// Two-worker Ring/LongNet experiment for TechJam case 8.
//
// This is a portable C++17 reference implementation.  It keeps the official
// Transformer block ordering and splits the expensive attention and FFN work
// into two independent worker shards.  The worker calls are deliberately
// isolated so they can later be replaced by ESP-NOW/TCP RPCs.
//
// Default shape (the leading "8" in the user's list is the case number):
//   B=64, D=1024, S=128, H=4, L=4, causal=true, F=1024.
//
// Modes:
//   baseline : dense causal attention, head-sharded over two workers.
//   ring     : exact online-softmax attention, visiting two KV shards.
//   longnet  : LongNet-style sparse/dilated causal attention.
//   ring-longnet: LongNet patterns with Ring Attention KV traversal; not
//                 baseline equivalent.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
  int batch = 64;
  int dim = 1024;
  int seq = 128;
  int heads = 4;
  int layers = 4;
  int ffn = 1024;
  bool causal = true;
  std::string mode = "baseline";
  uint32_t seed = 1234;
};

struct Matrix {
  int rows = 0;
  int cols = 0;
  std::vector<float> v;
  Matrix() = default;
  Matrix(int r, int c) : rows(r), cols(c), v(static_cast<size_t>(r) * c) {}
  float *row(int r) { return v.data() + static_cast<size_t>(r) * cols; }
  const float *row(int r) const { return v.data() + static_cast<size_t>(r) * cols; }
};

struct Layer {
  std::vector<float> n1w, n1b, n2w, n2b;
  Matrix qw, kw, vw, ow, f1w, f2w;
  std::vector<float> qb, kb, vb, ob, f1b, f2b;

  explicit Layer(const Config &c)
      : n1w(c.dim), n1b(c.dim), n2w(c.dim), n2b(c.dim),
        qw(c.dim, c.dim), kw(c.dim, c.dim), vw(c.dim, c.dim),
        ow(c.dim, c.dim), f1w(c.ffn, c.dim), f2w(c.dim, c.ffn),
        qb(c.dim), kb(c.dim), vb(c.dim), ob(c.dim), f1b(c.ffn), f2b(c.dim) {}
};

using Rows = Matrix; // rows x features

// Parameters resident on one physical board. Worker methods receive only this
// shard, never the full layer weights.
struct LayerShard {
  int head_begin, head_end, feature_dim;
  int ffn_begin, ffn_end;
  Matrix qw, kw, vw;       // owned Q/K/V rows x D
  Matrix ow;               // D x owned head features
  Matrix f1w, f2w;         // owned FFN rows, then D x owned FFN columns
  std::vector<float> qb, kb, vb, f1b;

  LayerShard(const Layer &l, const Config &c, int worker)
      : head_begin(worker == 0 ? 0 : c.heads / 2),
        head_end(worker == 0 ? c.heads / 2 : c.heads),
        feature_dim((head_end - head_begin) * (c.dim / c.heads)),
        ffn_begin(worker == 0 ? 0 : c.ffn / 2),
        ffn_end(worker == 0 ? c.ffn / 2 : c.ffn),
        qw(feature_dim, c.dim), kw(feature_dim, c.dim), vw(feature_dim, c.dim),
        ow(c.dim, feature_dim), f1w(ffn_end - ffn_begin, c.dim),
        f2w(c.dim, ffn_end - ffn_begin), qb(feature_dim), kb(feature_dim),
        vb(feature_dim), f1b(ffn_end - ffn_begin) {
    for (int r = 0; r < feature_dim; ++r) {
      const int global = head_begin * (c.dim / c.heads) + r;
      std::copy(l.qw.row(global), l.qw.row(global) + c.dim, qw.row(r));
      std::copy(l.kw.row(global), l.kw.row(global) + c.dim, kw.row(r));
      std::copy(l.vw.row(global), l.vw.row(global) + c.dim, vw.row(r));
      qb[r] = l.qb[global]; kb[r] = l.kb[global]; vb[r] = l.vb[global];
      for (int out = 0; out < c.dim; ++out)
        ow.row(out)[r] = l.ow.row(out)[global];
    }
    for (int r = ffn_begin; r < ffn_end; ++r) {
      const int local = r - ffn_begin;
      std::copy(l.f1w.row(r), l.f1w.row(r) + c.dim, f1w.row(local));
      f1b[local] = l.f1b[r];
      for (int out = 0; out < c.dim; ++out)
        f2w.row(out)[local] = l.f2w.row(out)[r];
    }
  }

  size_t bytes() const {
    return (qw.v.size() + kw.v.size() + vw.v.size() + ow.v.size() +
            f1w.v.size() + f2w.v.size() + qb.size() + kb.size() + vb.size() +
            f1b.size()) * sizeof(float);
  }
};

struct LayerShards {
  LayerShard worker0;
  LayerShard worker1;
  LayerShards(const Layer &l, const Config &c) : worker0(l, c, 0), worker1(l, c, 1) {}
};

void fill_random(std::vector<float> &x, std::mt19937 &rng, float scale = 0.02f) {
  std::normal_distribution<float> d(0.0f, scale);
  for (float &v : x) v = d(rng);
}

void fill_random(Matrix &x, std::mt19937 &rng, float scale = 0.02f) {
  fill_random(x.v, rng, scale);
}

void initialize(Layer &l, std::mt19937 &rng) {
  fill_random(l.n1w, rng, 0.02f);
  fill_random(l.n2w, rng, 0.02f);
  fill_random(l.n1b, rng, 0.002f);
  fill_random(l.n2b, rng, 0.002f);
  fill_random(l.qw, rng); fill_random(l.kw, rng); fill_random(l.vw, rng);
  fill_random(l.ow, rng); fill_random(l.f1w, rng); fill_random(l.f2w, rng);
  fill_random(l.qb, rng, 0.002f); fill_random(l.kb, rng, 0.002f);
  fill_random(l.vb, rng, 0.002f); fill_random(l.ob, rng, 0.002f);
  fill_random(l.f1b, rng, 0.002f); fill_random(l.f2b, rng, 0.002f);
}

struct Model {
  std::vector<Layer> layers;       // canonical host copy for reference setup
  std::vector<LayerShards> shards; // only these are visible to worker calls
  std::vector<float> final_w, final_b;

  Model(const Config &c, std::mt19937 &rng) : final_w(c.dim), final_b(c.dim) {
    layers.reserve(c.layers);
    for (int i = 0; i < c.layers; ++i) {
      layers.emplace_back(c);
      initialize(layers.back(), rng);
    }
    shards.reserve(c.layers);
    for (const Layer &layer : layers) shards.emplace_back(layer, c);
    fill_random(final_w, rng, 0.02f);
    fill_random(final_b, rng, 0.002f);
  }
};

void layer_norm(const Rows &x, Rows &y, const std::vector<float> &weight,
                const std::vector<float> &bias) {
  constexpr float eps = 1e-5f;
  for (int r = 0; r < x.rows; ++r) {
    const float *src = x.row(r);
    float *dst = y.row(r);
    float mean = 0.0f;
    for (int j = 0; j < x.cols; ++j) mean += src[j];
    mean /= static_cast<float>(x.cols);
    float variance = 0.0f;
    for (int j = 0; j < x.cols; ++j) {
      const float z = src[j] - mean;
      variance += z * z;
    }
    const float inv = 1.0f / std::sqrt(variance / x.cols + eps);
    for (int j = 0; j < x.cols; ++j)
      dst[j] = (src[j] - mean) * inv * weight[j] + bias[j];
  }
}

void linear_range(const Rows &x, const Matrix &w, const std::vector<float> &b,
                  int begin, int end, Rows &y) {
  if (y.rows != x.rows || y.cols != end - begin)
    throw std::runtime_error("linear_range shape mismatch");
  for (int r = 0; r < x.rows; ++r) {
    const float *src = x.row(r);
    float *dst = y.row(r);
    for (int o = begin; o < end; ++o) {
      const float *wr = w.row(o);
      float sum = b[o];
      for (int i = 0; i < x.cols; ++i) sum += src[i] * wr[i];
      dst[o - begin] = sum;
    }
  }
}

float gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// A worker owns a contiguous group of attention heads.  It returns a partial
// output projection, so the coordinator only has to add two D-wide tensors.
Rows attention_worker(const Rows &x, const LayerShard &s, const Config &c) {
  const int hd = c.dim / c.heads;
  Rows q(c.seq, s.feature_dim), k(c.seq, s.feature_dim), v(c.seq, s.feature_dim);
  linear_range(x, s.qw, s.qb, 0, s.feature_dim, q);
  linear_range(x, s.kw, s.kb, 0, s.feature_dim, k);
  linear_range(x, s.vw, s.vb, 0, s.feature_dim, v);
  Rows partial(c.seq, c.dim);
  std::fill(partial.v.begin(), partial.v.end(), 0.0f);
  std::vector<float> scores(c.seq), context(hd);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int h = 0; h < s.feature_dim / hd; ++h) {
    const int local = h * hd;
    for (int qi = 0; qi < c.seq; ++qi) {
      const int last = c.causal ? qi + 1 : c.seq;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int kj = 0; kj < last; ++kj) {
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += q.row(qi)[local + d] * k.row(kj)[local + d];
        scores[kj] = dot * scale;
        maximum = std::max(maximum, scores[kj]);
      }
      float denom = 0.0f;
      for (int kj = 0; kj < last; ++kj) denom += std::exp(scores[kj] - maximum);
      std::fill(context.begin(), context.end(), 0.0f);
      for (int d = 0; d < hd; ++d)
        for (int kj = 0; kj < last; ++kj)
          context[d] += std::exp(scores[kj] - maximum) * v.row(kj)[local + d] / denom;
      for (int out = 0; out < c.dim; ++out) {
        float projected = 0.0f;
        for (int d = 0; d < hd; ++d) projected += s.ow.row(out)[local + d] * context[d];
        partial.row(qi)[out] += projected;
      }
    }
  }
  return partial;
}

// Exact ring-style softmax.  The two workers own KV blocks.  Online merging
// avoids materializing the SxS score matrix and is algebraically equivalent to
// dense causal attention (within normal floating-point reduction order).
Rows ring_attention_worker(const Rows &x, const LayerShard &s, const Config &c) {
  const int hd = c.dim / c.heads;
  Rows q(c.seq, s.feature_dim), k(c.seq, s.feature_dim), v(c.seq, s.feature_dim);
  linear_range(x, s.qw, s.qb, 0, s.feature_dim, q);
  linear_range(x, s.kw, s.kb, 0, s.feature_dim, k);
  linear_range(x, s.vw, s.vb, 0, s.feature_dim, v);
  Rows partial(c.seq, c.dim); std::fill(partial.v.begin(), partial.v.end(), 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int h = 0; h < s.feature_dim / hd; ++h) {
    const int local = h * hd;
    for (int qi = 0; qi < c.seq; ++qi) {
      float global_max = -std::numeric_limits<float>::infinity();
      float global_sum = 0.0f;
      std::vector<float> numerator(hd, 0.0f);
      // Each ring step consumes one KV tile. A physical worker would send its
      // tile to the peer and receive the peer tile here.
      for (int ring_step = 0; ring_step < 2; ++ring_step) {
        const int begin = ring_step * (c.seq / 2);
        const int end = std::min(c.seq, begin + c.seq / 2);
        float block_max = -std::numeric_limits<float>::infinity();
        std::vector<float> score(c.seq / 2); int count = 0;
        for (int kj = begin; kj < end; ++kj) {
          if (c.causal && kj > qi) continue;
          float dot = 0.0f;
          for (int d = 0; d < hd; ++d) dot += q.row(qi)[local+d] * k.row(kj)[local+d];
          score[count++] = dot * scale; block_max = std::max(block_max, score[count-1]);
        }
        if (count == 0) continue;
        float block_sum = 0.0f; std::vector<float> block_num(hd, 0.0f); int p = 0;
        for (int kj = begin; kj < end; ++kj) {
          if (c.causal && kj > qi) continue;
          const float w = std::exp(score[p++] - block_max);
          block_sum += w;
          for (int d = 0; d < hd; ++d) block_num[d] += w * v.row(kj)[local+d];
        }
        const float merged_max = std::max(global_max, block_max);
        const float old_scale = std::isinf(global_max) ? 0.0f : std::exp(global_max - merged_max);
        const float block_scale = std::exp(block_max - merged_max);
        for (int d = 0; d < hd; ++d) numerator[d] = numerator[d] * old_scale + block_num[d] * block_scale;
        global_sum = global_sum * old_scale + block_sum * block_scale;
        global_max = merged_max;
      }
      for (int out = 0; out < c.dim; ++out) {
        float projected = 0.0f;
        for (int d = 0; d < hd; ++d) projected += s.ow.row(out)[local+d] * numerator[d] / global_sum;
        partial.row(qi)[out] += projected;
      }
    }
  }
  return partial;
}

// LongNet-like sparse causal pattern: local window plus dilated global links.
// It is intentionally separate from the baseline/ring path because it changes
// the attention graph and therefore cannot pass the official exact test.
Rows longnet_worker(const Rows &x, const LayerShard &s, const Config &c) {
  const int hd = c.dim / c.heads;
  Rows q(c.seq, s.feature_dim), k(c.seq, s.feature_dim), v(c.seq, s.feature_dim);
  linear_range(x, s.qw, s.qb, 0, s.feature_dim, q);
  linear_range(x, s.kw, s.kb, 0, s.feature_dim, k);
  linear_range(x, s.vw, s.vb, 0, s.feature_dim, v);
  Rows partial(c.seq, c.dim); std::fill(partial.v.begin(), partial.v.end(), 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  // LongNet's geometric (segment length, dilation) mixture. For S=128 the
  // local pattern is exact within a 16-token segment; larger patterns provide
  // exponentially wider, dilated context with bounded working memory.
  constexpr int pattern_count = 3;
  const int widths[pattern_count] = {16, 32, 128};
  const int dilations[pattern_count] = {1, 2, 8};

  for (int h = 0; h < s.feature_dim / hd; ++h) {
    const int local = h * hd;
    const int global_head = s.head_begin + h;
    for (int qi = 0; qi < c.seq; ++qi) {
      std::vector<float> mixed(hd, 0.0f);
      float mixture_weight = 0.0f;
      for (int pattern = 0; pattern < pattern_count; ++pattern) {
        const int w = widths[pattern], r = dilations[pattern];
        const int segment_begin = (qi / w) * w;
        const int segment_end = std::min(c.seq, segment_begin + w);
        const int offset = global_head % r;
        std::vector<int> keys;
        for (int kj = segment_begin; kj < segment_end && kj <= qi; ++kj)
          if ((kj - segment_begin - offset) % r == 0) keys.push_back(kj);
        if (keys.empty()) continue;

        float maximum = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(keys.size());
        for (size_t p = 0; p < keys.size(); ++p) {
          float dot = 0.0f;
          for (int d = 0; d < hd; ++d) dot += q.row(qi)[local+d] * k.row(keys[p])[local+d];
          scores[p] = dot * scale; maximum = std::max(maximum, scores[p]);
        }
        float denom = 0.0f;
        for (float score : scores) denom += std::exp(score - maximum);
        // Equation (9)-(10): use the softmax denominator as the dynamic
        // mixture weight, without storing a dense SxS attention matrix.
        for (size_t p = 0; p < keys.size(); ++p) {
          const float weight = std::exp(scores[p] - maximum);
          for (int d = 0; d < hd; ++d)
            mixed[d] += denom * weight * v.row(keys[p])[local+d] / denom;
        }
        mixture_weight += denom;
      }
      if (mixture_weight == 0.0f) continue;
      for (int d = 0; d < hd; ++d) mixed[d] /= mixture_weight;
      for (int out = 0; out < c.dim; ++out) {
        float projected = 0.0f;
        for (int d = 0; d < hd; ++d) projected += s.ow.row(out)[local+d] * mixed[d];
        partial.row(qi)[out] += projected;
      }
    }
  }
  return partial;
}

// LongNet sparse attention with Ring Attention traversal. The selected K/V
// tokens are consumed one sequence block at a time, as if they had arrived
// from the neighboring board. Online max/sum/numerator statistics keep the
// result bounded without materializing scores.
Rows ring_longnet_worker(const Rows &x, const LayerShard &s, const Config &c) {
  const int hd = c.dim / c.heads;
  Rows q(c.seq, s.feature_dim), k(c.seq, s.feature_dim), v(c.seq, s.feature_dim);
  linear_range(x, s.qw, s.qb, 0, s.feature_dim, q);
  linear_range(x, s.kw, s.kb, 0, s.feature_dim, k);
  linear_range(x, s.vw, s.vb, 0, s.feature_dim, v);
  Rows partial(c.seq, c.dim); std::fill(partial.v.begin(), partial.v.end(), 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  constexpr int pattern_count = 3;
  const int widths[pattern_count] = {16, 32, 128};
  const int dilations[pattern_count] = {1, 2, 8};

  for (int h = 0; h < s.feature_dim / hd; ++h) {
    const int local = h * hd, global_head = s.head_begin + h;
    for (int qi = 0; qi < c.seq; ++qi) {
      std::vector<float> mixed(hd, 0.0f);
      float total_pattern_denom = 0.0f;
      for (int pattern = 0; pattern < pattern_count; ++pattern) {
        const int w = widths[pattern], r = dilations[pattern];
        const int segment_begin = (qi / w) * w;
        const int segment_end = std::min(c.seq, segment_begin + w);
        const int offset = global_head % r;
        float maximum = -std::numeric_limits<float>::infinity();
        // Ring rotations: block 0 is local, block 1 is the peer's K/V tile.
        for (int ring_step = 0; ring_step < 2; ++ring_step) {
          const int block_begin = ring_step * (c.seq / 2);
          const int block_end = std::min(c.seq, block_begin + c.seq / 2);
          for (int kj = block_begin; kj < block_end && kj <= qi; ++kj) {
            if (kj < segment_begin || kj >= segment_end ||
                (kj - segment_begin - offset) % r != 0) continue;
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += q.row(qi)[local+d] * k.row(kj)[local+d];
            maximum = std::max(maximum, dot * scale);
          }
        }
        if (std::isinf(maximum)) continue;
        float denom = 0.0f; std::vector<float> numerator(hd, 0.0f);
        for (int ring_step = 0; ring_step < 2; ++ring_step) {
          const int block_begin = ring_step * (c.seq / 2);
          const int block_end = std::min(c.seq, block_begin + c.seq / 2);
          for (int kj = block_begin; kj < block_end && kj <= qi; ++kj) {
            if (kj < segment_begin || kj >= segment_end ||
                (kj - segment_begin - offset) % r != 0) continue;
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += q.row(qi)[local+d] * k.row(kj)[local+d];
            const float weight = std::exp(dot * scale - maximum);
            denom += weight;
            for (int d = 0; d < hd; ++d) numerator[d] += weight * v.row(kj)[local+d];
          }
        }
        for (int d = 0; d < hd; ++d) mixed[d] += numerator[d];
        total_pattern_denom += denom;
      }
      if (total_pattern_denom == 0.0f) continue;
      for (int d = 0; d < hd; ++d) mixed[d] /= total_pattern_denom;
      for (int out = 0; out < c.dim; ++out) {
        float projected = 0.0f;
        for (int d = 0; d < hd; ++d) projected += s.ow.row(out)[local+d] * mixed[d];
        partial.row(qi)[out] += projected;
      }
    }
  }
  return partial;
}

Rows run_attention(const Rows &x, const Layer &l, const LayerShards &s,
                   const Config &c) {
  auto fn = [&](int worker) -> Rows {
    const LayerShard &shard = worker == 0 ? s.worker0 : s.worker1;
    if (c.mode == "ring") return ring_attention_worker(x, shard, c);
    if (c.mode == "longnet") return longnet_worker(x, shard, c);
    if (c.mode == "ring-longnet") return ring_longnet_worker(x, shard, c);
    return attention_worker(x, shard, c);
  };
  auto a = std::async(std::launch::async, fn, 0);
  auto b = std::async(std::launch::async, fn, 1);
  Rows out = a.get(), other = b.get();
  for (size_t i = 0; i < out.v.size(); ++i) out.v[i] += other.v[i];
  for (int r = 0; r < c.seq; ++r)
    for (int j = 0; j < c.dim; ++j) out.row(r)[j] += l.ob[j];
  return out;
}

Rows run_ffn(const Rows &x, const Layer &l, const LayerShards &s,
             const Config &c) {
  auto fn = [&](int worker) -> Rows {
    const LayerShard &shard = worker == 0 ? s.worker0 : s.worker1;
    const int count = shard.ffn_end - shard.ffn_begin;
    Rows hidden(c.seq, count);
    linear_range(x, shard.f1w, shard.f1b, 0, count, hidden);
    for (float &v : hidden.v) v = gelu(v);
    Rows partial(c.seq, c.dim);
    std::fill(partial.v.begin(), partial.v.end(), 0.0f);
    for (int r = 0; r < c.seq; ++r)
      for (int o = 0; o < c.dim; ++o) {
        float sum = 0.0f;
        for (int j = 0; j < count; ++j) sum += hidden.row(r)[j] * shard.f2w.row(o)[j];
        partial.row(r)[o] = sum;
      }
    return partial;
  };
  auto a = std::async(std::launch::async, fn, 0);
  auto b = std::async(std::launch::async, fn, 1);
  Rows out = a.get(), other = b.get();
  for (size_t i = 0; i < out.v.size(); ++i) out.v[i] += other.v[i];
  for (int r = 0; r < c.seq; ++r)
    for (int j = 0; j < c.dim; ++j) out.row(r)[j] += l.f2b[j];
  return out;
}

Rows forward(Rows x, const Model &model, const Config &c) {
  Rows norm(c.seq, c.dim);
  for (const Layer &l : model.layers) {
    layer_norm(x, norm, l.n1w, l.n1b);
    Rows attn = run_attention(norm, l, model.shards[&l - model.layers.data()], c);
    for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += attn.v[i];
    layer_norm(x, norm, l.n2w, l.n2b);
    Rows ffn = run_ffn(norm, l, model.shards[&l - model.layers.data()], c);
    for (size_t i = 0; i < x.v.size(); ++i) x.v[i] += ffn.v[i];
  }
  layer_norm(x, norm, model.final_w, model.final_b);
  return norm;
}

void usage(const char *name) {
  std::cout << "usage: " << name << " [--mode baseline|ring|longnet|ring-longnet] [--batch N] [--repeats N]\n";
}

} // namespace

int main(int argc, char **argv) {
  Config c;
  int repeats = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char *flag) -> int {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return std::stoi(argv[++i]);
    };
    if (a == "--mode") { if (i + 1 >= argc) throw std::runtime_error("missing --mode"); c.mode = argv[++i]; }
    else if (a == "--batch") c.batch = value("--batch");
    else if (a == "--repeats") repeats = value("--repeats");
    else if (a == "--seed") c.seed = static_cast<uint32_t>(value("--seed"));
    else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (c.mode != "baseline" && c.mode != "ring" && c.mode != "longnet" &&
      c.mode != "ring-longnet")
    throw std::runtime_error("mode must be baseline, ring, longnet, or ring-longnet");
  if (c.dim % c.heads || c.ffn % 2 || c.heads != 4)
    throw std::runtime_error("this experiment requires D divisible by H=4 and even F");

  std::mt19937 rng(c.seed);
  Model model(c, rng);

  std::cout << "mode=" << c.mode << " workers=2 B=" << c.batch << " D=" << c.dim
            << " S=" << c.seq << " H=" << c.heads << " L=" << c.layers
            << " F=" << c.ffn << " causal=" << (c.causal ? "true" : "false") << "\n";
  size_t shard0_bytes = 0, shard1_bytes = 0;
  for (const LayerShards &shards : model.shards) {
    shard0_bytes += shards.worker0.bytes();
    shard1_bytes += shards.worker1.bytes();
  }
  std::cout << "worker0_fp32_shard_MB=" << (shard0_bytes / 1048576.0)
            << " worker1_fp32_shard_MB=" << (shard1_bytes / 1048576.0)
            << " note=int8_storage_is_required_on_4MB_C3\n";
  double total_ms = 0.0;
  std::vector<float> checksum;
  for (int b = 0; b < c.batch; ++b) {
    Rows input(c.seq, c.dim);
    std::mt19937 input_rng(c.seed + 10000u + static_cast<uint32_t>(b));
    fill_random(input.v, input_rng, 0.02f);
    const auto begin = std::chrono::steady_clock::now();
    Rows output;
    for (int r = 0; r < repeats; ++r) output = forward(input, model, c);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - begin).count() / repeats;
    total_ms += ms;
    if (b == 0) checksum = output.v;
    std::cout << "sample=" << b << " ms=" << std::fixed << std::setprecision(2) << ms << "\n";
  }
  double sum = 0.0;
  for (float v : checksum) sum += v;
  std::cout << "mean_ms=" << std::fixed << std::setprecision(2) << total_ms / c.batch
            << " checksum=" << std::setprecision(6) << sum << "\n";
  if (c.mode == "longnet" || c.mode == "ring-longnet")
    std::cout << "warning=LongNet attention is sparse and not baseline-equivalent\n";
  return 0;
}
