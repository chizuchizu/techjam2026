#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::size_t sequence = 2048;
  std::size_t heads = 1;
  std::size_t head_dim = 64;
  int repeats = 3;
  std::size_t batches = 1;
  std::size_t validate_sequence = 256;
  bool causal = true;
  bool list_devices = false;
};

std::size_t parse_size(const char *text, const char *flag) {
  const auto value = std::stoull(text);
  if (value == 0) throw std::runtime_error(std::string(flag) + " must be positive");
  return static_cast<std::size_t>(value);
}

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *flag) {
      if (++i >= argc) throw std::runtime_error(std::string("missing ") + flag);
      return argv[i];
    };
    if (arg == "--sequence") o.sequence = parse_size(next("--sequence"), "--sequence");
    else if (arg == "--heads") o.heads = parse_size(next("--heads"), "--heads");
    else if (arg == "--head-dim") o.head_dim = parse_size(next("--head-dim"), "--head-dim");
    else if (arg == "--repeats") o.repeats = static_cast<int>(parse_size(next("--repeats"), "--repeats"));
    else if (arg == "--batches") o.batches = parse_size(next("--batches"), "--batches");
    else if (arg == "--validate-sequence")
      o.validate_sequence = std::stoull(next("--validate-sequence"));
    else if (arg == "--non-causal") o.causal = false;
    else if (arg == "--list-devices") o.list_devices = true;
    else if (arg == "--help" || arg == "-h") {
      std::cout << "sycl_attention [--sequence N] [--heads N] [--head-dim N] "
                   "[--repeats N] [--batches N] [--validate-sequence N] [--non-causal] "
                   "[--list-devices]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (o.head_dim > 256 || (o.head_dim & (o.head_dim - 1)) != 0)
    throw std::runtime_error("head_dim must be a power of two no larger than 256");
  return o;
}

void list_devices() {
  for (const auto &platform : sycl::platform::get_platforms()) {
    std::cout << "platform=" << platform.get_info<sycl::info::platform::name>() << '\n';
    for (const auto &device : platform.get_devices())
      std::cout << "  device=" << device.get_info<sycl::info::device::name>()
                << " type=" << static_cast<int>(device.get_info<sycl::info::device::device_type>())
                << " local_mem=" << device.get_info<sycl::info::device::local_mem_size>()
                << " max_wg=" << device.get_info<sycl::info::device::max_work_group_size>()
                << '\n';
  }
}

class OnlineCausalAttention;
class TiledOnlineCausalAttention;

sycl::event submit_simple_attention(sycl::queue &queue, const float *q, const float *k,
                                    const float *v, float *output, std::size_t sequence,
                                    std::size_t heads, std::size_t head_dim, bool causal) {
  const std::size_t groups = heads * sequence;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  return queue.submit([&](sycl::handler &handler) {
    handler.parallel_for<OnlineCausalAttention>(
        sycl::nd_range<1>(groups * head_dim, head_dim),
        [=](sycl::nd_item<1> item) {
          const std::size_t group_index = item.get_group_linear_id();
          const std::size_t lane = item.get_local_linear_id();
          const std::size_t head = group_index / sequence;
          const std::size_t query_index = group_index % sequence;
          const std::size_t head_base = head * sequence * head_dim;
          const std::size_t query_base = head_base + query_index * head_dim;
          const std::size_t key_count = causal ? query_index + 1 : sequence;
          const float q_value = q[query_base + lane];

          // One work-group owns one query row. Each lane owns one output
          // feature. Online max/sum/output state makes storage O(S*D), not O(S^2).
          float running_max = -std::numeric_limits<float>::infinity();
          float running_sum = 0.0f;
          float accumulator = 0.0f;
          constexpr std::size_t key_tile = 128;

          for (std::size_t begin = 0; begin < key_count; begin += key_tile) {
            const std::size_t end = sycl::min(begin + key_tile, key_count);
            float block_max = -std::numeric_limits<float>::infinity();
            for (std::size_t key_index = begin; key_index < end; ++key_index) {
              const std::size_t key_base = head_base + key_index * head_dim;
              const float partial = q_value * k[key_base + lane];
              const float dot = sycl::reduce_over_group(
                  item.get_group(), partial, sycl::plus<float>());
              block_max = sycl::fmax(block_max, dot * scale);
            }

            float block_sum = 0.0f;
            float block_output = 0.0f;
            for (std::size_t key_index = begin; key_index < end; ++key_index) {
              const std::size_t key_base = head_base + key_index * head_dim;
              const float partial = q_value * k[key_base + lane];
              const float dot = sycl::reduce_over_group(
                  item.get_group(), partial, sycl::plus<float>());
              const float weight = sycl::exp(dot * scale - block_max);
              block_sum += weight;
              block_output += weight * v[key_base + lane];
            }

            const float merged_max = sycl::fmax(running_max, block_max);
            const float old_scale = sycl::isinf(running_max)
                                        ? 0.0f
                                        : sycl::exp(running_max - merged_max);
            const float new_scale = sycl::exp(block_max - merged_max);
            accumulator = accumulator * old_scale + block_output * new_scale;
            running_sum = running_sum * old_scale + block_sum * new_scale;
            running_max = merged_max;
          }
          output[query_base + lane] = accumulator / running_sum;
        });
  });
}

sycl::event submit_attention(sycl::queue &queue, const float *q, const float *k,
                             const float *v, float *output, std::size_t sequence,
                             std::size_t heads, std::size_t head_dim, bool causal) {
  if (head_dim != 64)
    return submit_simple_attention(queue, q, k, v, output, sequence, heads,
                                   head_dim, causal);

  constexpr std::size_t head_width = 64;
  constexpr std::size_t subgroup_width = 16;
  constexpr std::size_t query_tile = 16;
  constexpr std::size_t key_tile = 32;
  constexpr std::size_t work_group_size = subgroup_width * query_tile;
  const std::size_t query_blocks = (sequence + query_tile - 1) / query_tile;
  const std::size_t groups = heads * query_blocks;
  const float scale = 1.0f / 8.0f;

  return queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<float, 1> local_kv(2 * key_tile * head_width, handler);
    sycl::local_accessor<float, 1> local_scores(query_tile * key_tile, handler);
    handler.parallel_for<TiledOnlineCausalAttention>(
        sycl::nd_range<1>(groups * work_group_size, work_group_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::size_t group_index = item.get_group_linear_id();
          const std::size_t local_id = item.get_local_linear_id();
          const auto subgroup = item.get_sub_group();
          const std::size_t query_in_tile = subgroup.get_group_linear_id();
          const std::size_t lane = subgroup.get_local_linear_id();
          const std::size_t head = group_index / query_blocks;
          const std::size_t query_block = group_index % query_blocks;
          const std::size_t query_index = query_block * query_tile + query_in_tile;
          const bool valid_query = query_index < sequence;
          const std::size_t head_base = head * sequence * head_width;
          const std::size_t query_base = head_base + query_index * head_width;

          float q_registers[4];
          float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (std::size_t part = 0; part < 4; ++part) {
            const std::size_t d = lane + part * subgroup_width;
            q_registers[part] = valid_query ? q[query_base + d] : 0.0f;
          }
          float running_max = -std::numeric_limits<float>::infinity();
          float running_sum = 0.0f;
          const std::size_t group_key_count =
              causal ? sycl::min(sequence, (query_block + 1) * query_tile) : sequence;

          for (std::size_t begin = 0; begin < group_key_count; begin += key_tile) {
            const std::size_t count = sycl::min(key_tile, group_key_count - begin);
            const std::size_t tile_elements = count * head_width;
            for (std::size_t index = local_id; index < tile_elements;
                 index += work_group_size) {
              const std::size_t key_index = begin + index / head_width;
              const std::size_t d = index % head_width;
              local_kv[index] = k[head_base + key_index * head_width + d];
              local_kv[key_tile * head_width + index] =
                  v[head_base + key_index * head_width + d];
            }
            item.barrier(sycl::access::fence_space::local_space);

            float block_max = -std::numeric_limits<float>::infinity();
            for (std::size_t key_offset = 0; key_offset < count; ++key_offset) {
              float partial = 0.0f;
              for (std::size_t part = 0; part < 4; ++part) {
                const std::size_t d = lane + part * subgroup_width;
                partial += q_registers[part] *
                           local_kv[key_offset * head_width + d];
              }
              const float dot = sycl::reduce_over_group(subgroup, partial,
                                                        sycl::plus<float>());
              const std::size_t key_index = begin + key_offset;
              const float score = valid_query && (!causal || key_index <= query_index)
                                      ? dot * scale
                                      : -std::numeric_limits<float>::infinity();
              if (lane == 0)
                local_scores[query_in_tile * key_tile + key_offset] = score;
              block_max = sycl::fmax(block_max, score);
            }
            item.barrier(sycl::access::fence_space::local_space);

            if (valid_query) {
              float block_sum = 0.0f;
              float block_outputs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
              for (std::size_t key_offset = 0; key_offset < count; ++key_offset) {
                const float score =
                    local_scores[query_in_tile * key_tile + key_offset];
                const float weight = sycl::exp(score - block_max);
                block_sum += weight;
                for (std::size_t part = 0; part < 4; ++part) {
                  const std::size_t d = lane + part * subgroup_width;
                  block_outputs[part] +=
                      weight * local_kv[key_tile * head_width +
                                        key_offset * head_width + d];
                }
              }
              const float merged_max = sycl::fmax(running_max, block_max);
              const float old_scale = sycl::isinf(running_max)
                                          ? 0.0f
                                          : sycl::exp(running_max - merged_max);
              const float new_scale = sycl::exp(block_max - merged_max);
              for (std::size_t part = 0; part < 4; ++part)
                accumulators[part] = accumulators[part] * old_scale +
                                     block_outputs[part] * new_scale;
              running_sum = running_sum * old_scale + block_sum * new_scale;
              running_max = merged_max;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          if (valid_query) {
            for (std::size_t part = 0; part < 4; ++part) {
              const std::size_t d = lane + part * subgroup_width;
              output[query_base + d] = accumulators[part] / running_sum;
            }
          }
        });
  });
}

void fill_inputs(float *q, float *k, float *v, std::size_t elements) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> distribution(0.0f, 0.5f);
  for (std::size_t i = 0; i < elements; ++i) {
    q[i] = distribution(rng);
    k[i] = distribution(rng);
    v[i] = distribution(rng);
  }
}

void fill_benchmark_inputs(float *q, float *k, float *v, std::size_t elements,
                           std::uint32_t seed = 1234u) {
  // The accuracy test above retains Gaussian inputs. The large performance
  // case only needs finite representative values; a cheap LCG avoids spending
  // minutes in std::normal_distribution before a Case 14 kernel launch.
  std::uint32_t q_state = seed;
  std::uint32_t k_state = seed + 4444u;
  std::uint32_t v_state = seed + 8888u;
  constexpr float to_float = 1.0f / 16777216.0f;
  for (std::size_t i = 0; i < elements; ++i) {
    q_state = q_state * 1664525u + 1013904223u;
    k_state = k_state * 1664525u + 1013904223u;
    v_state = v_state * 1664525u + 1013904223u;
    q[i] = static_cast<float>(q_state >> 8) * to_float - 0.5f;
    k[i] = static_cast<float>(k_state >> 8) * to_float - 0.5f;
    v[i] = static_cast<float>(v_state >> 8) * to_float - 0.5f;
  }
}

std::vector<float> dense_reference(const float *q, const float *k, const float *v,
                                   std::size_t sequence, std::size_t heads,
                                   std::size_t head_dim, bool causal) {
  std::vector<float> output(heads * sequence * head_dim);
  std::vector<float> scores(sequence);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t head_base = head * sequence * head_dim;
    for (std::size_t qi = 0; qi < sequence; ++qi) {
      const std::size_t count = causal ? qi + 1 : sequence;
      const std::size_t query_base = head_base + qi * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t ki = 0; ki < count; ++ki) {
        const std::size_t key_base = head_base + ki * head_dim;
        float dot = 0.0f;
        for (std::size_t d = 0; d < head_dim; ++d)
          dot += q[query_base + d] * k[key_base + d];
        scores[ki] = dot * scale;
        maximum = std::max(maximum, scores[ki]);
      }
      float denominator = 0.0f;
      for (std::size_t ki = 0; ki < count; ++ki)
        denominator += std::exp(scores[ki] - maximum);
      for (std::size_t d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (std::size_t ki = 0; ki < count; ++ki)
          sum += std::exp(scores[ki] - maximum) * v[head_base + ki * head_dim + d];
        output[query_base + d] = sum / denominator;
      }
    }
  }
  return output;
}

void validate(sycl::queue &queue, std::size_t sequence, std::size_t heads,
              std::size_t head_dim, bool causal) {
  if (sequence == 0) return;
  const std::size_t elements = sequence * heads * head_dim;
  float *q = sycl::malloc_shared<float>(elements, queue);
  float *k = sycl::malloc_shared<float>(elements, queue);
  float *v = sycl::malloc_shared<float>(elements, queue);
  float *output = sycl::malloc_shared<float>(elements, queue);
  if (!q || !k || !v || !output) throw std::bad_alloc();
  fill_inputs(q, k, v, elements);
  submit_attention(queue, q, k, v, output, sequence, heads, head_dim, causal).wait_and_throw();
  const auto reference = dense_reference(q, k, v, sequence, heads, head_dim, causal);

  std::size_t failed = 0;
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < elements; ++i) {
    const float error = std::abs(output[i] - reference[i]);
    max_abs = std::max(max_abs, error);
    if (!(error <= 0.001f || error <= 0.01f * std::abs(reference[i]))) ++failed;
  }
  std::cout << "validation_sequence=" << sequence << " max_abs=" << max_abs
            << " failed=" << failed << '/' << elements << '\n';
  sycl::free(q, queue); sycl::free(k, queue); sycl::free(v, queue); sycl::free(output, queue);
  if (failed) throw std::runtime_error("accuracy validation failed");
}

void benchmark(sycl::queue &queue, const Options &o) {
  const std::size_t elements = o.sequence * o.heads * o.head_dim;
  const double mib_per_tensor = static_cast<double>(elements * sizeof(float)) / 1048576.0;
  std::cout << "shape=H" << o.heads << "xS" << o.sequence << "xD" << o.head_dim
            << " tensor_MiB=" << mib_per_tensor << " working_set_MiB=" << 4.0 * mib_per_tensor
            << " causal=" << std::boolalpha << o.causal << '\n';
  float *q = sycl::malloc_shared<float>(elements, queue);
  float *k = sycl::malloc_shared<float>(elements, queue);
  float *v = sycl::malloc_shared<float>(elements, queue);
  float *output = sycl::malloc_shared<float>(elements, queue);
  if (!q || !k || !v || !output)
    throw std::runtime_error("USM allocation failed; reduce --sequence or --heads");
  const auto initialization_start = std::chrono::steady_clock::now();
  fill_benchmark_inputs(q, k, v, elements, 1234u);
  const auto initialization_end = std::chrono::steady_clock::now();
  std::cout << "input_initialization_ms="
            << std::chrono::duration<double, std::milli>(initialization_end -
                                                          initialization_start)
                   .count()
            << '\n';

  submit_attention(queue, q, k, v, output, o.sequence, o.heads, o.head_dim, o.causal)
      .wait_and_throw();
  std::vector<double> milliseconds;
  for (std::size_t batch = 0; batch < o.batches; ++batch) {
    if (batch != 0) fill_benchmark_inputs(q, k, v, elements,
                                          1234u + static_cast<std::uint32_t>(batch));
    for (int repeat = 0; repeat < o.repeats; ++repeat) {
      auto event = submit_attention(queue, q, k, v, output, o.sequence,
                                    o.heads, o.head_dim, o.causal);
      event.wait_and_throw();
      const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
      const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
      const double ms = static_cast<double>(end - start) * 1e-6;
      milliseconds.push_back(ms);
      std::cout << "batch=" << batch << " repeat=" << repeat
                << " kernel_ms=" << ms << '\n';
    }
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const double median = milliseconds[milliseconds.size() / 2];
  const double pairs = o.causal
                           ? static_cast<double>(o.sequence) * (o.sequence + 1) / 2.0
                           : static_cast<double>(o.sequence) * o.sequence;
  const double flops = static_cast<double>(o.heads) * pairs * o.head_dim * 4.0;
  std::cout << "batches=" << o.batches << " median_ms=" << median << " attention_GFLOP_s="
            << flops / (median * 1e6) << " checksum=" << output[elements / 2] << '\n';
  sycl::free(q, queue); sycl::free(k, queue); sycl::free(v, queue); sycl::free(output, queue);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    // Flush each diagnostic immediately so driver/runtime failures cannot hide
    // the last successfully completed initialization step.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    const Options options = parse_args(argc, argv);
    if (options.list_devices) list_devices();
    std::cout << "selecting_gpu=true\n";
    sycl::queue queue(sycl::gpu_selector_v,
                      sycl::property_list{sycl::property::queue::enable_profiling{}});
    const auto device = queue.get_device();
    std::cout << "device=" << device.get_info<sycl::info::device::name>()
              << " driver=" << device.get_info<sycl::info::device::driver_version>() << '\n';
    if (options.head_dim > device.get_info<sycl::info::device::max_work_group_size>())
      throw std::runtime_error("head_dim exceeds device maximum work-group size");
    validate(queue, std::min(options.validate_sequence, options.sequence),
             options.heads, options.head_dim, options.causal);
    benchmark(queue, options);
    return 0;
  } catch (const sycl::exception &error) {
    std::cerr << "SYCL error: " << error.what() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
  }
  return 1;
}
