#!/usr/bin/env python3
"""score.py - model-FLOPs utilization (MFU) & bandwidth-aware execution score.

Computes, for the ESP32-C3 case-2 transformer, per-test-case MFU (weighted-sum
methodology of the benchmark evaluation) plus an execution score that accounts
for the memory-bandwidth roofline.

Methodology (documented so the evaluator can confirm/adjust the constants):
  * One forward = fixed model work (see flops_forward() below; ~122.6 MFLOP
    nominal, 2 FLOP/MAC).
  * MFU case-i = achieved_flops / (t_i * peak). Two peaks are reported:
      - P_MIX : the board's realistic mixed-precision peak, i.e.
                total_flops/(gemm_flops/P_INT + fp_flops/P_SOFTFP)
                with P_INT the int16-MAC ceiling (160 MMAC/s = 320 MFLOP/s) and
                P_SOFTFP the measured fp32 soft-float peak (~2 MFLOP/s, see
                docs/esp32_fp32_emulation_research.md + firecrawl research).
      - P_INT : strict raw scalar ceiling 320 MFLOP/s (every FLOP counted as
                int16 MAC; this is the "what's the raw core capable of" view).
  * Execution score (bandwidth considered): per op-class the roofline peak is
    min(compute_peak, BW * AI), AI = flops/bytes moved. GEMM bytes use the
    real kernel traffic (4 B/MAC, unblocked: 1 A-read + 1 W-read per MAC);
    fp32 parts use their operand traffic. Sum per-class times -> total
    roofline time -> ExScore = achieved_time / roofline_time.

Sources for peaks (firecrawl-verified, primary Espressif docs):
  ESP32-C3: RV32IMC @160 MHz, NO FPU, single scalar core, 4-stage in-order.
  int16 MAC ceiling = 160 MMAC/s (one 32-bit MUL/cycle max, no SIMD/HW-MAC).
  fp32 soft-float measured ~100 cyc/add (Espressif blog 2025-10) -> ~160 cyc
  per fused fp32 MAC -> ~2.0 MFLOP/s (range 1.5-2.7).
  SRAM load BW: 4 B/cycle scalar -> 640 MB/s theoretical @160 MHz; realistic
  sustained ~320-640 MB/s (estimate).
"""
import argparse, json, statistics

S, D, H, HD, F, L = 128, 128, 4, 32, 128, 4

def flops_forward():
    g  = 2.0 * (((3 + 1) * S * D * D) + (S * D * F) + (S * F * D)) * L
    cp = S * (S + 1) // 2
    a  = 2.0 * (2 * cp * HD) * H * L
    ln = (4 * L + 1) * S * D * 9.0
    g_ = L * S * F * 36.0
    r  = 2.0 * L * S * D
    return {"gemm": g, "attn": a, "ln": ln, "gelu": g_, "res": r,
            "gemm_int_mac": g / 2.0, "total": g + a + ln + g_ + r}

def bytes_gemm_forward(fl):
    # unblocked kernel: 1 int16 A-read + 1 int16 W-read per MAC = 4 B/MAC
    return 4.0 * fl["gemm_int_mac"]

def bytes_aux_forward(fl):
    # fp32 attention + LN + GELU + residual operand traffic (estimate).
    # attention: qi/kj/vj int16 rows are reused across the S window - amortized
    # ~2 B/MAC (int16 operand reads) + fp32 ctx dot-accumulate, generous upper
    # bound so AI_fp is a lower bound.
    attn = fl["attn"] / 2.0 * 2.0
    aux  = (fl["ln"] + fl["gelu"]) * 4.0 + fl["res"] * 8.0
    return attn + aux

# peaks ---------------------------------------------------------------- (FLOP/s)
P_INT    = 320.0e6      # 160 MMAC/s * 2
P_SOFTFP = 2.0e6        # range 1.5-2.7e6
P_SOFTFP_LO, P_SOFTFP_HI = 1.5e6, 2.7e6
BW_LO, BW_HI = 320.0e6, 640.0e6           # B/s (sustained SRAM, estimate)

def p_mix(fl, soft=P_SOFTFP, bw=None, gemm_ai=None):
    gemm_pk = min(P_INT, bw * gemm_ai) if bw else P_INT
    fp_pk   = P_SOFTFP
    return fl["total"] / (fl["gemm"] / gemm_pk + (fl["total"] - fl["gemm"]) / fp_pk)

def score_case(fl, t_s, soft=P_SOFTFP, bw=None, link=None):
    """Per-case MFU/ExScore.  `link` is an optional dict of a measured
    node-to-node data link:
        node_bw, node_traf : partner link peak (B/s) and bytes each node must
                             exchange with its partner per forward when the
                             workload is split across 2 nodes.
        host_bw, host_traf : optional host<->node USB-CDC link (B/s) and the
                             bytes the host moves per forward (counted once
                             per node pair, i.e. host_traf/2 adds to t_transfer).
    When given, a per-case *link-bound* view is computed:
        t_transfer = node_traf/node_bw + (host_traf/2)/host_bw
        link_scale = 1 / max(1, t_transfer/t_s)   # communication that does not
                                                  # bind leaves the score unchanged
    """
    ach = fl["total"] / t_s
    geo = fl["gemm"] / P_INT + (fl["total"] - fl["gemm"]) / soft
    mix_pk = fl["total"] / geo
    mfu_mix = ach / mix_pk
    mfu_int = ach / P_INT
    ai_total = fl["total"] / (bytes_gemm_forward(fl) + bytes_aux_forward(fl))
    ai_gemm = fl["gemm"] / bytes_gemm_forward(fl)          # ~0.5 FLOP/B
    ai_fp   = (fl["total"] - fl["gemm"]) / bytes_aux_forward(fl)
    gemm_pk = min(P_INT, bw * ai_gemm) if bw else P_INT
    fp_pk   = min(soft, bw * ai_fp) if bw else soft
    roof_t  = fl["gemm"] / gemm_pk + (fl["total"] - fl["gemm"]) / fp_pk
    ex = (fl["total"] / t_s) / (fl["total"] / roof_t)      # = roof_t / t_s
    r = {"t_s": t_s, "achieved_mflop_s": ach / 1e6,
         "ai_flop_b": ai_total, "ai_gemm": ai_gemm, "ai_fp": ai_fp,
         "mfu_mix": mfu_mix, "mfu_int": mfu_int,
         "exscore_lo": (roof_t/t_s) if bw else None, "roof_t": roof_t}
    if link and link.get("node_bw"):
        t_transfer = link["node_traf"] / link["node_bw"]
        if link.get("host_bw"):
            t_transfer += (link.get("host_traf", 0.0) / 2.0) / link["host_bw"]
        scale = 1.0 / max(1.0, t_transfer / t_s)
        r["link"] = {"node_bw": link["node_bw"], "node_traf": link["node_traf"],
                     "host_bw": link.get("host_bw"), "host_traf": link.get("host_traf", 0.0),
                     "t_transfer_s": t_transfer, "link_bound": t_transfer > t_s,
                     "link_scale": scale,
                     "mfu_mix_link": mfu_mix * scale,
                     "exscore_link": ex * scale}
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tools/runs.json",
                    help="JSON: {cases:[{case,t_s}]} measured device forwards")
    ap.add_argument("--weights", default=None,
                    help="JSON list of per-case weights (default: equal)")
    ap.add_argument("--output", default="scores.json",
                    help="output JSON path (use /dev/null for a smoke test)")
    ap.add_argument("--node-bw", type=float, default=None,
                    help="measured 2-node link peak in bytes/s (e.g. ESP-NOW)")
    ap.add_argument("--host-bw", type=float, default=None,
                    help="host<->node USB-CDC peak in bytes/s")
    ap.add_argument("--node-traf", type=float, default=0.0,
                    help="bytes each node exchanges with its partner per forward (2-node split)")
    ap.add_argument("--host-traf", type=float, default=0.0,
                    help="bytes the host moves per forward (counted once per node pair)")
    a = ap.parse_args()

    fl = flops_forward()
    runs = json.load(open(a.runs)) if __import__("os").path.exists(a.runs) \
           else {"cases": []}
    cases = runs["cases"]
    if not cases:
        # fall back to the device-measured forward times (esp32-baseline)
        cases = [{"case": "seed%d"%i, "t_s": t}
                 for i, t in enumerate(
                     [42.129714, 42.152393, 42.134257, 42.151013, 42.138329,
                      42.102588, 42.072949])]
    wts = a.weights and json.loads(a.weights)
    wts = wts or [1.0 / len(cases)] * len(cases)
    assert len(wts) == len(cases)

    link = None
    if a.node_bw:
        link = {"node_bw": a.node_bw, "node_traf": a.node_traf,
                "host_bw": a.host_bw, "host_traf": a.host_traf}

    rows = []
    for c, w in zip(cases, wts):
        r = score_case(fl, c["t_s"], bw=BW_HI, link=link)
        r["case"], r["w"] = c["case"], w
        rows.append(r)
    wsum_mix = sum(r["w"] * r["mfu_mix"] for r in rows)
    wsum_int = sum(r["w"] * r["mfu_int"] for r in rows)
    wsum_ex  = sum(r["w"] * r["exscore_lo"] for r in rows)
    wsum_link_mix = sum(r["w"] * r["link"]["mfu_mix_link"] for r in rows) if link else None
    wsum_link_ex  = sum(r["w"] * r["link"]["exscore_link"]  for r in rows) if link else None

    print("model flops (M):  gemm %.1f attn %.1f ln %.1f gelu %.1f res %.1f | total %.2f" % (
        fl["gemm"]/1e6, fl["attn"]/1e6, fl["ln"]/1e6, fl["gelu"]/1e6, fl["res"]/1e6, fl["total"]/1e6))
    print("bytes/fwd (MB):   gemm(naive) %.1f aux %.1f total %.1f  AI_total=%.2f FLOP/B" % (
        bytes_gemm_forward(fl)/1e6, bytes_aux_forward(fl)/1e6,
        (bytes_gemm_forward(fl)+bytes_aux_forward(fl))/1e6,
        fl["total"]/(bytes_gemm_forward(fl)+bytes_aux_forward(fl))))
    print("peaks: P_INT=%.0f MFLOP/s  P_SOFTFP=%.1f MFLOP/s  BW=%.0f MB/s" % (
        P_INT/1e6, P_SOFTFP/1e6, BW_HI/1e6))
    print("\nper-case (equal weight default):")
    if link:
        print("  %-10s %8s %10s %7s %7s | %7s %7s" % (
            "case","t(s)","MFU_mix%","MFU_int%","ExScore%","link_mix%","link_ex%"))
        for r in rows:
            print("  %-10s %8.3f %10.1f %7.2f %7.1f | %7.1f %7.1f" % (
                r["case"], r["t_s"], 100*r["mfu_mix"], 100*r["mfu_int"],
                100*r["exscore_lo"], 100*r["link"]["mfu_mix_link"],
                100*r["link"]["exscore_link"]))
    else:
        print("  %-10s %8s %10s %7s %7s" % ("case","t(s)","MFU_mix%","MFU_int%","ExScore%"))
        for r in rows:
            print("  %-10s %8.3f %10.1f %7.2f %7.1f" % (
                r["case"], r["t_s"], 100*r["mfu_mix"], 100*r["mfu_int"], 100*r["exscore_lo"]))
    print("\nWEIGHTED SUM  MFU(mix)=%.1f%%  MFU(raw-int)=%.2f%%  ExScore=%.1f%%" % (
        100*wsum_mix, 100*wsum_int, 100*wsum_ex))
    if link:
        print("LINK  node_bw=%.0f B/s node_traf=%.0f B/fwd host_bw=%s host_traf=%.0f" % (
            link["node_bw"], link["node_traf"],
            link["host_bw"] if link["host_bw"] else "n/a", link["host_traf"]))
        print("  worst-case t_transfer=%.3f s (%.1f%% of t_measured=%.2fs) -> link scales scores by %.3f..%.3f" % (
            max(r["link"]["t_transfer_s"] for r in rows),
            100*max(r["link"]["t_transfer_s"] for r in rows) / rows[0]["t_s"], rows[0]["t_s"],
            min(r["link"]["link_scale"] for r in rows), max(r["link"]["link_scale"] for r in rows)))
        print("  LINK-BOUND WEIGHTED SUM  MFU(mix)=%.1f%%  ExScore=%.1f%%" % (
            100*wsum_link_mix, 100*wsum_link_ex))
    # sensitivity to fp32-peak uncertainty
    hi = sum(w*score_case(fl, c["t_s"], soft=P_SOFTFP_HI, bw=BW_HI)["exscore_lo"]
             for w,c in zip(wts,cases))
    lo = sum(w*score_case(fl, c["t_s"], soft=P_SOFTFP_LO, bw=BW_HI)["exscore_lo"]
             for w,c in zip(wts,cases))
    print("  ExScore sensitivity to P_SOFTFP 1.5..2.7: %.1f%%..%.1f%%" % (100*lo, 100*hi))
    print("  ExScore sensitivity to BW 320MB/s: %.1f%%" % (100*sum(
        w*score_case(fl, c["t_s"], soft=P_SOFTFP, bw=BW_LO)["exscore_lo"]
        for w,c in zip(wts,cases))))

    out = {"flops": {k: v for k, v in fl.items()}, "runs": len(cases),
           "weighted_mfu_mix": wsum_mix, "weighted_mfu_raw_int": wsum_int,
           "weighted_exscore": wsum_ex, "peaks": {"P_INT": P_INT,
           "P_SOFTFP": P_SOFTFP, "BW": BW_HI}, "cases": rows}
    if link:
        out["link"] = {"node_bw": link["node_bw"], "node_traf": link["node_traf"],
                       "host_bw": link.get("host_bw"), "host_traf": link.get("host_traf"),
                       "weighted_mfu_mix_link": wsum_link_mix,
                       "weighted_exscore_link": wsum_link_ex}
    with open(a.output, "w") as output:
        json.dump(out, output, indent=1)
    print("\nwrote %s" % a.output)

if __name__ == "__main__":
    main()
