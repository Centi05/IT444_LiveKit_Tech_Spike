"""Summarize all spike CSVs into results/summary.csv, results/summary.md and results/spike_results.png.
Run from the repo root:  python analysis/summarize.py
"""
import glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob("data/with_inference/*.csv")):
    d = pd.read_csv(f)
    t = d.drop_duplicates("time")
    tag = os.path.basename(f).split("_")
    n, rate = int(tag[1][0]), tag[2].replace("fps", "").replace("p", ".")
    rows.append(dict(
        participants=n, target_fps=float(rate), intervals=len(t),
        infer_fps_mean=d.vid_infer_fps.mean(), infer_fps_min=d.vid_infer_fps.min(),
        dropped_fps_mean=d.vid_dropped_fps.mean(),
        model_ms_median=d.vid_infer_ms_med.median(), model_ms_p95=d.vid_infer_ms_p95.median(),
        cpu_mean=t.cpu_total_pct.mean(), cpu_max=t.cpu_total_pct.max(),
        intervals_over_50=int((t.cpu_total_pct > 50).sum()),
        temp_max=t.temp_c.max(), throttled=sorted(set(d.throttled))))
s = pd.DataFrame(rows).sort_values(["target_fps", "participants"], ascending=[False, True])
s["meets_target_fps"] = s.infer_fps_mean >= 0.95 * s.target_fps
s["meets_cpu_50"] = s.cpu_mean <= 50
s.round(2).to_csv("results/summary.csv", index=False)
with open("results/summary.md", "w") as fh:
    fh.write(s.round(2).to_markdown(index=False))

rcv = []
for f in sorted(glob.glob("data/receive_only/receive_*pub_*.csv")):
    d = pd.read_csv(f); t = d.drop_duplicates("time")
    rcv.append(dict(participants=int(d.n_streams.min()), recv_fps=d.recv_fps.mean(),
                    kept_fps=d.kept_fps.mean(), cpu_mean=t.cpu_total_pct.mean(), temp_max=t.temp_c.max()))
pd.DataFrame(rcv).round(2).to_csv("results/receive_only_summary.csv", index=False)

# figure: small multiples, shared x
C5, C25 = "#2a6fb0", "#c9721c"
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for rate, c, m in ((5.0, C5, "o"), (2.5, C25, "s")):
    g = s[s.target_fps == rate]
    ax[0].plot(g.participants, g.infer_fps_mean, c=c, marker=m, label=f"sampled at {rate:g} fps")
    ax[1].plot(g.participants, g.cpu_mean, c=c, marker=m, label=f"sampled at {rate:g} fps")
ax[0].set_title("Inference rate per participant (fps)", fontsize=10, loc="left")
ax[1].set_title("Whole-Pi CPU, mean (%)", fontsize=10, loc="left")
ax[1].axhline(50, c="#666", ls="--", lw=1); ax[1].text(1.02, 51.5, "NFR-01 limit 50%", fontsize=8, color="#444")
ax[0].axhline(5, c="#666", ls="--", lw=1); ax[0].text(1.02, 5.15, "spike target 5 fps", fontsize=8, color="#444")
for a in ax:
    a.set_xticks([1, 2, 3, 4]); a.set_xlabel("participants"); a.grid(alpha=.25)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
ax[0].set_ylim(0, 6); ax[1].set_ylim(0, 100); ax[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("results/spike_results.png", dpi=160)
print(s.round(2).to_string(index=False))
