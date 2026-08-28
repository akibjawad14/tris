"""
Plot Exp4 results:
  Figure 1 (exp4a): ASR vs adv_per_query — baseline vs Tri-Layer Sieve
  Figure 2 (exp4b): ASR heatmap — top_k × cluster_count (defense ASR reduction)

Usage:
  python plot_exp4.py --outdir figures/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def plot_asr_vs_injection(results_base: Path, outdir: Path):
    apq_values = [1, 2, 3, 5, 10]
    baseline_asrs, defense_asrs = [], []

    for apq in apq_values:
        summary_path = results_base / f"exp4_sweep_apq{apq}" / f"summary_apq{apq}.json"
        if not summary_path.exists():
            print(f"[WARN] Missing: {summary_path} — skipping apq={apq}")
            baseline_asrs.append(None)
            defense_asrs.append(None)
            continue
        s = load_summary(summary_path)
        baseline_asrs.append(s.get("baseline_run", {}).get("asr"))
        defense_asrs.append(s.get("defense_run", {}).get("asr"))

    valid = [(a, b, d) for a, b, d in zip(apq_values, baseline_asrs, defense_asrs)
             if b is not None and d is not None]
    if not valid:
        print("[WARN] No valid data for Exp4a — skipping plot.")
        return

    xs, bs, ds = zip(*valid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, bs, marker="o", label="No Defense (Baseline)", color="tab:red", linewidth=2)
    ax.plot(xs, ds, marker="s", label="Tri-Layer Sieve", color="tab:blue", linewidth=2)
    ax.fill_between(xs, ds, bs, alpha=0.12, color="tab:blue")
    ax.set_xlabel("Injected adversarial docs per query ($m$)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_title("ASR vs. Injection Ratio — NQ (GPT-3.5)", fontsize=13)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out_path = outdir / "exp4a_asr_vs_injection.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_heatmap(results_base: Path, outdir: Path):
    topk_values = [3, 5, 10, 20]
    cluster_values = [2, 3, 5]

    # defense ASR and delta (baseline_asr - defense_asr) grids
    defense_grid = np.full((len(topk_values), len(cluster_values)), np.nan)
    delta_grid = np.full((len(topk_values), len(cluster_values)), np.nan)

    for ki, k in enumerate(topk_values):
        for ci, c in enumerate(cluster_values):
            summary_path = (
                results_base / f"exp4_heatmap_k{k}_c{c}" / f"defense_heatmap_k{k}_c{c}.json"
            )
            # run_defense.py sweep naming: <defense_name>_k{k}_c{c}.json
            if not summary_path.exists():
                # try alternate location produced by sweep
                alt = results_base / f"exp4_heatmap_k{k}_c{c}" / "defense_heatmap.json"
                if alt.exists():
                    summary_path = alt

            # summary JSON from --make_summary is not written in sweep mode;
            # fall back to summarizing inline
            asr_path = results_base / f"exp4_heatmap_k{k}_c{c}" / f"defense_heatmap_k{k}_c{c}.json"
            baseline_path = results_base / f"exp4_heatmap_k{k}_c{c}" / f"baseline_heatmap_k{k}_c{c}.json"

            if not asr_path.exists():
                print(f"[WARN] Missing defense results for k={k}, c={c}: {asr_path}")
                continue

            defense_asr = _compute_asr_from_result_json(asr_path)
            defense_grid[ki, ci] = defense_asr

            if baseline_path.exists():
                baseline_asr = _compute_asr_from_result_json(baseline_path)
                delta_grid[ki, ci] = baseline_asr - defense_asr

    _save_heatmap(
        defense_grid,
        row_labels=[f"k={k}" for k in topk_values],
        col_labels=[f"C={c}" for c in cluster_values],
        title="Defense ASR (↓ better) — NQ",
        out_path=outdir / "exp4b_heatmap_defense_asr.pdf",
        cmap="YlOrRd",
        fmt=".2f",
    )

    if not np.all(np.isnan(delta_grid)):
        _save_heatmap(
            delta_grid,
            row_labels=[f"k={k}" for k in topk_values],
            col_labels=[f"C={c}" for c in cluster_values],
            title="ASR Reduction (Baseline − Defense, ↑ better) — NQ",
            out_path=outdir / "exp4b_heatmap_asr_reduction.pdf",
            cmap="YlGn",
            fmt="+.2f",
        )


def _compute_asr_from_result_json(path):
    with open(path) as f:
        raw = json.load(f)
    hits, total = 0, 0
    for iter_block in raw:
        for entries in iter_block.values():
            for row in entries:
                output = str(row.get("output_poison", row.get("output", ""))).lower()
                incorrect = str(row.get("incorrect_answer", "")).lower()
                if incorrect and incorrect in output:
                    hits += 1
                total += 1
    return hits / total if total > 0 else 0.0


def _save_heatmap(data, row_labels, col_labels, title, out_path, cmap, fmt):
    fig, ax = plt.subplots(figsize=(5, 4))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Cluster count (C)", fontsize=12)
    ax.set_ylabel("Retrieval top-k", fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)

    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val = data[ri, ci]
            if not np.isnan(val):
                text = format(val, fmt)
                ax.text(ci, ri, text, ha="center", va="center", fontsize=10,
                        color="black" if val < 0.6 else "white")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="results/query_results",
                        help="Base results directory")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Plotting Exp4a: ASR vs injection ratio ===")
    plot_asr_vs_injection(results_base, outdir)

    print("=== Plotting Exp4b: top_k x cluster_count heatmap ===")
    plot_heatmap(results_base, outdir)

    print("Done.")


if __name__ == "__main__":
    main()
