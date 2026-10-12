#!/usr/bin/env python3
"""Generate the tracked figures embedded in REPORT.md from campaign aggregates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_DATA = ROOT / "report_data"
OUTPUT = ROOT / "report_assets"

ALGORITHMS = ["direct-concurrent", "spreadout", "roundrobin", "fast"]
LABELS = {
    "direct-concurrent": "Direct",
    "spreadout": "SpreadOut",
    "roundrobin": "RoundRobin",
    "fast": "FAST",
}
COLORS = {
    "direct-concurrent": "#4C78A8",
    "spreadout": "#F58518",
    "roundrobin": "#54A24B",
    "fast": "#E45756",
}


def rows(campaign: str) -> list[dict]:
    snapshot = REPORT_DATA / f"{campaign}.json"
    source = snapshot if snapshot.is_file() else RESULTS / campaign / "aggregate.json"
    return json.loads(source.read_text(encoding="utf-8"))


def finish(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def e2_balanced_large() -> None:
    data = rows("e2-one-seed")
    profiles = ["T0", "T1", "T2", "T3", "T4"]
    width = 0.19
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for index, algorithm in enumerate(ALGORITHMS):
        values = []
        for profile in profiles:
            match = next(
                (
                    row
                    for row in data
                    if row["profile_id"] == profile
                    and row["workload_id"] == "balanced-large"
                    and row["algorithm"] == algorithm
                ),
                None,
            )
            values.append(
                math.nan
                if match is None
                else match["actual_makespan_ns"] / 1_000_000
            )
        positions = x + (index - 1.5) * width
        ax.bar(
            positions,
            values,
            width,
            label=LABELS[algorithm],
            color=COLORS[algorithm],
        )
        for pos, value in zip(positions, values):
            if math.isnan(value):
                ax.text(pos, 0.012, "failed", rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, profiles)
    ax.set_ylabel("Collective makespan (ms, log scale)")
    ax.set_title("E2: balanced-large scheduler baseline")
    ax.legend(ncol=4, loc="upper left")
    finish(fig, "e2-balanced-large-makespan.png")


def e3_profile_distribution() -> None:
    data = [row for row in rows("e3-native-five-seed") if row["algorithm"] == "fast"]
    profiles = ["T0", "T1", "T2", "T3", "T4"]
    values = [
        [row["speedup_vs_spreadout"] for row in data if row["profile_id"] == profile]
        for profile in profiles
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    parts = ax.boxplot(values, labels=profiles, patch_artist=True, showmeans=True)
    for box in parts["boxes"]:
        box.set_facecolor(COLORS["fast"])
        box.set_alpha(0.55)
    for median in parts["medians"]:
        median.set_color("#222222")
    for index, series in enumerate(values, 1):
        ax.scatter(
            np.full(len(series), index),
            series,
            s=10,
            alpha=0.25,
            color="#8E2A2A",
            zorder=2,
        )
    ax.axhline(1, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylabel("FAST speedup vs SpreadOut")
    ax.set_xlabel("Topology profile")
    ax.set_title("E3: FAST speedup distribution across native workloads")
    finish(fig, "e3-fast-speedup-by-profile.png")


def e4_skew() -> None:
    data = [row for row in rows("e4-fixed-total") if row["algorithm"] == "fast"]
    workloads = [
        "balanced-fixed-large",
        "uniform-fixed-large",
        "zipf05-fixed-large",
        "zipf09-fixed-large",
    ]
    names = ["Balanced", "Uniform", "Zipf 0.5", "Zipf 0.9"]
    gini = [mean(row["cell_gini"] for row in data if row["workload_id"] == work) for work in workloads]
    amplification = [
        mean(row["schedule_byte_amplification"] for row in data if row["workload_id"] == work)
        for work in workloads
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for profile, marker, color in [("T3", "o", "#4C78A8"), ("T4", "s", "#E45756")]:
        speedup = [
            mean(
                row["speedup_vs_spreadout"]
                for row in data
                if row["profile_id"] == profile and row["workload_id"] == work
            )
            for work in workloads
        ]
        ax.plot(gini, speedup, marker=marker, linewidth=2.2, label=f"{profile} FAST/SpreadOut", color=color)
    ax.set_xlabel("Mean cell Gini (fixed 240 MiB)")
    ax.set_ylabel("FAST speedup vs SpreadOut")
    ax.set_title("E4: skew causality under a fixed byte budget")
    for xpos, label in zip(gini, names):
        ax.annotate(label, (xpos, 0.93), xycoords=("data", "axes fraction"), ha="center", fontsize=8)
    second = ax.twinx()
    second.plot(gini, amplification, color="#7A5195", marker="^", linestyle="--", linewidth=1.8, label="Byte amplification")
    second.set_ylabel("FAST scheduled-byte amplification")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left")
    finish(fig, "e4-skew-causality.png")


def e5_scaling() -> None:
    data = rows("e5-fixed-per-device-scale")
    profile_devices = {"T0": 4, "T1": 8, "T3": 16}
    workloads = [
        ("balanced-fixed-per-device", "Balanced"),
        ("zipf09-fixed-per-device", "Zipf 0.9"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (workload, title) in zip(axes, workloads):
        for algorithm in ALGORITHMS:
            values = []
            devices = []
            for profile in ["T0", "T1", "T3"]:
                selected = [
                    row["per_device_algorithm_bandwidth_gbps"]
                    for row in data
                    if row["profile_id"] == profile
                    and row["workload_id"] == workload
                    and row["algorithm"] == algorithm
                ]
                devices.append(profile_devices[profile])
                values.append(mean(selected))
            ax.plot(devices, values, marker="o", linewidth=2, label=LABELS[algorithm], color=COLORS[algorithm])
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16])
        ax.set_xlabel("Number of GPUs")
        ax.set_title(title)
    axes[0].set_ylabel("Per-GPU effective bandwidth (Gbps, log scale)")
    axes[1].legend(loc="upper right")
    fig.suptitle("E5: scaling at a fixed 15 MiB sent per GPU", y=1.02)
    finish(fig, "e5-scaling-bandwidth.png")


def e6_queues() -> None:
    data = rows("e4-fixed-total")
    profiles = ["T3", "T4"]
    width = 0.19
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for index, algorithm in enumerate(ALGORITHMS):
        values = []
        for profile in profiles:
            selected = [
                row["tor_peak_queue_bytes"] / (1024 * 1024)
                for row in data
                if row["profile_id"] == profile and row["algorithm"] == algorithm
            ]
            values.append(mean(selected))
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=LABELS[algorithm],
            color=COLORS[algorithm],
        )
    ax.set_yscale("log")
    ax.set_xticks(x, ["T3 (1:1)", "T4 (2:1)"])
    ax.set_ylabel("Mean ToR peak queue (MiB, log scale)")
    ax.set_title("E6: scheduler pressure at the ToR")
    ax.legend(ncol=2)
    finish(fig, "e6-tor-queue-peaks.png")


def e7_load() -> None:
    data = [row for row in rows("e7-load-sweep") if row["algorithm"] == "fast"]
    loads = [1, 15, 64]
    workloads = [
        ("balanced-perdev-{load}m", "Balanced", "#4C78A8"),
        ("zipf09-perdev-{load}m", "Zipf 0.9", "#E45756"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, profile in zip(axes, ["T1", "T4"]):
        for template, label, color in workloads:
            values = []
            for load in loads:
                work = template.format(load=load)
                row = next(
                    item
                    for item in data
                    if item["profile_id"] == profile and item["workload_id"] == work
                )
                values.append(row["speedup_vs_spreadout"])
            ax.plot(loads, values, marker="o", linewidth=2.2, label=label, color=color)
        ax.set_xscale("log")
        ax.set_xticks(loads, [str(value) for value in loads])
        ax.set_xlabel("Sent bytes per GPU (MiB, log scale)")
        ax.set_title(profile)
        ax.axhline(1, color="#333333", linewidth=1, linestyle="--")
    axes[0].set_ylabel("FAST speedup vs SpreadOut")
    axes[1].legend()
    fig.suptitle("E7: load-size sensitivity", y=1.02)
    finish(fig, "e7-load-sensitivity.png")


def e8_wins() -> None:
    campaigns = [("e3-native-five-seed", "E3"), ("e4-fixed-total", "E4")]
    labels: list[str] = []
    rates: list[float] = []
    counts: list[str] = []
    for campaign, short in campaigns:
        data = rows(campaign)
        scenarios: dict[str, list[dict]] = {}
        for row in data:
            scenarios.setdefault(row["case_id"], []).append(row)
        for baseline, baseline_label in [("spreadout", "SpreadOut"), ("direct-concurrent", "Direct")]:
            won = 0
            available = 0
            for scenario in scenarios.values():
                fast = next(row for row in scenario if row["algorithm"] == "fast")
                other = next((row for row in scenario if row["algorithm"] == baseline), None)
                if other is None:
                    continue
                available += 1
                won += fast["actual_makespan_ns"] < other["actual_makespan_ns"]
            labels.append(f"{short} vs\n{baseline_label}")
            rates.append(100 * won / available)
            counts.append(f"{won}/{available}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, rates, color=["#F58518", "#4C78A8", "#F58518", "#4C78A8"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, count, ha="center", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("FAST win rate (%)")
    ax.set_title("E8: FAST win rates (ties are not wins)")
    finish(fig, "e8-fast-win-rates.png")


def main() -> None:
    configure()
    e2_balanced_large()
    e3_profile_distribution()
    e4_skew()
    e5_scaling()
    e6_queues()
    e7_load()
    e8_wins()


if __name__ == "__main__":
    main()
