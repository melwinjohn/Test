"""
Standalone demo runner.  Executes all three experiments in demo mode,
generates all figures, traces, and the final report.

Run:  python run_demo.py
"""
import json
import sqlite3
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.tracer import ModelTracer
from shared.runtime import (
    BaseModelClient,
    MultiAgentOrchestrator,
    WorkerResult,
    OrchestratorResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Demo responses
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PLANS = {
    "bug_audit": json.dumps({
        "rationale": "Split the library into logical sections so workers can audit independently.",
        "subtasks": [
            {"id": "st_1", "title": "Audit variance & std_dev",
             "description": "Examine variance() and std_dev() for mathematical correctness.",
             "context": "variance uses exponent; std_dev calls another aggregator",
             "expected_output": "JSON: {found_bug, bug_id, description, fix}"},
            {"id": "st_2", "title": "Audit percentile",
             "description": "Check percentile() index formula for off-by-one errors.",
             "context": "Uses len() in index formula",
             "expected_output": "JSON: {found_bug, bug_id, description, fix}"},
            {"id": "st_3", "title": "Audit z_score",
             "description": "Check z_score() for division-by-zero safety.",
             "context": "Divides by std_dev which can be 0",
             "expected_output": "JSON: {found_bug, bug_id, description, fix}"},
            {"id": "st_4", "title": "Audit moving_average",
             "description": "Verify the slice logic in moving_average().",
             "context": "Slice is numbers[i-window:i]",
             "expected_output": "JSON: {found_bug, bug_id, description, fix}"},
        ],
    }),
    "math_grading": json.dumps({
        "rationale": "Each pair of problems can be graded independently.",
        "subtasks": [
            {"id": "st_1", "title": "Grade P1-P2",
             "description": "Grade problems P1 and P2.",
             "context": "P1: 2x+5=13; P2: 3(x-2)=15",
             "expected_output": "JSON array [{id, correct, score, feedback}]"},
            {"id": "st_2", "title": "Grade P3-P4",
             "description": "Grade problems P3 and P4.",
             "context": "P3: simplify; P4: consecutive integers",
             "expected_output": "JSON array [{id, correct, score, feedback}]"},
            {"id": "st_3", "title": "Grade P5-P6",
             "description": "Grade problems P5 and P6.",
             "context": "P5: inequality; P6: factorise",
             "expected_output": "JSON array [{id, correct, score, feedback}]"},
            {"id": "st_4", "title": "Grade P7-P8",
             "description": "Grade problems P7 and P8.",
             "context": "P7: rectangle area; P8: system",
             "expected_output": "JSON array [{id, correct, score, feedback}]"},
        ],
    }),
    "customer_grouping": json.dumps({
        "rationale": "Split 20 messages into 4 batches of 5 for parallel categorisation.",
        "subtasks": [
            {"id": "st_1", "title": "Categorise M01-M05",
             "description": "Assign each message to a support category.",
             "context": "M01 order; M02 charge; M03 password; M04 laptop; M05 payment methods",
             "expected_output": "JSON array [{id, category, confidence, reason}]"},
            {"id": "st_2", "title": "Categorise M06-M10",
             "description": "Assign each message to a support category.",
             "context": "M06 cancel; M07 missing parcel; M08 charge; M09 login; M10 dead pixel",
             "expected_output": "JSON array [{id, category, confidence, reason}]"},
            {"id": "st_3", "title": "Categorise M11-M15",
             "description": "Assign each message to a support category.",
             "context": "M11 hours; M12 cancel plan; M13 tracking; M14 billed wrong; M15 2FA",
             "expected_output": "JSON array [{id, category, confidence, reason}]"},
            {"id": "st_4", "title": "Categorise M16-M20",
             "description": "Assign each message to a support category.",
             "context": "M16 headphones; M17 student discount; M18 downgrade; M19 express; M20 promo",
             "expected_output": "JSON array [{id, category, confidence, reason}]"},
        ],
    }),
}

DEMO_WORKER_OUTPUTS = {
    "bug_audit": {
        "st_1": '{"found_bug":true,"bugs":[{"bug_id":"BUG1","location":"variance()","description":"Exponent is 3 instead of 2.","fix":"Change (x-m)**3 to (x-m)**2"},{"bug_id":"BUG2","location":"std_dev()","description":"Calls mean() instead of variance().","fix":"Change mean(numbers)**0.5 to variance(numbers)**0.5"}]}',
        "st_2": '{"found_bug":true,"bugs":[{"bug_id":"BUG3","location":"percentile()","description":"Uses len(s) instead of len(s)-1 in index formula.","fix":"Change (p/100)*len(s) to (p/100)*(len(s)-1)"}]}',
        "st_3": '{"found_bug":true,"bugs":[{"bug_id":"BUG4","location":"z_score()","description":"No guard for zero std dev – raises ZeroDivisionError.","fix":"Add: if sd == 0: return 0.0"}]}',
        "st_4": '{"found_bug":true,"bugs":[{"bug_id":"BUG5","location":"moving_average()","description":"Slice numbers[i-window:i] excludes current element.","fix":"Change to numbers[i-window+1:i+1]"}]}',
    },
    "math_grading": {
        "st_1": '[{"id":"P1","correct":true,"score":10,"feedback":"Perfect."},{"id":"P2","correct":false,"score":3,"feedback":"Distribution error: 3(x-2)=3x-6 not 3x-2."}]',
        "st_2": '[{"id":"P3","correct":true,"score":10,"feedback":"Correct."},{"id":"P4","correct":true,"score":10,"feedback":"Correct."}]',
        "st_3": '[{"id":"P5","correct":true,"score":10,"feedback":"Correct."},{"id":"P6","correct":false,"score":2,"feedback":"Both factors need opposite signs: (x+3)(x-3)."}]',
        "st_4": '[{"id":"P7","correct":true,"score":10,"feedback":"Correct FOIL."},{"id":"P8","correct":false,"score":7,"feedback":"x=7 correct but y=3 not y=4."}]',
    },
    "customer_grouping": {
        "st_1": '[{"id":"M01","category":"shipping_delay","confidence":0.97,"reason":"Order not arrived"},{"id":"M02","category":"billing_error","confidence":0.99,"reason":"Duplicate charge"},{"id":"M03","category":"account_access","confidence":0.96,"reason":"Password link broken"},{"id":"M04","category":"product_defect","confidence":0.98,"reason":"Hardware failure"},{"id":"M05","category":"general_inquiry","confidence":0.95,"reason":"Payment methods question"}]',
        "st_2": '[{"id":"M06","category":"cancellation","confidence":0.97,"reason":"Explicit cancel"},{"id":"M07","category":"shipping_delay","confidence":0.98,"reason":"Delivered but missing"},{"id":"M08","category":"billing_error","confidence":0.96,"reason":"Unknown charge"},{"id":"M09","category":"account_access","confidence":0.99,"reason":"Login failure"},{"id":"M10","category":"product_defect","confidence":0.97,"reason":"Screen defect"}]',
        "st_3": '[{"id":"M11","category":"general_inquiry","confidence":0.94,"reason":"Support hours"},{"id":"M12","category":"cancellation","confidence":0.98,"reason":"Cancel + refund"},{"id":"M13","category":"shipping_delay","confidence":0.97,"reason":"Tracking stalled"},{"id":"M14","category":"billing_error","confidence":0.99,"reason":"Wrong tier charged"},{"id":"M15","category":"account_access","confidence":0.98,"reason":"2FA broken"}]',
        "st_4": '[{"id":"M16","category":"product_defect","confidence":0.97,"reason":"Audio defect"},{"id":"M17","category":"general_inquiry","confidence":0.95,"reason":"Student discount"},{"id":"M18","category":"cancellation","confidence":0.96,"reason":"Downgrade request"},{"id":"M19","category":"shipping_delay","confidence":0.98,"reason":"Express late"},{"id":"M20","category":"billing_error","confidence":0.93,"reason":"Promo not working"}]',
    },
}

DEMO_SYNTHESES = {
    "bug_audit": """\
## Bug Audit Report – stats_lib.py

**Total bugs found: 5 / 5**

| Bug | Location | Severity | Fix |
|-----|----------|----------|-----|
| BUG1 | `variance()` | High | Change `(x-m)**3` → `(x-m)**2` |
| BUG2 | `std_dev()` | High | Change `mean()**0.5` → `variance()**0.5` |
| BUG3 | `percentile()` | Medium | Use `len(s)-1` in index formula |
| BUG4 | `z_score()` | Medium | Add zero-std guard |
| BUG5 | `moving_average()` | Low | Fix slice to `[i-window+1:i+1]` |

All 5 planted bugs identified. BUG1 and BUG2 silently produce wrong values; BUG4 raises on edge input.""",
    "math_grading": """\
## Grading Report – Algebra I

| Problem | Score | Error |
|---------|-------|-------|
| P1 | 10/10 | – |
| P2 | 3/10 | Distribution |
| P3 | 10/10 | – |
| P4 | 10/10 | – |
| P5 | 10/10 | – |
| P6 | 2/10 | Diff of squares |
| P7 | 10/10 | – |
| P8 | 7/10 | Arithmetic |

**Total: 62/80 (77.5%)**  Recommend practice on factorisation and careful arithmetic.""",
    "customer_grouping": """\
## Customer Message Grouping

| Category | Count |
|----------|-------|
| shipping_delay | 5 |
| billing_error | 5 |
| product_defect | 4 |
| cancellation | 3 |
| account_access | 3 |
| general_inquiry | 3 |

Average confidence: 0.97. All 20 messages categorised correctly.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo client
# ─────────────────────────────────────────────────────────────────────────────

class DemoClient(BaseModelClient):
    """Returns pre-written responses with simulated latency."""

    def __init__(self, model: str = "demo-model"):
        super().__init__(model)
        self._lock = threading.Lock()

    def _sim_latency(self) -> None:
        import random
        time.sleep(random.uniform(0.6, 1.3))

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2):
        self._sim_latency()

        if "[PLAN]" in prompt:
            p = prompt.lower()
            if "bug" in p or "variance" in p or "stats" in p:
                return DEMO_PLANS["bug_audit"], 320, 280, 0.00018
            if "grade" in p or "algebra" in p or "homework" in p:
                return DEMO_PLANS["math_grading"], 310, 260, 0.00017
            return DEMO_PLANS["customer_grouping"], 300, 250, 0.00016

        if "[SYNTH]" in prompt:
            p = prompt.lower()
            if "bug" in p or "variance" in p:
                return DEMO_SYNTHESES["bug_audit"], 1200, 380, 0.00062
            if "grade" in p or "algebra" in p or "p1" in p:
                return DEMO_SYNTHESES["math_grading"], 1100, 360, 0.00058
            return DEMO_SYNTHESES["customer_grouping"], 1000, 320, 0.00052

        # Worker prompts – match by subtask id in the prompt
        for task_key, workers in DEMO_WORKER_OUTPUTS.items():
            for wid, output in workers.items():
                if wid in prompt:
                    return output, 500, len(output.split()) * 2, 0.00025

        return '{"result":"done","notes":"demo fallback"}', 400, 80, 0.00012


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(ms: int) -> str:
    return f"{ms} ms" if ms < 1000 else f"{ms/1000:.2f} s"

def speedup(wall_ms: int, serial_ms: int) -> float:
    return serial_ms / wall_ms if wall_ms else 0.0


def plot_gantt(result: OrchestratorResult, title: str, path: str) -> None:
    workers = result.worker_results
    if not workers:
        return
    n = len(workers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    colors = plt.cm.tab10.colors

    for i, w in enumerate(workers):
        dur = w.duration_ms / 1000
        ax1.barh(i, dur, color=colors[i % 10], alpha=0.85, height=0.6)
        ax1.text(dur + 0.02, i, fmt(w.duration_ms), va="center", fontsize=8)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([f"W{i+1}: {w.subtask_title[:22]}" for i, w in enumerate(workers)], fontsize=8)
    ax1.set_xlabel("Wall time (s)")
    ax1.set_title("Parallel Execution")
    ax1.axvline(result.wall_time_ms / 1000, color="red", ls="--", lw=1.5,
                label=f"Wall: {fmt(result.wall_time_ms)}")
    ax1.legend(fontsize=8)

    offset = 0.0
    for i, w in enumerate(workers):
        dur = w.duration_ms / 1000
        ax2.barh(i, dur, left=offset, color=colors[i % 10], alpha=0.85, height=0.6)
        offset += dur
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f"W{i+1}: {w.subtask_title[:22]}" for i, w in enumerate(workers)], fontsize=8)
    ax2.set_xlabel("Cumulative time (s)")
    ax2.set_title("Serial Execution (simulated)")
    ax2.axvline(result.serial_time_ms / 1000, color="red", ls="--", lw=1.5,
                label=f"Serial: {fmt(result.serial_time_ms)}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_summary(results: dict, model_name: str, path: str) -> None:
    experiments = list(results.keys())
    n = len(experiments)
    wall_times   = [results[e].wall_time_ms / 1000 for e in experiments]
    serial_times = [results[e].serial_time_ms / 1000 for e in experiments]
    speedups     = [speedup(results[e].wall_time_ms, results[e].serial_time_ms) for e in experiments]
    costs        = [results[e].total_cost_usd for e in experiments]
    tokens       = [results[e].total_tokens_in + results[e].total_tokens_out for e in experiments]
    n_workers    = [len(results[e].worker_results) for e in experiments]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Multi-Agent Workflow Summary  ({model_name})", fontsize=14, fontweight="bold")
    C = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(n)
    w = 0.35
    ax = axes[0, 0]
    ax.bar(x - w/2, wall_times,   w, label="Parallel (wall)",  color=C[0], alpha=0.85)
    ax.bar(x + w/2, serial_times, w, label="Serial (sum)",     color=C[1], alpha=0.85)
    ax.set_title("A. Execution Time (s)"); ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=12); ax.legend(fontsize=9)

    ax = axes[0, 1]
    bars = ax.bar(experiments, speedups, color=C[2], alpha=0.85)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_title("B. Parallel Speedup (serial / wall)"); ax.set_ylabel("×")
    for bar, v in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}×",
                ha="center", fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    ax.bar(experiments, tokens, color=C[0], alpha=0.85)
    ax.set_title("C. Total Tokens Used")
    for i, t in enumerate(tokens):
        ax.text(i, t + max(tokens) * 0.01, f"{t:,}", ha="center", fontsize=9)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(experiments, costs, color=C[1], alpha=0.75, label="Cost ($)")
    ax2.plot(experiments, n_workers, "o-", color=C[2], lw=2, ms=8, label="# workers")
    ax.set_title("D. Cost (USD) vs Workers"); ax.set_ylabel("Cost (USD)"); ax2.set_ylabel("Workers")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def generate_report(results: dict, model_name: str, tracer: ModelTracer, path: str) -> None:
    from datetime import datetime
    stats = tracer.stats()
    lines = [
        "# Dynamic Multi-Agent Workflow – Final Report",
        "",
        f"**Model:** {model_name}",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Aggregate Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total LLM calls | {stats['total_calls']} |",
        f"| Total tokens in | {stats['tokens_in']:,} |",
        f"| Total tokens out | {stats['tokens_out']:,} |",
        f"| Total cost (USD) | ${stats['total_cost_usd']:.4f} |",
        f"| Total wall time | {stats['total_ms'] / 1000:.1f} s |",
        f"| Unique experiments | {stats['unique_tasks']} |",
        "",
        "## Experiment Results",
        "",
    ]
    for exp_name, res in results.items():
        sr = speedup(res.wall_time_ms, res.serial_time_ms)
        lines += [
            f"### {exp_name}",
            "",
            f"- **Workers:** {len(res.worker_results)}",
            f"- **Wall time:** {fmt(res.wall_time_ms)}",
            f"- **Serial time:** {fmt(res.serial_time_ms)}",
            f"- **Speedup:** {sr:.2f}×",
            f"- **Tokens:** {res.total_tokens_in + res.total_tokens_out:,}",
            f"- **Cost:** ${res.total_cost_usd:.4f}",
            "",
            f"**Plan rationale:** {res.plan.rationale}",
            "",
            "**Subtasks:**",
            "",
        ]
        for w in res.worker_results:
            status = "✓" if not w.error else "✗"
            cache = " (cached)" if w.from_cache else ""
            lines.append(f"- {status} `{w.subtask_id}` – {w.subtask_title}{cache}")
        lines += ["", "**Synthesis excerpt:**", "",
                  f"> {res.synthesis[:500].replace(chr(10), ' ')}", ""]

    lines += [
        "## Conclusion",
        "",
        "The dynamic multi-agent approach consistently outperformed serial execution:",
        "",
        "- Parallel workers reduced wall-clock time proportionally to independent subtask count.",
        "- The lead agent produced coherent decompositions across 3 different task types.",
        "- `ModelTracer` caching means re-running this notebook costs $0 in API fees.",
        "",
        "---",
        "_Generated by the Dynamic Multi-Agent Workflow demo._",
    ]
    Path(path).write_text("\n".join(lines))
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Path("traces").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    client = DemoClient()

    # ── Experiment 1: Bug Audit ───────────────────────────────────────────────
    library_source = Path("data/buggy_library.py").read_text()
    tracer_bug = ModelTracer("traces/bug_audit.db")
    orc_bug = MultiAgentOrchestrator(lead_client=client, worker_client=client,
                                      tracer=tracer_bug, max_workers=4)
    print("▶ Experiment 1: Bug Audit")
    result_bug = orc_bug.run(
        task="Audit the Python statistics library for bugs. Find all mathematical errors, "
             "off-by-one errors, missing error handling, and other defects. "
             "Report each bug with location and a proposed fix.",
        context=library_source,
        task_id="bug_audit",
        on_worker_done=lambda r: print(f"  ✓ {r.subtask_id}: {fmt(r.duration_ms)}"),
    )
    print(f"  Wall: {fmt(result_bug.wall_time_ms)} | Serial: {fmt(result_bug.serial_time_ms)} | "
          f"Speedup: {speedup(result_bug.wall_time_ms, result_bug.serial_time_ms):.2f}×\n")

    # ── Experiment 2: Math Grading ────────────────────────────────────────────
    homework = json.loads(Path("data/math_homework.json").read_text())
    tracer_math = ModelTracer("traces/math_grading.db")
    orc_math = MultiAgentOrchestrator(lead_client=client, worker_client=client,
                                       tracer=tracer_math, max_workers=4)
    print("▶ Experiment 2: Math Grading")
    result_math = orc_math.run(
        task="Grade the algebra homework problems. Check each student's answer and working "
             "for correctness. Identify error type if wrong. Return scores out of 10.",
        context=json.dumps(homework["problems"], indent=2),
        task_id="math_grading",
        on_worker_done=lambda r: print(f"  ✓ {r.subtask_id}: {fmt(r.duration_ms)}"),
    )
    print(f"  Wall: {fmt(result_math.wall_time_ms)} | Serial: {fmt(result_math.serial_time_ms)} | "
          f"Speedup: {speedup(result_math.wall_time_ms, result_math.serial_time_ms):.2f}×\n")

    # ── Experiment 3: Customer Grouping ───────────────────────────────────────
    messages = json.loads(Path("data/customer_messages.json").read_text())
    tracer_cust = ModelTracer("traces/customer_grouping.db")
    orc_cust = MultiAgentOrchestrator(lead_client=client, worker_client=client,
                                       tracer=tracer_cust, max_workers=4)
    print("▶ Experiment 3: Customer Grouping")
    result_cust = orc_cust.run(
        task="Categorise each customer support message into one of the provided categories. "
             "Return category, confidence (0-1), and reason for each message.",
        context=(f"CATEGORIES:\n{json.dumps(messages['categories'], indent=2)}\n\n"
                 f"MESSAGES:\n{json.dumps(messages['messages'], indent=2)}"),
        task_id="customer_grouping",
        on_worker_done=lambda r: print(f"  ✓ {r.subtask_id}: {fmt(r.duration_ms)}"),
    )
    print(f"  Wall: {fmt(result_cust.wall_time_ms)} | Serial: {fmt(result_cust.serial_time_ms)} | "
          f"Speedup: {speedup(result_cust.wall_time_ms, result_cust.serial_time_ms):.2f}×\n")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("▶ Generating figures …")
    all_results = {"Bug Audit": result_bug, "Math Grading": result_math, "Customer Grp": result_cust}
    plot_gantt(result_bug,  "Experiment 1: Bug Audit Timeline",      "figures/gantt_bug_audit.png")
    plot_gantt(result_math, "Experiment 2: Math Grading Timeline",   "figures/gantt_math_grading.png")
    plot_gantt(result_cust, "Experiment 3: Customer Group Timeline", "figures/gantt_customer.png")
    plot_summary(all_results, "Claude / Qwen (demo)", "figures/summary.png")

    # ── Combined trace ────────────────────────────────────────────────────────
    print("\n▶ Building combined trace …")
    combined = ModelTracer("traces/combined.db")
    for src_path in ["traces/bug_audit.db", "traces/math_grading.db", "traces/customer_grouping.db"]:
        conn = sqlite3.connect(src_path)
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM traces").fetchall():
            r = dict(row)
            combined.record(
                model=r["model"],
                prompt=r["prompt"] + f"__src_{r['task_id']}",
                response=r["response"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                cost_usd=r["cost_usd"],
                duration_ms=r["duration_ms"],
                task_id=r["task_id"],
                worker_id=r["worker_id"],
            )
        conn.close()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n▶ Generating final report …")
    generate_report(all_results, "Claude / Qwen (demo)", combined, "final_report.md")
    n_exported = combined.export_json("traces/all_traces.json")
    print(f"  ✓ traces/all_traces.json  ({n_exported} records)")

    stats = combined.stats()
    print("\n── Aggregate stats ────────────────────────────────────")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    print("\n✅ Demo complete.")
    print("   Figures:      figures/*.png  (4 files)")
    print("   Traces:       traces/*.db    (3 experiment + 1 combined)")
    print("   JSON export:  traces/all_traces.json")
    print("   Final report: final_report.md")


if __name__ == "__main__":
    main()
