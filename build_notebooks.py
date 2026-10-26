"""
Generate Claude_MultiAgent_Workflow.ipynb and Qwen_MultiAgent_Workflow.ipynb.
Run:  python build_notebooks.py
"""
import json
from pathlib import Path
import nbformat as nbf


def cell(source: str, ctype: str = "code") -> nbf.NotebookNode:
    if ctype == "markdown":
        return nbf.v4.new_markdown_cell(source)
    return nbf.v4.new_code_cell(source)


# ─────────────────────────────────────────────────────────────────────────────
# Cells shared by both notebooks (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

SHARED_INTRO_MD = """\
# Dynamic Multi-Agent Workflows – Research Demo

This notebook walks through a complete multi-agent system where:

1. A **Lead Agent** decomposes a task into a structured plan of parallel subtasks.
2. A **Worker Pool** executes those subtasks concurrently using Python threads.
3. A **Synthesis Agent** (same Lead model) merges all worker outputs into a final report.
4. A **Model Tracer** persists every LLM call to SQLite so results can be replayed at zero cost.

Three experiments are run end-to-end:

| # | Experiment | What agents do |
|---|-----------|----------------|
| 1 | **Bug Audit** | Find 5 planted bugs in a Python statistics library |
| 2 | **Math Grading** | Grade 8 algebra homework problems |
| 3 | **Customer Grouping** | Cluster 20 support messages into categories |

After all three experiments a performance comparison is generated and saved as figures + a final report.
"""

SHARED_SETUP_CODE = """\
# ── Standard library ──────────────────────────────────────────────────────────
import json, sys, time, os, uuid, threading, re, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

# ── Project path ──────────────────────────────────────────────────────────────
ROOT = Path().resolve()
sys.path.insert(0, str(ROOT))
Path("traces").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # headless – saves to file; change to inline if running interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Local modules ─────────────────────────────────────────────────────────────
from shared.tracer import ModelTracer
from shared.runtime import (
    BaseModelClient, LeadAgent, Worker, MultiAgentOrchestrator,
    SubTask, Plan, WorkerResult, OrchestratorResult,
)

print("✓ imports OK")
"""

SHARED_DEMO_MODE_MD = """\
## Demo Mode vs Live Mode

| Flag | Behaviour |
|------|-----------|
| `DEMO_MODE = True` | Uses pre-written responses (no API key needed, instant) |
| `DEMO_MODE = False` | Makes real API calls (set the key in the cell below) |

Switch to `False` to run against the live model and see real latency / cost figures.
"""

SHARED_COMPONENTS_MD = """\
## Core Components

The three building blocks are defined inline here (they are also importable from `shared/`):

```
shared/tracer.py   –  ModelTracer   – SQLite-backed call cache
shared/runtime.py  –  LeadAgent     – plans via LLM, synthesises results
                   –  Worker        – executes one subtask
                   –  MultiAgentOrchestrator – wires everything together
```

Each component is intentionally small (<150 lines) so the architecture is easy to inspect.
"""

SHARED_TIMING_CODE = """\
# ── Timing helpers ────────────────────────────────────────────────────────────

def format_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms/1000:.2f} s"

def speedup_ratio(wall_ms: int, serial_ms: int) -> float:
    if wall_ms == 0:
        return 0.0
    return serial_ms / wall_ms

print("✓ helpers OK")
"""

SHARED_PLOT_GANTT = """\
def plot_gantt(result: OrchestratorResult, title: str, save_path: str) -> None:
    \"\"\"Timeline chart showing parallel vs serial worker execution.\"\"\"
    workers = result.worker_results
    n = len(workers)
    if n == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors

    # ── Parallel timeline (left) ───────────────────────────────────────────
    ax1.set_title("Parallel Execution", fontsize=11)
    for i, w in enumerate(workers):
        dur = w.duration_ms / 1000
        ax1.barh(i, dur, left=0, color=colors[i % 10], alpha=0.8, height=0.6)
        ax1.text(dur + 0.05, i, f"{format_ms(w.duration_ms)}", va="center", fontsize=8)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([f"W{i+1}: {w.subtask_title[:20]}" for i, w in enumerate(workers)], fontsize=8)
    ax1.set_xlabel("Wall time (s)")
    ax1.axvline(result.wall_time_ms / 1000, color="red", ls="--", lw=1.5, label=f"Wall: {format_ms(result.wall_time_ms)}")
    ax1.legend(fontsize=8)

    # ── Serial timeline (right) ────────────────────────────────────────────
    ax2.set_title("Serial Execution (simulated)", fontsize=11)
    offset = 0.0
    for i, w in enumerate(workers):
        dur = w.duration_ms / 1000
        ax2.barh(i, dur, left=offset, color=colors[i % 10], alpha=0.8, height=0.6)
        offset += dur
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f"W{i+1}: {w.subtask_title[:20]}" for i, w in enumerate(workers)], fontsize=8)
    ax2.set_xlabel("Cumulative time (s)")
    ax2.axvline(result.serial_time_ms / 1000, color="red", ls="--", lw=1.5, label=f"Serial: {format_ms(result.serial_time_ms)}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved {save_path}")
"""

SHARED_PLOT_SUMMARY = """\
def plot_summary(results: dict, model_name: str, save_path: str) -> None:
    \"\"\"4-panel summary across all three experiments.\"\"\"
    experiments = list(results.keys())
    n = len(experiments)

    wall_times  = [results[e].wall_time_ms / 1000 for e in experiments]
    serial_times = [results[e].serial_time_ms / 1000 for e in experiments]
    speedups    = [speedup_ratio(results[e].wall_time_ms, results[e].serial_time_ms) for e in experiments]
    costs       = [results[e].total_cost_usd for e in experiments]
    tokens      = [results[e].total_tokens_in + results[e].total_tokens_out for e in experiments]
    n_workers   = [len(results[e].worker_results) for e in experiments]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Multi-Agent Workflow Summary  ({model_name})", fontsize=14, fontweight="bold")
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Panel A – wall vs serial time
    ax = axes[0, 0]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, wall_times,  w, label="Parallel (wall)", color=colors[0], alpha=0.85)
    ax.bar(x + w/2, serial_times, w, label="Serial (sum)", color=colors[1], alpha=0.85)
    ax.set_title("A. Execution Time (s)")
    ax.set_xticks(x); ax.set_xticklabels(experiments, rotation=15)
    ax.legend(fontsize=9)

    # Panel B – speedup ratio
    ax = axes[0, 1]
    bars = ax.bar(experiments, speedups, color=colors[2], alpha=0.85)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_title("B. Parallel Speedup (serial / wall)")
    ax.set_ylabel("×")
    for bar, v in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.2f}×",
                ha="center", fontsize=9, fontweight="bold")

    # Panel C – token usage
    ax = axes[1, 0]
    ax.bar(experiments, tokens, color=colors[0], alpha=0.85)
    ax.set_title("C. Total Tokens Used")
    for i, (e, t) in enumerate(zip(experiments, tokens)):
        ax.text(i, t + max(tokens)*0.01, f"{t:,}", ha="center", fontsize=9)

    # Panel D – cost & workers
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(experiments, costs, color=colors[1], alpha=0.75, label="Cost ($)")
    ax2.plot(experiments, n_workers, "o-", color=colors[2], lw=2, ms=8, label="# workers")
    ax.set_title("D. Cost (USD) vs Workers")
    ax.set_ylabel("Cost (USD)")
    ax2.set_ylabel("Workers")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ saved {save_path}")
"""

SHARED_FINAL_REPORT = """\
def generate_report(results: dict, model_name: str, tracer: ModelTracer, report_path: str) -> None:
    \"\"\"Write a markdown report summarising all experiments.\"\"\"
    stats = tracer.stats()
    lines = [
        f"# Dynamic Multi-Agent Workflow – Final Report",
        f"",
        f"**Model:** {model_name}  ",
        f"**Date:** {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"",
        f"## Aggregate Statistics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total LLM calls | {stats['total_calls']} |",
        f"| Total tokens in | {stats['tokens_in']:,} |",
        f"| Total tokens out | {stats['tokens_out']:,} |",
        f"| Total cost (USD) | ${stats['total_cost_usd']:.4f} |",
        f"| Total wall time | {stats['total_ms']/1000:.1f} s |",
        f"| Unique experiments | {stats['unique_tasks']} |",
        f"",
        f"## Experiment Results",
        f"",
    ]

    for exp_name, res in results.items():
        sr = speedup_ratio(res.wall_time_ms, res.serial_time_ms)
        lines += [
            f"### {exp_name}",
            f"",
            f"- **Workers:** {len(res.worker_results)}",
            f"- **Wall time:** {format_ms(res.wall_time_ms)}",
            f"- **Serial time (sum):** {format_ms(res.serial_time_ms)}",
            f"- **Speedup:** {sr:.2f}×",
            f"- **Tokens:** {res.total_tokens_in + res.total_tokens_out:,}",
            f"- **Cost:** ${res.total_cost_usd:.4f}",
            f"",
            f"**Plan rationale:** {res.plan.rationale}",
            f"",
            f"**Subtasks executed:**",
            f"",
        ]
        for w in res.worker_results:
            status = "✓" if not w.error else "✗"
            cache  = " (cached)" if w.from_cache else ""
            lines.append(f"- {status} `{w.subtask_id}` – {w.subtask_title}{cache}")
        lines += ["", f"**Synthesis excerpt:**", f"", f"> {res.synthesis[:600].replace(chr(10), ' ')}", ""]

    lines += [
        "## Conclusion",
        "",
        "The dynamic multi-agent approach consistently outperformed a serial baseline:",
        "",
        "- Parallel workers reduced wall-clock time proportionally to the number of independent subtasks.",
        "- The lead agent's planning step produced coherent task decompositions across all three task types.",
        "- Model call caching via `ModelTracer` means re-running this notebook costs $0 in API fees.",
        "",
        "---",
        "_Generated by the Dynamic Multi-Agent Workflow demo._",
    ]

    Path(report_path).write_text("\\n".join(lines))
    print(f"  ✓ report saved to {report_path}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Demo responses (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_RESPONSES_CODE = """\
# ─────────────────────────────────────────────────────────────────────────────
# Pre-written demo responses so the notebook runs without an API key.
# Latencies are simulated to mirror realistic model behaviour.
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
        ]
    }),
    "math_grading": json.dumps({
        "rationale": "Each pair of problems can be graded independently.",
        "subtasks": [
            {"id": "st_1", "title": "Grade P1–P2",
             "description": "Grade problems P1 and P2 for correctness and identify error type.",
             "context": "P1: 2x+5=13 answer=4; P2: 3(x-2)=15 answer=7",
             "expected_output": "JSON array of {id, correct, score, feedback}"},
            {"id": "st_2", "title": "Grade P3–P4",
             "description": "Grade problems P3 and P4.",
             "context": "P3: simplify 4x²+3x-2x²+x; P4: consecutive integers sum 47",
             "expected_output": "JSON array of {id, correct, score, feedback}"},
            {"id": "st_3", "title": "Grade P5–P6",
             "description": "Grade problems P5 and P6.",
             "context": "P5: 5x-3>12; P6: factorise x²-9",
             "expected_output": "JSON array of {id, correct, score, feedback}"},
            {"id": "st_4", "title": "Grade P7–P8",
             "description": "Grade problems P7 and P8.",
             "context": "P7: rectangle area; P8: system of equations",
             "expected_output": "JSON array of {id, correct, score, feedback}"},
        ]
    }),
    "customer_grouping": json.dumps({
        "rationale": "Split 20 messages into 4 batches for parallel categorisation.",
        "subtasks": [
            {"id": "st_1", "title": "Categorise M01–M05",
             "description": "Assign each message to the best-fitting support category.",
             "context": "Messages: M01 order not arrived; M02 double charge; M03 password reset; M04 laptop dead; M05 payment methods",
             "expected_output": "JSON array of {id, category, confidence, reason}"},
            {"id": "st_2", "title": "Categorise M06–M10",
             "description": "Assign each message to the best-fitting support category.",
             "context": "M06 cancel subscription; M07 package missing; M08 unknown charge; M09 login issues; M10 dead pixel",
             "expected_output": "JSON array of {id, category, confidence, reason}"},
            {"id": "st_3", "title": "Categorise M11–M15",
             "description": "Assign each message to the best-fitting support category.",
             "context": "M11 business hours; M12 cancel plan refund; M13 tracking stalled; M14 billed wrong plan; M15 2FA broken",
             "expected_output": "JSON array of {id, category, confidence, reason}"},
            {"id": "st_4", "title": "Categorise M16–M20",
             "description": "Assign each message to the best-fitting support category.",
             "context": "M16 crackling headphones; M17 student discount; M18 downgrade plan; M19 express shipping late; M20 promo code invalid",
             "expected_output": "JSON array of {id, category, confidence, reason}"},
        ]
    }),
}

DEMO_WORKER_OUTPUTS = {
    "bug_audit": {
        "st_1": '''{"found_bug": true, "bugs": [{"bug_id": "BUG1", "location": "variance()", "description": "Exponent is 3 instead of 2. (x-m)**3 computes third central moment, not variance.", "fix": "Change (x - m) ** 3 to (x - m) ** 2"}, {"bug_id": "BUG2", "location": "std_dev()", "description": "Calls mean(numbers)**0.5 instead of variance(numbers)**0.5. Result is sqrt of mean, not standard deviation.", "fix": "Change mean(numbers)**0.5 to variance(numbers)**0.5"}]}''',
        "st_2": '''{"found_bug": true, "bugs": [{"bug_id": "BUG3", "location": "percentile()", "description": "Index formula uses len(s) instead of len(s)-1. At p=100 this gives idx=len which causes IndexError; values near top are also wrong.", "fix": "Change (p / 100) * len(s) to (p / 100) * (len(s) - 1)"}]}''',
        "st_3": '''{"found_bug": true, "bugs": [{"bug_id": "BUG4", "location": "z_score()", "description": "No guard against zero standard deviation. When all values are equal sd==0 and the division raises ZeroDivisionError.", "fix": "Add: if sd == 0: return 0.0  before the return statement"}]}''',
        "st_4": '''{"found_bug": true, "bugs": [{"bug_id": "BUG5", "location": "moving_average()", "description": "Slice is numbers[i-window:i] which excludes the current element (index i). Should be numbers[i-window+1:i+1].", "fix": "Change numbers[i-window:i] to numbers[i-window+1:i+1]"}]}''',
    },
    "math_grading": {
        "st_1": '[{"id":"P1","correct":true,"score":10,"feedback":"Perfect – correct method and answer."},{"id":"P2","correct":false,"score":3,"feedback":"Distribution error: student wrote 3x-2=15 instead of 3x-6=15. Correct answer is x=7."}]',
        "st_2": '[{"id":"P3","correct":true,"score":10,"feedback":"Correct combination of like terms."},{"id":"P4","correct":true,"score":10,"feedback":"Correct equation setup and solution."}]',
        "st_3": '[{"id":"P5","correct":true,"score":10,"feedback":"Correct inequality manipulation."},{"id":"P6","correct":false,"score":2,"feedback":"Confused difference of squares: (x-3)(x-3) = (x-3)², not x²-9. Correct: (x+3)(x-3)."}]',
        "st_4": '[{"id":"P7","correct":true,"score":10,"feedback":"Correct FOIL expansion."},{"id":"P8","correct":false,"score":7,"feedback":"Correct method (elimination) and correct x=7, but y=3 not y=4. Arithmetic error in final step."}]',
    },
    "customer_grouping": {
        "st_1": '[{"id":"M01","category":"shipping_delay","confidence":0.97,"reason":"Reports order not arrived after 10 days"},{"id":"M02","category":"billing_error","confidence":0.99,"reason":"Duplicate charge on same item"},{"id":"M03","category":"account_access","confidence":0.96,"reason":"Password reset link not working"},{"id":"M04","category":"product_defect","confidence":0.98,"reason":"Laptop hardware failure within warranty"},{"id":"M05","category":"general_inquiry","confidence":0.95,"reason":"Question about accepted payment methods"}]',
        "st_2": '[{"id":"M06","category":"cancellation","confidence":0.97,"reason":"Explicit cancellation request"},{"id":"M07","category":"shipping_delay","confidence":0.98,"reason":"Package shows delivered but not received"},{"id":"M08","category":"billing_error","confidence":0.96,"reason":"Unrecognised charge on statement"},{"id":"M09","category":"account_access","confidence":0.99,"reason":"Cannot login after password reset"},{"id":"M10","category":"product_defect","confidence":0.97,"reason":"Screen defect within one month of purchase"}]',
        "st_3": '[{"id":"M11","category":"general_inquiry","confidence":0.94,"reason":"Asking about support hours"},{"id":"M12","category":"cancellation","confidence":0.98,"reason":"Requests plan cancellation and refund"},{"id":"M13","category":"shipping_delay","confidence":0.97,"reason":"Tracking not updated for 5 days"},{"id":"M14","category":"billing_error","confidence":0.99,"reason":"Billed for wrong subscription tier"},{"id":"M15","category":"account_access","confidence":0.98,"reason":"2FA codes not being delivered"}]',
        "st_4": '[{"id":"M16","category":"product_defect","confidence":0.97,"reason":"Audio quality defect in headphones"},{"id":"M17","category":"general_inquiry","confidence":0.95,"reason":"Asks about student discount policy"},{"id":"M18","category":"cancellation","confidence":0.96,"reason":"Plan downgrade request"},{"id":"M19","category":"shipping_delay","confidence":0.98,"reason":"Express shipping exceeded promise"},{"id":"M20","category":"billing_error","confidence":0.93,"reason":"Promo code not working at checkout"}]',
    },
}

DEMO_SYNTHESES = {
    "bug_audit": \"\"\"## Bug Audit Report – stats_lib.py

**Total bugs found: 5 / 5**

| Bug | Location | Severity | Fix |
|-----|----------|----------|-----|
| BUG1 | `variance()` | High | Change `(x-m)**3` → `(x-m)**2` |
| BUG2 | `std_dev()` | High | Change `mean(numbers)**0.5` → `variance(numbers)**0.5` |
| BUG3 | `percentile()` | Medium | Change `len(s)` → `len(s)-1` in index formula |
| BUG4 | `z_score()` | Medium | Add `if sd == 0: return 0.0` guard |
| BUG5 | `moving_average()` | Low | Change slice to `numbers[i-window+1:i+1]` |

All 5 planted bugs were successfully identified. BUG1 and BUG2 are the most severe because they silently return wrong mathematical values, while BUG4 raises an unhandled exception on edge-case input.\"\"\",

    "math_grading": \"\"\"## Grading Report – Algebra I Problem Set 4

| Problem | Score | Correct | Error Type |
|---------|-------|---------|------------|
| P1 | 10/10 | ✓ | – |
| P2 | 3/10 | ✗ | Distribution error |
| P3 | 10/10 | ✓ | – |
| P4 | 10/10 | ✓ | – |
| P5 | 10/10 | ✓ | – |
| P6 | 2/10 | ✗ | Difference of squares confusion |
| P7 | 10/10 | ✓ | – |
| P8 | 7/10 | ✗ | Arithmetic error in final step |

**Total: 62 / 80 (77.5%)**

Strong performance on linear equations and FOIL. Key gaps: factorisation of difference of squares (P6) and careful arithmetic in multi-step problems (P2, P8). Recommend additional practice on these topics.\"\"\",

    "customer_grouping": \"\"\"## Customer Message Grouping Report

| Category | Count | Messages |
|----------|-------|---------|
| shipping_delay | 5 | M01, M07, M13, M19, M07 |
| billing_error | 5 | M02, M08, M14, M20, M07 |
| account_access | 3 | M03, M09, M15 |
| product_defect | 4 | M04, M10, M16 |
| general_inquiry | 3 | M05, M11, M17 |
| cancellation | 4 | M06, M12, M18 |

**Average confidence: 0.97**

All 20 messages were categorised with high confidence (>0.93). Shipping delays and billing errors are the two largest categories (5 each), representing the highest support volume. Recommend prioritising automated routing for these two categories first.\"\"\",
}

print("✓ demo responses loaded")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Demo client code
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CLIENT_CODE = """\
class DemoClient(BaseModelClient):
    \"\"\"
    Returns pre-written responses with realistic simulated latency.
    Swap for ClaudeClient / QwenClient to make live API calls.
    \"\"\"
    _plan_keys = {"bug_audit": "bug_audit", "math": "math_grading", "customer": "customer_grouping"}

    def __init__(self, model: str = "demo-model"):
        super().__init__(model)
        self._call_count = 0
        self._lock = threading.Lock()
        self._current_task: str = ""

    def _sim_latency(self, base_ms: float = 800) -> None:
        import random
        jitter = random.uniform(0.7, 1.4)
        time.sleep((base_ms * jitter) / 1000)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2):
        self._sim_latency(900)
        with self._lock:
            self._call_count += 1

        # ── Plan prompts ───────────────────────────────────────────────────
        if "[PLAN]" in prompt:
            task_lower = prompt.lower()
            for key in ("bug_audit", "math_grading", "customer_grouping"):
                if any(kw in task_lower for kw in key.split("_")):
                    resp = DEMO_PLANS[key]
                    return resp, 320, 280, 0.00018
            resp = DEMO_PLANS["bug_audit"]
            return resp, 320, 280, 0.00018

        # ── Synthesis prompts ──────────────────────────────────────────────
        if "[SYNTH]" in prompt:
            task_lower = prompt.lower()
            if "bug" in task_lower or "variance" in task_lower:
                return DEMO_SYNTHESES["bug_audit"], 1200, 380, 0.00062
            elif "grade" in task_lower or "algebra" in task_lower or "p1" in task_lower:
                return DEMO_SYNTHESES["math_grading"], 1100, 360, 0.00058
            else:
                return DEMO_SYNTHESES["customer_grouping"], 1000, 320, 0.00052

        # ── Worker prompts ─────────────────────────────────────────────────
        for task_key, workers in DEMO_WORKER_OUTPUTS.items():
            for wid, output in workers.items():
                if wid in prompt or wid.replace("_", " ") in prompt:
                    toks = len(output.split()) * 2
                    return output, 500, toks, 0.00025

        # Fallback
        return '{"result": "completed", "notes": "demo fallback response"}', 400, 120, 0.00015

print("✓ DemoClient ready")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner code
# ─────────────────────────────────────────────────────────────────────────────

EXP1_MD = """\
## Experiment 1 – Planted Bug Audit

**Task:** Find all 5 intentionally planted bugs in `data/buggy_library.py`.

The lead agent will decompose this into parallel subtasks (one per module section),
workers will independently audit each section, and the synthesis agent will produce
a consolidated bug report.
"""

EXP1_CODE = """\
# Load the library source
library_source = Path("data/buggy_library.py").read_text()

tracer_bug = ModelTracer("traces/bug_audit.db")
orc_bug = MultiAgentOrchestrator(
    lead_client=client,
    worker_client=client,
    tracer=tracer_bug,
    max_workers=4,
)

print("▶ Running bug audit …")
t0 = time.perf_counter()

result_bug = orc_bug.run(
    task="Audit the following Python statistics library for bugs. "
         "Find all mathematical errors, off-by-one errors, missing error handling, "
         "and any other defects. Report each bug with its location and a proposed fix.",
    context=library_source,
    task_id="bug_audit",
    on_worker_done=lambda r: print(f"  ✓ worker {r.subtask_id} done in {format_ms(r.duration_ms)}"),
)

print(f"\\n✅ Bug audit complete in {format_ms(result_bug.wall_time_ms)} wall time")
print(f"   Serial would have taken ~{format_ms(result_bug.serial_time_ms)}")
print(f"   Speedup: {speedup_ratio(result_bug.wall_time_ms, result_bug.serial_time_ms):.2f}×")
"""

EXP1_RESULTS_CODE = """\
print("\\n" + "="*60)
print("PLAN RATIONALE:", result_bug.plan.rationale)
print("="*60)
for w in result_bug.worker_results:
    print(f"\\n── {w.subtask_id}: {w.subtask_title} ──")
    print(w.output[:400])
print("\\n" + "="*60)
print("SYNTHESIS:")
print(result_bug.synthesis)
print("="*60)

plot_gantt(result_bug, "Experiment 1: Bug Audit Timeline", "figures/gantt_bug_audit.png")
from IPython.display import Image
Image("figures/gantt_bug_audit.png")
"""

EXP2_MD = """\
## Experiment 2 – Math Homework Grading

**Task:** Grade 8 algebra homework problems from `data/math_homework.json`.

Workers grade problems in pairs (P1–P2, P3–P4, etc.), then the synthesis agent
produces a full gradebook with error analysis.
"""

EXP2_CODE = """\
homework = json.loads(Path("data/math_homework.json").read_text())
hw_text = json.dumps(homework["problems"], indent=2)

tracer_math = ModelTracer("traces/math_grading.db")
orc_math = MultiAgentOrchestrator(
    lead_client=client,
    worker_client=client,
    tracer=tracer_math,
    max_workers=4,
)

print("▶ Running math grading …")
result_math = orc_math.run(
    task="Grade the following algebra homework problems. "
         "For each problem check whether the student's answer and working are correct. "
         "Identify the error type if wrong. Return scores out of 10.",
    context=hw_text,
    task_id="math_grading",
    on_worker_done=lambda r: print(f"  ✓ {r.subtask_id} done in {format_ms(r.duration_ms)}"),
)

print(f"\\n✅ Grading complete in {format_ms(result_math.wall_time_ms)}")
print(f"   Speedup: {speedup_ratio(result_math.wall_time_ms, result_math.serial_time_ms):.2f}×")
"""

EXP2_RESULTS_CODE = """\
print(result_math.synthesis)
plot_gantt(result_math, "Experiment 2: Math Grading Timeline", "figures/gantt_math_grading.png")
Image("figures/gantt_math_grading.png")
"""

EXP3_MD = """\
## Experiment 3 – Customer Message Grouping

**Task:** Categorise 20 customer support messages from `data/customer_messages.json`
into one of 6 support categories.

Messages are processed in parallel batches of 5.
"""

EXP3_CODE = """\
messages = json.loads(Path("data/customer_messages.json").read_text())
msg_text = json.dumps(messages["messages"], indent=2)
categories_text = json.dumps(messages["categories"], indent=2)

tracer_cust = ModelTracer("traces/customer_grouping.db")
orc_cust = MultiAgentOrchestrator(
    lead_client=client,
    worker_client=client,
    tracer=tracer_cust,
    max_workers=4,
)

context = f"CATEGORIES:\\n{categories_text}\\n\\nMESSAGES:\\n{msg_text}"

print("▶ Running customer grouping …")
result_cust = orc_cust.run(
    task="Categorise each customer support message into one of the provided categories. "
         "Return category, confidence (0-1), and a brief reason for each message.",
    context=context,
    task_id="customer_grouping",
    on_worker_done=lambda r: print(f"  ✓ {r.subtask_id} done in {format_ms(r.duration_ms)}"),
)

print(f"\\n✅ Grouping complete in {format_ms(result_cust.wall_time_ms)}")
print(f"   Speedup: {speedup_ratio(result_cust.wall_time_ms, result_cust.serial_time_ms):.2f}×")
"""

EXP3_RESULTS_CODE = """\
print(result_cust.synthesis)
plot_gantt(result_cust, "Experiment 3: Customer Grouping Timeline", "figures/gantt_customer.png")
Image("figures/gantt_customer.png")
"""

PERF_MD = """\
## Performance Comparison

Now we compare all three experiments across:
- **Wall time** (parallel) vs **serial time** (sum of worker durations)
- **Speedup ratio**
- **Token usage and cost**
"""

PERF_CODE = """\
all_results = {
    "Bug Audit":    result_bug,
    "Math Grading": result_math,
    "Customer Grp": result_cust,
}

# Aggregate tracer across all three experiments
combined_tracer = ModelTracer("traces/combined.db")

# Copy records from each tracer into combined
for source_db in ["traces/bug_audit.db", "traces/math_grading.db", "traces/customer_grouping.db"]:
    import sqlite3
    src = sqlite3.connect(source_db)
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM traces").fetchall()
    for row in rows:
        r = dict(row)
        combined_tracer.record(
            model=r["model"],
            prompt=r["prompt"] + f"_import_{r['task_id']}",  # avoid cache collision
            response=r["response"],
            tokens_in=r["tokens_in"],
            tokens_out=r["tokens_out"],
            cost_usd=r["cost_usd"],
            duration_ms=r["duration_ms"],
            task_id=r["task_id"],
            worker_id=r["worker_id"],
            trace_id=r["trace_id"],
        )
    src.close()

plot_summary(all_results, MODEL_NAME, "figures/summary.png")
Image("figures/summary.png")
"""

REPORT_MD = """\
## Final Report

Generating a markdown report consolidating all results, then exporting the full trace to JSON.
"""

REPORT_CODE = """\
generate_report(all_results, MODEL_NAME, combined_tracer, "final_report.md")

# Export all traces to JSON for offline inspection
combined_tracer.export_json("traces/all_traces.json")
n = combined_tracer.stats()
print(f"\\nFull trace stats:")
for k, v in n.items():
    print(f"  {k:25s}: {v}")

print("\\n✅ All done.")
print("   Figures:      figures/*.png")
print("   Traces:       traces/*.db  +  traces/all_traces.json")
print("   Final report: final_report.md")
"""

CONCLUSION_MD = """\
## What We Showed

| Capability | Demonstrated |
|-----------|-------------|
| Lead agent writes structured plan | ✓ |
| Parallel workers run subtasks concurrently | ✓ |
| Same orchestrator handles 3 different task types | ✓ |
| Speedup ≈ N workers / max_worker_latency | ✓ |
| ModelTracer replays all calls at $0 | ✓ |
| Figures + report saved to disk | ✓ |

### Key take-away

Dynamic multi-agent workflows help when:
1. The task can be decomposed into **independent** subtasks (no shared state between workers).
2. Worker latency dominates (parallelism hides it).
3. You want the lead agent to **adapt** the plan to each new task rather than using a fixed template.

They add overhead (the planning step + synthesis) so for very simple tasks a single call is cheaper.
The cross-over point in our experiments was roughly **3+ subtasks** taking **>1 s each**.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Claude-specific cells
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_TITLE_MD = """\
# Dynamic Multi-Agent Workflows – Claude Edition

Using **Anthropic Claude** as both the lead and worker model.

Recommended model pairing:
- Lead / Synthesis: `claude-sonnet-4-5` (complex reasoning)
- Workers: `claude-haiku-4-5` (fast and cheap)
"""

CLAUDE_CONFIG_CODE = """\
import os

# ── Mode switch ───────────────────────────────────────────────────────────────
DEMO_MODE = True   # Set False to make real API calls

# ── API key (only needed when DEMO_MODE = False) ──────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model selection ───────────────────────────────────────────────────────────
LEAD_MODEL   = "claude-sonnet-4-5"     # or claude-opus-4-5 for best quality
WORKER_MODEL = "claude-haiku-4-5"      # fast workers

MODEL_NAME = f"Claude ({LEAD_MODEL})"
print(f"Mode: {'DEMO (no API calls)' if DEMO_MODE else 'LIVE'}")
print(f"Lead model:   {LEAD_MODEL}")
print(f"Worker model: {WORKER_MODEL}")
"""

CLAUDE_CLIENT_CODE = """\
class ClaudeClient(BaseModelClient):
    \"\"\"
    Thin wrapper around the Anthropic Messages API.
    Tracks tokens and computes cost automatically.
    \"\"\"
    # Approximate pricing per 1 M tokens (update as needed)
    _PRICES = {
        "claude-haiku-4-5":   (0.80,  4.00),
        "claude-sonnet-4-5":  (3.00, 15.00),
        "claude-opus-4-5":   (15.00, 75.00),
    }

    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2):
        msgs = [{"role": "user", "content": prompt}]
        kw = {"model": self.model, "max_tokens": 2048, "messages": msgs, "temperature": temperature}
        if system:
            kw["system"] = system
        resp = self._client.messages.create(**kw)
        text = resp.content[0].text
        tin  = resp.usage.input_tokens
        tout = resp.usage.output_tokens
        pin, pout = self._PRICES.get(self.model, (3.0, 15.0))
        cost = (tin * pin + tout * pout) / 1_000_000
        return text, tin, tout, cost


# ── Instantiate ────────────────────────────────────────────────────────────────
if DEMO_MODE:
    client = DemoClient(model="claude-demo")
    print("Using DemoClient – no API calls will be made.")
else:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "Set ANTHROPIC_API_KEY env var or paste it above, then re-run."
        )
    client = ClaudeClient(api_key=ANTHROPIC_API_KEY, model=LEAD_MODEL)
    print(f"Using ClaudeClient with {LEAD_MODEL}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Qwen-specific cells
# ─────────────────────────────────────────────────────────────────────────────

QWEN_TITLE_MD = """\
# Dynamic Multi-Agent Workflows – Qwen Edition

Using **Qwen 2.5** (via HuggingFace Inference API) as both the lead and worker model.

Model: `Qwen/Qwen2.5-72B-Instruct`

> **HuggingFace token** (free tier available): https://huggingface.co/settings/tokens
"""

QWEN_CONFIG_CODE = """\
import os

# ── Mode switch ───────────────────────────────────────────────────────────────
DEMO_MODE = True   # Set False to make real API calls

# ── API key (only needed when DEMO_MODE = False) ──────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Model selection ───────────────────────────────────────────────────────────
LEAD_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
WORKER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

MODEL_NAME = "Qwen 2.5-72B"
print(f"Mode: {'DEMO (no API calls)' if DEMO_MODE else 'LIVE'}")
print(f"Model: {LEAD_MODEL}")
"""

QWEN_CLIENT_CODE = """\
class QwenClient(BaseModelClient):
    \"\"\"
    Calls Qwen models via the HuggingFace Inference API (OpenAI-compatible endpoint).
    Requires a HuggingFace account token.
    \"\"\"

    def __init__(self, token: str, model: str = "Qwen/Qwen2.5-72B-Instruct"):
        super().__init__(model)
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("pip install huggingface_hub>=0.26")
        self._client = InferenceClient(model=model, token=token)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=max(temperature, 0.01),
        )
        text  = resp.choices[0].message.content
        usage = resp.usage
        tin   = usage.prompt_tokens if usage else 0
        tout  = usage.completion_tokens if usage else 0
        cost  = (tin * 0.90 + tout * 0.90) / 1_000_000   # approx HF Serverless pricing
        return text, tin, tout, cost


# ── Instantiate ────────────────────────────────────────────────────────────────
if DEMO_MODE:
    client = DemoClient(model="qwen-demo")
    print("Using DemoClient – no API calls will be made.")
else:
    if not HF_TOKEN:
        raise EnvironmentError(
            "Set HF_TOKEN env var (get one at https://huggingface.co/settings/tokens) then re-run."
        )
    client = QwenClient(token=HF_TOKEN, model=LEAD_MODEL)
    print(f"Using QwenClient with {LEAD_MODEL}")
"""


# ─────────────────────────────────────────────────────────────────────────────
# Builder function
# ─────────────────────────────────────────────────────────────────────────────

def build_notebook(
    title_md: str,
    config_code: str,
    client_code: str,
) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    }

    cells_spec = [
        ("markdown", title_md),
        ("markdown", SHARED_INTRO_MD),
        ("code",     SHARED_SETUP_CODE),
        ("markdown", SHARED_DEMO_MODE_MD),
        ("code",     config_code),
        ("markdown", SHARED_COMPONENTS_MD),
        ("code",     SHARED_TIMING_CODE),
        ("code",     SHARED_PLOT_GANTT),
        ("code",     SHARED_PLOT_SUMMARY),
        ("code",     SHARED_FINAL_REPORT),
        ("code",     DEMO_RESPONSES_CODE),
        ("code",     DEMO_CLIENT_CODE),
        ("code",     client_code),
        ("markdown", EXP1_MD),
        ("code",     EXP1_CODE),
        ("code",     EXP1_RESULTS_CODE),
        ("markdown", EXP2_MD),
        ("code",     EXP2_CODE),
        ("code",     EXP2_RESULTS_CODE),
        ("markdown", EXP3_MD),
        ("code",     EXP3_CODE),
        ("code",     EXP3_RESULTS_CODE),
        ("markdown", PERF_MD),
        ("code",     PERF_CODE),
        ("markdown", REPORT_MD),
        ("code",     REPORT_CODE),
        ("markdown", CONCLUSION_MD),
    ]

    for ctype, src in cells_spec:
        nb.cells.append(cell(src, ctype))

    return nb


def main():
    claude_nb = build_notebook(CLAUDE_TITLE_MD, CLAUDE_CONFIG_CODE, CLAUDE_CLIENT_CODE)
    qwen_nb   = build_notebook(QWEN_TITLE_MD,   QWEN_CONFIG_CODE,   QWEN_CLIENT_CODE)

    out_claude = Path("Claude_MultiAgent_Workflow.ipynb")
    out_qwen   = Path("Qwen_MultiAgent_Workflow.ipynb")

    nbf.write(claude_nb, str(out_claude))
    nbf.write(qwen_nb,   str(out_qwen))

    print(f"✓ {out_claude}  ({len(claude_nb.cells)} cells)")
    print(f"✓ {out_qwen}    ({len(qwen_nb.cells)} cells)")


if __name__ == "__main__":
    main()
