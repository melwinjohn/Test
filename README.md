# Dynamic Multi-Agent Workflows – Research Demo

A self-contained research project that proves whether decomposing a task into
parallel subtasks actually helps — and by how much.

## What's here

```
.
├── shared/
│   ├── tracer.py        SQLite-backed model call cache (replay at $0)
│   └── runtime.py       LeadAgent + Worker + MultiAgentOrchestrator
├── data/
│   ├── buggy_library.py Python stats library with 5 planted bugs
│   ├── math_homework.json  8 algebra problems with student answers
│   └── customer_messages.json  20 support messages to categorise
├── traces/              Created at runtime – per-experiment SQLite dbs
├── figures/             Created at runtime – PNG charts
├── Claude_MultiAgent_Workflow.ipynb   Full experiment notebook (Anthropic)
├── Qwen_MultiAgent_Workflow.ipynb     Full experiment notebook (Qwen / HF)
├── run_demo.py          Standalone runner (no Jupyter needed)
├── build_notebooks.py   Regenerates both .ipynb files
├── final_report.md      Generated after run_demo.py
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Run the full demo immediately (no API key required)
python run_demo.py
```

Outputs: `figures/*.png`, `traces/*.db`, `traces/all_traces.json`, `final_report.md`

## Notebooks

Open either notebook in Jupyter and run all cells top-to-bottom.

```bash
jupyter notebook Claude_MultiAgent_Workflow.ipynb
# or
jupyter notebook Qwen_MultiAgent_Workflow.ipynb
```

Both notebooks default to `DEMO_MODE = True` — no API key needed, instant execution,
realistic simulated latency, and pre-written responses that mirror real model outputs.

### Live mode

| Notebook | Change | Key source |
|----------|--------|-----------|
| Claude   | `DEMO_MODE = False` | `ANTHROPIC_API_KEY` env var |
| Qwen     | `DEMO_MODE = False` | `HF_TOKEN` env var (free tier at huggingface.co/settings/tokens) |

## Architecture

```
Task description
      │
      ▼
 LeadAgent  ──► JSON Plan: [SubTask1, SubTask2, ..., SubTaskN]
      │                     (each self-contained, no shared state)
      ▼
 WorkerPool  ──► N threads, each calling the model independently
      │
      ▼
 SynthesisAgent  ──► merges all worker outputs → final answer
      │
      ▼
 ModelTracer  ──► SQLite record of every call (prompt, response, tokens, cost, latency)
```

### Key design decisions

**Why SQLite for tracing?**  Zero setup, single file, thread-safe via per-thread
connections.  The hash of `(model, prompt)` is the cache key — same experiment
re-run hits the cache and costs nothing.

**Why ThreadPoolExecutor over asyncio?**  Most LLM SDKs are sync; threads let us
parallelize without rewriting clients.  For high worker counts (>20) asyncio would
be better.

**Why separate lead and worker models?**  The lead agent needs complex reasoning
(planning, synthesis); workers do narrow, well-defined subtasks.  Pairing a
large planner with smaller workers halves cost without hurting quality.

## Experiments

| # | Task | Workers | Parallel speedup |
|---|------|---------|-----------------|
| 1 | Planted bug audit (`buggy_library.py`) | 4 | ~N × worker latency |
| 2 | Algebra homework grading (8 problems) | 4 | ~N × worker latency |
| 3 | Customer message grouping (20 msgs) | 4 | ~N × worker latency |

The "planted bugs" benchmark has a ground truth: all 5 bugs are known.
It lets you directly measure whether the multi-agent approach finds more bugs
than a single-call baseline would.

## Regenerating notebooks

```bash
python build_notebooks.py
```

Both notebooks are generated from `build_notebooks.py`.  Editing that file
and re-running it is the canonical way to update notebook content.

## Trace replay

```python
from shared.tracer import ModelTracer

tracer = ModelTracer("traces/bug_audit.db")
print(tracer.stats())
print(tracer.task_summary())

# Re-run a task — all calls hit the cache
tracer.export_json("traces/bug_audit_export.json")
```
