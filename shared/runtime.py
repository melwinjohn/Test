"""
Dynamic multi-agent runtime.

Architecture
------------
  Task description
       │
       ▼
  LeadAgent  ──► structured Plan  (JSON with N SubTasks)
       │
       ▼
  WorkerPool  ──► N threads run SubTasks in parallel
       │
       ▼
  LeadAgent  ──► SynthesisAgent merges worker outputs → final answer

Any model backend that implements BaseModelClient can be swapped in.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tracer import ModelTracer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    id: str
    title: str
    description: str
    context: str
    expected_output: str


@dataclass
class Plan:
    task_id: str
    original_task: str
    rationale: str
    subtasks: List[SubTask]
    raw_plan: str = ""


@dataclass
class WorkerResult:
    subtask_id: str
    subtask_title: str
    output: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    from_cache: bool = False
    error: Optional[str] = None


@dataclass
class OrchestratorResult:
    task_id: str
    task: str
    plan: Plan
    worker_results: List[WorkerResult]
    synthesis: str
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    wall_time_ms: int = 0
    serial_time_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Model client interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseModelClient:
    """
    Implement this to plug in any LLM.
    complete() must return (text, tokens_in, tokens_out, cost_usd).
    """

    def __init__(self, model: str):
        self.model = model

    def complete(
        self, prompt: str, system: str = "", temperature: float = 0.2
    ) -> tuple[str, int, int, float]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Lead agent
# ─────────────────────────────────────────────────────────────────────────────

PLAN_SYSTEM = """You are a lead agent that decomposes complex tasks into parallel subtasks.
Output ONLY valid JSON matching this schema (no markdown fences):
{
  "rationale": "<why this decomposition>",
  "subtasks": [
    {
      "id": "st_1",
      "title": "<short title>",
      "description": "<what the worker must do>",
      "context": "<relevant snippets or data>",
      "expected_output": "<format the worker should return>"
    }
  ]
}
Rules:
- Create 2-6 subtasks that can be executed independently and in parallel.
- Each subtask must be self-contained (include all context it needs).
- expected_output must be a concrete format (e.g. JSON object, bullet list).
"""

SYNTHESIS_SYSTEM = """You are a synthesis agent.  You receive the outputs of several parallel workers
and produce a single, structured final report.  Be concise but complete.
"""


def _extract_json(text: str) -> str:
    """Strip markdown fences if model added them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class LeadAgent:
    def __init__(
        self,
        client: BaseModelClient,
        tracer: ModelTracer,
        task_id: str,
    ):
        self.client = client
        self.tracer = tracer
        self.task_id = task_id

    def plan(self, task: str, context: str = "") -> Plan:
        _base = f"TASK:\n{task}\n\nCONTEXT:\n{context}" if context else f"TASK:\n{task}"
        prompt = f"[PLAN]{_base}"

        cached = self.tracer.get_cached(self.client.model, prompt)
        if cached:
            raw = cached["response"]
        else:
            t0 = time.perf_counter()
            raw, tin, tout, cost = self.client.complete(prompt, system=PLAN_SYSTEM)
            dur = int((time.perf_counter() - t0) * 1000)
            self.tracer.record(
                model=self.client.model,
                prompt=prompt,
                response=raw,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                duration_ms=dur,
                task_id=self.task_id,
                worker_id="lead:plan",
            )

        parsed = json.loads(_extract_json(raw))
        subtasks = [
            SubTask(
                id=st["id"],
                title=st["title"],
                description=st["description"],
                context=st.get("context", ""),
                expected_output=st.get("expected_output", ""),
            )
            for st in parsed["subtasks"]
        ]
        return Plan(
            task_id=self.task_id,
            original_task=task,
            rationale=parsed.get("rationale", ""),
            subtasks=subtasks,
            raw_plan=raw,
        )

    def synthesize(self, task: str, worker_results: List[WorkerResult]) -> tuple[str, int, int, float]:
        worker_block = "\n\n".join(
            f"=== Worker {r.subtask_id}: {r.subtask_title} ===\n{r.output}"
            for r in worker_results
            if not r.error
        )
        prompt = f"[SYNTH]ORIGINAL TASK:\n{task}\n\nWORKER OUTPUTS:\n{worker_block}"

        cached = self.tracer.get_cached(self.client.model, prompt)
        if cached:
            return cached["response"], 0, 0, 0.0

        t0 = time.perf_counter()
        raw, tin, tout, cost = self.client.complete(prompt, system=SYNTHESIS_SYSTEM, temperature=0.1)
        dur = int((time.perf_counter() - t0) * 1000)
        self.tracer.record(
            model=self.client.model,
            prompt=prompt,
            response=raw,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            duration_ms=dur,
            task_id=self.task_id,
            worker_id="lead:synthesis",
        )
        return raw, tin, tout, cost


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

WORKER_SYSTEM = """You are a specialist worker in a multi-agent pipeline.
Complete exactly the subtask you are given.
Return your output in the format specified by expected_output.
Be thorough and precise.
"""


class Worker:
    def __init__(
        self,
        client: BaseModelClient,
        tracer: ModelTracer,
        task_id: str,
    ):
        self.client = client
        self.tracer = tracer
        self.task_id = task_id

    def execute(self, subtask: SubTask) -> WorkerResult:
        prompt = (
            f"SUBTASK: {subtask.title}\n\n"
            f"DESCRIPTION:\n{subtask.description}\n\n"
            f"CONTEXT:\n{subtask.context}\n\n"
            f"EXPECTED OUTPUT FORMAT:\n{subtask.expected_output}"
        )
        cached = self.tracer.get_cached(self.client.model, prompt)
        if cached:
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=cached["response"],
                tokens_in=cached["tokens_in"],
                tokens_out=cached["tokens_out"],
                cost_usd=cached["cost_usd"],
                duration_ms=cached["duration_ms"],
                from_cache=True,
            )

        try:
            t0 = time.perf_counter()
            raw, tin, tout, cost = self.client.complete(prompt, system=WORKER_SYSTEM)
            dur = int((time.perf_counter() - t0) * 1000)
            self.tracer.record(
                model=self.client.model,
                prompt=prompt,
                response=raw,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                duration_ms=dur,
                task_id=self.task_id,
                worker_id=subtask.id,
            )
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output=raw,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                duration_ms=dur,
            )
        except Exception as exc:
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                output="",
                error=str(exc),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class MultiAgentOrchestrator:
    """
    Runs the full lead → parallel workers → synthesis pipeline.

    Parameters
    ----------
    lead_client  : model used for planning + synthesis (larger/smarter)
    worker_client: model used for individual subtasks (can be same or smaller)
    tracer       : shared trace store
    max_workers  : thread-pool size for parallel execution
    """

    def __init__(
        self,
        lead_client: BaseModelClient,
        worker_client: BaseModelClient,
        tracer: ModelTracer,
        max_workers: int = 4,
    ):
        self.lead_client = lead_client
        self.worker_client = worker_client
        self.tracer = tracer
        self.max_workers = max_workers

    def run(
        self,
        task: str,
        context: str = "",
        task_id: Optional[str] = None,
        on_worker_done: Optional[Callable[[WorkerResult], None]] = None,
    ) -> OrchestratorResult:
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"

        lead = LeadAgent(self.lead_client, self.tracer, task_id)
        worker = Worker(self.worker_client, self.tracer, task_id)

        # ── 1. Plan ──────────────────────────────────────────────────────────
        t_wall_start = time.perf_counter()
        plan = lead.plan(task, context)

        # ── 2. Execute in parallel ───────────────────────────────────────────
        worker_results: List[WorkerResult] = []
        serial_ms = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_st = {
                pool.submit(worker.execute, st): st for st in plan.subtasks
            }
            for future in as_completed(future_to_st):
                result = future.result()
                serial_ms += result.duration_ms
                worker_results.append(result)
                if on_worker_done:
                    on_worker_done(result)

        # Sort by subtask id for consistent ordering
        worker_results.sort(key=lambda r: r.subtask_id)

        # ── 3. Synthesise ────────────────────────────────────────────────────
        synthesis, s_tin, s_tout, s_cost = lead.synthesize(task, worker_results)
        wall_ms = int((time.perf_counter() - t_wall_start) * 1000)

        # ── 4. Aggregate stats ───────────────────────────────────────────────
        total_tin = sum(r.tokens_in for r in worker_results) + s_tin
        total_tout = sum(r.tokens_out for r in worker_results) + s_tout
        total_cost = sum(r.cost_usd for r in worker_results) + s_cost

        return OrchestratorResult(
            task_id=task_id,
            task=task,
            plan=plan,
            worker_results=worker_results,
            synthesis=synthesis,
            total_tokens_in=total_tin,
            total_tokens_out=total_tout,
            total_cost_usd=total_cost,
            wall_time_ms=wall_ms,
            serial_time_ms=serial_ms,
        )
