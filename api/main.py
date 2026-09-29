from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.workflow import DeepShieldWorkflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepShield AI",
    version="0.2.0",
    description=(
        "Multi-agent self-healing operations API. "
        "Diagnoses, retrieves runbooks, remediates, and verifies service health."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single workflow instance shared across requests (knowledge base init is expensive).
_KNOWLEDGE_DIR = os.getenv("DEEPSHIELD_KNOWLEDGE_DIR", "knowledge")
_workflow: Optional[DeepShieldWorkflow] = None


def _get_workflow() -> DeepShieldWorkflow:
    """Lazy-initialise the workflow singleton."""
    global _workflow
    if _workflow is None:
        _workflow = DeepShieldWorkflow(knowledge_dir=_KNOWLEDGE_DIR)
    return _workflow


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class WorkflowRequest(BaseModel):
    application: str = Field(default="Payment API", description="Name of the affected application.")
    issue: str = Field(
        default="Payment API cannot connect to MySQL",
        description="Free-text description of the observed problem.",
    )
    deployment: Dict[str, Any] = Field(
        default_factory=lambda: {"status": "unhealthy"},
        description="Current deployment context (e.g. status, replicas).",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, plan the remediation but do not execute it.",
    )


class DiagnosisInfo(BaseModel):
    root_cause: str
    contributing_factors: List[str] = []
    severity: str
    confidence: float
    diagnosis_source: str


class RunbookInfo(BaseModel):
    title: str
    summary: str
    causes: List[str] = []
    recommended_actions: List[str] = []
    severity_hint: str = "unknown"
    retrieval_source: str = "none"
    score: float = 0.0


class RemediationInfo(BaseModel):
    action: str
    action_type: str
    target: str
    risk: str
    approved: bool
    safe: bool
    mutating: bool
    details: str
    action_sequence: List[str] = []


class VerificationInfo(BaseModel):
    status: str
    service_health: str
    container_statuses: Dict[str, str] = {}
    issues: List[str] = []


class WorkflowResponse(BaseModel):
    application: str
    issue: str
    execution_mode: str = "loop"
    # Diagnosis
    diagnosis: DiagnosisInfo
    # Knowledge retrieval
    runbook: RunbookInfo
    # Remediation
    remediation: RemediationInfo
    action_executed: bool
    # Verification
    verification: VerificationInfo
    resolved: bool
    # Supervisor trail
    workflow_trail: List[Dict[str, Any]] = []
    supervisor_status: str = "complete"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "DeepShield AI", "version": "0.2.0"}


@app.post("/workflow/run", response_model=WorkflowResponse, tags=["Workflow"])
def run_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """
    Run the full self-healing workflow synchronously and return the complete result.

    Steps: Diagnosis → Knowledge retrieval → Remediation → Verification.
    """
    workflow = _get_workflow()

    # Honour per-request dry_run without mutating the singleton.
    if request.dry_run and not workflow.dry_run:
        workflow = DeepShieldWorkflow(
            knowledge_dir=_KNOWLEDGE_DIR,
            dry_run=True,
        )

    try:
        state = workflow.execute(
            application=request.application,
            issue=request.issue,
            deployment=request.deployment,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc

    return _build_response(request, state)


@app.post("/workflow/stream", tags=["Workflow"])
def stream_workflow(request: WorkflowRequest) -> StreamingResponse:
    """
    Run the workflow and stream each step's result as newline-delimited JSON (NDJSON).

    Emits one JSON object per pipeline step so the caller can show progress in real time.
    """
    def _generate() -> Generator[str, None, None]:
        yield _ndjson({"event": "start", "application": request.application, "issue": request.issue})

        # Run each agent step individually and stream the partial result.
        wf = DeepShieldWorkflow(
            knowledge_dir=_KNOWLEDGE_DIR,
            dry_run=request.dry_run,
            use_graph=False,  # Always use the loop for streaming so we control the ticks.
        )

        state: Dict[str, Any] = {
            "application": request.application,
            "issue": request.issue,
            "deployment": request.deployment,
            "service_health": "unhealthy",
        }

        # Step 1 — Diagnosis
        try:
            partial = wf.diagnosis.analyze(state)
            state.update(partial)
            yield _ndjson({
                "event": "diagnosis",
                "root_cause": state.get("root_cause"),
                "severity": state.get("severity"),
                "diagnosis_source": state.get("diagnosis_source"),
            })
        except Exception as exc:  # noqa: BLE001
            yield _ndjson({"event": "error", "step": "diagnosis", "detail": str(exc)})
            return

        # Step 2 — Knowledge retrieval
        try:
            runbook = wf.knowledge_base.get_runbook(state.get("issue", ""))
            state["runbook"] = runbook
            yield _ndjson({
                "event": "knowledge",
                "runbook_title": runbook.get("title"),
                "retrieval_source": runbook.get("retrieval_source"),
                "score": runbook.get("score"),
            })
        except Exception as exc:  # noqa: BLE001
            yield _ndjson({"event": "error", "step": "knowledge", "detail": str(exc)})
            return

        # Step 3 — Remediation
        try:
            partial = wf.remediation.plan(state)
            state.update(partial)
            plan = state.get("remediation_plan", {})
            yield _ndjson({
                "event": "remediation",
                "action": plan.get("action"),
                "action_type": plan.get("action_type"),
                "target": plan.get("target"),
                "risk": plan.get("risk"),
                "action_executed": state.get("action_executed", False),
                "action_sequence": plan.get("action_sequence", []),
            })
        except Exception as exc:  # noqa: BLE001
            yield _ndjson({"event": "error", "step": "remediation", "detail": str(exc)})
            return

        # Step 4 — Verification
        try:
            partial = wf.verification.verify(state)
            state.update(partial)
            vr = state.get("verification_result", {})
            yield _ndjson({
                "event": "verification",
                "status": vr.get("status"),
                "service_health": vr.get("service_health"),
                "issues": vr.get("issues", []),
                "resolved": state.get("resolved", False),
            })
        except Exception as exc:  # noqa: BLE001
            yield _ndjson({"event": "error", "step": "verification", "detail": str(exc)})
            return

        # Final summary
        yield _ndjson({"event": "complete", **_build_response(request, state)})

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/demo", tags=["Workflow"])
def demo() -> Dict[str, Any]:
    """
    Run the canonical MySQL-failure demo scenario and return the full result.
    Equivalent to POST /workflow/run with default parameters.
    """
    return run_workflow(WorkflowRequest())


@app.get("/workflow/status", tags=["Workflow"])
def workflow_info() -> Dict[str, Any]:
    """Return metadata about the loaded workflow (knowledge base, execution mode)."""
    wf = _get_workflow()
    return {
        "version": "0.2.0",
        "execution_mode": "graph" if wf._graph is not None else "loop",
        "knowledge_dir": _KNOWLEDGE_DIR,
        "runbooks": wf.knowledge_base.list_runbooks(),
        "dry_run": wf.dry_run,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_response(request: WorkflowRequest, state: Dict[str, Any]) -> Dict[str, Any]:
    """Map the raw workflow state dict to the WorkflowResponse schema."""
    runbook = state.get("runbook") or {}
    plan = state.get("remediation_plan") or {}
    vr = state.get("verification_result") or {}

    return {
        "application": request.application,
        "issue": request.issue,
        "execution_mode": state.get("execution_mode", "loop"),
        "diagnosis": {
            "root_cause": state.get("root_cause", "Unknown"),
            "contributing_factors": state.get("contributing_factors", []),
            "severity": state.get("severity", "medium"),
            "confidence": state.get("confidence", 0.0),
            "diagnosis_source": state.get("diagnosis_source", "rules"),
        },
        "runbook": {
            "title": runbook.get("title", "No matching runbook"),
            "summary": runbook.get("summary", ""),
            "causes": runbook.get("causes", []),
            "recommended_actions": runbook.get("recommended_actions", []),
            "severity_hint": runbook.get("severity_hint", "unknown"),
            "retrieval_source": runbook.get("retrieval_source", "none"),
            "score": runbook.get("score", 0.0),
        },
        "remediation": {
            "action": plan.get("action", "none"),
            "action_type": plan.get("action_type", "unknown"),
            "target": plan.get("target", "unknown"),
            "risk": plan.get("risk", "unknown"),
            "approved": plan.get("approved", False),
            "safe": plan.get("safe", False),
            "mutating": plan.get("mutating", False),
            "details": plan.get("details", ""),
            "action_sequence": plan.get("action_sequence", []),
        },
        "action_executed": state.get("action_executed", False),
        "verification": {
            "status": vr.get("status", "unknown"),
            "service_health": vr.get("service_health", "unknown"),
            "container_statuses": vr.get("container_statuses", {}),
            "issues": vr.get("issues", []),
        },
        "resolved": bool(state.get("resolved", False)),
        "workflow_trail": state.get("workflow_trail", []),
        "supervisor_status": state.get("supervisor_status", "complete"),
    }


def _ndjson(obj: Dict[str, Any]) -> str:
    """Serialise a dict to a newline-terminated JSON string."""
    return json.dumps(obj, default=str) + "\n"
