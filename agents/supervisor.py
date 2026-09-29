from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowStep(str, Enum):
    """Ordered steps in the self-healing workflow."""
    DIAGNOSIS = "diagnosis"
    KNOWLEDGE = "knowledge"
    REMEDIATION = "remediation"
    VERIFICATION = "verification"
    DONE = "done"


# The canonical step order drives the loop.
STEP_ORDER: List[WorkflowStep] = [
    WorkflowStep.DIAGNOSIS,
    WorkflowStep.KNOWLEDGE,
    WorkflowStep.REMEDIATION,
    WorkflowStep.VERIFICATION,
    WorkflowStep.DONE,
]

# Conditions that must be satisfied before a step is considered complete.
STEP_COMPLETION_KEYS: Dict[WorkflowStep, str] = {
    WorkflowStep.DIAGNOSIS: "root_cause",
    WorkflowStep.KNOWLEDGE: "runbook",
    WorkflowStep.REMEDIATION: "remediation_plan",
    WorkflowStep.VERIFICATION: "verification_result",
}

MAX_RETRIES = 2  # Maximum retries per step before escalating.


class SupervisorAgent:
    """
    Coordinates the self-healing workflow via a deterministic state-machine loop.

    Each call to ``decide_next_step`` examines the current shared state and
    returns the next agent to run together with a human-readable reason.
    ``run_loop`` drives the full pipeline, collecting a decision trail so the
    caller can reconstruct exactly what the supervisor decided at each tick.
    """

    def __init__(self) -> None:
        self._step_attempts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def decide_next_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the next agent to invoke based on what is present in *state*.

        Decision priority
        -----------------
        1. Walk ``STEP_ORDER`` in sequence.
        2. For each step, check whether the required output key exists and is
           non-empty in state.
        3. Return the first incomplete step.
        4. If all steps are complete, return ``"done"``.
        """
        current_step = self._identify_current_step(state)

        # Check retry budget.
        attempt_key = current_step.value
        attempts = self._step_attempts.get(attempt_key, 0)

        if attempts >= MAX_RETRIES and current_step != WorkflowStep.DONE:
            logger.warning(
                "Step '%s' has been attempted %d time(s) without producing output — escalating.",
                current_step.value,
                attempts,
            )
            return {
                "next_agent": "escalate",
                "step": current_step.value,
                "reason": (
                    f"Step '{current_step.value}' failed to produce output after "
                    f"{attempts} attempt(s). Manual intervention may be required."
                ),
                "escalated": True,
            }

        self._step_attempts[attempt_key] = attempts + 1

        reason = self._build_reason(current_step, state)
        logger.debug("Supervisor → next_agent=%s  reason=%s", current_step.value, reason)

        return {
            "next_agent": current_step.value,
            "step": current_step.value,
            "reason": reason,
            "escalated": False,
        }

    # ------------------------------------------------------------------
    # Full-loop driver (used by DeepShieldWorkflow)
    # ------------------------------------------------------------------

    def run_loop(
        self,
        state: Dict[str, Any],
        agent_map: Dict[str, Any],
        *,
        max_iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Drive the workflow loop until completion or escalation.

        Parameters
        ----------
        state:
            The shared mutable state dict.
        agent_map:
            Mapping of step name → callable(state) → Dict.  Every callable
            must accept the current state and return a partial state dict that
            will be merged back.
        max_iterations:
            Hard cap to prevent infinite loops.

        Returns
        -------
        The final state dict, augmented with ``workflow_trail`` — a list of
        each decision made by the supervisor.
        """
        trail: List[Dict[str, Any]] = []
        state.setdefault("workflow_trail", trail)

        for iteration in range(max_iterations):
            decision = self.decide_next_step(state)
            trail.append({"iteration": iteration, **decision})

            next_agent = decision["next_agent"]

            if next_agent == "done":
                logger.info("Supervisor: workflow complete after %d iteration(s).", iteration + 1)
                state["supervisor_status"] = "complete"
                break

            if decision.get("escalated"):
                logger.error("Supervisor escalated at step '%s'.", decision.get("step"))
                state["supervisor_status"] = "escalated"
                state["supervisor_escalation_reason"] = decision["reason"]
                break

            handler = agent_map.get(next_agent)
            if handler is None:
                logger.error("No handler registered for agent '%s'.", next_agent)
                state["supervisor_status"] = "error"
                state["supervisor_error"] = f"No handler for agent '{next_agent}'."
                break

            try:
                partial = handler(state)
                if isinstance(partial, dict):
                    state.update(partial)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent '%s' raised an exception: %s", next_agent, exc)
                state["supervisor_status"] = "error"
                state["supervisor_error"] = str(exc)
                break
        else:
            logger.warning("Supervisor hit max_iterations=%d without completing.", max_iterations)
            state["supervisor_status"] = "timeout"

        return state

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_workflow_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a snapshot of the current workflow progress."""
        completed = []
        pending = []
        for step, key in STEP_COMPLETION_KEYS.items():
            if state.get(key):
                completed.append(step.value)
            else:
                pending.append(step.value)

        current = self._identify_current_step(state)

        return {
            "current_step": current.value,
            "completed_steps": completed,
            "pending_steps": pending,
            "resolved": bool(state.get("resolved")),
            "supervisor_status": state.get("supervisor_status", "running"),
        }

    def reset(self) -> None:
        """Reset internal retry counters for a new workflow run."""
        self._step_attempts.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _identify_current_step(self, state: Dict[str, Any]) -> WorkflowStep:
        """Walk the step order and return the first step whose output is absent."""
        for step in STEP_ORDER:
            if step == WorkflowStep.DONE:
                return WorkflowStep.DONE
            required_key = STEP_COMPLETION_KEYS[step]
            if not state.get(required_key):
                return step
        return WorkflowStep.DONE

    def _build_reason(self, step: WorkflowStep, state: Dict[str, Any]) -> str:
        reasons: Dict[WorkflowStep, str] = {
            WorkflowStep.DIAGNOSIS: (
                "Deployment status and dependencies must be inspected to determine the root cause."
            ),
            WorkflowStep.KNOWLEDGE: (
                f"Root cause identified as: '{state.get('root_cause', 'unknown')}'. "
                "Searching the knowledge base for a matching runbook."
            ),
            WorkflowStep.REMEDIATION: (
                "Runbook retrieved. Selecting an approved, safe remediation action."
            ),
            WorkflowStep.VERIFICATION: (
                "Remediation action executed. Verifying that the service has recovered."
            ),
            WorkflowStep.DONE: "All workflow steps completed successfully.",
        }
        return reasons.get(step, "Proceeding to next step.")
