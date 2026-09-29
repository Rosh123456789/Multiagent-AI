from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tools.docker_tools import (
    get_container_logs,
    get_container_status,
    health_check,
    restart_container,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action catalogue — the ONLY actions this agent is allowed to execute.
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"        # Read-only or entirely non-destructive.
    MEDIUM = "medium"  # Causes a brief service interruption.
    HIGH = "high"      # May cause data loss or prolonged downtime.


class ActionType(str, Enum):
    INSPECT_LOGS = "inspect_logs"
    CHECK_STATUS = "check_status"
    HEALTH_CHECK = "health_check"
    RESTART_CONTAINER = "restart_container"
    ESCALATE = "escalate"


@dataclass
class CatalogueEntry:
    """A single approved, bounded remediation action."""
    action_type: ActionType
    label: str                        # Human-readable name shown in the plan.
    description: str                  # What the action does.
    target: str                       # Docker container or service name.
    risk: RiskLevel
    # Keywords from runbook recommended-action text that map to this entry.
    trigger_keywords: List[str] = field(default_factory=list)
    # Minimum severity required before this action is considered.
    min_severity: str = "low"
    # Whether this action changes system state (False = read-only).
    mutating: bool = False


# The catalogue is ordered by preference: less destructive first.
ACTION_CATALOGUE: List[CatalogueEntry] = [
    CatalogueEntry(
        action_type=ActionType.INSPECT_LOGS,
        label="Inspect payment-api logs",
        description="Retrieve the last 50 log lines from the payment-api container.",
        target="payment-api",
        risk=RiskLevel.LOW,
        trigger_keywords=["inspect logs", "check logs", "inspect container logs", "logs"],
        mutating=False,
    ),
    CatalogueEntry(
        action_type=ActionType.INSPECT_LOGS,
        label="Inspect mysql logs",
        description="Retrieve the last 50 log lines from the mysql container.",
        target="mysql",
        risk=RiskLevel.LOW,
        trigger_keywords=["mysql log", "database log", "check database"],
        mutating=False,
    ),
    CatalogueEntry(
        action_type=ActionType.CHECK_STATUS,
        label="Check payment-api container status",
        description="Check the current runtime status of the payment-api container.",
        target="payment-api",
        risk=RiskLevel.LOW,
        trigger_keywords=["check status", "container status", "check container"],
        mutating=False,
    ),
    CatalogueEntry(
        action_type=ActionType.CHECK_STATUS,
        label="Check mysql container status",
        description="Check the current runtime status of the mysql container.",
        target="mysql",
        risk=RiskLevel.LOW,
        trigger_keywords=[
            "check database container", "database container status",
            "mysql status", "database status",
        ],
        mutating=False,
    ),
    CatalogueEntry(
        action_type=ActionType.HEALTH_CHECK,
        label="Health check payment-api",
        description="Query the health endpoint of the payment-api service.",
        target="payment-api",
        risk=RiskLevel.LOW,
        trigger_keywords=[
            "health endpoint", "verify health", "health check",
            "health endpoint responds", "check health",
        ],
        mutating=False,
    ),
    CatalogueEntry(
        action_type=ActionType.RESTART_CONTAINER,
        label="Restart mysql container",
        description="Restart the mysql container to recover from an exited or crashed state.",
        target="mysql",
        risk=RiskLevel.MEDIUM,
        trigger_keywords=[
            "restart mysql", "restart database", "restart db",
            "restart mysql if it exited",
        ],
        min_severity="high",
        mutating=True,
    ),
    CatalogueEntry(
        action_type=ActionType.RESTART_CONTAINER,
        label="Restart payment-api container",
        description="Restart the payment-api container after dependencies are confirmed healthy.",
        target="payment-api",
        risk=RiskLevel.MEDIUM,
        trigger_keywords=[
            "restart payment-api", "restart payment api", "restart the affected service",
            "restart the app", "restart service", "restart only after",
            "restart the service",
        ],
        min_severity="high",
        mutating=True,
    ),
    CatalogueEntry(
        action_type=ActionType.ESCALATE,
        label="Escalate to on-call engineer",
        description=(
            "No safe automated action could be selected. "
            "An on-call engineer must investigate manually."
        ),
        target="ops-team",
        risk=RiskLevel.LOW,
        trigger_keywords=[],
        mutating=False,
    ),
]

# Fast lookup: action_type → entry list.
_CATALOGUE_BY_TYPE: Dict[ActionType, List[CatalogueEntry]] = {}
for _entry in ACTION_CATALOGUE:
    _CATALOGUE_BY_TYPE.setdefault(_entry.action_type, []).append(_entry)

# Severity ordering for comparisons.
_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ---------------------------------------------------------------------------
# RemediationAgent
# ---------------------------------------------------------------------------

class RemediationAgent:
    """
    Selects and executes a safe, bounded remediation action.

    Selection flow
    --------------
    1. Collect all ``recommended_actions`` from the runbook.
    2. Score each catalogue entry against those action strings using keyword
       matching.
    3. Filter out entries whose ``min_severity`` is not met.
    4. Pick the highest-scoring, lowest-risk entry.
    5. If nothing matches, fall back to the ESCALATE sentinel.
    6. Execute the selected action via the docker_tools layer (unless
       ``dry_run=True`` or the risk gate blocks it).

    Output keys added to state
    --------------------------
    - ``remediation_plan``   (dict)  Full plan record.
    - ``planned_action``     (str)   Short label for downstream use.
    - ``action_executed``    (bool)  Whether the action was actually run.
    - ``execution_result``   (dict)  Raw result from the tool call.
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        """
        Parameters
        ----------
        dry_run:
            When True the agent selects and plans the action but does NOT
            execute it.  Useful for testing and supervised-mode deployments.
        """
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runbook: Dict[str, Any] = state.get("runbook", {})
        severity: str = state.get("severity", "medium")
        root_cause: str = state.get("root_cause", "")

        recommended_actions: List[str] = runbook.get("recommended_actions", [])
        if not recommended_actions:
            recommended_actions = ["inspect logs", "check status"]

        logger.info(
            "RemediationAgent: selecting actions for severity=%s  runbook_actions=%s",
            severity,
            recommended_actions,
        )

        # Build the full ordered action sequence from the runbook.
        sequence = self._build_action_sequence(recommended_actions, severity, root_cause)

        # The primary action is the highest-priority executable entry.
        # For mutating actions (restart) we first run any read-only pre-checks
        # in the sequence, but expose the first mutating action as the "main" action
        # so downstream (verification) knows a real remediation was applied.
        primary = self._pick_primary(sequence)

        logger.info(
            "RemediationAgent: primary action_type=%s  label=%s  risk=%s  sequence_len=%d",
            primary.action_type.value,
            primary.label,
            primary.risk.value,
            len(sequence),
        )

        execution_results: List[Dict[str, Any]] = []
        action_executed = False
        primary_result: Dict[str, Any] = {}

        if not self.dry_run:
            for entry in sequence:
                result, executed = self._execute(entry, severity)
                execution_results.append({"action": entry.label, **result})
                if entry is primary:
                    primary_result = result
                    action_executed = executed
                # Stop the sequence on any blocked or failed execution.
                if result.get("status") in {"blocked", "rejected", "error"}:
                    logger.warning(
                        "RemediationAgent: halting sequence at '%s' (status=%s).",
                        entry.label,
                        result.get("status"),
                    )
                    break
        else:
            logger.info("RemediationAgent: dry_run=True — skipping execution.")
            primary_result = {
                "status": "dry_run",
                "message": "Action planned but not executed (dry_run mode).",
            }

        plan = {
            "action": primary.label,
            "action_type": primary.action_type.value,
            "target": primary.target,
            "risk": primary.risk.value,
            "approved": True,
            "safe": primary.risk != RiskLevel.HIGH,
            "mutating": primary.mutating,
            "details": primary.description,
            "action_sequence": [e.label for e in sequence],
            "execution_results": execution_results,
        }

        return {
            "remediation_plan": plan,
            "planned_action": primary.label,
            "action_executed": action_executed,
            "execution_result": primary_result,
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _build_action_sequence(
        self,
        recommended_actions: List[str],
        severity: str,
        root_cause: str,
    ) -> List[CatalogueEntry]:
        """
        Build an ordered list of catalogue entries that match the runbook actions.

        Each recommended-action string is matched independently against the
        catalogue, preserving the runbook's intended order (read-only checks
        before mutating restarts).  Duplicate action types are de-duplicated.
        Escalate is only returned if nothing else matched.
        """
        combined_context = root_cause.lower()
        severity_rank = _SEVERITY_ORDER.get(severity, 1)
        seen: set[str] = set()   # dedupe by (action_type, target)
        sequence: List[CatalogueEntry] = []

        for action_text in recommended_actions:
            action_lower = action_text.lower()
            best_score = 0
            best_entry: Optional[CatalogueEntry] = None

            for entry in ACTION_CATALOGUE:
                if entry.action_type == ActionType.ESCALATE:
                    continue

                dedup_key = f"{entry.action_type.value}:{entry.target}"
                if dedup_key in seen:
                    continue

                # Minimum severity gate.
                if severity_rank < _SEVERITY_ORDER.get(entry.min_severity, 0):
                    continue

                # Score against the specific action text (weight 2) and the
                # broader root-cause context (weight 1) for target disambiguation.
                score = (
                    sum(2 for kw in entry.trigger_keywords if kw in action_lower)
                    + sum(1 for kw in entry.trigger_keywords if kw in combined_context)
                )
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score > 0:
                seen.add(f"{best_entry.action_type.value}:{best_entry.target}")
                sequence.append(best_entry)

        if not sequence:
            logger.warning("RemediationAgent: no catalogue match — escalating.")
            return [self._escalate_entry()]

        return sequence

    @staticmethod
    def _pick_primary(sequence: List[CatalogueEntry]) -> CatalogueEntry:
        """
        Return the primary (most significant) entry from a sequence.

        Prefers the first mutating action (restart) as the headline so the
        verification agent can confirm recovery.  Falls back to the last entry
        when the sequence contains only read-only checks.
        """
        for entry in sequence:
            if entry.mutating:
                return entry
        return sequence[-1]

    @staticmethod
    def _escalate_entry() -> CatalogueEntry:
        return next(e for e in ACTION_CATALOGUE if e.action_type == ActionType.ESCALATE)

    # ------------------------------------------------------------------
    # Action execution — fully bounded, no arbitrary shell commands.
    # ------------------------------------------------------------------

    def _execute(
        self, entry: CatalogueEntry, severity: str
    ) -> tuple[Dict[str, Any], bool]:
        """
        Dispatch to the correct docker_tools function.

        Returns (result_dict, was_executed).
        """
        try:
            if entry.action_type == ActionType.INSPECT_LOGS:
                logs = get_container_logs(entry.target)
                return {"status": "completed", "logs": logs}, True

            if entry.action_type == ActionType.CHECK_STATUS:
                status = get_container_status(entry.target)
                return {"status": "completed", "container_status": status}, True

            if entry.action_type == ActionType.HEALTH_CHECK:
                result = health_check(entry.target)
                return {"status": "completed", "health": result}, True

            if entry.action_type == ActionType.RESTART_CONTAINER:
                # Extra safety gate: never restart on LOW severity.
                if _SEVERITY_ORDER.get(severity, 0) < _SEVERITY_ORDER["medium"]:
                    logger.warning(
                        "RemediationAgent: restart blocked — severity '%s' is below threshold.",
                        severity,
                    )
                    return {
                        "status": "blocked",
                        "message": (
                            f"Restart of '{entry.target}' blocked: "
                            f"severity '{severity}' is below the required minimum."
                        ),
                    }, False

                result = restart_container(entry.target)
                return {"status": "completed", "restart": result}, True

            if entry.action_type == ActionType.ESCALATE:
                return {
                    "status": "escalated",
                    "message": "No safe automated action available. Escalated to on-call.",
                }, False

        except ValueError as exc:
            # docker_tools raises ValueError for non-allowlisted containers.
            logger.error("RemediationAgent: action rejected by tools layer: %s", exc)
            return {"status": "rejected", "message": str(exc)}, False
        except Exception as exc:  # noqa: BLE001
            logger.exception("RemediationAgent: unexpected error during execution: %s", exc)
            return {"status": "error", "message": str(exc)}, False

        return {"status": "unknown_action"}, False
