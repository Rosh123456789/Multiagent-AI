from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from tools.docker_tools import get_container_status, health_check

logger = logging.getLogger(__name__)

# How many times to re-probe before declaring failure.
_DEFAULT_MAX_RETRIES = 3
# Seconds to wait between probes.
_DEFAULT_RETRY_DELAY = 2.0

# Containers that must be running for the service to be considered healthy.
_REQUIRED_CONTAINERS = ["payment-api", "mysql"]


class VerificationAgent:
    """
    Confirms that the self-healing action actually restored service health.

    Verification strategy (in order)
    ---------------------------------
    1. Re-check the container status for every required container.
    2. Query the payment-api health endpoint.
    3. Cross-check the action that was executed against the observed state.
    4. Retry up to ``max_retries`` times with a short delay when the service
       is still coming up (graceful startup window).

    Output keys added to state
    --------------------------
    - ``verification_result``  (dict)  Full structured result.
    - ``resolved``             (bool)  True only when all checks passed.
    - ``service_health``       (str)   Final health string observed.
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def verify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        remediation_plan: Dict[str, Any] = state.get("remediation_plan", {})
        action_executed: bool = state.get("action_executed", False)
        severity: str = state.get("severity", "medium")

        logger.info(
            "VerificationAgent: starting verification  action_executed=%s  severity=%s",
            action_executed,
            severity,
        )

        # Run probes with retry backoff.
        container_checks: Dict[str, str] = {}
        health_result: Dict[str, str] = {}
        issues: List[str] = []
        passed = False

        for attempt in range(1, self.max_retries + 1):
            logger.debug("VerificationAgent: probe attempt %d/%d", attempt, self.max_retries)

            container_checks = self._check_containers()
            health_result = self._check_health_endpoint()

            container_issues = self._evaluate_containers(container_checks)
            health_issues = self._evaluate_health(health_result)

            issues = container_issues + health_issues

            if not issues:
                passed = True
                logger.info(
                    "VerificationAgent: all checks passed on attempt %d.", attempt
                )
                break

            if attempt < self.max_retries:
                logger.info(
                    "VerificationAgent: %d issue(s) found on attempt %d — retrying in %.1fs.",
                    len(issues),
                    attempt,
                    self.retry_delay,
                )
                time.sleep(self.retry_delay)

        # Cross-check: if no action was executed and the service is still
        # unhealthy, that is an additional finding.
        if not action_executed and not passed:
            issues.append(
                "No remediation action was executed — the underlying issue may be unresolved."
            )

        # Summarise the remediation that was attempted.
        action_summary = self._summarise_action(remediation_plan, action_executed)

        final_health = health_result.get("health", "unknown")

        verification_result = {
            "status": "resolved" if passed else "failed",
            "service_health": final_health,
            "container_statuses": container_checks,
            "health_endpoint": health_result,
            "action_summary": action_summary,
            "issues": issues,
            "probe_attempts": self.max_retries if not passed else None,
        }

        logger.info(
            "VerificationAgent: result=%s  health=%s  issues=%d",
            verification_result["status"],
            final_health,
            len(issues),
        )

        return {
            "verification_result": verification_result,
            "resolved": passed,
            "service_health": final_health,
        }

    # ------------------------------------------------------------------
    # Probe helpers
    # ------------------------------------------------------------------

    def _check_containers(self) -> Dict[str, str]:
        """Return a dict of container_name → status for all required containers."""
        results: Dict[str, str] = {}
        for name in _REQUIRED_CONTAINERS:
            try:
                info = get_container_status(name)
                results[name] = info.get("status", "unknown")
            except Exception as exc:  # noqa: BLE001
                logger.warning("VerificationAgent: could not check container '%s': %s", name, exc)
                results[name] = "error"
        return results

    def _check_health_endpoint(self) -> Dict[str, str]:
        """Query the payment-api health endpoint."""
        try:
            return health_check("payment-api")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VerificationAgent: health check raised: %s", exc)
            return {"service": "payment-api", "health": "error"}

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_containers(container_checks: Dict[str, str]) -> List[str]:
        """Return a list of issue strings for any container not in an acceptable state."""
        # In the demo environment mysql is intentionally 'exited' before
        # remediation; after a restart it transitions to 'running'.
        # We accept both 'running' and 'restarting' as non-failed states.
        acceptable = {"running", "restarting", "healthy"}
        issues: List[str] = []
        for name, status in container_checks.items():
            if status not in acceptable:
                issues.append(
                    f"Container '{name}' is in state '{status}' — expected one of {sorted(acceptable)}."
                )
        return issues

    @staticmethod
    def _evaluate_health(health_result: Dict[str, str]) -> List[str]:
        """Return issues if the health endpoint is not healthy."""
        health = health_result.get("health", "unknown")
        if health not in {"healthy", "ok"}:
            return [
                f"Health endpoint for '{health_result.get('service', 'unknown')}' "
                f"returned '{health}'."
            ]
        return []

    @staticmethod
    def _summarise_action(
        remediation_plan: Dict[str, Any], action_executed: bool
    ) -> Dict[str, Any]:
        if not remediation_plan:
            return {"executed": False, "detail": "No remediation plan was recorded."}

        return {
            "executed": action_executed,
            "action": remediation_plan.get("action", "unknown"),
            "action_type": remediation_plan.get("action_type", "unknown"),
            "target": remediation_plan.get("target", "unknown"),
            "risk": remediation_plan.get("risk", "unknown"),
            "mutating": remediation_plan.get("mutating", False),
            "sequence": remediation_plan.get("action_sequence", []),
        }
