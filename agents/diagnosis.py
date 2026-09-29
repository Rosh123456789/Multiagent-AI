from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama integration (optional — graceful fallback when unavailable)
# ---------------------------------------------------------------------------
try:
    import ollama  # type: ignore[import-not-found]

    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.info("ollama package not installed — DiagnosisAgent will use rule-based fallback.")

# The model name can be overridden via the DEEPSHIELD_OLLAMA_MODEL env var.
_DEFAULT_MODEL = "llama3.2"

# System prompt that shapes the LLM's output into a predictable JSON structure.
_SYSTEM_PROMPT = """\
You are an expert SRE / platform engineer diagnosing operational incidents.

Given an issue description and deployment context you must:
1. Identify the most probable root cause.
2. List up to three contributing factors.
3. Suggest a severity level: critical | high | medium | low.
4. Return ONLY a valid JSON object — no prose, no markdown, no code fences.

Response schema (all fields required):
{
  "root_cause": "<one concise sentence>",
  "contributing_factors": ["<factor 1>", "<factor 2>"],
  "severity": "critical | high | medium | low",
  "confidence": 0.0–1.0
}
"""

# ---------------------------------------------------------------------------
# Rule-based fallback patterns
# ---------------------------------------------------------------------------
_RULE_PATTERNS: list[Dict[str, Any]] = [
    {
        "keywords": ["mysql", "database", "db connection", "cannot connect"],
        "root_cause": (
            "MySQL dependency is unavailable or misconfigured; "
            "the payment API cannot establish a database connection."
        ),
        "contributing_factors": [
            "MySQL container may be stopped or crash-looping.",
            "Network policy or firewall is blocking port 3306.",
            "Database credentials or host configuration are incorrect.",
        ],
        "severity": "critical",
    },
    {
        "keywords": ["crashloop", "crash loop", "oomkilled", "restart"],
        "root_cause": (
            "The container is in a crash-loop, likely due to a missing dependency "
            "or an unhandled startup error."
        ),
        "contributing_factors": [
            "Dependent service is not yet ready at startup.",
            "Missing or invalid environment variable.",
            "Application throws an unhandled exception on boot.",
        ],
        "severity": "high",
    },
    {
        "keywords": ["cpu", "high cpu", "resource", "throttl"],
        "root_cause": "Resource saturation is causing elevated latency and degraded throughput.",
        "contributing_factors": [
            "No CPU limit set on the container.",
            "Unexpected traffic spike or runaway goroutine/thread.",
            "Inefficient query or tight loop in application code.",
        ],
        "severity": "high",
    },
    {
        "keywords": ["memory", "oom", "out of memory", "heap"],
        "root_cause": "Memory pressure is causing the service to be OOM-killed or swap-thrashing.",
        "contributing_factors": [
            "Memory limit too low for the workload.",
            "Memory leak in application code.",
            "Unbounded cache growth without eviction policy.",
        ],
        "severity": "high",
    },
    {
        "keywords": ["health", "unhealthy", "health check", "liveness", "readiness"],
        "root_cause": (
            "The service health endpoint is returning a non-2xx status; "
            "the container has been marked unhealthy by the orchestrator."
        ),
        "contributing_factors": [
            "Downstream dependency failure propagating to the health check.",
            "Application is still initialising when the probe fires.",
            "Health endpoint has a logic bug returning 5xx unconditionally.",
        ],
        "severity": "high",
    },
    {
        "keywords": ["service down", "not responding", "timeout", "connection refused"],
        "root_cause": "The service is not accepting connections; it may have exited or be overwhelmed.",
        "contributing_factors": [
            "Process crashed without a restart policy.",
            "Port binding conflict with another process.",
            "Load balancer routing to a dead instance.",
        ],
        "severity": "critical",
    },
]


class DiagnosisAgent:
    """
    Analyses deployment health and identifies the likely root cause of an incident.

    Strategy
    --------
    1. Try to call the local Ollama LLM for a structured JSON diagnosis.
    2. If Ollama is unavailable or returns unparseable output, fall back to
       the deterministic rule-based engine.
    3. Always populate the same output keys regardless of which path was taken.

    Output keys added to state
    --------------------------
    - ``root_cause``        (str)  One-sentence root-cause description.
    - ``contributing_factors`` (list[str])
    - ``severity``          (str)  critical | high | medium | low
    - ``confidence``        (float) 0.0–1.0
    - ``diagnosis_source``  (str)  "ollama" | "rules"
    - ``deployment_checked`` (bool) Always True.
    - ``status``            (str)  Deployment status from input state.
    - ``analysis``          (dict) Full structured analysis for downstream use.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("DEEPSHIELD_OLLAMA_MODEL", _DEFAULT_MODEL)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        deployment = state.get("deployment", {})
        status = deployment.get("status", "unknown")
        issue = state.get("issue", "")
        application = state.get("application", "unknown application")

        logger.info("DiagnosisAgent: analysing issue='%s' status='%s'", issue, status)

        # Attempt LLM diagnosis first.
        llm_result = self._diagnose_with_ollama(application, issue, status)

        if llm_result:
            source = "ollama"
            result = llm_result
        else:
            source = "rules"
            result = self._diagnose_with_rules(issue, status)

        logger.info(
            "DiagnosisAgent: source=%s  severity=%s  root_cause=%s",
            source,
            result["severity"],
            result["root_cause"],
        )

        analysis = {
            "deployment_status": status,
            "issue": issue,
            "likely_root_cause": result["root_cause"],
            "contributing_factors": result["contributing_factors"],
            "severity": result["severity"],
            "confidence": result["confidence"],
            "diagnosis_source": source,
        }

        return {
            "deployment_checked": True,
            "status": status,
            "root_cause": result["root_cause"],
            "contributing_factors": result["contributing_factors"],
            "severity": result["severity"],
            "confidence": result["confidence"],
            "diagnosis_source": source,
            "analysis": analysis,
        }

    # ------------------------------------------------------------------
    # Ollama path
    # ------------------------------------------------------------------

    def _diagnose_with_ollama(
        self, application: str, issue: str, status: str
    ) -> Optional[Dict[str, Any]]:
        if not _OLLAMA_AVAILABLE:
            return None

        user_prompt = (
            f"Application: {application}\n"
            f"Reported issue: {issue}\n"
            f"Deployment status: {status}\n\n"
            "Diagnose this incident and respond with the JSON schema described in the system prompt."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.1},  # Low temperature for deterministic output.
            )
            raw = response["message"]["content"].strip()
            parsed = self._parse_llm_json(raw)
            if parsed and self._validate_llm_result(parsed):
                return parsed
            logger.warning("DiagnosisAgent: LLM returned invalid/incomplete JSON — falling back to rules.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DiagnosisAgent: Ollama call failed (%s) — falling back to rules.", exc)

        return None

    def _parse_llm_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Extract and parse a JSON object from raw LLM output."""
        # Strip markdown code fences if the model included them despite the prompt.
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        # Find the first {...} block.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.debug("DiagnosisAgent: JSON parse error: %s", exc)
            return None

    @staticmethod
    def _validate_llm_result(result: Dict[str, Any]) -> bool:
        required = {"root_cause", "contributing_factors", "severity", "confidence"}
        if not required.issubset(result.keys()):
            return False
        if result["severity"] not in {"critical", "high", "medium", "low"}:
            return False
        try:
            conf = float(result["confidence"])
            if not (0.0 <= conf <= 1.0):
                return False
        except (TypeError, ValueError):
            return False
        return True

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _diagnose_with_rules(self, issue: str, status: str) -> Dict[str, Any]:
        """Match the issue string against known keyword patterns."""
        combined = f"{issue} {status}".lower()

        for pattern in _RULE_PATTERNS:
            if any(kw in combined for kw in pattern["keywords"]):
                return {
                    "root_cause": pattern["root_cause"],
                    "contributing_factors": pattern["contributing_factors"],
                    "severity": pattern["severity"],
                    "confidence": 0.85,  # Rule matches are high-confidence but not perfect.
                }

        # Generic fallback when no pattern matches.
        return {
            "root_cause": (
                f"Unrecognised issue reported for deployment in '{status}' state. "
                "Manual investigation recommended."
            ),
            "contributing_factors": [
                "Issue does not match any known failure pattern.",
                "Deployment is in an unexpected state.",
            ],
            "severity": "medium",
            "confidence": 0.3,
        }
