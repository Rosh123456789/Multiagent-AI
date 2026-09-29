from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from agents.diagnosis import DiagnosisAgent
from agents.remediation import RemediationAgent
from agents.supervisor import SupervisorAgent, WorkflowStep
from agents.verification import VerificationAgent
from rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LangGraph import — the workflow runs without it.
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.info("langgraph not available — DeepShieldWorkflow will use the fallback loop.")


# ---------------------------------------------------------------------------
# Shared state type alias (plain dict for LangGraph compat)
# ---------------------------------------------------------------------------
WorkflowState = Dict[str, Any]


def _make_initial_state(
    application: str,
    issue: str,
    deployment: Optional[Dict[str, Any]],
) -> WorkflowState:
    return {
        "application": application,
        "issue": issue,
        "deployment": deployment or {"status": "unhealthy"},
        "service_health": "unhealthy",
    }


# ---------------------------------------------------------------------------
# DeepShieldWorkflow
# ---------------------------------------------------------------------------

class DeepShieldWorkflow:
    """
    Orchestrates the self-healing pipeline:

        Diagnosis → Knowledge → Remediation → Verification

    Execution modes
    ---------------
    1. **LangGraph graph** (preferred when langgraph is installed):
       A compiled ``StateGraph`` with conditional edges driven by the
       Supervisor's routing logic.  Each node is an agent step; the supervisor
       decides the next node after every step and can short-circuit to END
       on escalation or completion.

    2. **Fallback loop** (always available):
       ``SupervisorAgent.run_loop()`` drives the same agents through an
       ``agent_map`` dict without requiring langgraph.

    Usage
    -----
    ::

        wf = DeepShieldWorkflow()
        result = wf.execute(
            application="Payment API",
            issue="Payment API cannot connect to MySQL",
        )
    """

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        *,
        dry_run: bool = False,
        use_graph: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        knowledge_dir:
            Path to the markdown runbook directory (relative to CWD or absolute).
        dry_run:
            Passed to ``RemediationAgent`` — plans actions without executing them.
        use_graph:
            Force graph mode (True) or fallback loop (False).
            Defaults to auto-detect based on langgraph availability.
        """
        self.dry_run = dry_run
        self._use_graph = _LANGGRAPH_AVAILABLE if use_graph is None else use_graph

        self.knowledge_base = KnowledgeBase(knowledge_dir)
        self.supervisor = SupervisorAgent()
        self.diagnosis = DiagnosisAgent()
        self.remediation = RemediationAgent(dry_run=dry_run)
        self.verification = VerificationAgent()

        self._graph = None
        if self._use_graph:
            self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        application: str = "Payment API",
        issue: str = "Payment API cannot connect to MySQL",
        deployment: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Run the full self-healing workflow and return the final state.

        The returned dict always contains:
        - ``root_cause``, ``severity``, ``confidence``, ``diagnosis_source``
        - ``runbook``
        - ``remediation_plan``, ``planned_action``, ``action_executed``
        - ``verification_result``, ``resolved``, ``service_health``
        - ``workflow_trail``  (list of supervisor decisions)
        - ``execution_mode``  ("graph" | "loop")
        """
        state = _make_initial_state(application, issue, deployment)
        self.supervisor.reset()

        if self._graph is not None:
            logger.info("DeepShieldWorkflow: executing via LangGraph StateGraph.")
            state = self._run_graph(state)
            state["execution_mode"] = "graph"
        else:
            logger.info("DeepShieldWorkflow: executing via supervisor fallback loop.")
            state = self._run_loop(state)
            state["execution_mode"] = "loop"

        return state

    # ------------------------------------------------------------------
    # LangGraph graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build and compile the LangGraph StateGraph."""
        if not _LANGGRAPH_AVAILABLE:
            return None

        try:
            graph: StateGraph = StateGraph(dict)

            # ── Nodes ──────────────────────────────────────────────────
            graph.add_node(WorkflowStep.DIAGNOSIS.value, self._node_diagnosis)
            graph.add_node(WorkflowStep.KNOWLEDGE.value, self._node_knowledge)
            graph.add_node(WorkflowStep.REMEDIATION.value, self._node_remediation)
            graph.add_node(WorkflowStep.VERIFICATION.value, self._node_verification)

            # ── Entry point ────────────────────────────────────────────
            graph.set_entry_point(WorkflowStep.DIAGNOSIS.value)

            # ── Conditional edges — the supervisor decides next step ───
            # After diagnosis: route to knowledge or END (escalate).
            graph.add_conditional_edges(
                WorkflowStep.DIAGNOSIS.value,
                self._route_after_diagnosis,
                {
                    WorkflowStep.KNOWLEDGE.value: WorkflowStep.KNOWLEDGE.value,
                    END: END,
                },
            )

            # After knowledge: always go to remediation.
            graph.add_edge(WorkflowStep.KNOWLEDGE.value, WorkflowStep.REMEDIATION.value)

            # After remediation: always go to verification.
            graph.add_edge(WorkflowStep.REMEDIATION.value, WorkflowStep.VERIFICATION.value)

            # After verification: always end.
            graph.add_edge(WorkflowStep.VERIFICATION.value, END)

            compiled = graph.compile()
            logger.info("DeepShieldWorkflow: LangGraph StateGraph compiled successfully.")
            return compiled

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DeepShieldWorkflow: LangGraph graph build failed (%s) — falling back to loop.",
                exc,
            )
            return None

    # ── Node functions (called by LangGraph) ──────────────────────────
    # LangGraph with StateGraph(dict) replaces state with the node's return
    # value — so every node must return the FULL merged state, not a partial.

    def _node_diagnosis(self, state: WorkflowState) -> WorkflowState:
        logger.debug("LangGraph node: diagnosis")
        partial = self.diagnosis.analyze(state)
        return {**state, **partial}

    def _node_knowledge(self, state: WorkflowState) -> WorkflowState:
        logger.debug("LangGraph node: knowledge")
        runbook = self.knowledge_base.get_runbook(state.get("issue", ""))
        return {**state, "runbook": runbook}

    def _node_remediation(self, state: WorkflowState) -> WorkflowState:
        logger.debug("LangGraph node: remediation")
        partial = self.remediation.plan(state)
        return {**state, **partial}

    def _node_verification(self, state: WorkflowState) -> WorkflowState:
        logger.debug("LangGraph node: verification")
        partial = self.verification.verify(state)
        return {**state, **partial}

    # ── Conditional routing functions ──────────────────────────────────

    def _route_after_diagnosis(self, state: WorkflowState) -> str:
        """Route to knowledge retrieval, or END if escalation is needed."""
        decision = self.supervisor.decide_next_step(state)
        if decision.get("escalated"):
            logger.warning(
                "DeepShieldWorkflow: supervisor escalated after diagnosis — ending graph."
            )
            return END
        next_agent = decision.get("next_agent", WorkflowStep.KNOWLEDGE.value)
        # Map "done" → END as well.
        if next_agent in {WorkflowStep.DONE.value, "done", "escalate"}:
            return END
        return next_agent

    # ------------------------------------------------------------------
    # Graph execution (LangGraph invoke)
    # ------------------------------------------------------------------

    def _run_graph(self, state: WorkflowState) -> WorkflowState:
        """Invoke the compiled LangGraph graph and merge the result back."""
        try:
            trail: list[Dict[str, Any]] = []
            state.setdefault("workflow_trail", trail)

            result = self._graph.invoke(state)

            # LangGraph returns a new dict; merge it back so we always have
            # the full state including keys the graph may not have touched.
            merged = {**state, **result}
            merged.setdefault("workflow_trail", trail)
            return merged
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "DeepShieldWorkflow: LangGraph invoke failed (%s) — falling back to loop.", exc
            )
            # Fall through to the supervisor loop.
            return self._run_loop(state)

    # ------------------------------------------------------------------
    # Fallback loop (supervisor-driven, no langgraph dependency)
    # ------------------------------------------------------------------

    def _run_loop(self, state: WorkflowState) -> WorkflowState:
        """Drive the pipeline via SupervisorAgent.run_loop()."""
        agent_map = {
            WorkflowStep.DIAGNOSIS.value: self.diagnosis.analyze,
            WorkflowStep.KNOWLEDGE.value: self._knowledge_handler,
            WorkflowStep.REMEDIATION.value: self.remediation.plan,
            WorkflowStep.VERIFICATION.value: self.verification.verify,
        }
        return self.supervisor.run_loop(state, agent_map)

    def _knowledge_handler(self, state: WorkflowState) -> WorkflowState:
        """Wrap KnowledgeBase.get_runbook so it fits the agent_map signature."""
        runbook = self.knowledge_base.get_runbook(state.get("issue", ""))
        return {"runbook": runbook}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_status(self, state: WorkflowState) -> Dict[str, Any]:
        """Return a human-readable progress snapshot for the given state."""
        return self.supervisor.get_workflow_status(state)
