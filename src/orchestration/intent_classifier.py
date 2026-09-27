from typing import Any, Dict, Iterable, List, Optional
import json
import logging

from context import SharedContext

logger = logging.getLogger(__name__)

_INTENT_TO_WORKFLOW = {
    "code_change": "default_build",
    "deploy": "default_build",
}

# Intents are router-level concepts (what kind of action to take), not
# workflow-level (which phase). Workflows can add new phases / gates without
# adding new intents — feedback on a custom gate is still `intent=feedback`.
_ALLOWED_INTENTS = {
    "code_change", "deploy", "question", "revert", "approval", "feedback", "general",
}


def _summarize_workflow_nodes(workflow_def: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Compact node summary for the LLM to reason about.

    Returns one entry per non-terminal node so the classifier can map user
    feedback onto specific workflow targets (e.g. "redo requirements" picks
    the `requirements` phase out of an arbitrary workflow). No hardcoded
    filtering on phase names — every workflow-defined node passes through.
    """
    if not workflow_def:
        return []
    out: List[Dict[str, str]] = []
    for n in workflow_def.get("nodes") or []:
        kind = (n.get("type") or "").strip()
        if kind in ("start", "end"):
            continue
        out.append({
            "id": n.get("id") or "",
            "kind": kind,
            "phase": n.get("phase_label") or "",
            "label": n.get("label") or n.get("task_type") or "",
            "agent_type": n.get("agent_type") or "",
        })
    return out


def _derive_allowed_agents(
    workflow_def: Optional[Dict[str, Any]],
    available_agent_types: Optional[Iterable[str]],
) -> set:
    """Union of agent_types declared in the workflow and agents registered in
    the orchestrator pool. Empty set means "we have no list to validate
    against" — caller passes through whatever the LLM returned.
    """
    agents: set = set()
    if workflow_def:
        for n in workflow_def.get("nodes") or []:
            t = n.get("agent_type")
            if t:
                agents.add(t)
    if available_agent_types:
        agents.update(t for t in available_agent_types if t)
    return agents


class IntentClassifier:
    def __init__(self, llm_client, storage=None):
        self.llm_client = llm_client
        self.storage = storage

    async def classify_user_intent(
            self,
            message: str,
            project_state: Dict[str, Any],
            shared_context: Optional[SharedContext] = None,
            workflow_def: Optional[Dict[str, Any]] = None,
            pending_approval_type: Optional[str] = None,
            available_agent_types: Optional[Iterable[str]] = None,
            tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            phase = project_state.get("phase") or project_state.get("current_phase") or ""
            nodes_summary = _summarize_workflow_nodes(workflow_def)
            allowed_agents = _derive_allowed_agents(workflow_def, available_agent_types)

            workflow_block = ""
            if nodes_summary:
                workflow_block = (
                    "\nThe active workflow has these nodes (phases, gates, etc.):\n"
                    f"{json.dumps(nodes_summary, ensure_ascii=False)}\n"
                )
            approval_block = ""
            if pending_approval_type:
                approval_block = (
                    f"\nA '{pending_approval_type}' approval gate is currently pending — "
                    "the workflow is waiting for the user to approve, reject, or refine the "
                    "artifact produced for it. If the user's message looks like feedback on "
                    f"the pending '{pending_approval_type}' artifact, return intent='feedback' "
                    "and set target_phase to the matching phase from the workflow nodes above. "
                    "If the user is clearly approving ('looks good', 'ship it', 'ok') or rejecting "
                    "('no', 'scrap this'), return intent='approval'.\n"
                )

            system_prompt = (
                "Analyze user intent in a workflow-driven project. Return JSON:\n"
                "{\n"
                '  "intent": "code_change" | "deploy" | "question" | "revert" | "approval" | "feedback" | "general",\n'
                '  "agent": "<agent_type from workflow>" | null,\n'
                '  "target_phase": "<phase_label from workflow>" | null,\n'
                '  "summary": "brief description of what user wants"\n'
                "}\n\n"
                "Intent definitions:\n"
                "- code_change: User wants to ADD, MODIFY, FIX, or IMPLEMENT something in the "
                "code. Includes bug reports — if user says something is broken, they want it FIXED.\n"
                "- question: User wants INFORMATION or EXPLANATION without code changes.\n"
                "- deploy: User wants to deploy/publish the project.\n"
                "- revert: User wants to undo/go back/revert changes.\n"
                "- approval: User is approving or rejecting the current pending approval "
                "('looks good', 'ship it', 'no, reject').\n"
                "- feedback: User is providing feedback on a pending approval artifact OR on a "
                "previously-completed phase, asking for changes/revisions ('single task please', "
                "'redo the requirements', 'add more detail to the plan').\n"
                "- general: Casual conversation or genuinely unclear intent. Use sparingly — "
                "if the user mentions a workflow node or artifact, prefer feedback/code_change.\n\n"
                f"Current workflow phase: {phase or '(not started)'}"
                + workflow_block + approval_block +
                "\nIMPORTANT: User is NOT locked to phases. They can ask to change a completed "
                "phase from a later phase. Set `target_phase` to the workflow phase the user is "
                "talking about, which may differ from the current phase. When in doubt between "
                "code_change and question, prefer code_change if user implies they want something DONE."
            )

            if shared_context:
                model = shared_context.get_model("intent_classifier")
            else:
                # No loaded project (evicted or pre-load): resolve the tenant's
                # admin-configured default instead of a hardcoded gpt-5-mini,
                # which is banned on some tenants (e.g. VK prod).
                from llm.agent_model_params import resolve_tenant_default_model
                model, _ = await resolve_tenant_default_model(self.storage, tenant_id)

            payload = await self.llm_client.chat_completion_with_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                model=model,
                temperature=0,
            )

            if not isinstance(payload, dict):
                raise TypeError("LLM returned non-dict JSON")

            intent = (payload.get("intent") or "general").strip().lower()
            agent = payload.get("agent")
            if isinstance(agent, str):
                agent = agent.strip().lower() or None
            else:
                agent = None
            target_phase = payload.get("target_phase")
            if isinstance(target_phase, str):
                target_phase = target_phase.strip().lower() or None
            else:
                target_phase = None
            summary = payload.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                summary = message.strip()[:200]

            if intent not in _ALLOWED_INTENTS:
                intent = "general"

            # When we have a list of valid agents (from workflow + pool), an
            # agent value outside that list is logged but kept — the router
            # downstream can decide whether to ignore. Silently nulling here
            # is the bug we're avoiding: it dropped legitimate workflow
            # agents like `requirements_finalizer` because they weren't in
            # the old hardcoded {"coding","planner","qa"} set.
            if agent and allowed_agents and agent not in allowed_agents:
                logger.info(
                    "[INTENT] agent='%s' not in workflow-allowed set %s — keeping as-is",
                    agent, sorted(allowed_agents),
                )

            return {
                "intent": intent,
                "agent": agent,
                "target_phase": target_phase,
                "summary": summary.strip(),
                "workflow_id": (workflow_def or {}).get("_id") or _INTENT_TO_WORKFLOW.get(intent),
            }
        except Exception:
            return self._heuristic_fallback(message)

    def _heuristic_fallback(self, message: str) -> Dict[str, Any]:
        """Used only when the LLM call itself errors. Keyword-based, intentionally crude."""
        msg_lower = (message or "").lower().strip()
        summary = msg_lower[:200]

        code_keywords = [
            "fix", "add", "change", "update", "implement", "modify", "edit",
            "create", "remove", "delete", "refactor", "bug", "error", "broken",
            "not working", "doesn't work", "doesnt work", "is not defined",
            "make it", "improve", "enhance",
        ]

        for kw in code_keywords:
            if kw in msg_lower:
                return {"intent": "code_change", "agent": "coding", "target_phase": None, "summary": summary, "workflow_id": "default_build"}

        if any(kw in msg_lower for kw in ["deploy", "publish", "release", "ship"]):
            return {"intent": "deploy", "agent": None, "target_phase": None, "summary": summary, "workflow_id": "default_build"}

        if any(kw in msg_lower for kw in ["revert", "undo", "go back", "rollback"]):
            return {"intent": "revert", "agent": None, "target_phase": None, "summary": summary, "workflow_id": None}

        return {"intent": "general", "agent": None, "target_phase": None, "summary": summary, "workflow_id": None}
