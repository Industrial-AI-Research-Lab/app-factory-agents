"""
Generic Agent — config-driven agent loaded from MongoDB.

One class replaces ALL specialized agents. Behavior is entirely
determined by the configuration dictionary (system_prompt, model,
temperature, allowed_tools, allowed_phases, etc.).

For agents with tools (e.g. coding), uses StreamingAgentRunner or
SimpleAgentRunner + AgentToolDispatcher.
"""

from __future__ import annotations

import json
import logging
import math
import time
from copy import deepcopy
from numbers import Real
from typing import Any, Dict, List, Optional

from llm.agent_model_params import DEFAULT_AGENT_MODEL
from .base import BaseAgent, AgentType
from .followup_scope import bind_tool_constraints, call_matches_scope
from .tool_dispatcher import AgentToolDispatcher
from .streaming_agent_runner import StreamingAgentRunner
from .simple_agent_runner import SimpleAgentRunner
from .tool_failure_budget import ToolFailureLimitExceeded
from storage.tool_value_store import ValueReferenceError
from .output_repair import (
    announce_output_repair,
    output_rejection_message,
    output_repair_budget,
    output_repairable,
)
from .run_rehydration import build_resume_input_items, load_attempt_pairs
from .terminal_contract import (
    apply_terminal_gate,
    tool_events_from_simple_runner,
    tool_events_from_streaming_runner,
)
from schemas import ResultSchema, TaskSchema, TaskStatus
from orchestration.workflow_task_lifecycle import (
    EXTERNAL_TOOL_OUTCOME_UNKNOWN,
    SESSION_UNAVAILABLE_STOP,
    ExternalToolOutcomeUnknownError,
    SessionUnavailableStopError,
)
from tools.agent_allowed_tools import (
    effective_allowed_tool_ids,
    normalize_agent_tool_allowlists,
)
from tools.agent_tool_schemas import get_tool_schemas_for_agent
from tools.archive_tools import (
    ARCHIVE_TOOL_IDS,
    agent_can_archive_fetch,
    archive_tool_schemas,
)
from tools.attachment_tools import attachment_tool_schemas, project_has_attachments
from tools.followup_limits import MAX_CARD_ROWS
from tools.tenant_artifact_tools import tenant_artifact_tool_schemas, tenant_has_artifacts
from config.configuration_resolution import agent_display_name_from_doc, agent_wire_name_from_doc
from tools.prompt_tool_catalog import (
    ALLOWED_TOOLS_CATALOG_PLACEHOLDER,
    TOOL_CATALOG_INTRO_LINE,
    apply_tool_catalog_placeholder,
    build_tool_catalog_text,
)
from plugins.host import build_plugin_host
from plugins.config_chain import plugin_flag
from .delegation_policy import DELEGATE_TO_AGENT_TOOL, delegation_activation_error
from .delegation_targets import (
    bind_delegation_targets_to_tool_schemas,
    format_delegation_targets_prompt,
    resolve_a2a_delegation_targets,
    resolve_project_delegation_targets,
)
from utils.llm_json import LlmJsonParseError, parse_llm_json_object

logger = logging.getLogger(__name__)

_RUNNER_STOPS = (
    SessionUnavailableStopError,
    ExternalToolOutcomeUnknownError,
    ToolFailureLimitExceeded,
)


class GenericAgent(BaseAgent):
    """Config-driven agent. All behavior comes from `config` dict."""

    def __init__(self, config: Dict[str, Any]):
        agent_type = _resolve_agent_type(config.get("type", "generic"))

        runtime_tenant = str(config.get("tenant_id") or "").strip()
        wire_id = agent_wire_name_from_doc(config, runtime_tenant_id=runtime_tenant) or str(
            config.get("_id") or config.get("agent_id") or "generic_agent",
        )
        super().__init__(
            agent_id=wire_id,
            agent_type=agent_type,
            model=config.get("model", DEFAULT_AGENT_MODEL),
            temperature=config.get("temperature", 1),
            evaluation_model=config.get("evaluation_model"),
            allowed_phases=config.get("allowed_phases", "all"),
        )

        reg_tools, mcp_tools = normalize_agent_tool_allowlists(
            config.get("allowed_tools"),
            config.get("allowed_mcp_tools"),
        )
        self.config = dict(config)
        self.config["allowed_tools"] = reg_tools
        self.config["allowed_mcp_tools"] = mcp_tools
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        self.output_save_key: Optional[str] = config.get("output_save_key")
        self.allowed_tools: List[str] = reg_tools
        self.allowed_mcp_tools: List[str] = mcp_tools
        self._effective_allowed_tools: List[str] = effective_allowed_tool_ids(
            self.config,
            allowed_tools=reg_tools,
            allowed_mcp_tools=mcp_tools,
        )
        self.allowed_delegation_targets: List[str] | None = config.get("allowed_delegation_targets")
        self.use_streaming: bool = config.get("use_streaming", True)
        self.reasoning_effort: Optional[str] = config.get("reasoning_effort")
        self.display_name: str = agent_display_name_from_doc(config)
        self.step_limit: int = config.get("step_limit") or 30

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def get_display_name(self) -> str:
        return self.display_name

    # ------------------------------------------------------------------
    # evaluate_task — LLM fit evaluation
    # ------------------------------------------------------------------

    async def evaluate_task(self, task: Dict[str, Any]) -> float:
        # eval_keywords substring matching was removed: "plan" matched
        # "explain the plan", scores were a binary 0.9/0.1. Config may still
        # carry the field (old bundles) — it is deliberately ignored.
        try:
            task_context = await self._build_task_context_async(task)
            eval_prompt = (
                f"You are evaluating whether you should handle a task.\n"
                f"Your role: {self.display_name}\n"
                f"Your capabilities (from system prompt): {self.system_prompt[:300]}\n\n"
                f"{task_context}\n\n"
                f"Respond with JSON: {{\"should_handle\": true/false, \"fit_score\": 0.0-1.0, "
                f"\"reasoning\": \"brief\"}}"
            )
            messages = [
                {"role": "system", "content": "You evaluate task-agent fit. Respond ONLY with JSON."},
                {"role": "user", "content": eval_prompt},
            ]
            response = await self.call_llm(
                messages,
                model=self.evaluation_model,
                temperature=0.3,
                # Resolve through run_config.models["evaluation"] so bids stay
                # on a cheap model even when agents run expensive main models.
                subsystem="evaluation",
            )
            result = parse_llm_json_object(response)
            should_handle = result.get("should_handle")
            fit_score = result.get("fit_score")
            if not isinstance(should_handle, bool):
                raise LlmJsonParseError("bid should_handle must be a boolean")
            if (
                isinstance(fit_score, bool)
                or not isinstance(fit_score, Real)
                or not math.isfinite(fit_score)
                or not 0.0 <= fit_score <= 1.0
            ):
                raise LlmJsonParseError("bid fit_score must be a finite number from 0 to 1")
            return fit_score if should_handle else 0.0
        except Exception as e:
            logger.warning(
                "[AUCTION] agent=%s bid_evaluation_failed error_type=%s",
                self.agent_id,
                type(e).__name__,
            )
            return 0.0

    # ------------------------------------------------------------------
    # execute_task — tool-using (runner) or simple (prompt→LLM→result)
    # ------------------------------------------------------------------

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.current_task = task
        self._exact_tool_runtime = None
        self.ensure_not_cancelled("execute_task")
        description = task.get("description", "")

        if self._effective_allowed_tools:
            return await self._execute_with_tools(task, description)
        # F4: tool-less agents (requirements/planner/QA/…) still need attachment_*
        # when the project already has uploads — otherwise they only see inline meta.
        if await self._attachments_present() or await self._tenant_artifacts_present():
            return await self._execute_with_tools(task, description)
        return await self._execute_simple(task, description)

    # ------------------------------------------------------------------
    # Tool-using execution (StreamingAgentRunner / SimpleAgentRunner)
    # ------------------------------------------------------------------

    async def _fetch_registry_tool_schemas(self, allowed_tool_ids) -> List[Dict]:
        """OpenAI function schemas for ``allowed_tool_ids``, tenant-scoped.

        Prefers the in-memory tool_registry; falls back to a direct DB read.
        Returns [] when neither is available or the allow-list is empty.
        """
        allow = list(allowed_tool_ids or []) or None
        tenant_id = self.shared_context.tenant_id if self.shared_context else None
        if self.tool_registry and hasattr(self.tool_registry, "get_schemas_for_agent") and self.tool_registry.tools:
            tools = self.tool_registry.get_schemas_for_agent(
                self.agent_id,
                allowed_tool_ids=allow,
                tenant_id=tenant_id,
            )
            logger.info(
                "[AGENT] %s loaded %d tool schemas from registry "
                "(allowed_tools=%s allowed_mcp_tools=%s registry_count=%d)",
                self.agent_id,
                len(tools),
                self.allowed_tools,
                self.allowed_mcp_tools,
                len(self.tool_registry.tools),
            )
            return tools
        storage = getattr(self.shared_context, "storage", None) if self.shared_context else None
        tools = await get_tool_schemas_for_agent(
            self.agent_id,
            storage,
            allowed_tool_ids=allow,
            tenant_id=tenant_id,
        ) if storage else []
        logger.info(f"[AGENT] {self.agent_id} loaded {len(tools)} tool schemas from DB fallback (registry={bool(self.tool_registry)}, has_storage={bool(storage)})")
        return tools

    async def _splice_conditional_tools(self, tools: List[Dict]) -> List[Dict]:
        """Append archive / attachment / tenant-artifact schemas when this
        project already holds the matching data.

        These ride OUTSIDE the per-agent allowlist by design; archive_fetch still
        requires bash/read in the effective allowlist. Kept append-only so a
        tool-less agent can still run with attachment_* alone.
        """
        if await self._archive_refs_present():
            tools = tools + self._archive_tool_schemas()
        if await self._attachments_present():
            tools = tools + attachment_tool_schemas()
        if await self._tenant_artifacts_present():
            tools = tools + tenant_artifact_tool_schemas()
        return tools

    async def _execute_with_tools(self, task: Dict[str, Any], description: str) -> Dict[str, Any]:
        self.ensure_not_cancelled("execute_with_tools")
        tools = await self._fetch_registry_tool_schemas(self._effective_allowed_tools)
        delegation_targets = (
            await self._resolve_delegation_targets()
            if self._can_show_delegation_tool(task)
            else []
        )
        tools = self._filter_tool_schemas_for_task(
            tools,
            task,
            delegation_targets=delegation_targets,
        )
        tools = await self._splice_conditional_tools(tools)

        self._exact_tool_runtime = None
        if self.config.get("exact_tool_values") is True:
            from agents.exact_tool_runtime import ExactToolRuntime

            self._exact_tool_runtime = ExactToolRuntime(self, tools)
            tools = self._exact_tool_runtime.tools

        if not tools:
            logger.error(
                "[AGENT] %s has no tool schemas (allowed_tools=%s allowed_mcp_tools=%s)",
                self.agent_id,
                self.allowed_tools,
                self.allowed_mcp_tools,
            )
            return ResultSchema.create(
                status=TaskStatus.FAILED,
                output={},
                reasoning="No tool schemas available for tool-enabled agent",
            )

        effective_system_prompt = await self._build_effective_system_prompt(
            task,
            delegation_targets=delegation_targets,
        )
        dispatcher = AgentToolDispatcher(self)

        # Plugin host (ADR-0004): streaming path only, config chain resolved
        # once per execution. pre_flight fires before the Assembled Context is
        # built, context_render while it is — the spec's lifecycle order.
        plugin_host = None
        if self.use_streaming:
            plugin_host = await build_plugin_host(
                agent_config=self.config,
                shared_context=self.shared_context,
                event_emitter=self.event_emitter,
                agent_id=self.agent_id,
                model=self._resolve_project_model(),
                task_id=TaskSchema.get_id(task),
                cancellation_token=self.cancellation_token,
                llm_caller=self._plugin_llm_call,
            )
        if plugin_host is not None:
            await plugin_host.dispatch_pre_flight(
                tools=tools, system_prompt=effective_system_prompt
            )

        # Restart resume: a task marked by the resume trigger re-enters
        # mid-attempt, so its journaled pairs return as structured transcript
        # items and are excluded from the history text render — one copy, not
        # two.
        resume_pairs = await self._load_resume_pairs(task)

        # Conversation-fold seeder (AppFactory-149 slice 5, ADR-0014): its OWN switch,
        # separate from the plugin `enabled` gate above. Default on; when off the
        # valve truncates cross-run history instead of folding it. summary_model
        # keeps the fold on a cheap model (ADR-0014 Decision 8). Streaming only.
        seeder_on = False
        summary_model = None
        keep_recent_conversation_tokens = None
        if self.use_streaming:
            compaction_cfg = await self._resolve_compaction_config()
            seeder_on = plugin_flag(compaction_cfg, "summarize_overflow_conversation", default=True)
            summary_model = compaction_cfg.get("summary_model") or None
            keep_recent_conversation_tokens = compaction_cfg.get("keep_recent_conversation_tokens")

        # Build context. When the seeder is on, streaming folds cross-run
        # conversation into the rolling summary (AppFactory-149 slice 5) instead of the
        # valve dropping it; the summary rides as a separate seeded item below. The
        # sync simple/eval paths — and a disabled seeder — keep the valve.
        task_context = await self._build_task_context_async(
            task,
            exclude_journal_task_id=TaskSchema.get_id(task) if resume_pairs else None,
            fold_conversation=self.use_streaming and seeder_on,
            summary_model=summary_model,
            keep_recent_conversation_tokens=keep_recent_conversation_tokens,
        )
        if plugin_host is not None:
            task_context = await plugin_host.dispatch_context_render(task_context)
        user_content = f"{task_context}\n\nUse available tools to complete the task."

        if self.use_streaming:
            return await self._run_streaming(
                task, user_content, tools, dispatcher, effective_system_prompt,
                resume_pairs=resume_pairs,
                plugin_host=plugin_host,
                seeder_on=seeder_on,
            )
        else:
            return await self._run_simple_runner(
                task, user_content, tools, dispatcher, effective_system_prompt
            )

    async def _archive_refs_present(self) -> bool:
        """True when this (project, run) already holds at least one archive ref
        — the attach condition for the archive retrieval tools.

        Run-scoped on purpose: a ref from a previous run is not in this run's
        context, so its tools would be dead weight in the schema. Any failure
        (no storage, no run, query error) means "don't attach" — the mid-run
        dispatcher hook still covers a fresh spill.
        """
        sc = self.shared_context
        storage = getattr(sc, "storage", None) if sc else None
        project_id = getattr(sc, "project_id", None) if sc else None
        run_id = getattr(sc, "run_id", None) if sc else None
        refs = getattr(storage, "archive_refs", None)
        if refs is None or not project_id or not run_id:
            return False
        try:
            doc = await refs.find_one({"project_id": project_id, "run_id": run_id}, {"_id": 1})
            return doc is not None
        except Exception as exc:
            logger.warning("[AGENT] %s archive-ref presence check failed: %s", self.agent_id, exc)
            return False

    async def _attachments_present(self) -> bool:
        """True when this project already has user attachments for its tenant.

        Project-scoped, not run-scoped: files belong to the project, not a run.
        Missing tenant/storage/collection or a query error means do not attach.
        Mid-run upload is not spliced (ponytail: files exist at execute start).
        """
        sc = self.shared_context
        storage = getattr(sc, "storage", None) if sc else None
        project_id = getattr(sc, "project_id", None) if sc else None
        tenant_id = getattr(sc, "tenant_id", None) if sc else None
        return await project_has_attachments(storage, project_id, tenant_id)

    async def _tenant_artifacts_present(self) -> bool:
        """True when this project's tenant already has shared tenant artifacts.

        Tenant-scoped, not project-scoped. Missing tenant/storage/collection or a
        query error means do not attach. Mid-run admin upload is not spliced.
        """
        sc = self.shared_context
        storage = getattr(sc, "storage", None) if sc else None
        tenant_id = getattr(sc, "tenant_id", None) if sc else None
        return await tenant_has_artifacts(storage, tenant_id)

    async def _load_resume_pairs(
        self, task: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Capped attempts require readable journal history before resuming."""
        if not task.get("resume_from_journal") or not self.use_streaming:
            return None
        sc = self.shared_context
        store = getattr(sc, "message_store", None) if sc else None
        run_id = getattr(sc, "run_id", None) if sc else None
        task_id = TaskSchema.get_id(task)
        if not (store and run_id and task_id):
            logger.warning(
                "[REHYDRATE] %s: resume requested but store/run_id/task_id "
                "unavailable — running fresh",
                self.agent_id,
            )
            if task.get("max_tool_failures") is not None:
                raise RuntimeError("Cannot restore tool failure budget without journal access")
            return None
        try:
            pairs = await load_attempt_pairs(
                store, sc.project_id, run_id, task_id,
                complete=task.get("max_tool_failures") is not None,
            )
        except Exception as e:
            if task.get("max_tool_failures") is not None:
                logger.warning(
                    "[REHYDRATE] project_id=%s task_id=%s — failure budget restore failed",
                    sc.project_id, task_id,
                )
                raise RuntimeError("Cannot restore tool failure budget from journal") from e
            logger.warning(
                "[REHYDRATE] %s: journal read failed (%s) — running fresh",
                self.agent_id, e,
            )
            return None
        if pairs is None:
            logger.warning(
                "[REHYDRATE] %s: no journal records for resumed task %s — "
                "running fresh",
                self.agent_id, task_id,
            )
        return pairs

    async def _run_streaming(
        self, task: Dict[str, Any], user_content: str,
        tools: List[Dict], dispatcher: AgentToolDispatcher, effective_system_prompt: str,
        resume_pairs: Optional[List[Dict[str, Any]]] = None,
        plugin_host=None,
        seeder_on: bool = False,
    ) -> Dict[str, Any]:
        _seq = [0]

        async def emit_streaming_event(event: Dict[str, Any]):
            if self.event_emitter:
                event_type = event.get("type", "unknown")
                _seq[0] += 1
                await self.event_emitter.emit(
                    f"agent.streaming.{event_type}",
                    self.shared_context.run_id if self.shared_context else None,
                    {
                        "project_id": self.shared_context.project_id if self.shared_context else None,
                        "agent_id": self.agent_id,
                        "task_id": TaskSchema.get_id(task),
                        "timestamp": f"{time.time()}-{_seq[0]}",
                        **event,
                    },
                )

        input_items = [
            {"type": "message", "role": "system", "content": effective_system_prompt},
            {"type": "message", "role": "user", "content": user_content},
        ]
        # Rolling summary (AppFactory-149 slice 5): folded conversation rides here as a
        # strippable item after the task message — not baked into it — so an in-run
        # compaction fold recognises and replaces it instead of duplicating it.
        # None until something has been folded, or when the store is unreachable.
        # Only when the seeder is on: with it off the valve already put cross-run
        # history into the task message, so a summary item would duplicate it.
        summary_item = await self._rolling_summary_item() if seeder_on else None
        if summary_item:
            input_items.append(summary_item)
        if resume_pairs:
            hanging = [p for p in resume_pairs if p.get("call") and not p.get("result")]
            if hanging:
                # The trigger parks a run while a call hangs, so reaching here
                # means that contract broke upstream. The call is dropped from
                # the transcript (an output-less function_call is a protocol
                # error) and the model may legitimately re-issue it.
                logger.warning(
                    "[REHYDRATE] %s: %d hanging call(s) at seed time — "
                    "excluded from the rebuilt transcript",
                    self.agent_id, len(hanging),
                )
            input_items.extend(build_resume_input_items(resume_pairs))
            logger.info(
                "[REHYDRATE] %s: resumed task %s with %d journaled pair(s)",
                self.agent_id,
                TaskSchema.get_id(task),
                len(resume_pairs) - len(hanging),
            )

        runner = StreamingAgentRunner(
            llm_client=self.llm_client,
            tools=tools,
            tool_handler=dispatcher.handle,
            model=self._resolve_project_model(),
            reasoning_effort=self._resolve_reasoning_effort(self.reasoning_effort),
            temperature=self._resolve_temperature(self.temperature),
            max_tool_rounds=self._resolve_step_limit(self.step_limit),
            event_callback=emit_streaming_event,
            cancellation_token=self.cancellation_token,
            stream_round_timeout_seconds=StreamingAgentRunner.stream_round_timeout_from_env(),
            plugin_host=plugin_host,
            max_tool_failures=task.get("max_tool_failures"),
            **self._capture_kwargs_for_runner(),
        )

        dispatcher.on_archive_ref = self._archive_attach_hook(runner)

        api_key_override = getattr(self.shared_context, "_ephemeral_api_key", None) if self.shared_context else None
        fallback_models_override = getattr(self.shared_context, "_ephemeral_fallback_models", None) if self.shared_context else None
        if self.shared_context and getattr(self.shared_context, "_force_model_override", False):
            fallback_models_override = []
        try:
            if task.get("max_tool_failures") is not None:
                runner.tool_failure_budget.restore(resume_pairs or [])
                runner.tool_failure_budget.check()
            rr = await self.await_with_cancellation(
                runner.run(
                    input_items,
                    api_key_override=api_key_override,
                    fallback_models_override=fallback_models_override,
                ),
                "streaming_runner",
            )
        except _RUNNER_STOPS as exc:
            return self._runner_stop_result(task, exc)

        def gate(rr):
            return self._gated_final(
                rr,
                tool_events_from_streaming_runner(rr),
                f"Completed via StreamingAgentRunner ({len(rr.thinking_blocks)} thinking, {len(rr.tool_calls)} tools)",
            )

        repair = self._repair_turns(
            task,
            runner,
            rr,
            gate,
            "streaming_runner",
            api_key_override=api_key_override,
            fallback_models_override=fallback_models_override,
        )
        return await self._finish_run(task, gate(rr), repair)

    def _gated_final(self, rr, tool_events: list, reasoning: str) -> tuple:
        final = (rr.content or "").strip()
        parsed = self._try_parse_json(final) if final else None
        result = apply_terminal_gate(
            final_text=final,
            parsed_output=parsed,
            stop_reason=getattr(rr, "stop_reason", None),
            provider_error=(getattr(rr, "error", None) or "").strip() or None,
            tool_events=tool_events,
            reasoning=reasoning,
            final_envelope=True,
        )
        return final, parsed, result

    def _repair_turns(self, task, runner, rr, gate, location, **run_kwargs):
        """Each call continues the runner session with one user message and
        returns the gated (final, parsed, result) of that turn."""
        session = {"previous": rr}

        async def turn(message: str) -> tuple:
            try:
                session["previous"] = await self.await_with_cancellation(
                    runner.continue_run(session["previous"], message, **run_kwargs),
                    location,
                )
            except _RUNNER_STOPS as exc:
                return "", None, self._runner_stop_result(task, exc)
            return gate(session["previous"])

        return turn

    async def _finish_run(self, task: dict, gated: tuple, repair) -> dict:
        final, parsed, result = gated
        exact = getattr(self, "_exact_tool_runtime", None) is not None
        if not exact:
            await self._save_conversation_message(task, final)
        if ResultSchema.is_completed(result):
            return await self._completed_output(task, final, parsed, result, repair)
        if exact:
            await self._save_conversation_message(task, final)
        return result

    async def _completed_output(
        self, task: dict, final: str, parsed: Any, result: dict, repair=None
    ) -> dict:
        runtime = getattr(self, "_exact_tool_runtime", None)
        exact = runtime is not None
        budget = output_repair_budget(task) if exact and repair else 0
        attempt = 0
        failure = None
        while True:
            output = parsed if parsed is not None else {"final_output": final}
            try:
                output = await self._maybe_save_output(output, task)
                break
            except ValueReferenceError as exc:
                if attempt >= budget or not output_repairable(exc):
                    failure = self._output_reference_failure(task, exc)
                    break
                attempt += 1
                logger.warning(
                    "[SAVE_OUTPUT] project_id=%s agent=%s task_id=%s code=%s path=%s "
                    "attempt=%s/%s — output rejected, asking the model to repair it",
                    task.get("project_id"),
                    self.agent_id,
                    TaskSchema.get_id(task),
                    exc.code,
                    exc.path,
                    attempt,
                    budget,
                )
                await announce_output_repair(self, task, exc, attempt)
                message = await output_rejection_message(runtime, exc)
                final, parsed, result = await repair(message)
                if not ResultSchema.is_completed(result):
                    failure = result
                    break
        if exact:
            # Chat cards read this message, so it carries the resolved output
            # rather than the references the model wrote.
            if failure is None:
                final = json.dumps(output, ensure_ascii=False)
            await self._save_conversation_message(task, final)
        if failure is not None:
            return failure
        return {**result, ResultSchema.OUTPUT: {"final_output": final}}

    def _output_reference_failure(self, task: dict, exc: ValueReferenceError) -> dict:
        logger.warning(
            "[SAVE_OUTPUT] project_id=%s agent=%s task_id=%s code=%s path=%s — output rejected",
            task.get("project_id"), self.agent_id, TaskSchema.get_id(task),
            exc.code, exc.path,
        )
        return ResultSchema.create(
            status=TaskStatus.FAILED,
            output=None,
            reasoning=f"Final output rejected: {exc.code} at {exc.path}",
            error=f"{exc.code} at {exc.path}",
            error_type="output_reference_invalid",
        )

    def _runner_stop_result(self, task: dict, exc: Exception) -> dict:
        if isinstance(exc, ToolFailureLimitExceeded):
            return self._tool_failure_result(task, exc)
        if isinstance(exc, SessionUnavailableStopError):
            reason, error_type = SESSION_UNAVAILABLE_STOP, "session_unavailable"
        else:
            reason, error_type = (
                EXTERNAL_TOOL_OUTCOME_UNKNOWN,
                "external_tool_outcome_unknown",
            )
        return ResultSchema.create(
            status=TaskStatus.FAILED,
            output=None,
            reasoning=reason,
            error=reason,
            error_type=error_type,
            tool_name=exc.tool_name,
            tool_call_id=exc.call_id,
        )

    def _tool_failure_result(self, task: dict, exc: ToolFailureLimitExceeded) -> dict:
        logger.warning(
            "[EXECUTE] project_id=%s agent=%s task_id=%s tool=%s limit=%s — stopping phase",
            task.get("project_id"), self.agent_id, TaskSchema.get_id(task),
            exc.tool_name, exc.limit,
        )
        return ResultSchema.create(
            status=TaskStatus.FAILED,
            output=None,
            reasoning=str(exc),
            error=str(exc),
            error_type="tool_failure_limit_exhausted",
            tool_name=exc.tool_name,
            max_tool_failures=exc.limit,
        )

    def _archive_tool_schemas(self) -> list:
        """Archive schemas for this agent: fetch only when bash/read allowed."""
        return archive_tool_schemas(
            include_fetch=agent_can_archive_fetch(self._effective_allowed_tools),
        )

    def _archive_attach_hook(self, runner):
        """First spill of the run: make the retrieval tools callable from the
        NEXT round (both runners re-read their ``tools`` list per request).
        Idempotent — later spills find the schemas already spliced. Wired on
        BOTH runner paths: the placeholder names these tools, so whichever
        runner produced the spill must be able to call them. Fetch is gated
        the same way as the start-of-round splice (AppFactory-315)."""
        def _attach() -> None:
            present = {
                t.get("function", {}).get("name")
                for t in runner.tools
                if isinstance(t, dict)
            }
            if present.isdisjoint(ARCHIVE_TOOL_IDS):
                runner.tools.extend(self._archive_tool_schemas())

        return _attach

    async def _run_simple_runner(
        self, task: Dict[str, Any], user_content: str,
        tools: List[Dict], dispatcher: AgentToolDispatcher, effective_system_prompt: str,
    ) -> Dict[str, Any]:
        runner = SimpleAgentRunner(
            llm_client=self.llm_client,
            tools=tools,
            tool_handler=dispatcher.handle,
            model=self._resolve_project_model(),
            reasoning_effort=self._resolve_reasoning_effort(self.reasoning_effort),
            temperature=self._resolve_temperature(self.temperature),
            max_steps=self._resolve_step_limit(self.step_limit),
            cancellation_token=self.cancellation_token,
            max_tool_failures=task.get("max_tool_failures"),
            **self._capture_kwargs_for_runner(),
        )
        dispatcher.on_archive_ref = self._archive_attach_hook(runner)

        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            rr = await self.await_with_cancellation(runner.run(messages), "simple_runner")
        except _RUNNER_STOPS as exc:
            return self._runner_stop_result(task, exc)

        def gate(rr):
            return self._gated_final(
                rr,
                tool_events_from_simple_runner(rr),
                "Completed via SimpleAgentRunner",
            )

        repair = self._repair_turns(task, runner, rr, gate, "simple_runner")
        return await self._finish_run(task, gate(rr), repair)

    # Post-run follow-up chat tools that mutate the sandbox / repo. When one
    # runs, the caller must reopen the output approval — the chat changed the
    # deliverable, same as an ad-hoc coding task.
    FOLLOWUP_WRITE_TOOLS = frozenset(
        {"create", "edit", "bash", "deploy_from_artifacts", "analyze_and_repair"}
    )

    async def answer_followup(
        self,
        messages: List[Dict[str, Any]],
        *,
        allowed_tool_ids: Optional[List[str]] = None,
        max_steps: int = 12,
        tool_argument_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Explicit constraints bind both advertised schemas and execution;
        omitting them preserves ordinary project follow-up permissions.
        """
        self._exact_tool_runtime = None
        allow = (
            allowed_tool_ids
            if allowed_tool_ids is not None
            else self._effective_allowed_tools
        )
        tools = await self._fetch_registry_tool_schemas(allow)
        tools = await self._splice_conditional_tools(tools)
        if tool_argument_constraints is not None:
            tools = bind_tool_constraints(tools, tool_argument_constraints, allow)
        if self.config.get("exact_tool_values") is True and tools:
            from agents.exact_tool_runtime import ExactToolRuntime
            from tools.value_reference_schema import REFERENCE_INSTRUCTIONS

            self._exact_tool_runtime = ExactToolRuntime(self, tools)
            tools = self._exact_tool_runtime.tools
            guidance = REFERENCE_INSTRUCTIONS
            if tool_argument_constraints:
                selected = await self._exact_tool_runtime.project_context(tool_argument_constraints)
                guidance += "\nSaved tool arguments: " + json.dumps(selected, ensure_ascii=False)
            messages = [*messages, {"role": "system", "content": guidance}]
        allowed_names = {tool["function"]["name"] for tool in tools}
        if not tools:
            return {"content": "", "tools_used": [], "wrote": False, "had_tools": False, "tool_results": []}

        dispatcher = AgentToolDispatcher(self)
        used: List[str] = []
        results: List[Dict[str, Any]] = []
        base_handler = dispatcher.handle

        async def _recording_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            scoped_payload = payload
            if tool_argument_constraints is not None and self._exact_tool_runtime is not None:
                try:
                    resolved = await self._exact_tool_runtime.prepare(payload.get("tool"), payload.get("args", {}))
                    scoped_payload = {**payload, "args": resolved}
                except ValueReferenceError as error:
                    return {"status": "error", "error": {"code": error.code, "path": error.path}}
            if tool_argument_constraints is not None and not call_matches_scope(
                scoped_payload, tool_argument_constraints, allowed_names
            ):
                logger.warning(
                    "[POSTRUN_CHAT] agent=%s — tool call rejected by collection scope",
                    self.agent_id,
                )
                return {
                    "status": "error",
                    "error": {
                        "code": "FOLLOWUP_SCOPE_VIOLATION",
                        "message": "Use only the saved collection and its bound tool arguments.",
                    },
                }
            name = payload.get("tool") if isinstance(payload, dict) else None
            if name:
                used.append(name)
            result = await base_handler(payload)
            if tool_argument_constraints is not None:
                recorded_result = result
                skip_card_item = False
                is_query = name == "query_collection" or (
                    isinstance(name, str) and name.endswith("_query_collection")
                )
                if self._exact_tool_runtime is not None and is_query:
                    bounded = deepcopy(result)
                    if isinstance(bounded, dict):
                        body = bounded.get("data") if isinstance(bounded.get("data"), dict) else bounded
                        if isinstance(body, dict):
                            rows = body.get("rows")
                            if isinstance(rows, list):
                                body.setdefault("row_count", len(rows))
                                body["rows"] = rows[:MAX_CARD_ROWS]
                            refs = body.get("source_refs")
                            if isinstance(refs, list):
                                body["source_refs"] = refs[:MAX_CARD_ROWS]
                    try:
                        recorded_result = await self._exact_tool_runtime.references.resolve(bounded)
                    except ValueReferenceError:
                        logger.warning(
                            "[POSTRUN_CHAT] agent=%s tool=%s — query result not resolvable "
                            "for the card, skipped",
                            self.agent_id,
                            name,
                        )
                        skip_card_item = True
                if not skip_card_item:
                    results.append({
                        "tool": name,
                        "sql": (scoped_payload.get("args") or {}).get("sql"),
                        "result": recorded_result,
                    })
            return result

        runner = SimpleAgentRunner(
            llm_client=self.llm_client,
            tools=tools,
            tool_handler=_recording_handler,
            model=self._resolve_project_model(subsystem="question_handler"),
            reasoning_effort=self._resolve_reasoning_effort(self.reasoning_effort),
            temperature=self._resolve_temperature(self.temperature),
            max_steps=max_steps,
            cancellation_token=self.cancellation_token,
            **self._capture_kwargs_for_runner(),
        )
        if tool_argument_constraints is None:
            dispatcher.on_archive_ref = self._archive_attach_hook(runner)
        content = ""
        try:
            rr = await runner.run(messages)
            content = (rr.content or "").strip()
            if self._exact_tool_runtime is not None:
                from agents.exact_value_context import materialize_reference_text

                content = await materialize_reference_text(
                    self._exact_tool_runtime.store, content
                )
        except SessionUnavailableStopError:
            content = (
                "The sandbox session is unavailable — restore the environment "
                "(/recover) then resume the project."
            )
        except ExternalToolOutcomeUnknownError:
            # A write tool may have already mutated the sandbox before an
            # external call's outcome went unknown (`used` is recorded before
            # each call), so still report the write, as the task path does.
            content = (
                "I attempted changes but a tool's outcome could not be "
                "confirmed — please review the updated output."
            )
        return {
            "content": content,
            "tools_used": used,
            "wrote": any(t in self.FOLLOWUP_WRITE_TOOLS for t in used),
            "had_tools": True,
            "tool_results": results,
        }

    # ------------------------------------------------------------------
    # Simple execution (no tools) — prompt → LLM → structured output
    # ------------------------------------------------------------------

    async def _execute_simple(self, task: Dict[str, Any], description: str) -> Dict[str, Any]:
        self.ensure_not_cancelled("execute_simple")
        # Tool-less agents run here — one call_llm, no streaming item channel. Fold
        # the conversation as the streaming path does, but deliver the summary
        # inline: with no item to seed, prepend it into the flat string, else the
        # valve just drops the early project context (ADR-0014 Amendment).
        compaction_cfg = await self._resolve_compaction_config()
        seeder_on = plugin_flag(compaction_cfg, "summarize_overflow_conversation", default=True)
        task_context = await self._build_task_context_async(
            task,
            fold_conversation=seeder_on,
            summary_model=compaction_cfg.get("summary_model") or None,
            keep_recent_conversation_tokens=compaction_cfg.get("keep_recent_conversation_tokens"),
        )
        if seeder_on:
            summary_item = await self._rolling_summary_item()
            if summary_item:
                task_context = f"{summary_item['content']}\n\n{task_context}"

        user_content = task_context

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.call_llm(
                messages,
                reasoning_effort=self._resolve_reasoning_effort(self.reasoning_effort),
                return_details=True,
            )
            if isinstance(response, dict):
                content = response.get("content") or ""
                stop_reason = response.get("finish_reason") or response.get("stop_reason")
                provider_err = (response.get("error") or "").strip() or None
            else:
                content = response if isinstance(response, str) else ""
                stop_reason = None
                provider_err = None
            output = self._try_parse_json(content)
        except Exception as e:
            logger.error("[GenericAgent] %s execute_task failed: %s", self.agent_id, e)
            return ResultSchema.create(
                status=TaskStatus.FAILED,
                output=None,
                reasoning=f"LLM call failed: {e}",
            )

        await self._save_conversation_message(
            task, content if isinstance(content, str) else json.dumps(output)
        )
        result = apply_terminal_gate(
            final_text=content if isinstance(content, str) else "",
            parsed_output=output,
            stop_reason=stop_reason,
            provider_error=provider_err,
            tool_events=[],
            reasoning=f"Completed by {self.display_name}",
            final_envelope=False,
        )
        if ResultSchema.is_completed(result):
            await self._maybe_save_output(output, task)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _save_conversation_message(self, task: Dict[str, Any], content: str):
        if not self.shared_context:
            return
        if TaskSchema.should_suppress_assistant_message(task):
            # Gated phase: the assistant-final is suppressed (the approval card is
            # the single chat view). The durable completion marker is written by
            # the ENGINE after the writes-contract is satisfied
            # (WorkflowEngine._mark_gated_phase_completed) — NOT here at turn-end,
            # where a contract-failed or interrupted attempt would wrongly look
            # done to restart recovery and gate a phase whose outputs never
            # landed (F2).
            logger.warning(
                "[CHAT_OUTPUT] agent=%s task_id=%s — raw assistant message suppressed; "
                "persistent agent-result card is the single chat representation",
                self.agent_id,
                TaskSchema.get_id(task),
            )
            return
        if not content:
            return
        self.ensure_not_cancelled("save_conversation_message")
        try:
            phase = "unknown"
            if self.allowed_phases and isinstance(self.allowed_phases, list) and len(self.allowed_phases) > 0:
                phase = self.allowed_phases[0]
            metadata = {"agent_id": self.agent_id, "task_id": TaskSchema.get_id(task)}
            if TaskSchema.get_type(task):
                metadata["task_type"] = TaskSchema.get_type(task)
            metadata.update(TaskSchema.get_workflow_stage(task))
            await self.shared_context.add_conversation_message(
                role="assistant",
                content=content if isinstance(content, str) else str(content),
                phase=phase,
                metadata=metadata,
            )
        except Exception:
            pass

    async def _maybe_save_output(
        self,
        output: Any,
        task: Optional[Dict[str, Any]] = None,
    ):
        """Save output using node writes first, then legacy output_save_key."""
        self.ensure_not_cancelled("save_output")

        exact_runtime = getattr(self, "_exact_tool_runtime", None)
        if exact_runtime is not None:
            output = await exact_runtime.resolve_output(output)

        save_keys, mode = self._resolve_output_save_keys(task)
        if not save_keys:
            logger.info("[SAVE_OUTPUT] agent=%s save_key=None mode=%s - skipping save", self.agent_id, mode)
            return output
        if not self.shared_context or not output:
            logger.warning(
                "[SAVE_OUTPUT] agent=%s keys=%s mode=%s shared_context=%s output=%s - skipping save",
                self.agent_id,
                save_keys,
                mode,
                bool(self.shared_context),
                bool(output),
            )
            return output

        logger.info("[SAVE_OUTPUT] agent=%s saving to keys=%s mode=%s", self.agent_id, save_keys, mode)
        try:
            if mode == "contract":
                saved_keys = await self._save_contract_outputs(save_keys, output)
            else:
                legacy_key = save_keys[0]
                await self._save_legacy_output(legacy_key, output)
                saved_keys = [legacy_key]
            logger.info(
                "[SAVE_OUTPUT] agent=%s keys=%s mode=%s - saved successfully",
                self.agent_id,
                saved_keys,
                mode,
            )
        except Exception as e:
            logger.error(
                "[SAVE_OUTPUT] agent=%s keys=%s mode=%s - FAILED: %s",
                self.agent_id,
                save_keys,
                mode,
                e,
            )
        return output

    def _resolve_output_save_keys(self, task: Optional[Dict[str, Any]]) -> tuple[List[str], str]:
        """Resolve direct-save keys for this task."""
        if task is not None and TaskSchema.has_writes(task):
            writes = [
                str(key or "").strip()
                for key in (TaskSchema.get_writes(task) or [])
                if str(key or "").strip()
            ]
            if not writes:
                return [], "contract"
            return writes, "contract"

        return ([self.output_save_key] if self.output_save_key else []), "legacy"

    async def _save_contract_outputs(
        self, keys: List[str], output: Any
    ) -> List[str]:
        """Save contract outputs through the SharedContext logical write path.

        The agent's final output is the source of truth: a single declared key
        receives the whole output; a multi-key contract is split from a dict by key.
        """
        if not hasattr(self.shared_context, "write_context_key"):
            raise RuntimeError("shared_context does not support contract writes")

        if len(keys) == 1:
            await self.shared_context.write_context_key(keys[0], output)
            return keys

        if not isinstance(output, dict):
            logger.warning(
                "[SAVE_OUTPUT] agent=%s writes=%s output_type=%s - cannot split multi-key output",
                self.agent_id,
                keys,
                type(output).__name__,
            )
            return []

        saved_keys = []
        for key in keys:
            if key not in output:
                logger.warning(
                    "[SAVE_OUTPUT] agent=%s key='%s' missing in structured output - skipping",
                    self.agent_id,
                    key,
                )
                continue
            await self.shared_context.write_context_key(key, output[key])
            saved_keys.append(key)
        return saved_keys

    async def _save_legacy_output(self, key: str, output: Any) -> None:
        """Preserve legacy output_save_key behavior for workflows without writes."""
        if key == "requirements":
            if hasattr(self.shared_context, "update_requirements"):
                await self.shared_context.update_requirements(output if isinstance(output, dict) else {"raw": output})
            else:
                await self.shared_context.add(key, output, category="requirements")
        elif key == "plan":
            if hasattr(self.shared_context, "update_plan"):
                await self.shared_context.update_plan(output if isinstance(output, dict) else {"raw": output})
            else:
                await self.shared_context.add(key, output, category="planning")
        else:
            await self.shared_context.add(key, output, category="general")

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        """Try to parse LLM response as JSON, fall back to raw string."""
        if not isinstance(text, str):
            return text
        cleaned = text.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.strip().endswith("```"):
                cleaned = cleaned.strip()[:-3]
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {"raw_output": text}

    async def _resolve_delegation_targets(self) -> List[Dict[str, str]]:
        """Pool agents + explicitly-allowed a2a servers, as one delegation catalog.

        Both kinds flow through the same enum + prompt block so the LLM sees an external
        a2a server beside pool agents; DelegationManager routes each by re-resolving the
        chosen name. a2a resolution reads storage, hence async — computed once here on the
        execute path and passed down to both the schema binder and the prompt builder.
        """
        targets = list(resolve_project_delegation_targets(self))
        targets.extend(await resolve_a2a_delegation_targets(self))
        return targets

    def _filter_tool_schemas_for_task(
        self,
        tools: List[Dict[str, Any]],
        task: Dict[str, Any],
        delegation_targets: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Hide delegation unless both agent config and the current task allow it.

        Allow-list intersection already happened in ``get_schemas_for_agent`` /
        ``get_tool_schemas_for_agent`` (via ``select_effective_tool_docs_for_allow``,
        which matches public id, storage id, and ``llm_function_name``). Re-filtering
        by ``function.name`` here dropped MCP tools whose wire name differs from the
        allow-list ref and caused delegated specialists to fail with no tools.
        """
        can_delegate = self._can_show_delegation_tool(task)
        filtered = [
            tool
            for tool in tools
            if can_delegate
            or self._tool_schema_name(tool) != DELEGATE_TO_AGENT_TOOL
        ]
        if not can_delegate:
            return filtered
        if delegation_targets is None:
            # POOL-ONLY fallback: this sync path can't await the async a2a resolver, so a2a
            # targets are absent here. The production caller (execute_task) always passes a
            # precomputed pool+a2a list via _resolve_delegation_targets, so this branch never
            # runs for real; a future caller relying on the None default would silently drop
            # a2a targets from the enum — pass an explicit list instead.
            if self.agent_pool is None:
                return filtered
            delegation_targets = resolve_project_delegation_targets(self)
        return bind_delegation_targets_to_tool_schemas(filtered, delegation_targets)

    def _allowed_tool_ids_for_task(self, task: Dict[str, Any]) -> List[str]:
        allowed_tool_ids = list(self._effective_allowed_tools or [])
        if self._can_show_delegation_tool(task):
            return allowed_tool_ids
        return [
            tool_id
            for tool_id in allowed_tool_ids
            if tool_id != DELEGATE_TO_AGENT_TOOL
        ]

    def _can_show_delegation_tool(self, task: Dict[str, Any]) -> bool:
        return delegation_activation_error(self, task) is None

    @staticmethod
    def _tool_schema_name(tool: Dict[str, Any]) -> Optional[str]:
        if not isinstance(tool, dict):
            return None
        function = tool.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return str(name) if name else None

    async def _build_effective_system_prompt(
        self,
        task: Optional[Dict[str, Any]] = None,
        delegation_targets: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build final prompt with dynamic allowed-tools catalog block."""
        storage = getattr(self.shared_context, "storage", None) if self.shared_context else None
        tenant_id = self.shared_context.tenant_id if self.shared_context else None
        registry_tools = self.tool_registry.tools if self.tool_registry else None
        can_delegate = self._can_show_delegation_tool(task or {})
        if delegation_targets is None and can_delegate:
            # POOL-ONLY fallback (a2a targets absent): same caveat as _filter_tool_schemas_for_task
            # — the async a2a resolver can't be awaited here, so callers must pass an explicit
            # pool+a2a list (execute_task does) to keep a2a servers in the prompt block.
            delegation_targets = resolve_project_delegation_targets(self)
        allowed_tool_ids = self._allowed_tool_ids_for_task(task or {})
        if can_delegate and not delegation_targets:
            allowed_tool_ids = [
                tool_id
                for tool_id in allowed_tool_ids
                if tool_id != DELEGATE_TO_AGENT_TOOL
            ]
        exact_runtime = getattr(self, "_exact_tool_runtime", None)
        if exact_runtime is not None:
            catalog_text = "\n".join(
                f"- {tool['function']['name']}: {tool['function'].get('description', '')}"
                for tool in exact_runtime.tools
            )
        else:
            catalog_text = await build_tool_catalog_text(
                allowed_tool_ids=allowed_tool_ids,
                storage=storage,
                tenant_id=tenant_id,
                registry_tools=registry_tools,
                agent_context=self.agent_id,
                agent_id=self.agent_id,
            )
        if ALLOWED_TOOLS_CATALOG_PLACEHOLDER in self.system_prompt:
            prompt = apply_tool_catalog_placeholder(self.system_prompt, catalog_text)
        elif not catalog_text:
            prompt = self.system_prompt
        else:
            prompt = (
                f"{self.system_prompt}\n\n"
                f"{TOOL_CATALOG_INTRO_LINE}\n"
                f"{catalog_text}"
            )

        delegation_block = format_delegation_targets_prompt(delegation_targets or [])
        if delegation_block:
            prompt = f"{prompt}\n\n{delegation_block}"
        return prompt


# ------------------------------------------------------------------
# Helper: resolve AgentType from string
# ------------------------------------------------------------------

def _resolve_agent_type(type_str: str) -> AgentType:
    """Map config type string to AgentType enum, with fallback."""
    type_map = {v.value: v for v in AgentType}
    if type_str in type_map:
        return type_map[type_str]
    # Fallback: try case-insensitive
    for val, member in type_map.items():
        if val.lower() == type_str.lower():
            return member
    # Default to GENERIC if unknown
    logger.warning("[GenericAgent] Unknown agent type '%s', defaulting to GENERIC", type_str)
    return AgentType.GENERIC
