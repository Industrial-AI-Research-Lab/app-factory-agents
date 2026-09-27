"""Adapter: run a delegated sub-task on an external a2a agent (AppFactory-182, MVP).

Happy-path port of WorkflowEngine._run_a2a_agent_node, minus the DAG machinery — no
node id, no reads/writes contract, no SharedContext keys — because delegation is invoked
from an agent's reasoning, not from a graph node. Reuses the shared a2a client, the
card-driven parameter extraction, and the artifact helpers unchanged; it only sends,
waits for the terminal task, persists artifacts, and maps the reply into a delegation
tool result.

Not this task (AppFactory-181, on the ledger): the input_required dialog, approval/HITL,
and resilience. input_required here terminates with a distinct typed error rather than
hanging or 500ing — the MVP contract.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from integrations import A2AClientFactory
from integrations.a2a_artifacts import (
    a2a_artifact_scope,
    a2a_artifact_to_files,
    a2a_response_text,
    dedupe_a2a_path,
    merge_a2a_artifact_update,
)
from integrations.a2a_params import A2AParamExtractionError, extract_extension_dataparts
from integrations.a2a_output_modes import A2AOutputModeError
from integrations.a2a_contracts import A2AContractError
from storage.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

TARGET_KIND_A2A = "a2a"

# Terminal A2A task states -> (error_type, message) for the non-completed outcomes the
# MVP must end predictably instead of hanging. input_required is a first-class case:
# the static node can't answer a follow-up and neither can this path yet, so it fails
# with its own type so the orchestrator can branch on it.
_NON_COMPLETED_STATES = {
    "TASK_STATE_INPUT_REQUIRED": (
        "a2a_input_required",
        "A2A agent requires additional input (input_required)",
    ),
    "TASK_STATE_FAILED": ("a2a_failed", "A2A task explicitly failed"),
    "TASK_STATE_CANCELED": ("a2a_canceled", "A2A task was cancelled"),
}


def _error(server_name: str, error_type: str, error: str) -> Dict[str, Any]:
    """Uniform a2a-delegation failure result (status is stripped by the runner, so the
    error rides dedicated top-level keys the orchestrator will still see)."""
    return {
        "status": "error",
        "target": server_name,
        "target_kind": TARGET_KIND_A2A,
        "error_type": error_type,
        "error": error,
    }


async def delegate_to_a2a_agent(
    server: Dict[str, Any],
    instruction: str,
    *,
    storage: Any,
    llm_client: Any,
    model: Optional[str] = None,
    api_key_override: Optional[str] = None,
    fallback_models_override: Optional[List[str]] = None,
    tenant_id: str,
    project_id: str,
    run_id: Optional[str] = None,
    delegation_id: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    message_store: Any = None,
) -> Dict[str, Any]:
    """Delegate ``instruction`` to the external a2a ``server`` and return a tool result.

    ``server`` is the resolved a2a server document (``_id``, ``name``,
    ``cached_agent_card_summary``). ``delegation_id`` scopes saved artifacts so two
    delegations in one run don't clobber each other's files. When ``artifact_store`` is
    not supplied, one is built as ``ArtifactStore(storage, message_store=message_store)`` —
    ``message_store`` is mandatory for ``save_file`` to claim a checkpoint slot, so without
    it every artifact would silently fail to persist. Terminal COMPLETED returns
    ``{status: success, target, target_kind, result, artifacts}`` where ``artifacts`` lists
    only the paths that actually persisted (never fabricated); every other terminal state
    (incl. input_required) returns a typed error.
    """
    server_id = server.get("_id")
    server_name = str(server.get("name") or server_id or "a2a")
    summary = server.get("cached_agent_card_summary") or {}

    if not server_id:
        return _error(server_name, "a2a_configuration_error", "a2a server has no id")

    # Build the request: one text part carrying the instruction, plus any DataPart the
    # agent card REQUIRES (e.g. scenario_id), extracted from the instruction by a
    # schema-constrained LLM call. A required param we can't extract fails HERE, before
    # contacting the agent, so the reason is the missing param rather than a downstream
    # rejection. Non-extension agents make no LLM call and get just the text part.
    base_parts: List[Dict[str, Any]] = [{"kind": "text", "text": instruction}]
    try:
        extension_parts = await extract_extension_dataparts(
            instruction,
            summary.get("capabilities"),
            llm_client,
            model=model,
            api_key_override=api_key_override,
            fallback_models_override=fallback_models_override,
        )
    except A2AParamExtractionError as exc:
        logger.info("[A2A_DELEGATE] target=%s param_extraction_failed: %s", server_name, exc)
        return _error(server_name, "a2a_param_extraction_failed", str(exc))
    except Exception as exc:
        # A transient extraction-LLM error is not an A2AParamExtractionError; fail cleanly
        # with its own type instead of letting it escape as an opaque 500.
        logger.exception("[A2A_DELEGATE] target=%s param_extraction_error", server_name)
        return _error(server_name, "a2a_param_extraction_error", str(exc))

    message = base_parts + extension_parts if extension_parts else base_parts

    client = A2AClientFactory.get_client(storage)
    final_task: Optional[Dict[str, Any]] = None
    final_artifacts: List[Dict[str, Any]] = []
    last_state: Optional[str] = None
    is_final = False
    message_reply_parts: Optional[List[Dict[str, Any]]] = None

    try:
        async for event in client.send_message(
            server_id=server_id,
            tenant_id=tenant_id,
            message=message,
            context_id=run_id,
            metadata={"project_id": project_id, "delegation": True},
        ):
            event_type = event.get("type")
            event_data = event.get("data", {}) or {}

            if event_type == "task":
                final_task = event_data
                last_state = (event_data.get("status") or {}).get("state")
                if event_data.get("artifacts"):
                    final_artifacts = event_data.get("artifacts", [])
            elif event_type == "status_update":
                state = event_data.get("state")
                if state:
                    last_state = state
                if event_data.get("final"):
                    is_final = True
                artifacts = event_data.get("artifacts") or []
                if artifacts:
                    final_artifacts.extend(artifacts)
            elif event_type == "artifact_update":
                artifact = event_data.get("artifact", event_data)
                if artifact:
                    final_artifacts = merge_a2a_artifact_update(final_artifacts, artifact)
            elif event_type == "message":
                # Non-streaming send_message may return a bare Message instead of a Task.
                # Its parts carry the reply; presence signals completion (no Task state).
                if event_data.get("parts"):
                    message_reply_parts = event_data["parts"]
    except (A2AOutputModeError, A2AContractError) as exc:
        logger.error(
            "[A2A_CONTRACT] target=%s error_type=%s location=%s — %s",
            server_name,
            exc.error_type,
            exc.location,
            exc,
        )
        return _error(server_name, exc.error_type, str(exc))
    except Exception as exc:
        logger.exception("[A2A_DELEGATE] target=%s send failed", server_name)
        return _error(server_name, "a2a_send_failed", str(exc))

    # A bare Message reply counts as completion ONLY when no Task state was seen; a
    # terminal FAILED/CANCELED task after a message frame must still decide the outcome.
    message_only_completion = message_reply_parts is not None and last_state is None
    # is_final is gated on state: a final status_update carries a terminal state that may be
    # failed/input_required, so treating ANY final frame as "completed" would mislabel those
    # as success. (Today _serialize_stream_response drops the final flag so is_final never
    # fires; this keeps the contract correct if that lossy path is later fixed, instead of
    # relying on a neighboring serializer's data loss.)
    completed = (
        last_state == "TASK_STATE_COMPLETED"
        or (is_final and last_state in (None, "TASK_STATE_COMPLETED"))
        or message_only_completion
    )

    if not completed:
        error_type, error_msg = _NON_COMPLETED_STATES.get(
            last_state or "",
            ("a2a_unexpected_state", f"A2A task ended with unexpected state: {last_state}"),
        )
        # status.message may be absent OR present-but-null; guard both before .get.
        status_message = ((final_task or {}).get("status") or {}).get("message") or {}
        reason = a2a_response_text([{"parts": status_message.get("parts") or []}])
        if reason:
            error_msg = f"{error_msg}: {reason}"
        logger.info("[A2A_DELEGATE] target=%s not_completed state=%s", server_name, last_state)
        return _error(server_name, error_type, error_msg)

    # A data-only Message reply still carries output in its parts; persist them as one
    # artifact so they reach the Artifacts tab instead of vanishing. Skipped when a Task
    # already returned artifacts (those are authoritative).
    if not final_artifacts and message_reply_parts:
        final_artifacts = [{"name": f"{server_name}-response", "parts": message_reply_parts}]

    scope = a2a_artifact_scope(run_id, delegation_id or server_id)
    saved_paths: List[str] = []
    if final_artifacts:
        try:
            store = artifact_store
            if store is None:
                # message_store is REQUIRED by save_file to claim a checkpoint slot; a store
                # built without it raises on every write. Building it here (not deferring to
                # a message_store-less default) is what makes the artifacts actually persist.
                store = ArtifactStore(storage, message_store=message_store)
                await store.initialize()
            used_paths: set = set()
            for artifact in final_artifacts:
                for path, content in a2a_artifact_to_files(artifact, scope):
                    path = dedupe_a2a_path(path, used_paths)
                    try:
                        await store.save_file(project_id, path, content, run_id)
                        saved_paths.append(path)
                    except Exception as exc:
                        logger.warning("[A2A_DELEGATE] failed to save artifact '%s': %s", path, exc)
        except Exception as exc:
            # Store construction/init failure (e.g. DB down, missing message_store) must not
            # crash the delegation nor fabricate paths — degrade to zero saved artifacts.
            logger.warning("[A2A_DELEGATE] artifact store unavailable, nothing persisted: %s", exc)

    response_text = a2a_response_text(final_artifacts)
    if not response_text:
        response_text = (
            f"A2A agent '{server_name}' completed with {len(final_artifacts)} artifact(s)."
        )

    logger.info(
        "[A2A_DELEGATE] target=%s completed artifacts=%d saved=%d state=%s",
        server_name, len(final_artifacts), len(saved_paths), last_state,
    )
    return {
        "status": "success",
        "target": server_name,
        "target_kind": TARGET_KIND_A2A,
        "result": response_text,
        # Only paths that actually persisted — never fabricated, so the orchestrator
        # never cites a file the Artifacts tab doesn't have.
        "artifacts": saved_paths,
        "final_state": last_state,
    }
