"""Register immutable platform invocations before checkpoint-enabled A2A work."""

from copy import deepcopy
import logging
import os

from integrations.a2a_client import A2AClientFactory
from integrations.checkpoint_client import CheckpointClient, CheckpointError
from schemas.checkpoints import A2ACheckpointConfig
from storage.checkpoint_store import CheckpointStore, CheckpointConflict

logger = logging.getLogger(__name__)
RUNTIME_FIELDS = (
    "workflow_id",
    "approval_mode",
    "run_config",
    "run_config_id",
    "model_id",
    "model_override",
    "force_model",
    "reasoning_effort",
    "temperature",
)
CONTEXT_FIELDS = (
    "user_prompt",
    "requirements",
    "plan",
    "custom_context",
    "run_config",
    "project_attachments",
    "tenant_artifacts",
)


def enabled(config):
    return (
        isinstance(config, dict)
        and (config.get("checkpoints") or {}).get("enabled") is True
    )


def checked_client(config):
    settings = A2ACheckpointConfig.model_validate(config.get("checkpoints") or {})
    nonce = os.getenv(settings.callback_nonce_env or "", "")
    if (
        not settings.enabled
        or len(nonce) < 32
        or not nonce.isascii()
        or not all(c.isalnum() or c in "-_" for c in nonce)
    ):
        raise CheckpointError("checkpoint_configuration_invalid", 409, False)
    return CheckpointClient(config)


async def resolve_client(storage, binding):
    config = await A2AClientFactory.get_client(storage).get_server_config(
        binding["server_id"], binding["tenant_id"]
    )
    return checked_client(config)


async def register_invocation(
    orch, project, node, workflow_def, message, config, *, traceparent=None
):
    restored = project.get("_checkpoint_binding")
    if (
        restored
        and restored.get("run_id") == project.get("run_id")
        and restored.get("node_id") == node.get("id")
    ):
        return deepcopy(restored["message"]), restored["context_id"]
    if not enabled(config):
        return message, project.get("run_id")
    client = checked_client(config)
    # Callback correlation carries run_id only. Validate the whole pilot DAG
    # before its first scientific submission, including repeated server nodes.
    candidates = []
    for other in (workflow_def or {}).get("nodes", []):
        if other.get("type") != "a2a_agent":
            continue
        other_config = (
            config
            if other.get("server_id") == node["server_id"]
            else await A2AClientFactory.get_client(orch.storage).get_server_config(
                other["server_id"], project.get("tenant_id")
            )
        )
        if enabled(other_config):
            candidates.append(other["id"])
    if candidates != [node["id"]]:
        raise CheckpointConflict(
            "checkpoint workflow must have exactly one enabled A2A node"
        )
    sc = project["shared_context"]
    binding = {
        "run_id": project["run_id"],
        "project_id": sc.project_id
        if hasattr(sc, "project_id")
        else project.get("project_id"),
        "tenant_id": project.get("tenant_id"),
        "server_id": node["server_id"],
        "node_id": node["id"],
        "context_id": project["run_id"],
        "workflow_def": deepcopy(workflow_def),
        "message": deepcopy(message),
        "inputs": {
            "context": {
                k: deepcopy(sc._cache[k]) for k in CONTEXT_FIELDS if k in sc._cache
            },
            "runtime": {
                k: deepcopy(project[k]) for k in RUNTIME_FIELDS if k in project
            },
        },
    }
    await CheckpointStore(orch.storage, orch.message_store).bind_run(binding)
    await client.register_run(binding["context_id"], binding["run_id"], traceparent)
    logger.info(
        "[CHECKPOINT] project_id=%s run_id=%s node_id=%s — invocation registered",
        binding["project_id"],
        binding["run_id"],
        node["id"],
    )
    return message, binding["context_id"]
