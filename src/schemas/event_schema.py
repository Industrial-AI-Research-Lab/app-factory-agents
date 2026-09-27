"""
Event Schema Definition

Defines event type constants for the event system.
If you need to change event names, change them HERE only.
"""

from dataclasses import dataclass


@dataclass
class EventSchema:
    """
    Event type constants.
    
    DO NOT use hardcoded event strings in business logic!
    Use EventSchema constants instead.
    """
    
    # Project events
    PROJECT_STARTED = "project_started"
    PROJECT_COMPLETED = "project_completed"
    PROJECT_FAILED = "project_failed"
    
    # Phase events (use format_phase_event() to construct)
    PHASE_STARTED = "started"
    PHASE_COMPLETED = "completed"
    PHASE_FAILED = "failed"
    
    # Approval events
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GIVEN = "approval_given"
    APPROVAL_UPDATED = "approval_updated"
    
    # Auction events (use format_auction_event() to construct)
    AUCTION_STARTED = "auction_started"
    AUCTION_NO_BIDS = "auction_no_bids"
    AUCTION_BEST_BID = "auction_best_bid"
    AUCTION_REJECTED = "auction_rejected"
    AUCTION_COMPLETED = "auction_completed"
    
    # Project control events (stop/revert)
    PROJECT_STOPPING = "project_stopping"
    PROJECT_STOPPED = "project_stopped"
    PROJECT_REVERTING = "project_reverting"
    PROJECT_REVERTED = "project_reverted"
    PROJECT_REVERT_FAILED = "project_revert_failed"
    
    # Snapshot lifecycle
    SNAPSHOT_CREATED = "snapshot_created"
    
    # Agent events
    TASK_ASSIGNED = "task_assigned"
    TASK_ATTEMPT = "task_attempt"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    CODE_GENERATED = "code_generated"
    QA_COMPLETE = "qa_complete"
    PLANNING_COMPLETE = "planning_complete"
    AGENT_DELEGATION_STARTED = "agent.delegation.started"
    AGENT_DELEGATION_COMPLETED = "agent.delegation.completed"
    AGENT_DELEGATION_FAILED = "agent.delegation.failed"

    # Validator events (AppFactory-77). A rejection routes the run back around the
    # retry loop showing no card and writing no chat message, so without this
    # the UI shows a phase silently re-running with no stated reason.
    VALIDATOR_REJECTED = "validator.rejected"
    OUTPUT_REPAIR_STARTED = "output_repair_started"

    # A workflow `tool` node calls one MCP operation with no agent, so these
    # are the only signal the UI gets that the node ran.
    TOOL_NODE_STARTED = "tool.started"
    TOOL_NODE_FINISHED = "tool.finished"
    TOOL_NODE_SKIPPED = "tool.skipped"

    # Item tasks of a workflow `map` node run in parallel with their chat
    # messages suppressed; these carry the progress, never the item content.
    MAP_STARTED = "map.started"
    MAP_ITEM_STARTED = "map.item_started"
    MAP_ITEM_FINISHED = "map.item_finished"
    MAP_COMPLETED = "map.completed"

    # Container events
    CONTAINER_CREATED = "container_created"
    CONTAINER_CHECKOUT = "container_checkout"
    CONTAINER_TESTED = "container_tested"
    CONTAINER_APPLIED = "container_applied"

    # Agent termination / abandonment signals (used by UI to badge thoughts
    # with the reason a thinking block ended other than naturally).
    # AGENT_VALIDATION_TIMEOUT: auction.wait_for(critic.execute_task) hit its
    #   timeout and the auction defaulted to approve. The critic agent may
    #   still be streaming in the background — but from the orchestration
    #   perspective its contribution was abandoned at this moment.
    # AGENT_STREAM_TERMINATED: the LLM streaming call exited via an exception
    #   (CancelledError / generic Exception) — i.e. it did NOT complete naturally.
    # AGENT_STREAM_CLOSED: the LLM streaming call completed naturally. Pairs
    #   with AGENT_STREAM_TERMINATED for symmetry; the UI uses `elapsed` to
    #   detect provider-buffered responses (one SSE chunk after a long wire
    #   wait, e.g. OpenRouter free-tier endpoints).
    AGENT_VALIDATION_TIMEOUT = "agent.validation.timeout"
    AGENT_STREAM_TERMINATED = "agent.streaming.terminated"
    AGENT_STREAM_CLOSED = "agent.streaming.closed"
    # Inspector capture signal: emitted once per LLM invocation after the
    # capture doc lands in `agent_llm_calls`. UI uses {call_id} to fetch
    # full doc on demand. See storage/agent_llm_calls_store.py.
    AGENT_INVOCATION_CAPTURED = "agent.invocation.captured"
    # Concurrency instrumentation: emitted when a new stream opens while
    # another is already open for the same project. The UI's
    # ConcurrencyDebugPanel surfaces these so the user can bundle evidence.
    AGENT_STREAM_CONCURRENCY = "agent.streaming.concurrency_warning"

    # Plugin system (ADR-0004). PLUGIN_ERROR: a plugin raised inside a hook —
    # the execution continues, the event names the plugin and hook point.
    # PLUGIN_MARKER: observability marker emitted by a plugin (the AppFactory-147
    # skeleton emits one per hook call to prove lifecycle placement).
    PLUGIN_ERROR = "plugin.error"
    PLUGIN_MARKER = "plugin.marker"
    
    @staticmethod
    def format_phase_event(phase: str, status: str) -> str:
        """
        Format a phase event name.
        
        Args:
            phase: Phase name (requirements, planning, execution)
            status: Status (started, completed, failed)
        
        Returns:
            Formatted event name: "phase.{phase}.{status}"
        """
        return f"phase.{phase}.{status}"
    
    @staticmethod
    def format_auction_event(event_type: str) -> str:
        """
        Format an auction event name.
        
        Args:
            event_type: Event type (auction_started, auction_completed, etc.)
        
        Returns:
            Formatted event name: "auction.{event_type}"
        """
        return f"auction.{event_type}"
