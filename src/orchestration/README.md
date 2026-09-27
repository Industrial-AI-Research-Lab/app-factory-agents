# Orchestration Overview

This directory contains the core orchestration logic for AppFactory.

## Responsibilities

- Own the project lifecycle (creation -> requirements -> plan -> execution -> output -> done).
- Coordinate agents via auctions and phase runners.
- Maintain the in-memory view of project runtime state and bind it to `SharedContext`.
- Implement revert / retry / stop semantics consistent with the state models.

## Key entry points

- `orchestrator.py`
  - Project startup and workflow (`run_workflow`, phase runners).
  - Revert and retry (`revert_project`, `revert_to_user_message`).
  - Approval handling and `_request_approval`.
- `auction.py`
  - Task auctioning and agent selection.

## Design references

- Business logic and project lifecycle: [`docs/business-logic.md`](../../docs/business-logic.md).
- Detailed docs: [`docs/`](../../docs/).
- Agent context contracts: [`docs/agent-context-contracts.md`](../../docs/agent-context-contracts.md).

When changing orchestrator behavior, keep state transitions and events consistent with the project contracts.
