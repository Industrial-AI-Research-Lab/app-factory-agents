"""
Task Auction System

Implements the auction model where:
1. Tasks are broadcast to all agents
2. Agents submit bids (confidence scores)
3. CriticExpert validates the best bid
4. Task is assigned to winning agent
"""

from typing import Dict, List, Any
from datetime import datetime
import logging
import time
import uuid

from schemas import TaskSchema, EventSchema
from telemetry.tracer import get_tracer, create_task_with_context
import asyncio


logger = logging.getLogger(__name__)


class TaskAuction:
    """
    Manages the auction process for task assignment.
    
    Flow:
    1. broadcast_task() → all agents
    2. collect_bids() → gather responses
    3. validate_winner() → critic reviews
    4. assign_task() → winner gets task
    """
    
    def __init__(self, critic_agent=None, timeout: int = 10):
        """
        Args:
            critic_agent: CriticExpert agent for validation
            timeout: Max seconds to wait for bids
        """
        self.critic_agent = critic_agent
        self.timeout = timeout
        self.auction_history: List[Dict] = []
        self.event_emitter = None
        self.tracer = get_tracer()
    
    def set_event_emitter(self, emitter):
        """Inject event emitter for UI updates"""
        self.event_emitter = emitter
    
    async def run_auction(
        self,
        task: Dict[str, Any],
        agent_pool: List[Any],
        cancellation_token: Any | None = None,
        phase: str | None = None,
        run_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Run complete auction for a task.
        
        Args:
            task: Task to assign
            agent_pool: List of available agents
        
        Returns:
            Auction result:
                - winner: BaseAgent (assigned agent)
                - winning_bid: Dict (bid details)
                - all_bids: List[Dict] (all submitted bids)
                - critic_validation: Dict (critic's assessment)
        """
        auction_id = str(uuid.uuid4())
        if self.event_emitter:
            try:
                logger.info("[AUCTION] emit.started.fire_and_forget task_id=%s", TaskSchema.get_id(task))
                asyncio.create_task(self._emit_event(EventSchema.AUCTION_STARTED, run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "agent_count": len(agent_pool),
                    "auction_id": auction_id,
                }))
            except Exception:
                pass
        try:
            logger.info("[AUCTION] start task_id=%s agents=%d", TaskSchema.get_id(task), len(agent_pool))
        except Exception:
            pass
        
        # Create tracing span for the entire auction
        task_type = TaskSchema.get_type(task) or "unknown"
        with self.tracer.start_span(
            f"auction.run.{task_type}",
            attributes={
                "project.id": TaskSchema.get_project_id(task),
                "task.id": TaskSchema.get_id(task),
                "task.type": task_type,
                "agent.count": len(agent_pool),
                "AppFactory.auction.type": "task_assignment",
                "AppFactory.auction.timeout": self.timeout,
                "AppFactory.auction.critic_enabled": self.critic_agent is not None
            },
        ) as auction_span:
            # Early cancel
            if cancellation_token and getattr(cancellation_token, "is_cancelled", lambda: False)():
                return {
                    "winner": None,
                    "winning_bid": None,
                    "all_bids": [],
                    "critic_validation": {"approved": False, "reason": "Cancelled"},
                    "timestamp": datetime.utcnow().isoformat(),
                    "auction_id": auction_id,
                }
            # Step 1: Broadcast and collect bids
            bids = await self._collect_bids(
                task, agent_pool, cancellation_token, phase, run_id, auction_id=auction_id
            )
            try:
                logger.info("[AUCTION] bids_collected task_id=%s count=%d", TaskSchema.get_id(task), len(bids))
            except Exception:
                pass
        
            # Step 2: Find best bid
            if not bids:
                # Fallback: if no bids (LLM rate limited), directly assign based on task type
                task_type = TaskSchema.get_type(task)
                fallback_agent = None
                fallback_pool: List[Any] = []
                for agent in agent_pool:
                    try:
                        if hasattr(agent, "can_bid_on_phase") and not agent.can_bid_on_phase(phase):
                            continue
                    except Exception:
                        # Fail open for unexpected agent config, consistent with bid filtering
                        pass
                    fallback_pool.append(agent)

                for agent in fallback_pool:
                    agent_type = getattr(getattr(agent, 'agent_type', None), 'value', '')
                    if task_type == "coding" and agent_type == "coding":
                        fallback_agent = agent
                        break
                    elif task_type == "qa" and agent_type == "qa":
                        fallback_agent = agent
                        break
                
                if fallback_agent:
                    logger.info("[AUCTION] fallback_assign task_id=%s agent=%s (no bids collected)", 
                               TaskSchema.get_id(task), fallback_agent.agent_id)
                    fallback_bid = {
                        "agent_id": fallback_agent.agent_id,
                        "fit_score": 0.8,
                        "reasoning": "Fallback assignment (LLM unavailable for auction)",
                    }
                    if self.event_emitter:
                        asyncio.create_task(self._emit_event(EventSchema.AUCTION_COMPLETED, run_id, {
                            "project_id": TaskSchema.get_project_id(task),
                            "task_id": TaskSchema.get_id(task),
                            "winner_id": fallback_agent.agent_id,
                            "winner_display_name": getattr(fallback_agent, "get_display_name", lambda: None)(),
                            "fit_score": fallback_bid["fit_score"],
                            "bids": [fallback_bid],
                            "selection_mode": "fallback",
                            "auction_id": auction_id,
                        }))
                    return {
                        "winner": fallback_agent,
                        "winning_bid": fallback_bid,
                        "all_bids": [fallback_bid],
                        "critic_validation": {"approved": True, "reason": "Fallback assignment"},
                        "timestamp": datetime.utcnow().isoformat(),
                        "auction_id": auction_id,
                    }
                
                # No fallback possible - emit event and fail
                if self.event_emitter:
                    asyncio.create_task(self._emit_event(EventSchema.AUCTION_NO_BIDS, run_id, {
                        "project_id": TaskSchema.get_project_id(task),
                        "task_id": TaskSchema.get_id(task),
                        "task_description": TaskSchema.get_description(task),
                        "agent_count": len(agent_pool),
                        "auction_id": auction_id,
                    }))
                if auction_span:
                    self.tracer.set_error(auction_span, RuntimeError("No agents submitted bids"))
                return {
                    "winner": None,
                    "winning_bid": None,
                    "all_bids": [],
                    "critic_validation": {"approved": False, "reason": "No agents submitted bids"},
                    "timestamp": datetime.utcnow().isoformat(),
                    "auction_id": auction_id,
                }
                
            best_bid = max(bids, key=lambda b: b["fit_score"])
        
            # Find display name for best bid agent
            best_agent_display = None
            try:
                for a in agent_pool:
                    if a.agent_id == best_bid["agent_id"]:
                        best_agent_display = getattr(a, "get_display_name", lambda: None)()
                        break
            except Exception:
                pass
                
            if self.event_emitter:
                # Non-blocking emit prevents auction stall if any subscriber is backpressured
                asyncio.create_task(self._emit_event(EventSchema.AUCTION_BEST_BID, run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "task_type": TaskSchema.get_type(task),
                    "agent_id": best_bid["agent_id"],
                    "agent_display_name": best_agent_display,
                    "fit_score": best_bid["fit_score"],
                    "auction_id": auction_id,
                }))
        
            # Step 3: Critic validation (if available)
            critic_validation = {"approved": True}  # Default to approved
            if self.critic_agent:
                try:
                    logger.info("[AUCTION] critic_validate.start task_id=%s best_agent=%s timeout=%ss", TaskSchema.get_id(task), best_bid.get("agent_id"), self.timeout)
                except Exception:
                    pass
                _cv_t0 = time.monotonic()
                critic_validation = await self._validate_with_critic(
                    task,
                    best_bid,
                    bids,
                    run_id=run_id,
                )
                try:
                    logger.info(
                        "[AUCTION] critic_validate.end task_id=%s approved=%s elapsed=%.2fs reason=%r",
                        TaskSchema.get_id(task),
                        critic_validation.get("approved"),
                        time.monotonic() - _cv_t0,
                        critic_validation.get("reason"),
                    )
                except Exception:
                    pass

            # Fail-open: None or missing "approved" should be treated as approved
            # Only explicitly False should trigger rejection
            if critic_validation.get("approved") is False:
                if self.event_emitter:
                    # Fire-and-forget so validation emits never block auction progress
                    asyncio.create_task(self._emit_event(EventSchema.AUCTION_REJECTED, run_id, {
                        "project_id": TaskSchema.get_project_id(task),
                        "task_id": TaskSchema.get_id(task),
                        "reason": critic_validation.get("reason", "Unknown")
                    }))
                
                # Try second-best bid if available
                if len(bids) > 1:
                    sorted_bids = sorted(bids, key=lambda b: b["fit_score"], reverse=True)
                    best_bid = sorted_bids[1]  # Second best

                    try:
                        logger.info("[AUCTION] critic_validate.start task_id=%s best_agent=%s timeout=%ss attempt=2", TaskSchema.get_id(task), best_bid.get("agent_id"), self.timeout)
                    except Exception:
                        pass
                    _cv_t0 = time.monotonic()
                    critic_validation = await self._validate_with_critic(
                        task,
                        best_bid,
                        bids,
                        run_id=run_id,
                    )
                    try:
                        logger.info(
                            "[AUCTION] critic_validate.end task_id=%s approved=%s elapsed=%.2fs reason=%r attempt=2",
                            TaskSchema.get_id(task),
                            critic_validation.get("approved"),
                            time.monotonic() - _cv_t0,
                            critic_validation.get("reason"),
                        )
                    except Exception:
                        pass
        
            # Step 4: Find winner agent
            winner = None
            for agent in agent_pool:
                if agent.agent_id == best_bid["agent_id"]:
                    winner = agent
                    break
            try:
                logger.info("[AUCTION] winner task_id=%s agent=%s conf=%.2f", TaskSchema.get_id(task), best_bid.get("agent_id"), best_bid.get("fit_score", 0))
            except Exception:
                pass
        
            result = {
                "winner": winner,
                "winning_bid": best_bid,
                "all_bids": sorted(bids, key=lambda b: b["fit_score"], reverse=True),
                "critic_validation": critic_validation,
                "timestamp": datetime.utcnow().isoformat(),
                "auction_id": auction_id,
            }
        
            # Record auction history
            self.auction_history.append({
                "task_id": TaskSchema.get_id(task),
                "winner_id": best_bid["agent_id"],
                "fit_score": best_bid["fit_score"],
                "bid_count": len(bids),
                "timestamp": datetime.utcnow().isoformat()
            })
        
            if self.event_emitter:
                # Do not await completion here; keep auction hot path non-blocking
                asyncio.create_task(self._emit_event(EventSchema.AUCTION_COMPLETED, run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "winner_id": best_bid["agent_id"],
                    "winner_type": getattr(winner.agent_type, "value", str(getattr(winner, "agent_type", "unknown"))) if winner else "unknown",
                    "winner_display_name": getattr(winner, "get_display_name", lambda: None)() if winner else None,
                    "fit_score": best_bid["fit_score"],
                    "auction_id": auction_id,
                }))
            try:
                logger.info("[AUCTION] completed task_id=%s", TaskSchema.get_id(task))
            except Exception:
                pass
            
            if auction_span:
                auction_span.set_attribute("auction.bids", len(bids))
                auction_span.set_attribute("auction.winner", best_bid["agent_id"])
                auction_span.set_attribute("auction.winning_confidence", best_bid["fit_score"])
                # Avoid attaching list of dicts (unsupported); record bidder IDs instead
                try:
                    bidder_ids = [b.get("agent_id", "unknown") for b in bids if isinstance(b, dict)]
                    auction_span.set_attribute("auction.bidders", bidder_ids)
                except Exception:
                    pass
                self.tracer.set_success(auction_span)
            return result
    
    async def _collect_bids(
        self,
        task: Dict[str, Any],
        agent_pool: List[Any],
        cancellation_token: Any | None = None,
        phase: str | None = None,
        run_id: str | None = None,
        auction_id: str | None = None,
    ) -> List[Dict]:
        """
        Collect bids from all agents concurrently.
        
        Returns:
            List of bid dictionaries
        """
        # Filter agents by phase eligibility when available
        eligible_agents: List[Any] = []
        filtered_agents: List[str] = []
        for a in agent_pool:
            try:
                if hasattr(a, "can_bid_on_phase") and not a.can_bid_on_phase(phase):
                    filtered_agents.append(getattr(a, "agent_id", "unknown"))
                    continue
            except Exception:
                # Fail open if agent has unexpected config
                pass
            eligible_agents.append(a)
        if filtered_agents:
            logger.info(
                "[AUCTION] phase=%s filtered_agents=%s",
                phase or "unknown",
                ",".join(filtered_agents),
            )

        # Simple case fast-path: one eligible agent, skip LLM bidding.
        if len(eligible_agents) == 1:
            agent = eligible_agents[0]
            bid = {
                "agent_id": getattr(agent, "agent_id", "unknown"),
                "agent_type": getattr(agent, "agent_type", None),
                "fit_score": 1.0,
                "reasoning": "Single eligible agent for this phase",
                "timestamp": datetime.utcnow().isoformat(),
            }
            if self.event_emitter:
                try:
                    asyncio.create_task(self.event_emitter.emit("auction_bid_completed", run_id, {
                        "project_id": TaskSchema.get_project_id(task),
                        "task_id": TaskSchema.get_id(task),
                        "task_description": TaskSchema.get_description(task),
                        "task_type": TaskSchema.get_type(task),
                        "agent_id": getattr(agent, "agent_id", None),
                        "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                        "fit_score": 1.0,
                        "reasoning": "Single eligible agent for this phase",
                        "auction_id": auction_id,
                    }))
                except Exception:
                    pass
            return [bid]

        print(f"\n🔍 AUCTION: Starting bid collection from {len(eligible_agents)} agents")
        print(f"   Task: {TaskSchema.get_description(task)[:80]}...")
        print(f"   Timeout: {self.timeout}s")
        
        # Emit: Auction started
        if self.event_emitter:
            try:
                asyncio.create_task(self._emit_event("auction_started", run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "agent_count": len(eligible_agents),
                    "agents": [{"id": a.agent_id, "type": a.agent_type.value} for a in eligible_agents],
                    "auction_id": auction_id,
                }))
            except Exception:
                pass
        
        # Create bid tasks with per-agent timeout
        print("   Creating bid tasks with 60s per-agent timeout...")
        
        async def bid_with_timeout(agent):
            """Wrap bid_on_task with timeout and emit events"""
            # Emit: Agent started bidding
            if self.event_emitter:
                try:
                    asyncio.create_task(self.event_emitter.emit("auction_bid_started", run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "task_type": TaskSchema.get_type(task),
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                    "auction_id": auction_id,
                }))
                except Exception:
                    pass
            
            try:
                bid = await asyncio.wait_for(
                    agent.bid_on_task(task),
                    timeout=60.0  # 60s per agent
                )
                
                # Emit: Agent completed bid
                if self.event_emitter:
                    try:
                        asyncio.create_task(self.event_emitter.emit("auction_bid_completed", run_id, {
                        "project_id": TaskSchema.get_project_id(task),
                        "task_id": TaskSchema.get_id(task),
                        "task_description": TaskSchema.get_description(task),
                        "task_type": TaskSchema.get_type(task),
                        "agent_id": agent.agent_id,
                        "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                        "fit_score": bid.get("fit_score", 0.0),
                        "reasoning": bid.get("reasoning", ""),
                        "auction_id": auction_id,
                    }))
                    except Exception:
                        pass
                
                return bid
                
            except asyncio.TimeoutError:
                print(f"   ⏱️  {agent.agent_id} bid timed out after 60s")
                
                # Emit: Agent timed out
                if self.event_emitter:
                    try:
                        asyncio.create_task(self.event_emitter.emit("auction_bid_timeout", run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "task_type": TaskSchema.get_type(task),
                    "agent_id": agent.agent_id,
                    "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                    "auction_id": auction_id,
                    }))
                    except Exception:
                        pass
                
                return {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "fit_score": 0.0,
                    "reasoning": "Bid evaluation timed out"
                }
            except Exception as e:
                print(f"   ❌ {agent.agent_id} bid failed: {e}")
                
                # Emit: Agent failed
                if self.event_emitter:
                    try:
                        asyncio.create_task(self.event_emitter.emit("auction_bid_failed", run_id, {
                    "project_id": TaskSchema.get_project_id(task),
                    "task_id": TaskSchema.get_id(task),
                    "task_description": TaskSchema.get_description(task),
                    "task_type": TaskSchema.get_type(task),
                    "agent_id": agent.agent_id,
                    "agent_display_name": getattr(agent, "get_display_name", lambda: None)(),
                    "error": str(e),
                    "auction_id": auction_id,
                    }))
                    except Exception:
                        pass
                
                return {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "fit_score": 0.0,
                    "reasoning": f"Bid evaluation failed: {e}"
                }
        
        if cancellation_token and getattr(cancellation_token, "is_cancelled", lambda: False)():
            return []
        # Use context-preserving task creation so bid spans are children of auction span
        tasks = [create_task_with_context(bid_with_timeout(agent)) for agent in eligible_agents]
        print(f"   ✓ Created {len(tasks)} bid tasks")
        
        # Gather all bids; each bid task has its own per-agent timeout.
        print("   ⏳ Waiting for all bids (per-agent timeout: 60s)...")
        if cancellation_token and getattr(cancellation_token, "is_cancelled", lambda: False)():
            from orchestration.workflow_task_lifecycle import cancel_and_await

            await cancel_and_await(tasks, label="auction_bid")
            return []
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        bids = []
        for res in gather_results:
            bids.append(res)
        print(f"   ✓ Received {len(bids)} responses; 0 pending (per-agent timeouts handled inside tasks)")
        
        # Filter out exceptions and zero-confidence bids
        print(f"\n🎯 AUCTION DEBUG - Task: {TaskSchema.get_description(task)[:50]}...")
        print(f"   Total agents: {len(eligible_agents)}")
        print(f"   Raw bids received: {len(bids)}")
        
        for i, bid in enumerate(bids):
            if isinstance(bid, Exception):
                print(f"   ❌ Bid {i}: Exception - {bid}")
            elif isinstance(bid, dict):
                agent_id = bid.get("agent_id", "unknown")
                confidence = bid.get("fit_score", 0)
                print(f"   {'✅' if confidence > 0 else '❌'} Bid {i}: {agent_id} = {confidence:.2f}")
            else:
                print(f"   ❌ Bid {i}: Invalid type - {type(bid)}")
        
        valid_bids = [
            bid for bid in bids
            if isinstance(bid, dict) and bid.get("fit_score", 0) > 0
        ]
        
        print(f"   Valid bids: {len(valid_bids)}\n")
        
        return valid_bids
    
    async def _validate_with_critic(
        self,
        task: Dict[str, Any],
        best_bid: Dict,
        all_bids: List[Dict],
        run_id: str | None = None,
    ) -> Dict:
        """
        Ask CriticExpert to validate the winning bid.
        
        Returns:
            Validation result:
                - approved: bool
                - reason: str
                - confidence_adjustment: float (optional)
        """
        if not self.critic_agent:
            return {"approved": True, "reason": "No critic available"}
        
        validation_task = TaskSchema.create(
            task_id=f"validation_{TaskSchema.get_id(task)}",
            project_id=TaskSchema.get_project_id(task),
            task_type="bid_validation",
            description="Validate task assignment",
            context={
                "original_task": task,
                "best_bid": best_bid,
                "all_bids": all_bids
            }
        )

        # Mark the critic's call as auction-internal so its streaming events
        # are tagged phase="bidding" instead of inheriting the workflow phase
        # (e.g. "planning"). Without this the UI receives critic-validation
        # thinking events labeled identically to the winner agent's real work,
        # which makes the auction look concurrent. Restore the prior value in
        # `finally` so this never leaks outside the validation window.
        prior_bidding_flag = getattr(self.critic_agent, "_bidding_phase", False)
        self.critic_agent._bidding_phase = True
        try:
            result = await asyncio.wait_for(
                self.critic_agent.execute_task(validation_task),
                timeout=self.timeout
            )
            output = result.get("output") if isinstance(result, dict) else None
            if isinstance(output, dict):
                return output

            logger.warning(
                "[AUCTION] critic returned non-dict output; task_id=%s output_type=%s default_approved=true",
                TaskSchema.get_id(task),
                type(output).__name__,
            )
            return {
                "approved": True,
                "reason": "Critic returned invalid output format",
            }
        except asyncio.TimeoutError:
            # The auction abandoned the critic at self.timeout. The streaming
            # task may continue in the background (separate orphan-cancellation
            # bug) — but from the auction's perspective this contribution is
            # gone. Emit a public event so the UI can badge the critic's most
            # recent thought as "auction timed out".
            critic_agent_id = getattr(self.critic_agent, "agent_id", None)
            if self.event_emitter and critic_agent_id:
                # Emit only the plain form (not the auction.* prefix variant
                # `_emit_event` adds) — the prefixed copy would double the
                # MongoDB persistence and isn't consumed anywhere.
                asyncio.create_task(self.event_emitter.emit(
                    EventSchema.AGENT_VALIDATION_TIMEOUT,
                    run_id,
                    {
                        "project_id": TaskSchema.get_project_id(task),
                        "task_id": TaskSchema.get_id(task),
                        "agent_id": critic_agent_id,
                        "timeout_s": self.timeout,
                        "phase": "bidding",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ))
            return {
                "approved": True,
                "reason": f"Critic validation timed out after {self.timeout}s",
            }
        except Exception as e:
            # If critic fails, default to approval
            return {
                "approved": True,
                "reason": f"Critic validation failed: {str(e)}"
            }
        finally:
            self.critic_agent._bidding_phase = prior_bidding_flag
    
    async def _emit_event(self, event_type: str, run_id: str | None, data: Dict):
        """Emit event for UI updates: both namespaced and plain"""
        if self.event_emitter:
            # Namespaced form (for internal consumers)
            try:
                await self.event_emitter.emit(f"auction.{event_type}", run_id, data)
            except Exception:
                pass
            # Plain form expected by UI (e.g., 'auction_started')
            try:
                await self.event_emitter.emit(event_type, run_id, data)
            except Exception:
                # Swallow emitter failures so auction never stalls on emit
                pass
    
    def get_stats(self) -> Dict:
        """Get auction statistics"""
        if not self.auction_history:
            return {
                "total_auctions": 0,
                "avg_confidence": 0,
                "avg_bids_per_task": 0
            }
        
        return {
            "total_auctions": len(self.auction_history),
            "avg_confidence": sum(a["fit_score"] for a in self.auction_history) / len(self.auction_history),
            "avg_bids_per_task": sum(a["bid_count"] for a in self.auction_history) / len(self.auction_history)
        }
