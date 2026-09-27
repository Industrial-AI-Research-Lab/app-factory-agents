"""
Tool Registry with Keyword-Based Search

Manages available tools and provides keyword-based search
to match tools to agent needs.

Tools are functions/APIs agents can use during development.

Note: Uses simple keyword matching for POC. 
For production, consider semantic search with sentence-transformers.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry of available tools with keyword-based search.
    
    Tools can be:
    - Python functions (create_file, run_command, etc.)
    - External APIs (web scraping, API calls, etc.)
    - MCP servers
    """
    
    def __init__(self):
        """Initialize tool registry"""
        # Tool catalog
        self.tools: List[Dict] = []
        
        # Access control (which agents can use which tools)
        self.access_rules: Dict[str, List[str]] = {}

        # Storage backend reference (set by load_from_db)
        self._storage = None
    
    def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        category: str,
        function: Optional[callable] = None,
        mcp_server: Optional[str] = None,
        metadata: Optional[Dict] = None,
        source: str = "builtin",
        schema: Optional[Dict] = None,
        enabled: bool = True,
        **_: Any,
    ):
        """
        Register a tool in the registry.

        Access control is exclusively via ``agent_configurations.allowed_tools``;
        registry rows describe tools only.
        """
        tool = {
            "tool_id": tool_id,
            "name": name,
            "description": description,
            "category": category,
            "function": function,
            "mcp_server": mcp_server,
            "metadata": metadata or {},
            "source": source,
            "schema": schema,
            "enabled": enabled,
        }

        self.tools.append(tool)

    def register_tools_from_config(self, config: List[Dict]):
        """Bulk register tools from config file (e.g. tools.yaml during tests)."""
        for raw in config:
            tc = {k: v for k, v in raw.items() if k != "allowed_agents"}
            tid = tc.get("_id") or tc.get("tool_id")
            if not tid:
                continue
            self.register_tool(
                tool_id=str(tid),
                name=str(tc.get("name", tid)),
                description=str(tc.get("description", "")),
                category=str(tc.get("category") or "").strip() or "general",
                mcp_server=tc.get("mcp_server"),
                metadata=tc.get("metadata") if isinstance(tc.get("metadata"), dict) else {},
                source=str(tc.get("source", "builtin")),
                schema=tc.get("schema"),
                enabled=bool(tc.get("enabled", True)),
            )

    async def load_from_db(self, storage, enabled_only: bool = True) -> int:
        """Load tool configurations from MongoDB into the registry.

        Replaces current in-memory tools with what's in the DB.
        Does not use code-level defaults: DB/seed is the single source of truth.

        Returns:
            Number of tools loaded.
        """
        self._storage = storage
        from storage.tool_doc_storage import get_all_tool_configurations_for_registry

        docs = await get_all_tool_configurations_for_registry(
            storage,
            enabled_only=enabled_only,
        )
        if not docs:
            logger.error("[TOOL_REGISTRY] tool_count=0 in DB — refusing fallback to hardcoded defaults")
            self.tools = []
            raise RuntimeError("No tool configurations found in MongoDB")

        from copy import deepcopy

        from config.tool_configuration_schema import (
            render_tool_openai_schema,
            tool_description_from_doc,
            tool_wire_name_from_doc,
        )
        from tools.mcp_tool_ids import mcp_public_tool_id_from_doc

        self.tools = []
        for doc in docs:
            registry_id = str(doc["_id"])
            pub = None
            if doc.get("source") == "mcp_server":
                pub = mcp_public_tool_id_from_doc(doc)
            schema = render_tool_openai_schema(doc)
            if pub and isinstance(schema, dict):
                schema = deepcopy(schema)
            wire = tool_wire_name_from_doc(doc) if doc.get("source") == "mcp_server" else str(
                doc.get("name") or registry_id,
            )
            tool = {
                "tool_id": registry_id,
                "public_tool_id": pub,
                "tenant_id": doc.get("tenant_id"),
                "name": wire,
                "rpc_name": doc.get("rpc_name"),
                "description": tool_description_from_doc(doc),
                "category": str(doc.get("category") or "").strip() or "unknown",
                "function": None,
                "mcp_server": doc.get("mcp_server"),
                "metadata": doc.get("metadata", {}),
                "source": doc.get("source", "builtin"),
                "schema": schema,
                # MCP docs load unfiltered so a disabled fork keeps shadowing the
                # shared doc; the flag must ride along because selection here is
                # synchronous and cannot re-read the DB.
                "enabled": bool(doc.get("enabled", True)),
            }
            self.tools.append(tool)

        logger.info("[TOOL_REGISTRY] Loaded %d tools from MongoDB", len(self.tools))
        return len(self.tools)

    def get_tools_for_agent(self, agent_id: str) -> List[Dict]:
        """Return the registered tools an agent could call (agent_id retained for
        API compatibility).

        Which tools an agent may *invoke* is enforced via ``allowed_tools`` when
        building schemas and dispatching; the registry is not agent-scoped.
        Disabled MCP docs are dropped: they only live in ``self.tools`` so a
        disabled fork keeps shadowing its shared doc during ranking (ADR-0013),
        and dispatch refuses every one of them.
        """
        return [
            t
            for t in self.tools
            if t.get("source") != "mcp_server" or t.get("enabled", True)
        ]

    @staticmethod
    def _tool_matches_tenant(tool: Dict, tenant_id: Optional[str]) -> bool:
        # Workspace builtins (create, read, …) are global for every project tenant.
        if tool.get("source") != "mcp_server":
            return True
        if not tenant_id:
            return False
        doc_tenant = str(tool.get("tenant_id") or "__root__")
        # Shared platform offering (ADR-0013): visible to every tenant, same rule
        # as dispatch. Selection ranking still prefers the tenant's own fork.
        if doc_tenant == "__system__":
            return True
        return doc_tenant == str(tenant_id)

    @staticmethod
    def _tool_allowed_keys(tool: Dict) -> set[str]:
        """Registry storage id plus wire/public refs for ``allowed_tools``."""
        from config.tool_configuration_schema import tool_wire_name_from_doc

        keys = {str(tool.get("tool_id") or "")}
        pub = tool.get("public_tool_id")
        if isinstance(pub, str) and pub:
            keys.add(pub)
        wire = tool_wire_name_from_doc(tool)
        if wire:
            keys.add(wire)
        stored_name = str(tool.get("name") or "").strip()
        if stored_name:
            keys.add(stored_name)
        return {k for k in keys if k}

    def _tool_allowed_for_agent(
        self,
        tool: Dict,
        allow: frozenset[str],
        *,
        tenant_id: Optional[str] = None,
    ) -> bool:
        if not self._tool_matches_tenant(tool, tenant_id):
            return False
        return bool(self._tool_allowed_keys(tool) & allow)

    @staticmethod
    def _registry_tool_as_doc(tool: Dict) -> Dict:
        return {
            "_id": tool.get("tool_id"),
            "tenant_id": tool.get("tenant_id"),
            "name": tool.get("name"),
            "mcp_server": tool.get("mcp_server"),
            "rpc_name": tool.get("rpc_name"),
            "source": tool.get("source", "builtin"),
            "schema": tool.get("schema"),
            "description": tool.get("description", ""),
            "enabled": tool.get("enabled", True),
        }

    def _select_tools_for_allow_list(
        self,
        allow: frozenset[str],
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Dict]:
        from config.tool_configuration_schema import (
            drop_disabled_effective_docs,
            select_effective_tool_docs_for_allow,
        )

        visible = [t for t in self.tools if self._tool_matches_tenant(t, tenant_id)]
        selected_docs = drop_disabled_effective_docs(
            select_effective_tool_docs_for_allow(
                [self._registry_tool_as_doc(tool) for tool in visible],
                allow,
                tenant_id=tenant_id,
            )
        )
        selected_ids = {str(doc.get("_id") or "") for doc in selected_docs}
        return [tool for tool in visible if str(tool.get("tool_id") or "") in selected_ids]

    def get_schemas_for_agent(
        self,
        agent_id: str,
        allowed_tool_ids: Optional[List[str]] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return OpenAI function-calling schemas intersected with ``allowed_tool_ids``.

        When ``allowed_tool_ids`` is missing or empty, returns no schemas (strict
        agent-centric model). ``agent_id`` is unused but kept for call-site compatibility.
        """
        from copy import deepcopy

        if not allowed_tool_ids:
            logger.info(
                "[TOOL_REGISTRY] agent_id=%s — get_schemas_for_agent empty allowed_tool_ids (strict)",
                agent_id,
            )
            return []

        allow = frozenset(allowed_tool_ids)
        schemas = []
        seen_public: set[str] = set()
        for tool in self._select_tools_for_allow_list(allow, tenant_id=tenant_id):
            pub = tool.get("public_tool_id")
            if isinstance(pub, str) and pub:
                if pub in seen_public:
                    continue
                seen_public.add(pub)
            if tool.get("schema"):
                schemas.append(deepcopy(tool["schema"]))
        return schemas
    
    async def search_tools(
        self,
        query: str,
        agent_type: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        allowed_tool_ids: Optional[List[str]] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Keyword-based search for relevant tools.

        Args:
            query: What the agent needs (e.g., "create a Python file")
            agent_type: Deprecated, ignored (kept for signature compatibility).
            category: Filter by category
            top_k: Number of results
            allowed_tool_ids: When set, only search within these tool ids (agent ``allowed_tools``).
            tenant_id: Project/tenant scope (MCP tools excluded when unset).
        """
        if not self.tools:
            return []

        available_tools = self.tools
        if allowed_tool_ids is not None:
            allow = frozenset(allowed_tool_ids)
            available_tools = self._select_tools_for_allow_list(allow, tenant_id=tenant_id)
        
        # Filter by category
        if category:
            available_tools = [
                t for t in available_tools
                if t["category"] == category
            ]
        
        if not available_tools:
            return []
        
        # Simple keyword matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each tool based on keyword matches
        scored_tools = []
        for tool in available_tools:
            description_lower = tool["description"].lower()
            name_lower = tool["name"].lower()
            
            # Count keyword matches
            score = 0.0
            
            # Exact phrase match (highest score)
            if query_lower in description_lower:
                score += 10.0
            
            # Name match
            if query_lower in name_lower:
                score += 8.0
            
            # Individual word matches in description
            description_words = set(description_lower.split())
            matching_words = query_words.intersection(description_words)
            score += len(matching_words) * 2.0
            
            # Individual word matches in name
            name_words = set(name_lower.split())
            matching_words_name = query_words.intersection(name_words)
            score += len(matching_words_name) * 3.0
            
            if score > 0:
                scored_tools.append({
                    **tool,
                    "relevance_score": score
                })
        
        # Sort by score (descending) and return top_k
        scored_tools.sort(key=lambda t: t["relevance_score"], reverse=True)
        return scored_tools[:top_k]
    
    def get_tool(self, tool_id: str) -> Optional[Dict]:
        """Get tool by ID"""
        for tool in self.tools:
            if tool["tool_id"] == tool_id:
                return tool
        return None
    
    def get_tools_by_category(self, category: str) -> List[Dict]:
        """Get all tools in a category"""
        return [t for t in self.tools if t["category"] == category]
    
    def get_stats(self) -> Dict:
        """Get registry statistics"""
        categories = {}
        for tool in self.tools:
            cat = tool["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_tools": len(self.tools),
            "categories": categories
        }

