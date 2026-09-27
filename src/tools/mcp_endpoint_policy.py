"""SSRF policy for remote HTTP MCP discovery endpoints."""

from __future__ import annotations

import ipaddress
import os
import re
from functools import lru_cache
from typing import List, Tuple

# Hard-coded MCP discover allowlist (no env required).
# Optional: EXTERNAL_MCP_ENDPOINT_ALLOWLIST adds more hosts/CIDRs on top of these.
_BUILTIN_MCP_ALLOWLIST_HOSTS: Tuple[str, ...] = ("admin.example.com",)
_BUILTIN_MCP_ALLOWLIST_CIDRS: Tuple[str, ...] = ("10.0.0.0/16",)

_WILDCARD_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.\*\.\*$")


def local_mcp_endpoints_allowed() -> bool:
    """When true, allow localhost and private/LAN addresses for remote MCP discovery."""
    return os.environ.get("EXTERNAL_MCP_ALLOW_LOCAL_ENDPOINTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def parse_mcp_allowlist(entries: List[str]) -> Tuple[List[str], List[ipaddress.IPv4Network | ipaddress.IPv6Network]]:
    """Split allowlist entries into host suffixes and IP networks (CIDR or ``10.0.*.*``)."""
    hosts: List[str] = list(_BUILTIN_MCP_ALLOWLIST_HOSTS)
    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for cidr in _BUILTIN_MCP_ALLOWLIST_CIDRS:
        networks.append(ipaddress.ip_network(cidr, strict=False))

    for raw in entries:
        entry = str(raw or "").strip().lower()
        if not entry:
            continue
        wildcard = _WILDCARD_IPV4_RE.match(entry)
        if wildcard:
            first, second = wildcard.group(1), wildcard.group(2)
            try:
                networks.append(ipaddress.ip_network(f"{first}.{second}.0.0/16", strict=False))
            except ValueError:
                pass
            continue
        if "/" in entry:
            try:
                networks.append(ipaddress.ip_network(entry, strict=False))
                continue
            except ValueError:
                pass
        try:
            ip_obj = ipaddress.ip_address(entry.strip("[]"))
            prefix = 128 if ip_obj.version == 6 else 32
            networks.append(ipaddress.ip_network(f"{entry}/{prefix}", strict=False))
            continue
        except ValueError:
            pass
        if entry not in hosts:
            hosts.append(entry)
    return hosts, networks


@lru_cache(maxsize=1)
def _cached_env_allowlist_entries() -> Tuple[str, ...]:
    raw = os.environ.get("EXTERNAL_MCP_ENDPOINT_ALLOWLIST", "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def get_mcp_allowlist_entries() -> List[str]:
    return list(_cached_env_allowlist_entries())


def clear_mcp_allowlist_cache() -> None:
    _cached_env_allowlist_entries.cache_clear()


def is_cluster_internal_hostname(host: str) -> bool:
    h = str(host or "").strip().lower()
    return h.endswith(".cluster.local") or h.endswith(".svc.cluster.local")


def is_local_mcp_hostname(host: str) -> bool:
    """True for loopback, RFC1918 literals, and common dev hostnames."""
    h = str(host or "").strip().lower()
    if not h or is_cluster_internal_hostname(h):
        return False
    if h in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "host.docker.internal"}:
        return True
    if h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        ip_obj = ipaddress.ip_address(h.strip("[]"))
    except ValueError:
        return False
    return any(
        (
            ip_obj.is_private,
            ip_obj.is_loopback,
            ip_obj.is_link_local,
        )
    )


def is_host_in_allowlist(host: str, hosts: List[str]) -> bool:
    h = str(host or "").strip().lower()
    for entry in hosts:
        if h == entry or h.endswith(f".{entry}"):
            return True
    return False


def is_ip_in_allowlist_networks(
    ip_value: str,
    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    try:
        ip_obj = ipaddress.ip_address(str(ip_value or "").strip("[]"))
    except ValueError:
        return False
    return any(ip_obj in net for net in networks)


def is_blocked_mcp_hostname(host: str) -> bool:
    h = str(host or "").strip().lower()
    hosts, networks = parse_mcp_allowlist(get_mcp_allowlist_entries())
    if local_mcp_endpoints_allowed() and is_local_mcp_hostname(h):
        return False
    if is_host_in_allowlist(h, hosts) or is_ip_in_allowlist_networks(h, networks):
        return False
    if h == "host.docker.internal":
        return True
    if h in {"localhost"}:
        return True
    if h.endswith(".localhost"):
        return True
    if h.endswith(".local"):
        return True
    if h.endswith(".localdomain"):
        return True
    if is_cluster_internal_hostname(h):
        return True
    return is_private_or_restricted_ip(h)


def is_allowlisted_mcp_hostname(host: str, *, allowlist: List[str]) -> bool:
    h = str(host or "").strip().lower()
    hosts, networks = parse_mcp_allowlist(allowlist)
    if is_host_in_allowlist(h, hosts) or is_ip_in_allowlist_networks(h, networks):
        return True
    if local_mcp_endpoints_allowed() and is_local_mcp_hostname(h):
        return True
    if not allowlist:
        return True
    return False


def is_private_or_restricted_ip(ip_value: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_value)
    except ValueError:
        return False
    return any(
        (
            ip_obj.is_private,
            ip_obj.is_loopback,
            ip_obj.is_link_local,
            ip_obj.is_multicast,
            ip_obj.is_reserved,
            ip_obj.is_unspecified,
        )
    )


def is_blocked_resolved_mcp_ip(ip_value: str) -> bool:
    _, networks = parse_mcp_allowlist(get_mcp_allowlist_entries())
    if is_ip_in_allowlist_networks(ip_value, networks):
        return False
    if local_mcp_endpoints_allowed() and is_private_or_restricted_ip(ip_value):
        return False
    return is_private_or_restricted_ip(ip_value)
