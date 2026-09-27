"""Per-call trace propagation for A2A SDK requests."""

from a2a.client import ClientCallContext

from telemetry.run_context import outgoing_trace_headers
from telemetry.tracer import get_tracer


def build_call_context(timeout: float) -> ClientCallContext:
    headers = outgoing_trace_headers(get_tracer())
    if headers:
        return ClientCallContext(timeout=timeout, service_parameters=headers)
    return ClientCallContext(timeout=timeout)
