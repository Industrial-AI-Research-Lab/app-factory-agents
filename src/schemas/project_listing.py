"""Shared API/storage contract constants for project listing."""

PROJECT_LIST_SORT_FIELDS = {
    "created_at",
    "updated_at",
    "title",
    "status",
    "current_phase",
}

PROJECT_LIST_DEFAULT_SORT = "created_at"
PROJECT_LIST_DEFAULT_LIMIT = 50
PROJECT_LIST_MAX_LIMIT = 100
PROJECT_LIST_UNKNOWN_FACET = "unknown"
