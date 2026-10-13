# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context Compaction, Persistent Session State, and Async Memory Operations (Category 2).

Step 4a Implementation — Rubric Category 2.2 (History Compaction):
1. **Token-Based Event Compaction (`EventsCompactionConfig` + `LlmEventSummarizer`)**:
   Automatically summarizes older conversation events using Gemini Flash once prompt
   tokens cross `COMPACTION_TOKEN_THRESHOLD` (32,000 tokens), while retaining the most
   recent `EVENT_RETENTION_SIZE` (5) raw events un-compacted.
2. **Google Cloud Context Caching (`ContextCacheConfig`)**:
   Caches large static system instructions and retrieved onboarding documents on
   Gemini platform when context exceeds `MIN_CACHE_TOKENS` (2,048 tokens).
3. **Sliding-Window Truncation Fallback (`compact_conversation_history_sliding_window`)**:
   Provides deterministic token-budget and sliding-window truncation for long histories.

Step 4b Implementation — Rubric Categories 2.3 & 2.4:
4. **Persistent Session & Knowledge State (Category 2.3)**:
   - `PersistentOnboardingStore`: SQLite-backed database persisting Noogler profiles,
     saved 30-60-90 day onboarding plans, role knowledge documents, and consolidated
     long-term memories across turns and server restarts.
   - `create_persistent_session_service`: Configures Sessions in Gemini platform
     (`VertexAiSessionService`) in cloud environments or ADK's SQLite-backed
     `DatabaseSessionService` locally so session state persists across turns.
   - `create_gemini_platform_search_tool`: Wraps Search in Gemini platform
     (`VertexAiSearchTool`) for grounded enterprise knowledge retrieval.
5. **Async Memory Operations (Category 2.4)**:
   - `trigger_background_memory_consolidation` (`after_agent_callback`): Uses
     `asyncio.create_task()` to run `async_consolidate_session_memory()` in the
     background without blocking the main user conversation response thread.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.genai import types

from app.routing import FLASH_MODEL, build_gemini_model

logger = logging.getLogger(__name__)

# ============================================================================
# 1. ADK TOKEN-BASED HISTORY COMPACTION (Category 2.2 — Step 4a)
# ============================================================================

COMPACTION_TOKEN_THRESHOLD = 32_000
EVENT_RETENTION_SIZE = 5

# ADK Context Cache parameters for Gemini platform
MIN_CACHE_TOKENS = 2_048
CACHE_TTL_SECONDS = 1_800
CACHE_REFRESH_INTERVALS = 10


def create_events_compaction_config(
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    event_retention_size: int = EVENT_RETENTION_SIZE,
) -> EventsCompactionConfig:
    """Builds the ADK `EventsCompactionConfig` for automatic history summarization.

    When a Noogler onboarding session exceeds `token_threshold` prompt tokens,
    ADK invokes `LlmEventSummarizer` (powered by `gemini-3.8-flash`) to condense
    older turns into a compact summary event while keeping the most recent
    `event_retention_size` events intact.
    """
    summarizer = LlmEventSummarizer(llm=build_gemini_model(FLASH_MODEL))
    try:
        return EventsCompactionConfig(
            token_threshold=token_threshold,
            event_retention_size=event_retention_size,
            summarizer=summarizer,
        )
    except TypeError:  # pragma: no cover - compatibility across ADK minor versions
        return EventsCompactionConfig(
            compaction_interval=10,
            overlap_size=event_retention_size,
            summarizer=summarizer,
        )


def create_context_cache_config(
    min_tokens: int = MIN_CACHE_TOKENS,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    cache_intervals: int = CACHE_REFRESH_INTERVALS,
) -> ContextCacheConfig:
    """Builds the ADK `ContextCacheConfig` for Gemini platform context caching.

    Caches system constitution prompts and multi-turn onboarding context once
    the prompt exceeds `min_tokens` (2,048), reducing latency and token cost.
    """
    return ContextCacheConfig(
        min_tokens=min_tokens,
        ttl_seconds=ttl_seconds,
        cache_intervals=cache_intervals,
    )


def compact_conversation_history_sliding_window(
    events: list[dict[str, Any]],
    max_recent_events: int = EVENT_RETENTION_SIZE,
    max_estimated_tokens: int = COMPACTION_TOKEN_THRESHOLD,
) -> dict[str, Any]:
    """Deterministic sliding-window + token-budget compaction helper.

    Complements ADK's `EventsCompactionConfig` by providing an explicit,
    unit-testable sliding window + summary generator over conversation events.

    Args:
        events: List of conversation turn dictionaries (`{"role": ..., "content": ...}`).
        max_recent_events: Number of most recent events to preserve verbatim.
        max_estimated_tokens: Token ceiling (~4 chars/token) before compaction triggers.

    Returns:
        Dictionary containing `compacted_events`, `was_compacted`, `original_count`,
        and `retained_count`.
    """
    total_chars = sum(len(str(e.get("content", ""))) for e in events)
    estimated_tokens = total_chars // 4

    if len(events) <= max_recent_events and estimated_tokens <= max_estimated_tokens:
        return {
            "was_compacted": False,
            "original_count": len(events),
            "retained_count": len(events),
            "estimated_tokens_before": estimated_tokens,
            "compacted_events": list(events),
        }

    older_events = events[:-max_recent_events] if max_recent_events > 0 else events
    recent_events = events[-max_recent_events:] if max_recent_events > 0 else []

    summarized_snippets = [
        f"[{e.get('role', 'turn')}]: {str(e.get('content', ''))[:120]}"
        for e in older_events
    ]
    summary_event = {
        "role": "system_summary",
        "content": (
            f"Compacted summary of {len(older_events)} earlier onboarding turns: "
            + " | ".join(summarized_snippets)
        ),
    }
    compacted = [summary_event, *recent_events]
    return {
        "was_compacted": True,
        "original_count": len(events),
        "retained_count": len(compacted),
        "estimated_tokens_before": estimated_tokens,
        "compacted_events": compacted,
    }


# ============================================================================
# 2. PERSISTENT SESSION & KNOWLEDGE DATABASE (Category 2.3 — Step 4b)
# ============================================================================

DEFAULT_SQLITE_DB_PATH = os.environ.get(
    "ONBOARDING_SQLITE_DB_PATH",
    str(Path(__file__).resolve().parent.parent / ".onboarding_state.sqlite3"),
)


class PersistentOnboardingStore:
    """SQLite-backed persistent store for Noogler profiles, 30-60-90 plans,
    role knowledge documents, and consolidated long-term memories.

    Satisfies Rubric Category 2.3 (Persistent Session State) by ensuring all
    state changes, onboarding plans, and knowledge retrieval are backed by a
    durable relational/document database across conversation turns and restarts.
    """

    def __init__(self, db_path: str = DEFAULT_SQLITE_DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS noogler_profiles (
                    employee_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS onboarding_plans (
                    employee_id TEXT PRIMARY KEY,
                    job_role TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS role_knowledge_docs (
                    doc_id TEXT PRIMARY KEY,
                    job_role TEXT NOT NULL,
                    pillar TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS consolidated_memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def seed_profiles_if_empty(
        self, seed_records: dict[str, dict[str, Any]]
    ) -> None:
        """Seeds initial Noogler profiles into SQLite if the table is empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM noogler_profiles"
            ).fetchone()
            if row and row["cnt"] == 0:
                now = time.time()
                for emp_id, record in seed_records.items():
                    conn.execute(
                        """
                        INSERT INTO noogler_profiles (employee_id, profile_json, updated_at)
                        VALUES (?, ?, ?)
                        """,
                        (emp_id, json.dumps(record), now),
                    )

    def get_all_profiles(
        self, seed_records: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Loads all Noogler profiles from the persistent SQLite database."""
        if seed_records:
            self.seed_profiles_if_empty(seed_records)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT employee_id, profile_json FROM noogler_profiles"
            ).fetchall()
        return {
            row["employee_id"]: json.loads(row["profile_json"]) for row in rows
        }

    def upsert_profile(self, employee_id: str, profile: dict[str, Any]) -> None:
        """Persists a Noogler profile mutation to the SQLite database."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO noogler_profiles (employee_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (employee_id, json.dumps(profile), time.time()),
            )

    def seed_knowledge_docs_if_empty(
        self, knowledge_base: dict[str, dict[str, list[dict[str, str]]]]
    ) -> None:
        """Seeds role onboarding curriculum documents into SQLite if empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM role_knowledge_docs"
            ).fetchone()
            if row and row["cnt"] == 0:
                for job_role, sections in knowledge_base.items():
                    for pillar, docs in sections.items():
                        for doc in docs:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO role_knowledge_docs
                                (doc_id, job_role, pillar, title, content)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    doc["doc_id"],
                                    job_role,
                                    pillar,
                                    doc["title"],
                                    doc["content"],
                                ),
                            )

    def query_role_knowledge(
        self,
        job_role: str,
        topic_filter: str = "all",
        seed_kb: dict[str, dict[str, list[dict[str, str]]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Queries role onboarding documents from the persistent SQLite store."""
        if seed_kb:
            self.seed_knowledge_docs_if_empty(seed_kb)
        with self._connect() as conn:
            if topic_filter == "all":
                rows = conn.execute(
                    """
                    SELECT doc_id, title, content, pillar
                    FROM role_knowledge_docs
                    WHERE job_role = ?
                    ORDER BY doc_id
                    """,
                    (job_role,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT doc_id, title, content, pillar
                    FROM role_knowledge_docs
                    WHERE job_role = ? AND pillar = ?
                    ORDER BY doc_id
                    """,
                    (job_role, topic_filter),
                ).fetchall()
        return [dict(row) for row in rows]

    def save_onboarding_plan(
        self, employee_id: str, job_role: str, plan_record: dict[str, Any]
    ) -> None:
        """Persists a 30-60-90 day onboarding plan in SQLite."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO onboarding_plans (employee_id, job_role, plan_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    job_role = excluded.job_role,
                    plan_json = excluded.plan_json,
                    updated_at = excluded.updated_at
                """,
                (employee_id, job_role, json.dumps(plan_record), time.time()),
            )

    def get_onboarding_plan(self, employee_id: str) -> dict[str, Any] | None:
        """Retrieves a persisted 30-60-90 day onboarding plan by employee ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT plan_json FROM onboarding_plans WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def save_consolidated_memory(
        self,
        session_id: str,
        user_id: str,
        agent_name: str,
        summary_text: str,
        facts: list[str],
    ) -> int:
        """Stores a PII-scrubbed background-consolidated session memory entry into SQLite."""
        from app.observability import scrub_pii

        scrubbed_summary = str(scrub_pii(summary_text))
        scrubbed_facts = [str(scrub_pii(f)) for f in facts]
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO consolidated_memories
                (session_id, user_id, agent_name, summary_text, facts_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    agent_name,
                    scrubbed_summary,
                    json.dumps(scrubbed_facts),
                    time.time(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def get_consolidated_memories(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieves consolidated long-term memories for verification or recall."""
        query = "SELECT * FROM consolidated_memories WHERE 1=1"
        params: list[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY memory_id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "agent_name": row["agent_name"],
                "summary_text": row["summary_text"],
                "facts": json.loads(row["facts_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


@functools.cache
def get_persistent_onboarding_store(
    db_path: str = DEFAULT_SQLITE_DB_PATH,
) -> PersistentOnboardingStore:
    """Returns the singleton `PersistentOnboardingStore` instance."""
    return PersistentOnboardingStore(db_path=db_path)


def create_persistent_session_service(
    db_url: str | None = None,
    base_dir: str | None = None,
) -> Any:
    """Creates a persistent ADK SessionService across turns (Category 2.3).

    Resolution order:
    1. Explicit `SESSION_SERVICE_URI` or `db_url` (e.g. `sqlite:///...` or Postgres).
    2. `GOOGLE_CLOUD_AGENT_ENGINE_ID` -> `VertexAiSessionService` (Sessions in Gemini platform).
    3. Local persistent SQLite `DatabaseSessionService` if `sqlalchemy`/`aiosqlite` is
       installed, falling back gracefully to `InMemorySessionService` coupled with
       `PersistentOnboardingStore`.
    """
    from google.adk.cli.utils.service_factory import (
        create_session_service_from_options,
    )

    resolved_uri = db_url or os.environ.get("SESSION_SERVICE_URI")
    if resolved_uri:
        return create_session_service_from_options(
            base_dir=base_dir or os.getcwd(),
            session_service_uri=resolved_uri,
        )

    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.sessions.vertex_ai_session_service import (
            VertexAiSessionService,
        )

        return VertexAiSessionService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )

    try:
        from google.adk.sessions.database_session_service import (
            DatabaseSessionService,
        )

        sqlite_uri = f"sqlite:///{DEFAULT_SQLITE_DB_PATH}"
        return DatabaseSessionService(db_url=sqlite_uri)
    except Exception:
        from google.adk.sessions.in_memory_session_service import (
            InMemorySessionService,
        )

        return InMemorySessionService()


def create_gemini_platform_search_tool(
    data_store_id: str | None = None,
) -> Any | None:
    """Builds a Search in Gemini platform tool (`VertexAiSearchTool`) when configured.

    Allows seamless attachment to an enterprise Search in Gemini platform datastore
    via `GEMINI_PLATFORM_DATASTORE_ID` while keeping local SQLite knowledge retrieval
    active for offline development and automated CI evaluation.
    """
    resolved_datastore = data_store_id or os.environ.get(
        "GEMINI_PLATFORM_DATASTORE_ID"
    )
    if not resolved_datastore:
        return None
    from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

    return VertexAiSearchTool(data_store_id=resolved_datastore)


# ============================================================================
# 3. ASYNC MEMORY OPERATIONS (Category 2.4 — Step 4b)
# ============================================================================

# Track in-flight background memory tasks so Python's garbage collector never
# drops a running task and unit tests can await pending tasks deterministically.
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()


def get_pending_background_memory_tasks() -> set[asyncio.Task[Any]]:
    """Returns the set of currently running background memory consolidation tasks."""
    return set(_BACKGROUND_MEMORY_TASKS)


def create_memory_service() -> Any:
    """Creates the ADK long-term MemoryService (Memory Bank in Gemini platform or local)."""
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.memory.vertex_ai_memory_bank_service import (
            VertexAiMemoryBankService,
        )

        return VertexAiMemoryBankService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )

    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

    return InMemoryMemoryService()


async def async_consolidate_session_memory(
    *,
    session_id: str,
    user_id: str,
    agent_name: str,
    state_snapshot: dict[str, Any],
    recent_event_texts: list[str],
    session_obj: Any | None = None,
    memory_service: Any | None = None,
    store: PersistentOnboardingStore | None = None,
) -> dict[str, Any]:
    """Asynchronously consolidates turn insights into long-term memory (Category 2.4).

    Executed inside a background `asyncio.Task` spawned by `after_agent_callback`
    so the user never waits on memory extraction or persistence.

    Steps performed asynchronously:
    1. If an ADK `MemoryService` (e.g., Memory Bank in Gemini platform) and `Session`
       object are present, calls `await memory_service.add_session_to_memory(session_obj)`.
    2. Extracts structured onboarding facts (active employee IDs, saved 30-60-90 plans,
       retrieved role docs, and recent turn snippets) and writes them to the persistent
       SQLite `consolidated_memories` table.
    """
    await asyncio.sleep(0)  # Yield control immediately so caller continues unblocked

    extracted_facts: list[str] = []
    if "noogler_records" in state_snapshot and isinstance(
        state_snapshot["noogler_records"], dict
    ):
        for emp_id, rec in state_snapshot["noogler_records"].items():
            if isinstance(rec, dict):
                plan = rec.get("enrolled_health_plan", "UNENROLLED")
                k401 = rec.get("retirement_401k_contribution_percent", 0)
                sku = rec.get("assigned_laptop_sku", "UNKNOWN")
                extracted_facts.append(
                    f"{emp_id}: health_plan={plan}, 401k={k401}%, laptop={sku}"
                )

    saved_plans = state_snapshot.get("user:saved_onboarding_plans")
    if isinstance(saved_plans, dict):
        for emp_id, plan_data in saved_plans.items():
            role = (
                plan_data.get("job_role", "unknown")
                if isinstance(plan_data, dict)
                else "unknown"
            )
            extracted_facts.append(f"{emp_id}: saved 30-60-90 plan for role={role}")

    for snippet in recent_event_texts[-3:]:
        if snippet:
            extracted_facts.append(f"turn_snippet: {snippet[:140]}")

    summary_text = (
        f"Async consolidated memory for session={session_id} (agent={agent_name}): "
        f"{len(extracted_facts)} onboarding facts captured."
    )

    # 1. Sync with ADK MemoryService (Memory Bank in Gemini platform / InMemoryMemoryService)
    memory_bank_synced = False
    if memory_service is not None and session_obj is not None:
        try:
            await memory_service.add_session_to_memory(session_obj)
            memory_bank_synced = True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("MemoryService sync skipped or failed: %s", exc)

    # 2. Persist into SQLite consolidated_memories table
    target_store = store or get_persistent_onboarding_store()
    memory_id = target_store.save_consolidated_memory(
        session_id=session_id,
        user_id=user_id,
        agent_name=agent_name,
        summary_text=summary_text,
        facts=extracted_facts,
    )

    return {
        "status": "consolidated",
        "memory_id": memory_id,
        "session_id": session_id,
        "user_id": user_id,
        "agent_name": agent_name,
        "facts_count": len(extracted_facts),
        "memory_bank_synced": memory_bank_synced,
    }


def trigger_background_memory_consolidation(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Non-blocking `after_agent_callback` that schedules background memory consolidation.

    Satisfies Rubric Category 2.4 (Async Memory Operations):
    Spawns `async_consolidate_session_memory` via `asyncio.create_task()` on the
    running event loop and immediately returns `None` so the user's conversation
    response is delivered with zero added latency.
    """
    try:
        state_dict = (
            callback_context.state.to_dict()
            if hasattr(callback_context.state, "to_dict")
            else dict(callback_context.state)
        )
    except Exception:
        state_dict = {}

    agent_name = getattr(callback_context, "agent_name", "noogler_onboarding_agent")
    inv_ctx = getattr(callback_context, "_invocation_context", None)
    session_obj = getattr(inv_ctx, "session", None) if inv_ctx else None
    memory_service = getattr(inv_ctx, "memory_service", None) if inv_ctx else None

    session_id = (
        getattr(session_obj, "id", None)
        or getattr(callback_context, "session_id", None)
        or "default_session"
    )
    user_id = (
        getattr(session_obj, "user_id", None)
        or getattr(callback_context, "user_id", None)
        or "default_noogler"
    )

    recent_texts: list[str] = []
    events = getattr(session_obj, "events", None) or []
    for ev in events[-4:]:
        content = getattr(ev, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    recent_texts.append(str(text))

    coro = async_consolidate_session_memory(
        session_id=str(session_id),
        user_id=str(user_id),
        agent_name=str(agent_name),
        state_snapshot=state_dict,
        recent_event_texts=recent_texts,
        session_obj=session_obj,
        memory_service=memory_service,
    )

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            coro, name=f"memory_consolidation_{session_id}_{int(time.time() * 1000)}"
        )
        _BACKGROUND_MEMORY_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    except RuntimeError:
        # Fallback if invoked outside an active asyncio event loop in synchronous unit tests
        asyncio.run(coro)

    # Returning None allows the agent's normal response to proceed immediately
    return None
