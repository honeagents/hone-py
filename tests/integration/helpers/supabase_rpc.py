"""
Supabase RPC helper for integration tests.

Exact replica of TypeScript supabase-rpc.ts - makes direct RPC calls
to the sync_prompts and insert_runs functions.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import httpx

SUPABASE_API_URL = os.environ.get("SUPABASE_API_URL", "http://127.0.0.1:54321")
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9."
    "CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
)


class PromptRequestItem(TypedDict, total=False):
    id: str
    name: Optional[str]
    version: Optional[str]
    prompt: str
    paramKeys: List[str]
    childrenIds: List[str]


class SyncPromptsRequest(TypedDict):
    rootId: str
    map: Dict[str, PromptRequestItem]


class SyncPromptsResponse(TypedDict):
    prompt: str


class InsertRunsRequest(TypedDict, total=False):
    id: str  # prompt_id
    messages: List[Dict[str, str]]
    sessionId: str
    timestamp: Optional[str]


@dataclass
class InsertRunsResponse:
    id: str
    project_id: str
    prompt_id: str
    session_id: str
    created_at: str


@dataclass
class RpcError:
    code: str
    message: str
    details: Optional[str]
    hint: Optional[str]


class RpcCallError(Exception):
    """Error from RPC call."""

    def __init__(self, error: Dict[str, Any]):
        super().__init__(error.get("message", "RPC error"))
        self.code = error.get("code", "UNKNOWN")
        self.details = error.get("details")
        self.hint = error.get("hint")


class SupabaseRpc:
    """Supabase RPC helper for integration tests."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def sync_prompts(
        self,
        request: SyncPromptsRequest,
    ) -> Dict[str, SyncPromptsResponse]:
        """Call the sync_prompts RPC function."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/sync_prompts"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={"prompts": request},
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            return response.json()

    async def insert_runs(
        self,
        request: InsertRunsRequest,
    ) -> InsertRunsResponse:
        """Call the insert_runs RPC function."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/insert_runs"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "id": request["id"],
                    "messages": request["messages"],
                    "sessionId": request["sessionId"],
                    "timestamp": request.get("timestamp"),
                },
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            data = response.json()
            return InsertRunsResponse(
                id=data["id"],
                project_id=data["projectId"],
                prompt_id=data["promptId"],
                session_id=data["sessionId"],
                created_at=data["createdAt"],
            )

    @staticmethod
    async def sync_prompts_without_api_key(
        request: SyncPromptsRequest,
    ) -> Dict[str, SyncPromptsResponse]:
        """Call sync_prompts without an API key (for testing auth errors)."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/sync_prompts"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                },
                json={"prompts": request},
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            return response.json()

    @staticmethod
    async def sync_prompts_with_invalid_api_key(
        request: SyncPromptsRequest,
    ) -> Dict[str, SyncPromptsResponse]:
        """Call sync_prompts with an invalid API key (for testing auth errors)."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/sync_prompts"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "x-api-key": "invalid_api_key_12345",
                    "Content-Type": "application/json",
                },
                json={"prompts": request},
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            return response.json()

    @staticmethod
    async def insert_runs_without_api_key(
        request: InsertRunsRequest,
    ) -> InsertRunsResponse:
        """Call insert_runs without an API key (for testing auth errors)."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/insert_runs"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "id": request["id"],
                    "messages": request["messages"],
                    "sessionId": request["sessionId"],
                    "timestamp": request.get("timestamp"),
                },
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            data = response.json()
            return InsertRunsResponse(
                id=data["id"],
                project_id=data["projectId"],
                prompt_id=data["promptId"],
                session_id=data["sessionId"],
                created_at=data["createdAt"],
            )

    @staticmethod
    async def insert_runs_with_invalid_api_key(
        request: InsertRunsRequest,
    ) -> InsertRunsResponse:
        """Call insert_runs with an invalid API key (for testing auth errors)."""
        url = f"{SUPABASE_API_URL}/rest/v1/rpc/insert_runs"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "x-api-key": "invalid_api_key_12345",
                    "Content-Type": "application/json",
                },
                json={
                    "id": request["id"],
                    "messages": request["messages"],
                    "sessionId": request["sessionId"],
                    "timestamp": request.get("timestamp"),
                },
            )

            if not response.is_success:
                error = response.json()
                raise RpcCallError(error)

            data = response.json()
            return InsertRunsResponse(
                id=data["id"],
                project_id=data["projectId"],
                prompt_id=data["promptId"],
                session_id=data["sessionId"],
                created_at=data["createdAt"],
            )
