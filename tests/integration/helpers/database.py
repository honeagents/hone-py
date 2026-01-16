"""
Database helper for integration tests.

Exact replica of TypeScript database.ts - provides direct database access for:
- Creating test fixtures (organizations, projects)
- Verifying database state after RPC calls
- Cleaning up test data
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import httpx

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
)

SUPABASE_API_URL = os.environ.get("SUPABASE_API_URL", "http://127.0.0.1:54321")
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9."
    "CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
)
# Service role key for admin operations (bypasses RLS)
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0."
    "EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
)


@dataclass
class TestProject:
    id: str
    name: str
    api_key: str
    organization_id: str


@dataclass
class TestOrganization:
    id: str
    name: str
    slug: str


@dataclass
class Prompt:
    id: str
    project_id: str
    name: Optional[str]
    version: Optional[str]
    param_keys: List[str]
    text: str


@dataclass
class PromptHierarchy:
    id: str
    project_id: str
    parent_prompt_id: str
    child_prompt_id: str


class Message(TypedDict):
    role: str
    content: str


@dataclass
class Run:
    id: str
    project_id: str
    prompt_id: str
    session_id: str
    messages: List[Message]
    created_at: str


@dataclass
class Session:
    session_id: str
    project_id: str
    prompt_id: str
    run_id: str
    updated_at: str


class TestDatabase:
    """Database helper for integration tests."""

    def __init__(self):
        self._connected = False

    async def check_connection(self) -> bool:
        """Check if the database is accessible."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SUPABASE_API_URL}/rest/v1/",
                    headers={
                        "apikey": SUPABASE_ANON_KEY,
                        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    },
                )
                self._connected = response.is_success
                return self._connected
        except Exception:
            return False

    async def patch(
        self,
        table: str,
        data: Dict[str, Any],
        params: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Patch database rows.

        Args:
            table: The table name
            data: The data to update
            params: The query parameters to select rows
        """
        # PostgREST requires filter operators like eq., gt., etc.
        query_params = {key: f"eq.{value}" for key, value in params.items()}
        url = f"{SUPABASE_API_URL}/rest/v1/{table}"

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                url,
                params=query_params,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                json=data,
            )

            if not response.is_success:
                raise Exception(f"Patch failed: {response.text}")

            return response.json()

    async def _query(
        self,
        table: str,
        params: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a query using Supabase's REST API (for reads)."""
        url = f"{SUPABASE_API_URL}/rest/v1/{table}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params or {},
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
            )

            if not response.is_success:
                raise Exception(f"Query failed: {response.text}")

            return response.json()

    async def _insert(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Insert a row using Supabase's REST API."""
        url = f"{SUPABASE_API_URL}/rest/v1/{table}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                json=data,
            )

            if not response.is_success:
                raise Exception(f"Insert failed: {response.text}")

            result = response.json()
            return result[0]

    async def _delete(
        self,
        table: str,
        params: Dict[str, str],
    ) -> None:
        """Delete rows using Supabase's REST API."""
        url = f"{SUPABASE_API_URL}/rest/v1/{table}"

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                url,
                params=params,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )

            if not response.is_success:
                raise Exception(f"Delete failed: {response.text}")

    async def create_organization(
        self,
        id: str,
        name: str,
        slug: str,
    ) -> TestOrganization:
        """Create a test organization."""
        result = await self._insert("organizations", {
            "id": id,
            "name": name,
            "slug": slug,
        })
        return TestOrganization(
            id=result["id"],
            name=result["name"],
            slug=result["slug"],
        )

    async def create_project(
        self,
        id: str,
        name: str,
        organization_id: str,
        api_key: str,
    ) -> TestProject:
        """Create a test project with a known API key."""
        result = await self._insert("projects", {
            "id": id,
            "name": name,
            "organization_id": organization_id,
            "api_key": api_key,
        })
        return TestProject(
            id=result["id"],
            name=result["name"],
            api_key=result["api_key"],
            organization_id=result["organization_id"],
        )

    async def get_prompts(self, project_id: str) -> List[Prompt]:
        """Get all prompts for a project."""
        results = await self._query("prompts", {
            "project_id": f"eq.{project_id}",
        })

        return [
            Prompt(
                id=r["id"],
                project_id=r["project_id"],
                name=r.get("name"),
                version=r.get("version"),
                param_keys=r.get("param_keys", []),
                text=r["text"],
            )
            for r in results
        ]

    async def get_prompt(
        self,
        project_id: str,
        prompt_id: str,
    ) -> Optional[Prompt]:
        """Get a specific prompt by ID."""
        results = await self._query("prompts", {
            "project_id": f"eq.{project_id}",
            "id": f"eq.{prompt_id}",
        })

        if not results:
            return None

        r = results[0]
        return Prompt(
            id=r["id"],
            project_id=r["project_id"],
            name=r.get("name"),
            version=r.get("version"),
            param_keys=r.get("param_keys", []),
            text=r["text"],
        )

    async def update_prompt_text(
        self,
        project_id: str,
        prompt_id: str,
        new_text: str,
    ) -> None:
        """Updates a prompt's text directly in the database."""
        await self.patch(
            "prompts",
            {"text": new_text},
            {
                "project_id": project_id,
                "id": prompt_id,
            },
        )

    async def get_hierarchy(self, project_id: str) -> List[PromptHierarchy]:
        """Get all prompt hierarchy entries for a project."""
        results = await self._query("prompt_hierarchy", {
            "project_id": f"eq.{project_id}",
        })

        return [
            PromptHierarchy(
                id=r["id"],
                project_id=r["project_id"],
                parent_prompt_id=r["parent_prompt_id"],
                child_prompt_id=r["child_prompt_id"],
            )
            for r in results
        ]

    async def create_run(
        self,
        id: str,
        project_id: str,
        prompt_id: str,
        session_id: str,
        messages: List[Message],
    ) -> Run:
        """Create a run in the database."""
        result = await self._insert("runs", {
            "id": id,
            "project_id": project_id,
            "prompt_id": prompt_id,
            "session_id": session_id,
            "messages": messages,
        })

        return Run(
            id=result["id"],
            project_id=result["project_id"],
            prompt_id=result["prompt_id"],
            session_id=result["session_id"],
            messages=result["messages"],
            created_at=result["created_at"],
        )

    async def get_runs(self, project_id: str) -> List[Run]:
        """Get all runs for a project."""
        results = await self._query("runs", {
            "project_id": f"eq.{project_id}",
            "order": "created_at.asc",
        })

        return [
            Run(
                id=r["id"],
                project_id=r["project_id"],
                prompt_id=r["prompt_id"],
                session_id=r["session_id"],
                messages=r["messages"],
                created_at=r["created_at"],
            )
            for r in results
        ]

    async def get_run(self, run_id: str) -> Optional[Run]:
        """Get a specific run by ID."""
        results = await self._query("runs", {
            "id": f"eq.{run_id}",
        })

        if not results:
            return None

        r = results[0]
        return Run(
            id=r["id"],
            project_id=r["project_id"],
            prompt_id=r["prompt_id"],
            session_id=r["session_id"],
            messages=r["messages"],
            created_at=r["created_at"],
        )

    async def get_sessions(self, project_id: str) -> List[Session]:
        """Get all sessions for a project."""
        results = await self._query("sessions", {
            "project_id": f"eq.{project_id}",
            "order": "updated_at.asc",
        })

        return [
            Session(
                session_id=r["session_id"],
                project_id=r["project_id"],
                prompt_id=r["prompt_id"],
                run_id=r["run_id"],
                updated_at=r["updated_at"],
            )
            for r in results
        ]

    async def get_session(
        self,
        project_id: str,
        session_id: str,
    ) -> Optional[Session]:
        """Get a specific session by session_id and project_id."""
        results = await self._query("sessions", {
            "project_id": f"eq.{project_id}",
            "session_id": f"eq.{session_id}",
        })

        if not results:
            return None

        r = results[0]
        return Session(
            session_id=r["session_id"],
            project_id=r["project_id"],
            prompt_id=r["prompt_id"],
            run_id=r["run_id"],
            updated_at=r["updated_at"],
        )

    async def cleanup_project(self, project_id: str) -> None:
        """Delete all prompts, hierarchy, runs, and sessions for a project."""
        # Delete sessions first (foreign key to runs)
        await self._delete("sessions", {"project_id": f"eq.{project_id}"})
        # Delete runs (foreign key to prompts)
        await self._delete("runs", {"project_id": f"eq.{project_id}"})
        # Delete hierarchy (foreign key to prompts)
        await self._delete("prompt_hierarchy", {"project_id": f"eq.{project_id}"})
        # Delete archived prompts
        await self._delete("archived_prompts", {"project_id": f"eq.{project_id}"})
        # Delete prompts
        await self._delete("prompts", {"project_id": f"eq.{project_id}"})

    async def delete_project(self, project_id: str) -> None:
        """Delete a project and its organization."""
        await self.cleanup_project(project_id)
        await self._delete("projects", {"id": f"eq.{project_id}"})

    async def delete_organization(self, organization_id: str) -> None:
        """Delete an organization."""
        await self._delete("organizations", {"id": f"eq.{organization_id}"})

    async def close(self) -> None:
        """Close database connection."""
        self._connected = False
