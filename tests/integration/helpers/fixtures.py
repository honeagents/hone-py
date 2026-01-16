"""
Test fixtures for integration tests.

Exact replica of TypeScript fixtures.ts - provides consistent test data
and cleanup utilities.
"""

import uuid
from dataclasses import dataclass

from .database import TestDatabase, TestProject, TestOrganization


@dataclass
class TestFixture:
    organization: TestOrganization
    project: TestProject


async def create_test_fixture(
    db: TestDatabase,
    prefix: str = "integ",
) -> TestFixture:
    """Create a unique test fixture with organization and project."""
    unique_id = str(uuid.uuid4())[:8]
    org_id = f"{prefix}-org-{unique_id}"
    project_id = f"{prefix}-project-{unique_id}"
    api_key = f"hone_{prefix}_{unique_id}"

    organization = await db.create_organization(
        org_id,
        f"Test Org {unique_id}",
        f"test-org-{unique_id}",
    )

    project = await db.create_project(
        project_id,
        f"Test Project {unique_id}",
        org_id,
        api_key,
    )

    return TestFixture(organization=organization, project=project)


async def cleanup_test_fixture(
    db: TestDatabase,
    fixture: TestFixture,
) -> None:
    """Clean up a test fixture."""
    await db.delete_project(fixture.project.id)
    await db.delete_organization(fixture.organization.id)


def unique_prompt_id(prefix: str = "prompt") -> str:
    """Generate unique prompt IDs for tests."""
    return f"{prefix}-{str(uuid.uuid4())[:8]}"
