"""
Integration test configuration and fixtures.

Sets up database connection and test fixtures for integration tests.
"""

import pytest
import pytest_asyncio

from .helpers import (
    TestDatabase,
    TestFixture,
    create_test_fixture,
    cleanup_test_fixture,
)


@pytest_asyncio.fixture(scope="module")
async def db():
    """Create database connection for the test module."""
    database = TestDatabase()
    connected = await database.check_connection()
    if not connected:
        pytest.skip(
            "Supabase is not running. Start with: supabase start"
        )
    yield database
    await database.close()


@pytest_asyncio.fixture(scope="module")
async def fixture(db: TestDatabase):
    """Create test fixture (organization and project) for the test module."""
    test_fixture = await create_test_fixture(db, "sdkclient")
    yield test_fixture
    await cleanup_test_fixture(db, test_fixture)


@pytest_asyncio.fixture
async def clean_project(db: TestDatabase, fixture: TestFixture):
    """Clean up project data before each test."""
    await db.cleanup_project(fixture.project.id)
    yield
