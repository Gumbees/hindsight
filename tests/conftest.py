"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
from dotenv import load_dotenv
from memory import TemporalSemanticMemory
import psycopg2

load_dotenv()


@pytest.fixture(scope="function")
def memory():
    """
    Provide a clean memory system instance for each test.
    """
    mem = TemporalSemanticMemory()
    yield mem
    # Cleanup is handled by individual tests


@pytest.fixture(scope="function")
def clean_agent(memory):
    """
    Provide a clean agent ID and clean up data after test.
    Uses agent_id='test' for all tests (multi-tenant isolation).
    """
    agent_id = "test"

    # Clean up before test
    memory.delete_agent(agent_id)

    yield agent_id

    # Clean up after test
    memory.delete_agent(agent_id)


@pytest.fixture
def db_connection():
    """
    Provide a database connection for direct DB queries in tests.
    """
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    yield conn
    conn.close()
