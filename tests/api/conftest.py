"""Pytest configuration for API tests - setup test database before any imports."""

import os

# MUST be set BEFORE importing any fastapi/database modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client with SQLite in-memory database."""
    # Now safe to import after DATABASE_URL is set
    from src.api.database import Base, engine, init_db
    from src.api.main import app

    # Initialize database tables
    Base.metadata.create_all(engine)
    init_db()

    return TestClient(app)
