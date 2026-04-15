"""Pytest shared configuration for import paths."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest - force test database URL before any imports."""
    # MUST set this BEFORE any config/database imports
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    # Register custom pytest markers
    config.addinivalue_line("markers", "streamlit_integration: mark test as a Streamlit integration test (requires running app)")
