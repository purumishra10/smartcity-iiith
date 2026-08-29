from pathlib import Path

import pytest

from app.config import settings
from app.database import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure storage directory exists and database is initialized before any tests run."""
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    yield
