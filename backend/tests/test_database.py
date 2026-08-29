from app.database import _ensure_sqlite_parent_dir


def test_ensure_sqlite_parent_dir_creates_directory(tmp_path):
    db_path = tmp_path / "nested" / "clinic.db"
    assert not db_path.parent.exists()
    _ensure_sqlite_parent_dir(f"sqlite:///{db_path.as_posix()}")
    assert db_path.parent.exists()
