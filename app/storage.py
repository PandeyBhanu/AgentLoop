"""Persistent, bounded SQLite run records for a single API worker."""
import json
import os
import sqlite3
import threading
import time
from collections.abc import MutableMapping, Iterator
from pathlib import Path
from typing import Any


class RunStore(MutableMapping[str, dict]):
    def __init__(self, path: str | None = None, max_runs: int = 1000):
        path = path or os.getenv("AGENTLOOP_DB_PATH", ".agentloop/runs.sqlite3")
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self._namespace = "default"
        self.max_runs = max_runs
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (namespace TEXT NOT NULL, run_id TEXT NOT NULL, "
                "owner_id TEXT NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL, "
                "PRIMARY KEY (namespace, run_id))"
            )
            self._connection.execute("CREATE INDEX IF NOT EXISTS runs_owner ON runs(namespace, owner_id)")

    def use_namespace(self, namespace: str) -> None:
        self._namespace = namespace

    def __getitem__(self, key: str) -> dict:
        with self._lock:
            row = self._connection.execute(
                "SELECT data FROM runs WHERE namespace=? AND run_id=?", (self._namespace, key)
            ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key: str, value: dict) -> None:
        if not isinstance(value, dict) or not value.get("owner_id"):
            raise ValueError("Run owner is required")
        with self._lock, self._connection:
            count = self._connection.execute(
                "SELECT COUNT(*) FROM runs WHERE namespace=?", (self._namespace,)
            ).fetchone()[0]
            exists = self._connection.execute(
                "SELECT 1 FROM runs WHERE namespace=? AND run_id=?", (self._namespace, key)
            ).fetchone()
            if not exists and count >= self.max_runs:
                raise OverflowError("Run retention capacity reached")
            self._connection.execute(
                "INSERT INTO runs(namespace,run_id,owner_id,updated,data) VALUES(?,?,?,?,?) "
                "ON CONFLICT(namespace,run_id) DO UPDATE SET updated=excluded.updated,data=excluded.data",
                (self._namespace, key, value["owner_id"], time.time(), json.dumps(value, default=str)),
            )

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("Run records are not deleted through the API")

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id FROM runs WHERE namespace=?", (self._namespace,)
            ).fetchall()
        return iter(row[0] for row in rows)

    def __len__(self) -> int:
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM runs WHERE namespace=?", (self._namespace,)
            ).fetchone()[0]

    def mark_interrupted_runs(self) -> int:
        interrupted = 0
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, data FROM runs WHERE namespace=?", (self._namespace,)
            ).fetchall()
        for run_id, raw in rows:
            value = json.loads(raw)
            if value.get("status") != "running":
                continue
            value.update({
                "status": "interrupted",
                "error": "Process stopped before the run completed",
                "termination_reason": "process_restart",
            })
            self[run_id] = value
            interrupted += 1
        return interrupted

    def close(self) -> None:
        self._connection.close()


runs: RunStore = RunStore()
