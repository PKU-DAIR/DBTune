"""Abstract execution of tasks: submit now, run everything pending at the next wait.

Parallelism is expressed by the caller as *batching*: all tasks submitted before a
`wait()` are handed to the backend together, and a backend may run them concurrently.
Results are cached by task key, so re-running a program skips finished work.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .tasks import EvalResult, TuneResult, TuneTask


class Future:
    def __init__(self, task):
        self.task = task
        self.result = None
        self.done = False


class ResultCache:
    """One JSON file per finished task, keyed by the task's content hash."""

    def __init__(self, directory: str | None):
        self.directory = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def get(self, task):
        path = self._path(task)
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            return _result_from_dict(task, json.load(f)["result"])

    def put(self, task, result) -> None:
        path = self._path(task)
        if not path:
            return
        with open(path, "w") as f:
            json.dump({"task": _task_to_dict(task), "result": result.to_dict()}, f, indent=1)

    def _path(self, task) -> str | None:
        return os.path.join(self.directory, task.key() + ".json") if self.directory else None


class Executor(ABC):
    """submit() queues; wait() runs every pending task through the backend, then returns."""

    def __init__(self, cache: ResultCache | None = None):
        self.cache = cache or ResultCache(None)
        self._pending: dict[str, tuple[object, list[Future]]] = {}
        self._inflight: dict[str, list[Future]] = {}      # key -> futures, while a batch runs
        self._lock = threading.Condition()

    def submit(self, task) -> Future:
        future = Future(task)
        cached = self.cache.get(task)
        if cached is not None:
            _resolve(future, cached)
            return future
        with self._lock:
            if task.key() in self._inflight:                  # already running in another batch: join it
                self._inflight[task.key()].append(future)
            else:
                self._pending.setdefault(task.key(), (task, []))[1].append(future)
        return future

    def wait(self, futures: Iterable[Future]) -> list:
        futures = list(futures)
        while not all(f.done for f in futures):
            self.flush()
            with self._lock:
                if not all(f.done for f in futures) and not self._pending:
                    self._lock.wait(timeout=1.0)   # another thread is running our batch
        return [f.result for f in futures]

    def run_all(self, tasks: Sequence) -> list:
        return self.wait([self.submit(t) for t in tasks])

    def flush(self) -> None:
        """Run everything pending (as one batch) and resolve its futures."""
        with self._lock:
            batch = list(self._pending.items())
            self._pending = {}
            for key, (task, futures) in batch:
                self._inflight[key] = futures
        if not batch:
            return
        results = self._run([task for _, (task, _) in batch])
        with self._lock:
            for (key, (task, _)), result in zip(batch, results):
                self.cache.put(task, result)
                for f in self._inflight.pop(key):
                    _resolve(f, result)
            self._lock.notify_all()

    @abstractmethod
    def _run(self, tasks: list) -> list:
        """Execute the batch (possibly in parallel); return results in the same order."""


def _resolve(future: Future, result) -> None:
    future.result, future.done = result, True


def _task_to_dict(task) -> dict:
    d = {"kind": task.kind, "workload": task.workload.as_dict()}
    if isinstance(task, TuneTask):
        d["budget"] = task.budget.__dict__
    else:
        d["configs"] = [dict(c) for c in task.configs]
    return d


def _result_from_dict(task, d):
    return TuneResult.from_dict(d) if isinstance(task, TuneTask) else EvalResult.from_dict(d)
