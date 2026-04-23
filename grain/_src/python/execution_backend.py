# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ExecutionBackend layer for cross-platform process/thread management."""

import abc
import multiprocessing as mp
from multiprocessing import queues
from multiprocessing import sharedctypes
from multiprocessing import synchronize
import os
import platform
import queue
import threading
from typing import Any, Callable

class ExecutionBackend(abc.ABC):
    """Abstract mapping for concurrency primitives."""

    @abc.abstractmethod
    def Queue(self, maxsize: int = 0) -> "queue.Queue[Any]":
        """Returns a Queue instance."""

    @abc.abstractmethod
    def Event(self) -> synchronize.Event | threading.Event:
        """Returns an Event instance."""

    @abc.abstractmethod
    def SynchronizedInt(self, initial_value: int) -> Any:
        """Returns a synchronized integer with .value and .get_lock()."""

    @abc.abstractmethod
    def Process(self, target: Callable, kwargs: dict, daemon: bool, name: str) -> Any:
        """Returns a Process or Thread instance."""

    @abc.abstractmethod
    def is_multiprocess(self) -> bool:
        """Returns True if this backend runs tasks in separate processes."""


class MultiprocessingBackend(ExecutionBackend):
    """Execution backend utilizing Linux-optimized multiprocessing 'spawn'."""

    def __init__(self):
        start_method = os.environ.get("GRAIN_MP_START", "fork")
        if start_method not in ["fork", "spawn", "forkserver"]:
            raise ValueError(f"Invalid GRAIN_MP_START: {start_method}")
        try:
            self._ctx = mp.get_context(start_method)
        except ValueError:
            self._ctx = mp.get_context("spawn")

    def Queue(self, maxsize: int = 0) -> "queue.Queue[Any]":
        return self._ctx.Queue(maxsize=maxsize)

    def Event(self) -> synchronize.Event:
        return self._ctx.Event()

    def SynchronizedInt(self, initial_value: int) -> Any:
        return self._ctx.Value("i", initial_value)

    def Process(self, target: Callable, kwargs: dict, daemon: bool, name: str):
        return self._ctx.Process(target=target, kwargs=kwargs, daemon=daemon, name=name)

    def is_multiprocess(self) -> bool:
        return True


class _ThreadSynchronizedInt:
    def __init__(self, initial_value: int):
        self._value = initial_value
        self._lock = threading.Lock()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def get_lock(self):
        return self._lock


class ThreadingBackend(ExecutionBackend):
    """Execution backend utilizing threading for platforms lacking solid fork support or shared_memory (Windows/macOS)."""

    def Queue(self, maxsize: int = 0) -> "queue.Queue[Any]":
        return queue.Queue(maxsize=maxsize)

    def Event(self) -> threading.Event:
        return threading.Event()

    def SynchronizedInt(self, initial_value: int) -> _ThreadSynchronizedInt:
        return _ThreadSynchronizedInt(initial_value)

    def Process(self, target: Callable, kwargs: dict, daemon: bool, name: str) -> threading.Thread:
        return threading.Thread(target=target, kwargs=kwargs, daemon=daemon, name=name)

    def is_multiprocess(self) -> bool:
        return False


def get_execution_backend() -> ExecutionBackend:
    """Returns the optimal ExecutionBackend based on the platform and environment."""
    env_backend = os.environ.get("GRAIN_EXECUTION_BACKEND", "").lower()
    if env_backend and env_backend not in ["multiprocessing", "threading"]:
        raise ValueError(f"Invalid GRAIN_EXECUTION_BACKEND: {env_backend}")
    if env_backend == "multiprocessing":
        return MultiprocessingBackend()
    elif env_backend == "threading":
        return ThreadingBackend()

    if platform.system() == "Linux":
        return MultiprocessingBackend()
    else:
        return ThreadingBackend()
