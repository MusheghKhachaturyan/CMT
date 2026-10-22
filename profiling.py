"""Opt-in, synchronized wall-clock stage measurements."""

from contextlib import contextmanager
from time import perf_counter


class Profile:
    def __init__(self):
        self.enabled = False
        self.synchronize = lambda: None
        self.reset()

    def reset(self):
        self.stages = {}
        self.transfers = {"upload_bytes": 0, "download_bytes": 0}
        self.features = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.stages[name] = self.stages.get(name, 0.0) + perf_counter() - start

    def snapshot(self):
        return {"seconds": dict(self.stages), "transfers": dict(self.transfers),
                "features": dict(self.features)}
