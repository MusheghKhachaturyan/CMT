"""Optional CuPy startup, model ownership, and explicit host/device transfers."""

from dataclasses import dataclass
import importlib
import sys

import numpy as np


class BackendError(RuntimeError):
    pass


@dataclass
class GPUModel:
    descriptors: object
    classes: object
    springs: object
    distances: object
    angles: object
    selected_offset: int
    selected_count: int


def validate_descriptors(values):
    values = np.asarray(values)
    if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != 64:
        raise ValueError("CuPy backend requires packed BRISK uint8 descriptors of shape (N, 64)")
    return np.ascontiguousarray(values)


class Backend:
    def __init__(self, name, device, memory_mb, profile):
        if name not in ("cpu", "cupy", "auto"):
            raise ValueError("backend must be cpu, cupy, or auto")
        if not isinstance(device, (int, np.integer)) or device < 0:
            raise ValueError("device must be a non-negative integer")
        if not np.isfinite(memory_mb) or memory_mb < 1:
            raise ValueError("gpu_memory_mb must be at least 1 MiB")
        self.requested = name
        self.name = "cpu"
        self.device_id = int(device)
        self.workspace_bytes = int(memory_mb * 1024 ** 2)
        self.profile = profile
        self.cp = None
        self.startup_error = None
        self._geometry_notice = False
        if name != "cpu":
            try:
                self.cp = importlib.import_module("cupy")
                count = self.cp.cuda.runtime.getDeviceCount()
                if device >= count:
                    raise BackendError(f"device {device} unavailable; found {count} CUDA device(s)")
                self.device = self.cp.cuda.Device(device)
                with self.device:
                    self._probe()
                self.name = "cupy"
                profile.synchronize = self.synchronize
            except Exception as exc:
                self.startup_failure(exc)

    def startup_failure(self, exc):
        message = (f"CuPy backend could not initialize device {self.device_id}: {exc}. "
                   "Check GPU exposure, driver/runtime compatibility, and the optional CuPy installation.")
        if self.requested == "cupy":
            raise BackendError(message) from exc
        self.startup_error = message
        self.name = "cpu"
        self.cp = None
        self.profile.synchronize = lambda: None
        print("Backend auto selected CPU: " + message, file=sys.stderr)

    def _probe(self):
        # Exercise the operations actually shipped, including their JIT kernels.
        from gpu_matching import match_candidates
        from gpu_geometry import pair_pose
        descriptors = np.vstack((np.zeros((1, 64), np.uint8), np.full((1, 64), 255, np.uint8)))
        model = self.upload_model(descriptors, np.array([1, 2]), np.array([[0., 0.], [1., 0.]]),
                                  np.array([[0., 1.], [1., 0.]]), np.array([[0., 0.], [np.pi, 0.]]), 2)
        events = match_candidates(self, model, descriptors[:1], np.zeros((1, 2)),
                                  np.zeros(2), np.array([[0., 0.], [1., 0.]]), 20, .75, .8)
        pose = pair_pose(self, model, np.array([[0., 0., 1.], [1., 0., 2.]]), True, True)
        if events.tolist() != [[1, 1]] or not np.allclose(pose, [1., 0.]):
            raise BackendError("matching/geometry numerical startup probe failed")
        self.cp.cuda.get_current_stream().synchronize()

    def synchronize(self):
        if self.name == "cupy":
            with self.device:
                self.cp.cuda.get_current_stream().synchronize()

    def upload(self, values):
        with self.device, self.profile.stage("upload"):
            result = self.cp.asarray(values)
        if self.profile.enabled:
            self.profile.transfers["upload_bytes"] += np.asarray(values).nbytes
        return result

    def download(self, values):
        with self.device, self.profile.stage("download"):
            result = self.cp.asnumpy(values)
        if self.profile.enabled:
            self.profile.transfers["download_bytes"] += values.nbytes
        return result

    def available_bytes(self):
        with self.device:
            free, _ = self.cp.cuda.runtime.memGetInfo()
            return free + self.cp.get_default_memory_pool().free_bytes()

    def upload_model(self, descriptors, classes, springs, distances, angles, selected_count):
        values = [validate_descriptors(descriptors), np.asarray(classes, dtype=np.int32),
                  np.asarray(springs, dtype=np.float64), np.asarray(distances, dtype=np.float64),
                  np.asarray(angles, dtype=np.float64)]
        needed = sum(v.nbytes for v in values)
        if needed + 1024 ** 2 > self.available_bytes():
            raise BackendError(f"GPU model requires {needed} bytes; insufficient available device memory")
        return GPUModel(*(self.upload(v) for v in values), len(descriptors) - selected_count, selected_count)

    def geometry_fits(self, keypoints, scale, rotation):
        from gpu_geometry import workspace_bytes
        required = workspace_bytes(keypoints, scale, rotation)
        fits = required <= min(self.workspace_bytes, max(0, self.available_bytes() - 1024 ** 2))
        if not fits and not self._geometry_notice:
            print(f"CuPy geometry routed to CPU: estimated workspace {required} bytes exceeds available budget", file=sys.stderr)
            self._geometry_notice = True
        return fits

    def memory(self):
        if self.name != "cupy":
            return {"used_bytes": 0, "reserved_bytes": 0}
        with self.device:
            pool = self.cp.get_default_memory_pool()
            return {"used_bytes": pool.used_bytes(), "reserved_bytes": pool.total_bytes()}

    def diagnostics(self):
        result = {"requested": self.requested, "selected": self.name, "workspace_bytes": self.workspace_bytes}
        result["cpu_stages"] = ["BRISK", "Lucas-Kanade", "voting", "SciPy clustering", "I/O"]
        if self.name == "cpu":
            result["cpu_stages"] += ["descriptor matching", "structural scoring", "pair geometry"]
        if self.name == "cupy":
            with self.device:
                properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
                name = properties["name"]
                result.update(device=self.device_id, gpu=name.decode() if isinstance(name, bytes) else name,
                              cupy=self.cp.__version__, driver=self.cp.cuda.runtime.driverGetVersion(),
                              runtime=self.cp.cuda.runtime.runtimeGetVersion(), total_memory=properties["totalGlobalMem"])
        if self.startup_error:
            result["fallback_reason"] = self.startup_error
        return result
