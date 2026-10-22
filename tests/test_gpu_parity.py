"""Actual CUDA checks: skipped locally without GPU; mandatory in release runner."""
from contextlib import redirect_stderr
import io
import unittest

import numpy as np
from backend import BackendError
from CMT import CMT
from test_numerics import NumericalCases
from test_tracking import TrackingTests as _TrackingTests


def gpu_or_skip():
    try:
        return CMT(backend='cupy')
    except BackendError as exc:
        raise unittest.SkipTest(f'CUDA validation unavailable: {exc}')


class GPUNumericalTests(NumericalCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker=gpu_or_skip()
        cls.backend=cls.tracker._backend

    def test_long_run_memory_reuse(self):
        from test_numerics import model
        from gpu_matching import match_candidates
        from gpu_geometry import pair_pose
        rng=np.random.default_rng(93)
        db=rng.integers(0,256,(300,64),dtype=np.uint8)
        m=model(self.backend,db,300)
        samples=[]
        for i in range(80):
            q=(17,70,129,256)[i%4]
            match_candidates(self.backend,m,db[:q],np.zeros((q,2)),[0.,0.],np.zeros((300,2)),20,.75,.8)
            pair_pose(self.backend,m,np.column_stack((np.arange(q),np.zeros(q),np.arange(1,q+1))),True,True)
            self.backend.synchronize()
            samples.append(self.backend.memory()['reserved_bytes'])
        self.assertLessEqual(max(samples[-20:]),max(samples[20:40])+2*1024**2)


class GPUTrackingTests(_TrackingTests):
    tracker_backend='cupy'

    @classmethod
    def setUpClass(cls):
        gpu_or_skip()

    def test_reinitialize_replaces_model(self):
        self.tracker.initialise(self.image,(25,30),(100,90))
        old=self.tracker._gpu_model
        self.tracker.initialise(self.image,(0,0),(160,120))
        self.assertIsNot(self.tracker._gpu_model,old)
        self.assertEqual(self.tracker._gpu_model.selected_count,self.tracker.num_initial_keypoints)
        self.tracker.process_frame(self.image)
        self.assertTrue(self.tracker.has_result)

    def test_forced_memory_routing_matches_cpu(self):
        cpu=CMT();cpu.estimate_rotation=False
        cpu.initialise(self.image,(25,30),(100,90))
        self.tracker.initialise(self.image,(25,30),(100,90))
        self.tracker._backend.workspace_bytes=1
        points=cpu.active_keypoints
        expected=cpu.estimate(points)
        with redirect_stderr(io.StringIO()): actual=self.tracker.estimate(points)
        self.assertEqual(self.tracker.geometry_route,'cpu-memory')
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_sequence_decisions_match_cpu(self):
        import cv2
        cpu=CMT();cpu.estimate_rotation=False
        for tracker in (cpu,self.tracker):tracker.initialise(self.image,(25,30),(100,90))
        blank=np.full_like(self.image,40)
        frames=[cv2.warpAffine(self.image,np.float32([[1,0,2],[0,1,1]]),(160,120),borderValue=40),blank,blank,self.image,self.image]
        for image in frames:
            cpu.process_frame(image);self.tracker.process_frame(image)
            self.assertEqual(self.tracker.has_result,cpu.has_result)
            for name in ('active_keypoints','tracked_keypoints','outliers'):
                actual=getattr(self.tracker,name);expected=getattr(cpu,name)
                np.testing.assert_array_equal(actual[:,2],expected[:,2])
                np.testing.assert_allclose(actual[:,:2],expected[:,:2],rtol=1e-7,atol=1e-6)
            np.testing.assert_array_equal(np.isnan(self.tracker.bb),np.isnan(cpu.bb))
            np.testing.assert_allclose(self.tracker.center,cpu.center,rtol=1e-7,atol=1e-6)


del _TrackingTests


if __name__=='__main__':unittest.main()
