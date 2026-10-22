import contextlib
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from backend import Backend, BackendError
from CMT import CMT
from profiling import Profile

ROOT=Path(__file__).resolve().parents[1]


class BackendTests(unittest.TestCase):
    def test_optional_dispatch_with_numpy_device_adapter(self):
        # Integration check only. This adapter does not execute CUDA kernels.
        import cv2
        from contextlib import nullcontext

        class NumPyDevice:
            __version__='test-adapter'
            cuda=SimpleNamespace(
                Device=lambda _:nullcontext(),
                get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None),
                runtime=SimpleNamespace(getDeviceCount=lambda:1,memGetInfo=lambda:(2**30,2**30)))

            def __getattr__(self,name):
                if name=='asnumpy':return np.asarray
                if name=='get_default_memory_pool':
                    return lambda:SimpleNamespace(free_bytes=lambda:0)
                return getattr(np,name)

        with patch('backend.importlib.import_module',return_value=NumPyDevice()):
            tracker=CMT(backend='cupy')
        image=np.full((120,160),40,np.uint8)
        texture=np.random.default_rng(7).integers(0,256,(16,20),dtype=np.uint8)
        image[35:83,30:90]=cv2.resize(texture,(60,48),interpolation=cv2.INTER_NEAREST)
        tracker.estimate_rotation=False
        tracker.initialise(image,(25,30),(100,90))
        with np.load(ROOT/'tests/fixtures/tracking_reference.npz') as expected:
            for frame,(dx,dy) in enumerate(((2,1),(4,2),(6,3))):
                shifted=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(160,120),borderValue=40)
                tracker.process_frame(shifted)
                for name in ('active_keypoints','tracked_keypoints','votes','outliers','bb','center'):
                    np.testing.assert_allclose(getattr(tracker,name),expected[f'{frame}_{name}'],rtol=1e-7,atol=1e-6)
        # Reinitialization cannot retain the old GPU model.
        previous=tracker._gpu_model
        tracker.initialise(image,(0,0),(160,120))
        self.assertIsNot(previous,tracker._gpu_model)
        tracker._backend.requested='auto'
        with patch.object(tracker._backend,'upload_model',side_effect=RuntimeError('model allocation failed')):
            with contextlib.redirect_stderr(io.StringIO()):
                tracker.initialise(image,(25,30),(100,90))
        self.assertEqual(tracker.backend,'cpu')
        self.assertIsNone(tracker._gpu_model)

    def test_cpu_never_imports_cupy(self):
        with patch('backend.importlib.import_module',side_effect=AssertionError('unexpected import')):
            self.assertEqual(CMT().backend,'cpu')

    def test_missing_cupy_explicit_and_auto(self):
        with patch('backend.importlib.import_module',side_effect=ImportError('not installed')):
            with self.assertRaisesRegex(BackendError,'could not initialize.*not installed'):
                CMT(backend='cupy')
            with contextlib.redirect_stderr(io.StringIO()) as output:
                tracker=CMT(backend='auto')
            self.assertEqual(tracker.backend,'cpu')
            self.assertIn('auto selected CPU',output.getvalue())

    def test_invalid_device(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda:1)))
        with patch('backend.importlib.import_module',return_value=fake):
            with self.assertRaisesRegex(BackendError,'device 1 unavailable'):
                CMT(backend='cupy',device=1)
        for kwargs in ({'backend':'invalid'},{'device':-1},{'gpu_memory_mb':0},{'gpu_memory_mb':np.nan}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): CMT(**kwargs)

    def test_cpu_cli_and_actionable_cupy_error(self):
        help_result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(help_result.returncode,0)
        self.assertIn('--gpu-memory-mb',help_result.stdout)
        result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--backend=cupy','--device=-1'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('non-negative',result.stderr)

    def test_memory_route_uses_original_estimator(self):
        t=CMT();t.springs=np.array([[-1.,0.],[1.,0.]])
        t.squareform=np.array([[0.,2.],[2.,0.]]);t.angles=np.array([[0.,0.],[np.pi,0.]])
        points=np.array([[2.,3.,1.],[4.,3.,2.]])
        expected=t.estimate(points)
        t._backend=SimpleNamespace(name='cupy',geometry_fits=lambda *args:False)
        actual=t.estimate(points)
        self.assertEqual(t.geometry_route,'cpu-memory')
        for x,y in zip(expected,actual): np.testing.assert_array_equal(x,y)

    def test_runtime_failure_propagates(self):
        t=CMT(); t._backend=SimpleNamespace(name='cupy',geometry_fits=lambda *args:True)
        t._gpu_model=object()
        with patch('gpu_geometry.pair_pose',side_effect=RuntimeError('CUDA failure')):
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):
                t.estimate(np.array([[0.,0.,1.],[1.,0.,2.]]))
        self.assertEqual(t.backend,'cupy')

    def test_disabled_profiling_has_no_synchronization(self):
        profile=Profile();profile.synchronize=lambda: self.fail('unexpected synchronization')
        with profile.stage('test'): pass
        self.assertEqual(profile.stages,{})

    def test_duplicate_unmatched_tracked_rows_survive(self):
        points=np.array([[1.,2.,1.],[3.,4.,1.]])
        tracker=CMT()
        tracker.im_prev=np.zeros((20,20),np.uint8)
        tracker.active_keypoints=points
        tracker.num_initial_keypoints=2
        tracker.features_database=np.zeros((2,64),np.uint8)
        tracker.springs=np.zeros((2,2))
        tracker.detector=SimpleNamespace(detect=lambda image:[])
        tracker.descriptor=SimpleNamespace(compute=lambda image,keypoints:([],None))
        with patch('util.track',return_value=(points,None)),patch.object(tracker,'estimate',return_value=(np.array([np.nan,np.nan]),np.nan,np.nan,points)):
            tracker.process_frame(tracker.im_prev)
        np.testing.assert_array_equal(tracker.active_keypoints,points)


if __name__=='__main__':unittest.main()
