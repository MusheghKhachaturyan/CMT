import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from CMT import CMT


ROOT = Path(__file__).resolve().parents[1]


class TrackingTests(unittest.TestCase):
    tracker_backend = 'cpu'

    def setUp(self):
        self.image = np.full((120, 160), 40, dtype=np.uint8)
        texture = np.random.default_rng(7).integers(0, 256, (16, 20), dtype=np.uint8)
        self.image[35:83, 30:90] = cv2.resize(texture, (60, 48), interpolation=cv2.INTER_NEAREST)
        self.tracker = CMT(backend=self.tracker_backend)
        self.tracker.estimate_rotation = False

    def test_tracks_known_translation(self):
        self.tracker.initialise(self.image, (25, 30), (100, 90))
        for dx, dy in ((2, 1), (4, 2), (6, 3)):
            shifted = cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, dy]]), (160, 120), borderValue=40)
            self.tracker.process_frame(shifted)
            self.assertTrue(self.tracker.has_result)
            np.testing.assert_allclose(self.tracker.bb[:2], [25 + dx, 30 + dy], atol=2)

    def test_can_select_entire_frame_without_background_features(self):
        self.tracker.initialise(self.image, (0, 0), (160, 120))
        self.tracker.process_frame(self.image)
        self.assertTrue(self.tracker.has_result)
        np.testing.assert_allclose(self.tracker.bb[:2], [0, 0], atol=1)

    def test_single_keypoint_has_no_scale_estimate(self):
        self.tracker.initialise(self.image, (25, 30), (100, 90))
        center, scale, _, _ = self.tracker.estimate(self.tracker.active_keypoints[:1])
        self.assertTrue(np.isnan(center).all())
        self.assertTrue(np.isnan(scale))

    def test_headless_image_sequence_survives_loss_and_recovers(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            blank = np.full_like(self.image, 40)
            frames = [self.image, blank, blank, blank, self.image, self.image]
            for number, image in enumerate(frames, 1):
                self.assertTrue(cv2.imwrite(str(directory / '{:03d}.png'.format(number)), image))
            result = subprocess.run([
                sys.executable, str(ROOT / 'run.py'), str(directory / '{:03d}.png'),
                '--bbox=25,30,75,60', '--quiet', '--output-dir', str(directory / 'output'),
                '--backend', self.tracker_backend,
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            boxes = sorted((directory / 'output').glob('bbox_*.csv'))
            self.assertEqual(len(boxes), len(frames) - 1)
            self.assertTrue(np.isnan(np.loadtxt(boxes[2], skiprows=1)).all())
            self.assertTrue(np.isfinite(np.loadtxt(boxes[-1], skiprows=1)).all())

    def test_captured_cpu_tracking_reference(self):
        reference = np.load(ROOT / 'tests/fixtures/tracking_reference.npz')
        self.tracker.initialise(self.image, (25, 30), (100, 90))
        for frame, (dx, dy) in enumerate(((2, 1), (4, 2), (6, 3))):
            shifted = cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, dy]]), (160, 120), borderValue=40)
            self.tracker.process_frame(shifted)
            for name in ('active_keypoints', 'tracked_keypoints', 'votes', 'outliers', 'bb', 'center', 'scale_estimate', 'rotation_estimate'):
                actual = np.asarray(getattr(self.tracker, name))
                expected = reference[f'{frame}_{name}']
                if name in ('active_keypoints', 'tracked_keypoints', 'outliers'):
                    np.testing.assert_array_equal(actual[:, 2], expected[:, 2])
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
