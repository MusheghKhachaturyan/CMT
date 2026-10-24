"""CPU oracles and array-level checks; these do not claim CUDA validation."""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import scipy.spatial.distance

from backend import GPUModel, validate_descriptors
from CMT import CMT
from gpu_geometry import pair_pose, workspace_bytes
from gpu_matching import POPCOUNT, hamming_tile, match_candidates, replay_events, tile_shape


class ArrayBackend:
    """Run the shared array expressions with NumPy to check algorithm semantics."""
    cp = np
    device = nullcontext()
    workspace_bytes = 1024 ** 2

    @staticmethod
    def upload(x):
        return np.asarray(x).copy()

    @staticmethod
    def download(x):
        return np.asarray(x).copy()

    @staticmethod
    def available_bytes():
        return 1024 ** 3


def model(backend, descriptors, selected, springs=None, reference=None):
    n = selected
    if springs is None:
        springs = np.zeros((n, 2))
    if reference is None:
        reference = np.arange(n * 2, dtype=float).reshape(n, 2)
    delta = reference[None, :, :] - reference[:, None, :]
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(reference))
    angles = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    classes = np.r_[np.zeros(len(descriptors) - n), np.arange(1, n + 1)].astype(np.int32)
    return GPUModel(*(backend.upload(x) for x in (descriptors, classes, springs, distances, angles)), len(descriptors) - n, n)


def descriptors_with_bits(counts):
    return np.stack([np.packbits(np.r_[np.ones(n, np.uint8), np.zeros(512-n, np.uint8)]) for n in counts])


def _literal_replay(positions, events):
    """Original append/delete loop, kept as the oracle for replay_events."""
    active = np.zeros((0, 3))
    for location, (global_class, structural_class) in zip(positions, events):
        if global_class:
            active = np.append(active, [np.append(location, global_class)], axis=0)
        if structural_class:
            if active.size:
                active = np.delete(active, np.nonzero(active[:, 2] == structural_class), axis=0)
            active = np.append(active, [np.append(location, structural_class)], axis=0)
    return active


class NumericalCases:
    def test_exact_packed_hamming_and_tiles(self):
        rng = np.random.default_rng(8)
        queries = rng.integers(0, 256, (19, 64), dtype=np.uint8)
        database = np.vstack((queries[:2], np.zeros((1, 64), np.uint8), np.full((1, 64), 255, np.uint8), rng.integers(0, 256, (263, 64), dtype=np.uint8)))
        with self.backend.device:
            got = hamming_tile(self.backend.cp, self.backend.upload(queries), self.backend.upload(database), self.backend.upload(POPCOUNT))
            got = self.backend.download(got)
        expected = np.array([[cv2.norm(q, d, cv2.NORM_HAMMING) for d in database] for q in queries], np.uint16)
        np.testing.assert_array_equal(got, expected)
        extremes = np.vstack((np.zeros((1,64),np.uint8), np.full((1,64),255,np.uint8)))
        with self.backend.device:
            values = self.backend.download(hamming_tile(self.backend.cp, self.backend.upload(extremes), self.backend.upload(extremes), self.backend.upload(POPCOUNT)))
        np.testing.assert_array_equal(values, [[0,512],[512,0]])
        m = model(self.backend, database, len(database))
        events = match_candidates(self.backend, m, queries, np.zeros((len(queries),2)), [np.nan,np.nan], None, 20,.75,.8)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(queries, database, k=2)
        expected_classes=[]
        for first, second in matches:
            accepted=second.distance != 0 and first.distance/second.distance < .8 and 1-first.distance/512 > .75
            expected_classes.append(first.trainIdx+1 if accepted else 0)
        np.testing.assert_array_equal(events[:,0],expected_classes)

    def test_matching_thresholds_and_duplicates(self):
        query=np.zeros((1,64),np.uint8)
        for counts, expected in [([128,256],0),([80,100],0),([79,100],1),([0,0],0),([0,512],1),([50,50],0)]:
            with self.subTest(counts=counts):
                db=descriptors_with_bits(counts)
                events=match_candidates(self.backend,model(self.backend,db,2),query,[[0.,0.]],[np.nan,np.nan],None,20,.75,.8)
                self.assertEqual(events[0,0],expected)

    def test_structural_spatial_boundary(self):
        db=descriptors_with_bits([79,100])
        m=model(self.backend,db,2)
        for x, expected in [(19.999,1),(20.,0),(20.001,0)]:
            events=match_candidates(self.backend,m,np.zeros((1,64),np.uint8),[[x,0.]],[0.,0.],[[0.,0.],[100.,0.]],20,.75,.8)
            self.assertEqual(events[0,1],expected)

    def test_background_invalid_pose_and_small_inputs(self):
        db=descriptors_with_bits([0,100,200])
        events=match_candidates(self.backend,model(self.backend,db,2),db[:1],[[0.,0.]],[np.nan,np.nan],None,20,.75,.8)
        np.testing.assert_array_equal(events,[[0,0]])
        for d in (1,2):
            m=model(self.backend,db[:d],1)
            empty=match_candidates(self.backend,m,np.empty((0,64),np.uint8),np.empty((0,2)),[0.,0.],[[0.,0.]],20,.75,.8)
            self.assertEqual(empty.shape,(0,2))
            events=match_candidates(self.backend,m,db[:1],[[0.,0.]],[0.,0.],[[0.,0.]],20,.75,.8)
            self.assertEqual(events[0,1],0)
            if d==1:
                np.testing.assert_array_equal(events,[[0,0]])

    def test_fused_structural_matches_cpu_oracle(self):
        rng=np.random.default_rng(42)
        db=rng.integers(0,256,(269,64),dtype=np.uint8)
        query=db[np.r_[0,2,5,90,268]].copy()
        springs=rng.normal(0,10,(267,2))
        positions=springs[[0,0,3,88,266]]+np.array([7.,9.])
        t=CMT(); t.features_database=db; t.selected_features=db[2:]
        t.database_classes=np.r_[0,0,np.arange(1,268)]; t.selected_classes=np.arange(1,268)
        t.springs=springs; t.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        keypoints=[cv2.KeyPoint(float(x),float(y),1) for x,y in positions]
        positions=np.array([k.pt for k in keypoints])
        expected=t._match_cpu(query,keypoints,np.array([7.,9.]),1.,0.)
        events=match_candidates(self.backend,model(self.backend,db,267,springs),query,positions,[7.,9.],springs,20,.75,.8)
        np.testing.assert_array_equal(replay_events(positions,events),expected)

    def test_geometry_against_captured_cpu(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/geometry_reference.json').read_text())
        ref=np.asarray(fixture['reference_points'])
        m=model(self.backend,np.zeros((len(ref),64),np.uint8),len(ref),ref-ref.mean(axis=0),ref)
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                points=np.asarray(case['keypoints'],dtype=float).reshape(-1,3)
                points=points[np.argsort(points[:,2].astype(int))] if len(points)>1 else points
                pose=pair_pose(self.backend,m,points,case['estimate_scale'],case['estimate_rotation'])
                np.testing.assert_allclose(pose,[case['scale'],case['rotation']],rtol=1e-7,atol=1e-6,equal_nan=True)
                t=CMT(); t.springs=ref-ref.mean(axis=0)
                if not np.isnan(pose[0]):
                    center,inliers=t._vote_and_cluster(points,*pose)
                    np.testing.assert_allclose(center,case['center'],rtol=1e-7,atol=1e-6)
                    np.testing.assert_allclose(inliers,np.array(case['inliers']).reshape(-1,3),rtol=1e-7,atol=1e-6)
                    np.testing.assert_allclose(t.votes,np.array(case['votes']).reshape(-1,2),rtol=1e-7,atol=1e-6)
                    np.testing.assert_array_equal(t.outliers,np.array(case['outliers']).reshape(-1,3))

    def test_zero_reference_distances_and_coincident_points(self):
        ref=np.zeros((2,2))
        m=model(self.backend,np.zeros((2,64),np.uint8),2,reference=ref)
        pose=pair_pose(self.backend,m,np.array([[0.,0.,1.],[0.,0.,2.]]),True,True)
        np.testing.assert_allclose(pose,[1,0])

    def test_rotation_wrap_and_disabled_flags(self):
        ref=np.array([[0.,0.],[10.,0.],[0.,10.],[10.,10.]])
        m=model(self.backend,np.zeros((4,64),np.uint8),4,reference=ref)
        delta=ref[None,:,:]-ref[:,None,:]
        t=CMT();t.springs=ref-ref.mean(axis=0)
        t.squareform=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ref))
        t.angles=np.arctan2(delta[:,:,1],delta[:,:,0])
        for angle in (np.pi,-np.pi,np.pi-1e-9,-np.pi+1e-9,0.3):
            rotated=ref@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
            points=np.column_stack((rotated,[1,2,3,4]))
            for scale,rotation in ((True,True),(True,False),(False,True),(False,False)):
                with self.subTest(angle=angle,scale=scale,rotation=rotation):
                    t.estimate_scale=scale;t.estimate_rotation=rotation
                    _,s,r,_=t.estimate(points)
                    np.testing.assert_allclose(pair_pose(self.backend,m,points,scale,rotation),[s,r],rtol=1e-7,atol=1e-6)

    def test_tile_shape_does_not_change_events(self):
        rng=np.random.default_rng(11)
        db=rng.integers(0,256,(41,64),dtype=np.uint8); db[25]=db[20]
        springs=rng.normal(0,30,(29,2)); center=np.array([5.,7.])
        sources=np.r_[rng.choice(np.arange(12,41),12,replace=False),rng.choice(12,6,replace=False)]
        bits=np.unpackbits(db[sources],axis=1)
        flips=rng.random(bits.shape)<rng.uniform(0,.04,(len(sources),1))
        query=np.vstack((np.packbits(bits^flips,axis=1),rng.integers(0,256,(5,64),dtype=np.uint8)))
        positions=rng.uniform(-60,60,(len(query),2))+center
        positions[:12]=springs[sources[:12]-12]+center+rng.normal(0,12,(12,2))
        m=model(self.backend,db,29,springs)
        expected=match_candidates(self.backend,m,query,positions,center,springs,20,.75,.8)
        self.assertTrue(expected[:,0].any() and expected[:,1].any())
        # Single-column, ragged, unaligned-with-selected-offset and whole-database tiles.
        for shape in ((3,1),(4,8),(7,41),(len(query),41)):
            with self.subTest(shape=shape),patch('gpu_matching.tile_shape',return_value=shape):
                np.testing.assert_array_equal(match_candidates(self.backend,m,query,positions,center,springs,20,.75,.8),expected)

    def test_structural_zero_denominator(self):
        db=descriptors_with_bits([0,0])
        m=model(self.backend,db,2)
        for springs, expected in [([[0.,0.],[0.,0.]],0),([[0.,0.],[100.,0.]],1)]:
            events=match_candidates(self.backend,m,db[:1],[[0.,0.]],[0.,0.],springs,20,.75,.8)
            self.assertEqual(events[0,1],expected)


class ArrayExpressionTests(NumericalCases, unittest.TestCase):
    def setUp(self):
        self.backend=ArrayBackend()


class ReferenceTests(unittest.TestCase):
    def test_original_cpu_fixture(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/geometry_reference.json').read_text())
        ref=np.array(fixture['reference_points']); delta=ref[None,:,:]-ref[:,None,:]
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                t=CMT(); t.springs=ref-ref.mean(axis=0)
                t.squareform=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ref))
                t.angles=np.array([[math.atan2(*(b-a)[::-1]) for b in ref] for a in ref])
                t.estimate_scale=case['estimate_scale']; t.estimate_rotation=case['estimate_rotation']
                center,scale,rotation,inliers=t.estimate(np.array(case['keypoints'],dtype=float).reshape(-1,3))
                np.testing.assert_array_equal(center,case['center'])
                np.testing.assert_allclose([scale,rotation],[case['scale'],case['rotation']],equal_nan=True,rtol=0,atol=0)
                np.testing.assert_array_equal(inliers,np.array(case['inliers']).reshape(-1,3))
                np.testing.assert_array_equal(t.votes,np.array(case['votes']).reshape(-1,2))

    def test_literal_ordered_events(self):
        positions=np.array([[1.,2.],[3.,4.],[5.,6.]])
        for events, expected in [([[1,0],[1,0],[0,0]],[[1,2,1],[3,4,1]]),
                                 ([[1,0],[0,1],[0,0]],[[3,4,1]]),
                                 ([[0,1],[1,0],[0,0]],[[1,2,1],[3,4,1]]),
                                 ([[1,1],[1,2],[0,1]],[[3,4,2],[5,6,1]])]:
            np.testing.assert_array_equal(replay_events(positions,np.array(events)),expected)

    def test_vectorized_replay_matches_literal_loop(self):
        rng=np.random.default_rng(5)
        for q in (0,1,2,17,200):
            for classes in (1,3,40):
                positions=rng.normal(0,100,(q,2))
                events=rng.integers(0,classes+1,(q,2))*(rng.random((q,2))<.7)
                with self.subTest(q=q,classes=classes):
                    np.testing.assert_array_equal(replay_events(positions,events),_literal_replay(positions,events))

    def test_clustering_connectivity_and_tied_largest(self):
        for votes, expected in [([[0,0],[20,0],[40,0],[100,0]],[20,0]),
                                ([[0,0],[10,0],[100,0],[101,0]],[100.5,0])]:
            t=CMT(); t.springs=np.zeros((4,2))
            center,_=t._vote_and_cluster(np.column_stack((votes,np.arange(1,5))),1.,0.)
            np.testing.assert_array_equal(center,expected)

    def test_layout_and_memory_checks(self):
        with self.assertRaises(ValueError): validate_descriptors(np.zeros((2,64),float))
        with self.assertRaises(ValueError): validate_descriptors(np.zeros((2,32),np.uint8))
        self.assertTrue(validate_descriptors(np.zeros((2,128),np.uint8)[:,::2]).flags.c_contiguous)
        self.assertEqual(workspace_bytes(np.array([[0,0,1],[0,0,1]])),0)
        self.assertEqual(workspace_bytes(np.array([[0,0,1],[0,0,2]]),False,False),0)
        self.assertGreater(workspace_bytes(np.array([[0,0,1],[0,0,2]])),0)
        rows,cols=tile_shape(10000,10000,4*1024**2)
        self.assertLessEqual(rows*cols*256+10000*192+65536,4*1024**2)
        # The default budget takes a laptop-sized database in one column tile.
        rows,cols=tile_shape(1650,1573,256*1024**2)
        self.assertEqual(cols,1573)
        self.assertGreater(rows,128)
        self.assertLessEqual(rows*cols*256+1650*192+65536,256*1024**2)


if __name__=='__main__': unittest.main()
