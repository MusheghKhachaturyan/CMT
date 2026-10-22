import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class CLIIntegrationTests(unittest.TestCase):
    def test_benchmark_parity_decisions_and_truncation(self):
        import copy
        from benchmarks.benchmark_tracking import sequence_parity
        frame=dict(has_result=True,active=2,class_signature='same',center=[1.,2.],
                   scale=1.,rotation=0.,corners_float=[[1.,2.]],corners=[[1,2]])
        original=[dict(results=[frame])]
        self.assertEqual(sequence_parity(original,copy.deepcopy(original))['status'],'passed')
        changed=copy.deepcopy(original)
        changed[0]['results'][0]['class_signature']='different'
        self.assertEqual(sequence_parity(original,changed)['status'],'failed')
        changed=copy.deepcopy(original)
        changed[0]['results'][0]['corners']=[[0,2]]
        changed[0]['results'][0]['corners_float']=[[1.-1e-8,2.]]
        self.assertEqual(sequence_parity(original,changed)['status'],'passed')
        changed[0]['results'][0]['corners_float']=[[0.5,2.]]
        self.assertEqual(sequence_parity(original,changed)['status'],'failed')

    def test_challenge_mode_and_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            image=np.full((120,160),40,np.uint8)
            texture=np.random.default_rng(7).integers(0,256,(16,20),dtype=np.uint8)
            image[35:83,30:90]=cv2.resize(texture,(60,48),interpolation=cv2.INTER_NEAREST)
            files=[]
            for i in range(3):
                path=directory/f'{i}.png';self.assertTrue(cv2.imwrite(str(path),image));files.append(str(path))
            (directory/'images.txt').write_text('\n'.join(files)+'\n')
            (directory/'region.txt').write_text('25,30,75,60\n')
            result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--challenge','--quiet','--profile'],cwd=directory,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(np.loadtxt(directory/'output.txt',delimiter=',').shape,(3,4))
            profiles=[json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
            self.assertEqual([p['frame'] for p in profiles],[0,1,2])
            self.assertIn('pair_geometry',profiles[-1]['profile']['seconds'])

    def test_benchmark_smoke_report_and_missing_input(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'result.json'
            result=subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_tracking.py'),str(ROOT/'tests/fixtures/does-not-exist.avi'),'--bbox=0,0,10,10','--backends=cpu','--repeats=1','--output',str(output)],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['backends']['cpu']['status'],'failed')


if __name__=='__main__':unittest.main()
