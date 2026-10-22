#!/usr/bin/env python3
"""Release gate: a usable CuPy backend and zero skipped tests are mandatory."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from backend import BackendError
from CMT import CMT


def main():
    try:
        tracker=CMT(backend='cupy')
        print(tracker._backend.diagnostics(),flush=True)
    except BackendError as exc:
        print(f'GPU release validation INCOMPLETE: {exc}',file=sys.stderr)
        return 1
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    if result.skipped:
        print('GPU release validation INCOMPLETE: tests were skipped',file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': sys.exit(main())
