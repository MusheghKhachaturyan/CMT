# Validation status

The hybrid CuPy backend is experimental. No CUDA correctness or acceleration
claim is supported by the current environment.

The ordinary test suite passes 30 CPU/reference tests. Two CUDA suites skip
because CuPy is unavailable. `python tests/run_gpu_validation.py` correctly
returns a failure with `GPU release validation INCOMPLETE` in that situation.
The tests include original CPU geometry/tracking fixtures, packed Hamming
oracles, structural boundaries, ordered events, angular/median cases, optional
backend dispatch with a NumPy adapter, CLI/challenge output, and memory routing.
The NumPy adapter is not a CUDA test.

The supplied environment is Python 3.12.3, OpenCV 4.14.0, NumPy 2.5.4, SciPy
1.18.1, and WSL2. `nvidia-smi` reports GPU access blocked by the operating system.
An elevated visibility check timed out in automatic approval. Dependency
resolution of `requirements-gpu.txt` also did not succeed under the supplied
package source, so a tested GPU dependency lock is intentionally absent.

Benchmark reports are generated under ignored `benchmark-results/`. Both
backends are requested: completed CPU results are retained, unavailable CuPy
runs are marked failed, and the overall command exits nonzero. Measurements use
12 OpenCV threads, disposable warmup trackers, fresh measured trackers, three
repeats, and a separate 30-frame instrumented stage sample. Latency/throughput
passes run without stage instrumentation. These local diagnostic measurements
were collected during implementation and are not GPU performance evidence.

The full synthetic demo completed three CPU runs of 60 tracking frames each,
including shared runner PNG/CSV output. Tracker throughput was 1.726, 1.732, and
1.836 FPS (median 1.732); application throughput was 1.702, 1.707, and 1.813 FPS.

The phone-video benchmark uses `demo/IMG_3542.MOV`, the entire laptop selection
`0,520,325,238`, and rotation enabled. All three CPU runs processed 388 tracking
frames, at 0.710, 0.729, and 0.690 tracker FPS (median 0.710). The first run had
median latency 1.385 seconds and p95 1.883 seconds. All frames in that run
produced a result. Full repeat measurements and the separate stage sample are
recorded in `benchmark-results/laptop.json`.

Before changing the support status, run the zero-skip CUDA validation command
and both complete benchmark commands from README in the same resolved GPU
environment. Check sequence parity, per-run speedup, geometry routes, and the
reserved-memory plateau. In the current conservative workspace preflight, the
256 MiB default can route the phone video's largest geometry stages to CPU;
512 MiB can permit those stages when the actual device has sufficient memory.
