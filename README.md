# Introduction
CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is
a novel keypoint-based method for long-term model-free object tracking in a
combined matching-and-tracking framework.
Details can be found on the [project page](http://www.gnebehay.com/cmt)
and in our [publication](http://www.gnebehay.com/publications/wacv_2014/wacv_2014.pdf).
The Python implementation in this repository is platform-independent and runs
on Linux, Windows and OS X.

# License
CMT is freely available under the [3-clause BSD license][1],
meaning that you can basically do with the code whatever you want.
If you use our algorithm in scientific work, please cite our publication
```
@inproceedings{Nebehay2015CVPR,
    author = {Nebehay, Georg and Pflugfelder, Roman},
    booktitle = {Computer Vision and Pattern Recognition},
    month = jun,
    publisher = {IEEE},
    title = {Clustering of {Static-Adaptive} Correspondences for Deformable Object Tracking},
    year = {2015}
}
```

# Dependencies
* Python 3.10 or newer (tested with Python 3.12)
* OpenCV-Python 4
* NumPy
* SciPy
* optional: ipdb (for debugging the code)

## Setup and quick start (Linux / WSL)

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Create and track a sample video without a camera or display:

```bash
python make_demo.py
python run.py demo/input.avi --bbox=50,50,100,80 --quiet --output-dir demo/output
```

The sample contains 61 frames. The first frame initializes the tracker and the
remaining 60 frames are tracked. Annotated frames are written to
`demo/output/output_00000001.png` through `output_00000060.png`, alongside the
input images and CSV files for keypoints, outliers, votes, and bounding boxes.
The sample is synthetic; it checks that the pipeline works and is not a benchmark
of tracking quality on real footage.

For your own video, supply the initial object's bounding box in pixels:

```bash
python run.py /absolute/path/to/video.mp4 --bbox=123,85,60,140
```

The box format is `x,y,width,height`, measured from the top left of the first
frame. With a working desktop display you can omit `--bbox` and select two
corners with the mouse. Press `q` to quit the tracking window. For a terminal
without a display, use both `--quiet` and `--bbox`:

```bash
python run.py /absolute/path/to/video.mp4 --bbox=123,85,60,140 --quiet --output-dir output
```

For a webcam, run `python run.py`. Press any key to stop the preview, then select
the object. The camera must be accessible to Linux; WSL may require camera
passthrough. A video file works without a camera.

## CPU / GPU

CPU tracking remains the default and does not import CuPy. The optional hybrid
CuPy backend accelerates packed BRISK descriptor matching, structural scoring,
and ordered-pair scale/rotation estimation. BRISK detection/description,
Lucas–Kanade optical flow, voting, SciPy clustering, and input/output stay on the
CPU. A CUDA-enabled OpenCV build is not required.

**The CuPy backend is experimental. CUDA correctness, memory stability, and
speed have not been verified on this machine.** WSL currently reports blocked
GPU access, and CuPy is absent. CPU/NumPy expression checks do not establish
CUDA support or acceleration.

```bash
# Existing CPU behavior.
python run.py demo/input.avi --bbox=50,50,100,80 --quiet

# Explicit hybrid backend; initialization errors are fatal.
python run.py demo/input.avi --bbox=50,50,100,80 --quiet --backend cupy --device 0

# Availability selection at startup, with a visible CPU fallback reason.
python run.py demo/input.avi --bbox=50,50,100,80 --quiet --backend auto
```

`--gpu-memory-mb` sets the temporary GPU workspace budget (default 256 MiB).
Persistent descriptors and reference tables are checked separately against
available device memory. Matching uses bounded tiles. Exact pair medians need
quadratic storage: when the conservative geometry preflight exceeds the budget,
the existing CPU estimator runs and reports the route. For the laptop workload,
the default budget can route geometry to CPU; use a larger budget, such as
`--gpu-memory-mb 512`, if the device has room. There is no pair sampling, feature
cap, or runtime recovery from CUDA execution errors. `auto` does not tune speed;
select CPU explicitly for small workloads that run faster there.

Python callers can use `CMT(backend='cupy', device=0, gpu_memory_mb=256)`.
All published results remain host NumPy arrays. Enable detailed synchronized
wall-clock measurements with `tracker.profile.enabled = True` or `--profile`.
The CLI writes JSON stage records to stderr; nested stage times overlap.
`--quiet` continues to suppress windows while retaining drawing, output, and
frame summaries.

### GPU environment and validation

First establish GPU access in the same WSL/container environment using
`nvidia-smi`. Preserve the original `.venv`; create a separate GPU environment:

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip check
export CUPY_CACHE_DIR="$PWD/.cupy-cache"
python tests/run_gpu_validation.py
# Save a tested lock only AFTER the CUDA validation run succeeds.
python -m pip freeze > requirements-gpu-tested.lock
```

The optional requirements choose CUDA 12 CuPy with CUDA component wheels and
constrain NumPy/SciPy to the documented CuPy 14 test ranges. They are an
installation recipe, not a validated lock. This environment could not resolve
them during implementation, so no tested GPU lock is supplied. For a CUDA 13
installation, install `cupy-cuda13x[ctk]>=14.2,<15` with `requirements.txt` and
`constraints-gpu.txt` instead. Install exactly one CuPy distribution. A compatible
NVIDIA driver is required even with component wheels. See the official
[CuPy installation instructions](https://docs.cupy.dev/en/stable/install.html).

The ordinary suite clearly skips CUDA checks without a usable backend. The
release runner fails if CUDA startup fails or any tests are skipped. Startup
itself executes the actual Hamming/structural and median geometry operations,
including device allocation, generated kernels, and transfers.

### Reproducible benchmarks

```bash
python benchmarks/benchmark_tracking.py demo/IMG_3542.MOV \
  --bbox=0,520,325,238 --with-rotation --backends cpu,cupy \
  --repeats 3 --output benchmark-results/laptop.json

python benchmarks/benchmark_tracking.py demo/input.avi \
  --bbox=50,50,100,80 --backends cpu,cupy --repeats 3 \
  --application-output benchmark-results/demo-output \
  --output benchmark-results/demo.json
```

Each backend warms a disposable tracker, then starts fresh trackers at the
original initialization frame for measured runs. Reports record dependency
versions, thread count, input checksum, startup/initialization, median/p95 frame
latency, throughput, feature counts, pool memory, geometry routes, and outputs.
Pool memory is sampled at frame boundaries: reserved memory retains temporary
allocations, while live temporary allocation peaks are not measured. Tracker
throughput includes CPU stages and transfers; application throughput
also includes decoding and optional PNG/CSV output. A separate instrumented
sample (first 30 tracking frames by default) records stage/transfer costs and is
excluded from throughput measurements. Its nested times must not be added.

Use `--max-frames N` for a smoke run; reports mark it incomplete. Use
`--opencv-threads N` to fix thread settings across runs. CPU and CuPy comparisons
must use the same resolved environment. The harness synchronizes GPU work at
measurement boundaries, following
[CuPy performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html).
Available CPU/CuPy runs compare result flags, ordered class signatures, poses,
and floating corners before allowing a one-pixel integer-corner difference.
Unavailable backends are recorded as failures and produce a nonzero exit status;
completed CPU results are retained. The target is 1.25× warmed tracker throughput
on the full laptop video over at least three runs, after correctness validation.
Until measured CUDA results meet that target, this backend remains experimental.
See [the validation record](benchmarks/VALIDATION.md) for the current CPU
baseline and unresolved GPU environment checks.

## Validation

Run the regression checks from the project directory after activating `.venv`:

```bash
python -m unittest discover -s tests -v
```

These checks cover known object motion, selecting the full frame, a single
remaining keypoint, and tracking loss/recovery through the headless image
sequence runner.

# Usage
```
usage: run.py [-h] [--challenge] [--preview] [--no-preview] [--no-scale]
               [--with-rotation] [--bbox BBOX] [--pause] [--output-dir OUTPUT]
               [--quiet] [--skip SKIP] [--backend {cpu,cupy,auto}]
               [--device DEVICE] [--gpu-memory-mb GPU_MEMORY_MB] [--profile]
               [inputpath]
```

## Optional arguments
* `inputpath` The input path.
* `-h, --help` show help message and exit
* `--challenge` Enter challenge mode.
* `--preview` Force preview
* `--no-preview` Disable preview
* `--no-scale` Disable scale estimation
* `--with-rotation` Enable rotation estimation
* `--bbox BBOX` Specify initial bounding box. Format: x,y,w,h
* `--pause` Pause after each frame
* `--skip N` Skips N frames of the video input
* `--output-dir OUTPUT` Specify a directory for output data.
* `--quiet` Do not show graphical output (Useful in combination with --output-dir).

## Object Selection
Press any key to stop the preview stream. Left click to select the
top left bounding box corner and left click again to select the bottom right corner.

## Examples
When using a webcam, no arguments are necessary:
```
python run.py
```

When using a video, the path to the file has to be given as an input parameter:
```
python run.py /home/cmt/test.avi
```

It is also possible to specify the initial bounding box on the command line.
```
python run.py --bbox=123,85,60,140 /home/cmt/test.avi
```

Use a sequence of numbered image files as an input:
```
python run.py sequence_dir/{:08d}.jpg
```
Here, {:08d} is a python format string that is expanded to 00000001.jpg, 00000002.jpg, etc.

[1]: http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29
