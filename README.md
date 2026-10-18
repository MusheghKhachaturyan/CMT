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

This implementation runs tracking on the **CPU**. It uses BRISK features,
Hamming descriptor matching, Lucas–Kanade optical flow, NumPy arrays, and SciPy
clustering. It does not use `cv2.cuda`, CUDA device memory, PyTorch, or CuPy, and
there is no GPU command-line switch. Having an NVIDIA GPU or installing CUDA
does not move this algorithm to the GPU. GPU acceleration would require code
changes and an appropriate GPU-enabled OpenCV build.

The compatibility updates in this checkout support Python 3 and OpenCV 4;
the original upstream implementation targeted Python 2 and OpenCV 2.4.

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
               [--quiet] [--skip SKIP]
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
