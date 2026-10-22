#!/usr/bin/env python3

import argparse
import json
import cv2
from numpy import empty, nan
import os
import sys
import time

import CMT
import numpy as np
import util
from backend import BackendError
from tracking_output import draw_and_write


parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('inputpath', nargs='?', help='The input path.')
parser.add_argument('--challenge', dest='challenge', action='store_true', help='Enter challenge mode.')
parser.add_argument('--preview', dest='preview', action='store_const', const=True, default=None, help='Force preview')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None, help='Disable preview')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--pause', dest='pause', action='store_true', help='Pause after every frame and wait for any key.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
parser.add_argument('--quiet', dest='quiet', action='store_true', help='Do not show graphical output (Useful in combination with --output-dir ).')
parser.add_argument('--skip', dest='skip', action='store', default=None, help='Skip the first n frames', type=int)
parser.add_argument('--backend', choices=('cpu', 'cupy', 'auto'), default='cpu', help='Numerical backend (default: cpu)')
parser.add_argument('--device', type=int, default=0, help='CUDA device index (default: 0)')
parser.add_argument('--gpu-memory-mb', type=float, default=256, help='Temporary GPU workspace budget in MiB')
parser.add_argument('--profile', action='store_true', help='Report synchronized stage timings to stderr as JSON')

args = parser.parse_args()

if args.quiet and args.bbox is None and not args.challenge:
	parser.error('--quiet requires --bbox=x,y,width,height')
if args.quiet and args.preview:
	parser.error('--quiet cannot be combined with --preview')
if args.skip is not None and args.skip < 0:
	parser.error('--skip must be non-negative')

try:
	CMT = CMT.CMT(backend=args.backend, device=args.device, gpu_memory_mb=args.gpu_memory_mb)
except (BackendError, ValueError) as exc:
	parser.error(str(exc))
CMT.profile.enabled = args.profile
if args.backend != 'cpu' or args.profile:
	print('Backend: ' + json.dumps(CMT._backend.diagnostics()), file=sys.stderr)


def report_profile(frame):
	if args.profile:
		print(json.dumps(dict(frame=frame, geometry_route=CMT.geometry_route,
			profile=CMT.profile.snapshot(), memory=CMT._backend.memory())), file=sys.stderr)


CMT.estimate_scale = args.estimate_scale
CMT.estimate_rotation = args.estimate_rotation

if args.pause:
	pause_time = 0
else:
	pause_time = 10

if args.output is not None:
	if not os.path.exists(args.output):
		os.makedirs(args.output)
	elif not os.path.isdir(args.output):
		raise Exception(args.output + ' exists, but is not a directory')

if args.challenge:
	with open('images.txt') as f:
		images = [line.strip() for line in f]

	init_region = np.genfromtxt('region.txt', delimiter=',')
	num_frames = len(images)

	results = empty((num_frames, 4))
	results[:] = nan

	results[0, :] = init_region

	frame = 0

	im0 = cv2.imread(images[frame])
	im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im0)

	tl, br = (util.array_to_int_tuple(init_region[:2]), util.array_to_int_tuple(init_region[:2] + init_region[2:4] - 1))

	try:
		CMT.initialise(im_gray0, tl, br)
		report_profile(0)
		frame = 1
		while frame < num_frames:
			im = cv2.imread(images[frame])
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			CMT.process_frame(im_gray)
			report_profile(frame)
			results[frame, :] = CMT.bb

			# Advance frame number
			frame += 1
	except Exception as exc:
		np.savetxt('output.txt', results, delimiter=',')
		print('Tracking failed: {}'.format(exc), file=sys.stderr)
		sys.exit(1)

	np.savetxt('output.txt', results, delimiter=',')

else:
	# Clean up
	if not args.quiet:
		cv2.destroyAllWindows()

	preview = args.preview

	if args.inputpath is not None:

		# If a path to a file was given, assume it is a single video file
		if os.path.isfile(args.inputpath):
			cap = cv2.VideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip)


		# Otherwise assume it is a format string for reading images
		else:
			cap = util.FileVideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.frame = 1 + args.skip

		# By default do not show preview in both cases
		if preview is None:
			preview = False

	else:
		# If no input path was specified, open camera device
		cap = cv2.VideoCapture(0)
		if preview is None:
			preview = not args.quiet

	# Check if videocapture is working
	if not cap.isOpened():
		print('Unable to open video input.', file=sys.stderr)
		sys.exit(1)

	while preview:
		status, im = cap.read()
		if not status:
			break
		cv2.imshow('Preview', im)
		k = cv2.waitKey(10)
		if not k == -1:
			break

	# Read first frame
	status, im0 = cap.read()
	if not status:
		print('Unable to read the first frame; check the input and --skip.', file=sys.stderr)
		cap.release()
		sys.exit(1)
	im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im0)

	if args.bbox is not None:
		# Try to disassemble user specified bounding box
		values = args.bbox.split(',')
		try:
			values = [int(v) for v in values]
		except:
			raise Exception('Unable to parse bounding box')
		if len(values) != 4:
			raise Exception('Bounding box must have exactly 4 elements')
		x, y, width, height = values
		if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > im0.shape[1] or y + height > im0.shape[0]:
			parser.error('Bounding box must have positive width/height and fit inside the first frame')
		bbox = np.array(values)

		# Convert to point representation, adding singleton dimension
		bbox = util.bb2pts(bbox[None, :])

		# Squeeze
		bbox = bbox[0, :]

		tl = bbox[:2]
		br = bbox[2:4]
	else:
		# Get rectangle input from user
		(tl, br) = util.get_rect(im_draw)

	print('using', tl, br, 'as init bb')


	CMT.initialise(im_gray0, tl, br)
	report_profile(0)

	frame = 1
	while True:
		# Read image
		status, im = cap.read()
		if not status:
			break
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_draw = np.copy(im)

		CMT.synchronize()
		tic = time.perf_counter()
		CMT.process_frame(im_gray)
		CMT.synchronize()
		toc = time.perf_counter()

		# Display results

		with CMT.profile.stage('drawing_output'):
			draw_and_write(CMT, im, im_draw, args.output, frame)
		report_profile(frame)

		if not args.quiet:
			cv2.imshow('main', im_draw)

			# Check key input
			k = cv2.waitKey(pause_time)
			key = chr(k & 255)
			if key == 'q':
				break
			if key == 'd':
				import ipdb; ipdb.set_trace()

		# Remember image
		im_prev = im_gray

		# Advance frame number
		frame += 1

		print('{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame))

	cap.release()
	if not args.quiet:
		cv2.destroyAllWindows()
