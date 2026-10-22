"""Shared runner/benchmark annotation and legacy image/CSV output formats."""
import cv2
import numpy as np
import util


def draw_and_write(tracker, im, im_draw, output, frame):
	# Draw updated estimate
	if tracker.has_result:

		cv2.line(im_draw, tracker.tl, tracker.tr, (255, 0, 0), 4)
		cv2.line(im_draw, tracker.tr, tracker.br, (255, 0, 0), 4)
		cv2.line(im_draw, tracker.br, tracker.bl, (255, 0, 0), 4)
		cv2.line(im_draw, tracker.bl, tracker.tl, (255, 0, 0), 4)

	util.draw_keypoints(tracker.tracked_keypoints, im_draw, (255, 255, 255))
	# this is from simplescale
	util.draw_keypoints(tracker.votes[:, :2], im_draw)  # blue
	util.draw_keypoints(tracker.outliers[:, :2], im_draw, (0, 0, 255))

	if output is not None:
		# Original image
		cv2.imwrite('{0}/input_{1:08d}.png'.format(output, frame), im)
		# Output image
		cv2.imwrite('{0}/output_{1:08d}.png'.format(output, frame), im_draw)

		# Keypoints
		with open('{0}/keypoints_{1:08d}.csv'.format(output, frame), 'w') as f:
			f.write('x y\n')
			np.savetxt(f, tracker.tracked_keypoints[:, :2], fmt='%.2f')

		# Outlier
		with open('{0}/outliers_{1:08d}.csv'.format(output, frame), 'w') as f:
			f.write('x y\n')
			np.savetxt(f, tracker.outliers, fmt='%.2f')

		# Votes
		with open('{0}/votes_{1:08d}.csv'.format(output, frame), 'w') as f:
			f.write('x y\n')
			np.savetxt(f, tracker.votes, fmt='%.2f')

		# Bounding box
		with open('{0}/bbox_{1:08d}.csv'.format(output, frame), 'w') as f:
			f.write('x y\n')
			# Duplicate entry tl is not a mistake, as it is used as a drawing instruction
			corners = np.array((tracker.tl, tracker.tr, tracker.br, tracker.bl, tracker.tl)) if tracker.has_result else np.full((5, 2), np.nan)
			np.savetxt(f, corners, fmt='%.2f')
