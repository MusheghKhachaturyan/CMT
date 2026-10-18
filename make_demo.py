#!/usr/bin/env python3
"""Create a reproducible moving-object video for trying CMT without a webcam."""

from pathlib import Path

import cv2
import numpy as np


def main():
    output = Path(__file__).resolve().parent / 'demo'
    output.mkdir(exist_ok=True)
    path = output / 'input.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
    if not writer.isOpened():
        raise RuntimeError('Unable to create the demo video with the MJPG codec')

    rng = np.random.default_rng(42)
    texture = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
    cv2.putText(texture, 'CMT', (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    try:
        for frame in range(61):
            image = np.full((240, 320, 3), 40, dtype=np.uint8)
            x, y = 50 + 2 * frame, 50 + frame
            image[y:y + 80, x:x + 100] = texture
            writer.write(image)
            if frame == 0:
                cv2.imwrite(str(output / 'first_frame.png'), image)
    finally:
        writer.release()

    print('Created {} (61 frames, 320x240, 30 FPS)'.format(path))
    print('Run: .venv/bin/python run.py demo/input.avi --bbox=50,50,100,80 --quiet --output-dir demo/output')


if __name__ == '__main__':
    main()
