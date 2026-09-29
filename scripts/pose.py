#!/usr/bin/env python3
"""The camera side of Tomo's health programme: where the user's joints are.

Runs as a long-lived child of Tomo's brain (tomo/health.py).
It opens the camera, runs MediaPipe Pose Lite (a ~6 MB pose model, on the
CPU, ~10 ms a frame) on a few frames a second, and prints each result as one
JSON line on stdout. Everything after that — counting reps, noticing breaks,
the lock — is Tomo's own plain code; nothing here decides anything.

  {"event": "camera", "ok": true | false}   a camera appeared / there is none
  {"event": "pose", "world": [[x, y, z, visibility], … 33] | null}
                                              one frame's joints, in metres from
                                              the hips (x right, y down, z away
                                              from the camera); null: nobody seen
  {"event": "error", "message": "..."}

Without a camera it says so and looks again every 30 s. Lines on stdin steer
it: "fps N" — frames a second to look at (a couple while watching for breaks,
15 during a round). No image leaves this process; none is kept.

Testing without a camera (debug builds of Tomo only):
  pose.py --model pose_landmarker_lite.task --replay joints.jsonl
plays recorded joints — lines of {"t": seconds, "world": [...] | null} —
in a loop, as if from a camera.
"""

import argparse
import json
import sys
import threading
import time

CAMERA_RETRY_SECONDS = 30
# Cameras to try, in order: most machines have one, at 0.
CAMERAS = range(4)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


class Steering:
    """What stdin asks for: frames a second."""

    def __init__(self, fps):
        self.fps = max(1, fps)

    def listen(self):
        for line in sys.stdin:
            parts = line.split()
            if len(parts) == 2 and parts[0] == "fps" and parts[1].isdigit():
                self.fps = max(1, min(30, int(parts[1])))
        # stdin closed: Tomo is gone; let go of the camera.
        import os

        os._exit(0)


def open_camera():
    """The first camera that gives a frame, or None."""
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for index in CAMERAS:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened() and camera.read()[0]:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return camera
        camera.release()
    return None


def landmarker(model):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def joints(result):
    """The first person's world joints as [[x, y, z, visibility], …], or None."""
    if not result.pose_world_landmarks:
        return None
    return [
        [round(j.x, 4), round(j.y, 4), round(j.z, 4), round(j.visibility or 0.0, 3)]
        for j in result.pose_world_landmarks[0]
    ]


def watch(model, steering):
    import cv2
    import mediapipe as mp

    detector = landmarker(model)
    started = time.monotonic()
    while True:
        camera = open_camera()
        if camera is None:
            emit("camera", ok=False)
            time.sleep(CAMERA_RETRY_SECONDS)
            continue
        emit("camera", ok=True)
        failures = 0
        while failures < 5:
            tick = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                failures += 1
                time.sleep(0.2)
                continue
            failures = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int((time.monotonic() - started) * 1000))
            emit("pose", world=joints(result))
            time.sleep(max(0.0, 1.0 / steering.fps - (time.monotonic() - tick)))
        # Unplugged (or taken by another app): say so, and look for it again.
        camera.release()
        emit("camera", ok=False)
        time.sleep(2)


def replay(path, steering):
    frames = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not frames:
        raise RuntimeError(f"{path} has no frames")
    emit("camera", ok=True)
    while True:
        start = time.monotonic()
        for frame in frames:
            wait = frame.get("t", 0.0) - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
            emit("pose", world=frame.get("world"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="MediaPipe pose landmarker (.task)")
    parser.add_argument("--fps", type=int, default=2, help="frames a second to start with")
    parser.add_argument("--replay", help="play recorded joints instead of the camera (testing)")
    args = parser.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    # Load the compiled modules before a thread blocks reading stdin: on
    # Windows, loading one while stdin is being read deadlocks.
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401
    import numpy  # noqa: F401

    steering = Steering(args.fps)
    threading.Thread(target=steering.listen, daemon=True).start()
    try:
        if args.replay:
            replay(args.replay, steering)
        else:
            watch(args.model, steering)
    except Exception as error:  # report, don't traceback: the brain reads stdout
        emit("error", message=str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
