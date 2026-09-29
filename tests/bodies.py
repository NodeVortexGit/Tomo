"""Synthetic camera poses for the tests: a body built from a few angles, in
MediaPipe's world coordinates (metres, y down), all 33 joints visible."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tomo.health import (JOINT_COUNT, LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE,
                         RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, Joint)

# Each side's joints, and how far that side is from the camera's middle plane.
SIDES = ((0.1, (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)),
         (-0.1, (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)))


@dataclass(frozen=True)
class Body:
    torso: float = 0.0  # lean from upright, degrees (90: lying, head toward −x)
    hips: float = 180.0  # angle at the hips between torso and thighs (180: straight)
    knees: float = 180.0
    elbows: float = 170.0
    shoulders: float = 10.0  # upper arm's angle from the torso

    @staticmethod
    def standing() -> "Body":
        return Body()

    def but(self, **changes) -> "Body":
        return replace(self, **changes)

    def pose(self) -> list[Joint]:
        """In the image plane (z = 0 but for the sides), facing +x."""

        def direction(deg: float) -> tuple[float, float]:
            r = math.radians(deg)
            return math.sin(r), -math.cos(r)  # 0° points up (−y)

        def along(start, deg, length):
            d = direction(deg)
            return start[0] + d[0] * length, start[1] + d[1] * length

        hip = (0.0, 0.0)
        up = -self.torso
        shoulder = along(hip, up, 0.5)
        thigh = up + 180.0 - (180.0 - self.hips)
        knee = along(hip, thigh, 0.45)
        ankle = along(knee, thigh + (180.0 - self.knees), 0.45)
        arm = up + 180.0 - self.shoulders
        elbow = along(shoulder, arm, 0.3)
        wrist = along(elbow, arm - (180.0 - self.elbows), 0.28)
        head = along(shoulder, up, 0.25)

        pose = [Joint(0.0, 0.0, 0.0, 1.0)] * JOINT_COUNT
        pose[NOSE] = Joint(head[0], head[1], 0.0, 1.0)
        for side, joints in SIDES:
            for index, point in zip(joints, (shoulder, elbow, wrist, hip, knee, ankle)):
                pose[index] = Joint(point[0], point[1], side, 1.0)
        return pose


def hidden(pose: list[Joint], *indices: int) -> list[Joint]:
    """The pose with these joints out of view."""
    return [Joint(j.x, j.y, j.z, 0.0) if i in indices else j for i, j in enumerate(pose)]


def cycle(n: int, top: Body, bottom: Body) -> list[Body]:
    """``n`` reps between two bodies, a few frames each way."""
    frames = [top, top]
    for _ in range(n):
        frames += [bottom, bottom, bottom, top, top, top]
    return frames
