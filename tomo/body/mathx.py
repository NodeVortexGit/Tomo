"""Rotations and transforms.

Quaternions are ``(x, y, z, w)``, glTF's order. Single ones are plain tuples
— for a handful of bones Python's own arithmetic beats numpy's per-call cost;
the batch functions (``*_batch``) take numpy arrays for whole skeletons.

``qmul(a, b)`` turns by ``b`` first, then ``a`` (as glam's ``a * b``).
Matrices are numpy, row-major in memory, acting on column vectors: ``M @ v``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]

IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Vec2:
    """A point or direction on the screen (logical px, +y down)."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def dot(self, o: "Vec2") -> float:
        return self.x * o.x + self.y * o.y

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance(self, o: "Vec2") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def lerp(self, o: "Vec2", t: float) -> "Vec2":
        return Vec2(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)

    def clamp_length_max(self, most: float) -> "Vec2":
        n = self.length()
        return self * (most / n) if n > most else self


# ---- vectors -----------------------------------------------------------------------------


def add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def scale(a: Vec3, k: float) -> Vec3:
    return a[0] * k, a[1] * k, a[2] * k


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def length(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize_or_zero(a: Vec3) -> Vec3:
    n = length(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 and math.isfinite(n) else (0.0, 0.0, 0.0)


def is_zero(a: Vec3) -> bool:
    return a == (0.0, 0.0, 0.0)


# ---- quaternions -------------------------------------------------------------------------


def qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qconj(q: Quat) -> Quat:
    """The inverse of a unit quaternion."""
    return -q[0], -q[1], -q[2], q[3]


def qnormalize(q: Quat) -> Quat:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n) if n > 1e-12 else IDENTITY


def qrot(q: Quat, v: Vec3) -> Vec3:
    """``v`` turned by ``q``."""
    x, y, z, w = q
    # t = 2 q.xyz × v; v' = v + w t + q.xyz × t
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty,
            v[1] + w * ty + z * tx - x * tz,
            v[2] + w * tz + x * ty - y * tx)


def qaxis(axis: Vec3, radians: float) -> Quat:
    s = math.sin(radians * 0.5)
    ax = normalize_or_zero(axis)
    return ax[0] * s, ax[1] * s, ax[2] * s, math.cos(radians * 0.5)


def rx(degrees: float) -> Quat:
    return qaxis((1.0, 0.0, 0.0), math.radians(degrees))


def ry(degrees: float) -> Quat:
    return qaxis((0.0, 1.0, 0.0), math.radians(degrees))


def rz(degrees: float) -> Quat:
    return qaxis((0.0, 0.0, 1.0), math.radians(degrees))


def qslerp(a: Quat, b: Quat, t: float) -> Quat:
    """Spherical interpolation, the short way round."""
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if d < 0.0:
        b, d = (-b[0], -b[1], -b[2], -b[3]), -d
    if d > 0.9995:  # nearly the same: a straight blend is exact enough
        return qnormalize(tuple(a[i] + (b[i] - a[i]) * t for i in range(4)))
    theta = math.acos(min(1.0, d))
    s = math.sin(theta)
    wa, wb = math.sin((1.0 - t) * theta) / s, math.sin(t * theta) / s
    return tuple(a[i] * wa + b[i] * wb for i in range(4))


def qarc(src: Vec3, dst: Vec3) -> Quat:
    """The shortest turn taking unit vector ``src`` to unit vector ``dst``."""
    d = dot(src, dst)
    if d < -1.0 + 1e-6:
        # Opposite: half a turn about any axis at right angles to both.
        axis = cross((1.0, 0.0, 0.0), src)
        if dot(axis, axis) < 1e-8:
            axis = cross((0.0, 1.0, 0.0), src)
        return qaxis(axis, math.pi)
    c = cross(src, dst)
    return qnormalize((c[0], c[1], c[2], 1.0 + d))


def qfrom_mat3(m) -> Quat:
    """A rotation matrix (3×3, columns the turned axes) as a quaternion."""
    m00, m01, m02 = float(m[0][0]), float(m[0][1]), float(m[0][2])
    m10, m11, m12 = float(m[1][0]), float(m[1][1]), float(m[1][2])
    m20, m21, m22 = float(m[2][0]), float(m[2][1]), float(m[2][2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    return qnormalize(q)


def qfrom_axes(x: Vec3, y: Vec3, z: Vec3) -> Quat:
    """The rotation taking the unit axes to ``x``, ``y``, ``z`` (orthonormal)."""
    return qfrom_mat3(((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2])))


# ---- batches (numpy) ---------------------------------------------------------------------


def quat_to_mat3_batch(q: np.ndarray) -> np.ndarray:
    """(n, 4) quaternions → (n, 3, 3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    m = np.empty((len(q), 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (yy + zz)
    m[:, 0, 1] = 2 * (xy - wz)
    m[:, 0, 2] = 2 * (xz + wy)
    m[:, 1, 0] = 2 * (xy + wz)
    m[:, 1, 1] = 1 - 2 * (xx + zz)
    m[:, 1, 2] = 2 * (yz - wx)
    m[:, 2, 0] = 2 * (xz - wy)
    m[:, 2, 1] = 2 * (yz + wx)
    m[:, 2, 2] = 1 - 2 * (xx + yy)
    return m


def compose_batch(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Translation (n, 3), rotation (n, 4) and scale (n, 3) → (n, 4, 4) T·R·S."""
    n = len(t)
    m = np.zeros((n, 4, 4), dtype=np.float64)
    m[:, :3, :3] = quat_to_mat3_batch(q) * s[:, None, :]
    m[:, :3, 3] = t
    m[:, 3, 3] = 1.0
    return m


def mat3_to_quat_batch(m: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation matrices → (n, 4) quaternions."""
    return np.array([qfrom_mat3(r) for r in m], dtype=np.float64).reshape(len(m), 4)


def rotation_of(m: np.ndarray) -> np.ndarray:
    """The rotation part of an affine matrix (…, 4, 4) or (…, 3, 3): its
    columns with the scale taken out."""
    r = np.asarray(m)[..., :3, :3]
    return r / np.linalg.norm(r, axis=-2, keepdims=True)


def compose(t: Vec3, q: Quat, s: Vec3 = (1.0, 1.0, 1.0)) -> np.ndarray:
    """One 4×4 T·R·S matrix."""
    return compose_batch(np.array([t], dtype=np.float64), np.array([q], dtype=np.float64),
                         np.array([s], dtype=np.float64))[0]
