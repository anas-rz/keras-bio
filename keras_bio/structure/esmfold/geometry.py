"""
Multi-backend 3D Lie Group geometry primitives for protein structure modeling.

Implements SO(3) rotations and SE(3) rigid-body transformations using keras.ops
with exact parity to OpenFold and AlphaFold 2 implementations.
"""

from collections.abc import Callable, Sequence
from typing import Any

import keras
from keras import ops


def rot_matmul(a, b):
    """Batch matrix multiplication of two rotation matrix tensors [..., 3, 3]."""
    return ops.matmul(a, b)


def rot_vec_mul(r, v):
    """
    Applies rotation matrix tensor [..., 3, 3] to 3D point tensor [..., 3].

    Computes R @ v along the last two dimensions.
    """
    # Using component multiplication for maximum numerical stability & multi-backend parity
    x = r[..., 0, 0] * v[..., 0] + r[..., 0, 1] * v[..., 1] + r[..., 0, 2] * v[..., 2]
    y = r[..., 1, 0] * v[..., 0] + r[..., 1, 1] * v[..., 1] + r[..., 1, 2] * v[..., 2]
    z = r[..., 2, 0] * v[..., 0] + r[..., 2, 1] * v[..., 1] + r[..., 2, 2] * v[..., 2]
    return ops.stack([x, y, z], axis=-1)


def invert_rot_mat(r):
    """Inverts a rotation matrix by taking its matrix transpose on the last two dimensions."""
    return ops.swapaxes(r, -1, -2)


def quat_to_rot(quat):
    """
    Converts a unit quaternion [..., 4] (w, x, y, z) into a 3x3 rotation matrix [..., 3, 3].
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    r00 = 1.0 - 2.0 * (ops.square(y) + ops.square(z))
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)

    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (ops.square(x) + ops.square(z))
    r12 = 2.0 * (y * z - w * x)

    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (ops.square(x) + ops.square(y))

    row0 = ops.stack([r00, r01, r02], axis=-1)
    row1 = ops.stack([r10, r11, r12], axis=-1)
    row2 = ops.stack([r20, r21, r22], axis=-1)

    return ops.stack([row0, row1, row2], axis=-2)


def quat_multiply_by_vec(quat, vec):
    """
    Multiplies a quaternion [..., 4] by a pure-vector quaternion [..., 3] (0, vec).

    Resulting quaternion update is:
      r_w = -(q_x * v_x + q_y * v_y + q_z * v_z)
      r_x =  q_w * v_x + q_y * v_z - q_z * v_y
      r_y =  q_w * v_y + q_z * v_x - q_x * v_z
      r_z =  q_w * v_z + q_x * v_y - q_y * v_x
    """
    w = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    vx = vec[..., 0]
    vy = vec[..., 1]
    vz = vec[..., 2]

    rw = -(qx * vx + qy * vy + qz * vz)
    rx = w * vx + qy * vz - qz * vy
    ry = w * vy + qz * vx - qx * vz
    rz = w * vz + qx * vy - qy * vx

    return ops.stack([rw, rx, ry, rz], axis=-1)


class Rotation:
    """
    Represents 3D rotations in SO(3), parameterized as 3x3 rotation matrices or unit quaternions.
    """

    def __init__(self, rot_mats=None, quats=None, normalize_quats=True):
        if rot_mats is None and quats is None:
            raise ValueError("Either rot_mats or quats must be provided")

        if quats is not None and normalize_quats:
            norm = ops.sqrt(ops.sum(ops.square(quats), axis=-1, keepdims=True) + 1e-8)
            quats = quats / norm

        self._rot_mats = rot_mats
        self._quats = quats

    @property
    def shape(self):
        if self._rot_mats is not None:
            return ops.shape(self._rot_mats)[:-2]
        return ops.shape(self._quats)[:-1]

    def get_rot_mats(self):
        if self._rot_mats is None:
            self._rot_mats = quat_to_rot(self._quats)
        return self._rot_mats

    def get_quats(self):
        if self._quats is None:
            raise ValueError("Quaternions not explicitly stored in this Rotation")
        return self._quats

    def apply(self, pts):
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts):
        rot_mats = self.get_rot_mats()
        inv_rot = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot, pts)

    def compose_r(self, other):
        r1 = self.get_rot_mats()
        r2 = other.get_rot_mats()
        return Rotation(rot_mats=rot_matmul(r1, r2), quats=None)

    def compose_q_update_vec(self, q_update_vec, normalize_quats=True):
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def stop_rot_gradient(self):
        rot_mats = ops.stop_gradient(self._rot_mats) if self._rot_mats is not None else None
        quats = ops.stop_gradient(self._quats) if self._quats is not None else None
        return Rotation(rot_mats=rot_mats, quats=quats, normalize_quats=False)

    def __mul__(self, right):
        if self._rot_mats is not None:
            rot_mats = self._rot_mats * ops.expand_dims(ops.expand_dims(right, axis=-1), axis=-1)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * ops.expand_dims(right, axis=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations representations are None")

    def map_tensor_fn(self, fn: Callable):
        if self._rot_mats is not None:
            orig_shape = ops.shape(self._rot_mats)
            flat = ops.reshape(self._rot_mats, tuple(orig_shape[:-2]) + (9,))
            mapped = fn(flat)
            new_shape = tuple(ops.shape(mapped)[:-1]) + (3, 3)
            return Rotation(rot_mats=ops.reshape(mapped, new_shape), quats=None)
        elif self._quats is not None:
            mapped = fn(self._quats)
            return Rotation(rot_mats=None, quats=mapped, normalize_quats=False)
        else:
            raise ValueError("Both rotations representations are None")

    def __getitem__(self, index: Any):
        if not isinstance(index, tuple):
            index = (index,)
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats[index + (slice(None), slice(None))], quats=None)
        return Rotation(rot_mats=None, quats=self._quats[index + (slice(None),)], normalize_quats=False)

    @classmethod
    def cat(cls, rs: Sequence["Rotation"], dim: int) -> "Rotation":
        rot_mats_list = [r.get_rot_mats() for r in rs]
        dim = dim if dim >= 0 else dim - 2
        return cls(rot_mats=ops.concatenate(rot_mats_list, axis=dim), quats=None)

    @classmethod
    def identity(cls, shape, dtype="float32"):
        zeros = ops.zeros(shape + (3,), dtype=dtype)
        ones = ops.ones(shape + (1,), dtype=dtype)
        quats = ops.concatenate([ones, zeros], axis=-1)
        return cls(rot_mats=None, quats=quats, normalize_quats=False)


class Rigid:
    """
    Represents 3D rigid-body transformations in SE(3), combining a Rotation and Translation vector.
    """

    def __init__(self, rots: Rotation, trans):
        self._rots = rots
        self._trans = trans

    @property
    def shape(self):
        return ops.shape(self._trans)[:-1]

    def get_rots(self) -> Rotation:
        return self._rots

    def get_trans(self):
        return self._trans

    def apply(self, pts):
        """Applies (R, t) to coordinates: R @ pts + t."""
        return self._rots.apply(pts) + self._trans

    def invert_apply(self, pts):
        """Applies inverse transformation: R^T @ (pts - t)."""
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def compose(self, other: "Rigid") -> "Rigid":
        """Composes two rigid transformations: (R1, t1) o (R2, t2) = (R1 R2, R1 t2 + t1)."""
        new_rots = self._rots.compose_r(other.get_rots())
        new_trans = self._trans + self._rots.apply(other.get_trans())
        return Rigid(new_rots, new_trans)

    def compose_q_update_vec(self, q_update_vec) -> "Rigid":
        """
        Composes with a 6D update vector [q_x, q_y, q_z, t_x, t_y, t_z].
        """
        q_vec = q_update_vec[..., :3]
        t_vec = q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec)
        new_trans = self._trans + self._rots.apply(t_vec)
        return Rigid(new_rots, new_trans)

    def scale_translation(self, factor) -> "Rigid":
        return Rigid(self._rots, self._trans * factor)

    def stop_rot_gradient(self) -> "Rigid":
        return Rigid(self._rots.stop_rot_gradient(), self._trans)

    def __mul__(self, right) -> "Rigid":
        new_rots = self._rots * right
        new_trans = self._trans * ops.expand_dims(right, axis=-1)
        return Rigid(new_rots, new_trans)

    def map_tensor_fn(self, fn: Callable) -> "Rigid":
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = fn(self._trans)
        return Rigid(new_rots, new_trans)

    def __getitem__(self, index: Any) -> "Rigid":
        if not isinstance(index, tuple):
            index = (index,)
        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    @classmethod
    def cat(cls, ts: Sequence["Rigid"], dim: int) -> "Rigid":
        rots = Rotation.cat([t.get_rots() for t in ts], dim)
        trans_list = [t.get_trans() for t in ts]
        cat_dim = dim if dim >= 0 else dim - 1
        trans = ops.concatenate(trans_list, axis=cat_dim)
        return cls(rots, trans)

    @classmethod
    def identity(cls, shape, dtype="float32") -> "Rigid":
        rots = Rotation.identity(shape, dtype=dtype)
        trans = ops.zeros(shape + (3,), dtype=dtype)
        return cls(rots, trans)

    @classmethod
    def from_tensor_4x4(cls, t) -> "Rigid":
        rot_mats = t[..., :3, :3]
        trans = t[..., :3, 3]
        return cls(Rotation(rot_mats=rot_mats), trans)

    def to_tensor_4x4(self):
        rot_mats = self._rots.get_rot_mats()
        trans = ops.expand_dims(self._trans, axis=-1)
        top = ops.concatenate([rot_mats, trans], axis=-1)  # [..., 3, 4]
        batch_shape = ops.shape(self._trans)[:-1]
        zeros = ops.zeros(tuple(batch_shape) + (1, 3), dtype=top.dtype)
        ones = ops.ones(tuple(batch_shape) + (1, 1), dtype=top.dtype)
        bottom = ops.concatenate([zeros, ones], axis=-1)  # [..., 1, 4]
        return ops.concatenate([top, bottom], axis=-2)

    def to_tensor_7(self):
        quats = self._rots.get_quats()
        return ops.concatenate([quats, self._trans], axis=-1)

    @classmethod
    def from_tensor_7(cls, t, normalize_quats=True) -> "Rigid":
        quats = t[..., :4]
        trans = t[..., 4:]
        return cls(Rotation(quats=quats, normalize_quats=normalize_quats), trans)

