"""NumPy serial chains in metres, kilograms, seconds, and radians."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Vector3 = tuple[float, float, float]
FloatArray = NDArray[np.float64]


def _number(
    value: object, name: str, *, minimum: float | None = None, positive: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _vector(value: object, size: int, name: str) -> FloatArray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have shape ({size},)") from exc
    if raw.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.array([_number(item, name) for item in raw], dtype=np.float64)


def _triple(value: object, name: str) -> Vector3:
    vector = _vector(value, 3, name)
    return float(vector[0]), float(vector[1]), float(vector[2])


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Link:
    """A local hinge followed by an offset in its rotated frame."""

    name: str
    axis: Vector3
    offset: Vector3
    mass: float
    limits: tuple[float, float] = (-math.pi, math.pi)
    radius: float = 0.015
    damping: float = 0.05
    armature: float = 0.01
    torque_limit: float = 10.0
    kp: float = 20.0
    kd: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", self.name
        ):
            raise ValueError("link name must be a safe identifier")
        axis = _vector(self.axis, 3, f"{self.name}.axis")
        scale = float(np.max(np.abs(axis)))
        if scale == 0:
            raise ValueError(f"{self.name}.axis must be nonzero")
        # Scale first so finite very large or tiny axes normalize safely.
        axis /= scale
        axis /= np.linalg.norm(axis)
        object.__setattr__(self, "axis", _triple(axis, "axis"))
        object.__setattr__(self, "offset", _triple(self.offset, "offset"))
        limits = _vector(self.limits, 2, "limits")
        if limits[0] >= limits[1]:
            raise ValueError("limits must have lower < upper")
        object.__setattr__(self, "limits", (float(limits[0]), float(limits[1])))
        for name in ("mass", "radius", "torque_limit", "kp"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )
        for name in ("damping", "armature", "kd"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, minimum=0)
            )


@dataclass(frozen=True)
class SerialArm:
    """Translate base, then rotate(axis, q) and translate(offset) per link.

    The tool translation is in the final rotated link frame. All values are SI.
    """

    links: tuple[Link, ...]
    name: str = "serial_arm"
    base: Vector3 = (0.0, 0.0, 0.0)
    tool: Vector3 = (0.0, 0.0, 0.0)
    payload_mass: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if (
            not isinstance(self.links, (tuple, list))
            or not self.links
            or any(not isinstance(link, Link) for link in self.links)
        ):
            raise ValueError("links must be a nonempty sequence of Link objects")
        if len({link.name for link in self.links}) != len(self.links):
            raise ValueError("link names must be unique")
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "base", _triple(self.base, "base"))
        object.__setattr__(self, "tool", _triple(self.tool, "tool"))
        object.__setattr__(
            self, "payload_mass", _number(self.payload_mass, "payload_mass", minimum=0)
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SerialArm:
        """Load a robot-only configuration; scenario settings are not accepted."""
        if not isinstance(config, dict):
            raise ValueError("robot configuration must be an object")
        unknown = set(config) - {
            "name",
            "units",
            "base",
            "tool",
            "payload_mass",
            "links",
        }
        if unknown:
            raise ValueError(f"unknown robot keys: {unknown}")
        if not isinstance(config.get("units"), str) or config["units"] != "SI":
            raise ValueError("units must be exactly 'SI'")
        raw_links = config.get("links")
        if not isinstance(raw_links, list) or not raw_links:
            raise ValueError("links must be a nonempty array")
        links = []
        allowed = {field.name for field in fields(Link)}
        for index, raw in enumerate(raw_links):
            if not isinstance(raw, dict):
                raise ValueError(f"links[{index}] must be an object")
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"unknown link keys: {unknown}")
            missing = {"name", "axis", "offset", "mass"} - set(raw)
            if missing:
                raise ValueError(f"links[{index}] missing required keys: {missing}")
            links.append(Link(**raw))
        return cls(
            links=tuple(links),
            name=config.get("name", "serial_arm"),
            base=config.get("base", (0, 0, 0)),
            tool=config.get("tool", (0, 0, 0)),
            payload_mass=config.get("payload_mass", 0),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SerialArm:
        """Read strict JSON, rejecting duplicate keys and nonfinite constants."""

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> Any:
            raise ValueError(f"invalid JSON constant: {value}")

        with Path(path).open(encoding="utf-8") as stream:
            config = json.load(
                stream, object_pairs_hook=unique_object, parse_constant=reject_constant
            )
        return cls.from_dict(config)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible configuration with defaults."""
        links = []
        for link in self.links:
            values = asdict(link)
            for name in ("axis", "offset", "limits"):
                values[name] = list(values[name])
            links.append(values)
        return {
            "name": self.name,
            "units": "SI",
            "base": list(self.base),
            "tool": list(self.tool),
            "payload_mass": self.payload_mass,
            "links": links,
        }

    @property
    def n_joints(self) -> int:
        return len(self.links)

    @property
    def joint_limits(self) -> FloatArray:
        return np.array([link.limits for link in self.links], dtype=np.float64)

    def validate_configuration(
        self, q: object, *, enforce_limits: bool = False
    ) -> FloatArray:
        values = _vector(q, self.n_joints, "q")
        limits = self.joint_limits
        if enforce_limits and np.any((values < limits[:, 0]) | (values > limits[:, 1])):
            raise ValueError("q is outside joint limits")
        return values

    def _frames(self, q: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        position = np.array(self.base, dtype=np.float64)
        rotation = np.eye(3)
        points = np.empty((self.n_joints + 2, 3))
        axes = np.empty((self.n_joints, 3))
        for index, (link, angle) in enumerate(zip(self.links, q, strict=True)):
            points[index] = position
            axis = np.array(link.axis)
            axes[index] = rotation @ axis
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            local_rotation = (
                np.eye(3)
                + math.sin(angle) * skew
                + (1 - math.cos(angle)) * (skew @ skew)
            )
            rotation = rotation @ local_rotation
            position = position + rotation @ link.offset
        points[-2] = position
        points[-1] = position + rotation @ self.tool
        return points, rotation, axes

    def forward_kinematics(self, q: object) -> tuple[FloatArray, FloatArray]:
        points, rotation, _ = self._frames(self.validate_configuration(q))
        return points[-1].copy(), rotation

    def joint_positions(self, q: object) -> FloatArray:
        """Joint origins, final link endpoint, then tool tip (including repeats)."""
        return self._frames(self.validate_configuration(q))[0]

    def jacobian(self, q: object) -> FloatArray:
        points, _, axes = self._frames(self.validate_configuration(q))
        return np.cross(axes, points[-1] - points[: self.n_joints]).T

    def inverse_kinematics(
        self,
        target: object,
        q0: object = None,
        *,
        tol: float = 1e-5,
        max_iter: int = 200,
        restarts: int = 4,
        seed: int = 0,
    ) -> tuple[FloatArray, bool, float]:
        """Bounded position-only DLS with an initial guess plus random restarts.

        Initial guesses are clipped to limits. Failure is not proof of unreachability.
        """
        goal = _vector(target, 3, "target")
        tolerance = _number(tol, "tol", positive=True)
        iterations = _integer(max_iter, "max_iter", minimum=1)
        restart_count = _integer(restarts, "restarts")
        rng = np.random.default_rng(_integer(seed, "seed"))
        lower, upper = self.joint_limits.T
        initial = (
            np.zeros(self.n_joints) if q0 is None else self.validate_configuration(q0)
        )
        best_q = np.clip(initial, lower, upper)
        best_error = math.inf
        for attempt in range(restart_count + 1):
            q = best_q.copy() if attempt == 0 else rng.uniform(lower, upper)
            for iteration in range(iterations + 1):
                points, _, axes = self._frames(q)
                error = goal - points[-1]
                residual = float(np.linalg.norm(error))
                if residual < best_error:
                    best_q, best_error = q.copy(), residual
                if residual <= tolerance:
                    return q.copy(), True, residual
                if iteration == iterations:
                    break
                jacobian = np.cross(axes, points[-1] - points[: self.n_joints]).T
                damping = 1e-3 * max(float(np.linalg.norm(jacobian)), 1e-3)
                step = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(3), error
                )
                step *= min(1.0, 0.5 / max(float(np.linalg.norm(step)), 1e-15))
                for backtrack in range(12):
                    candidate = np.clip(q + step * (0.5**backtrack), lower, upper)
                    candidate_error = float(
                        np.linalg.norm(goal - self._frames(candidate)[0][-1])
                    )
                    if candidate_error < residual:
                        q = candidate
                        break
                else:
                    break
        return best_q, False, best_error

    def workspace(self, n_samples: int = 1000, seed: int = 0) -> FloatArray:
        """Sample joint limits uniformly, not Cartesian volume or collision-free space."""
        count = _integer(n_samples, "n_samples", minimum=1)
        rng = np.random.default_rng(_integer(seed, "seed"))
        lower, upper = self.joint_limits.T
        samples = rng.uniform(lower, upper, size=(count, self.n_joints))
        return np.array([self._frames(q)[0][-1] for q in samples], dtype=np.float64)

    def to_mjcf(self, *, timestep: float = 0.002) -> str:
        """Generate primitive geometry with ALL contacts disabled, not clearance guarantees.

        Default motor limits and gains are examples, not hardware tuning.
        MuJoCo is not needed to generate this XML.
        """
        dt = _number(timestep, "timestep", positive=True)

        def text(values: object) -> str:
            return " ".join(
                format(float(value), ".17g") for value in np.asarray(values).flat
            )

        root = ET.Element("mujoco", model=self.name)
        ET.SubElement(
            root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true"
        )
        ET.SubElement(
            root,
            "option",
            timestep=str(dt),
            gravity="0 0 -9.81",
            integrator="implicitfast",
        )
        defaults = ET.SubElement(root, "default")
        ET.SubElement(defaults, "geom", contype="0", conaffinity="0")
        body = ET.SubElement(root, "worldbody")
        actuators = ET.SubElement(root, "actuator")
        colors = ("0.20 0.48 0.70 1", "0.90 0.55 0.20 1", "0.25 0.65 0.48 1")
        for index, link in enumerate(self.links):
            origin = self.base if index == 0 else self.links[index - 1].offset
            body = ET.SubElement(body, "body", name=f"body_{index}", pos=text(origin))
            ET.SubElement(
                body,
                "joint",
                name=link.name,
                type="hinge",
                axis=text(link.axis),
                range=text(link.limits),
                limited="true",
                damping=str(link.damping),
                armature=str(link.armature),
            )
            geometry = {
                "name": f"geom_{index}",
                "size": str(link.radius),
                "mass": str(link.mass),
                "rgba": colors[index % len(colors)],
            }
            if any(link.offset):
                geometry.update(type="capsule", fromto=text((0, 0, 0, *link.offset)))
            else:
                geometry.update(type="sphere", pos="0 0 0")
            ET.SubElement(body, "geom", geometry)
            ET.SubElement(
                actuators,
                "motor",
                name=f"motor_{index}",
                joint=link.name,
                gear="1",
                ctrllimited="true",
                ctrlrange=text((-link.torque_limit, link.torque_limit)),
            )
        tip = np.array(self.links[-1].offset) + self.tool
        ET.SubElement(
            body,
            "site",
            name="end_effector",
            pos=text(tip),
            size="0.008",
            rgba="0.9 0.2 0.2 1",
        )
        if self.payload_mass > 0:
            ET.SubElement(
                body,
                "geom",
                name="payload",
                type="sphere",
                pos=text(tip),
                size=str(self.links[-1].radius),
                mass=str(self.payload_mass),
                rgba="0.3 0.3 0.3 1",
            )
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")
