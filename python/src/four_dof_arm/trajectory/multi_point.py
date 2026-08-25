from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .joint import JointTrajectory
from .validation import _finite_vector, _method_name, _point_count


class MultiPointTrajectory:
    """Joint-space trajectory through multiple via-points.

    Concatenates piecewise polynomial segments with smooth transitions.

    Parameters
    ----------
    waypoints : list of arrays of shape (4,)
        Joint angle waypoints.
    durations : list of float
        Duration for each segment (len = len(waypoints) - 1).
    n_points_per_segment : int
        Points per segment.
    method : str
        'cubic' or 'quintic'.
    """

    def __init__(
        self,
        waypoints: list[NDArray[np.float64]],
        durations: list[float],
        n_points_per_segment: int = 50,
        method: str = "quintic",
    ) -> None:
        try:
            n_waypoints = len(waypoints)
        except TypeError as exc:
            raise ValueError(
                "waypoints must contain at least two arrays with shape (4,)"
            ) from exc
        if n_waypoints < 2:
            raise ValueError(
                "waypoints must contain at least two arrays with shape (4,)"
            )

        waypoint_values = [
            _finite_vector(waypoint, (4,), f"waypoints[{i}]")
            for i, waypoint in enumerate(waypoints)
        ]

        try:
            duration_values = np.asarray(durations, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"durations must contain exactly {n_waypoints - 1} values"
            ) from exc
        if duration_values.shape != (n_waypoints - 1,):
            raise ValueError(f"durations must contain exactly {n_waypoints - 1} values")
        if not np.all(np.isfinite(duration_values)) or np.any(duration_values <= 0):
            raise ValueError("durations must contain only finite positive values")
        if not np.isfinite(np.sum(duration_values)):
            raise ValueError("total duration must be finite")

        n_points_per_segment = _point_count(
            n_points_per_segment, "n_points_per_segment"
        )
        method = _method_name(method)

        segments = []
        for i, segment_duration in enumerate(duration_values):
            seg = JointTrajectory(
                waypoint_values[i],
                waypoint_values[i + 1],
                duration=float(segment_duration),
                n_points=n_points_per_segment,
                method=method,
            )
            segments.append(seg)

        # Concatenate (skip duplicate points at boundaries)
        t_offset = 0.0
        t_list, q_list, qd_list, qdd_list = [], [], [], []
        for i, seg in enumerate(segments):
            start = 1 if i > 0 else 0
            t_list.append(seg.t[start:] + t_offset)
            q_list.append(seg.q[start:])
            qd_list.append(seg.qd[start:])
            if seg.qdd is not None:
                qdd_list.append(seg.qdd[start:])
            t_offset += seg.duration

        timestamps = np.concatenate(t_list)
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(
                "segment durations must produce strictly increasing timestamps"
            )

        self.t = timestamps
        self.q = np.concatenate(q_list, axis=0)
        self.qd = np.concatenate(qd_list, axis=0)
        self.qdd = np.concatenate(qdd_list, axis=0) if qdd_list else None
        self.duration = t_offset
        self.n_points = len(self.t)
