import numpy as np
import pytest

import four_dof_arm
import four_dof_arm.trajectory as trajectory_facade
from four_dof_arm import Robot4DOF
from four_dof_arm.trajectory import (
    CartesianTrajectory,
    JointTrajectory,
    MultiPointTrajectory,
    cubic_polynomial,
    quintic_polynomial,
)
from four_dof_arm.trajectory.cartesian import (
    CartesianTrajectory as CartesianTrajectoryOwner,
)
from four_dof_arm.trajectory.joint import JointTrajectory as JointTrajectoryOwner
from four_dof_arm.trajectory.multi_point import (
    MultiPointTrajectory as MultiPointTrajectoryOwner,
)
from four_dof_arm.trajectory.polynomials import (
    cubic_polynomial as cubic_polynomial_owner,
)
from four_dof_arm.trajectory.polynomials import (
    quintic_polynomial as quintic_polynomial_owner,
)


class StubRobot:
    def __init__(self, results):
        self.results = iter(results)
        self.seeds = []
        self.targets = []
        self.joint_limits = np.array(
            [
                [-np.pi, np.pi],
                [-np.pi / 2, np.pi / 2],
                [-np.pi, np.pi],
                [-np.pi / 2, np.pi / 2],
            ]
        )

    def inverse_kinematics(self, target, q0=None, tol=0.1):
        self.targets.append(target.copy())
        self.seeds.append(q0.copy())
        return next(self.results)


def test_trajectory_facade_preserves_public_exports():
    expected = {
        "cubic_polynomial": cubic_polynomial_owner,
        "quintic_polynomial": quintic_polynomial_owner,
        "JointTrajectory": JointTrajectoryOwner,
        "CartesianTrajectory": CartesianTrajectoryOwner,
        "MultiPointTrajectory": MultiPointTrajectoryOwner,
    }

    assert trajectory_facade.__all__ == list(expected)
    for name, owner in expected.items():
        assert getattr(trajectory_facade, name) is owner
        assert getattr(four_dof_arm, name) is owner


@pytest.mark.parametrize("polynomial", [cubic_polynomial, quintic_polynomial])
def test_polynomial_endpoint_conditions(polynomial):
    duration = 2.5
    result = polynomial(-0.4, 1.2, np.array([0.0, duration]), duration)

    np.testing.assert_allclose(result[0], [-0.4, 1.2], atol=1e-14)
    np.testing.assert_allclose(result[1], [0.0, 0.0], atol=1e-14)
    if polynomial is quintic_polynomial:
        np.testing.assert_allclose(result[2], [0.0, 0.0], atol=1e-14)


def test_cubic_polynomial_matches_independent_interior_oracle():
    q, qd = cubic_polynomial(-2.0, 6.0, np.array([1.0]), 4.0)

    np.testing.assert_allclose(q, [-0.75], atol=1e-14)
    np.testing.assert_allclose(qd, [2.25], atol=1e-14)


def test_quintic_polynomial_matches_independent_interior_derivative_oracle():
    q, qd, qdd = quintic_polynomial(-2.0, 6.0, np.array([1.0]), 4.0)

    np.testing.assert_allclose(q, [-1.171875], atol=1e-14)
    np.testing.assert_allclose(qd, [2.109375], atol=1e-14)
    np.testing.assert_allclose(qdd, [2.8125], atol=1e-14)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("polynomial", [cubic_polynomial, quintic_polynomial])
def test_polynomial_rejects_invalid_duration(polynomial, duration):
    with pytest.raises(ValueError, match="finite positive"):
        polynomial(0.0, 1.0, np.array([0.0, 1.0]), duration)


@pytest.mark.parametrize(
    "times",
    [np.array([[0.0, 1.0]]), np.array([]), np.array([0.0, np.nan])],
)
def test_polynomial_rejects_invalid_time_vector(times):
    with pytest.raises(ValueError, match="t must"):
        cubic_polynomial(0.0, 1.0, times, 1.0)


@pytest.mark.parametrize("method", ["linear", "CUBIC", "", None])
def test_joint_trajectory_rejects_invalid_method(method):
    with pytest.raises(ValueError, match="method"):
        JointTrajectory(np.zeros(4), np.ones(4), method=method)


@pytest.mark.parametrize(
    ("q_start", "q_end"),
    [
        (np.zeros(3), np.ones(4)),
        (np.zeros(4), np.ones((4, 1))),
        (np.array([0.0, 0.0, 0.0, np.nan]), np.ones(4)),
    ],
)
def test_joint_trajectory_rejects_invalid_joint_vectors(q_start, q_end):
    with pytest.raises(ValueError):
        JointTrajectory(q_start, q_end)


def test_joint_trajectory_rejects_one_point():
    with pytest.raises(ValueError, match="n_points"):
        JointTrajectory(np.zeros(4), np.ones(4), n_points=1)


def test_cubic_joint_trajectory_shapes_and_interior_velocity():
    q_start = np.array([-1.0, 0.0, 1.0, 2.0])
    q_end = np.array([3.0, 2.0, 0.0, -2.0])
    trajectory = JointTrajectory(
        q_start, q_end, duration=2.0, n_points=5, method="cubic"
    )
    delta = q_end - q_start

    assert trajectory.t.shape == (5,)
    assert trajectory.q.shape == (5, 4)
    assert trajectory.qd.shape == (5, 4)
    assert trajectory.qdd is None
    np.testing.assert_allclose(trajectory.q[2], (q_start + q_end) / 2)
    np.testing.assert_allclose(trajectory.qd[1], 0.5625 * delta)
    np.testing.assert_allclose(trajectory.qd[2], 0.75 * delta)


def test_quintic_joint_trajectory_shapes_and_interior_derivatives():
    q_start = np.array([-1.0, 0.0, 1.0, 2.0])
    q_end = np.array([3.0, 2.0, 0.0, -2.0])
    trajectory = JointTrajectory(
        q_start, q_end, duration=2.0, n_points=5, method="quintic"
    )
    delta = q_end - q_start

    assert trajectory.t.shape == (5,)
    assert trajectory.q.shape == (5, 4)
    assert trajectory.qd.shape == (5, 4)
    assert trajectory.qdd is not None
    assert trajectory.qdd.shape == (5, 4)
    np.testing.assert_allclose(trajectory.q[2], (q_start + q_end) / 2)
    np.testing.assert_allclose(trajectory.qd[1], 0.52734375 * delta)
    np.testing.assert_allclose(trajectory.qd[2], 0.9375 * delta)
    np.testing.assert_allclose(trajectory.qdd[1], 1.40625 * delta)
    np.testing.assert_allclose(trajectory.qdd[2], np.zeros(4), atol=1e-14)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p_start": np.zeros(2)}, "p_start"),
        ({"p_end": np.array([0.0, np.nan, 0.0])}, "p_end"),
        ({"q_seed": np.zeros(3)}, "q_seed"),
        ({"q_seed": np.array([0.0, 0.0, 0.0, np.inf])}, "q_seed"),
        ({"duration": 0.0}, "duration"),
        ({"duration": np.nan}, "duration"),
        ({"n_points": 1}, "n_points"),
    ],
)
def test_cartesian_trajectory_rejects_invalid_inputs(kwargs, message):
    robot = StubRobot([])
    arguments = {
        "p_start": np.zeros(3),
        "p_end": np.ones(3),
        "n_points": 2,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CartesianTrajectory(robot, **arguments)


def test_cartesian_trajectory_uses_last_successful_warm_start_and_target_blend():
    seed = np.array([0.1, 0.2, 0.3, 0.4])
    failed_q = np.full(4, 9.0)
    first_success = np.array([0.5, 0.6, 0.7, 0.8])
    second_success = np.array([0.6, 0.7, 0.8, 0.9])
    robot = StubRobot(
        [
            (failed_q, False, 12.5),
            (first_success, True, 0.02),
            (second_success, True, 0.01),
        ]
    )
    p_start = np.array([1.0, 2.0, 3.0])
    p_end = np.array([5.0, 8.0, 11.0])

    trajectory = CartesianTrajectory(
        robot,
        p_start,
        p_end,
        n_points=3,
        q_seed=seed,
    )

    np.testing.assert_array_equal(trajectory.ik_success, [False, True, True])
    np.testing.assert_allclose(trajectory.ik_error, [12.5, 0.02, 0.01])
    np.testing.assert_allclose(trajectory.q[0], failed_q)
    np.testing.assert_allclose(robot.seeds, [seed, seed, first_success])
    np.testing.assert_allclose(
        robot.targets,
        [p_start, (p_start + p_end) / 2.0, p_end],
        atol=1e-14,
    )
    np.testing.assert_allclose(trajectory.positions, robot.targets)
    assert not trajectory.success


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ((np.zeros(3), True, 0.0), r"q with shape \(4,\)"),
        (([10**1000, 0.0, 0.0, 0.0], True, 0.0), r"q with shape \(4,\)"),
        ((np.zeros(4), True, np.zeros(1)), "scalar error"),
        ((np.zeros(4), True, 10**1000), "scalar error"),
        ((np.zeros(4), np.array([True]), 0.0), "boolean success"),
    ],
)
def test_cartesian_trajectory_rejects_malformed_ik_returns(result, message):
    robot = StubRobot([result])

    with pytest.raises(ValueError, match=message):
        CartesianTrajectory(robot, np.zeros(3), np.ones(3), n_points=2)


def test_cartesian_trajectory_marks_nonfinite_and_out_of_limit_returns_failed():
    seed = np.array([0.1, 0.2, 0.3, 0.4])
    robot = StubRobot(
        [
            (np.array([np.nan, 0.0, 0.0, 0.0]), True, 0.01),
            (np.array([0.0, np.pi, 0.0, 0.0]), True, 0.01),
            (np.zeros(4), True, np.inf),
        ]
    )

    trajectory = CartesianTrajectory(
        robot, np.zeros(3), np.ones(3), n_points=3, q_seed=seed
    )

    np.testing.assert_array_equal(trajectory.ik_success, [False, False, False])
    np.testing.assert_allclose(robot.seeds, [seed, seed, seed])
    assert not trajectory.success


def test_cartesian_trajectory_rejects_periodic_branch_crossing():
    first = np.deg2rad([179.0, 10.0, 20.0, -10.0])
    crossed = np.deg2rad([-178.0, 11.0, 21.0, -9.0])
    recovered = np.deg2rad([178.0, 12.0, 22.0, -8.0])
    robot = StubRobot(
        [
            (first, True, 0.01),
            (crossed, True, 0.01),
            (recovered, True, 0.01),
        ]
    )

    trajectory = CartesianTrajectory(robot, np.zeros(3), np.ones(3), n_points=3)

    np.testing.assert_array_equal(trajectory.ik_success, [True, False, True])
    np.testing.assert_allclose(robot.seeds, [np.zeros(4), first, first])
    assert np.rad2deg(abs(trajectory.q[1, 0] - trajectory.q[0, 0])) == pytest.approx(
        357.0
    )
    assert not trajectory.success


def test_cartesian_trajectory_reuses_equivalent_hard_limit_endpoint():
    positive_endpoint = np.array([np.pi, 0.0, 0.0, 0.0])
    negative_endpoint = np.array([-np.pi, 0.0, 0.0, 0.0])
    robot = StubRobot(
        [
            (positive_endpoint, True, 0.0),
            (negative_endpoint, True, 0.0),
        ]
    )

    trajectory = CartesianTrajectory(robot, np.zeros(3), np.ones(3), n_points=2)

    assert trajectory.success
    np.testing.assert_allclose(trajectory.q[:, 0], np.pi)


def test_cartesian_trajectory_real_robot_fk_residuals():
    robot = Robot4DOF()
    q_start = np.deg2rad([20.0, 30.0, -15.0, 10.0])
    q_end = np.deg2rad([25.0, 35.0, -10.0, 15.0])
    p_start = robot.forward_kinematics(q_start)[0]
    p_end = robot.forward_kinematics(q_end)[0]

    trajectory = CartesianTrajectory(
        robot,
        p_start,
        p_end,
        duration=1.5,
        n_points=7,
        q_seed=q_start,
    )

    assert trajectory.success
    residuals = [
        np.linalg.norm(robot.forward_kinematics(q)[0] - target)
        for q, target in zip(trajectory.q, trajectory.positions, strict=True)
    ]
    assert max(residuals) < 0.1


def test_multi_point_trajectory_requires_two_waypoints():
    with pytest.raises(ValueError, match="at least two"):
        MultiPointTrajectory([np.zeros(4)], [])


def test_multi_point_trajectory_rejects_duration_count_mismatch():
    waypoints = [np.zeros(4), np.ones(4), np.full(4, 2.0)]

    with pytest.raises(ValueError, match="exactly 2"):
        MultiPointTrajectory(waypoints, [1.0])


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, np.nan])
def test_multi_point_trajectory_rejects_invalid_segment_duration(duration):
    with pytest.raises(ValueError, match="finite positive"):
        MultiPointTrajectory([np.zeros(4), np.ones(4)], [duration])


def test_multi_point_trajectory_rejects_collapsed_timestamps():
    waypoints = [np.zeros(4), np.ones(4), np.full(4, 2.0)]

    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        MultiPointTrajectory(
            waypoints,
            [1.0, 1e-17],
            n_points_per_segment=2,
            method="cubic",
        )


def test_multi_point_trajectory_rejects_invalid_waypoint_and_point_count():
    with pytest.raises(ValueError, match=r"waypoints\[1\]"):
        MultiPointTrajectory([np.zeros(4), np.ones(3)], [1.0])

    with pytest.raises(ValueError, match="n_points_per_segment"):
        MultiPointTrajectory([np.zeros(4), np.ones(4)], [1.0], 1)


def test_multi_point_trajectory_rejects_invalid_method():
    with pytest.raises(ValueError, match="method"):
        MultiPointTrajectory([np.zeros(4), np.ones(4)], [1.0], method="linear")


@pytest.mark.parametrize("method", ["cubic", "quintic"])
def test_multi_point_trajectory_preserves_endpoints_and_monotonic_time(method):
    waypoints = [np.zeros(4), np.ones(4), np.array([2.0, 1.0, 0.0, -1.0])]
    trajectory = MultiPointTrajectory(
        waypoints,
        [0.75, 1.25],
        n_points_per_segment=5,
        method=method,
    )

    np.testing.assert_allclose(trajectory.q[0], waypoints[0], atol=1e-14)
    np.testing.assert_allclose(trajectory.q[4], waypoints[1], atol=1e-14)
    np.testing.assert_allclose(trajectory.q[-1], waypoints[-1], atol=1e-14)
    np.testing.assert_allclose(trajectory.qd[4], np.zeros(4), atol=1e-14)
    assert np.all(np.diff(trajectory.t) > 0)
    assert trajectory.t[-1] == pytest.approx(2.0)
    assert trajectory.n_points == 9
    assert (trajectory.qdd is not None) is (method == "quintic")
    if trajectory.qdd is not None:
        np.testing.assert_allclose(trajectory.qdd[4], np.zeros(4), atol=1e-14)
        np.testing.assert_allclose(
            trajectory.qdd[5],
            3.6 * (waypoints[2] - waypoints[1]),
            atol=1e-14,
        )
