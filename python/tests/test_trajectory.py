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


@pytest.mark.parametrize(
    ("polynomial", "duration", "displacement"),
    [
        (cubic_polynomial, 1e-104, 1.0),
        (quintic_polynomial, 1e-62, 1.0),
        (cubic_polynomial, 1e65, 1.0),
        (quintic_polynomial, 1e65, 1.0),
        (quintic_polynomial, 1e-162, 1e-20),
    ],
)
def test_polynomial_extreme_representable_durations(polynomial, duration, displacement):
    times = np.linspace(0.0, duration, 5)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = polynomial(0.0, displacement, times, duration)

    assert all(np.all(np.isfinite(profile)) for profile in result)
    if polynomial is cubic_polynomial:
        expected_q = [0.0, 0.15625, 0.5, 0.84375, 1.0]
        expected_qd = [0.0, 1.125, 1.5, 1.125, 0.0]
    else:
        expected_q = [0.0, 0.103515625, 0.5, 0.896484375, 1.0]
        expected_qd = [0.0, 1.0546875, 1.875, 1.0546875, 0.0]
        np.testing.assert_allclose(
            result[2] * duration * duration / displacement,
            [0.0, 5.625, 0.0, -5.625, 0.0],
            atol=1e-14,
        )
    np.testing.assert_allclose(result[0] / displacement, expected_q, atol=1e-14)
    np.testing.assert_allclose(
        result[1] * duration / displacement, expected_qd, atol=1e-14
    )


@pytest.mark.parametrize(
    ("polynomial", "duration"),
    [(cubic_polynomial, 1e-309), (quintic_polynomial, 1e-160)],
)
def test_polynomial_rejects_unrepresentable_output(polynomial, duration):
    with pytest.raises(ValueError, match="output.*finite"):
        polynomial(0.0, 1.0, np.linspace(0.0, duration, 5), duration)


@pytest.mark.parametrize("polynomial", [cubic_polynomial, quintic_polynomial])
def test_polynomial_rejects_nonfinite_extrapolated_output(polynomial):
    with pytest.raises(ValueError, match="output.*finite"):
        polynomial(0.0, 1.0, np.array([1e200]), 1.0)


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


@pytest.mark.parametrize(
    "max_joint_step",
    [
        0.0,
        -0.5,
        np.inf,
        -np.inf,
        np.nan,
        True,
        np.bool_(False),
        np.array(True),
        [0.5],
        np.full(4, 0.5),
        None,
        10**1000,
    ],
)
def test_cartesian_trajectory_rejects_invalid_max_joint_step(max_joint_step):
    robot = StubRobot([])

    with pytest.raises(ValueError, match="max_joint_step.*positive finite scalar"):
        CartesianTrajectory(
            robot, np.zeros(3), np.ones(3), max_joint_step=max_joint_step
        )
    assert not robot.seeds


@pytest.mark.parametrize("joint", range(4))
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("max_joint_step", [0.125, 0.5, 0.75])
def test_cartesian_trajectory_bounds_all_joints_from_last_accepted_solution(
    joint, direction, max_joint_step
):
    first = np.zeros(4)
    rejected = first.copy()
    rejected[joint] = direction * np.nextafter(max_joint_step, np.inf)
    boundary = first.copy()
    boundary[joint] = direction * max_joint_step
    following = 2.0 * boundary
    seed = np.ones(4)
    robot = StubRobot([(q, True, 0.01) for q in (first, rejected, boundary, following)])

    trajectory = CartesianTrajectory(
        robot,
        np.zeros(3),
        np.ones(3),
        n_points=4,
        q_seed=seed,
        max_joint_step=max_joint_step,
    )

    np.testing.assert_array_equal(trajectory.ik_success, [True, False, True, True])
    np.testing.assert_array_equal(trajectory.q, [first, rejected, boundary, following])
    np.testing.assert_array_equal(robot.seeds, [seed, first, first, boundary])
    np.testing.assert_allclose(trajectory.ik_error, 0.01)
    assert not trajectory.success


def test_cartesian_trajectory_uses_last_successful_warm_start_and_target_blend():
    seed = np.array([0.1, 0.2, 0.3, 0.4])
    failed_q = np.full(4, 9.0)
    first_success = np.array([0.8, 0.9, 1.0, 1.1])
    second_success = np.array([0.9, 1.0, 1.1, 1.2])
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


@pytest.mark.parametrize("force_nonlocal_solution", [False, True])
def test_cartesian_trajectory_real_robot_near_limits(
    monkeypatch, force_nonlocal_solution
):
    robot = Robot4DOF()
    q_a = np.array([0.0, 1.569, -1.414, -np.pi / 2])
    q_b = np.array([0.016, np.pi / 2, -1.409, -1.557])
    p_start = robot.forward_kinematics(q_a)[0]
    p_end = robot.forward_kinematics(q_b)[0]
    solve = robot.inverse_kinematics
    seeds = []

    def inverse_kinematics(target, q0=None, tol=0.1):
        seeds.append(q0.copy())
        if force_nonlocal_solution and len(seeds) == 2:
            # Exercise acceptance independently of the warm-start solver's branch.
            q, success, error = solve(target, q0=np.zeros(4), tol=tol)
            assert success
            assert error < tol
            assert np.max(np.abs(q - q0)) > 0.5
            return q, success, error
        return solve(target, q0=q0, tol=tol)

    monkeypatch.setattr(robot, "inverse_kinematics", inverse_kinematics)
    trajectory = CartesianTrajectory(robot, p_start, p_end, n_points=51, q_seed=q_a)

    assert trajectory.max_joint_step == 0.5
    assert trajectory.ik_success[0]
    accepted = trajectory.q[trajectory.ik_success]
    assert np.all(np.abs(np.diff(accepted, axis=0)) <= 0.5)
    assert np.all(accepted >= robot.joint_limits[:, 0])
    assert np.all(accepted <= robot.joint_limits[:, 1])
    for q, target in zip(
        accepted, trajectory.positions[trajectory.ik_success], strict=True
    ):
        assert np.linalg.norm(robot.forward_kinematics(q)[0] - target) < 0.1
    if force_nonlocal_solution:
        assert not trajectory.ik_success[1]
        assert trajectory.ik_error[1] < 0.1
        assert not trajectory.success
        np.testing.assert_array_equal(seeds[2], trajectory.q[0])
    if trajectory.success:
        assert np.all(np.abs(np.diff(trajectory.q, axis=0)) <= 0.5)


@pytest.mark.parametrize("trajectory_type", [JointTrajectory, MultiPointTrajectory])
@pytest.mark.parametrize(
    ("method", "duration"),
    [("cubic", 1e-104), ("quintic", 1e-62), ("cubic", 1e65), ("quintic", 1e65)],
)
def test_joint_trajectory_constructors_support_extreme_durations(
    trajectory_type, method, duration
):
    if trajectory_type is JointTrajectory:
        trajectory = trajectory_type(
            np.zeros(4), np.ones(4), duration=duration, n_points=5, method=method
        )
    else:
        trajectory = trajectory_type(
            [np.zeros(4), np.ones(4), np.full(4, 2.0)],
            [duration, duration],
            n_points_per_segment=5,
            method=method,
        )

    assert np.all(np.diff(trajectory.t) > 0.0)
    assert np.all(np.isfinite(trajectory.q))
    assert np.all(np.isfinite(trajectory.qd))
    np.testing.assert_allclose(trajectory.q[0], 0.0)
    np.testing.assert_allclose(trajectory.q[4], 1.0)
    np.testing.assert_allclose(
        trajectory.qd[1] * duration, 1.125 if method == "cubic" else 1.0546875
    )
    if trajectory.qdd is not None:
        assert np.all(np.isfinite(trajectory.qdd))
        np.testing.assert_allclose(trajectory.qdd[1] * duration * duration, 5.625)


@pytest.mark.parametrize("trajectory_type", [JointTrajectory, MultiPointTrajectory])
@pytest.mark.parametrize(
    ("method", "duration"), [("cubic", 1e-309), ("quintic", 1e-160)]
)
def test_joint_trajectory_constructors_reject_unrepresentable_output(
    trajectory_type, method, duration
):
    with pytest.raises(ValueError, match="output.*finite"):
        if trajectory_type is JointTrajectory:
            trajectory_type(
                np.zeros(4), np.ones(4), duration=duration, n_points=5, method=method
            )
        else:
            trajectory_type(
                [np.zeros(4), np.ones(4)],
                [duration],
                n_points_per_segment=5,
                method=method,
            )


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
