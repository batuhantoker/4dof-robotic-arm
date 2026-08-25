import numpy as np
import pytest

from four_dof_arm import Robot4DOF
from four_dof_arm.robot import Robot4DOF as FacadeRobot4DOF
from four_dof_arm.robot.model import Robot4DOF as ModelRobot4DOF


def test_robot_facade_exports_model_class() -> None:
    assert Robot4DOF is FacadeRobot4DOF is ModelRobot4DOF


def test_forward_kinematics_matches_joint_chain() -> None:
    robot = Robot4DOF()
    q = np.deg2rad([30.0, 45.0, -30.0, 20.0])

    position, rotation = robot.forward_kinematics(q)
    rotations = robot.rotation_matrices(q)

    np.testing.assert_allclose(position, robot.joint_positions(q)[-1], atol=1e-12)
    np.testing.assert_allclose(rotation, np.linalg.multi_dot(rotations), atol=1e-12)


def test_rotation_matrices_match_independent_goldens() -> None:
    robot = Robot4DOF()
    q = np.array([np.pi / 2, -np.pi / 2, np.pi, np.pi / 2])
    expected = (
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    )

    for actual, golden in zip(robot.rotation_matrices(q), expected, strict=True):
        np.testing.assert_allclose(actual, golden, atol=1e-15)


def test_joint_positions_match_independent_golden() -> None:
    robot = Robot4DOF()
    q = np.array([np.pi / 2, np.pi / 2, 0.0, 0.0])
    expected_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 55.0],
            [0.0, 50.0, 55.0],
            [0.0, 150.0, 55.0],
            [90.0, 150.0, 55.0],
        ]
    )
    expected_rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    position, rotation = robot.forward_kinematics(q)

    np.testing.assert_allclose(robot.joint_positions(q), expected_positions, atol=1e-12)
    np.testing.assert_allclose(position, expected_positions[-1], atol=1e-12)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-15)


def test_analytical_jacobian_matches_finite_difference() -> None:
    robot = Robot4DOF()
    q = np.deg2rad([30.0, 45.0, -30.0, 20.0])

    np.testing.assert_allclose(
        robot.jacobian_analytical(q),
        robot.jacobian(q),
        rtol=1e-5,
        atol=1e-4,
    )


def test_inverse_kinematics_reaches_valid_target() -> None:
    robot = Robot4DOF()
    target, _ = robot.forward_kinematics(np.deg2rad([20.0, 30.0, -15.0, 10.0]))

    q, success, error = robot.inverse_kinematics(target)
    reached, _ = robot.forward_kinematics(q)

    assert success
    assert error < 1e-4
    np.testing.assert_allclose(reached, target, atol=1e-4)
    assert np.all(q >= robot.joint_limits[:, 0])
    assert np.all(q <= robot.joint_limits[:, 1])


def test_inverse_kinematics_escapes_zero_seed_singularity() -> None:
    robot = Robot4DOF()
    target = np.array([240.0, 0.0, 55.0])

    q, success, error = robot.inverse_kinematics(target)

    assert success
    assert error < 1e-4
    np.testing.assert_allclose(robot.forward_kinematics(q)[0], target, atol=1e-4)


def test_inverse_kinematics_clips_initial_seed_to_hard_limits() -> None:
    robot = Robot4DOF()
    expected = np.array(
        [
            np.pi,
            np.pi / 2,
            -np.pi,
            -np.pi / 2,
        ]
    )
    seed = np.array([4.5 * np.pi, 10.0, -4.5 * np.pi, -10.0])
    target, _ = robot.forward_kinematics(expected)

    q, success, error = robot.inverse_kinematics(target, q0=seed)

    assert success
    assert error < 1e-12
    np.testing.assert_allclose(q, expected, atol=1e-12)


def test_inverse_kinematics_keeps_failed_result_within_limits() -> None:
    robot = Robot4DOF()

    q, success, _ = robot.inverse_kinematics(
        np.array([10_000.0, -10_000.0, 10_000.0]),
        q0=np.array([10.0, 10.0, 10.0, 10.0]),
        max_iter=10,
    )

    assert not success
    assert np.all(q >= robot.joint_limits[:, 0])
    assert np.all(q <= robot.joint_limits[:, 1])


@pytest.mark.parametrize(
    "method_name",
    [
        "rotation_matrices",
        "forward_kinematics",
        "joint_positions",
        "jacobian",
        "jacobian_analytical",
    ],
)
def test_joint_methods_reject_invalid_q_shape(method_name: str) -> None:
    robot = Robot4DOF()

    with pytest.raises(ValueError, match=r"q.*shape \(4,\)"):
        getattr(robot, method_name)(np.zeros((4, 1)))


@pytest.mark.parametrize(
    "target, q0",
    [
        (np.zeros(4), None),
        (np.array([1.0, 2.0, np.nan]), None),
        (np.zeros(3), np.zeros(3)),
        (np.zeros(3), np.array([0.0, 0.0, np.inf, 0.0])),
    ],
)
def test_inverse_kinematics_rejects_invalid_vectors(
    target: np.ndarray, q0: np.ndarray | None
) -> None:
    with pytest.raises(ValueError):
        Robot4DOF().inverse_kinematics(target, q0=q0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tol": 0.0}, "tol"),
        ({"tol": np.nan}, "tol"),
        ({"damping": 0.0}, "damping"),
        ({"damping": np.inf}, "damping"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
    ],
)
def test_inverse_kinematics_rejects_invalid_solver_options(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Robot4DOF().inverse_kinematics(np.zeros(3), **kwargs)


@pytest.mark.parametrize("eps", [0.0, -1.0, np.inf, np.nan])
def test_jacobian_rejects_invalid_step(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        Robot4DOF().jacobian(np.zeros(4), eps=eps)


@pytest.mark.parametrize("name", ["Lax", "Laz", "Lb", "Lt"])
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        np.inf,
        -np.inf,
        np.nan,
        True,
        np.bool_(False),
        np.array(True),
    ],
)
def test_robot_rejects_nonpositive_nonfinite_and_boolean_link_lengths(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=rf"{name}.*positive finite"):
        Robot4DOF(**{name: value})


def test_robot_numeric_converters_translate_overflow_to_value_error() -> None:
    too_large = 10**1000

    with pytest.raises(ValueError, match="Lax"):
        Robot4DOF(Lax=too_large)

    with pytest.raises(ValueError, match=r"q.*shape \(4,\)"):
        Robot4DOF().forward_kinematics([too_large, 0.0, 0.0, 0.0])


def test_inverse_kinematics_multi_is_reproducible_with_generator() -> None:
    robot = Robot4DOF()
    target = np.array([1000.0, -1000.0, 1000.0])
    q0 = np.array([0.1, 0.2, 0.3, 0.4])

    first = robot.inverse_kinematics_multi(
        target,
        n_restarts=4,
        q0=q0,
        rng=np.random.default_rng(123),
        max_iter=3,
    )
    second = robot.inverse_kinematics_multi(
        target,
        n_restarts=4,
        q0=q0,
        rng=np.random.default_rng(123),
        max_iter=3,
    )

    np.testing.assert_allclose(first[0], second[0])
    assert first[1:] == second[1:]


def test_inverse_kinematics_multi_uses_initial_then_seeded_random_guesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot4DOF()
    target = np.array([10.0, 20.0, 30.0])
    initial = np.array([0.1, 0.2, 0.3, 0.4])
    seeds = []

    def fake_inverse_kinematics(target, q0=None, **kwargs):
        del target, kwargs
        seeds.append(None if q0 is None else q0.copy())
        return np.zeros(4), False, 1.0

    monkeypatch.setattr(robot, "inverse_kinematics", fake_inverse_kinematics)
    robot.inverse_kinematics_multi(
        target,
        n_restarts=4,
        q0=initial,
        rng=np.random.default_rng(123),
    )

    expected_rng = np.random.default_rng(123)
    expected = [initial]
    expected.extend(
        expected_rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        for _ in range(3)
    )
    np.testing.assert_allclose(seeds, expected)


def test_inverse_kinematics_multi_selects_lowest_error_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot4DOF()
    expected_q = np.full(4, 2.0)
    results = iter(
        [
            (np.full(4, 1.0), False, 3.0),
            (expected_q, False, 0.5),
            (np.full(4, 3.0), False, 2.0),
        ]
    )

    def fake_inverse_kinematics(target, **kwargs):
        del target, kwargs
        return next(results)

    monkeypatch.setattr(robot, "inverse_kinematics", fake_inverse_kinematics)

    q, success, error = robot.inverse_kinematics_multi(
        np.zeros(3), n_restarts=3, rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(q, expected_q)
    assert not success
    assert error == 0.5


def test_inverse_kinematics_multi_forwards_solver_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot4DOF()
    target = np.array([10.0, 20.0, 30.0])
    initial = np.array([0.1, 0.2, 0.3, 0.4])
    calls = []

    def fake_inverse_kinematics(target, q0=None, tol=None, max_iter=None, damping=None):
        calls.append((target.copy(), q0.copy(), tol, max_iter, damping))
        return np.zeros(4), False, 1.0

    monkeypatch.setattr(robot, "inverse_kinematics", fake_inverse_kinematics)
    robot.inverse_kinematics_multi(
        target,
        n_restarts=1,
        q0=initial,
        tol=0.25,
        max_iter=17,
        damping=0.75,
    )

    assert len(calls) == 1
    actual_target, actual_q0, tol, max_iter, damping = calls[0]
    np.testing.assert_array_equal(actual_target, target)
    np.testing.assert_array_equal(actual_q0, initial)
    assert (tol, max_iter, damping) == (0.25, 17, 0.75)


def test_inverse_kinematics_multi_stops_at_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot4DOF()
    successful_q = np.array([0.2, 0.3, 0.4, 0.5])
    results = iter(
        [
            (np.zeros(4), False, 1.0),
            (successful_q, True, 5e-5),
            (np.ones(4), True, 1e-5),
        ]
    )
    call_count = 0

    def fake_inverse_kinematics(target, **kwargs):
        nonlocal call_count
        del target, kwargs
        call_count += 1
        return next(results)

    monkeypatch.setattr(robot, "inverse_kinematics", fake_inverse_kinematics)

    q, success, error = robot.inverse_kinematics_multi(
        np.zeros(3), n_restarts=3, rng=np.random.default_rng(0)
    )

    assert call_count == 2
    np.testing.assert_array_equal(q, successful_q)
    assert success
    assert error == 5e-5


def test_inverse_kinematics_seeded_restarts_recover_reachable_target() -> None:
    robot = Robot4DOF()
    target_q = np.array([1.8, 0.16, -1.61, -0.52])
    target = robot.forward_kinematics(target_q)[0]
    trapped_seed = np.array([np.pi, np.pi / 2, np.pi, np.pi / 2])

    _, first_success, _ = robot.inverse_kinematics(target, q0=trapped_seed, max_iter=5)
    q, success, error = robot.inverse_kinematics_multi(
        target,
        n_restarts=4,
        q0=trapped_seed,
        rng=np.random.default_rng(7),
        max_iter=5,
    )

    assert not first_success
    assert success
    assert error < 1e-4
    np.testing.assert_allclose(robot.forward_kinematics(q)[0], target, atol=1e-4)


def test_inverse_kinematics_multi_rejects_invalid_restart_count() -> None:
    with pytest.raises(ValueError, match="n_restarts"):
        Robot4DOF().inverse_kinematics_multi(np.zeros(3), n_restarts=0)


def test_sample_workspace_matches_seeded_joint_samples() -> None:
    robot = Robot4DOF()
    sample_rng = np.random.default_rng(42)
    expected_rng = np.random.default_rng(42)
    expected = []
    for _ in range(5):
        q = expected_rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        expected.append(robot.forward_kinematics(q)[0])

    points = robot.sample_workspace(5, rng=sample_rng)

    assert points.shape == (5, 3)
    np.testing.assert_allclose(points, expected)


def test_sample_workspace_rejects_invalid_sample_count() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        Robot4DOF().sample_workspace(0)
