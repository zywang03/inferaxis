"""Shared dummy implementations for tests."""

from __future__ import annotations

import threading

import inferaxis as infra
import numpy as np

from inferaxis import Action, ChunkRequest, Frame
from inferaxis.core.schema import ComponentSpec, PolicyOutputSpec, PolicySpec, RobotSpec


def demo_image() -> np.ndarray:
    """Return one small deterministic image tensor for tests/examples."""

    return np.zeros((2, 2, 3), dtype=np.uint8)


def assert_array_equal(testcase: object, actual: object, expected: object) -> None:
    """Assert one array-like payload matches the expected numeric values."""

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


class DummyRobot:
    def __init__(self) -> None:
        self.last_action: Action | None = None

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="dummy_robot",
            image_keys=["front_rgb"],
            components=[
                ComponentSpec(
                    name="arm",
                    type="arm",
                    dof=6,
                    command=["cartesian_pose_delta"],
                )
            ],
        )

    def get_obs(self) -> Frame:
        return Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: Action) -> None:
        self.last_action = action

    def reset(self) -> Frame:
        return self.get_obs()


class DummyPolicy:
    def get_spec(self) -> PolicySpec:
        return PolicySpec(
            name="dummy_model",
            required_image_keys=["front_rgb"],
            required_state_keys=["arm"],
            outputs=[
                PolicyOutputSpec(
                    target="arm",
                    command="cartesian_pose_delta",
                    dim=6,
                )
            ],
        )

    def reset(self) -> None:
        return None

    def infer(
        self,
        obs: Frame,
        request: ChunkRequest,
    ) -> Action:
        del obs, request
        return Action.single(
            target="arm",
            command="cartesian_pose_delta",
            value=np.zeros(6, dtype=np.float64),
        )


def arm_value(action: infra.Action) -> float:
    """Return the first arm dimension from one action."""

    command = action.get_command("arm")
    assert command is not None
    return float(command.value[0])


def accept_scheduler_chunk(
    scheduler: object,
    actions: list[infra.Action],
    *,
    request_step: int | None = None,
    current_raw_step: int | None = None,
    source_plan_length: int | None = None,
) -> None:
    raw_buffer = scheduler._raw_buffer  # type: ignore[attr-defined]
    raw_step = raw_buffer.global_step
    raw_buffer.accept_chunk(
        actions=actions,
        request_step=raw_step if request_step is None else request_step,
        current_raw_step=raw_step if current_raw_step is None else current_raw_step,
        source_plan_length=len(actions)
        if source_plan_length is None
        else source_plan_length,
    )
    scheduler._execution_cursor.reset()  # type: ignore[attr-defined]


def scheduler_raw_actions(scheduler: object) -> list[infra.Action]:
    return scheduler._raw_buffer.remaining_actions()  # type: ignore[attr-defined]


def scheduler_execution_actions(scheduler: object) -> list[infra.Action]:
    return scheduler._execution_cursor.remaining_segment_actions()  # type: ignore[attr-defined]


def arm_action(value: float) -> infra.Action:
    """Build one single-arm action for compact scheduler tests."""

    return infra.Action.single(
        target="arm",
        command="cartesian_pose_delta",
        value=[value] * 6,
    )


def gripper_value(action: infra.Action, target: str = "gripper") -> float:
    """Return the first gripper dimension from one action."""

    command = action.get_command(target)
    assert command is not None
    return float(command.value[0])


def arm_and_gripper_action(
    *,
    arm: float,
    gripper: float,
    gripper_command: str = infra.BuiltinCommandKind.GRIPPER_POSITION,
) -> infra.Action:
    """Build one action containing both arm and gripper commands."""

    return infra.Action(
        commands={
            "arm": infra.Command(
                command="cartesian_pose_delta",
                value=[arm] * 6,
            ),
            "gripper": infra.Command(
                command=gripper_command,
                value=[gripper],
            ),
        }
    )


def make_chunk_request(**kwargs: object) -> infra.ChunkRequest:
    """Build one modern ChunkRequest for tests."""

    request_step = int(kwargs.pop("request_step", 0))
    request_time_s = float(kwargs.pop("request_time_s", 0.0))
    active_chunk_length = int(kwargs.pop("active_chunk_length", 0))
    remaining_steps = int(kwargs.pop("remaining_steps", 0))
    latency_steps = int(kwargs.pop("latency_steps", 0))
    prev_action_chunk = kwargs.pop("prev_action_chunk", None)
    inference_delay = kwargs.pop("inference_delay", None)
    execute_horizon = kwargs.pop("execute_horizon", None)
    action_prefix = kwargs.pop("action_prefix", None)
    prefix_length = kwargs.pop("prefix_length", None)
    rtc_args = kwargs.pop("rtc_args", None)

    if kwargs:
        raise AssertionError(
            f"Unexpected ChunkRequest test fields: {sorted(kwargs.keys())!r}"
        )

    return infra.ChunkRequest(
        request_step=request_step,
        request_time_s=request_time_s,
        active_chunk_length=active_chunk_length,
        remaining_steps=remaining_steps,
        latency_steps=latency_steps,
        prev_action_chunk=prev_action_chunk,  # type: ignore[arg-type]
        inference_delay=inference_delay,  # type: ignore[arg-type]
        execute_horizon=execute_horizon,  # type: ignore[arg-type]
        action_prefix=action_prefix,  # type: ignore[arg-type]
        prefix_length=prefix_length,  # type: ignore[arg-type]
        rtc_args=rtc_args,  # type: ignore[arg-type]
    )


class RuntimeRobot:
    """Tiny plain local executor used by inference-runtime tests."""

    def __init__(self) -> None:
        self.last_action: infra.Action | None = None

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="runtime_robot",
            image_keys=["front_rgb"],
            components=[
                ComponentSpec(
                    name="arm",
                    type="arm",
                    dof=6,
                    command=["cartesian_pose_delta"],
                )
            ],
        )

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: infra.Action) -> None:
        self.last_action = action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class RuntimePolicy:
    """Policy used by runtime tests through one infer(frame, request) entrypoint."""

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs
        start = float(request.request_step + 1)
        return [
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[start] * 6,
            ),
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[start + 1.0] * 6,
            ),
        ]


class RtcLoggingChunkPolicy:
    """Policy that records RTC hints from each chunk request."""

    def __init__(self) -> None:
        self.requests: list[infra.ChunkRequest] = []
        self.second_request_seen = threading.Event()

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs
        self.requests.append(request)
        if len(self.requests) >= 2:
            self.second_request_seen.set()

        base = 1.0 + float(request.request_step)
        return [arm_action(base + float(offset)) for offset in range(4)]


class RecordingRuntimePolicy(RuntimePolicy):
    """Runtime policy that records request metadata for comparison tests."""

    def __init__(self) -> None:
        super().__init__()
        self.request_summaries: list[dict[str, object]] = []

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        self.request_summaries.append(
            {
                "request_step": request.request_step,
                "active_chunk_length": request.active_chunk_length,
                "remaining_steps": request.remaining_steps,
                "latency_steps": request.latency_steps,
                "has_rtc_args": request.rtc_args is not None,
                "prev_action_chunk": (
                    None
                    if request.prev_action_chunk is None
                    else [arm_value(action) for action in request.prev_action_chunk]
                ),
                "inference_delay": request.inference_delay,
                "execute_horizon": request.execute_horizon,
            }
        )
        return super().infer(obs, request)


class PlainRuntimeExecutor:
    """Plain local executor without inferaxis mixins."""

    def __init__(self) -> None:
        self.last_action: infra.Action | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: infra.Action) -> None:
        self.last_action = action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class SingleActionChunkPolicy:
    """Policy that returns one action directly from infer(frame, request)."""

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> infra.Action:
        del obs, request
        value = float(1 + self.step_index)
        self.step_index += 1
        return infra.Action.single(
            target="arm",
            command="cartesian_pose_delta",
            value=[value] * 6,
        )


class PlanningSource:
    """Simple source exposing one future-action provider."""

    def __init__(self) -> None:
        self.plan_base = 10.0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs
        base = self.plan_base + float(request.request_step)
        return [
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[base] * 6,
            ),
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[base + 1.0] * 6,
            ),
        ]


class DeterministicClock:
    """Clock that returns one predefined timestamp per call."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("DeterministicClock exhausted.") from exc


def make_profile_clock(
    *,
    step_durations: list[float],
    inference_durations: list[float],
    observe_duration: float = 0.001,
    inter_step_gap: float = 0.001,
) -> DeterministicClock:
    """Build one deterministic clock for request timing tests."""

    timestamps: list[float] = []
    current = 0.0
    for step_duration, inference_duration in zip(
        step_durations,
        inference_durations,
        strict=True,
    ):
        inference_start = current + observe_duration
        timestamps.extend(
            [
                current,
                inference_start,
                inference_start + inference_duration,
                current + step_duration,
            ]
        )
        current += step_duration + inter_step_gap
    return DeterministicClock(timestamps)
