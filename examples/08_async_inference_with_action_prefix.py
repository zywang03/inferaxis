"""Example 8: async inference with action-prefix request hints.

Run with:

    PYTHONPATH=src python examples/08_async_inference_with_action_prefix.py
"""

from __future__ import annotations

import inferaxis as infra
import numpy as np
from inferaxis.core.transform import action_to_dict


class YourRobot:
    """Plain local executor used by the runtime loop."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={
                "YOUR_OWN_arm": np.full(6, 0.25, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        )

    def send_action(self, action: infra.Action) -> infra.Action:
        self.last_native_action = action
        return action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class YourPrefixPolicy:
    """Async-capable source that consumes prefix-style request hints."""

    def __init__(self) -> None:
        self.last_prefix_summary: dict[str, int | bool | None] | None = None

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        gripper_pos = float(obs.state["YOUR_OWN_gripper"][0])
        action_prefix = request.action_prefix
        prefix_length = request.prefix_length
        self.last_prefix_summary = {
            "has_action_prefix": action_prefix is not None,
            "action_prefix_len": 0 if action_prefix is None else len(action_prefix),
            "prefix_length": prefix_length,
        }

        plan: list[infra.Action] = []
        next_base = float(request.request_step * 3.0)

        while len(plan) < 4:
            plan.append(
                infra.Action(
                    commands={
                        "YOUR_OWN_arm": infra.Command(
                            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                            value=np.full(6, next_base, dtype=np.float64),
                        ),
                        "YOUR_OWN_gripper": infra.Command(
                            command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                            value=np.array(
                                [max(0.0, min(1.0, 1.0 - gripper_pos))],
                                dtype=np.float64,
                            ),
                        ),
                    }
                )
            )
            next_base += 3.0
        return plan


def main() -> None:
    robot = YourRobot()
    policy = YourPrefixPolicy()
    runtime = infra.InferenceRuntime.async_realtime(
        control_hz=50.0,
        execution_steps=3,
        enable_rtc=True,
        slow_rtc_bootstrap="warn",
    )

    for step_index in range(5):
        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        print(
            "step:",
            step_index,
            "action:",
            action_to_dict(result.action),
            "plan_refreshed:",
            result.plan_refreshed,
            "wait:",
            f"{result.control_wait_s:.4f}",
            "prefix:",
            policy.last_prefix_summary,
        )

    runtime.close()
    print("native_robot_received:", robot.last_native_action)
    print("example 8 passed.")


if __name__ == "__main__":
    main()
