"""Runtime-engine tests for inferaxis inference runtime utilities."""

from __future__ import annotations

import inspect
import threading
import unittest

import inferaxis as infra
from inferaxis.core.errors import InterfaceValidationError

from helpers import (
    PlanningSource,
    PlainRuntimeExecutor,
    RtcLoggingChunkPolicy,
    RuntimePolicy,
    RuntimeRobot,
    SingleActionChunkPolicy,
    arm_action,
    arm_value,
    scheduler_raw_actions,
)


class _BlockingPendingPolicy:
    def __init__(self) -> None:
        self.call_count = 0
        self.blocking_call_started = threading.Event()
        self.release_blocking_call = threading.Event()

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs, request
        self.call_count += 1
        if self.call_count > 1:
            self.blocking_call_started.set()
            self.release_blocking_call.wait(timeout=2.0)
        return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]


class RuntimeEngineTests(unittest.TestCase):
    """Coverage for sync/async runtime engine flow."""

    def test_runtime_accepts_scalar_ensemble_weight(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            ensemble_weight=0.5,
        )

        self.assertEqual(runtime.ensemble_weight, 0.5)

    def test_runtime_accepts_weight_schedule_ensemble_weight(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            ensemble_weight=(0.2, 0.8),
        )

        self.assertEqual(runtime.ensemble_weight, (0.2, 0.8))

    def test_runtime_defaults_to_no_ensemble_weight(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
        )

        self.assertIsNone(runtime.ensemble_weight)

    def test_inference_runtime_public_exports_include_runtime_only(self) -> None:
        import inferaxis.runtime.inference as inference_module

        self.assertIs(inference_module.InferenceRuntime, infra.InferenceRuntime)
        self.assertIs(inference_module.InferenceMode, infra.InferenceMode)
        self.assertFalse(hasattr(inference_module, "profile_sync_inference"))
        self.assertFalse(hasattr(inference_module, "recommend_inference_mode"))
        self.assertFalse(hasattr(infra, "profile_sync_inference"))
        self.assertFalse(hasattr(infra, "recommend_inference_mode"))

    def test_runtime_profile_requires_async_mode(self) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                profile=True,
            )

        self.assertIn("mode=ASYNC", str(ctx.exception))

    def test_runtime_rejects_invalid_profile_output_dir(self) -> None:
        with self.assertRaises(InterfaceValidationError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile_output_dir=object(),  # type: ignore[arg-type]
            )

    def test_runtime_accepts_interpolation_steps(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            interpolation_steps=2,
        )

        self.assertEqual(runtime.interpolation_steps, 2)

    def test_runtime_rejects_invalid_interpolation_steps(self) -> None:
        for invalid in (-1, 1.5, True):
            with self.assertRaises(InterfaceValidationError):
                infra.InferenceRuntime(
                    mode=infra.InferenceMode.SYNC,
                    interpolation_steps=invalid,  # type: ignore[arg-type]
                )

    def test_runtime_no_longer_accepts_removed_transition_bridge_configuration(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                enable_mismatch_bridge=False,  # type: ignore[call-arg]
            )
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                transition_bridge_steps=2,  # type: ignore[call-arg]
            )
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                transition_bridge_mismatch_threshold=1.5,  # type: ignore[call-arg]
            )

    def test_sync_runtime_skips_overlap_scheduler_for_single_action_source(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            ensemble_weight=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.raw_action), 1.0)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.raw_action), 2.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(robot.last_action), 2.0)  # type: ignore[arg-type]

    def test_runtime_enable_rtc_needs_no_extra_bootstrap_config(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            enable_rtc=True,
            execution_steps=1,
        )

        self.assertTrue(runtime.enable_rtc)
        self.assertEqual(runtime.execution_steps, 1)

    def test_runtime_accepts_control_hz_and_builds_internal_controller(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            control_hz=50.0,
        )

        self.assertEqual(runtime.control_hz, 50.0)
        self.assertIsNotNone(runtime.realtime_controller)
        assert runtime.realtime_controller is not None
        self.assertEqual(runtime.realtime_controller.hz, 50.0)

    def test_runtime_rejects_invalid_control_hz(self) -> None:
        for invalid in (0, -1, True, "50"):
            with self.assertRaises(InterfaceValidationError):
                infra.InferenceRuntime(
                    mode=infra.InferenceMode.SYNC,
                    control_hz=invalid,  # type: ignore[arg-type]
                )

    def test_runtime_accepts_latency_steps_offset(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            latency_steps_offset=-2,
        )

        self.assertEqual(runtime.latency_steps_offset, -2)

    def test_runtime_no_longer_accepts_startup_validation_only(self) -> None:
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                startup_validation_only=True,  # type: ignore[call-arg]
            )

    def test_runtime_accepts_validation_strategy(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="off",
        )

        self.assertEqual(runtime.validation, "off")

    def test_runtime_rejects_unknown_validation_strategy(self) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                validation="sometimes",
            )

        self.assertIn("validation", str(ctx.exception))

    def test_async_realtime_preset_builds_async_runtime(self) -> None:
        runtime = infra.InferenceRuntime.async_realtime(
            control_hz=50.0,
            execution_steps=3,
            enable_rtc=True,
        )

        self.assertEqual(runtime.mode, infra.InferenceMode.ASYNC)
        self.assertEqual(runtime.validation, "startup")
        self.assertEqual(runtime.control_hz, 50.0)
        self.assertEqual(runtime.execution_steps, 3)
        self.assertTrue(runtime.enable_rtc)
        self.assertFalse(runtime.profile)

    def test_async_realtime_signature_exposes_runtime_tuning_knobs(self) -> None:
        signature = inspect.signature(infra.InferenceRuntime.async_realtime)

        self.assertNotIn(
            inspect.Parameter.VAR_KEYWORD,
            [param.kind for param in signature.parameters.values()],
        )
        expected_parameters = [
            "profile",
            "profile_output_dir",
            "steps_before_request",
            "execution_steps",
            "warmup_requests",
            "profile_delay_requests",
            "interpolation_steps",
            "ensemble_weight",
            "control_hz",
            "enable_rtc",
            "slow_rtc_bootstrap",
            "latency_steps_offset",
            "validation",
        ]
        self.assertEqual(list(signature.parameters), expected_parameters)
        for name in expected_parameters:
            self.assertIs(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

    def test_runtime_defaults_zeroed_async_tuning_knobs(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
        )

        self.assertEqual(runtime.steps_before_request, 0)
        self.assertEqual(runtime.latency_steps_offset, 0)
        self.assertEqual(runtime.interpolation_steps, 0)
        self.assertEqual(runtime.validation, "startup")

    def test_runtime_rejects_invalid_latency_steps_offset(self) -> None:
        for invalid in (1.5, True, "2"):
            with self.assertRaises(InterfaceValidationError):
                infra.InferenceRuntime(
                    mode=infra.InferenceMode.SYNC,
                    latency_steps_offset=invalid,  # type: ignore[arg-type]
                )

    def test_runtime_no_longer_accepts_legacy_rtc_delay_offset_keyword(self) -> None:
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                rtc_inference_delay_offset_steps=1,  # type: ignore[call-arg]
            )

    def test_async_runtime_no_longer_requires_execution_steps_without_rtc(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
        )

        self.assertEqual(runtime.mode, infra.InferenceMode.ASYNC)
        self.assertIsNone(runtime.execution_steps)

    def test_async_runtime_accepts_execution_steps(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
        )

        self.assertEqual(runtime.mode, infra.InferenceMode.ASYNC)
        self.assertEqual(runtime.execution_steps, 1)

    def test_runtime_rejects_invalid_execution_steps(self) -> None:
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(InterfaceValidationError):
                infra.InferenceRuntime(
                    mode=infra.InferenceMode.ASYNC,
                    steps_before_request=0,
                    execution_steps=invalid,  # type: ignore[arg-type]
                )

    def test_runtime_accepts_slow_rtc_bootstrap_policy(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            execution_steps=2,
            enable_rtc=True,
            slow_rtc_bootstrap="error",
        )

        self.assertEqual(runtime.slow_rtc_bootstrap, "error")

    def test_runtime_rejects_invalid_slow_rtc_bootstrap_policy(self) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                execution_steps=2,
                enable_rtc=True,
                slow_rtc_bootstrap="ask-politely",
            )

        self.assertIn("slow_rtc_bootstrap", str(ctx.exception))

    def test_runtime_allows_steps_before_request_equal_to_execution_steps_without_rtc(
        self,
    ) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=2,
            execution_steps=2,
        )

        self.assertEqual(runtime.steps_before_request, 2)
        self.assertEqual(runtime.execution_steps, 2)

    def test_async_runtime_accepts_startup_request_count_config(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
            warmup_requests=2,
            profile_delay_requests=4,
        )

        self.assertEqual(runtime.warmup_requests, 2)
        self.assertEqual(runtime.profile_delay_requests, 4)

    def test_runtime_no_longer_accepts_legacy_latency_steps_keyword(self) -> None:
        with self.assertRaises(TypeError):
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                steps_before_request=0,
                execution_steps=1,
                latency_steps=4,  # type: ignore[call-arg]
            )

    def test_async_runtime_with_control_hz_warms_then_profiles_latency_before_execute(
        self,
    ) -> None:
        class CountingRobot(RuntimeRobot):
            def __init__(self) -> None:
                super().__init__()
                self.send_count = 0

            def send_action(self, action: infra.Action) -> None:
                self.send_count += 1
                super().send_action(action)

        class ConstantChunkPolicy:
            def __init__(self) -> None:
                self.request_count = 0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                self.request_count += 1
                return [
                    arm_action(7.0),
                    arm_action(8.0),
                    arm_action(9.0),
                    arm_action(10.0),
                ]

        robot = CountingRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            warmup_requests=2,
            profile_delay_requests=2,
            control_hz=50.0,
        )

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 7.0)
        self.assertEqual(robot.send_count, 1)
        self.assertGreaterEqual(policy.request_count, 4)
        self.assertIsNotNone(runtime._chunk_scheduler)
        self.assertTrue(runtime._chunk_scheduler.latency_estimate_ready())  # type: ignore[union-attr]
        self.assertGreaterEqual(runtime._chunk_scheduler.estimated_latency_steps(), 1)  # type: ignore[union-attr]

    def test_async_runtime_bootstrap_async_can_be_called_explicitly(self) -> None:
        class CountingRobot(RuntimeRobot):
            def __init__(self) -> None:
                super().__init__()
                self.send_count = 0

            def send_action(self, action: infra.Action) -> None:
                self.send_count += 1
                super().send_action(action)

        class ConstantChunkPolicy:
            def __init__(self) -> None:
                self.request_count = 0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                self.request_count += 1
                return [
                    arm_action(7.0),
                    arm_action(8.0),
                    arm_action(9.0),
                    arm_action(10.0),
                ]

        robot = CountingRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            warmup_requests=2,
            profile_delay_requests=2,
            control_hz=50.0,
        )

        bootstrapped = runtime.bootstrap_async(
            observe_fn=robot.get_obs,
            act_src_fn=policy.infer,
        )

        self.assertTrue(bootstrapped)
        self.assertEqual(robot.send_count, 0)
        self.assertEqual(policy.request_count, 4)
        self.assertIsNotNone(runtime._chunk_scheduler)
        self.assertGreater(len(scheduler_raw_actions(runtime._chunk_scheduler)), 0)

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 7.0)
        self.assertEqual(robot.send_count, 1)

    def test_async_runtime_bootstrap_async_enable_rtc_sends_prev_action_chunk_during_warmup(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            warmup_requests=1,
            profile_delay_requests=2,
            control_hz=50.0,
            enable_rtc=True,
        )

        bootstrapped = runtime.bootstrap_async(
            observe_fn=robot.get_obs,
            act_src_fn=policy.infer,
        )

        self.assertTrue(bootstrapped)
        self.assertEqual(len(policy.requests), 3)
        self.assertIsNone(policy.requests[0].rtc_args)
        self.assertIsNone(policy.requests[0].prev_action_chunk)
        assert policy.requests[1].prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in policy.requests[1].prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(policy.requests[1].execute_horizon, 3)
        assert policy.requests[2].prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in policy.requests[2].prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(policy.requests[2].execute_horizon, 3)

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)

    def test_async_runtime_bootstrap_async_enable_rtc_does_not_apply_latency_offset_during_warmup(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            warmup_requests=1,
            profile_delay_requests=2,
            control_hz=50.0,
            enable_rtc=True,
            latency_steps_offset=2,
        )

        bootstrapped = runtime.bootstrap_async(
            observe_fn=robot.get_obs,
            act_src_fn=policy.infer,
        )

        self.assertTrue(bootstrapped)
        self.assertEqual(len(policy.requests), 3)
        self.assertIsNone(policy.requests[0].rtc_args)
        self.assertEqual(policy.requests[1].latency_steps, 0)
        self.assertEqual(policy.requests[2].latency_steps, 0)
        self.assertEqual(policy.requests[1].inference_delay, 1)
        self.assertEqual(policy.requests[2].inference_delay, 1)

    def test_async_runtime_startup_validation_skips_steady_state_frame_validation(
        self,
    ) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="startup",
        )

        first = infra.run_step(
            frame=robot.get_obs(),
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        bad_frame = robot.get_obs()
        bad_frame.timestamp_ns = -1
        second = infra.run_step(
            frame=bad_frame,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        assert runtime._chunk_scheduler is not None
        self.assertTrue(runtime._chunk_scheduler._startup_validation_complete)

    def test_async_runtime_validation_off_skips_frame_validation(self) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="off",
        )
        bad_frame = robot.get_obs()
        bad_frame.state["arm"] = [0.0] * 6  # type: ignore[assignment]

        result = infra.run_step(
            frame=bad_frame,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertEqual(arm_value(result.action), 1.0)

    def test_async_runtime_validation_policy_change_resets_reused_scheduler_startup(
        self,
    ) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="off",
        )

        bad_frame = robot.get_obs()
        bad_frame.state["arm"] = [0.0] * 6  # type: ignore[assignment]
        infra.run_step(
            frame=bad_frame,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )
        assert runtime._chunk_scheduler is not None
        self.assertFalse(runtime._chunk_scheduler.runtime_validation_enabled())

        runtime.validation = "startup"
        refreshed_bad_frame = robot.get_obs()
        refreshed_bad_frame.state["arm"] = [0.0] * 6  # type: ignore[assignment]

        with self.assertRaises(InterfaceValidationError):
            infra.run_step(
                frame=refreshed_bad_frame,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
                pace_control=False,
            )

    def test_async_runtime_validation_policy_can_switch_reused_scheduler_off(
        self,
    ) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="startup",
        )

        infra.run_step(
            frame=robot.get_obs(),
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        runtime.validation = "off"
        bad_frame = robot.get_obs()
        bad_frame.state["arm"] = [0.0] * 6  # type: ignore[assignment]

        result = infra.run_step(
            frame=bad_frame,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertEqual(arm_value(result.action), 2.0)
        assert runtime._chunk_scheduler is not None
        self.assertEqual(runtime._chunk_scheduler.validation, "off")
        self.assertFalse(runtime._chunk_scheduler.runtime_validation_enabled())

    def test_async_runtime_reset_waits_for_running_pending_request(self) -> None:
        robot = RuntimeRobot()
        policy = _BlockingPendingPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )
        self.assertTrue(policy.blocking_call_started.wait(timeout=1.0))

        reset_returned = threading.Event()
        reset_thread = threading.Thread(
            target=lambda: (runtime.reset(), reset_returned.set()),
        )
        reset_thread.start()
        try:
            self.assertFalse(reset_returned.wait(timeout=0.05))
        finally:
            policy.release_blocking_call.set()
            reset_thread.join(timeout=1.0)
            runtime.close()

        self.assertFalse(reset_thread.is_alive())

    def test_async_runtime_close_waits_for_running_pending_request(self) -> None:
        robot = RuntimeRobot()
        policy = _BlockingPendingPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )
        self.assertTrue(policy.blocking_call_started.wait(timeout=1.0))

        close_returned = threading.Event()
        close_thread = threading.Thread(
            target=lambda: (runtime.close(), close_returned.set()),
        )
        close_thread.start()
        try:
            self.assertFalse(close_returned.wait(timeout=0.05))
        finally:
            policy.release_blocking_call.set()
            close_thread.join(timeout=1.0)

        self.assertFalse(close_thread.is_alive())

    def test_async_runtime_rejects_invalid_slow_rtc_bootstrap_on_scheduler_reuse(
        self,
    ) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            slow_rtc_bootstrap="warn",
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        assert runtime._chunk_scheduler is not None
        scheduler = runtime._chunk_scheduler
        self.assertEqual(scheduler.slow_rtc_bootstrap, "warn")

        runtime.slow_rtc_bootstrap = "ask-politely"

        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
                pace_control=False,
            )

        self.assertIn("slow_rtc_bootstrap", str(ctx.exception))
        self.assertIs(runtime._chunk_scheduler, scheduler)
        self.assertEqual(scheduler.slow_rtc_bootstrap, "warn")

    def test_async_runtime_reuse_syncs_scheduler_components_from_runtime_config(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=1,
            execution_steps=2,
            interpolation_steps=0,
            enable_rtc=False,
            latency_steps_offset=0,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        assert runtime._chunk_scheduler is not None
        scheduler = runtime._chunk_scheduler
        self.assertEqual(scheduler._execution_cursor.interpolation_steps, 0)
        self.assertEqual(scheduler._latency_tracker.latency_steps_offset, 0)
        self.assertFalse(scheduler._rtc_window_builder.enabled)
        self.assertEqual(scheduler._rtc_window_builder.execution_steps, 2)
        self.assertEqual(scheduler._rtc_window_builder.steps_before_request, 1)

        runtime.interpolation_steps = 3
        runtime.latency_steps_offset = 2
        runtime.enable_rtc = True
        runtime.execution_steps = 2
        runtime.steps_before_request = 0

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertIs(runtime._chunk_scheduler, scheduler)
        self.assertEqual(
            scheduler._execution_cursor.interpolation_steps,
            runtime.interpolation_steps,
        )
        self.assertEqual(
            scheduler._latency_tracker.latency_steps_offset,
            runtime.latency_steps_offset,
        )
        self.assertEqual(
            scheduler._rtc_window_builder.enabled,
            runtime.enable_rtc,
        )
        self.assertEqual(
            scheduler._rtc_window_builder.execution_steps,
            runtime.execution_steps,
        )
        self.assertEqual(
            scheduler._rtc_window_builder.steps_before_request,
            runtime.steps_before_request,
        )

    def test_async_runtime_reuse_preserves_learned_latency_tracker_state(
        self,
    ) -> None:
        class ConstantChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        robot = RuntimeRobot()
        policy = ConstantChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            control_hz=50.0,
            warmup_requests=0,
            profile_delay_requests=0,
            steps_before_request=99,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        assert runtime._chunk_scheduler is not None
        scheduler = runtime._chunk_scheduler
        scheduler._latency_tracker.estimate = 7.0
        scheduler._latency_tracker.bootstrap_complete = True
        scheduler._latency_tracker.observation_count = 5

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertIs(runtime._chunk_scheduler, scheduler)
        self.assertEqual(scheduler._latency_tracker.estimate, 7.0)
        self.assertTrue(scheduler._latency_tracker.bootstrap_complete)
        self.assertEqual(scheduler._latency_tracker.observation_count, 5)

    def test_async_runtime_reuse_applies_interpolation_change_only_after_raw_boundary(
        self,
    ) -> None:
        class InterpolatingChunkPolicy:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                return [arm_action(0.0), arm_action(3.0), arm_action(6.0)]

        robot = RuntimeRobot()
        policy = InterpolatingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            interpolation_steps=2,
            steps_before_request=99,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertEqual(arm_value(first.action), 0.0)
        assert runtime._chunk_scheduler is not None
        scheduler = runtime._chunk_scheduler
        self.assertEqual(scheduler._execution_cursor.interpolation_steps, 2)
        self.assertFalse(scheduler._execution_cursor.at_raw_boundary)

        runtime.interpolation_steps = 0

        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            pace_control=False,
        )

        self.assertIs(runtime._chunk_scheduler, scheduler)
        self.assertEqual(arm_value(second.action), 1.0)
        self.assertEqual(arm_value(third.action), 2.0)
        self.assertEqual(arm_value(fourth.action), 3.0)
        self.assertEqual(scheduler._execution_cursor.interpolation_steps, 0)

    def test_sync_runtime_enable_rtc_does_not_require_robot_spec(self) -> None:
        executor = PlainRuntimeExecutor()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)
        self.assertEqual(len(policy.requests), 3)
        self.assertIsNone(policy.requests[0].rtc_args)
        rtc_args = policy.requests[-1].rtc_args
        self.assertIsNotNone(rtc_args)

    def test_async_runtime_rejects_single_action_source(self) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
            ensemble_weight=0.5,
        )

        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )

        self.assertIn("more than one action", str(ctx.exception))

    def test_sync_runtime_does_not_filter_multi_step_chunks_without_overlap(
        self,
    ) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=2,
            ensemble_weight=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertTrue(first.plan_refreshed)
        self.assertFalse(second.plan_refreshed)
        self.assertEqual(arm_value(first.raw_action), 10.0)
        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 11.0)
        self.assertEqual(arm_value(second.action), 11.0)
        self.assertEqual(arm_value(robot.last_action), 11.0)  # type: ignore[arg-type]

    def test_async_runtime_does_not_filter_each_emitted_action(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
            ensemble_weight=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.raw_action), 10.0)
        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 11.0)
        self.assertEqual(arm_value(second.action), 11.0)
        self.assertEqual(arm_value(robot.last_action), 11.0)  # type: ignore[arg-type]

    def test_sync_runtime_memoizes_single_action_source_and_rejects_later_chunks(
        self,
    ) -> None:
        class InconsistentPolicy:
            def __init__(self) -> None:
                self.step_index = 0

            def reset(self) -> None:
                self.step_index = 0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> infra.Action | list[infra.Action]:
                del obs, request
                if self.step_index == 0:
                    self.step_index += 1
                    return infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[1.0] * 6,
                    )
                return [
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[2.0] * 6,
                    ),
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[3.0] * 6,
                    ),
                ]

        robot = RuntimeRobot()
        policy = InconsistentPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertIsNone(runtime._chunk_scheduler)

        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )

        self.assertIn("previously classified as single-step", str(ctx.exception))

    def test_sync_runtime_accepts_plain_local_executor(self) -> None:
        executor = PlainRuntimeExecutor()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(mode=infra.InferenceMode.SYNC)

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)
        self.assertEqual(arm_value(executor.last_action), 1.0)  # type: ignore[arg-type]

    def test_sync_runtime_prefers_robot_returned_action(self) -> None:
        class ReturningRobot(RuntimeRobot):
            def send_action(self, action: object) -> infra.Action:
                del action
                self.last_action = infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[9.0] * 6,
                )
                return self.last_action

        robot = ReturningRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(mode=infra.InferenceMode.SYNC)

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.raw_action), 1.0)
        self.assertEqual(arm_value(result.action), 9.0)
        self.assertEqual(arm_value(robot.last_action), 9.0)  # type: ignore[arg-type]

    def test_runtime_step_keeps_public_shape_small(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            ensemble_weight=0.5,
        )

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertFalse(hasattr(result, "timing"))
        self.assertGreaterEqual(result.control_wait_s, 0.0)

    def test_sync_runtime_can_use_overlap_with_overlap_aware_source(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)

    def test_async_runtime_can_use_internal_scheduler(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
        )

        first = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )
        second = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )
        third = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )

        self.assertTrue(first.plan_refreshed)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)

    def test_async_runtime_without_ensemble_weight_replaces_overlap_with_new_chunk(
        self,
    ) -> None:
        class FourStepPolicy:
            def __init__(self) -> None:
                self.base = 1.0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                base = self.base
                self.base += 4.0
                return [arm_action(base + float(offset)) for offset in range(4)]

        robot = RuntimeRobot()
        policy = FourStepPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=2,
            execution_steps=3,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)
        self.assertEqual(arm_value(fourth.action), 6.0)

    def test_async_runtime_enable_rtc_first_request_has_no_rtc_args_and_is_discarded(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)
        self.assertGreaterEqual(len(policy.requests), 2)
        self.assertIsNone(policy.requests[0].rtc_args)
        self.assertIsNone(policy.requests[0].prev_action_chunk)

    def test_async_runtime_enable_rtc_passes_full_chunk_context(self) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertTrue(policy.second_request_seen.wait(timeout=1.0))
        self.assertGreaterEqual(len(policy.requests), 2)

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertIsNone(policy.requests[0].rtc_args)
        self.assertIsNone(policy.requests[0].prev_action_chunk)

        first_rtc_args = policy.requests[1].rtc_args
        self.assertIsNotNone(first_rtc_args)
        assert first_rtc_args is not None
        self.assertEqual(first_rtc_args.inference_delay, 1)
        self.assertEqual(first_rtc_args.execute_horizon, 3)
        self.assertEqual(
            [arm_value(action) for action in first_rtc_args.prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(policy.requests[1].inference_delay, 1)
        self.assertEqual(policy.requests[1].execute_horizon, 3)
        assert policy.requests[1].prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in policy.requests[1].prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        self.assertIs(policy.requests[1].action_prefix, policy.requests[1].prev_action_chunk)
        self.assertEqual(policy.requests[1].prefix_length, 1)

    def test_async_runtime_enable_rtc_prev_chunk_tracks_current_active_chunk(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )

        for _ in range(8):
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )

        self.assertGreaterEqual(len(policy.requests), 5)

        self.assertIsNone(policy.requests[0].prev_action_chunk)
        assert policy.requests[1].prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in policy.requests[1].prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        tracked_request = policy.requests[-1]
        assert tracked_request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in tracked_request.prev_action_chunk],
            [5.0, 5.0, 5.0, 5.0],
        )
        self.assertEqual(tracked_request.execute_horizon, 3)
        self.assertEqual(tracked_request.inference_delay, 1)

    def test_async_runtime_interpolation_executes_smoothed_actions_but_rtc_stays_raw(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=3,
            interpolation_steps=2,
            enable_rtc=True,
        )

        emitted: list[float] = []
        for _ in range(6):
            result = infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            emitted.append(arm_value(result.action))

        self.assertTrue(policy.second_request_seen.wait(timeout=1.0))
        self.assertEqual(emitted[0], 1.0)
        self.assertAlmostEqual(emitted[1], 4.0 / 3.0)
        self.assertAlmostEqual(emitted[2], 5.0 / 3.0)
        self.assertEqual(emitted[3], 2.0)
        assert policy.requests[1].prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in policy.requests[1].prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(policy.requests[1].execute_horizon, 3)
        self.assertEqual(policy.requests[1].inference_delay, 1)

    def test_async_runtime_requires_action_source(self) -> None:
        robot = RuntimeRobot()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
        )

        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                runtime=runtime,
            )

        self.assertIn("act_src_fn", str(ctx.exception))

    def test_sync_overlap_requires_action_source(self) -> None:
        robot = RuntimeRobot()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
        )

        with self.assertRaises(InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                runtime=runtime,
            )

        self.assertIn("act_src_fn", str(ctx.exception))

    def test_sync_runtime_future_only_chunk_handoff_uses_request_step_origin(
        self,
    ) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.action), 11.0)
        self.assertEqual(arm_value(third.action), 12.0)
        self.assertEqual(arm_value(fourth.action), 13.0)

    def test_runtime_closes_old_scheduler_when_source_changes(self) -> None:
        robot = RuntimeRobot()
        first_policy = RuntimePolicy()
        second_policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=0,
            execution_steps=1,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        first_scheduler = runtime._chunk_scheduler
        self.assertIsNotNone(first_scheduler)
        self.assertIsNotNone(first_scheduler._pipeline.executor)

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=second_policy.infer,
            runtime=runtime,
        )

        self.assertIsNot(runtime._chunk_scheduler, first_scheduler)
        self.assertIsNone(first_scheduler._pipeline.executor)

    def test_runtime_restarts_scheduler_when_source_changes_with_ensemble_weight(
        self,
    ) -> None:
        robot = RuntimeRobot()
        first_policy = PlanningSource()
        second_policy = PlanningSource()
        second_policy.plan_base = 100.0
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
            ensemble_weight=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=second_policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 100.0)
        self.assertEqual(arm_value(second.action), 100.0)

    def test_async_runtime_blends_chunk_handoff_overlap_with_ensemble_weight(
        self,
    ) -> None:
        class FourStepPolicy:
            def __init__(self) -> None:
                self.base = 1.0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                base = self.base
                self.base += 4.0
                return [
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[base + offset] * 6,
                    )
                    for offset in range(4)
                ]

        robot = RuntimeRobot()
        policy = FourStepPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            steps_before_request=2,
            execution_steps=3,
            ensemble_weight=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)
        self.assertEqual(arm_value(fourth.raw_action), 5.0)
        self.assertEqual(arm_value(fourth.action), 5.0)


if __name__ == "__main__":
    unittest.main()
