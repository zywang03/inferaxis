"""Core ChunkScheduler state container and method wiring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

from ..optimizers import BlendWeight
from ..protocols import ActionSource, ActionSourceProtocol
from ..validation import ValidationMode
from . import actions, bootstrap, config, execution, latency, requests, rtc
from .buffers import ExecutionCursor, RawChunkBuffer
from .latency import LatencyTracker
from .pipeline import RequestPipeline
from .rtc import RtcWindowBuilder
from .state import _CompletedChunk


@dataclass(slots=True)
class ChunkScheduler:
    """Step-based async chunk scheduler."""

    action_source: ActionSourceProtocol | ActionSource | None = None
    steps_before_request: int = 0
    execution_steps: int | None = None
    latency_ema_beta: float = 0.5
    initial_latency_steps: float = 0.0
    fixed_latency_steps: float | None = None
    control_period_s: float | None = None
    warmup_requests: int = 3
    profile_delay_requests: int = 0
    interpolation_steps: int = 0
    max_chunk_size: int | None = None
    use_overlap_blend: bool = False
    overlap_current_weight: BlendWeight = 0.5
    enable_rtc: bool = False
    slow_rtc_bootstrap: str = "warn"
    latency_steps_offset: int = 0
    validation: str | None = None
    live_profile: object | None = None
    clock: Callable[[], float] = time.perf_counter
    _control_step: int = field(default=0, init=False, repr=False)
    _raw_buffer: RawChunkBuffer = field(init=False, repr=False)
    _execution_cursor: ExecutionCursor = field(init=False, repr=False)
    _latency_tracker: LatencyTracker = field(init=False, repr=False)
    _rtc_window_builder: RtcWindowBuilder = field(init=False, repr=False)
    _startup_execution_window_validated: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _pipeline: RequestPipeline = field(
        default_factory=RequestPipeline, init=False, repr=False
    )
    _startup_validation_complete: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        self._validate_configuration()
        self._raw_buffer = RawChunkBuffer()
        self._execution_cursor = ExecutionCursor(
            buffer=self._raw_buffer,
            interpolation_steps=self.interpolation_steps,
        )
        self._latency_tracker = LatencyTracker(
            latency_ema_beta=self.latency_ema_beta,
            initial_latency_steps=self.initial_latency_steps,
            fixed_latency_steps=self.fixed_latency_steps,
            control_period_s=self.control_period_s,
            warmup_requests=self.warmup_requests,
            profile_delay_requests=self.profile_delay_requests,
            interpolation_steps=self.interpolation_steps,
            latency_steps_offset=self.latency_steps_offset,
        )
        self._rtc_window_builder = RtcWindowBuilder(
            enabled=self.enable_rtc,
            execution_steps=self.execution_steps,
            steps_before_request=self.steps_before_request,
        )

    def reset(self) -> None:
        """Discard buffered and in-flight chunks but keep learned latency."""

        self._raw_buffer.reset()
        self._execution_cursor.reset()
        self._control_step = 0
        self._startup_execution_window_validated = False
        self._rtc_window_builder.reset()
        self._startup_validation_complete = False
        self._pipeline.discard_pending(wait=True)

    def close(self) -> None:
        """Shut down background request execution."""

        self._record_completed_pending_profile_request(wait=True)
        self.reset()
        self._pipeline.close()

    @property
    def active_source_plan_length(self) -> int:
        """Return the source chunk length most recently accepted."""

        return self._raw_buffer.active_source_plan_length

    @property
    def remaining_raw_count(self) -> int:
        """Return buffered raw actions without materializing a snapshot."""

        return self._raw_buffer.remaining_raw_count

    @property
    def remaining_execution_steps(self) -> int:
        """Return current execution segment size without materializing actions."""

        return self._execution_cursor.remaining_segment_steps

    @property
    def _global_step(self) -> int:
        return self._raw_buffer.global_step

    @_global_step.setter
    def _global_step(self, value: int) -> None:
        self._raw_buffer.global_step = value

    @property
    def _active_chunk_consumed_steps(self) -> int:
        return self._raw_buffer.active_chunk_consumed_steps

    @_active_chunk_consumed_steps.setter
    def _active_chunk_consumed_steps(self, value: int) -> None:
        self._raw_buffer.active_chunk_consumed_steps = value

    @property
    def _active_chunk_waited_raw_steps(self) -> int:
        return self._raw_buffer.active_chunk_waited_raw_steps

    @_active_chunk_waited_raw_steps.setter
    def _active_chunk_waited_raw_steps(self, value: int) -> None:
        self._raw_buffer.active_chunk_waited_raw_steps = value

    @property
    def _active_source_plan_length(self) -> int:
        return self._raw_buffer.active_source_plan_length

    @_active_source_plan_length.setter
    def _active_source_plan_length(self, value: int) -> None:
        self._raw_buffer.active_source_plan_length = value

    @property
    def _latency_steps_estimate(self) -> float:
        return self._latency_tracker.estimate

    @_latency_steps_estimate.setter
    def _latency_steps_estimate(self, value: float) -> None:
        self._latency_tracker.estimate = float(value)

    @property
    def _latency_observation_count(self) -> int:
        return self._latency_tracker.observation_count

    @_latency_observation_count.setter
    def _latency_observation_count(self, value: int) -> None:
        self._latency_tracker.observation_count = int(value)

    @property
    def _startup_latency_bootstrap_complete(self) -> bool:
        return self._latency_tracker.bootstrap_complete

    @_startup_latency_bootstrap_complete.setter
    def _startup_latency_bootstrap_complete(self, value: bool) -> None:
        self._latency_tracker.bootstrap_complete = bool(value)

    @property
    def _rtc_chunk_total_length(self) -> int | None:
        return self._rtc_window_builder.locked_chunk_total_length

    @_rtc_chunk_total_length.setter
    def _rtc_chunk_total_length(self, value: int | None) -> None:
        self._rtc_window_builder.locked_chunk_total_length = value

    def runtime_validation_enabled(self) -> bool:
        """Return whether hot-path frame/action validation should run."""

        if self.validation == ValidationMode.OFF:
            return False
        if self.validation != ValidationMode.STARTUP:
            return True
        return not self._startup_validation_complete

    _validate_configuration = config._validate_configuration
    refresh_latency_mode = config.refresh_latency_mode
    _validated_latency_steps_offset = config._validated_latency_steps_offset
    _sync_execution_cursor_config = config._sync_execution_cursor_config
    _sync_rtc_window_builder_config = config._sync_rtc_window_builder_config
    _sync_latency_tracker_config = config._sync_latency_tracker_config

    estimated_latency_steps = latency.estimated_latency_steps
    _base_estimated_latency_steps = latency._base_estimated_latency_steps
    latency_estimate_ready = latency.latency_estimate_ready
    _control_steps_for_raw_count = latency._control_steps_for_raw_count
    _control_steps_for_actions = latency._control_steps_for_actions
    _raw_segment_control_steps = latency._raw_segment_control_steps
    _remaining_control_steps = latency._remaining_control_steps
    _project_control_latency_to_raw_steps = (
        latency._project_control_latency_to_raw_steps
    )
    _estimated_request_latency_steps = latency._estimated_request_latency_steps
    _update_latency_estimate = latency._update_latency_estimate
    _observed_latency_steps_from_duration = (
        latency._observed_latency_steps_from_duration
    )

    _materialize_action = actions._materialize_action
    _materialize_command = actions._materialize_command
    _commands_share_layout = actions._commands_share_layout
    _commands_share_target_layout = actions._commands_share_target_layout
    _blend_overlap_action = actions._blend_overlap_action
    _overlap_new_weight = actions._overlap_new_weight
    _interpolate_action = actions._interpolate_action
    _build_execution_segment = actions._build_execution_segment
    _advance_raw_step = actions._advance_raw_step
    _normalize_plan = actions._normalize_plan

    _build_rtc_args = rtc._build_rtc_args
    _build_action_prefix = rtc._build_action_prefix
    _build_prev_action_chunk = rtc._build_prev_action_chunk
    _validate_chunk_length = rtc._validate_chunk_length
    _lock_rtc_chunk_total_length = rtc._lock_rtc_chunk_total_length
    _validate_rtc_execution_window_structure = (
        rtc._validate_rtc_execution_window_structure
    )
    _check_execution_window_delay = rtc._check_execution_window_delay

    _steps_before_request_satisfied = requests._steps_before_request_satisfied
    _build_request_job = requests._build_request_job
    _execute_request = requests._execute_request
    _ensure_executor = requests._ensure_executor

    _validate_startup_execution_window = bootstrap._validate_startup_execution_window
    _should_retry_with_local_rtc_seed = bootstrap._should_retry_with_local_rtc_seed
    _confirm_slow_rtc_bootstrap_request = bootstrap._confirm_slow_rtc_bootstrap_request
    _bootstrap_async_latency = bootstrap._bootstrap_async_latency
    _maybe_complete_startup_validation = bootstrap._maybe_complete_startup_validation
    bootstrap = bootstrap.bootstrap

    _integrate_completed_chunk = execution._integrate_completed_chunk
    _accept_pending_chunk = execution._accept_pending_chunk
    _accept_ready_pending_chunk = execution._accept_ready_pending_chunk
    _accept_blocking_pending_chunk = execution._accept_blocking_pending_chunk
    _record_completed_pending_profile_request = (
        execution._record_completed_pending_profile_request
    )
    _request_until_execution_buffer_ready = (
        execution._request_until_execution_buffer_ready
    )
    _ensure_executable_actions = execution._ensure_executable_actions
    _maybe_launch_next_request = execution._maybe_launch_next_request
    _pop_next_action = execution._pop_next_action
    next_action = execution.next_action


__all__ = ["ChunkScheduler"]
