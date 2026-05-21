"""Inference runtime engine built on inferaxis's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from os import PathLike
from pathlib import Path

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..flow import (
    StepResult,
    _execute_step_action,
    _resolve_step_frame,
)
from ...shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    callable_key,
    resolve_runtime_owner,
)
from .control import RealtimeController
from .optimizers import BlendWeight
from .scheduler import ChunkScheduler
from .engine_config import validate_runtime_config
from .engine_scheduler import (
    bootstrap_chunk_scheduler,
    ensure_chunk_scheduler,
    resolve_raw_action,
)
from .validation import ValidationMode


class InferenceMode(StrEnum):
    """Named runtime modes for :class:`InferenceRuntime`."""

    SYNC = "sync"
    ASYNC = "async"


@dataclass(slots=True)
class InferenceRuntime:
    """Composable runtime configuration for optimized :func:`run_step` calls."""

    mode: InferenceMode | str
    profile: bool = False
    profile_output_dir: str | PathLike[str] | None = None
    steps_before_request: int = 0
    execution_steps: int | None = None
    warmup_requests: int = 1
    profile_delay_requests: int = 3
    interpolation_steps: int = 0
    ensemble_weight: BlendWeight | None = None
    control_hz: float | None = None
    enable_rtc: bool = False
    slow_rtc_bootstrap: str = "warn"
    latency_steps_offset: int = 0
    validation: ValidationMode | str | None = None
    realtime_controller: RealtimeController | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler: ChunkScheduler | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler_key: object | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _single_step_source_keys: set[object] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _live_profile_recorder: object | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate mode-specific runtime configuration."""

        validate_runtime_config(self, mode_enum=InferenceMode)
        resolved_profile_output_dir: Path | None = None
        if self.profile:
            from .live_profile import (
                LiveRuntimeProfileRecorder,
                resolve_live_profile_output_dir,
            )

            resolved_profile_output_dir = resolve_live_profile_output_dir(
                self.profile_output_dir,
            )
            self._live_profile_recorder = LiveRuntimeProfileRecorder(
                output_dir=resolved_profile_output_dir,
            )
        elif self.profile_output_dir is not None:
            resolved_profile_output_dir = Path(self.profile_output_dir)

        self.profile_output_dir = resolved_profile_output_dir

    @classmethod
    def async_realtime(
        cls,
        *,
        profile: bool = False,
        profile_output_dir: str | PathLike[str] | None = None,
        steps_before_request: int = 0,
        execution_steps: int | None = None,
        warmup_requests: int = 3,
        profile_delay_requests: int = 3,
        interpolation_steps: int = 0,
        ensemble_weight: BlendWeight | None = None,
        control_hz: float | None = None,
        enable_rtc: bool = False,
        slow_rtc_bootstrap: str = "warn",
        latency_steps_offset: int = 0,
        validation: ValidationMode | str = ValidationMode.STARTUP,
    ) -> "InferenceRuntime":
        """Build an async runtime preset while keeping tuning knobs visible.

        The preset fixes ``mode=ASYNC`` and defaults validation to the startup
        pass, but leaves runtime latency, RTC, profiling, interpolation, and
        validation controls as explicit keyword-only parameters.
        """

        preset_kwargs: dict[str, object] = {
            "mode": InferenceMode.ASYNC,
            "validation": validation,
            "profile": profile,
            "profile_output_dir": profile_output_dir,
            "steps_before_request": steps_before_request,
            "execution_steps": execution_steps,
            "warmup_requests": warmup_requests,
            "profile_delay_requests": profile_delay_requests,
            "interpolation_steps": interpolation_steps,
            "ensemble_weight": ensemble_weight,
            "control_hz": control_hz,
            "enable_rtc": enable_rtc,
            "slow_rtc_bootstrap": slow_rtc_bootstrap,
            "latency_steps_offset": latency_steps_offset,
        }
        return cls(**preset_kwargs)

    def reset(self) -> None:
        """Reset source state and any attached runtime components."""

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.reset()
        self._chunk_scheduler_key = None

        if self.realtime_controller is not None:
            self.realtime_controller.reset()

    def close(self) -> None:
        """Release any background scheduler resources held by this runtime."""

        try:
            if self._chunk_scheduler is not None:
                self._chunk_scheduler.close()
                self._chunk_scheduler = None
            self._chunk_scheduler_key = None
        finally:
            self._flush_live_profile()

    def _ensure_chunk_scheduler(
        self,
        *,
        act_src_fn: ActionSource | None,
    ) -> ChunkScheduler | None:
        """Create one hidden chunk scheduler lazily when runtime mode needs it."""

        return ensure_chunk_scheduler(self, act_src_fn=act_src_fn)

    def _profile_config_snapshot(self) -> dict[str, object]:
        """Build one JSON-safe runtime configuration snapshot for profiling."""

        return {
            "mode": str(self.mode),
            "profile": self.profile,
            "profile_output_dir": (
                None
                if self.profile_output_dir is None
                else str(self.profile_output_dir)
            ),
            "steps_before_request": self.steps_before_request,
            "execution_steps": self.execution_steps,
            "warmup_requests": self.warmup_requests,
            "profile_delay_requests": self.profile_delay_requests,
            "interpolation_steps": self.interpolation_steps,
            "ensemble_weight": self.ensemble_weight,
            "control_hz": self.control_hz,
            "enable_rtc": self.enable_rtc,
            "slow_rtc_bootstrap": self.slow_rtc_bootstrap,
            "latency_steps_offset": self.latency_steps_offset,
            "validation": self.validation,
        }

    def _flush_live_profile(self) -> None:
        """Flush the live async profiling recorder when it is enabled."""

        if self._live_profile_recorder is None:
            return
        self._live_profile_recorder.flush(  # type: ignore[union-attr]
            config_snapshot=self._profile_config_snapshot(),
        )

    def _step_validation_enabled(self, *, source_key: object | None) -> bool:
        """Return whether the current runtime step should do full validation."""

        if self.validation == ValidationMode.OFF:
            return False
        if self.validation != ValidationMode.STARTUP:
            return True
        scheduler = self._chunk_scheduler
        if scheduler is None:
            return True
        if source_key is not None and self._chunk_scheduler_key != source_key:
            return True
        if scheduler.validation != self.validation:
            return True
        return scheduler.runtime_validation_enabled()

    def _bootstrap_chunk_scheduler(
        self,
        *,
        frame: Frame,
        chunk_scheduler: ChunkScheduler,
    ) -> bool:
        """Run async startup warmup/profile outside of ``next_action()``."""

        return bootstrap_chunk_scheduler(
            self,
            frame=frame,
            chunk_scheduler=chunk_scheduler,
        )

    def _resolve_raw_action(
        self,
        *,
        frame: Frame,
        act_src_fn: ActionSource | None,
        source_key: object | None,
    ) -> tuple[Action, bool, int]:
        """Resolve one raw action plus its plan metadata from the configured source."""

        return resolve_raw_action(
            self,
            frame=frame,
            act_src_fn=act_src_fn,
            source_key=source_key,
        )

    def bootstrap_async(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
    ) -> bool:
        """Explicitly run async startup warmup/profile before the first step."""

        if self.mode is not InferenceMode.ASYNC:
            raise InterfaceValidationError(
                "InferenceRuntime.bootstrap_async() requires mode=ASYNC."
            )

        metadata_owner = resolve_runtime_owner(
            observe_fn,
            act_src_fn,
        )
        source_key = callable_key(act_src_fn)
        normalized_frame = _resolve_step_frame(
            observe_fn,
            frame,
            owner=metadata_owner,
            validate=self._step_validation_enabled(source_key=source_key),
        )
        chunk_scheduler = self._ensure_chunk_scheduler(
            act_src_fn=act_src_fn,
        )
        if chunk_scheduler is None:
            raise InterfaceValidationError(
                "InferenceRuntime.bootstrap_async() requires async chunk scheduling."
            )

        bootstrapped = self._bootstrap_chunk_scheduler(
            frame=normalized_frame,
            chunk_scheduler=chunk_scheduler,
        )
        plan_length = chunk_scheduler.active_source_plan_length
        source_key = callable_key(act_src_fn)
        if plan_length == 1 and source_key is not None:
            self._single_step_source_keys.add(source_key)
            chunk_scheduler.close()
            self._chunk_scheduler = None
            self._chunk_scheduler_key = None
            raise InterfaceValidationError(
                "InferenceRuntime(mode=ASYNC) requires act_src_fn=... "
                "to return a chunk with more than one action."
            )
        return bootstrapped

    def _run_step_impl(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_fn: ActionSink | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
        execute_action: bool | None = None,
        pace_control: bool = True,
        metadata_owner: object | None = None,
    ) -> StepResult:
        """Internal implementation for runtime-managed local execution."""

        source_key = callable_key(act_src_fn)
        validate_step_values = self._step_validation_enabled(source_key=source_key)
        frame_owner = (
            metadata_owner
            if metadata_owner is not None
            else resolve_runtime_owner(observe_fn, act_src_fn)
        )
        normalized_frame = _resolve_step_frame(
            observe_fn,
            frame,
            owner=frame_owner,
            validate=validate_step_values,
        )

        raw_action, plan_refreshed, _ = self._resolve_raw_action(
            frame=normalized_frame,
            act_src_fn=act_src_fn,
            source_key=source_key,
        )
        action = raw_action

        should_execute = True if execute_action is None else execute_action
        if should_execute:
            action = _execute_step_action(
                act_fn,
                action,
                validate=validate_step_values,
            )

        control_wait_s = 0.0
        if pace_control and self.realtime_controller is not None:
            control_wait_s = self.realtime_controller.wait()

        if self._live_profile_recorder is not None:
            buffer_size = None
            execution_buffer_size = None
            if self._chunk_scheduler is not None:
                buffer_size = self._chunk_scheduler.remaining_raw_count
                execution_buffer_size = self._chunk_scheduler.remaining_execution_steps
            self._live_profile_recorder.record_action(  # type: ignore[attr-defined]
                raw_action=raw_action,
                action=action,
                plan_refreshed=plan_refreshed,
                control_wait_s=control_wait_s,
                buffer_size=buffer_size,
                execution_buffer_size=execution_buffer_size,
            )

        return StepResult(
            frame=normalized_frame,
            raw_action=raw_action,
            action=action,
            plan_refreshed=plan_refreshed,
            control_wait_s=control_wait_s,
        )

    def run_step(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_fn: ActionSink | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
        execute_action: bool | None = None,
        pace_control: bool = True,
    ) -> StepResult:
        """Run one runtime-managed step directly through this runtime instance."""

        metadata_owner = resolve_runtime_owner(
            observe_fn,
            act_fn,
            act_src_fn,
        )
        return self._run_step_impl(
            observe_fn=observe_fn,
            act_fn=act_fn,
            act_src_fn=act_src_fn,
            frame=frame,
            execute_action=True if execute_action is None else execute_action,
            pace_control=pace_control,
            metadata_owner=metadata_owner,
        )

    step = run_step


__all__ = ["InferenceMode", "InferenceRuntime"]
