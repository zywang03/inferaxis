"""RTC-specific request payload helpers for chunk scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

from ....core.errors import InterfaceValidationError
from ....core.schema import Action
from ..contracts import RtcArgs


@dataclass(slots=True)
class RtcWindowBuilder:
    enabled: bool = False
    execution_steps: int | None = None
    steps_before_request: int = 0
    locked_chunk_total_length: int | None = None

    def reset(self) -> None:
        self.locked_chunk_total_length = None

    def lock_chunk_total_length(self, chunk_length: int) -> None:
        if not self.enabled:
            return
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC chunk length locking requires execution_steps."
            )
        if chunk_length <= 0:
            raise InterfaceValidationError(
                f"RTC chunk_total_length must be > 0, got {chunk_length!r}."
            )
        if self.locked_chunk_total_length is None:
            self.locked_chunk_total_length = chunk_length
            self.validate_execution_window_structure(chunk_length)
            return
        if chunk_length != self.locked_chunk_total_length:
            raise InterfaceValidationError(
                "RTC requires a stable source raw chunk length once the first "
                "chunk is accepted. Got "
                f"chunk_total_length={chunk_length!r}, "
                f"locked_chunk_total_length={self.locked_chunk_total_length!r}."
            )

    def validate_execution_window_structure(self, chunk_total_length: int) -> None:
        if not self.enabled or self.execution_steps is None:
            return
        if self.execution_steps >= (chunk_total_length - self.steps_before_request):
            raise InterfaceValidationError(
                "RTC requires execution_steps < chunk_total_length - "
                "steps_before_request, got "
                f"execution_steps={self.execution_steps!r}, "
                f"chunk_total_length={chunk_total_length!r}, "
                f"steps_before_request={self.steps_before_request!r}."
            )

    def build_prev_action_chunk(
        self,
        *,
        source_chunk: Sequence[Action],
    ) -> tuple[list[Action], int]:
        if not source_chunk:
            raise InterfaceValidationError(
                "RTC prev_action_chunk source must contain at least one action."
            )
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC prev_action_chunk construction requires execution_steps."
            )
        total_length = (
            self.locked_chunk_total_length
            if self.locked_chunk_total_length is not None
            else len(source_chunk)
        )
        execute_horizon = self.execution_steps
        if total_length < execute_horizon:
            raise InterfaceValidationError(
                "RTC locked chunk_total_length must be >= execution_steps, got "
                f"chunk_total_length={total_length!r}, "
                f"execution_steps={execute_horizon!r}."
            )
        window_limit = min(len(source_chunk), execute_horizon, total_length)
        window = [source_chunk[index] for index in range(window_limit)]
        total_pad_count = total_length - len(window)
        if total_pad_count > 0:
            pad_action = window[-1]
            window.extend(pad_action for _ in range(total_pad_count))
        return window, execute_horizon

    def build_action_prefix(
        self,
        *,
        source_chunk: Sequence[Action],
    ) -> list[Action]:
        if not source_chunk:
            raise InterfaceValidationError(
                "Action prefix source must contain at least one action."
            )
        total_length = (
            self.locked_chunk_total_length
            if self.locked_chunk_total_length is not None
            else len(source_chunk)
        )
        window_limit = min(len(source_chunk), total_length)
        window = [source_chunk[index] for index in range(window_limit)]
        total_pad_count = total_length - len(window)
        if total_pad_count > 0:
            pad_action = window[-1]
            window.extend(pad_action for _ in range(total_pad_count))
        return window

    def build_args(
        self,
        *,
        remaining_chunk: Sequence[Action],
        inference_delay: int,
        rtc_seed_chunk: Sequence[Action] | None = None,
    ) -> RtcArgs | None:
        if not self.enabled:
            return None
        source_chunk = remaining_chunk if remaining_chunk else rtc_seed_chunk
        if not source_chunk:
            return None
        prev_action_chunk, execute_horizon = self.build_prev_action_chunk(
            source_chunk=source_chunk,
        )
        return RtcArgs(
            prev_action_chunk=prev_action_chunk,
            inference_delay=min(max(int(inference_delay), 1), execute_horizon),
            execute_horizon=execute_horizon,
        )


def _build_rtc_args(
    self,
    *,
    remaining_chunk: Sequence[Action],
    inference_delay: int,
    rtc_seed_chunk: Sequence[Action] | None = None,
) -> RtcArgs | None:
    """Build optional RTC hints for one policy request."""

    return self._rtc_window_builder.build_args(
        remaining_chunk=remaining_chunk,
        inference_delay=inference_delay,
        rtc_seed_chunk=rtc_seed_chunk,
    )


def _build_action_prefix(
    self,
    *,
    source_chunk: Sequence[Action],
) -> list[Action]:
    """Build the prefix-conditioning action window for one policy request."""

    return self._rtc_window_builder.build_action_prefix(
        source_chunk=source_chunk,
    )


def _build_prev_action_chunk(
    self,
    *,
    source_chunk: Sequence[Action],
) -> tuple[list[Action], int]:
    """Build one RTC prev-action chunk and its execution horizon.

    The live execution window stays prefix-aligned for RTC consumers such
    as pi06star/openpi, so any fixed-shape padding is appended on the
    right by repeating the last available action. The returned list shell is
    fresh for each request, but the inner action objects are shared
    read-only references into the live buffer / seed chunk.
    """

    return self._rtc_window_builder.build_prev_action_chunk(
        source_chunk=source_chunk,
    )


def _validate_chunk_length(self, chunk_length: int) -> None:
    """Validate and lock RTC raw chunk length constraints."""

    if not self.enable_rtc:
        return
    self._lock_rtc_chunk_total_length(chunk_length)


def _lock_rtc_chunk_total_length(self, chunk_length: int) -> None:
    """Lock one fixed RTC raw chunk length and enforce later consistency."""

    self._rtc_window_builder.lock_chunk_total_length(chunk_length)


def _validate_rtc_execution_window_structure(
    self,
    chunk_total_length: int,
) -> None:
    """Validate the fixed RTC request window against the locked chunk size."""

    self._rtc_window_builder.validate_execution_window_structure(chunk_total_length)


def _check_execution_window_delay(
    self,
    *,
    raw_delay_steps: int,
) -> None:
    """Warn when predicted raw delay exceeds the RTC execution horizon."""

    if not self.enable_rtc or self.execution_steps is None:
        return
    if raw_delay_steps <= self.execution_steps:
        return

    message = (
        "Estimated raw-step inference delay exceeds execution_steps: "
        f"inference_delay={raw_delay_steps}, "
        f"execution_steps={self.execution_steps}. "
        "The transmitted RTC inference_delay will still be clamped into "
        "[1, execute_horizon]."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
