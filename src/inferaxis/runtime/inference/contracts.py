"""Concrete request payloads and action-plan aliases for inference runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ...core.errors import InterfaceValidationError
from ...core.schema import Action


ActionChunk = Sequence[Action]
ActionPlan = Action | ActionChunk


@dataclass(slots=True)
class RtcArgs:
    """Optional runtime-to-policy chunk execution hints.

    ``prev_action_chunk`` is a shared read-only view over the runtime's current
    raw action window. Consumers should treat the contained actions as
    immutable.
    """

    prev_action_chunk: list[Action] = field(default_factory=list)
    inference_delay: int = 1
    execute_horizon: int = 0


@dataclass(slots=True)
class ChunkRequest:
    """Runtime context for one action request.

    ``rtc_args`` is populated when the owning runtime enables real-time chunk
    hints. In async RTC mode, it exposes a fixed-length raw
    ``prev_action_chunk`` derived from the current live buffer head, padded on
    the right to the locked source raw chunk length, together with the raw-step
    ``inference_delay`` hint and the fixed raw-step ``execute_horizon`` used
    by the runtime. For easier interoperability with RTC-style policy code,
    the same values are also mirrored onto ``prev_action_chunk``,
    ``inference_delay``, and ``execute_horizon`` directly on this object.
    For prefix-conditioning policy code, the current prefix window is also
    exposed as ``action_prefix`` with ``prefix_length``. The prefix length is
    the raw-step inference latency hint and may differ from the legacy RTC
    ``inference_delay`` mirror when the latter is clamped to the execute
    horizon for backward compatibility.
    Runtime hot paths treat any action objects exposed through this request as
    shared read-only references suitable for websocket / serialization
    consumers, not in-place mutation.
    """

    request_step: int
    request_time_s: float
    active_chunk_length: int
    remaining_steps: int
    latency_steps: int
    prev_action_chunk: list[Action] | None = None
    inference_delay: int | None = None
    execute_horizon: int | None = None
    action_prefix: list[Action] | None = None
    prefix_length: int | None = None
    rtc_args: RtcArgs | None = None

    def __post_init__(self) -> None:
        """Keep RTC mirrors synchronized when either form is provided."""

        if self.rtc_args is not None:
            if self.prev_action_chunk is None:
                self.prev_action_chunk = self.rtc_args.prev_action_chunk
            elif self.prev_action_chunk != self.rtc_args.prev_action_chunk:
                raise InterfaceValidationError(
                    "ChunkRequest.prev_action_chunk must match "
                    "ChunkRequest.rtc_args.prev_action_chunk when both are provided."
                )

            if self.inference_delay is None:
                self.inference_delay = self.rtc_args.inference_delay
            elif self.inference_delay != self.rtc_args.inference_delay:
                raise InterfaceValidationError(
                    "ChunkRequest.inference_delay must match "
                    "ChunkRequest.rtc_args.inference_delay when both are provided."
                )

            if self.execute_horizon is None:
                self.execute_horizon = self.rtc_args.execute_horizon
            elif self.execute_horizon != self.rtc_args.execute_horizon:
                raise InterfaceValidationError(
                    "ChunkRequest.execute_horizon must match "
                    "ChunkRequest.rtc_args.execute_horizon when both are provided."
                )
            self._sync_prefix_fields()
            return

        if (
            self.prev_action_chunk is not None
            or self.inference_delay is not None
            or self.execute_horizon is not None
        ):
            self.rtc_args = RtcArgs(
                prev_action_chunk=[]
                if self.prev_action_chunk is None
                else self.prev_action_chunk,
                inference_delay=1
                if self.inference_delay is None
                else self.inference_delay,
                execute_horizon=0
                if self.execute_horizon is None
                else self.execute_horizon,
            )
        self._sync_prefix_fields()

    def _sync_prefix_fields(self) -> None:
        """Fill prefix-style request fields from RTC hints when omitted."""

        if self.prev_action_chunk is not None and self.action_prefix is None:
            self.action_prefix = self.prev_action_chunk

        if self.inference_delay is not None and self.prefix_length is None:
            self.prefix_length = self.inference_delay


__all__ = [
    "ActionChunk",
    "ActionPlan",
    "ChunkRequest",
    "RtcArgs",
]
