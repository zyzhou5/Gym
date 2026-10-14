# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-owned, engine-neutral capture with stage-before-response ordering."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    build_staging_delta,
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.protocols import (
    AsyncStagingSink,
    CaptureAdapter,
    StagingSink,
    WeightVersionProvider,
)
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    CommitCoords,
    StagedCallRecord,
    StageResult,
)


LOGGER = logging.getLogger(__name__)


class CaptureError(Exception):
    """A caller violated the worker capture lifecycle."""


class StreamingUnsupportedError(CaptureError):
    """Worker-owned staging currently accepts only complete responses."""


@dataclass
class ActiveCall:
    """One admitted call, the policy version present at admission, and its resolved prefix.

    ``prefix_token_ids`` is the exact token prefix the engine prompt must begin
    with. ``begin_call`` resolves it from the admission's inline prefix or from
    the caller-supplied ids fetched for a ``staging_chain``. Empty for a text root.
    """

    admission: CaptureAdmission
    weight_version: int
    prefix_token_ids: list[int] = field(default_factory=list)
    completed: bool = field(default=False, init=False)

    @property
    def rollout_id(self) -> str:
        return self.admission.rollout_id

    @property
    def model_call_id(self) -> str:
        return self.admission.model_call_id


class RolloutTokenCapture:
    """Build, stage, and acknowledge exact per-call token deltas."""

    def __init__(
        self,
        *,
        sink: StagingSink,
        weight_version_fn: WeightVersionProvider,
        adapter: CaptureAdapter | None = None,
        async_sink: AsyncStagingSink | None = None,
    ) -> None:
        self._sink = sink
        self._async_sink = async_sink
        self._weight_version_fn = weight_version_fn
        self._adapter = adapter
        # Guards only the per-call single-completion transition. Sink writes
        # run unlocked and may overlap across calls (StagingSink contract).
        self._completion_lock = threading.Lock()

    @property
    def adapter(self) -> CaptureAdapter | None:
        return self._adapter

    def begin_call(
        self,
        admission: CaptureAdmission,
        *,
        prefix_token_ids: list[int] | None = None,
        stream: bool = False,
        weight_version: int | None = None,
    ) -> ActiveCall:
        """Admit a typed gate contract and stamp its generation weight version.

        ``prefix_token_ids`` is the resolved parent prefix for a ``token_in`` call.
        It is required when the admission stores its prefix as a ``staging_chain``:
        the caller fetches and concatenates those staged deltas and passes the ids here,
        the same list it hands to ``CaptureAdapter.enter_prefix``.
        When the admission carries the prefix inline the argument may be omitted;
        if given it must match. A text root accepts no prefix.
        Violations are caller bugs and raise ``CaptureError``; they never poison the call.

        ``weight_version`` is supplied when engine-finished metadata owns the
        authoritative per-call epoch. Other worker capture paths leave it unset
        and read the serving worker's current version provider instead.
        """
        if not isinstance(admission, CaptureAdmission):
            raise TypeError("admission must be a CaptureAdmission")
        if stream:
            raise StreamingUnsupportedError(
                f"rollout {admission.rollout_id} call {admission.model_call_id}: "
                "token capture does not support streaming responses"
            )
        resolved_prefix = self._resolve_prefix(admission, prefix_token_ids)
        if weight_version is None:
            weight_version = self._weight_version_fn()
        if type(weight_version) is not int or weight_version < 0:
            raise CaptureError(f"weight_version must be a non-negative int, got {weight_version!r}")
        return ActiveCall(admission=admission, weight_version=weight_version, prefix_token_ids=resolved_prefix)

    @staticmethod
    def _resolve_prefix(admission: CaptureAdmission, prefix_token_ids: list[int] | None) -> list[int]:
        where = f"rollout {admission.rollout_id} call {admission.model_call_id}"
        if prefix_token_ids is not None:
            if not isinstance(prefix_token_ids, list) or any(
                type(token_id) is not int or token_id < 0 for token_id in prefix_token_ids
            ):
                raise CaptureError(f"{where}: prefix_token_ids must be a list of non-negative ints")
        if admission.mode == "text":
            if prefix_token_ids:
                raise CaptureError(f"{where}: a text root admission accepts no prefix_token_ids")
            return []
        inline = list(admission.required_prefix_token_ids)
        if prefix_token_ids is None:
            if not inline:
                raise CaptureError(
                    f"{where}: staging_chain admission requires the caller to pass the resolved prefix_token_ids"
                )
            return inline
        if len(prefix_token_ids) != admission.prev_len:
            raise CaptureError(
                f"{where}: prefix_token_ids length {len(prefix_token_ids)} does not equal prev_len {admission.prev_len}"
            )
        if inline and inline != list(prefix_token_ids):
            raise CaptureError(f"{where}: prefix_token_ids conflict with the admission's inline prefix")
        return list(prefix_token_ids)

    def complete_call(
        self,
        call: ActiveCall,
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        extras: dict[str, Any] | None = None,
        attachments: Mapping[str, Any] | None = None,
    ) -> CommitCoords:
        """Stage tokens and attachments before returning acknowledged coordinates."""
        record = self._prepare_record(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
        )
        if record is None:
            return self._failed_coords(call)
        try:
            # The completion claim is per call; independent writes may overlap.
            # Omit the keyword for legacy text-only sinks.
            if attachments is None:
                result = self._sink.stage(record)
            else:
                result = self._sink.stage(record, attachments=attachments)
            return self._stage_coords(call, record, result)
        except Exception:
            # Sink exceptions poison capture, while cancellation propagates.
            LOGGER.exception(
                "token staging failed for rollout %s call %s",
                call.rollout_id,
                call.model_call_id,
            )
            return self._failed_coords(call)

    async def async_complete_call(
        self,
        call: ActiveCall,
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        extras: dict[str, Any] | None = None,
        attachments: Mapping[str, Any] | None = None,
    ) -> CommitCoords:
        """Stage tokens and attachments before returning acknowledged coordinates."""
        if self._async_sink is None:
            raise CaptureError("async_complete_call requires an AsyncStagingSink")
        record = self._prepare_record(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
        )
        if record is None:
            return self._failed_coords(call)
        try:
            # The completion claim is per call; independent writes may overlap.
            # Omit the keyword for legacy text-only sinks.
            if attachments is None:
                result = await self._async_sink.async_stage(record)
            else:
                result = await self._async_sink.async_stage(record, attachments=attachments)
            return self._stage_coords(call, record, result)
        except Exception:
            # Sink exceptions poison capture, while cancellation propagates.
            LOGGER.exception(
                "token staging failed for rollout %s call %s",
                call.rollout_id,
                call.model_call_id,
            )
            return self._failed_coords(call)

    def _prepare_record(
        self,
        call: ActiveCall,
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        extras: dict[str, Any] | None,
    ) -> StagedCallRecord | None:
        """Claim completion once and construct the shared, digest-covered record."""
        self._claim_completion(call)
        admission = call.admission
        try:
            if admission.mode == "token_in" and prompt_token_ids[: admission.prev_len] != call.prefix_token_ids:
                raise ValueError("generation prompt does not begin with the gate-authorized token prefix")
            token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_log_probs=generated_logprobs,
                prev_len=admission.prev_len,
            )
            delta_len = len(token_ids_delta)
            cum_len = admission.prev_len + delta_len
            # For a continuation, the comparison above verifies that the prompt starts with the required prefix.
            # Appending the generated IDs therefore yields the complete sequence for both hashes.
            chain_hash = compute_chain_hash(admission.parent_chain_hash, token_ids_delta)
            cumulative_hash = hash_token_ids(list(prompt_token_ids) + list(generated_token_ids))
            extras_digest = compute_extras_digest(extras)
            digest = compute_staging_digest(
                schema_version=admission.schema_version,
                digest_version=STAGING_DIGEST_VERSION,
                extras_digest_version=EXTRAS_DIGEST_VERSION,
                rollout_id=admission.rollout_id,
                model_call_id=admission.model_call_id,
                parent_call_id=admission.parent_call_id,
                mode=admission.mode,
                prev_len=admission.prev_len,
                delta_len=delta_len,
                cum_len=cum_len,
                weight_version=call.weight_version,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                generation_log_probs_delta=logprobs_delta,
                extras_digest=extras_digest,
                chain_hash=chain_hash,
                cumulative_hash=cumulative_hash,
            )
            record = StagedCallRecord(
                rollout_id=admission.rollout_id,
                model_call_id=admission.model_call_id,
                parent_call_id=admission.parent_call_id,
                mode=admission.mode,
                prev_len=admission.prev_len,
                delta_len=delta_len,
                cum_len=cum_len,
                weight_version=call.weight_version,
                digest=digest,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                generation_log_probs_delta=logprobs_delta,
                extras=extras,
                extras_digest=extras_digest,
                chain_hash=chain_hash,
                cumulative_hash=cumulative_hash,
            )
        except (TypeError, ValueError, OverflowError):
            LOGGER.exception(
                "token capture could not build rollout %s call %s",
                admission.rollout_id,
                admission.model_call_id,
            )
            return None
        return record

    def _stage_coords(self, call: ActiveCall, record: StagedCallRecord, result: StageResult) -> CommitCoords:
        """Publish coordinates only after a valid successful sink acknowledgment."""
        admission = call.admission
        if not isinstance(result, StageResult):
            raise TypeError(f"staging sink returned {type(result).__name__}, expected StageResult")
        if not result.ok:
            LOGGER.warning(
                "token staging sink rejected rollout %s call %s: %s",
                admission.rollout_id,
                admission.model_call_id,
                result.error,
            )
            return self._failed_coords(call)
        return CommitCoords(
            rollout_id=admission.rollout_id,
            model_call_id=admission.model_call_id,
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=record.delta_len,
            cum_len=record.cum_len,
            weight_version=call.weight_version,
            disposition="staged",
            digest=record.digest,
            extras_digest=record.extras_digest,
            staging_key=result.staging_key,
            chain_hash=record.chain_hash,
            cumulative_hash=record.cumulative_hash,
        )

    def _extract_response_material(
        self, call: ActiveCall, response_payload: Any
    ) -> tuple[list[int], list[int], list[float], dict[str, Any] | None] | None:
        """Extract engine material; extraction failure claims and poisons the call."""
        if self._adapter is None:
            raise CaptureError("complete_call_from_response requires a CaptureAdapter")
        try:
            prompt_token_ids = self._adapter.extract_prompt_ids(response_payload)
            generated_token_ids, generated_logprobs = self._adapter.extract_generation(response_payload)
            extras = self._adapter.extract_extras(response_payload)
        except Exception:
            # Adapters are engine/framework boundaries and may expose native
            # exception types. Extraction failure poisons capture only.
            LOGGER.exception(
                "token extraction failed for rollout %s call %s",
                call.rollout_id,
                call.model_call_id,
            )
            self._claim_completion(call)
            return None
        return prompt_token_ids, generated_token_ids, generated_logprobs, extras

    def complete_call_from_response(
        self,
        call: ActiveCall,
        response_payload: Any,
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> CommitCoords:
        """Extract engine-native material and stage it before acknowledgment."""
        material = self._extract_response_material(call, response_payload)
        if material is None:
            return self._failed_coords(call)
        prompt_token_ids, generated_token_ids, generated_logprobs, extras = material
        return self.complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
            attachments=attachments,
        )

    async def async_complete_call_from_response(
        self,
        call: ActiveCall,
        response_payload: Any,
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> CommitCoords:
        """Extract engine-native material and stage it before acknowledgment."""
        material = self._extract_response_material(call, response_payload)
        if material is None:
            return self._failed_coords(call)
        prompt_token_ids, generated_token_ids, generated_logprobs, extras = material
        return await self.async_complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
            attachments=attachments,
        )

    def fail_call(self, call: ActiveCall, *, reason: str) -> CommitCoords:
        """Poison one admitted call that failed before durable staging."""
        self._claim_completion(call)
        LOGGER.warning(
            "token capture failed for rollout %s call %s: %s",
            call.rollout_id,
            call.model_call_id,
            reason,
        )
        return self._failed_coords(call)

    def _claim_completion(self, call: ActiveCall) -> None:
        with self._completion_lock:
            if call.completed:
                raise CaptureError(f"rollout {call.rollout_id} call {call.model_call_id} was already completed")
            call.completed = True

    @staticmethod
    def _failed_coords(call: ActiveCall) -> CommitCoords:
        admission = call.admission
        return CommitCoords(
            rollout_id=admission.rollout_id,
            model_call_id=admission.model_call_id,
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=0,
            cum_len=admission.prev_len,
            weight_version=call.weight_version,
            disposition="capture_failed",
        )


class CaptureHost:
    """Minimal serving-layer seam used by ``install_capture``."""

    def __init__(self) -> None:
        self.token_capture: RolloutTokenCapture | None = None

    def install_token_capture(self, capture: RolloutTokenCapture) -> None:
        self.token_capture = capture


def install_capture(
    serving_layer: Any,
    *,
    sink: StagingSink,
    weight_version_fn: WeightVersionProvider,
    adapter: CaptureAdapter | None = None,
    async_sink: AsyncStagingSink | None = None,
) -> RolloutTokenCapture:
    """Wire worker-owned staging into one inference serving layer."""
    if not isinstance(sink, StagingSink):
        raise TypeError(f"sink does not implement StagingSink: {type(sink)!r}")
    if async_sink is not None and not isinstance(async_sink, AsyncStagingSink):
        raise TypeError(f"async_sink does not implement AsyncStagingSink: {type(async_sink)!r}")
    if not callable(weight_version_fn):
        raise TypeError(f"weight_version_fn must be callable, got {type(weight_version_fn)!r}")
    install = getattr(serving_layer, "install_token_capture", None)
    if not callable(install):
        raise TypeError(f"serving layer {type(serving_layer)!r} does not expose install_token_capture(capture)")
    capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=weight_version_fn,
        adapter=adapter,
        async_sink=async_sink,
    )
    install(capture)
    return capture
