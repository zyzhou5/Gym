# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async capture preserves the synchronous integrity and acknowledgment contract."""

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
from nemo_gym.token_id_capture.staging import CaptureAdmission, StagedCallRecord, StageResult
from nemo_gym.token_id_capture.staging.capture import CaptureError, RolloutTokenCapture


class _Sink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.attachments: list[Mapping[str, Any] | None] = []
        self.entered = asyncio.Queue()
        self.release = asyncio.Event()
        self.error: Exception | None = None
        self.result: StageResult | None = None

    def stage(self, record: StagedCallRecord, *, attachments: Mapping[str, Any] | None = None) -> StageResult:
        if self.error is not None:
            raise self.error
        self.records.append(record)
        self.attachments.append(attachments)
        return self.result or StageResult(ok=True, staging_key=record.staging_key)

    async def async_stage(
        self, record: StagedCallRecord, *, attachments: Mapping[str, Any] | None = None
    ) -> StageResult:
        self.entered.put_nowait(record.model_call_id)
        await self.release.wait()
        return self.stage(record, attachments=attachments)


def _capture(sink: _Sink) -> RolloutTokenCapture:
    return RolloutTokenCapture(sink=sink, async_sink=sink, weight_version_fn=lambda: 7, adapter=VLLMCaptureAdapter())


def _admission(call_id: str = "c1") -> CaptureAdmission:
    return CaptureAdmission(rollout_id="rollout", model_call_id=call_id, mode="text")


def _response() -> dict:
    return {
        "prompt_token_ids": [10, 11],
        "choices": [{"message": {"generation_token_ids": [12], "generation_log_probs": [-0.25]}}],
    }


def test_async_coords_wait_for_ack_and_match_sync_record() -> None:
    async def scenario() -> None:
        sink = _Sink()
        capture = _capture(sink)
        attachment = {"opaque": object()}
        task = asyncio.create_task(
            capture.async_complete_call_from_response(
                capture.begin_call(_admission()), _response(), attachments=attachment
            )
        )
        await sink.entered.get()
        assert not task.done()
        assert sink.records == []
        sink.release.set()
        actual = await task
        baseline_sink = _Sink()
        baseline = _capture(baseline_sink)
        expected = baseline.complete_call_from_response(
            baseline.begin_call(_admission()), _response(), attachments=attachment
        )
        assert actual == expected
        assert sink.records == baseline_sink.records
        assert sink.attachments == [attachment]

    asyncio.run(scenario())


def test_async_independent_calls_overlap_but_duplicate_completion_is_rejected() -> None:
    async def scenario() -> None:
        sink = _Sink()
        capture = _capture(sink)
        first_call = capture.begin_call(_admission("a"))
        first = asyncio.create_task(capture.async_complete_call_from_response(first_call, _response()))
        second = asyncio.create_task(
            capture.async_complete_call_from_response(capture.begin_call(_admission("b")), _response())
        )
        assert {await sink.entered.get(), await sink.entered.get()} == {"a", "b"}
        with pytest.raises(CaptureError, match="already completed"):
            await capture.async_complete_call_from_response(first_call, _response())
        sink.release.set()
        assert all(coords.disposition == "staged" for coords in await asyncio.gather(first, second))
        assert len(sink.records) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["raise", "reject", "bad_result", "bad_response"])
def test_async_failures_never_acknowledge_staging(failure: str) -> None:
    async def scenario() -> None:
        sink = _Sink()
        sink.release.set()
        if failure == "raise":
            sink.error = OSError("storage unavailable")
        elif failure == "reject":
            sink.result = StageResult(ok=False, error="rejected")
        elif failure == "bad_result":
            sink.result = "not a StageResult"  # type: ignore[assignment]
        capture = _capture(sink)
        call = capture.begin_call(_admission())
        response = {} if failure == "bad_response" else _response()
        coords = await capture.async_complete_call_from_response(call, response)
        assert coords.disposition == "capture_failed"
        assert coords.staging_key is None
        assert call.completed

    asyncio.run(scenario())


def test_async_cancellation_propagates_and_completion_cannot_be_replayed() -> None:
    async def scenario() -> None:
        sink = _Sink()
        capture = _capture(sink)
        call = capture.begin_call(_admission())
        task = asyncio.create_task(capture.async_complete_call_from_response(call, _response()))
        await sink.entered.get()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sink.records == []
        with pytest.raises(CaptureError, match="already completed"):
            capture.complete_call_from_response(call, _response())

    asyncio.run(scenario())


def test_async_completion_requires_explicit_async_sink() -> None:
    capture = RolloutTokenCapture(sink=_Sink(), weight_version_fn=lambda: 7)
    call = capture.begin_call(_admission())
    with pytest.raises(CaptureError, match="AsyncStagingSink"):
        asyncio.run(
            capture.async_complete_call(call, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1])
        )
    assert not call.completed
