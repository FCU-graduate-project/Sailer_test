"""GPU-free tests for decoded-audio and transcript semantic limits."""

from __future__ import annotations

import io
import unittest
import wave

from .inference import (
    CrabEmotionPredictor,
    InferenceInputError,
    TARGET_SR,
    TEXT_MAX_LEN,
)


def make_wav(duration_sec: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SR)
        handle.writeframes(b"\x00\x00" * round(duration_sec * TARGET_SR))
    return buffer.getvalue()


class FakeTokenizer:
    def __init__(self, token_count: int):
        self.token_count = token_count

    def __call__(self, text: str, **kwargs):
        return {"input_ids": list(range(self.token_count))}


class PredictorInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = CrabEmotionPredictor.__new__(CrabEmotionPredictor)
        self.predictor.wav_mean = 0.0
        self.predictor.wav_std = 1.0

    def test_exactly_12_seconds_is_accepted_without_truncation(self) -> None:
        waveform, samples, duration = self.predictor._preprocess_audio(make_wav(12.0))
        self.assertEqual(samples, 12 * TARGET_SR)
        self.assertEqual(waveform.shape[0], samples)
        self.assertAlmostEqual(duration, 12.0)

    def test_over_12_seconds_is_rejected_instead_of_silently_truncated(self) -> None:
        with self.assertRaises(InferenceInputError) as context:
            self.predictor._preprocess_audio(make_wav(12.01))
        self.assertEqual(context.exception.code, "AUDIO_TOO_LONG")
        self.assertIn("classify-long", context.exception.public_message)

    def test_decoder_details_are_not_exposed(self) -> None:
        with self.assertRaises(InferenceInputError) as context:
            self.predictor._preprocess_audio(b"not an audio container")
        self.assertEqual(context.exception.code, "INVALID_AUDIO")
        self.assertNotIn("ffmpeg", context.exception.public_message.lower())

    def test_transcript_token_boundary(self) -> None:
        self.predictor.tokenizer = FakeTokenizer(TEXT_MAX_LEN)
        self.predictor._validate_text("accepted")
        self.predictor.tokenizer = FakeTokenizer(TEXT_MAX_LEN + 1)
        with self.assertRaises(InferenceInputError) as context:
            self.predictor._validate_text("rejected")
        self.assertEqual(context.exception.code, "TEXT_TOO_LONG")

    def test_excessive_long_windows_are_rejected_before_materialization(self) -> None:
        self.predictor.tokenizer = FakeTokenizer(1)
        # If the implementation starts normalizing individual windows before
        # enforcing the count, this sentinel makes tensor subtraction fail.
        self.predictor.wav_mean = object()
        with self.assertRaises(InferenceInputError) as context:
            self.predictor.predict_long(
                make_wav(40.0),
                window_sec=12.0,
                stride_sec=0.5,
            )
        self.assertEqual(context.exception.code, "TOO_MANY_WINDOWS")
        self.assertIn("maximum is 64", context.exception.public_message)


if __name__ == "__main__":
    unittest.main()
