"""Minimal speaker recognition — HTTP client for voice-id REST API.

No PyTorch or ONNX Runtime dependencies. All recognition is delegated to
the voice-id service (voiceprint-api) via HTTP.

Usage::

    from wespeaker_deep_edge import WespeakerDeep

    client = WespeakerDeep(
        base_url="http://127.0.0.1:8005",
        api_key="your-api-key",
    )
    client.enroll("audio.wav", "voice_john.pkl")
    result = client.recognize("test.wav")
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import requests
import soundfile as sf

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  RecognitionResult
# --------------------------------------------------------------------------- #


class RecognitionResult(NamedTuple):
    """Speaker recognition result.

    Attributes:
        is_recognized: Whether the confidence meets the server's threshold.
        confidence: Cosine similarity score [0, 1].
        name: Matched speaker name (empty string if no match).
    """
    is_recognized: bool
    confidence: float
    name: str


# --------------------------------------------------------------------------- #
#  WespeakerDeep — HTTP Client
# --------------------------------------------------------------------------- #


class WespeakerDeep:
    """Voiceprint recognition client that delegates to a REST API.

    Wraps the voice-id HTTP API (``/voiceprint/register``,
    ``/voiceprint/identify``) behind the same method signatures
    as the original local engine for backward compatibility.

    Args:
        base_url: The voice-id service URL. Falls back to ``VOICE_ID_URL``
            env var, then ``http://127.0.0.1:8005``.
        api_key: API authentication token. Falls back to ``VOICE_ID_KEY``
            env var, then empty string.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("VOICE_ID_URL", "http://127.0.0.1:8005")).rstrip("/")
        self.api_key = api_key or os.getenv("VOICE_ID_KEY", "")
        self._speaker_ids: list[str] = []

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #

    def enroll(
        self,
        audio_path: str | Path,
        pk_path: str | Path = "voice.pkl",
    ) -> dict:
        """Register a speaker voiceprint via the REST API.

        The ``pk_path`` argument is retained for API compatibility but is
        **not** used to write a local file. The ``speaker_id`` is inferred
        from the ``pk_path`` filename: ``voice_john.pkl`` → ``"john"``.
        If the filename doesn't match ``voice_*.pkl``, the ``speaker_id``
        is derived from the audio filename instead.

        Args:
            audio_path: Path to the WAV audio file for enrollment.
            pk_path: Retained for backward compatibility. Speaker ID is
                inferred from the filename (``voice_X.pkl`` → ``"X"``).

        Returns:
            ``{"ok": True, "msg": "已登记: <speaker_id>"}`` on success.

        Raises:
            RuntimeError: If the API request fails.
        """
        speaker_id = self._resolve_speaker_id(audio_path, pk_path)

        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/voiceprint/register",
                headers=self._auth_headers,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"speaker_id": speaker_id},
            )

        try:
            resp.raise_for_status()
            result = resp.json()
            return {
                "ok": result.get("success", False),
                "msg": result.get("msg", ""),
            }
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"register failed for {speaker_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Recognition
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        audio_path: str | Path | object,
        voiceprint: object | None = None,
    ) -> dict:
        """Identify a speaker from an audio file.

        Args:
            audio_path: Path to a WAV file.
            voiceprint: Ignored for HTTP client (kept for API compat).
                The speaker ID list from ``load_templates()`` is used
                as candidates.

        Returns:
            ``{"is_recognized": bool, "confidence": float, "name": str}``

        Raises:
            RuntimeError: If the API request fails.
        """
        speaker_ids = self._speaker_ids or ["john"]

        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/voiceprint/identify",
                headers=self._auth_headers,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"speaker_ids": ",".join(speaker_ids)},
            )

        try:
            resp.raise_for_status()
            result = resp.json()
            speaker_id = result.get("speaker_id", "")
            score = result.get("score", 0.0)
            return {
                "is_recognized": bool(speaker_id),
                "confidence": float(score),
                "name": speaker_id,
            }
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"identify failed: {exc}"
            ) from exc

    def recognize_multi_pcm(
        self,
        pcm: object,
        sample_rate: int = 16000,
    ) -> RecognitionResult:
        """Identify a speaker from a raw PCM array.

        Writes the PCM to a temporary WAV file, calls the identify API,
        and cleans up.

        Args:
            pcm: 1-D int16 or float32 PCM array.
            sample_rate: Sample rate of the PCM data.

        Returns:
            RecognitionResult with the best match.

        Raises:
            RuntimeError: If the API request fails.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                sf.write(tmp_path, pcm, sample_rate)
                result = self.recognize(tmp_path)
            finally:
                os.unlink(tmp_path)

        return RecognitionResult(
            is_recognized=result["is_recognized"],
            confidence=result["confidence"],
            name=result["name"],
        )

    # ------------------------------------------------------------------ #
    #  Template management
    # ------------------------------------------------------------------ #

    def load_templates(
        self,
        indices: list[int] | None = None,
        files: dict[str, str | Path] | None = None,
    ) -> None:
        """Cache candidate speaker IDs for subsequent recognize calls.

        Args:
            indices: Built-in voiceprint indices (0-7).
            files: Custom voiceprint mappings, e.g. ``{"target": "path.pkl"}``.

        Raises:
            ValueError: If no templates are provided.
        """
        from ._voiceprints import get_voiceprint_name

        ids: list[str] = []

        if indices:
            for idx in indices:
                ids.append(get_voiceprint_name(idx))

        if files:
            ids.extend(files.keys())

        if not ids:
            raise ValueError("No templates provided")

        self._speaker_ids = ids
        logger.info("Loaded %d speaker IDs: %s", len(ids), ids)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _resolve_speaker_id(audio_path: str | Path, pk_path: str | Path) -> str:
        """Extract the speaker ID from the pk_path or audio filename."""
        pk_name = Path(pk_path).stem  # e.g. "voice_john" or "voice"
        if pk_name.startswith("voice_"):
            return pk_name[len("voice_"):]
        # Fallback: use audio filename stem
        return Path(audio_path).stem
