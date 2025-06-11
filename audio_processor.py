"""
Audio Processing Module - Whisper Integration

Handles local Whisper model loading, audio transcription, and analysis.
Completely self-contained module for speech recognition.
"""

import logging
from datetime import datetime
from data_models import AUDIO_CONFIG

# Whisper imports
try:
    import whisper
    WHISPER_TYPE = "openai"
except ImportError:
    try:
        from faster_whisper import WhisperModel
        WHISPER_TYPE = "faster"
    except ImportError:
        raise ImportError("Please install either 'openai-whisper' or 'faster-whisper'")
import torch

logger = logging.getLogger(__name__)


class LocalWhisperAnalyzer:
    def __init__(self):
        """Initialize local Whisper model with caching"""
        logger.info("Loading Whisper model...")
        if WHISPER_TYPE == "openai":
            self.model = whisper.load_model(AUDIO_CONFIG["whisper"]["model_size"])
            self.model_type = "openai"
        else:
            self.model = WhisperModel(AUDIO_CONFIG["whisper"]["model_size"], device="cpu", compute_type="int8")
            self.model_type = "faster"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper loaded with {self.model_type} backend on device: {self.device}")

        self.stats = {
            "total_transcriptions": 0,
            "successful_transcriptions": 0,
            "average_confidence": 0.0,
            "processing_times": [],
            "error_count": 0
        }

    def transcribe_with_analysis(self, audio_file_path):
        """Transcribe audio using local Whisper and extract detailed information"""
        try:
            self.stats["total_transcriptions"] += 1
            start_time = datetime.now()

            whisper_config = AUDIO_CONFIG["whisper"]

            if self.model_type == "openai":
                result = self.model.transcribe(
                    audio_file_path,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    temperature=whisper_config["temperature"],
                    no_speech_threshold=whisper_config["no_speech_threshold"],
                    logprob_threshold=whisper_config["logprob_threshold"],
                    compression_ratio_threshold=whisper_config["compression_ratio_threshold"],
                    language=whisper_config["language"]
                )
                transcript = result["text"].strip()
                segments = result.get("segments", [])

                words = []
                for segment in segments:
                    segment_words = segment.get("words", [])
                    for word_info in segment_words:
                        confidence = self._estimate_word_confidence(word_info, segment)
                        words.append({
                            "word": word_info["word"].strip(),
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "confidence": confidence
                        })
            else:
                segments_generator, info = self.model.transcribe(
                    audio_file_path,
                    word_timestamps=True,
                    temperature=whisper_config["temperature"],
                    condition_on_previous_text=False,
                    language=whisper_config["language"]
                )
                segments = list(segments_generator)
                transcript = " ".join(segment.text for segment in segments).strip()

                words = []
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word_info in segment.words:
                            words.append({
                                "word": word_info.word.strip(),
                                "start": word_info.start,
                                "end": word_info.end,
                                "confidence": getattr(word_info, 'probability', 0.8)
                            })

            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["processing_times"].append(processing_time)

            overall_confidence = self._calculate_overall_confidence_v2(segments, words)

            self.stats["successful_transcriptions"] += 1
            self.stats["average_confidence"] = (
                    (self.stats["average_confidence"] * (self.stats["successful_transcriptions"] - 1) + overall_confidence) /
                    self.stats["successful_transcriptions"]
            )

            return {
                "transcript": transcript,
                "confidence": overall_confidence,
                "words": words,
                "language": getattr(info, 'language', 'en') if self.model_type == "faster" else result.get("language", "en"),
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            self.stats["error_count"] += 1
            return {
                "transcript": "",
                "confidence": 0.0,
                "words": [],
                "language": "en",
                "processing_time": 0
            }

    def _estimate_word_confidence(self, word_info, segment):
        """Estimate word confidence based on available Whisper metrics"""
        base_confidence = 0.8

        if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
            logprob_confidence = max(0.0, min(1.0, segment.avg_logprob + 1.0))
            base_confidence = (base_confidence + logprob_confidence) / 2

        if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob is not None:
            speech_confidence = 1.0 - segment.no_speech_prob
            base_confidence = (base_confidence + speech_confidence) / 2

        duration = word_info["end"] - word_info["start"]
        if 0.1 <= duration <= 2.0:
            duration_bonus = 0.1
        elif duration < 0.05:
            duration_bonus = -0.2
        else:
            duration_bonus = -0.1

        word = word_info["word"].strip().lower()
        if len(word) >= 2 and word.isalpha():
            word_bonus = 0.1
        elif word in ["a", "i", "the", "and", "or"]:
            word_bonus = 0.05
        else:
            word_bonus = -0.05

        final_confidence = base_confidence + duration_bonus + word_bonus
        return max(0.0, min(1.0, final_confidence))

    def _calculate_overall_confidence_v2(self, segments, words):
        """Calculate overall confidence for both whisper types"""
        if not words:
            return 0.0

        if self.model_type == "faster":
            confidences = [w["confidence"] for w in words]
            return sum(confidences) / len(confidences) if confidences else 0.8

        return self._calculate_overall_confidence(segments, words)

    def _calculate_overall_confidence(self, segments, words):
        """Calculate overall transcription confidence"""
        if not words:
            return 0.0

        word_confidences = [w["confidence"] for w in words]
        avg_word_confidence = sum(word_confidences) / len(word_confidences)

        segment_confidence = 0.8
        if segments:
            total_logprob = 0
            total_no_speech = 0
            valid_segments = 0
            for segment in segments:
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                    total_logprob += max(0.0, segment.avg_logprob + 1.0)
                    valid_segments += 1
                if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob is not None:
                    total_no_speech += (1.0 - segment.no_speech_prob)

            if valid_segments > 0:
                segment_confidence = (total_logprob + total_no_speech) / (valid_segments * 2)

        overall = (avg_word_confidence * 0.7) + (segment_confidence * 0.3)
        return max(0.0, min(1.0, overall))

    def get_statistics(self):
        """Get usage statistics"""
        avg_processing_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0
        return {
            **self.stats,
            "average_processing_time": avg_processing_time,
            "success_rate": (self.stats["successful_transcriptions"] / max(1, self.stats["total_transcriptions"])) * 100
        }


# Export the WHISPER_TYPE for use in main app
__all__ = ['LocalWhisperAnalyzer', 'WHISPER_TYPE']