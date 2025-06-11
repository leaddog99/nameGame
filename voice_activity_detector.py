"""
Voice Activity Detection Module

Handles real-time speech detection for conversational flow:
- Detects when user starts speaking
- Monitors for silence periods
- Auto-stops recording after 2 seconds of silence
- Provides configuration for different sensitivity levels
"""

import logging
from typing import Dict, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VADSensitivity(Enum):
    """Voice Activity Detection sensitivity levels"""
    LOW = "low"  # Less sensitive, requires clearer speech
    MEDIUM = "medium"  # Balanced sensitivity
    HIGH = "high"  # More sensitive, picks up quieter speech
    ADAPTIVE = "adaptive"  # Adapts based on environment


class VADState(Enum):
    """Voice Activity Detection states"""
    IDLE = "idle"
    LISTENING = "listening"
    SPEECH_DETECTED = "speech_detected"
    SILENCE_DETECTED = "silence_detected"
    RECORDING_COMPLETE = "recording_complete"


class VoiceActivityDetector:
    """
    Manages voice activity detection for conversational interaction.
    Works with browser's Web Audio API for real-time audio analysis.
    """

    def __init__(self, sensitivity: VADSensitivity = VADSensitivity.MEDIUM):
        """
        Initialize Voice Activity Detector

        Args:
            sensitivity: Detection sensitivity level
        """
        self.sensitivity = sensitivity
        self.state = VADState.IDLE

        # Detection thresholds based on sensitivity
        self.thresholds = self._get_sensitivity_config(sensitivity)

        # Timing configuration
        self.silence_duration_ms = 2000  # 2 seconds of silence to stop
        self.min_speech_duration_ms = 500  # Minimum speech length
        self.max_recording_duration_ms = 30000  # 30 second max

        # State tracking
        self.speech_start_time: Optional[datetime] = None
        self.last_speech_time: Optional[datetime] = None
        self.recording_start_time: Optional[datetime] = None

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_silence_detected: Optional[Callable] = None
        self.on_recording_complete: Optional[Callable] = None

        # Audio analysis buffer
        self.audio_level_history = []
        self.max_history_length = 10

        logger.info(f"VoiceActivityDetector initialized with {sensitivity.value} sensitivity")

    def start_listening(self) -> Dict:
        """
        Start listening for voice activity

        Returns:
            Configuration dict for frontend implementation
        """
        self.state = VADState.LISTENING
        self.recording_start_time = datetime.now()
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_level_history.clear()

        config = self._get_frontend_config()

        logger.info("Started voice activity detection")
        return config

    def stop_listening(self) -> None:
        """Stop listening for voice activity"""
        self.state = VADState.IDLE
        self.speech_start_time = None
        self.last_speech_time = None
        self.recording_start_time = None

        logger.info("Stopped voice activity detection")

    def process_audio_level(self, audio_level: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        Process real-time audio level data

        Args:
            audio_level: Audio level (0.0 to 1.0)
            timestamp: Optional timestamp, uses current time if None

        Returns:
            Status dictionary with detection results
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Add to history
        self.audio_level_history.append({
            'level': audio_level,
            'timestamp': timestamp
        })

        # Maintain history size
        if len(self.audio_level_history) > self.max_history_length:
            self.audio_level_history.pop(0)

        # Analyze current state
        return self._analyze_voice_activity(audio_level, timestamp)

    def _analyze_voice_activity(self, audio_level: float, timestamp: datetime) -> Dict:
        """
        Analyze voice activity based on current audio level

        Args:
            audio_level: Current audio level
            timestamp: Current timestamp

        Returns:
            Analysis results
        """
        if self.state != VADState.LISTENING:
            return {"status": "not_listening"}

        # Check for speech detection
        is_speech = self._is_speech_detected(audio_level)

        result = {
            "timestamp": timestamp.isoformat(),
            "audio_level": audio_level,
            "is_speech": is_speech,
            "state": self.state.value
        }

        if is_speech:
            result.update(self._handle_speech_detected(timestamp))
        else:
            result.update(self._handle_silence_detected(timestamp))

        # Check for recording completion conditions
        completion_check = self._check_recording_completion(timestamp)
        if completion_check:
            result.update(completion_check)

        return result

    def _is_speech_detected(self, audio_level: float) -> bool:
        """
        Determine if current audio level indicates speech

        Args:
            audio_level: Current audio level (0.0 to 1.0)

        Returns:
            True if speech is detected
        """
        threshold = self.thresholds["speech_threshold"]

        # Simple threshold check
        if audio_level >= threshold:
            return True

        # Advanced: Check for sustained audio above noise floor
        if len(self.audio_level_history) >= 3:
            recent_levels = [h['level'] for h in self.audio_level_history[-3:]]
            avg_recent = sum(recent_levels) / len(recent_levels)

            # If recent average is above noise threshold, consider it speech
            if avg_recent >= self.thresholds["noise_threshold"]:
                return True

        return False

    def _handle_speech_detected(self, timestamp: datetime) -> Dict:
        """Handle speech detection"""
        if self.speech_start_time is None:
            self.speech_start_time = timestamp
            self.state = VADState.SPEECH_DETECTED

            if self.on_speech_start:
                self.on_speech_start()

            return {
                "action": "speech_started",
                "speech_duration_ms": 0
            }

        # Update last speech time
        self.last_speech_time = timestamp

        speech_duration = (timestamp - self.speech_start_time).total_seconds() * 1000

        return {
            "action": "speech_continuing",
            "speech_duration_ms": speech_duration
        }

    def _handle_silence_detected(self, timestamp: datetime) -> Dict:
        """Handle silence detection"""
        if self.speech_start_time is None:
            # No speech detected yet
            return {"action": "waiting_for_speech"}

        if self.last_speech_time is None:
            self.last_speech_time = timestamp

        # Calculate silence duration
        silence_duration = (timestamp - self.last_speech_time).total_seconds() * 1000

        if silence_duration >= self.silence_duration_ms:
            # Sufficient silence detected
            self.state = VADState.SILENCE_DETECTED

            if self.on_silence_detected:
                self.on_silence_detected()

            return {
                "action": "silence_detected",
                "silence_duration_ms": silence_duration,
                "should_stop_recording": True
            }

        return {
            "action": "silence_continuing",
            "silence_duration_ms": silence_duration
        }

    def _check_recording_completion(self, timestamp: datetime) -> Optional[Dict]:
        """Check if recording should be completed"""
        if self.recording_start_time is None:
            return None

        total_duration = (timestamp - self.recording_start_time).total_seconds() * 1000

        # Maximum duration reached
        if total_duration >= self.max_recording_duration_ms:
            self.state = VADState.RECORDING_COMPLETE

            if self.on_recording_complete:
                self.on_recording_complete()

            return {
                "completion_reason": "max_duration",
                "total_duration_ms": total_duration,
                "should_complete": True
            }

        # Minimum speech + sufficient silence
        if (self.speech_start_time and self.state == VADState.SILENCE_DETECTED):
            speech_duration = (self.last_speech_time - self.speech_start_time).total_seconds() * 1000

            if speech_duration >= self.min_speech_duration_ms:
                self.state = VADState.RECORDING_COMPLETE

                if self.on_recording_complete:
                    self.on_recording_complete()

                return {
                    "completion_reason": "speech_complete",
                    "speech_duration_ms": speech_duration,
                    "total_duration_ms": total_duration,
                    "should_complete": True
                }

        return None

    def _get_sensitivity_config(self, sensitivity: VADSensitivity) -> Dict:
        """Get configuration based on sensitivity level"""
        configs = {
            VADSensitivity.LOW: {
                "speech_threshold": 0.3,
                "noise_threshold": 0.15,
                "smoothing_factor": 0.7
            },
            VADSensitivity.MEDIUM: {
                "speech_threshold": 0.2,
                "noise_threshold": 0.1,
                "smoothing_factor": 0.5
            },
            VADSensitivity.HIGH: {
                "speech_threshold": 0.1,
                "noise_threshold": 0.05,
                "smoothing_factor": 0.3
            },
            VADSensitivity.ADAPTIVE: {
                "speech_threshold": 0.2,  # Will adapt based on environment
                "noise_threshold": 0.1,
                "smoothing_factor": 0.5
            }
        }

        return configs[sensitivity]

    def _get_frontend_config(self) -> Dict:
        """Get configuration for frontend implementation"""
        return {
            "sensitivity": self.sensitivity.value,
            "thresholds": self.thresholds,
            "timing": {
                "silence_duration_ms": self.silence_duration_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "max_recording_duration_ms": self.max_recording_duration_ms
            },
            "audio_analysis": {
                "fft_size": 256,
                "sample_rate": 16000,
                "analysis_interval_ms": 100
            }
        }

    def get_current_status(self) -> Dict:
        """Get current VAD status"""
        status = {
            "state": self.state.value,
            "sensitivity": self.sensitivity.value,
            "is_listening": self.state == VADState.LISTENING,
            "speech_detected": self.state == VADState.SPEECH_DETECTED
        }

        if self.speech_start_time:
            now = datetime.now()
            status["speech_duration_ms"] = (now - self.speech_start_time).total_seconds() * 1000

        if self.recording_start_time:
            now = datetime.now()
            status["total_duration_ms"] = (now - self.recording_start_time).total_seconds() * 1000

        return status

    def calibrate_environment(self, ambient_audio_levels: list) -> None:
        """
        Calibrate detection thresholds based on ambient noise

        Args:
            ambient_audio_levels: List of ambient audio levels for calibration
        """
        if not ambient_audio_levels:
            return

        avg_ambient = sum(ambient_audio_levels) / len(ambient_audio_levels)
        max_ambient = max(ambient_audio_levels)

        # Adjust thresholds based on ambient noise
        noise_margin = 0.05  # 5% above ambient
        speech_margin = 0.15  # 15% above ambient for speech

        self.thresholds["noise_threshold"] = min(0.2, max_ambient + noise_margin)
        self.thresholds["speech_threshold"] = min(0.4, avg_ambient + speech_margin)

        logger.info(f"Calibrated thresholds - Noise: {self.thresholds['noise_threshold']:.3f}, "
                    f"Speech: {self.thresholds['speech_threshold']:.3f}")


# Export the main classes
__all__ = ['VoiceActivityDetector', 'VADSensitivity', 'VADState']