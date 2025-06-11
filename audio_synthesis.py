"""
Audio Synthesis Module - Web Speech API Integration

Handles text-to-speech using browser's built-in Web Speech API for instant response.
Designed for conversational flow with zero latency.
"""

import logging
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class VoiceType(Enum):
    """Available voice types for different contexts"""
    QUESTION = "question"  # For asking questions
    FEEDBACK = "feedback"  # For providing feedback
    ENCOURAGEMENT = "encouragement"  # For positive reinforcement
    INSTRUCTION = "instruction"  # For game instructions


class WebSpeechSynthesizer:
    """
    Web Speech API integration for conversational TTS.
    Provides instant, zero-latency speech synthesis.
    """

    def __init__(self):
        """Initialize the speech synthesizer"""
        self.is_speaking = False
        self.speech_queue: List[Dict] = []
        self.on_speech_end_callbacks: List[Callable] = []

        # Voice configuration for different contexts
        self.voice_configs = {
            VoiceType.QUESTION: {
                "rate": 0.9,
                "pitch": 1.0,
                "volume": 0.8,
                "voice_preference": ["female", "pleasant"]
            },
            VoiceType.FEEDBACK: {
                "rate": 1.0,
                "pitch": 1.1,
                "volume": 0.9,
                "voice_preference": ["encouraging", "warm"]
            },
            VoiceType.ENCOURAGEMENT: {
                "rate": 1.1,
                "pitch": 1.2,
                "volume": 1.0,
                "voice_preference": ["enthusiastic", "happy"]
            },
            VoiceType.INSTRUCTION: {
                "rate": 0.8,
                "pitch": 0.9,
                "volume": 0.9,
                "voice_preference": ["clear", "authoritative"]
            }
        }

        logger.info("WebSpeechSynthesizer initialized")

    def speak_question(self, question_text: str, callback: Optional[Callable] = None) -> None:
        """
        Speak a question with appropriate voice settings

        Args:
            question_text: The question to speak
            callback: Function to call when speaking is complete
        """
        enhanced_text = self._enhance_question_text(question_text)
        self._speak_with_config(enhanced_text, VoiceType.QUESTION, callback)

    def speak_feedback(self, feedback_text: str, score: int, callback: Optional[Callable] = None) -> None:
        """
        Speak feedback with appropriate emotional tone based on score

        Args:
            feedback_text: The feedback to speak
            score: Score achieved (0-100) to determine tone
            callback: Function to call when speaking is complete
        """
        enhanced_text = self._enhance_feedback_text(feedback_text, score)
        voice_type = VoiceType.ENCOURAGEMENT if score >= 80 else VoiceType.FEEDBACK
        self._speak_with_config(enhanced_text, voice_type, callback)

    def speak_instruction(self, instruction_text: str, callback: Optional[Callable] = None) -> None:
        """
        Speak game instructions clearly

        Args:
            instruction_text: The instruction to speak
            callback: Function to call when speaking is complete
        """
        self._speak_with_config(instruction_text, VoiceType.INSTRUCTION, callback)

    def stop_speaking(self) -> None:
        """Stop current speech and clear queue"""
        self.speech_queue.clear()
        self.is_speaking = False
        # Frontend will handle actual speech cancellation

    def _enhance_question_text(self, text: str) -> str:
        """Enhance question text for better speech delivery"""
        # Add natural pauses and emphasis
        enhanced = text.replace(".", ". ").replace(",", ", ")

        # Add conversational elements
        if not text.lower().startswith(("who", "what", "when", "where", "why", "how")):
            enhanced = f"Here's your question: {enhanced}"

        return enhanced

    def _enhance_feedback_text(self, text: str, score: int) -> str:
        """Enhance feedback text based on performance"""
        if score >= 90:
            prefix = "Excellent! "
        elif score >= 70:
            prefix = "Great job! "
        elif score >= 50:
            prefix = "Good effort! "
        else:
            prefix = "Keep trying! "

        return f"{prefix}{text}"

    def _speak_with_config(self, text: str, voice_type: VoiceType, callback: Optional[Callable] = None) -> None:
        """
        Internal method to speak text with specific voice configuration

        Args:
            text: Text to speak
            voice_type: Type of voice configuration to use
            callback: Function to call when complete
        """
        config = self.voice_configs[voice_type]

        speech_data = {
            "text": text,
            "config": config,
            "callback_id": id(callback) if callback else None,
            "voice_type": voice_type.value
        }

        # Add to queue or speak immediately
        if not self.is_speaking:
            self._execute_speech(speech_data)
        else:
            self.speech_queue.append(speech_data)

        # Register callback if provided
        if callback:
            self.on_speech_end_callbacks.append(callback)

        logger.info(f"Queued speech: {voice_type.value} - {text[:50]}...")

    def _execute_speech(self, speech_data: Dict) -> None:
        """Execute speech synthesis (handled by frontend)"""
        self.is_speaking = True
        # The actual speech synthesis will be handled by the frontend JavaScript
        # This method prepares the data structure for the frontend

    def on_speech_complete(self, callback_id: Optional[int] = None) -> None:
        """Called when speech synthesis completes (from frontend)"""
        self.is_speaking = False

        # Execute callback if provided
        if callback_id and self.on_speech_end_callbacks:
            # Find and execute the callback
            for i, callback in enumerate(self.on_speech_end_callbacks):
                if id(callback) == callback_id:
                    try:
                        callback()
                        self.on_speech_end_callbacks.pop(i)
                        break
                    except Exception as e:
                        logger.error(f"Error executing speech callback: {e}")

        # Process next item in queue
        if self.speech_queue:
            next_speech = self.speech_queue.pop(0)
            self._execute_speech(next_speech)

    def get_pending_speech(self) -> Optional[Dict]:
        """Get the next speech to be synthesized (for frontend)"""
        if self.is_speaking or self.speech_queue:
            return None

        return None


class ConversationalPrompts:
    """Pre-built conversational prompts for the game"""

    GAME_START = [
        "Welcome to the Historical Figures Game! I'll describe someone famous, and you tell me who they are.",
        "Let's start your historical figures challenge! Listen carefully to each description.",
        "Ready to test your knowledge? I'll give you clues about famous people throughout history."
    ]

    TRANSITION = [
        "Great! Let's move on to the next figure.",
        "Excellent! Here's your next challenge.",
        "Perfect! Ready for the next one?",
        "Wonderful! Let's continue with another historical figure."
    ]

    ENCOURAGEMENT = [
        "You're doing great! Keep it up!",
        "Excellent progress! You really know your history!",
        "Fantastic! You're on a roll!",
        "Outstanding work! Your knowledge is impressive!"
    ]

    HINTS = [
        "Here's a hint to help you out:",
        "Let me give you a clue:",
        "This might help you:",
        "Here's something that might ring a bell:"
    ]

    COMPLETION = [
        "Congratulations! You've completed all the historical figures!",
        "Amazing work! You've finished the entire challenge!",
        "Fantastic! You've demonstrated excellent knowledge of history!",
        "Outstanding! You've successfully identified all the historical figures!"
    ]

    WAITING_FOR_NEXT = [
        "Say 'next' when you're ready to continue.",
        "Just say 'next' to move on to the next figure.",
        "When you're ready, say 'next' to continue.",
        "Say 'next' to proceed to the next challenge."
    ]


# Export the main classes
__all__ = ['WebSpeechSynthesizer', 'VoiceType', 'ConversationalPrompts']