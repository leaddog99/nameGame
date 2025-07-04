"""
Conversation Manager - Handles Conversational Game Flow

Manages the state machine for conversational interaction:
SPEAKING_QUESTION → LISTENING_FOR_ANSWER → PROCESSING → SPEAKING_FEEDBACK → WAITING_FOR_NEXT
"""

import logging
import random
from enum import Enum
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from audio_synthesis import WebSpeechSynthesizer, VoiceType, ConversationalPrompts

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States in the conversational flow"""
    IDLE = "idle"
    SPEAKING_QUESTION = "speaking_question"
    LISTENING_FOR_ANSWER = "listening_for_answer"
    PROCESSING = "processing"
    SPEAKING_FEEDBACK = "speaking_feedback"
    WAITING_FOR_NEXT = "waiting_for_next"
    GAME_COMPLETE = "game_complete"


class ConversationManager:
    """
    Manages the conversational flow and state transitions for the game.
    Coordinates between TTS, speech recognition, and game logic.
    """

    def __init__(self):
        """Initialize the conversation manager"""
        self.synthesizer = WebSpeechSynthesizer()
        self.current_state = ConversationState.IDLE
        self.current_figure = None
        self.conversation_start_time = None
        self.answer_start_time = None

        # Callbacks for different events
        self.on_state_change: Optional[Callable] = None
        self.on_start_listening: Optional[Callable] = None
        self.on_stop_listening: Optional[Callable] = None
        self.on_answer_received: Optional[Callable] = None
        self.on_next_requested: Optional[Callable] = None

        # Conversation settings
        self.auto_listening = True
        self.voice_commands_enabled = True
        self.encouragement_frequency = 3  # Every N correct answers
        self.correct_streak = 0

        logger.info("ConversationManager initialized")

    def start_conversation(self, figure_data: Dict, callback: Optional[Callable] = None) -> None:
        """
        Start a conversational interaction with a new figure

        Args:
            figure_data: Historical figure data
            callback: Optional callback when question speaking completes
        """
        self.current_figure = figure_data
        self.conversation_start_time = datetime.now()
        self._transition_to_state(ConversationState.SPEAKING_QUESTION)

        # Create conversational question text
        question_text = self._create_question_text(figure_data)

        # Speak the question, then transition to listening
        def on_question_complete():
            if callback:
                callback()
            self._transition_to_listening()

        self.synthesizer.speak_question(question_text, on_question_complete)
        logger.info(f"Started conversation for figure: {figure_data.get('name', 'Unknown')}")

    def process_voice_command(self, command_text: str) -> bool:
        """
        Process voice commands like 'next', 'repeat', 'hint'

        Args:
            command_text: Recognized speech text

        Returns:
            True if command was processed, False otherwise
        """
        if not self.voice_commands_enabled:
            return False

        command = command_text.lower().strip()

        # Next command
        if any(word in command for word in ['next', 'continue', 'go on', 'proceed']):
            if self.current_state == ConversationState.WAITING_FOR_NEXT:
                self._handle_next_command()
                return True

        # Repeat command
        elif any(word in command for word in ['repeat', 'again', 'say again', 'one more time']):
            if self.current_state == ConversationState.LISTENING_FOR_ANSWER:
                self._handle_repeat_command()
                return True

        # Hint command
        elif any(word in command for word in ['hint', 'clue', 'help']):
            if self.current_state == ConversationState.LISTENING_FOR_ANSWER:
                self._handle_hint_command()
                return True

        # Skip command
        elif any(word in command for word in ['skip', 'pass', 'don\'t know']):
            if self.current_state == ConversationState.LISTENING_FOR_ANSWER:
                self._handle_skip_command()
                return True

        return False

    def provide_feedback(self, analysis_result: Dict, callback: Optional[Callable] = None) -> None:
        """
        Provide conversational feedback based on analysis results

        Args:
            analysis_result: Result from pronunciation analysis
            callback: Optional callback when feedback speaking completes
        """
        self._transition_to_state(ConversationState.SPEAKING_FEEDBACK)

        # Create conversational feedback
        feedback_text = self._create_feedback_text(analysis_result)
        overall_score = analysis_result.get('overall_score', 0)

        # Update streak tracking
        if analysis_result.get('accuracy_score', 0) >= 70:
            self.correct_streak += 1
        else:
            self.correct_streak = 0

        def on_feedback_complete():
            if callback:
                callback()
            self._transition_to_waiting()

        self.synthesizer.speak_feedback(feedback_text, overall_score, on_feedback_complete)
        logger.info(f"Provided feedback with score: {overall_score}")

    def handle_game_completion(self, final_stats: Dict, callback: Optional[Callable] = None) -> None:
        """
        Handle game completion with congratulatory message

        Args:
            final_stats: Final game statistics
            callback: Optional callback when completion message finishes
        """
        self._transition_to_state(ConversationState.GAME_COMPLETE)

        completion_text = self._create_completion_text(final_stats)

        def on_completion_complete():
            if callback:
                callback()

        self.synthesizer.speak_instruction(completion_text, on_completion_complete)
        logger.info("Game completion message delivered")

    def _create_question_text(self, figure_data: Dict) -> str:
        """Create conversational question text from figure data"""
        description = figure_data.get('description', '')
        difficulty = figure_data.get('difficulty', 1)

        # Add difficulty-appropriate introductions
        if difficulty == 1:
            intro = random.choice([
                "Here's an easy one for you:",
                "Let's start with this:",
                "This should be familiar:"
            ])
        elif difficulty == 2:
            intro = random.choice([
                "Here's a moderate challenge:",
                "Let's see if you know this one:",
                "This might require some thought:"
            ])
        else:
            intro = random.choice([
                "Here's a tough one:",
                "This is challenging:",
                "Let's test your expertise:"
            ])

        # Add natural ending
        ending = " Who am I describing?"

        return f"{intro} {description}.{ending}"

    def _create_feedback_text(self, analysis_result: Dict) -> str:
        """Create conversational feedback from analysis results"""
        accuracy_score = analysis_result.get('accuracy_score', 0)
        overall_score = analysis_result.get('overall_score', 0)
        figure_name = analysis_result.get('figure_name', 'the figure')
        transcription = analysis_result.get('raw_transcript', '')

        # Build conversational feedback
        feedback_parts = []

        # Recognition feedback
        if accuracy_score >= 90:
            feedback_parts.append(f"Perfect! You said '{transcription}' and that's exactly right.")
        elif accuracy_score >= 70:
            feedback_parts.append(f"Correct! You said '{transcription}' - that's {figure_name}.")
        elif accuracy_score >= 40:
            feedback_parts.append(f"Close! You said '{transcription}' - I was looking for {figure_name}.")
        else:
            feedback_parts.append(f"Not quite. You said '{transcription}', but the answer was {figure_name}.")

        # Additional context or encouragement
        if overall_score >= 95:
            feedback_parts.append("Your pronunciation was excellent!")
        elif overall_score >= 80:
            feedback_parts.append("Great clarity and timing!")
        elif accuracy_score >= 70 and overall_score < 80:
            feedback_parts.append("Correct answer, but try to speak a bit more clearly next time.")

        # Streak encouragement
        if self.correct_streak >= 3:
            feedback_parts.append(f"You're on a {self.correct_streak} question streak! Fantastic!")

        return " ".join(feedback_parts)

    def _create_completion_text(self, final_stats: Dict) -> str:
        """Create conversational completion message"""
        total_score = final_stats.get('total_score', 0)
        average_score = final_stats.get('average_score', 0)
        attempts = final_stats.get('attempts', 0)

        intro = random.choice(ConversationalPrompts.COMPLETION)

        performance_text = f"You completed {attempts} questions with an average score of {average_score:.1f} points."

        if average_score >= 85:
            ending = "You're a true history expert! Outstanding work!"
        elif average_score >= 70:
            ending = "Excellent knowledge of historical figures!"
        elif average_score >= 50:
            ending = "Great effort! You're well on your way to becoming a history buff!"
        else:
            ending = "Keep practicing! Every expert was once a beginner."

        return f"{intro} {performance_text} {ending}"

    def _transition_to_state(self, new_state: ConversationState) -> None:
        """Transition to a new conversation state"""
        old_state = self.current_state
        self.current_state = new_state

        logger.info(f"State transition: {old_state.value} → {new_state.value}")

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _transition_to_listening(self) -> None:
        """Transition to listening for user answer"""
        self._transition_to_state(ConversationState.LISTENING_FOR_ANSWER)
        self.answer_start_time = datetime.now()

        if self.on_start_listening:
            self.on_start_listening()

    def _transition_to_waiting(self) -> None:
        """Transition to waiting for 'next' command"""
        self._transition_to_state(ConversationState.WAITING_FOR_NEXT)

        # Speak the waiting instruction
        waiting_text = random.choice(ConversationalPrompts.WAITING_FOR_NEXT)
        self.synthesizer.speak_instruction(waiting_text)

    def _handle_next_command(self) -> None:
        """Handle 'next' voice command"""
        if self.on_next_requested:
            self.on_next_requested()

        self._transition_to_state(ConversationState.IDLE)

    def _handle_repeat_command(self) -> None:
        """Handle 'repeat' voice command"""
        if self.current_figure:
            question_text = self._create_question_text(self.current_figure)
            self.synthesizer.speak_question(question_text)


    def _handle_hint_command(self) -> None:
        """Handle 'hint' voice command"""
        if self.current_figure:
            hint_intro = random.choice(ConversationalPrompts.HINTS)
            hint_text = self.current_figure.get('hint', 'No hint available for this figure.')
            full_hint = f"{hint_intro} {hint_text}"
            self.synthesizer.speak_instruction(full_hint)

    def _handle_skip_command(self) -> None:
        """Handle 'skip' voice command"""
        skip_text = f"No problem! The answer was {self.current_figure.get('name', 'unknown')}."
        self.synthesizer.speak_feedback(skip_text, 0)

        # Trigger next after short delay
        def delayed_next():
            self._transition_to_waiting()

        # Schedule transition to waiting state
        delayed_next()

    def stop_conversation(self) -> None:
        """Stop current conversation and reset state"""
        self.synthesizer.stop_speaking()
        self._transition_to_state(ConversationState.IDLE)
        self.current_figure = None
        self.conversation_start_time = None
        self.answer_start_time = None

    def get_state(self) -> ConversationState:
        """Get current conversation state"""
        return self.current_state

    def is_listening(self) -> bool:
        """Check if currently in listening state"""
        return self.current_state == ConversationState.LISTENING_FOR_ANSWER

    def is_waiting_for_next(self) -> bool:
        """Check if waiting for 'next' command"""
        return self.current_state == ConversationState.WAITING_FOR_NEXT


# Export the main classes
__all__ = ['ConversationManager', 'ConversationState']