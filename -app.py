from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import tempfile
import json
from datetime import datetime
from difflib import SequenceMatcher
import logging
from pydub import AudioSegment
import numpy as np
import random
import jellyfish
import uuid

from data_models import (
    HISTORICAL_FIGURES,
    GAME_MODES,
    SCORING_CONFIG,
    AUDIO_CONFIG,
    TIMING_CONFIG,
    MATCHING_CONFIG,
    ACHIEVEMENTS,
    MESSAGES,
    get_game_mode_config,
    validate_figure_data
)

from audio_processor import LocalWhisperAnalyzer, WHISPER_TYPE
from pronunciation_analyzer import PronunciationAnalyzer
from conversation_manager import ConversationManager, ConversationState
from voice_activity_detector import VoiceActivityDetector, VADSensitivity
from audio_synthesis import WebSpeechSynthesizer, VoiceType, ConversationalPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this-in-production'

user_sessions = {}
user_conversations = {}  # Track conversation managers per user

validate_figure_data()
logger.info(f"✅ Data validation passed! Loaded {len(HISTORICAL_FIGURES)} figures")


class GameSession:
    def __init__(self, mode="classic"):
        self.mode = mode
        self.current_figure_index = 0
        self.score = 0
        self.attempts = []
        self.start_time = datetime.now()
        self.hints_used = 0
        self.streak = 0
        self.setup_figures_for_mode()

    def setup_figures_for_mode(self):
        mode_config = get_game_mode_config(self.mode)

        available_figures = HISTORICAL_FIGURES
        if mode_config.get("difficulty_filter"):
            available_figures = [f for f in HISTORICAL_FIGURES if f["difficulty"] == mode_config["difficulty_filter"]]

        figure_count = mode_config["figure_count"]
        if figure_count == -1:
            figure_count = len(available_figures)

        if figure_count >= len(available_figures):
            self.figures_order = list(range(len(available_figures)))
            self.selected_figures = available_figures
        else:
            selected_indices = random.sample(range(len(available_figures)), figure_count)
            self.figures_order = selected_indices
            self.selected_figures = [available_figures[i] for i in selected_indices]

        random.shuffle(self.figures_order)

    def get_current_figure(self):
        if self.current_figure_index < len(self.figures_order):
            if hasattr(self, 'selected_figures'):
                return self.selected_figures[self.current_figure_index]
            else:
                return HISTORICAL_FIGURES[self.figures_order[self.current_figure_index]]
        return None

    def advance_figure(self):
        self.current_figure_index += 1

    def is_complete(self):
        return self.current_figure_index >= len(self.figures_order)

    def get_mode_config(self):
        return get_game_mode_config(self.mode)


def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"Created new user session: {session['user_id']}")
    return session['user_id']


def get_user_game_session(mode="classic"):
    user_id = get_user_id()
    if user_id not in user_sessions:
        user_sessions[user_id] = GameSession(mode)
        logger.info(f"Created new game session for user: {user_id}")
    return user_sessions[user_id]


def reset_user_game_session(mode="classic"):
    user_id = get_user_id()

    logger.info(f"🔧 RESET DEBUG: === STARTING RESET FOR USER {user_id} ===")
    logger.info(f"🔧 RESET DEBUG: Requested mode: {mode}")
    logger.info(f"🔧 RESET DEBUG: Old session exists: {user_id in user_sessions}")

    if user_id in user_sessions:
        old_session = user_sessions[user_id]
        logger.info(f"🔧 RESET DEBUG: Old session complete: {old_session.is_complete()}")
        logger.info(f"🔧 RESET DEBUG: Old session figure index: {old_session.current_figure_index}")
        del user_sessions[user_id]
        logger.info(f"🔧 RESET DEBUG: Old session deleted")

    # Reset conversation manager as well
    if user_id in user_conversations:
        user_conversations[user_id].stop_conversation()
        del user_conversations[user_id]
        logger.info(f"🔧 RESET DEBUG: Conversation manager reset")

    user_sessions[user_id] = GameSession(mode)
    new_session = user_sessions[user_id]

    logger.info(f"🔧 RESET DEBUG: New session created")
    logger.info(f"🔧 RESET DEBUG: New session complete: {new_session.is_complete()}")
    logger.info(f"🔧 RESET DEBUG: New session figure index: {new_session.current_figure_index}")
    logger.info(f"🔧 RESET DEBUG: === RESET COMPLETE FOR USER {user_id} ===")

    return new_session


def get_user_conversation_manager():
    """Get or create conversation manager for current user"""
    user_id = get_user_id()
    if user_id not in user_conversations:
        conversation_manager = ConversationManager()

        # Set up callbacks for conversation events
        def on_next_requested():
            logger.info(f"🎙️ CONVERSATION: User {user_id} requested next figure")

        def on_state_change(old_state, new_state):
            logger.info(f"🎙️ CONVERSATION: State change {old_state.value} → {new_state.value}")

        conversation_manager.on_next_requested = on_next_requested
        conversation_manager.on_state_change = on_state_change

        user_conversations[user_id] = conversation_manager
        logger.info(f"Created conversation manager for user: {user_id}")

    return user_conversations[user_id]


def reset_user_conversation():
    """Reset conversation manager for current user"""
    user_id = get_user_id()
    if user_id in user_conversations:
        user_conversations[user_id].stop_conversation()
        del user_conversations[user_id]
        logger.info(f"Reset conversation manager for user: {user_id}")


pronunciation_analyzer = PronunciationAnalyzer()


# ===== ORIGINAL MANUAL MODE ROUTES =====

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/conversational')
def conversational_interface():
    """Serve the conversational game interface"""
    return render_template('conversational.html')


@app.route('/api/game-state')
def get_game_state():
    user_id = get_user_id()
    logger.info(f"🔧 GAME STATE DEBUG: Getting state for user {user_id}")

    game_session = get_user_game_session()
    figure = game_session.get_current_figure()

    logger.info(f"🔧 GAME STATE DEBUG: Session complete: {game_session.is_complete()}")
    logger.info(f"🔧 GAME STATE DEBUG: Current figure index: {game_session.current_figure_index}")
    logger.info(f"🔧 GAME STATE DEBUG: Figure exists: {figure is not None}")

    return jsonify({
        "current_figure": {
            "description": figure["description"],
            "hint": figure["hint"],
            "difficulty": figure["difficulty"],
            "category": figure.get("category"),
            "nationality": figure.get("nationality"),
            "birth_year": figure.get("birth_year"),
            "pronunciation_tips": figure.get("pronunciation_tips"),
            "name": figure["name"]  # Add name for conversational mode
        } if figure else None,
        "progress": {
            "current": game_session.current_figure_index + 1,
            "total": len(game_session.figures_order)
        },
        "score": game_session.score,
        "streak": game_session.streak,
        "is_complete": game_session.is_complete(),
        "mode": game_session.mode,
        "hints_used": game_session.hints_used
    })


@app.route('/api/submit-audio', methods=['POST'])
def submit_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    game_session = get_user_game_session()
    figure = game_session.get_current_figure()
    if not figure:
        return jsonify({"error": "Game completed"}), 400

    time_to_record = request.form.get('time_to_record', type=float)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name

    try:
        wav_path = None
        try:
            audio = AudioSegment.from_file(temp_file_path)
            if len(audio) < AUDIO_CONFIG["recording"]["min_duration_ms"]:
                return jsonify({"error": MESSAGES["errors"]["audio_too_short"]}), 400

            wav_path = temp_file_path.replace('.webm', '.wav')
            audio.export(wav_path, format="wav")
            analysis_file_path = wav_path
        except Exception as e:
            logger.info(f"Audio conversion skipped, using original file: {e}")
            analysis_file_path = temp_file_path

        analysis = pronunciation_analyzer.analyze_pronunciation(analysis_file_path, figure)

        if time_to_record is not None:
            analysis["time_to_record"] = time_to_record

        attempt = {
            "figure": figure["name"],
            "transcription": analysis["raw_transcript"],
            "score": analysis["overall_score"],
            "feedback": analysis["feedback"],
            "accuracy": "exact" if analysis["accuracy_score"] >= 90 else
            "close" if analysis["accuracy_score"] >= 70 else
            "partial" if analysis["accuracy_score"] >= 40 else "incorrect",
            "highlights": analysis["highlights"],
            "timestamp": datetime.now().isoformat(),
            "analysis_details": analysis,
            "time_to_record": time_to_record
        }

        game_session.attempts.append(attempt)
        game_session.score += analysis["overall_score"]

        if analysis["accuracy_score"] > 0:
            game_session.streak += 1
        else:
            game_session.streak = 0

        is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)

        logger.info(f"🔧 FINAL CARD DEBUG: Current index: {game_session.current_figure_index}")
        logger.info(f"🔧 FINAL CARD DEBUG: Total figures: {len(game_session.figures_order)}")
        logger.info(f"🔧 FINAL CARD DEBUG: Is final figure: {is_final_figure}")

        response_data = {
            "transcription": analysis["raw_transcript"],
            "analysis": analysis,
            "figure_name": figure["name"],
            "total_score": game_session.score,
            "streak": game_session.streak,
            "attempt_number": len(game_session.attempts),
            "is_final_figure": is_final_figure
        }

        game_session.advance_figure()

        if game_session.is_complete():
            game_duration = (datetime.now() - game_session.start_time).total_seconds()
            avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0

            response_data["final_stats"] = {
                "total_score": round(game_session.score, 1),
                "average_score": round(avg_score, 1),
                "duration_seconds": round(game_duration),
                "attempts": len(game_session.attempts),
                "max_streak": game_session.streak,
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
                }
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500

    finally:
        cleanup_files = [temp_file_path]
        if 'wav_path' in locals() and wav_path and wav_path != temp_file_path:
            cleanup_files.append(wav_path)

        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass


@app.route('/api/complete-game', methods=['POST'])
def complete_game():
    game_session = get_user_game_session()

    if not game_session.is_complete():
        return jsonify({"error": "Game not complete"}), 400

    game_duration = (datetime.now() - game_session.start_time).total_seconds()
    avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0

    final_stats = {
        "total_score": round(game_session.score, 1),
        "average_score": round(avg_score, 1),
        "duration_seconds": round(game_duration),
        "attempts": len(game_session.attempts),
        "max_streak": game_session.streak,
        "accuracy_breakdown": {
            "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
            "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
            "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
            "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
        }
    }

    return jsonify({
        "game_complete": True,
        "final_stats": final_stats
    })


@app.route('/api/timeout', methods=['POST'])
def handle_timeout():
    game_session = get_user_game_session()
    figure = game_session.get_current_figure()
    if not figure:
        return jsonify({"error": "Game completed"}), 400

    attempt = {
        "figure": figure["name"],
        "transcription": "⏰ Time expired",
        "score": 0,
        "feedback": MESSAGES["feedback"]["timeout"],
        "accuracy": "timeout",
        "highlights": ["Practice responding faster! ⚡"],
        "timestamp": datetime.now().isoformat(),
        "analysis_details": {
            "overall_score": 0,
            "accuracy_score": 0,
            "clarity_score": 0,
            "speed_score": 0,
            "raw_transcript": "⏰ Time expired",
            "feedback": MESSAGES["feedback"]["timeout"],
            "highlights": ["Practice responding faster! ⚡"]
        },
        "time_to_record": TIMING_CONFIG["card_timeout_seconds"]
    }

    game_session.attempts.append(attempt)
    game_session.streak = 0

    is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)

    response_data = {
        "transcription": "⏰ Time expired",
        "analysis": attempt["analysis_details"],
        "figure_name": figure["name"],
        "total_score": game_session.score,
        "streak": game_session.streak,
        "attempt_number": len(game_session.attempts),
        "timeout": True,
        "is_final_figure": is_final_figure
    }

    game_session.advance_figure()

    if game_session.is_complete():
        game_duration = (datetime.now() - game_session.start_time).total_seconds()
        avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0

        response_data["final_stats"] = {
            "total_score": round(game_session.score, 1),
            "average_score": round(avg_score, 1),
            "duration_seconds": round(game_duration),
            "attempts": len(game_session.attempts),
            "max_streak": game_session.streak,
            "accuracy_breakdown": {
                "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"]),
                "timeout": len([a for a in game_session.attempts if a["accuracy"] == "timeout"])
            }
        }

    return jsonify(response_data)


@app.route('/api/restart-game', methods=['POST'])
def restart_game():
    try:
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = {}

        mode = data.get('mode', 'classic')

        user_id = get_user_id()
        logger.info(f"🔧 RESTART DEBUG: User {user_id} requesting restart with mode {mode}")

        game_session = reset_user_game_session(mode)

        current_figure = game_session.get_current_figure()
        logger.info(f"🔧 RESTART DEBUG: Current figure exists: {current_figure is not None}")

        return jsonify({
            "message": "Game restarted successfully",
            "mode": mode,
            "success": True,
            "debug_info": {
                "user_id": user_id,
                "session_complete": game_session.is_complete(),
                "current_figure_index": game_session.current_figure_index,
                "total_figures": len(game_session.figures_order),
                "has_current_figure": current_figure is not None
            }
        })

    except Exception as e:
        logger.error(f"🔧 RESTART DEBUG: Error during restart: {e}")
        return jsonify({"error": f"Failed to restart game: {str(e)}", "success": False}), 500


@app.route('/api/get-hint', methods=['POST'])
def get_hint():
    game_session = get_user_game_session()
    figure = game_session.get_current_figure()
    if not figure:
        return jsonify({"error": "No active figure"}), 400

    mode_config = game_session.get_mode_config()
    hints_allowed = mode_config.get("hints_allowed", 3)

    if game_session.hints_used >= hints_allowed:
        return jsonify({"error": "No more hints allowed in this mode"}), 400

    hints = [
        figure["hint"],
        f"Category: {figure.get('category', 'Historical Figure')}",
        f"Keywords: {', '.join(figure['keywords'][:3])}",
        f"Born in: {abs(figure.get('birth_year', 0))}{' BC' if figure.get('birth_year', 0) < 0 else ''}",
        f"Nationality: {figure.get('nationality', 'Unknown')}",
        f"Try saying: {figure['phonetic_variations'][0] if figure['phonetic_variations'] else figure['name']}",
        f"Full description: {figure['description']}"
    ]

    hint_index = min(game_session.hints_used, len(hints) - 1)
    game_session.hints_used += 1

    return jsonify({
        "hint": hints[hint_index],
        "hint_level": hint_index + 1,
        "hints_remaining": hints_allowed - game_session.hints_used
    })


@app.route('/api/stats')
def get_stats():
    whisper_stats = pronunciation_analyzer.whisper.get_statistics()
    game_session = get_user_game_session()

    session_stats = {
        "current_session_attempts": len(game_session.attempts),
        "current_session_score": game_session.score,
        "current_streak": game_session.streak,
        "current_mode": game_session.mode,
        "figures_completed": game_session.current_figure_index,
        "total_figures": len(game_session.figures_order),
        "active_users": len(user_sessions)
    }

    if game_session.attempts:
        accuracy_counts = {
            "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
            "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
            "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
            "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
        }
        session_stats["accuracy_breakdown"] = accuracy_counts
        session_stats["average_score"] = game_session.score / len(game_session.attempts)

    return jsonify({
        "whisper_stats": whisper_stats,
        "session_stats": session_stats,
        "game_modes": GAME_MODES,
        "cost_savings": {
            "api_calls_avoided": whisper_stats["total_transcriptions"],
            "money_saved": f"${whisper_stats['total_transcriptions'] * 0.006:.2f}",
            "message": "100% cost savings with local Whisper!"
        },
        "performance_metrics": {
            "total_figures_available": len(HISTORICAL_FIGURES),
            "avg_processing_time": whisper_stats.get("average_processing_time", 0),
            "success_rate": whisper_stats.get("success_rate", 0)
        }
    })


# ===== CONVERSATIONAL API ENDPOINTS =====

@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """Start conversational mode for the current figure"""
    try:
        game_session = get_user_game_session()
        conversation_manager = get_user_conversation_manager()

        figure = game_session.get_current_figure()
        if not figure:
            return jsonify({"error": "No current figure available"}), 400

        # Start the conversation
        conversation_manager.start_conversation(figure)

        return jsonify({
            "message": "Conversation started",
            "figure_name": figure["name"],
            "conversation_state": conversation_manager.get_state().value,
            "speech_synthesis_ready": True
        })

    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        return jsonify({"error": f"Failed to start conversation: {str(e)}"}), 500


@app.route('/api/conversation/state')
def get_conversation_state():
    """Get current conversation state"""
    try:
        conversation_manager = get_user_conversation_manager()
        game_session = get_user_game_session()

        state_info = {
            "conversation_state": conversation_manager.get_state().value,
            "is_listening": conversation_manager.is_listening(),
            "is_waiting_for_next": conversation_manager.is_waiting_for_next(),
            "current_figure": game_session.get_current_figure(),
            "game_progress": {
                "current": game_session.current_figure_index + 1,
                "total": len(game_session.figures_order)
            }
        }

        return jsonify(state_info)

    except Exception as e:
        logger.error(f"Error getting conversation state: {e}")
        return jsonify({"error": f"Failed to get conversation state: {str(e)}"}), 500


@app.route('/api/conversation/voice-command', methods=['POST'])
def process_voice_command():
    """Process voice commands like 'next', 'repeat', 'hint'"""
    try:
        data = request.get_json()
        command_text = data.get('command', '').strip()

        if not command_text:
            return jsonify({"error": "No command provided"}), 400

        conversation_manager = get_user_conversation_manager()
        command_processed = conversation_manager.process_voice_command(command_text)

        return jsonify({
            "command": command_text,
            "processed": command_processed,
            "conversation_state": conversation_manager.get_state().value
        })

    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        return jsonify({"error": f"Failed to process voice command: {str(e)}"}), 500


@app.route('/api/conversation/speech-synthesis')
def get_speech_synthesis():
    """Get pending speech synthesis data for frontend"""
    try:
        conversation_manager = get_user_conversation_manager()
        synthesizer = conversation_manager.synthesizer

        # This would normally return pending speech data
        # For now, return current state info
        return jsonify({
            "has_pending_speech": synthesizer.is_speaking,
            "speech_queue_length": len(synthesizer.speech_queue),
            "conversation_state": conversation_manager.get_state().value
        })

    except Exception as e:
        logger.error(f"Error getting speech synthesis: {e}")
        return jsonify({"error": f"Failed to get speech synthesis: {str(e)}"}), 500


@app.route('/api/conversation/voice-activity-config')
def get_voice_activity_config():
    """Get voice activity detection configuration"""
    try:
        # Create VAD instance for configuration
        vad = VoiceActivityDetector(VADSensitivity.MEDIUM)
        config = vad.start_listening()

        return jsonify({
            "vad_config": config,
            "conversational_mode": True,
            "auto_recording": True
        })

    except Exception as e:
        logger.error(f"Error getting VAD config: {e}")
        return jsonify({"error": f"Failed to get VAD config: {str(e)}"}), 500


@app.route('/api/conversation/submit-answer', methods=['POST'])
def submit_conversational_answer():
    """Submit answer in conversational mode"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        game_session = get_user_game_session()
        conversation_manager = get_user_conversation_manager()

        figure = game_session.get_current_figure()
        if not figure:
            return jsonify({"error": "Game completed"}), 400

        # Process the audio (existing logic)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            wav_path = None
            try:
                audio = AudioSegment.from_file(temp_file_path)
                if len(audio) < AUDIO_CONFIG["recording"]["min_duration_ms"]:
                    return jsonify({"error": MESSAGES["errors"]["audio_too_short"]}), 400

                wav_path = temp_file_path.replace('.webm', '.wav')
                audio.export(wav_path, format="wav")
                analysis_file_path = wav_path
            except Exception as e:
                logger.info(f"Audio conversion skipped, using original file: {e}")
                analysis_file_path = temp_file_path

            # Analyze pronunciation
            analysis = pronunciation_analyzer.analyze_pronunciation(analysis_file_path, figure)

            # Create attempt record
            attempt = {
                "figure": figure["name"],
                "transcription": analysis["raw_transcript"],
                "score": analysis["overall_score"],
                "feedback": analysis["feedback"],
                "accuracy": "exact" if analysis["accuracy_score"] >= 90 else
                "close" if analysis["accuracy_score"] >= 70 else
                "partial" if analysis["accuracy_score"] >= 40 else "incorrect",
                "highlights": analysis["highlights"],
                "timestamp": datetime.now().isoformat(),
                "analysis_details": analysis,
                "conversational_mode": True
            }

            game_session.attempts.append(attempt)
            game_session.score += analysis["overall_score"]

            if analysis["accuracy_score"] > 0:
                game_session.streak += 1
            else:
                game_session.streak = 0

            # Provide conversational feedback
            def on_feedback_complete():
                logger.info("🎙️ Conversational feedback complete")

            conversation_manager.provide_feedback(analysis, on_feedback_complete)

            is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)

            response_data = {
                "transcription": analysis["raw_transcript"],
                "analysis": analysis,
                "figure_name": figure["name"],
                "total_score": game_session.score,
                "streak": game_session.streak,
                "attempt_number": len(game_session.attempts),
                "is_final_figure": is_final_figure,
                "conversational_mode": True,
                "conversation_state": conversation_manager.get_state().value
            }

            # Advance to next figure
            game_session.advance_figure()

            if game_session.is_complete():
                game_duration = (datetime.now() - game_session.start_time).total_seconds()
                avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0

                final_stats = {
                    "total_score": round(game_session.score, 1),
                    "average_score": round(avg_score, 1),
                    "duration_seconds": round(game_duration),
                    "attempts": len(game_session.attempts),
                    "max_streak": game_session.streak,
                    "accuracy_breakdown": {
                        "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                        "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                        "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                        "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
                    }
                }

                response_data["final_stats"] = final_stats

                # Handle game completion conversationally
                conversation_manager.handle_game_completion(final_stats)

            return jsonify(response_data)

        finally:
            # Cleanup files
            cleanup_files = [temp_file_path]
            if 'wav_path' in locals() and wav_path and wav_path != temp_file_path:
                cleanup_files.append(wav_path)

            for file_path in cleanup_files:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass

    except Exception as e:
        logger.error(f"Error in conversational audio processing: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500


if __name__ == '__main__':
    logger.info("Starting Enhanced Historical Figures Name Game - Conversational Mode Ready!")
    logger.info(f"Whisper Type: {WHISPER_TYPE}")
    logger.info(f"Total Figures: {len(HISTORICAL_FIGURES)}")
    logger.info("🎙️ Conversational AI features enabled!")
    logger.info("📝 Manual mode: http://localhost:5000/")
    logger.info("🗣️ Conversational mode: http://localhost:5000/conversational")
    logger.info("No API costs - 100% local processing!")

    app.config['MAX_CONTENT_LENGTH'] = AUDIO_CONFIG["recording"]["max_file_size_mb"] * 1024 * 1024
    app.run(debug=True, host='0.0.0.0', port=5000)