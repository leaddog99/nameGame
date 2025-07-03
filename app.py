from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
import os
import tempfile
import json
import time
import threading
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import logging
from pydub import AudioSegment
import numpy as np
import random
import jellyfish
import uuid
from typing import Dict, Optional

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

# FFmpeg configuration (after logger is set up)
try:
    import imageio_ffmpeg as ffmpeg
    from pydub.utils import which

    if not which("ffmpeg"):
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        AudioSegment.converter = ffmpeg_exe
        AudioSegment.ffmpeg = ffmpeg_exe
        logger.info("✅ FFmpeg configured via imageio-ffmpeg")
    else:
        logger.info("✅ System FFmpeg found")
except ImportError:
    logger.warning("⚠️ imageio-ffmpeg not available, audio conversion may fail")
except Exception as e:
    logger.warning(f"⚠️ Error configuring FFmpeg: {e}")

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this-in-production'

validate_figure_data()
logger.info(f"✅ Data validation passed! Loaded {len(HISTORICAL_FIGURES)} figures")


class GameSession:
    def __init__(self, mode="classic"):
        self.mode = mode
        self.current_figure_index = 0
        self.score = 0
        self.attempts = []
        self.start_time = datetime.now()
        self.last_accessed = datetime.now()  # For session management
        self.hints_used = 0
        self.streak = 0  # Current streak (resets on wrong answer)
        self.max_streak = 0  # ✅ FIXED: Best streak achieved this session
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
        self.last_accessed = datetime.now()  # Update access time
        if self.current_figure_index < len(self.figures_order):
            if hasattr(self, 'selected_figures'):
                return self.selected_figures[self.current_figure_index]
            else:
                return HISTORICAL_FIGURES[self.figures_order[self.current_figure_index]]
        return None

    def advance_figure(self):
        self.current_figure_index += 1
        self.last_accessed = datetime.now()

    def is_complete(self):
        return self.current_figure_index >= len(self.figures_order)

    def get_mode_config(self):
        return get_game_mode_config(self.mode)

    def update_streak(self, is_correct):
        """✅ FIXED: Properly update both current streak and max streak"""
        if is_correct:
            self.streak += 1
            # Update max streak if current streak is higher
            if self.streak > self.max_streak:
                self.max_streak = self.streak
                logger.info(f"🔥 NEW BEST STREAK: {self.max_streak}")
        else:
            # Reset current streak but keep max_streak
            if self.streak > 0:
                logger.info(f"💔 Streak broken at {self.streak}, best remains {self.max_streak}")
            self.streak = 0


class SessionManager:
    """Enhanced session management with automatic cleanup and monitoring"""

    def __init__(self, session_timeout_minutes=30, cleanup_interval_minutes=10):
        self.user_sessions: Dict[str, GameSession] = {}
        self.user_conversations: Dict[str, ConversationManager] = {}
        self.session_metadata: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        # Session statistics
        self.total_sessions_created = 0
        self.total_sessions_cleaned = 0

        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        # Optional: Load persistent sessions
        self.load_sessions_from_disk()

        logger.info(
            f"🔧 SessionManager initialized - timeout: {session_timeout_minutes}min, cleanup: {cleanup_interval_minutes}min")

    def get_or_create_session(self, user_id: str, mode: str = "classic") -> GameSession:
        """Get existing session or create new one with metadata tracking"""
        current_time = datetime.now()

        if user_id not in self.user_sessions:
            # Create new session
            self.user_sessions[user_id] = GameSession(mode)
            self.session_metadata[user_id] = {
                'created_at': current_time,
                'last_accessed': current_time,
                'mode': mode,
                'total_requests': 0,
                'user_agent': request.headers.get('User-Agent', 'Unknown') if request else 'Unknown',
                'ip_address': request.remote_addr if request else 'Unknown'
            }
            self.total_sessions_created += 1
            logger.info(f"📱 Created new session for user: {user_id[:8]}... (mode: {mode})")

        # Update access time and request count
        self.session_metadata[user_id]['last_accessed'] = current_time
        self.session_metadata[user_id]['total_requests'] += 1

        return self.user_sessions[user_id]

    def get_or_create_conversation(self, user_id: str) -> ConversationManager:
        """Get existing conversation manager or create new one"""
        if user_id not in self.user_conversations:
            conversation_manager = ConversationManager()

            # Setup callbacks
            def on_next_requested():
                logger.info(f"🎙️ CONVERSATION: User {user_id[:8]}... requested next figure")

            def on_state_change(old_state, new_state):
                logger.info(f"🎙️ CONVERSATION: State change {old_state.value} → {new_state.value}")

            conversation_manager.on_next_requested = on_next_requested
            conversation_manager.on_state_change = on_state_change
            self.user_conversations[user_id] = conversation_manager
            logger.info(f"🗣️ Created conversation manager for user: {user_id[:8]}...")

        return self.user_conversations[user_id]

    def cleanup_expired_sessions(self):
        """Remove sessions that haven't been accessed recently"""
        current_time = datetime.now()
        expired_users = []

        for user_id, metadata in self.session_metadata.items():
            if current_time - metadata['last_accessed'] > self.session_timeout:
                expired_users.append(user_id)

        for user_id in expired_users:
            self.remove_user_session(user_id)
            self.total_sessions_cleaned += 1
            logger.info(f"🧹 Cleaned up expired session for user: {user_id[:8]}...")

        if expired_users:
            logger.info(f"🗑️ Session cleanup: removed {len(expired_users)} expired sessions")

    def remove_user_session(self, user_id: str):
        """Safely remove all data for a user"""
        # Stop conversation if active
        if user_id in self.user_conversations:
            try:
                self.user_conversations[user_id].stop_conversation()
            except Exception as e:
                logger.warning(f"Error stopping conversation for {user_id[:8]}...: {e}")
            del self.user_conversations[user_id]

        # Remove game session
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]

        # Remove metadata
        if user_id in self.session_metadata:
            del self.session_metadata[user_id]

    def get_session_stats(self) -> dict:
        """Get comprehensive statistics about active sessions"""
        current_time = datetime.now()
        active_sessions = len(self.user_sessions)
        recent_sessions = 0
        mode_breakdown = {}

        for user_id, metadata in self.session_metadata.items():
            # Count recently active
            if current_time - metadata['last_accessed'] < timedelta(minutes=5):
                recent_sessions += 1

            # Mode breakdown
            mode = metadata.get('mode', 'unknown')
            mode_breakdown[mode] = mode_breakdown.get(mode, 0) + 1

        return {
            'total_active_sessions': active_sessions,
            'recently_active_sessions': recent_sessions,
            'total_conversations': len(self.user_conversations),
            'total_sessions_created': self.total_sessions_created,
            'total_sessions_cleaned': self.total_sessions_cleaned,
            'mode_breakdown': mode_breakdown,
            'memory_usage_estimate': self._estimate_memory_usage(),
            'session_timeout_minutes': self.session_timeout.total_seconds() / 60,
            'uptime_hours': self._get_uptime_hours()
        }

    def get_detailed_session_info(self) -> list:
        """Get detailed info about each session (admin use)"""
        current_time = datetime.now()
        sessions_info = []

        for user_id, metadata in self.session_metadata.items():
            session = self.user_sessions.get(user_id)
            conversation = self.user_conversations.get(user_id)

            last_accessed_minutes = (current_time - metadata['last_accessed']).total_seconds() / 60

            sessions_info.append({
                'user_id': user_id[:8] + '...',  # Truncated for privacy
                'mode': metadata.get('mode', 'unknown'),
                'created_at': metadata['created_at'].isoformat(),
                'last_accessed_minutes_ago': round(last_accessed_minutes, 1),
                'total_requests': metadata['total_requests'],
                'game_progress': f"{session.current_figure_index}/{len(session.figures_order)}" if session else "No session",
                'score': round(session.score) if session else 0,
                'streak': session.streak if session else 0,
                'has_conversation': conversation is not None,
                'conversation_state': conversation.get_state().value if conversation else 'None'
            })

        # Sort by last accessed (most recent first)
        sessions_info.sort(key=lambda x: x['last_accessed_minutes_ago'])
        return sessions_info

    def _estimate_memory_usage(self) -> str:
        """Rough estimate of memory usage"""
        sessions_size = len(self.user_sessions) * 2048  # ~2KB per session estimate
        conversations_size = len(self.user_conversations) * 1024  # ~1KB per conversation
        metadata_size = len(self.session_metadata) * 512  # ~512B per metadata
        total_bytes = sessions_size + conversations_size + metadata_size

        if total_bytes < 1024:
            return f"{total_bytes} B"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.1f} KB"
        else:
            return f"{total_bytes / (1024 * 1024):.1f} MB"

    def _get_uptime_hours(self) -> float:
        """Calculate uptime since SessionManager was created"""
        if hasattr(self, '_start_time'):
            return (datetime.now() - self._start_time).total_seconds() / 3600
        return 0

    def _cleanup_loop(self):
        """Background thread for periodic cleanup"""
        self._start_time = datetime.now()

        while True:
            try:
                time.sleep(self.cleanup_interval.total_seconds())
                self.cleanup_expired_sessions()

                # Optional: Save sessions every hour
                if hasattr(self, '_last_save'):
                    if datetime.now() - self._last_save > timedelta(hours=1):
                        self.save_sessions_to_disk()
                        self._last_save = datetime.now()
                else:
                    self._last_save = datetime.now()

            except Exception as e:
                logger.error(f"❌ Error in session cleanup: {e}")

    def save_sessions_to_disk(self, filepath: str = "sessions_backup.json"):
        """Optional: Save session data to disk for persistence"""
        try:
            backup_data = {
                'sessions': {},
                'metadata': {},
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            # Only save serializable session data
            for user_id, session in self.user_sessions.items():
                backup_data['sessions'][user_id] = {
                    'mode': session.mode,
                    'current_figure_index': session.current_figure_index,
                    'score': session.score,
                    'streak': session.streak,
                    'max_streak': session.max_streak,  # ✅ FIXED: Include max_streak in backup
                    'hints_used': session.hints_used,
                    'attempts': session.attempts[-5:],  # Only save last 5 attempts
                    'start_time': session.start_time.isoformat(),
                    'last_accessed': session.last_accessed.isoformat()
                }

            # Save metadata (without sensitive info)
            for user_id, metadata in self.session_metadata.items():
                backup_data['metadata'][user_id] = {
                    'created_at': metadata['created_at'].isoformat(),
                    'last_accessed': metadata['last_accessed'].isoformat(),
                    'mode': metadata['mode'],
                    'total_requests': metadata['total_requests']
                }

            with open(filepath, 'w') as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"💾 Saved {len(self.user_sessions)} sessions to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save sessions: {e}")

    def load_sessions_from_disk(self, filepath: str = "sessions_backup.json"):
        """Optional: Load session data from disk"""
        try:
            if not os.path.exists(filepath):
                return

            with open(filepath, 'r') as f:
                backup_data = json.load(f)

            # Only load recent sessions (within last 2 hours)
            backup_time = datetime.fromisoformat(backup_data['timestamp'])
            if datetime.now() - backup_time > timedelta(hours=2):
                logger.info("📁 Session backup too old, skipping load")
                return

            # Restore metadata only (sessions will be recreated on demand)
            for user_id, metadata in backup_data.get('metadata', {}).items():
                # Convert ISO strings back to datetime
                metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
                metadata['last_accessed'] = datetime.fromisoformat(metadata['last_accessed'])
                # Add missing fields with defaults
                metadata['user_agent'] = 'Restored'
                metadata['ip_address'] = 'Restored'

                self.session_metadata[user_id] = metadata

            logger.info(f"📂 Loaded session metadata for {len(self.session_metadata)} users")

        except Exception as e:
            logger.error(f"❌ Failed to load sessions: {e}")


# Initialize enhanced session manager
session_manager = SessionManager(session_timeout_minutes=30, cleanup_interval_minutes=10)

pronunciation_analyzer = PronunciationAnalyzer()


def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"🆔 Created new user session: {session['user_id'][:8]}...")
    return session['user_id']


def get_user_game_session(mode="classic"):
    user_id = get_user_id()
    return session_manager.get_or_create_session(user_id, mode)


def reset_user_game_session(mode="classic"):
    user_id = get_user_id()
    logger.info(f"🔧 RESET DEBUG: === STARTING RESET FOR USER {user_id[:8]}... ===")
    logger.info(f"🔧 RESET DEBUG: Requested mode: {mode}")

    # Remove existing session
    session_manager.remove_user_session(user_id)
    logger.info(f"🔧 RESET DEBUG: Old session removed")

    # Create new session
    new_session = session_manager.get_or_create_session(user_id, mode)
    logger.info(f"🔧 RESET DEBUG: New session created")
    logger.info(f"🔧 RESET DEBUG: New session complete: {new_session.is_complete()}")
    logger.info(f"🔧 RESET DEBUG: === RESET COMPLETE FOR USER {user_id[:8]}... ===")
    return new_session


def get_user_conversation_manager():
    user_id = get_user_id()
    return session_manager.get_or_create_conversation(user_id)


def reset_user_conversation():
    user_id = get_user_id()
    if user_id in session_manager.user_conversations:
        session_manager.user_conversations[user_id].stop_conversation()
        del session_manager.user_conversations[user_id]
        logger.info(f"🔄 Reset conversation manager for user: {user_id[:8]}...")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/conversational')
def conversational_interface():
    return render_template('conversational.html')


@app.route('/api/game-state')
def get_game_state():
    user_id = get_user_id()
    logger.info(f"🔧 GAME STATE DEBUG: Getting state for user {user_id[:8]}...")
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
            "name": figure["name"]
        } if figure else None,
        "progress": {
            "current": game_session.current_figure_index + 1,
            "total": len(game_session.figures_order)
        },
        "score": game_session.score,
        "streak": game_session.streak,
        "max_streak": game_session.max_streak,  # ✅ FIXED: Include max_streak in game state
        "is_complete": game_session.is_complete(),
        "mode": game_session.mode,
        "hints_used": game_session.hints_used
    })


# New session management endpoints
@app.route('/api/session-stats')
def get_session_stats():
    """Get session statistics for monitoring"""
    stats = session_manager.get_session_stats()
    return jsonify(stats)


@app.route('/api/admin/session-details')
def get_session_details():
    """Get detailed session information (admin endpoint)"""
    try:
        detailed_info = session_manager.get_detailed_session_info()
        return jsonify({
            "sessions": detailed_info,
            "summary": session_manager.get_session_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/cleanup-sessions', methods=['POST'])
def force_cleanup_sessions():
    """Force immediate session cleanup (admin endpoint)"""
    try:
        before_count = len(session_manager.user_sessions)
        session_manager.cleanup_expired_sessions()
        after_count = len(session_manager.user_sessions)
        cleaned_count = before_count - after_count

        return jsonify({
            "message": "Session cleanup completed",
            "sessions_before": before_count,
            "sessions_after": after_count,
            "sessions_cleaned": cleaned_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/save-sessions', methods=['POST'])
def save_sessions():
    """Force save sessions to disk (admin endpoint)"""
    try:
        session_manager.save_sessions_to_disk()
        return jsonify({"message": "Sessions saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Keep all existing endpoints with streak fixes
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

        # ✅ FIXED: Use proper streak update method
        is_correct = analysis["accuracy_score"] > 0
        game_session.update_streak(is_correct)

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
            "max_streak": game_session.max_streak,  # ✅ FIXED: Include max_streak in response
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
                "max_streak": game_session.max_streak,  # ✅ FIXED: Use max_streak instead of current streak
                "current_streak": game_session.streak,  # ✅ BONUS: Also show current streak
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
        "max_streak": game_session.max_streak,  # ✅ FIXED: Use max_streak
        "current_streak": game_session.streak,  # ✅ BONUS: Also show current
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

    # ✅ FIXED: Use proper streak update method instead of direct assignment
    game_session.update_streak(False)  # Timeout counts as incorrect

    is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)
    response_data = {
        "transcription": "⏰ Time expired",
        "analysis": attempt["analysis_details"],
        "figure_name": figure["name"],
        "total_score": game_session.score,
        "streak": game_session.streak,
        "max_streak": game_session.max_streak,  # ✅ FIXED: Include max_streak
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
            "max_streak": game_session.max_streak,  # ✅ FIXED: Use max_streak
            "current_streak": game_session.streak,  # ✅ BONUS: Also show current
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
        logger.info(f"🔧 RESTART DEBUG: User {user_id[:8]}... requesting restart with mode {mode}")
        game_session = reset_user_game_session(mode)
        current_figure = game_session.get_current_figure()
        logger.info(f"🔧 RESTART DEBUG: Current figure exists: {current_figure is not None}")
        return jsonify({
            "message": "Game restarted successfully",
            "mode": mode,
            "success": True,
            "debug_info": {
                "user_id": user_id[:8] + '...',
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
        "max_streak": game_session.max_streak,  # ✅ FIXED: Include max_streak in stats
        "current_mode": game_session.mode,
        "figures_completed": game_session.current_figure_index,
        "total_figures": len(game_session.figures_order),
        "active_users": len(session_manager.user_sessions)
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

    # Add session management stats
    session_mgmt_stats = session_manager.get_session_stats()

    return jsonify({
        "whisper_stats": whisper_stats,
        "session_stats": session_stats,
        "session_management": session_mgmt_stats,
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


# Conversational endpoints with fixes
@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    try:
        game_session = get_user_game_session()

        # ✅ FIXED: Auto-restart if game is complete
        auto_restarted = False
        if game_session.is_complete():
            logger.info("🔄 Game complete, auto-restarting for new conversation...")
            game_session = reset_user_game_session('classic')
            auto_restarted = True

        conversation_manager = get_user_conversation_manager()
        figure = game_session.get_current_figure()
        if not figure:
            return jsonify({"error": "No current figure available"}), 400

        conversation_manager.start_conversation(figure)
        return jsonify({
            "message": "Conversation started",
            "figure_name": figure["name"],
            "conversation_state": conversation_manager.get_state().value,
            "speech_synthesis_ready": True,
            "auto_restarted": auto_restarted  # ✅ Tell frontend we restarted
        })
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        return jsonify({"error": f"Failed to start conversation: {str(e)}"}), 500


@app.route('/api/conversation/state')
def get_conversation_state():
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


@app.route('/api/conversation/voice-command-audio', methods=['POST'])
def process_voice_command_audio():
    """Process voice commands from audio without affecting game state - SIMPLE VERSION"""
    try:
        logger.info("🎙️ VOICE COMMAND: Processing voice command audio")

        if 'audio' not in request.files:
            logger.error("No audio file provided for voice command")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']

        # Save the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_file.save(temp_file.name)
            logger.info(f"💾 Saved voice command audio: {temp_file.name}, size: {os.path.getsize(temp_file.name)} bytes")
            temp_file_path = temp_file.name

        try:
            # FIXED: Use simpler approach - get current figure and use that for analysis
            transcription = ""

            try:
                # Get actual current figure for voice command processing
                game_session = get_user_game_session()
                figure = game_session.get_current_figure()

                if not figure:
                    # If no figure, create a minimal one
                    figure = {
                        "name": "unknown",
                        "keywords": [],
                        "phonetic_variations": [],
                        "difficulty": 1,
                        "description": "voice command",
                        "hint": "voice command"
                    }

                logger.info(f"🎯 Using figure for voice command: {figure['name']}")

                # Skip conversion entirely - use WebM directly
                analysis = pronunciation_analyzer.analyze_pronunciation(temp_file_path, figure)
                transcription = analysis.get('raw_transcript', '').strip()
                logger.info(f"✅ Voice command transcribed: '{transcription}'")

            except Exception as analysis_error:
                logger.error(f"❌ Voice command analysis failed: {analysis_error}")
                transcription = ""

            # Return just the transcription, no game state changes
            return jsonify({
                "transcription": transcription,
                "command_mode": True,
                "message": "Voice command processed successfully"
            })

        except Exception as e:
            logger.error(f"Error processing voice command audio: {e}")
            return jsonify({"error": f"Failed to process voice command: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error in voice command processing: {e}")
        return jsonify({"error": f"Failed to process voice command: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ Cleaned up voice command file: {temp_file_path}")
            except:
                pass


@app.route('/api/conversation/speech-synthesis')
def get_speech_synthesis():
    try:
        conversation_manager = get_user_conversation_manager()
        synthesizer = conversation_manager.synthesizer
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
    try:
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
    logger.info("🎙️ CONVERSATIONAL: Received audio submission")
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    game_session = get_user_game_session()
    conversation_manager = get_user_conversation_manager()
    figure = game_session.get_current_figure()
    if not figure:
        return jsonify({"error": "Game completed"}), 400
    time_to_record = request.form.get('time_to_record', type=float)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name
    try:
        # COPY EXACT LOGIC FROM WORKING MANUAL MODE (submit_audio)
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


        # COPY EXACT ANALYSIS FROM WORKING MANUAL MODE
        analysis = pronunciation_analyzer.analyze_pronunciation(analysis_file_path, figure)
        if time_to_record is not None:
            analysis["time_to_record"] = time_to_record

        # COPY EXACT ATTEMPT BUILDING FROM WORKING MANUAL MODE
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

        # ✅ FIXED: Use proper streak update method
        is_correct = analysis["accuracy_score"] > 0
        game_session.update_streak(is_correct)

        is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)
        logger.info(f"🔧 FINAL CARD DEBUG: Current index: {game_session.current_figure_index}")
        logger.info(f"🔧 FINAL CARD DEBUG: Total figures: {len(game_session.figures_order)}")
        logger.info(f"🔧 FINAL CARD DEBUG: Is final figure: {is_final_figure}")

        # ONLY DIFFERENCE: Build response for conversational mode
        response_data = {
            "transcription": analysis["raw_transcript"],
            "analysis": analysis,
            "figure_name": figure["name"],
            "total_score": game_session.score,
            "streak": game_session.streak,
            "max_streak": game_session.max_streak,  # ✅ FIXED: Include max_streak
            "attempt_number": len(game_session.attempts),
            "is_final_figure": is_final_figure,
            "conversational_mode": True,
            "conversation_state": conversation_manager.get_state().value
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
                "max_streak": game_session.max_streak,  # ✅ FIXED: Use max_streak
                "current_streak": game_session.streak,  # ✅ BONUS: Also show current
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
                }
            }
            conversation_manager.handle_game_completion(response_data["final_stats"])

        # ONLY DIFFERENCE: Add conversational feedback
        def on_feedback_complete():
            logger.info("Successfully provided conversational feedback")

        conversation_manager.provide_feedback(analysis, on_feedback_complete)

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    finally:
        # COPY EXACT CLEANUP FROM WORKING MANUAL MODE
        cleanup_files = [temp_file_path]
        if 'wav_path' in locals() and wav_path and wav_path != temp_file_path:
            cleanup_files.append(wav_path)
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass


@app.route('/api/calibrate-vad', methods=['POST'])
def calibrate_vad():
    try:
        data = request.get_json()
        ambient_levels = data.get('ambient_levels', [])
        if not ambient_levels:
            logger.warning('No ambient levels provided for VAD calibration')
            return jsonify({"error": "No ambient levels provided"}), 400
        vad = VoiceActivityDetector(VADSensitivity.MEDIUM)
        vad.calibrate_environment(ambient_levels)
        logger.info(
            f"VAD calibrated with baseline: {np.mean(ambient_levels):.3f}, min: {np.min(ambient_levels):.3f}, max: {np.max(ambient_levels):.3f}")
        return jsonify({"message": "VAD successfully calibrated successfully"})
    except Exception as e:
        logger.error(f"Error calibrating VAD: {e}")
        return jsonify({"error": f"Failed to calibrate VAD: {str(e)}"}), 400


if __name__ == '__main__':
    logger.info(f"🚀 Starting Enhanced Historical Figures Game Server!")
    logger.info(f"🎤 Conversational mode AI features enabled!")
    logger.info(f"🧠 Enhanced session management active!")
    logger.info(f"🔥 Fixed streak tracking system!")
    logger.info(f"🔄 Auto-restart for completed games!")
    logger.info(f"🔍 Whisper Type: {WHISPER_TYPE}")
    logger.info(f"📚 Total Figures Available: {len(HISTORICAL_FIGURES)}")
    logger.info("📝 Manual mode: http://localhost:5000/")
    logger.info("🗣️ Conversational mode: http://localhost:5000/conversational")
    logger.info("📊 Session stats: http://localhost:5000/api/session-stats")
    logger.info("🔧 Admin panel: http://localhost:5000/api/admin/session-details")
    logger.info("💰 No API costs - 100% local processing!")

    app.config['MAX_CONTENT_LENGTH'] = AUDIO_CONFIG["recording"]["max_file_size_mb"] * 1024 * 1024
    app.run(debug=True, host='0.0.0.0', port=5000)