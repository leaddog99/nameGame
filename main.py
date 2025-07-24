#from flask import Flask, request, jsonify, render_template, session, send_file
#from flask_cors import CORS
import os
import tempfile
import json
import time
import threading
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import logging

import builtins
if not hasattr(builtins, 'basestring'):
    builtins.basestring = str


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

from fastapi import FastAPI, Request, Depends, File, UploadFile, APIRouter, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from typing import Optional

from starlette.middleware.sessions import SessionMiddleware

from schemas import HistoricalFigureCreate
from postgres_models import HistoricalFigure, GameSession
from db import get_db
from sqlalchemy.orm import Session



templates = Jinja2Templates(directory="templates")

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

'''

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this-in-production'

'''

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

validate_figure_data()
logger.info(f"✅ Data validation passed! Loaded {len(HISTORICAL_FIGURES)} figures")

'''
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
'''

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

    def get_or_create_session(self, user_id: str, mode: str = "classic", request: Optional[Request] = None) -> GameSession:
        current_time = datetime.now()

        if user_id not in self.user_sessions:
            #self.user_sessions[user_id] = GameSession(mode)
            new_session = GameSession() ##
            new_session.mode = mode ##
            new_session.user_id = user_id ##
            # initialize other fields as needed ##
            self.user_sessions[user_id] = new_session ##
            self.session_metadata[user_id] = {
                'created_at': current_time,
                'last_accessed': current_time,
                'mode': mode,
                'total_requests': 0,
                'user_agent': request.headers.get('User-Agent', 'Unknown') if request else 'Unknown',
                'ip_address': request.client.host if request else 'Unknown'  # ✅ Proper FastAPI way
            }
            self.total_sessions_created += 1
            logger.info(f"📱 Created new session for user: {user_id[:8]}... (mode: {mode})")

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

now = datetime.now(timezone.utc)

def get_user_id(request: Request) -> str:
    if 'user_id' not in request.session:
        new_id = str(uuid.uuid4())
        request.session['user_id'] = new_id
        logger.info(f"🆔 Created new user session: {new_id[:8]}...")
    return request.session['user_id']


'''
def get_user_game_session(request: Request, mode="classic"):
    user_id = get_user_id(request)
    return session_manager.get_or_create_session(user_id, mode, request)
'''

def get_user_game_session(request: Request, db: Session) -> GameSession:
    user_id = get_user_id(request)
    session = db.query(GameSession).filter_by(user_id=user_id).first()

    if session:
        return session

    # Create a new game session
    session = GameSession(user_id=user_id)

    # Step 3: Initialize selected_figures, figures_order, etc.
    figures = random.sample(HISTORICAL_FIGURES, k=8)
    session.selected_figures = figures
    session.figures_order = list(range(len(figures)))
    session.current_figure_index = 0
    session.total_figures = len(figures)
    session.start_time = datetime.utcnow()
    session.score = 0
    session.streak = 0
    session.max_streak = 0
    session.attempts = []
    session.hints_used = 0
    session.is_complete = False

    db.add(session)
    db.commit()
    db.refresh(session)

    return session

def generate_figures_order(db: Session, mode: str = "classic") -> list[int]:
    query = db.query(HistoricalFigure.id)

    # Optional: You could add filtering here by mode
    # if mode == "easy":
    #     query = query.filter(HistoricalFigure.difficulty <= 2)

    figure_ids = [row.id for row in query.all()]
    random.shuffle(figure_ids)
    return figure_ids

def get_selected_figures(db: Session, figure_ids: list[int]) -> list[dict]:
    figures = db.query(HistoricalFigure).filter(HistoricalFigure.id.in_(figure_ids)).all()

    # Convert to dicts for JSON storage in GameSession
    id_to_figure = {fig.id: fig for fig in figures}
    return [
        {
            "id": id_to_figure[fig_id].id,
            "name": id_to_figure[fig_id].name,
            "description": id_to_figure[fig_id].description,
            "hint": id_to_figure[fig_id].hint,
            "difficulty": id_to_figure[fig_id].difficulty,
            "category": id_to_figure[fig_id].category,
            "nationality": id_to_figure[fig_id].nationality,
            "birth_year": id_to_figure[fig_id].birth_year,
            "pronunciation_tips": id_to_figure[fig_id].pronunciation_tips,
        }
        for fig_id in figure_ids
        if fig_id in id_to_figure
    ]

def update_streak(game_session, is_correct: bool):
    if is_correct:
        game_session.streak += 1
        if game_session.streak > game_session.max_streak:
            game_session.max_streak = game_session.streak
            logger.info(f"🔥 NEW BEST STREAK: {game_session.max_streak}")
    else:
        if game_session.streak > 0:
            logger.info(f"💔 Streak broken at {game_session.streak}, best remains {game_session.max_streak}")
        game_session.streak = 0

def advance_figure(game_session):
        game_session.current_figure_index += 1
        game_session.last_accessed = datetime.now()

def normalize_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: normalize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_numpy(i) for i in obj]
    return obj

def reset_user_game_session(request: Request, db: Session, mode: str = "classic") -> GameSession:
    user_id = get_user_id(request)

    # Delete old session if it exists
    old_session = db.query(GameSession).filter(GameSession.user_id == user_id).first()
    if old_session:
        db.delete(old_session)
        db.commit()
        logger.info("🔧 RESET DEBUG: Old session removed")

    # Generate new game data
    figures_order = generate_figures_order(db, mode)
    selected_figures = get_selected_figures(db, figures_order)

    # Create new session
    new_session = GameSession(
        user_id=user_id,
        current_figure_index=0,
        total_figures=len(figures_order),
        score=0,
        streak=0,
        max_streak=0,
        is_complete=False,
        mode=mode,
        hints_used=0,
        attempts=[],
        figures_order=figures_order,
        selected_figures=selected_figures,
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    logger.info(f"📱 Created new session for user: {user_id[:8]}... (mode: {mode})")
    return new_session


def get_user_conversation_manager(request: Request):
    user_id = get_user_id(request)
    return session_manager.get_or_create_conversation(user_id)


def reset_user_conversation(request: Request):
    user_id = get_user_id(request)
    if user_id in session_manager.user_conversations:
        session_manager.user_conversations[user_id].stop_conversation()
        del session_manager.user_conversations[user_id]
        logger.info(f"🔄 Reset conversation manager for user: {user_id[:8]}...")

def get_game_mode_config(mode_name):
    """Get configuration for a specific game mode"""
    return GAME_MODES.get(mode_name, GAME_MODES["classic"])

'''
def get_current_figure(session: GameSession):
    session.last_accessed = now

    if session.current_figure_index < len(session.figures_order):
        if session.selected_figures:
            return session.selected_figures[session.current_figure_index]
        else:
            return HISTORICAL_FIGURES[session.figures_order[session.current_figure_index]]
    return None
'''

def get_current_figure(session: GameSession) -> Optional[dict]:
    session.last_accessed = datetime.utcnow()

    if not session.figures_order or not isinstance(session.figures_order, list):
        logger.warning(f"[get_current_figure] Invalid or missing figures_order: {session.figures_order}")
        return None

    if session.current_figure_index is None:
        logger.warning(f"[get_current_figure] current_figure_index is None")
        return None

    if session.current_figure_index >= len(session.figures_order):
        logger.warning(f"[get_current_figure] Index {session.current_figure_index} out of range for figures_order length {len(session.figures_order)}")
        return None

    if session.selected_figures and isinstance(session.selected_figures, list):
        try:
            return session.selected_figures[session.current_figure_index]
        except IndexError:
            logger.warning(f"[get_current_figure] selected_figures IndexError: index {session.current_figure_index}, len {len(session.selected_figures)}")
            return None

    try:
        figure_idx = session.figures_order[session.current_figure_index]
        return HISTORICAL_FIGURES[figure_idx]
    except (IndexError, TypeError) as e:
        logger.warning(f"[get_current_figure] Failed to retrieve figure from HISTORICAL_FIGURES. Index: {figure_idx}, Error: {e}")
        return None


@app.post("/api/historical-figures")
def create_historical_figure(figure: HistoricalFigureCreate, db: Session = Depends(get_db)):
    db_figure = HistoricalFigure(**figure.model_dump())
    db.add(db_figure)
    db.commit()
    db.refresh(db_figure)
    return db_figure

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/conversational', response_class=HTMLResponse)
async def conversational_interface(request: Request):
    return templates.TemplateResponse('conversational.html', {"request": request})

'''
@app.get('/api/game-state')
def get_game_state(request: Request):
    user_id = get_user_id(request)
    logger.info(f"🔧 GAME STATE DEBUG: Getting state for user {user_id[:8]}...")
    game_session = get_user_game_session(request)
    figure = game_session.get_current_figure()
    logger.info(f"🔧 GAME STATE DEBUG: Session complete: {game_session.is_complete()}")
    logger.info(f"🔧 GAME STATE DEBUG: Current figure index: {game_session.current_figure_index}")
    logger.info(f"🔧 GAME STATE DEBUG: Figure exists: {figure is not None}")
    return JSONResponse(content={
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
'''

@app.get('/api/game-state')
def get_game_state(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)

    # Retrieve existing session
    game_session = db.query(GameSession).filter(GameSession.user_id == user_id).first()

    # Create new session if it doesn't exist
    if not game_session:
        total_figures = db.query(HistoricalFigure).count()
        if total_figures == 0:
            return JSONResponse(content={"error": "No historical figures found"}, status_code=404)

        game_session = GameSession(
            user_id=user_id,
            current_figure_index=0,
            total_figures=total_figures,
            score=0,
            streak=0,
            max_streak=0,
            is_complete=False,
            mode="classic",
            hints_used=0
        )
        db.add(game_session)
        db.commit()
        db.refresh(game_session)
    else:
        # Update total_figures in case new data was added
        updated_total = db.query(HistoricalFigure).count()
        if game_session.total_figures != updated_total:
            game_session.total_figures = updated_total
            db.commit()
            db.refresh(game_session)
            # Optional: print or log update
            # print(f"Updated total_figures for user {user_id}: {updated_total}")

    # Get the current figure using stable ordering
    current_index = game_session.current_figure_index
    figure = (
        db.query(HistoricalFigure)
        .order_by(HistoricalFigure.id)
        .offset(current_index)
        .limit(1)
        .first()
    )

    if not figure:
        return JSONResponse(
            content={"error": "Historical figure not found at current index"},
            status_code=404
        )

    # Update figure info in session
    game_session.current_figure_name = figure.name
    game_session.current_figure_description = figure.description
    game_session.current_figure_hint = figure.hint
    game_session.current_figure_difficulty = figure.difficulty
    game_session.current_figure_category = figure.category
    game_session.current_figure_nationality = figure.nationality
    game_session.current_figure_birth_year = figure.birth_year
    game_session.current_figure_pronunciation_tips = figure.pronunciation_tips
    db.commit()
    db.refresh(game_session)

    return JSONResponse(content={
        "current_figure": {
            "name": figure.name,
            "description": figure.description,
            "hint": figure.hint,
            "difficulty": figure.difficulty,
            "category": figure.category,
            "nationality": figure.nationality,
            "birth_year": figure.birth_year,
            "pronunciation_tips": figure.pronunciation_tips
        },
        "progress": {
            "current": current_index + 1,
            "total": game_session.total_figures
        },
        "score": game_session.score,
        "streak": game_session.streak,
        "max_streak": game_session.max_streak,
        "is_complete": game_session.is_complete,
        "mode": game_session.mode,
        "hints_used": game_session.hints_used
    })


# New session management endpoints
@app.get('/api/session-stats')
def get_session_stats():
    """Get session statistics for monitoring"""
    stats = session_manager.get_session_stats()
    return JSONResponse(content=stats)


@app.get('/api/admin/session-details')
def get_session_details():
    """Get detailed session information (admin endpoint)"""
    try:
        detailed_info = session_manager.get_detailed_session_info()
        return JSONResponse(content={
            "sessions": detailed_info,
            "summary": session_manager.get_session_stats()
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}), 500


@app.post('/api/admin/cleanup-sessions')
def force_cleanup_sessions():
    """Force immediate session cleanup (admin endpoint)"""
    try:
        before_count = len(session_manager.user_sessions)
        session_manager.cleanup_expired_sessions()
        after_count = len(session_manager.user_sessions)
        cleaned_count = before_count - after_count

        return JSONResponse(content={
            "message": "Session cleanup completed",
            "sessions_before": before_count,
            "sessions_after": after_count,
            "sessions_cleaned": cleaned_count
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}), 500


@app.post('/api/admin/save-sessions')
def save_sessions():
    """Force save sessions to disk (admin endpoint)"""
    try:
        session_manager.save_sessions_to_disk()
        return JSONResponse(content={"message": "Sessions saved successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}), 500


'''
# Keep all existing endpoints with streak fixes
@app.post('/api/submit-audio')
async def submit_audio(
    request: Request,
    audio: UploadFile = File(...),
    time_to_record: Optional[float] = Form(None)
):
    if not audio:
        return JSONResponse(content={"error": "No audio file provided"}, status_code=400)

    # Pass the request object here so get_user_game_session can access headers
    game_session = get_user_game_session(request)
    figure = game_session.get_current_figure()
    if not figure:
        return JSONResponse(content={"error": "Game completed"}, status_code=400)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            contents = await audio.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        wav_path = None
        try:
            audio_segment = AudioSegment.from_file(temp_file_path)
            if len(audio_segment) < AUDIO_CONFIG["recording"]["min_duration_ms"]:
                return JSONResponse(content={"error": MESSAGES["errors"]["audio_too_short"]}, status_code=400)

            wav_path = temp_file_path.replace('.webm', '.wav')
            audio_segment.export(wav_path, format="wav")
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
            "accuracy": (
                "exact" if analysis["accuracy_score"] >= 90 else
                "close" if analysis["accuracy_score"] >= 70 else
                "partial" if analysis["accuracy_score"] >= 40 else
                "incorrect"
            ),
            "highlights": analysis["highlights"],
            "timestamp": datetime.now().isoformat(),
            "analysis_details": analysis,
            "time_to_record": time_to_record
        }

        game_session.attempts.append(attempt)
        game_session.score += analysis["overall_score"]

        # Update streak properly
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
            "max_streak": game_session.max_streak,
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
                "max_streak": game_session.max_streak,
                "current_streak": game_session.streak,
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"]),
                }
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(content={"error": f"Failed to process audio: {str(e)}"}, status_code=500)

    finally:
        cleanup_files = [temp_file_path]
        if wav_path and wav_path != temp_file_path:
            cleanup_files.append(wav_path)
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass

'''

# Keep all existing endpoints with streak fixes
@app.post('/api/submit-audio')
async def submit_audio(
    request: Request,
    audio: UploadFile = File(...),
    time_to_record: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    if not audio:
        return JSONResponse(content={"error": "No audio file provided"}, status_code=400)

    game_session = get_user_game_session(request, db)

    # Extract current figure info
    figure = {
        "name": game_session.current_figure_name or "",
        "description": game_session.current_figure_description or "",
        "hint": game_session.current_figure_hint or "",
        "difficulty": game_session.current_figure_difficulty or 0,
        "category": game_session.current_figure_category or "",
        "nationality": game_session.current_figure_nationality or "",
        "birth_year": game_session.current_figure_birth_year or 0,
        "pronunciation_tips": game_session.current_figure_pronunciation_tips or "",
    }

    if not figure["name"]:
        return JSONResponse(content={"error": "Game completed"}, status_code=400)

    try:
        contents = await audio.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        wav_path = None
        try:
            audio_segment = AudioSegment.from_file(temp_file_path)

            if audio_segment is None:
                raise ValueError("Failed to decode audio. File may be corrupted or unsupported.")

            duration_ms = len(audio_segment)
            if duration_ms < AUDIO_CONFIG["recording"]["min_duration_ms"]:
                return JSONResponse(content={"error": MESSAGES["errors"]["audio_too_short"]}, status_code=400)

            wav_path = temp_file_path.replace(".webm", ".wav")
            audio_segment.export(wav_path, format="wav")
            analysis_file_path = wav_path
 
        except Exception as e:
            logger.warning(f"Audio conversion failed, using original file instead. Reason: {e}")
            analysis_file_path = temp_file_path

        # Run pronunciation analysis
        analysis = pronunciation_analyzer.analyze_pronunciation(analysis_file_path, figure)

        if time_to_record is not None:
            analysis["time_to_record"] = time_to_record

        # Build attempt record
        accuracy_score = analysis.get("accuracy_score", 0)
        attempt = {
            "figure": figure["name"],
            "transcription": analysis["raw_transcript"],
            "score": analysis["overall_score"],
            "feedback": analysis["feedback"],
            "accuracy": (
                "exact" if accuracy_score >= 90 else
                "close" if accuracy_score >= 70 else
                "partial" if accuracy_score >= 40 else
                "incorrect"
            ),
            "highlights": analysis["highlights"],
            "timestamp": datetime.now().isoformat(),
            "analysis_details": analysis,
            "time_to_record": time_to_record
        }

        game_session.attempts.append(attempt)
        game_session.score = (game_session.score or 0) + analysis["overall_score"]

        # Update streaks
        is_correct = accuracy_score > 0
        game_session.streak = (game_session.streak or 0) + 1 if is_correct else 0
        game_session.max_streak = max(game_session.max_streak or 0, game_session.streak)

        if not is_correct and game_session.streak == 0:
            logger.info(f"💔 Streak broken. Max streak remains {game_session.max_streak}")

        # Prepare response
        response_data = {
            "transcription": analysis["raw_transcript"],
            "analysis": analysis,
            "figure_name": figure["name"],
            "total_score": game_session.score,
            "streak": game_session.streak,
            "max_streak": game_session.max_streak,
            "attempt_number": len(game_session.attempts),
        }

        # Advance figure index
        game_session.current_figure_index = (game_session.current_figure_index or 0) + 1

        # Final stats if complete
        if game_session.is_complete:
            duration = (datetime.now() - game_session.start_time).total_seconds()
            avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0
            response_data["final_stats"] = {
                "total_score": round(game_session.score, 1),
                "average_score": round(avg_score, 1),
                "duration_seconds": round(duration),
                "attempts": len(game_session.attempts),
                "max_streak": game_session.max_streak,
                "current_streak": game_session.streak,
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"]),
                }
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(content={"error": f"Failed to process audio: {str(e)}"}, status_code=500)

    finally:
        # Clean up temporary files
        for file_path in [temp_file_path, wav_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass

@app.post('/api/complete-game')
def complete_game(
    request: Request,
    db: Session = Depends(get_db)
):
    game_session = get_user_game_session(request, db)
    if not game_session.is_complete:
        return JSONResponse(content={"error": "Game not complete"}), 400
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
    return JSONResponse(content={
        "game_complete": True,
        "final_stats": final_stats
    })


@app.post('/api/timeout')
async def handle_timeout(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        game_session = get_user_game_session(request, db)

        # Validate session exists
        if not game_session:
            return JSONResponse(content={"error": "No active session"}, status_code=400)

        # Validate current figure
        logger.debug(f"Getting current figure: index={game_session.current_figure_index}, total={len(game_session.figures_order or [])}")
        figure = get_current_figure(game_session)

        if not figure:
            return JSONResponse(content={"error": "Game completed or invalid state"}, status_code=400)

        # Ensure 'attempts' is initialized
        if game_session.attempts is None:
            game_session.attempts = []

        attempt = {
            "figure": figure.get("name", "Unknown"),
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

        # Defensive call to update streak
        if hasattr(game_session, "update_streak"):
            game_session.update_streak(False)

        is_final_figure = (
            game_session.current_figure_index + 1 >= len(game_session.figures_order or [])
        )

        response_data = {
            "transcription": "⏰ Time expired",
            "analysis": attempt["analysis_details"],
            "figure_name": figure.get("name", "Unknown"),
            "total_score": game_session.score,
            "streak": game_session.streak,
            "max_streak": game_session.max_streak,
            "attempt_number": len(game_session.attempts),
            "timeout": True,
            "is_final_figure": is_final_figure
        }

        # Advance to next figure
        if hasattr(game_session, "advance_figure"):
            advance_figure(game_session)

        # Final stats if game is complete
        if hasattr(game_session, "is_complete") and game_session.is_complete:
            game_duration = (datetime.now() - game_session.start_time).total_seconds()
            avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0
            response_data["final_stats"] = {
                "total_score": round(game_session.score, 1),
                "average_score": round(avg_score, 1),
                "duration_seconds": round(game_duration),
                "attempts": len(game_session.attempts),
                "max_streak": game_session.max_streak,
                "current_streak": game_session.streak,
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"]),
                    "timeout": len([a for a in game_session.attempts if a["accuracy"] == "timeout"])
                }
            }

        db.commit()
        db.refresh(game_session)
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error handling timeout: {e}", exc_info=True)
        db.rollback()
        return JSONResponse(content={"error": f"Failed to handle timeout: {str(e)}"}, status_code=500)



@app.post('/api/restart-game')
async def restart_game(request: Request, db: Session = Depends(get_db)):
    try:
        # Parse JSON body
        try:
            data = await request.json()
        except Exception:
            data = {}

        mode = data.get('mode', 'classic')
        user_id = get_user_id(request)

        logger.info(f"🔧 RESTART DEBUG: User {user_id[:8]}... requesting restart with mode {mode}")

        # ✅ Now pass `db` to reset_user_game_session
        game_session = reset_user_game_session(request, db, mode)

        current_figure = get_current_figure(game_session)

        logger.info(f"🔧 RESTART DEBUG: Current figure exists: {current_figure is not None}")

        return JSONResponse(content={
            "message": "Game restarted successfully",
            "mode": mode,
            "success": True,
            "debug_info": {
                "user_id": user_id[:8] + '...',
                "session_complete": game_session.is_complete,
                "current_figure_index": game_session.current_figure_index,
                "total_figures": len(game_session.figures_order or []),
                "has_current_figure": current_figure is not None
            }
        })

    except Exception as e:
        logger.error(f"🔧 RESTART DEBUG: Error during restart: {e}")
        return JSONResponse(
            content={"error": f"Failed to restart game: {str(e)}", "success": False},
            status_code=500
        )


@app.post('/api/get-hint')
async def get_hint(request: Request, db: Session = Depends(get_db)):
    try:
        # Get the user's game session
        game_session = get_user_game_session(request, db)
        figure = get_current_figure(game_session)

        if not figure:
            return JSONResponse(content={"error": "No active figure"}, status_code=400)

        # Defensive fallback for mode config
        mode_config = get_game_mode_config(game_session.mode) or {}
        hints_allowed = mode_config.get("hints_allowed", 3)

        # Enforce hint limit
        if game_session.hints_used >= hints_allowed:
            return JSONResponse(content={"error": "No more hints allowed in this mode"}, status_code=400)

        # Safely extract figure fields
        keywords = figure.get('keywords') or []
        if not isinstance(keywords, list):
            keywords = []

        phonetic_variations = figure.get('phonetic_variations') or []
        if not isinstance(phonetic_variations, list) or not phonetic_variations:
            phonetic_variations = [figure.get('name', 'Unknown')]

        birth_year = figure.get('birth_year')
        if not isinstance(birth_year, int):
            birth_year = 0

        # Build hints list
        hints = [
            figure.get("hint", "No hint available"),
            f"Category: {figure.get('category', 'Historical Figure')}",
            f"Keywords: {', '.join(keywords[:3]) or 'None'}",
            f"Born in: {abs(birth_year)}{' BC' if birth_year < 0 else ''}",
            f"Nationality: {figure.get('nationality', 'Unknown')}",
            f"Try saying: {phonetic_variations[0]}",
            f"Full description: {figure.get('description', 'No description available')}"
        ]

        # Determine current hint to return
        hint_index = min(game_session.hints_used, len(hints) - 1)
        game_session.hints_used += 1

        # Save session state
        db.commit()
        db.refresh(game_session)

        return JSONResponse(content={
            "hint": hints[hint_index],
            "hint_level": hint_index + 1,
            "hints_remaining": max(0, hints_allowed - game_session.hints_used)
        })

    except Exception as e:
        logger.error(f"Error getting hint: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to retrieve hint: {str(e)}"},
            status_code=500
        )

@app.get('/api/stats')
def get_stats(
    request: Request,
    db: Session = Depends(get_db)
):
    whisper_stats = pronunciation_analyzer.whisper.get_statistics()
    game_session = get_user_game_session(request, db)
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

    return JSONResponse(content={
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
@app.post('/api/conversation/start')
def start_conversation(
    request: Request,
    db: Session = Depends(get_db) 
):
    try:
        game_session = get_user_game_session(request,db)

        # ✅ FIXED: Auto-restart if game is complete
        auto_restarted = False
        if game_session.is_complete:
            logger.info("🔄 Game complete, auto-restarting for new conversation...")
            game_session = reset_user_game_session('classic')
            auto_restarted = True

        conversation_manager = get_user_conversation_manager()
        #figure = game_session.get_current_figure()
        figure = get_current_figure(game_session)
        if not figure:
            return JSONResponse(content={"error": "No current figure available"}), 400

        conversation_manager.start_conversation(figure)
        return JSONResponse(content={
            "message": "Conversation started",
            "figure_name": figure["name"],
            "conversation_state": conversation_manager.get_state().value,
            "speech_synthesis_ready": True,
            "auto_restarted": auto_restarted  # ✅ Tell frontend we restarted
        })
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        return JSONResponse(content={"error": f"Failed to start conversation: {str(e)}"}), 500


@app.get('/api/conversation/state')
def get_conversation_state(
    request: Request,
    db: Session = Depends(get_db) 
):
    try:
        conversation_manager = get_user_conversation_manager()
        game_session = get_user_game_session(request,db)
        state_info = {
            "conversation_state": conversation_manager.get_state().value,
            "is_listening": conversation_manager.is_listening(),
            "is_waiting_for_next": conversation_manager.is_waiting_for_next(),
            #"current_figure": game_session.get_current_figure(),
            "current_figure": get_current_figure(game_session),
            "game_progress": {
                "current": game_session.current_figure_index + 1,
                "total": len(game_session.figures_order)
            }
        }
        return JSONResponse(content=state_info)
    except Exception as e:
        logger.error(f"Error getting conversation state: {e}")
        return JSONResponse(content={"error": f"Failed to get conversation state: {str(e)}"}), 500


@app.post('/api/conversation/voice-command')
async def process_voice_command(request: Request):
    try:
        data = await request.json()
        command_text = data.get('command', '').strip()

        if not command_text:
            return JSONResponse(content={"error": "No command provided"}, status_code=400)

        conversation_manager = get_user_conversation_manager(request)  # ✅ Pass request if needed
        command_processed = conversation_manager.process_voice_command(command_text)

        return JSONResponse(content={
            "command": command_text,
            "processed": command_processed,
            "conversation_state": conversation_manager.get_state().value
        })

    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        return JSONResponse(
            content={"error": f"Failed to process voice command: {str(e)}"},
            status_code=500
        )
'''
@app.post('/api/conversation/voice-command-audio')
async def process_voice_command_audio(
    request: Request,
    audio: UploadFile = File(...)
):
    """Process voice commands from audio without affecting game state - SIMPLE VERSION"""
    temp_file_path = None

    try:
        logger.info("🎙️ VOICE COMMAND: Processing voice command audio")

        # Save the uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            contents = await audio.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        logger.info(f"💾 Saved voice command audio: {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")

        transcription = ""

        try:
            # Use current game session and figure for analysis context
            game_session = get_user_game_session(request)
            figure = game_session.get_current_figure()

            if not figure:
                figure = {
                    "name": "unknown",
                    "keywords": [],
                    "phonetic_variations": [],
                    "difficulty": 1,
                    "description": "voice command",
                    "hint": "voice command"
                }

            logger.info(f"🎯 Using figure for voice command: {figure['name']}")

            # Analyze pronunciation without conversion (assume .webm works)
            analysis = pronunciation_analyzer.analyze_pronunciation(temp_file_path, figure)
            transcription = analysis.get("raw_transcript", "").strip()

            logger.info(f"✅ Voice command transcribed: '{transcription}'")

        except Exception as analysis_error:
            logger.error(f"❌ Voice command analysis failed: {analysis_error}")
            transcription = ""

        return JSONResponse(content={
            "transcription": transcription,
            "command_mode": True,
            "message": "Voice command processed successfully"
        })

    except Exception as e:
        logger.error(f"Error in voice command processing: {e}")
        return JSONResponse(content={"error": f"Failed to process voice command: {str(e)}"}, status_code=500)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ Cleaned up voice command file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Could not delete temp file: {cleanup_error}")
'''

@app.post('/api/conversation/voice-command-audio')
async def process_voice_command_audio(
    request: Request,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db) 
):
    """Process voice commands from audio without affecting game state - SIMPLE VERSION"""
    temp_file_path = None

    try:
        logger.info("🎙️ VOICE COMMAND: Processing voice command audio")

        # Save the uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            contents = await audio.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        logger.info(f"💾 Saved voice command audio: {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")

        transcription = ""

        try:
            # Use current game session and figure for analysis context
            game_session = get_user_game_session(request,db)
            #figure = game_session.get_current_figure()
            figure = get_current_figure(game_session)

            if not figure:
                figure = {
                    "name": "unknown",
                    "keywords": [],
                    "phonetic_variations": [],
                    "difficulty": 1,
                    "description": "voice command",
                    "hint": "voice command"
                }

            logger.info(f"🎯 Using figure for voice command: {figure['name']}")

            # Analyze pronunciation without conversion (assume .webm works)
            analysis = pronunciation_analyzer.analyze_pronunciation(temp_file_path, figure)
            transcription = analysis.get("raw_transcript", "").strip()

            logger.info(f"✅ Voice command transcribed: '{transcription}'")

        except Exception as analysis_error:
            logger.error(f"❌ Voice command analysis failed: {analysis_error}")
            transcription = ""

        return JSONResponse(content={
            "transcription": transcription,
            "command_mode": True,
            "message": "Voice command processed successfully"
        })

    except Exception as e:
        logger.error(f"Error in voice command processing: {e}")
        return JSONResponse(content={"error": f"Failed to process voice command: {str(e)}"}, status_code=500)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ Cleaned up voice command file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Could not delete temp file: {cleanup_error}")

@app.get('/api/conversation/speech-synthesis')
def get_speech_synthesis():
    try:
        conversation_manager = get_user_conversation_manager()
        synthesizer = conversation_manager.synthesizer
        return JSONResponse(content={
            "has_pending_speech": synthesizer.is_speaking,
            "speech_queue_length": len(synthesizer.speech_queue),
            "conversation_state": conversation_manager.get_state().value
        })
    except Exception as e:
        logger.error(f"Error getting speech synthesis: {e}")
        return JSONResponse(content={"error": f"Failed to get speech synthesis: {str(e)}"}), 500


@app.get('/api/conversation/voice-activity-config')
def get_voice_activity_config():
    try:
        vad = VoiceActivityDetector(VADSensitivity.MEDIUM)
        config = vad.start_listening()
        return JSONResponse(content={
            "vad_config": config,
            "conversational_mode": True,
            "auto_recording": True
        })
    except Exception as e:
        logger.error(f"Error getting VAD config: {e}")
        return JSONResponse(content={"error": f"Failed to get VAD config: {str(e)}"}), 500


@app.post('/api/conversation/submit-answer')
async def submit_conversational_answer(
    request: Request,
    audio: UploadFile = File(...),
    time_to_record: float = Form(None),
    db: Session = Depends(get_db)
):
    logger.info("🎙️ CONVERSATIONAL: Received audio submission")
    temp_file_path = None
    wav_path = None

    try:
        game_session = get_user_game_session(request, db)
        conversation_manager = get_user_conversation_manager(request)
        figure = get_current_figure(game_session)

        if not figure:
            return JSONResponse(content={"error": "Game completed"}, status_code=400)

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            contents = await audio.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Audio conversion
        try:
            audio_segment = AudioSegment.from_file(temp_file_path)
            if len(audio_segment) < AUDIO_CONFIG["recording"]["min_duration_ms"]:
                return JSONResponse(content={"error": MESSAGES["errors"]["audio_too_short"]}, status_code=400)
            wav_path = temp_file_path.replace(".webm", ".wav")
            audio_segment.export(wav_path, format="wav")
            analysis_file_path = wav_path
        except Exception as e:
            logger.info(f"Audio conversion skipped, using original file: {e}")
            analysis_file_path = temp_file_path

        # Analyze pronunciation
        analysis = pronunciation_analyzer.analyze_pronunciation(analysis_file_path, figure)
        if time_to_record is not None:
            analysis["time_to_record"] = time_to_record

        # ✅ Normalize analysis to avoid np.float64 issues
        analysis = normalize_numpy(analysis)

        # Build attempt record (also normalizing score)
        attempt = {
            "figure": figure["name"],
            "transcription": analysis["raw_transcript"],
            "score": float(analysis["overall_score"]),
            "feedback": analysis["feedback"],
            "accuracy": (
                "exact" if analysis["accuracy_score"] >= 90 else
                "close" if analysis["accuracy_score"] >= 70 else
                "partial" if analysis["accuracy_score"] >= 40 else
                "incorrect"
            ),
            "highlights": analysis["highlights"],
            "timestamp": datetime.now().isoformat(),
            "analysis_details": analysis,
            "time_to_record": time_to_record
        }

        # Update session state
        game_session.attempts.append(attempt)
        game_session.score += float(analysis["overall_score"])
        is_correct = analysis["accuracy_score"] > 0

        # ✅ Update streak
        update_streak(game_session, is_correct)

        is_final_figure = (game_session.current_figure_index + 1) >= len(game_session.figures_order)

        response_data = {
            "transcription": analysis["raw_transcript"],
            "analysis": analysis,
            "figure_name": figure["name"],
            "total_score": game_session.score,
            "streak": game_session.streak,
            "max_streak": game_session.max_streak,
            "attempt_number": len(game_session.attempts),
            "is_final_figure": is_final_figure,
            "conversational_mode": True,
            "conversation_state": conversation_manager.get_state().value
        }

        # Advance figure
        advance_figure(game_session)

        # ✅ Commit DB-safe types only
        db.add(game_session)
        db.commit()

        if game_session.is_complete:
            game_duration = (datetime.now() - game_session.start_time).total_seconds()
            avg_score = game_session.score / len(game_session.attempts) if game_session.attempts else 0
            response_data["final_stats"] = {
                "total_score": round(game_session.score, 1),
                "average_score": round(avg_score, 1),
                "duration_seconds": round(game_duration),
                "attempts": len(game_session.attempts),
                "max_streak": game_session.max_streak,
                "current_streak": game_session.streak,
                "accuracy_breakdown": {
                    "exact": len([a for a in game_session.attempts if a["accuracy"] == "exact"]),
                    "close": len([a for a in game_session.attempts if a["accuracy"] == "close"]),
                    "partial": len([a for a in game_session.attempts if a["accuracy"] == "partial"]),
                    "incorrect": len([a for a in game_session.attempts if a["accuracy"] == "incorrect"])
                }
            }
            conversation_manager.handle_game_completion(response_data["final_stats"])

        def on_feedback_complete():
            logger.info("✅ Conversational feedback provided")

        conversation_manager.provide_feedback(analysis, on_feedback_complete)

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return JSONResponse(content={"error": f"Failed to process audio: {str(e)}"}, status_code=500)

    finally:
        for file_path in [temp_file_path, wav_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    logger.info(f"🗑️ Deleted temp file: {file_path}")
                except Exception:
                    pass



@app.post('/api/calibrate-vad')
async def calibrate_vad(request: Request):
    try:
        data = await request.json()
        ambient_levels = data.get('ambient_levels', [])

        if not ambient_levels:
            logger.warning('No ambient levels provided for VAD calibration')
            return JSONResponse(content={"error": "No ambient levels provided"}, status_code=400)

        vad = VoiceActivityDetector(VADSensitivity.MEDIUM)
        vad.calibrate_environment(ambient_levels)

        logger.info(
            f"VAD calibrated with baseline: {np.mean(ambient_levels):.3f}, "
            f"min: {np.min(ambient_levels):.3f}, max: {np.max(ambient_levels):.3f}"
        )

        return JSONResponse(content={"message": "VAD successfully calibrated"})

    except Exception as e:
        logger.error(f"Error calibrating VAD: {e}")
        return JSONResponse(
            content={"error": f"Failed to calibrate VAD: {str(e)}"},
            status_code=400
        )

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

'''
'''
import sys
import requests

API_BASE = "http://127.0.0.1"
PORTS = {
    "conversational": (8001, "conversational"),
    "game-state": (8002, "api/game-state"),
    "session-stats": (8003, "api/session-stats"),
    "session-details": (8004, "api/admin/session-details"),
    "cleanup-sessions": (8005, "api/admin/cleanup-sessions"),
    "save-sessions": (8006, "api/admin/save-sessions"),
    "submit-audio": (8007, "api/submit-audio"),
    "complete-game": (8008, "api/complete-game"),
    "timeout": (8009, "api/timeout"),
    "restart-game": (8010, "api/restart-game"),
    "get-hint": (8011, "api/get-hint"),
    "stats": (8012, "api/stats"),
    "conversation-stats": (8013, "api/conversation/stats"),
    "conversation-state": (8014, "api/conversation/state"),
    "voice-command": (8015, "api/conversation/voice-command"),
    "voice-command-audio": (8016, "api/conversation/voice-command-audio"),
    "speech-synthesis": (8017, "api/conversation/speech-synthesis"),
    "voice-activity-config": (8018, "api/conversation/voice-activity-config"),
    "submit-answer": (8019, "api/conversation/submit-answer"),
    "calibrate-vad": (8020, "api/calibrate-vad"),
}



def run_step(step, payload):
    port, path = PORTS[step]
    url = f"{API_BASE}:{port}/{path}"
    print(f"\n▶ Running {step} @ {url}")
    print(f"Payload: {payload}")

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        print(f"✅ {step} success: {resp.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ {step} failed: {e}")
        if e.response is not None:
            print("❌ Response content:", e.response.text)
        sys.exit(1)



def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <batch_id> <source_file>")
        sys.exit(1)

    batch_id = sys.argv[1]
    source_file = sys.argv[2]

    print(f"\n🚀 Starting pipeline for batch: {batch_id}, file: {source_file}\n")

    run_step("conversational", {"batch_id": batch_id})
    run_step("api/game-state", {"batch_id": batch_id})
    run_step("api/session-stats", {"batch_id": batch_id})
    run_step("api/admin/session-details", {"batch_id": batch_id})
    run_step("api/admin/cleanup-sessions", {"batch_id": batch_id})
    run_step("api/admin/save-sessions", {"batch_id": batch_id})
    run_step("api/submit-audio", {"batch_id": batch_id})
    run_step("api/complete-game", {"batch_id": batch_id})
    run_step("api/timeout", {"batch_id": batch_id})
    run_step("api/restart-game", {"batch_id": batch_id})
    run_step("api/get-hint", {"batch_id": batch_id})
    run_step("api/stats", {"batch_id": batch_id})
    run_step("api/conversation/stats", {"batch_id": batch_id})
    run_step("api/conversation/state", {"batch_id": batch_id})
    run_step("api/conversation/voice-command", {"batch_id": batch_id})
    run_step("api/conversation/voice-command-audio", {"batch_id": batch_id})
    run_step("api/conversation/speech-synthesis", {"batch_id": batch_id})
    run_step("api/conversation/voice-activity-config", {"batch_id": batch_id})
    run_step("api/conversation/submit-answer", {"batch_id": batch_id})
    run_step("api/calibrate-vad", {"batch_id": batch_id})

    print(f"\n✅ Pipeline completed successfully for batch: {batch_id}\n")

if __name__ == "__main__":
    main()