from sqlalchemy import Column, Integer, String, Text, ARRAY, DateTime, Boolean, JSON
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableList
from typing import List, Dict
from db import Base  # import the Base from db.py

class HistoricalFigure(Base):
    __tablename__ = "historical_figures"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    keywords = Column(ARRAY(String))
    difficulty = Column(Integer)
    hint = Column(Text)
    phonetic_variations = Column(ARRAY(String))
    category = Column(String)
    birth_year = Column(Integer)
    nationality = Column(String)
    famous_quote = Column(Text)
    pronunciation_tips = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class GameSession(Base):
    __tablename__ = 'game_sessions'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)

    current_figure_name = Column(String)
    current_figure_description = Column(String)
    current_figure_hint = Column(String)
    current_figure_difficulty = Column(String)
    current_figure_category = Column(String, nullable=True)
    current_figure_nationality = Column(String, nullable=True)
    current_figure_birth_year = Column(Integer, nullable=True)
    current_figure_pronunciation_tips = Column(String, nullable=True)

    current_figure_index = Column(Integer, default=0)
    total_figures = Column(Integer, default=0)
    score = Column(Integer, default=0)
    streak = Column(Integer, default=0, nullable=False)
    max_streak = Column(Integer, default=0)
    is_complete = Column(Boolean, default=False)
    mode = Column(String, default='classic')
    hints_used = Column(Integer, default=0)

    # ✅ JSONB list for attempts with Mutable tracking
    attempts = Column(MutableList.as_mutable(JSONB), default=list, nullable=False, server_default='[]')

    figures_order = Column(ARRAY(Integer), nullable=True)
    selected_figures = Column(JSONB, nullable=True)

    start_time = Column(DateTime, default=datetime.now, nullable=False)
    last_accessed = Column(DateTime, default=datetime.now, nullable=False)


class DBConversation(Base):
    __tablename__ = "conversations"

    user_id = Column(String, primary_key=True, index=True)
    state = Column(String, default="initial")
    history = Column(JSON, default=[])
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))