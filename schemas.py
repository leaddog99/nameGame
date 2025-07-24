from pydantic import BaseModel
from typing import List, Optional

class HistoricalFigureCreate(BaseModel):
    name: str
    description: str
    keywords: Optional[List[str]] = None
    difficulty: Optional[int] = None
    hint: Optional[str] = None
    phonetic_variations: Optional[List[str]] = None
    category: Optional[str] = None
    birth_year: Optional[int] = None
    nationality: Optional[str] = None
    famous_quote: Optional[str] = None
    pronunciation_tips: Optional[str] = None


class FigureState(BaseModel):
    name: str
    description: str
    hint: Optional[str] = None
    difficulty: Optional[int] = None
    category: Optional[str] = None
    nationality: Optional[str] = None
    birth_year: Optional[int] = None
    pronunciation_tips: Optional[str] = None


class ProgressState(BaseModel):
    current: int
    total: int


class GameStateResponse(BaseModel):
    current_figure: Optional[FigureState]
    progress: ProgressState
    score: int
    streak: int
    max_streak: int
    is_complete: bool
    mode: str
    hints_used: int