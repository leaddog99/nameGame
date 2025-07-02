"""
Historical Figures Game - Data Models and Configuration

This module contains all the static data, configuration constants,
and data structures used by the Historical Figures Name Game.
"""

# Enhanced historical figures with comprehensive metadata
HISTORICAL_FIGURES = [
    {
        "name": "Jason Tatum",
        "description": "#0, a 6 foot 8 inch small forward for the Celtics, excelled in college at Duke University. He contributed to the teams success before being drafted by Boston in the 2017 NBA Draft",
        "keywords": ["first president", "revolutionary war", "founding father", "mount vernon"],
        "difficulty": 1,
        "hint": "First President and founding father of America",
        "phonetic_variations": ["JT", "Tatum", "Jason T", "Jason", ""],
        "category": "American Presidents",
        "birth_year": 1732,
        "nationality": "American",
        "famous_quote": "It is better to offer no excuse than a bad one",
        "pronunciation_tips": "AL HOR-furd"
    },
    {
        "name": "Abraham Lincoln",
        "description": "16th President of the United States, led the nation during Civil War",
        "keywords": ["president", "civil war", "emancipation", "gettysburg", "honest abe"],
        "difficulty": 1,
        "hint": "Known as 'Honest Abe', he freed the slaves",
        "phonetic_variations": ["abraham lincoln", "abe lincoln", "honest abe", "president lincoln", "lincoln"],
        "category": "American Presidents",
        "birth_year": 1809,
        "nationality": "American",
        "famous_quote": "Government of the people, by the people, for the people",
        "pronunciation_tips": "AY-bruh-ham LINK-un"
    },
    {
        "name": "Albert Einstein",
        "description": "Theoretical physicist, developed theory of relativity",
        "keywords": ["relativity", "physics", "scientist", "e=mc2", "nobel prize", "theory"],
        "difficulty": 2,
        "hint": "Famous physicist who said 'E=mc²'",
        "phonetic_variations": ["albert einstein", "einstein", "professor einstein"],
        "category": "Scientists",
        "birth_year": 1879,
        "nationality": "German",
        "famous_quote": "Imagination is more important than knowledge",
        "pronunciation_tips": "AL-bert EYE-n-stine"
    },
    {
        "name": "George Washington",
        "description": "First President of the United States, Revolutionary War leader",
        "keywords": ["first president", "revolutionary war", "founding father", "mount vernon"],
        "difficulty": 1,
        "hint": "First President and founding father of America",
        "phonetic_variations": ["george washington", "president washington", "general washington", "washington"],
        "category": "American Presidents",
        "birth_year": 1732,
        "nationality": "American",
        "famous_quote": "It is better to offer no excuse than a bad one",
        "pronunciation_tips": "JORJ WASH-ing-tun"
    },
    {
        "name": "Marie Curie",
        "description": "Physicist and chemist, first woman to win Nobel Prize",
        "keywords": ["radioactivity", "nobel prize", "scientist", "radium", "polonium"],
        "difficulty": 3,
        "hint": "First woman to win a Nobel Prize, studied radioactivity",
        "phonetic_variations": ["marie curie", "madame curie", "maria curie", "curie"],
        "category": "Scientists",
        "birth_year": 1867,
        "nationality": "Polish",
        "famous_quote": "Nothing in life is to be feared, it is only to be understood",
        "pronunciation_tips": "mah-REE kyoor-EE"
    },
    {
        "name": "Mahatma Gandhi",
        "description": "Leader of Indian independence movement, advocate of non-violence",
        "keywords": ["india", "independence", "non-violence", "civil rights", "salt march"],
        "difficulty": 2,
        "hint": "Led India's independence through non-violent resistance",
        "phonetic_variations": ["mahatma gandhi", "gandhi", "mohandas gandhi"],
        "category": "Political Leaders",
        "birth_year": 1869,
        "nationality": "Indian",
        "famous_quote": "Be the change you wish to see in the world",
        "pronunciation_tips": "mah-HAHT-mah GAHN-dee"
    },
    {
        "name": "Leonardo da Vinci",
        "description": "Renaissance artist, inventor, and polymath",
        "keywords": ["renaissance", "mona lisa", "inventor", "artist", "flying machine"],
        "difficulty": 3,
        "hint": "Renaissance genius who painted the Mona Lisa",
        "phonetic_variations": ["leonardo da vinci", "leonardo", "da vinci", "davinci"],
        "category": "Artists",
        "birth_year": 1452,
        "nationality": "Italian",
        "famous_quote": "Learning never exhausts the mind",
        "pronunciation_tips": "lee-oh-NAR-doh dah VIN-chee"
    },
    {
        "name": "Martin Luther King Jr.",
        "description": "Civil rights leader, advocate for racial equality",
        "keywords": ["civil rights", "dream speech", "equality", "montgomery", "nobel peace"],
        "difficulty": 2,
        "hint": "Civil rights leader famous for 'I Have a Dream' speech",
        "phonetic_variations": ["martin luther king", "martin luther king jr", "martin luther king junior",
                                "doctor king", "mlk"],
        "category": "Civil Rights Leaders",
        "birth_year": 1929,
        "nationality": "American",
        "famous_quote": "I have a dream that one day this nation will rise up",
        "pronunciation_tips": "MAR-tin LOO-ther KING"
    }
]

# Game modes and difficulty settings
GAME_MODES = {
    "classic": {
        "name": "Classic Mode",
        "description": "Play through all figures in random order",
        "figure_count": len(HISTORICAL_FIGURES),
        "time_limit": None,
        "hints_allowed": 3,
        "score_multiplier": 1.0
    },
    "quick": {
        "name": "Quick Game",
        "description": "5 random figures for a quick challenge",
        "figure_count": 5,
        "time_limit": None,
        "hints_allowed": 2,
        "score_multiplier": 1.2
    }
}

# Scoring configuration constants
SCORING_CONFIG = {
    "accuracy": {
        "perfect_match": 100,
        "partial_match": 33,
        "no_match": 0
    },
    "difficulty_multipliers": {
        1: 1.0,  # Easy figures
        2: 1.05,  # Medium figures
        3: 1.1  # Hard figures
    },
    "penalties": {
        "clarity": {
            "poor_threshold": 70,
            "fair_threshold": 85,
            "poor_penalty_rate": 0.3,
            "fair_penalty_rate": 0.1
        },
        "speed": {
            "poor_threshold": 70,
            "fair_threshold": 85,
            "poor_penalty_rate": 0.2,
            "fair_penalty_rate": 0.05
        }
    }
}

# Audio processing configuration
AUDIO_CONFIG = {
    "whisper": {
        "model_size": "base",
        "language": "en",  # Force English
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4
    },
    "recording": {
        "min_duration_ms": 500,  # Minimum recording length
        "max_file_size_mb": 16,  # Maximum upload size
        "supported_formats": ["webm", "wav", "mp4"]
    }
}

# Game timing configuration
TIMING_CONFIG = {
    "card_timeout_seconds": 20,
    "timer_warning_seconds": 10,
    "timer_critical_seconds": 5,
    "speed_scoring": {
        "excellent_max": 5,  # Under 5 seconds = excellent
        "good_max": 10,  # Under 10 seconds = good
        "fair_max": 15,  # Under 15 seconds = fair
        "poor_max": 20  # Under 20 seconds = poor
    }
}

# Text matching configuration
MATCHING_CONFIG = {
    "similarity_thresholds": {
        "metaphone_exact": 0.9,  # Phonetic exact match
        "jaro_winkler_good": 0.75,  # Good similarity match
        "sequence_matcher_min": 0.7  # Minimum acceptable similarity
    },
    "give_up_phrases": [
        "i don't know", "don't know", "no idea", "no clue", "not sure",
        "beats me", "i give up", "give up", "next", "skip", "pass",
        "that'll trump", "trump"
    ]
}

# Achievement definitions
ACHIEVEMENTS = {
    "streak_master": {
        "name": "🔥 Streak Master",
        "description": "Got 5 figures correct in a row",
        "threshold": 5,
        "type": "streak"
    },
    "perfect_round": {
        "name": "🎯 Perfect Round",
        "description": "Got 100% accuracy on a figure",
        "threshold": 100,
        "type": "score"
    },
    "speed_demon": {
        "name": "⚡ Speed Demon",
        "description": "Answered in under 3 seconds",
        "threshold": 3,
        "type": "speed"
    },
    "scholar": {
        "name": "🎓 Scholar",
        "description": "Completed all difficulty 3 figures",
        "threshold": 3,
        "type": "difficulty"
    }
}

# Error messages and user feedback
MESSAGES = {
    "errors": {
        "audio_too_short": "Audio recording too short - please speak for at least 1 second",
        "microphone_access": "Could not access microphone. Please check permissions.",
        "processing_failed": "Failed to process audio",
        "game_complete": "Game completed",
        "no_active_figure": "No active figure",
        "server_error": "Server error occurred"
    },
    "feedback": {
        "timeout": "Time's up! Try to respond more quickly next time.",
        "excellent": "🎉 Perfect! Excellent pronunciation and recognition!",
        "great": "🌟 Great job! Correct answer with clear pronunciation.",
        "good": "👏 Correct! Good pronunciation with room for minor improvement.",
        "fair": "👍 Right answer! Try to speak a bit more clearly next time.",
        "needs_work": "✅ Correct, but work on your clarity and speaking pace.",
        "incorrect": "❌ Incorrect answer. Listen carefully to the description and try again!",
        "partial": "🔶 Partial credit! You got part of the name right, but work on clarity and speed."
    }
}


# Utility functions for data access
def get_figure_by_name(name):
    """Get a historical figure by name"""
    for figure in HISTORICAL_FIGURES:
        if figure["name"].lower() == name.lower():
            return figure
    return None


def get_figures_by_difficulty(difficulty):
    """Get all figures of a specific difficulty"""
    return [f for f in HISTORICAL_FIGURES if f["difficulty"] == difficulty]


def get_figures_by_category(category):
    """Get all figures in a specific category"""
    return [f for f in HISTORICAL_FIGURES if f.get("category") == category]


def get_game_mode_config(mode_name):
    """Get configuration for a specific game mode"""
    return GAME_MODES.get(mode_name, GAME_MODES["classic"])


def validate_figure_data():
    """Validate that all figures have required fields"""
    required_fields = ["name", "description", "difficulty", "hint", "phonetic_variations"]

    for i, figure in enumerate(HISTORICAL_FIGURES):
        for field in required_fields:
            if field not in figure:
                raise ValueError(f"Figure {i} missing required field: {field}")

        if not isinstance(figure["phonetic_variations"], list):
            raise ValueError(f"Figure {i} phonetic_variations must be a list")

        if figure["difficulty"] not in [1, 2, 3]:
            raise ValueError(f"Figure {i} difficulty must be 1, 2, or 3")

    return True


# Validate data on import
if __name__ == "__main__":
    validate_figure_data()
    print(f"✅ Data validation passed!")
    print(f"📊 Total figures: {len(HISTORICAL_FIGURES)}")
    print(f"🎮 Game modes: {len(GAME_MODES)}")
    print(f"🏆 Achievements: {len(ACHIEVEMENTS)}")