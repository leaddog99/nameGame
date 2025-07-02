"""
Pronunciation Analysis Module

Handles speech analysis, name matching, scoring, and feedback generation.
Uses sophisticated algorithms for fuzzy name matching and pronunciation scoring.
"""

import logging
from difflib import SequenceMatcher
import jellyfish
from audio_processor import LocalWhisperAnalyzer
from data_models import (
    SCORING_CONFIG,
    TIMING_CONFIG,
    MATCHING_CONFIG,
    MESSAGES
)

logger = logging.getLogger(__name__)


class PronunciationAnalyzer:
    def __init__(self):
        self.whisper = LocalWhisperAnalyzer()

    def analyze_pronunciation(self, audio_file_path, target_figure):
        """Complete pronunciation analysis using local Whisper"""
        whisper_result = self.whisper.transcribe_with_analysis(audio_file_path)
        analysis = self._analyze_pronunciation_quality(whisper_result, target_figure)
        return analysis

    def _analyze_pronunciation_quality(self, whisper_result, target_figure):
        transcript = whisper_result["transcript"].lower().strip()
        words = whisper_result["words"]
        overall_confidence = whisper_result["confidence"]

        accuracy, matched_name = self._calculate_name_accuracy(transcript, target_figure)

        analysis = {
            "overall_score": 0,
            "accuracy_score": accuracy,
            "clarity_score": overall_confidence * 100,
            "speed_score": self._calculate_speed_score(words),
            "confidence_score": overall_confidence * 100,
            "feedback": "",
            "word_breakdown": [],
            "highlights": [],
            "pronunciation_issues": [],
            "raw_transcript": whisper_result["transcript"],
            "matched_name": matched_name,
            "processing_stats": {
                "confidence": overall_confidence,
                "word_count": len(words),
                "duration": words[-1]["end"] - words[0]["start"] if len(words) >= 2 else 0,
                "processing_time": whisper_result["processing_time"]
            }
        }

        analysis["word_breakdown"] = self._analyze_individual_words(words, target_figure)
        analysis["overall_score"] = self._calculate_overall_score(analysis["accuracy_score"], analysis["clarity_score"], analysis["speed_score"], target_figure["difficulty"])
        analysis["feedback"] = self._generate_feedback(analysis)
        analysis["highlights"] = self._generate_highlights(analysis)
        analysis["pronunciation_issues"] = self._identify_pronunciation_issues(analysis, target_figure)

        return analysis

    def _calculate_name_accuracy(self, transcript, target_figure):
        transcript_clean = transcript.lower().strip()
        target_name = target_figure["name"].lower()

        if not transcript_clean or len(transcript_clean) < 2:
            return SCORING_CONFIG["accuracy"]["no_match"], None

        for give_up in MATCHING_CONFIG["give_up_phrases"]:
            if give_up in transcript_clean:
                return SCORING_CONFIG["accuracy"]["no_match"], None

        all_variations = [target_name] + target_figure.get("phonetic_variations", [])
        best_score = 0
        best_match = None

        for variation in all_variations:
            variation_clean = variation.strip().lower()
            if not variation_clean:
                continue
            score = self._calculate_word_similarity(transcript_clean, variation_clean)
            if score > best_score:
                best_score = score
                best_match = variation

        if best_score >= MATCHING_CONFIG["similarity_thresholds"]["metaphone_exact"]:
            return SCORING_CONFIG["accuracy"]["perfect_match"], best_match
        elif best_score >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
            return SCORING_CONFIG["accuracy"]["partial_match"], best_match
        else:
            return SCORING_CONFIG["accuracy"]["no_match"], None

    def _calculate_word_similarity(self, target, spoken):
        if target == spoken:
            return 1.0

        try:
            if jellyfish.metaphone(target) == jellyfish.metaphone(spoken):
                return MATCHING_CONFIG["similarity_thresholds"]["metaphone_exact"]
        except:
            pass

        try:
            jaro = jellyfish.jaro_winkler_similarity(target, spoken)
            if jaro >= MATCHING_CONFIG["similarity_thresholds"]["jaro_winkler_good"]:
                return jaro
        except:
            pass

        return SequenceMatcher(None, target, spoken).ratio()

    def _calculate_speed_score(self, words, time_to_record=None):
        timing = TIMING_CONFIG["speed_scoring"]
        if time_to_record is not None:
            if time_to_record <= timing["excellent_max"]:
                response_speed_score = 100
            elif time_to_record <= timing["good_max"]:
                response_speed_score = 85
            elif time_to_record <= timing["fair_max"]:
                response_speed_score = 70
            elif time_to_record <= timing["poor_max"]:
                response_speed_score = 50
            else:
                response_speed_score = 0
        else:
            response_speed_score = 80

        if len(words) < 2:
            speaking_pace_score = 80
        else:
            duration = words[-1]["end"] - words[0]["start"]
            if duration <= 0:
                speaking_pace_score = 80
            else:
                wps = len(words) / duration
                if 1.5 <= wps <= 3.0:
                    speaking_pace_score = 100
                elif 1.0 <= wps <= 4.0:
                    speaking_pace_score = 85
                elif 0.5 <= wps < 1.0:
                    speaking_pace_score = 70
                else:
                    speaking_pace_score = 60

        return 0.7 * response_speed_score + 0.3 * speaking_pace_score

    def _analyze_individual_words(self, words, target_figure):
        return []  # Placeholder

    def _calculate_overall_score(self, accuracy, clarity, speed, difficulty):
        base = (accuracy + clarity + speed) / 3
        multiplier = SCORING_CONFIG["difficulty_multipliers"].get(difficulty, 1.0)
        return base * multiplier

    def _generate_feedback(self, analysis):
        return "Analysis complete."

    def _generate_highlights(self, analysis):
        return []

    def _identify_pronunciation_issues(self, analysis, target_figure):
        return []
