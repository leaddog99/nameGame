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
        """Analyze pronunciation quality comprehensively"""
        transcript = whisper_result["transcript"].lower().strip()
        words = whisper_result["words"]
        overall_confidence = whisper_result["confidence"]

        analysis = {
            "overall_score": 0,
            "accuracy_score": 0,
            "clarity_score": 0,
            "speed_score": 0,
            "confidence_score": 0,
            "feedback": "",
            "word_breakdown": [],
            "highlights": [],
            "pronunciation_issues": [],
            "raw_transcript": whisper_result["transcript"],
            "processing_stats": {
                "confidence": overall_confidence,
                "word_count": len(words),
                "duration": words[-1]["end"] - words[0]["start"] if len(words) >= 2 else 0,
                "processing_time": whisper_result["processing_time"]
            }
        }

        accuracy = self._calculate_name_accuracy(transcript, target_figure)
        analysis["accuracy_score"] = accuracy

        clarity = overall_confidence * 100
        analysis["clarity_score"] = clarity
        analysis["confidence_score"] = clarity

        speed = self._calculate_speed_score(words, analysis.get("time_to_record"))
        analysis["speed_score"] = speed

        analysis["word_breakdown"] = self._analyze_individual_words(words, target_figure)
        analysis["overall_score"] = self._calculate_overall_score(accuracy, clarity, speed, target_figure["difficulty"])
        analysis["feedback"] = self._generate_feedback(analysis)
        analysis["highlights"] = self._generate_highlights(analysis)
        analysis["pronunciation_issues"] = self._identify_pronunciation_issues(analysis, target_figure)

        return analysis

    def _calculate_name_accuracy(self, transcript, target_figure):
        """FIXED: Proper accuracy scoring that prevents false full credit from partial matches"""
        logger.info(f"=== ACCURACY CALCULATION DEBUG ===")
        logger.info(f"Transcript: '{transcript}'")
        logger.info(f"Target Figure: '{target_figure['name']}'")

        transcript_clean = transcript.lower().strip()
        target_name = target_figure["name"].lower()

        if not transcript_clean or len(transcript_clean) < 2:
            logger.info(f"EMPTY/SHORT TRANSCRIPT: returning {SCORING_CONFIG['accuracy']['no_match']}")
            return SCORING_CONFIG["accuracy"]["no_match"]

        # Check for give up responses
        for give_up in MATCHING_CONFIG["give_up_phrases"]:
            if give_up in transcript_clean:
                logger.info(
                    f"GIVE UP PHRASE DETECTED: '{give_up}' - returning {SCORING_CONFIG['accuracy']['no_match']}")
                return SCORING_CONFIG["accuracy"]["no_match"]

        # Get target name parts
        target_words = target_name.split()
        target_first = target_words[0] if target_words else ""
        target_last = target_words[-1] if len(target_words) > 1 else ""
        is_single_name = len(target_words) == 1

        logger.info(f"Target words: first='{target_first}', last='{target_last}', single_name={is_single_name}")

        # FIXED: Check for EXACT full name matches - filter out partial variations
        target_variations = target_figure.get("phonetic_variations", [])

        # Only include full name variations for multi-word targets
        full_name_variations = []
        for variation in target_variations:
            variation_words = variation.lower().split()
            if is_single_name:
                # For single names, all variations are valid
                full_name_variations.append(variation.lower())
            else:
                # For multi-word names, only include multi-word variations
                if len(variation_words) > 1:
                    full_name_variations.append(variation.lower())

        all_correct_variations = [target_name] + full_name_variations
        logger.info(f"Checking against FULL NAME variations: {all_correct_variations}")

        for correct_variation in all_correct_variations:
            if correct_variation in transcript_clean:
                logger.info(
                    f"FULL NAME MATCH found: '{correct_variation}' - returning {SCORING_CONFIG['accuracy']['perfect_match']}")
                return SCORING_CONFIG["accuracy"]["perfect_match"]

        # Check individual name parts
        transcript_words = transcript_clean.split()
        first_name_found = False
        last_name_found = False

        logger.info(f"Transcript words: {transcript_words}")

        if is_single_name:
            # For single names, check all spoken words
            for transcript_word in transcript_words:
                similarity = self._calculate_word_similarity(target_first, transcript_word)
                logger.info(f"Single name check: '{target_first}' vs '{transcript_word}' = {similarity:.3f}")
                if similarity >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                    first_name_found = True
                    logger.info(f"SINGLE NAME MATCH FOUND!")
                    break
        else:
            # For multi-word names, check positional matching first
            if target_first and len(transcript_words) > 0:
                first_similarity = self._calculate_word_similarity(target_first, transcript_words[0])
                logger.info(
                    f"First name positional check: '{target_first}' vs '{transcript_words[0]}' = {first_similarity:.3f}")
                if first_similarity >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                    first_name_found = True
                    logger.info(f"FIRST NAME FOUND (positional)!")

            if target_last and len(transcript_words) > 1:
                last_similarity = self._calculate_word_similarity(target_last, transcript_words[-1])
                logger.info(
                    f"Last name positional check: '{target_last}' vs '{transcript_words[-1]}' = {last_similarity:.3f}")
                if last_similarity >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                    last_name_found = True
                    logger.info(f"LAST NAME FOUND (positional)!")

            # Check any word matches (for cases like "Billy Einstein")
            if not first_name_found:
                for word in transcript_words:
                    similarity = self._calculate_word_similarity(target_first, word)
                    logger.info(f"First name anywhere check: '{target_first}' vs '{word}' = {similarity:.3f}")
                    if similarity >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                        first_name_found = True
                        logger.info(f"FIRST NAME FOUND (anywhere)!")
                        break

            if not last_name_found and target_last:
                for word in transcript_words:
                    similarity = self._calculate_word_similarity(target_last, word)
                    logger.info(f"Last name anywhere check: '{target_last}' vs '{word}' = {similarity:.3f}")
                    if similarity >= MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                        last_name_found = True
                        logger.info(f"LAST NAME FOUND (anywhere)!")
                        break

            # Check single-word phonetic variations for partial credit
            if len(transcript_words) == 1:
                single_word = transcript_words[0]
                single_word_variations = [v.lower() for v in target_variations if len(v.split()) == 1]

                for variation in single_word_variations:
                    if self._calculate_word_similarity(variation, single_word) >= \
                            MATCHING_CONFIG["similarity_thresholds"]["sequence_matcher_min"]:
                        # Determine which name part this represents
                        if self._calculate_word_similarity(target_first, variation) >= 0.8:
                            first_name_found = True
                            logger.info(f"Single word variation '{variation}' matches FIRST name!")
                        elif target_last and self._calculate_word_similarity(target_last, variation) >= 0.8:
                            last_name_found = True
                            logger.info(f"Single word variation '{variation}' matches LAST name!")

        # Final scoring
        if is_single_name:
            result = SCORING_CONFIG["accuracy"]["perfect_match"] if first_name_found else SCORING_CONFIG["accuracy"][
                "no_match"]
            logger.info(f"SINGLE NAME RESULT: {result}")
            return result
        else:
            if first_name_found and last_name_found:
                logger.info(f"BOTH NAMES FOUND - returning {SCORING_CONFIG['accuracy']['perfect_match']}")
                return SCORING_CONFIG["accuracy"]["perfect_match"]
            elif first_name_found or last_name_found:
                logger.info(
                    f"ONE NAME FOUND: first={first_name_found}, last={last_name_found} - returning {SCORING_CONFIG['accuracy']['partial_match']}")
                return SCORING_CONFIG["accuracy"]["partial_match"]
            else:
                logger.info(f"NO NAMES FOUND - returning {SCORING_CONFIG['accuracy']['no_match']}")
                return SCORING_CONFIG["accuracy"]["no_match"]

    def _calculate_word_similarity(self, target_word, spoken_word):
        """Enhanced word similarity using jellyfish algorithms"""
        if target_word == spoken_word:
            return 1.0

        target_word = target_word.lower().strip()
        spoken_word = spoken_word.lower().strip()

        if target_word == spoken_word:
            return 1.0

        # Metaphone matching
        try:
            target_metaphone = jellyfish.metaphone(target_word)
            spoken_metaphone = jellyfish.metaphone(spoken_word)
            if target_metaphone == spoken_metaphone:
                return MATCHING_CONFIG["similarity_thresholds"]["metaphone_exact"]
        except:
            pass

        # Jaro-Winkler matching
        try:
            jaro_similarity = jellyfish.jaro_winkler_similarity(target_word, spoken_word)
            threshold = MATCHING_CONFIG["similarity_thresholds"]["jaro_winkler_good"]
            if jaro_similarity >= threshold:
                return jaro_similarity
        except:
            pass

        # Fallback to sequence matcher
        return SequenceMatcher(None, target_word, spoken_word).ratio()

    def _calculate_speed_score(self, words, time_to_record=None):
        """Calculate speed score based on response time and speaking pace"""
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

        # Calculate speaking pace
        if len(words) < 2:
            speaking_pace_score = 80
        else:
            total_duration = words[-1]["end"] - words[0]["start"]
            if total_duration <= 0:
                speaking_pace_score = 80
            else:
                words_per_second = len(words) / total_duration
                if 1.5 <= words_per_second <= 3.0:
                    speaking_pace_score = 100
                elif 1.0 <= words_per_second <= 4.0:
                    speaking_pace_score = 85
                elif 0.5 <= words_per_second < 1.0:
                    speaking_pace_score = 70
                else:
                    speaking_pace_score = 60

        return (response_speed_score * 0.7) + (speaking_pace_score * 0.3)

    def _analyze_individual_words(self, words, target_figure):
        """Analyze each word individually"""
        target_words = set(word.lower() for word in target_figure["name"].split())
        breakdown = []

        for word_info in words:
            word = word_info["word"].lower().strip()
            confidence = word_info["confidence"]

            is_target_word = any(
                target_word in word or word in target_word or
                SequenceMatcher(None, target_word, word).ratio() > 0.8
                for target_word in target_words
            )

            quality = "excellent" if confidence >= 0.9 else "good" if confidence >= 0.8 else "fair" if confidence >= 0.6 else "poor"

            breakdown.append({
                "word": word_info["word"],
                "confidence": confidence,
                "is_target_word": is_target_word,
                "quality": quality,
                "duration": word_info["end"] - word_info["start"],
                "timing": {"start": word_info["start"], "end": word_info["end"]}
            })

        return breakdown

    def _calculate_overall_score(self, accuracy, clarity, speed, difficulty):
        """Calculate overall score using configuration constants"""
        if accuracy <= 0:
            return 0

        base_score = accuracy

        # Apply penalties from configuration
        clarity_config = SCORING_CONFIG["penalties"]["clarity"]
        speed_config = SCORING_CONFIG["penalties"]["speed"]

        # Clarity penalty
        if clarity < clarity_config["poor_threshold"]:
            clarity_penalty = (clarity_config["poor_threshold"] - clarity) * clarity_config["poor_penalty_rate"]
            base_score = max(0, base_score - clarity_penalty)
        elif clarity < clarity_config["fair_threshold"]:
            clarity_penalty = (clarity_config["fair_threshold"] - clarity) * clarity_config["fair_penalty_rate"]
            base_score = max(0, base_score - clarity_penalty)

        # Speed penalty
        if speed < speed_config["poor_threshold"]:
            speed_penalty = (speed_config["poor_threshold"] - speed) * speed_config["poor_penalty_rate"]
            base_score = max(0, base_score - speed_penalty)
        elif speed < speed_config["fair_threshold"]:
            speed_penalty = (speed_config["fair_threshold"] - speed) * speed_config["fair_penalty_rate"]
            base_score = max(0, base_score - speed_penalty)

        # Apply difficulty bonus
        difficulty_modifier = SCORING_CONFIG["difficulty_multipliers"].get(difficulty, 1.0)
        final_score = base_score * difficulty_modifier

        return min(100, max(0, final_score))

    def _generate_feedback(self, analysis):
        """Generate feedback using configuration constants"""
        accuracy = analysis["accuracy_score"]
        overall = analysis["overall_score"]

        if accuracy <= 0:
            return MESSAGES["feedback"]["incorrect"]
        elif accuracy == SCORING_CONFIG["accuracy"]["partial_match"]:
            return MESSAGES["feedback"]["partial"]
        elif accuracy == SCORING_CONFIG["accuracy"]["perfect_match"]:
            if overall >= 95:
                return MESSAGES["feedback"]["excellent"]
            elif overall >= 85:
                return MESSAGES["feedback"]["great"]
            elif overall >= 75:
                return MESSAGES["feedback"]["good"]
            elif overall >= 65:
                return MESSAGES["feedback"]["fair"]
            else:
                return MESSAGES["feedback"]["needs_work"]
        else:
            return MESSAGES["feedback"]["needs_work"]

    def _generate_highlights(self, analysis):
        """Generate highlights based on performance"""
        highlights = []
        accuracy = analysis["accuracy_score"]
        clarity = analysis["clarity_score"]
        speed = analysis["speed_score"]

        if accuracy == SCORING_CONFIG["accuracy"]["perfect_match"]:
            highlights.append("Perfect name recognition! 🎯")
        elif accuracy == SCORING_CONFIG["accuracy"]["partial_match"]:
            highlights.append("Partial name recognition 🔶")

        if clarity >= 85:
            highlights.append("Excellent clarity 🔊")
        elif clarity >= 70:
            highlights.append("Good pronunciation clarity")

        if speed >= 90:
            highlights.append("Perfect response timing ⏱️")
        elif speed >= 80:
            highlights.append("Good response speed")

        if not highlights:
            highlights.append("Keep practicing! 💪")

        return highlights

    def _identify_pronunciation_issues(self, analysis, target_figure):
        """Identify specific areas for improvement"""
        issues = []

        poor_clarity_words = [w for w in analysis["word_breakdown"] if w["confidence"] < 0.6 and w["is_target_word"]]
        if poor_clarity_words:
            issues.append(f"Try pronouncing '{poor_clarity_words[0]['word']}' more clearly")

        if analysis["speed_score"] < 70:
            if analysis["processing_stats"]["duration"] > 0:
                wps = len(analysis["word_breakdown"]) / analysis["processing_stats"]["duration"]
                if wps < 1.0:
                    issues.append("Try speaking a bit faster")
                elif wps > 4.0:
                    issues.append("Try speaking a bit slower")

        if analysis["accuracy_score"] < 70:
            issues.append(f"Practice saying '{target_figure['name']}' clearly")

        return issues


# Export the main class
__all__ = ['PronunciationAnalyzer']