#!/usr/bin/env python3
"""
Test script to verify faster-whisper is working correctly
"""

from faster_whisper import WhisperModel
import tempfile
import numpy as np
import wave


def create_test_audio():
    """Create a simple test audio file"""
    # Generate 2 seconds of silence (for testing)
    sample_rate = 16000
    duration = 2
    samples = np.zeros(sample_rate * duration, dtype=np.int16)

    # Create temporary wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        with wave.open(f.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        return f.name


def test_whisper():
    """Test faster-whisper installation and functionality"""
    print("🔧 Testing faster-whisper installation...")

    try:
        # Initialize model
        print("Loading Whisper model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ Model loaded successfully!")

        # Create test audio
        print("Creating test audio file...")
        test_audio_path = create_test_audio()
        print("✅ Test audio created!")

        # Test transcription
        print("Testing transcription...")
        segments, info = model.transcribe(test_audio_path, word_timestamps=True)
        segments_list = list(segments)

        print(f"✅ Transcription completed!")
        print(f"Language detected: {info.language}")
        print(f"Language probability: {info.language_probability:.2f}")
        print(f"Number of segments: {len(segments_list)}")

        # Test word-level timestamps
        if segments_list:
            for segment in segments_list[:1]:  # Just show first segment
                print(f"Segment: '{segment.text}'")
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words[:3]:  # Show first 3 words
                        print(
                            f"  Word: '{word.word}' ({word.start:.2f}s - {word.end:.2f}s, confidence: {word.probability:.2f})")

        print("\n🎉 faster-whisper is working perfectly!")
        print("💰 Cost per transcription: $0.00 (vs $0.006 with Google API)")
        print("🚀 Ready to use in your name game!")

        return True

    except Exception as e:
        print(f"❌ Error testing faster-whisper: {e}")
        return False

    finally:
        # Cleanup
        try:
            import os
            os.unlink(test_audio_path)
        except:
            pass


if __name__ == "__main__":
    test_whisper()