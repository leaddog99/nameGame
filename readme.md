# Name Recognition Game

A web-based name recognition game that uses AI to intelligently score speech responses with interpolated scoring from 0-100 points.

## Quick Start

### 1. File Structure
Create this folder structure:
```
name-game/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

### 2. Installation

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**For Windows users who have trouble with PyAudio:**
```bash
pip install pipwin
pipwin install pyaudio
```

**For macOS users:**
```bash
brew install portaudio
pip install pyaudio
```

**For Linux users:**
```bash
sudo apt-get install python3-pyaudio
# or
sudo apt-get install portaudio19-dev python3-all-dev
pip install pyaudio
```

### 3. Running the Game

**Start the server:**
```bash
python app.py
```

**Open your browser:**
```
http://localhost:5000
```

## Features

### 🎯 Intelligent AI Scoring
- **0-100 point interpolation**: AI judges responses with nuanced scoring
- **Multiple factors considered**: Accuracy, confidence, timing, hesitation
- **Real-time analysis**: Instant feedback with detailed reasoning

### 🎤 Advanced Speech Recognition
- Browser-based audio recording
- Google Speech-to-Text integration
- Noise filtering and confidence scoring
- Support for various audio formats

### 🎮 Game Features
- 5 famous historical figures
- Progressive difficulty
- Cumulative scoring (max 500 points)
- Real-time feedback and reasoning
- Mobile-friendly responsive design

## Scoring System

The AI evaluates responses using these guidelines:

- **0 points**: No response or "I don't know"
- **1-25 points**: Attempted but unclear/wrong
- **26-40 points**: Some similarity to expected name
- **41-60 points**: Partial accuracy with hesitation
- **61-80 points**: Correct but with issues (mispronunciation, hesitation)
- **81-95 points**: Mostly correct with minor issues
- **96-100 points**: Perfect delivery

### Example Scores:
- "Abraham Lincoln" (clear) = **100 points**
- "Abraham... um... Lincoln?" (hesitant) = **72 points**
- "Abraham" (partial) = **33 points**
- "Abe Lincoln" (nickname) = **95 points**
- "Albert Lincoln" (confused) = **15 points**

## Customization

### Enable Real OpenAI Integration
1. Get an OpenAI API key from https://platform.openai.com
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. In `app.py`, uncomment line 21:
   ```python
   openai.api_key = os.getenv('OPENAI_API_KEY')
   ```
4. In the `analyze_response_with_llm` method, comment out the simulation line and uncomment the real LLM code.

### Add New Images
Edit the `game_images` list in `app.py`:
```python
{"id": 6, "image": "tesla.jpg", "expected_name": "Nikola Tesla"}
```

### Modify Scoring Criteria
Adjust the scoring prompt in the `analyze_response_with_llm` method to change how the AI evaluates responses.

## Troubleshooting

### Common Issues

**Microphone not working:**
- Check browser permissions (click the microphone icon in address bar)
- Ensure HTTPS (use localhost for development)
- Try a different browser

**PyAudio installation fails:**
- Windows: Use `pipwin install pyaudio`
- macOS: Install portaudio with Homebrew first
- Linux: Install system audio development packages

**Speech recognition errors:**
- Check internet connection (Google Speech API required)
- Speak clearly and avoid background noise
- Try recording again if recognition fails

**Server won't start:**
- Check if port 5000 is available
- Install all dependencies from requirements.txt
- Make sure you're in the correct directory

### Performance Tips

- **Use a good microphone** for better speech recognition
- **Speak clearly** without background noise
- **Quick responses** get timing bonuses
- **Full names** score higher than partial names

## Game Images

The demo uses these historical figures:
1. Abraham Lincoln
2. Albert Einstein  
3. George Washington
4. Marie Curie
5. Mahatma Gandhi

*Note: This is a demo version. For production, replace with actual images in a static folder.*

## Technical Architecture

- **Backend**: Flask web server with speech recognition
- **Frontend**: Vanilla JavaScript with Web Audio API
- **AI**: OpenAI GPT integration for intelligent scoring
- **Speech**: Google Speech-to-Text API
- **Audio**: Browser MediaRecorder API

## License

MIT License - Feel free to modify and use for your projects!

---

**Ready to play?** Run `python app.py` and visit http://localhost:5000!