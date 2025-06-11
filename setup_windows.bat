@echo off
echo ================================================
echo   Historical Figures Game - Windows Setup
echo   Installing Local Whisper (NO API COSTS!)
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!

REM Create virtual environment (recommended)
echo.
echo 🔧 Creating virtual environment...
python -m venv whisper_env
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created!

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call whisper_env\Scripts\activate.bat

REM Upgrade pip
echo.
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo 🔧 Installing Python packages (this may take a few minutes)...
pip install Flask==2.3.3
pip install Flask-CORS==4.0.0
pip install Werkzeug==2.3.7
pip install openai-whisper==20231117
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pydub==0.25.1
pip install librosa==0.10.1
pip install numpy
pip install scipy

if errorlevel 1 (
    echo ❌ Failed to install packages
    pause
    exit /b 1
)

echo.
echo ✅ All packages installed successfully!

REM Download Whisper model
echo.
echo 🔧 Downloading Whisper 'base' model (this will take 1-2 minutes)...
python -c "import whisper; whisper.load_model('base')"

if errorlevel 1 (
    echo ❌ Failed to download Whisper model
    pause
    exit /b 1
)

echo.
echo 🎉 Setup complete!

echo.
echo ================================================
echo   NEXT STEPS:
echo ================================================
echo 1. Replace your app.py with the new Whisper version
echo 2. Run: whisper_env\Scripts\activate.bat
echo 3. Run: python app.py
echo 4. Run: ngrok http 5000 (in another terminal)
echo.
echo 💰 COST SAVINGS: $0 per user (vs $3.60 with Google API)!
echo ================================================

pause