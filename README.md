# Scam Call Detection Backend

Real-time AI-powered scam call detection system using multi-modal analysis of audio features, speech patterns, and conversation context.

## Features

- **Real-time Audio Processing**: 4-second sliding window analysis
- **Multi-Model Fusion**: Combines 4 AI models for comprehensive scam detection
- **WebSocket API**: Live audio streaming from mobile clients
- **Voice Activity Detection**: Filters silence and non-speech segments
- **Conversation Stage Tracking**: Monitors call progression patterns

## Architecture

The system uses a 4-model ensemble approach:

1. **Phoneme CNN**: Analyzes speech phonetic patterns
2. **Urgency Detector**: Measures pitch, energy, and speech rate anomalies  
3. **Repetition CNN**: Detects repetitive keyword patterns
4. **Sentence Transformer**: Tracks conversation stages via intent classification

## Quick Start

### Prerequisites

- Python 3.12.6+
- Whisper.cpp (for speech transcription)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build Whisper.cpp
cd whisper.cpp
make

# Download Whisper model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin -P models/
```

### Running the Server

```bash
python newServer.py
```

Server starts on `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

### WebSocket Audio Stream
```
WS /ws/audio
```

Expects 4-second int16 PCM audio chunks (128KB at 16kHz).
Returns risk assessment: `✅ SAFE`, `🟡 LOW RISK`, `🟠 MODERATE`, `🔴 HIGH RISK`, `🚨 SCAM ALERT`

## Models

Place trained models in `models/`:
- `best_phoneme_model.keras` - Phoneme pattern classifier
- `best_repetition_model.keras` - Keyword repetition detector  
- `ggml-tiny.en.bin` - Whisper speech-to-text model

## Configuration

Key parameters in `config.py`:
- `SR = 16000` - Audio sample rate
- `WINDOW_SEC = 4` - Analysis window size
- `VAD_MODE = 2` - Voice activity detection sensitivity
- Model fusion weights (`W_STAGE`, `W_REP`, `W_PHONEME`, `W_URGENCY`)

## Risk Assessment

The system outputs a continuous risk score (0.0-1.0) mapped to 5 categories:
- **< 0.15**: Safe call
- **0.15-0.35**: Low risk  
- **0.35-0.55**: Moderate risk
- **0.55-0.80**: High risk
- **> 0.80**: Scam alert

## Development

### Project Structure
```
Backend/
├── models/              # Trained AI models
├── whisper.cpp/         # Speech transcription engine
├── newServer.py         # Main FastAPI server
├── config.py           # Configuration parameters
├── feature_extraction.py # Audio preprocessing
└── audio_pipeline.py   # Real-time processing pipeline
```

### Dependencies

Core libraries:
- FastAPI + Uvicorn (web server)
- TensorFlow/Keras (neural networks)
- Librosa (audio processing)
- Sentence Transformers (text embeddings)
- WebRTC VAD (voice activity detection)

## License

MIT License