"""
FastAPI Scam Detection Server
==============================
- Android sends one 4-second int16 PCM chunk per WebSocket message
- Runs all 4 models in parallel via ThreadPoolExecutor
- Per-connection SessionState (running risk, stage progression)
- Returns risk_label string back to Android
"""

import os
import json
import asyncio
import warnings
import tempfile
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import librosa
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

phoneme_model = None
repetition_model = None
embedder = None
intent_embeddings = None

executor = ThreadPoolExecutor(max_workers=4)

warnings.filterwarnings("ignore")

from config import SR, MIN_SPEECH_S
from audio_pipeline import vad_filter, dominant_speaker_filter
from feature_extraction import extract_mfcc
from repetition_preprocessing import extract_mfcc_raw

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXPECTED_SAMPLES      = 4 * SR        # exactly what Android should send

PHONEME_MODEL_PATH    = "models/best_phoneme_model.keras"
MAX_LEN               = 370

REPETITION_MODEL_PATH = "models/best_repetition_model.keras"
MAX_PHRASE_LEN        = 470
SLICE_SEC             = 1
REP_THRESHOLD         = 0.6

WINDOW_SEC      = 4          # seconds captured per window
WHISPER_CLI = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL   = "tiny"     # tiny / base / small / medium
WHISPER_LANG    = "en"     # auto-detect, or "en", "hi"
MODEL_PATH = r"models/ggml-tiny.en.bin"
SIMILARITY_GATE = 0.55       # minimum cosine similarity to count as a stage match

URGENCY_MAP = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.9}

W_STAGE   = 0.60
W_REP     = 0.20
W_PHONEME = 0.10
W_URGENCY = 0.10

ALPHA     = 0.7
BETA      = 0.3
THRESHOLD = 0.45

BASELINE = {
    "pitch_mean": 372.61,   "pitch_std": 0.0,
    "energy_mean": 3.23e-6, "energy_var": 1.33e-9,
    "spectral_centroid": 49257.50, "speech_rate": 12.5
}

INTENTS = {

"greeting": [
"hello sir how are you",
"hello maam how are you",
"good morning sir",
"good afternoon maam",
"good evening sir",
"am I speaking with the account holder",
"is this the owner of this number",
"am I speaking with Arya",
"can you hear me clearly",
"this is a service call regarding your account",
"this call is for account verification",
"namaste sir",
"namaste maam",
"hello this is customer service",
"hello I hope you are doing well",
"hello I am calling regarding your bank account",
"hello this call is about your bank services",
"hello we are calling from bank support",
"hello this is an official call",
"hello this is customer care calling",

],

"authority": [
"I am calling from SBI bank",
"I am calling from HDFC bank",
"I am calling from ICICI bank",
"I am calling from Axis bank",
"I am calling from bank verification team",
"I am calling from card security department",
"I am calling from ATM department",
"I am calling from KYC verification department",
"I am calling from bank security team",
"I am calling from banking support",
"This is the bank verification team",
"This is your bank customer care",
"This is bank technical department",
"This is UPI support team",
"I am calling from payment verification team",
"I am calling from banking headquarters",
"This call is from official bank department",
"This is your bank security desk",
"This is financial services department",
"This is your registered bank support",
"main bank se bol raha hoon",
"main SBI bank se bol raha hoon",
"main bank verification department se bol raha hoon",
"main customer support se bol raha hoon",
"main bank security team se bol raha hoon",
"ye bank verification call hai",
"main card department se bol raha hoon"
],

"problem": [
"your account has suspicious activity",
"we detected unusual transactions",
"your ATM card has been temporarily blocked",
"your bank account is under review",
"your card has been restricted",
"someone attempted login to your account",
"your account has security issues",
"your account verification failed",
"your KYC is incomplete",
"your KYC update is pending",
"your account is temporarily suspended",
"your debit card is flagged",
"your account has been locked",
"your account needs re verification",
"your bank services may stop",
"your account security is compromised",
"someone tried accessing your banking account",
"your account has abnormal activity",
"your account is under investigation",
"your card access is limited",
"aapka account block ho sakta hai",
"aapka ATM card block ho gaya hai",
"aapka KYC pending hai",
"aapke account mein suspicious activity hai",
"aapka bank account verify nahi hua",
"aapka account temporarily block ho gaya hai",
"aapka account security issue mein hai"
],

"urgency": [
"you must verify immediately",
"this is very urgent",
"immediate action is required",
"please verify now",
"your account will be blocked today",
"your account will be permanently blocked",
"this needs to be done right now",
"you need to confirm immediately",
"please do this quickly",
"we need to complete verification now",
"your banking service will stop",
"your card will be blocked today",
"your account will be frozen",
"this is time sensitive",
"verification must be done now",
"please respond quickly",
"this process must be completed immediately",
"this cannot be delayed",
"we must finish verification now",
"your service may stop today",
"abhi verify karna hoga",
"turant verify karna padega",
"abhi confirm karna hoga",
"warna account block ho jayega",
"jaldi karna hoga",
"abhi process complete karna hoga"
],

"data_request": [
"tell me the OTP sent to your phone",
"share the OTP you received",
"read the verification code",
"tell me the six digit code",
"read the SMS you received",
"confirm the number sent to your phone",
"share the verification code with me",
"tell me the security code",
"read the digits sent to your mobile",
"confirm the code sent to your number",
"tell me the message you received",
"share the OTP quickly",
"tell me the code for verification",
"confirm the SMS code",
"read the message from the bank",
"tell me the numbers you received",
"share the code sent to your phone",
"confirm the OTP message",
"tell me the verification number",
"read the code you just received",
"OTP bata dijiye",
"OTP bol dijiye",
"jo OTP aya hai wo bataiye",
"SMS mein jo code aya hai wo boliye",
"verification code bata dijiye",
"mobile pe jo code aya hai wo boliye",
"jo number aya hai wo bataiye",
"OTP share kar dijiye",
"message mein jo code hai wo boliye",
"OTP confirm kar dijiye"
]

}

STAGE_MAP    = {"greeting":1, "authority":2, "problem":3, "urgency":4, "data_request":5}
STAGE_RISK   = {0:0.05, 1:0.10, 2:0.25, 3:0.50, 4:0.70, 5:0.95}
STAGE_LABELS = {0:"none", 1:"GREETING", 2:"AUTHORITY", 3:"PROBLEM", 4:"URGENCY", 5:"DATA_REQUEST"}

# ─────────────────────────────────────────────
# LOAD MODELS ONCE AT STARTUP
# ─────────────────────────────────────────────
#print("Loading models...")
# phoneme_model    = load_model(PHONEME_MODEL_PATH)
# repetition_model = load_model(REPETITION_MODEL_PATH)
# embedder         = SentenceTransformer("all-MiniLM-L6-v2")
# intent_embeddings = {
#     intent: embedder.encode(sentences)
#     for intent, sentences in INTENTS.items()
# }
# executor = ThreadPoolExecutor(max_workers=4)
# print("All models loaded.\n")
@app.on_event("startup")
def load_models():

    global phoneme_model, repetition_model, embedder, intent_embeddings

    print("Loading models...")

    phoneme_model = load_model(PHONEME_MODEL_PATH)
    repetition_model = load_model(REPETITION_MODEL_PATH)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    intent_embeddings = {
        intent: embedder.encode(sentences)
        for intent, sentences in INTENTS.items()
    }

    print("All models loaded.")

app = FastAPI()


# ─────────────────────────────────────────────
# SESSION STATE — one instance per connection
# ─────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.current_stage = 0
        self.stage_history = []
        self.running_risk  = 0.0
        self.keyword_hits  = 0
        self.processed     = 0

    def update_running_risk(self, raw_risk: float) -> float:
        self.running_risk = ALPHA * self.running_risk + BETA * raw_risk
        return self.running_risk


# ─────────────────────────────────────────────
# MODEL 1 — PHONEME CNN
# ─────────────────────────────────────────────
def run_phoneme(dominant: np.ndarray) -> float:
    mfcc = extract_mfcc(dominant)
    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.vstack((mfcc, np.zeros((MAX_LEN - mfcc.shape[0], mfcc.shape[1]))))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    return float(phoneme_model.predict(np.expand_dims(mfcc, 0), verbose=0)[0][0])


# ─────────────────────────────────────────────
# MODEL 2 — URGENCY DETECTOR
# ─────────────────────────────────────────────
def run_urgency(audio: np.ndarray) -> tuple:
    pitches, magnitudes = librosa.piptrack(y=audio, sr=SR)
    pitch_vals = [
        pitches[magnitudes[:, i].argmax(), i]
        for i in range(pitches.shape[1])
        if pitches[magnitudes[:, i].argmax(), i] > 0
    ]
    pitch_vals = np.array(pitch_vals)

    f = {
        "pitch_mean":        float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0,
        "pitch_std":         float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0.0,
        "energy_mean":       float(np.mean(librosa.feature.rms(y=audio)[0])),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR))),
        "speech_rate":       float(len(librosa.util.peak_pick(
            librosa.onset.onset_strength(y=audio, sr=SR),
            pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=5
        )) / (len(audio) / SR + 1e-8)),
    }

    if f["energy_mean"] < 0.005:
        return 0, 0.0
    if BASELINE["energy_mean"] > 0.001 and f["energy_mean"] < BASELINE["energy_mean"] * 1.3:
        return 0, 0.0

    score = sum([
        f["energy_mean"]       > max(BASELINE["energy_mean"] * 1.8, 0.002),
        f["pitch_mean"]        > BASELINE["pitch_mean"]        * 1.3,
        f["pitch_std"]         > BASELINE["pitch_std"]         * 1.5,
        f["speech_rate"]       > BASELINE["speech_rate"]       * 1.5,
        f["spectral_centroid"] > BASELINE["spectral_centroid"] * 1.25,
    ])
    score = min(score, 3)
    return score, URGENCY_MAP.get(score, 0.0)


# ─────────────────────────────────────────────
# MODEL 3 — REPETITION CNN
# ─────────────────────────────────────────────
def run_repetition(speech: np.ndarray) -> float:
    slice_samples = int(SLICE_SEC * SR)
    probs = []
    for start in range(0, len(speech), slice_samples):
        chunk = speech[start: start + slice_samples]
        if len(chunk) < SR * 0.3:
            continue
        mfcc = extract_mfcc_raw(chunk)
        T = mfcc.shape[0]
        if T < MAX_PHRASE_LEN:
            mfcc = np.vstack((mfcc, np.zeros((MAX_PHRASE_LEN - T, mfcc.shape[1]), dtype=np.float32)))
        else:
            mfcc = mfcc[:MAX_PHRASE_LEN, :]
        probs.append(float(repetition_model.predict(np.expand_dims(mfcc, 0), verbose=0)[0][0]))
    return float(np.max(probs)) if probs else 0.0


# ─────────────────────────────────────────────
# MODEL 4 — SENTENCE TRANSFORMER (via Whisper)
# ─────────────────────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        wav.write(tmp_path, SR, (audio * 32767).astype(np.int16))
        result = subprocess.run(
            [WHISPER_CLI, "-m", MODEL_PATH, "-f", tmp_path,
             "-l", WHISPER_LANG, "--no-timestamps"],
            capture_output=True, text=True, timeout=30,
        )
        transcript = result.stdout.strip()
        for artifact in ["[BLANK_AUDIO]", "(music)", "[Music]", "(Music)"]:
            transcript = transcript.replace(artifact, "").strip()
        return transcript
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def run_stage(audio: np.ndarray, state: SessionState) -> tuple:
    transcript = transcribe(audio)
    if not transcript:
        return STAGE_RISK[state.current_stage], state.current_stage, "none", 0.0, ""

    text_emb = embedder.encode([transcript.lower().strip()])
    scores   = {
        intent: float(np.max(cosine_similarity(text_emb, embs)))
        for intent, embs in intent_embeddings.items()
    }
    best_intent = max(scores.keys(), key=lambda k: scores[k])
    best_score  = scores[best_intent]

    if best_score >= SIMILARITY_GATE:
        stage = STAGE_MAP[best_intent]
        if stage > state.current_stage:
            state.current_stage = stage
        state.stage_history.append((best_intent, best_score))

    return STAGE_RISK[state.current_stage], state.current_stage, best_intent, best_score, transcript


# ─────────────────────────────────────────────
# CORE INFERENCE — runs on thread pool
# ─────────────────────────────────────────────
def process_window(window: np.ndarray, state: SessionState) -> str:
    """
    Runs all 4 models in parallel on a 4-second audio window.
    Returns just the risk_label string for Android.
    """
    # VAD — skip silent windows
    speech = vad_filter(window)
    if len(speech) < SR * MIN_SPEECH_S:
        return "✅ safe"

    dominant = dominant_speaker_filter(speech)
    if len(dominant) < SR * MIN_SPEECH_S:
        dominant = speech   # fallback if filter too aggressive

    # ── All 4 models in parallel ──────────────────────────────
    f_phoneme = executor.submit(run_phoneme,    dominant)
    f_urgency = executor.submit(run_urgency,    dominant)
    f_rep     = executor.submit(run_repetition, speech)
    f_stage   = executor.submit(run_stage,      speech, state)

    phoneme_prob            = f_phoneme.result()
    urgency_score, urg_norm = f_urgency.result()
    rep_prob                = f_rep.result()
    stage_risk, stage_num, intent, sim_score, transcript = f_stage.result()

    # ── Track keyword hits ────────────────────────────────────
    if rep_prob >= REP_THRESHOLD:
        state.keyword_hits += 1
    state.processed += 1

    # ── Fuse scores ───────────────────────────────────────────
    raw_risk = (
        W_STAGE   * stage_risk   +
        W_REP     * rep_prob     +
        W_PHONEME * phoneme_prob +
        W_URGENCY * urg_norm
    )
    running_risk = state.update_running_risk(raw_risk)

    # ── Map to label ──────────────────────────────────────────
    if running_risk < 0.15:   return "✅ SAFE"
    elif running_risk < 0.25: return "🟡 LOW RISK"
    elif running_risk < 0.45: return "🟠 MODERATE"
    elif running_risk < 0.60: return "🔴 HIGH RISK"
    else:                     return "🚨 SCAM ALERT"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "OK"}


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    print("📱 Android client connected.")

    state = SessionState()   # fresh state per call/connection

    try:
        while True:
            # ── Receive 4-second int16 PCM bytes from Android ─────────
            # Expected size: 4 * SR * 2 bytes  (e.g. 4 * 16000 * 2 = 128000)
            audio_bytes = await websocket.receive_bytes()

            print(f"Length of incoming audio: {len(audio_bytes)}")
            # ── Sanity check ──────────────────────────────────────────
            expected_bytes = EXPECTED_SAMPLES * 2   # int16 = 2 bytes per sample
            if len(audio_bytes) != expected_bytes:
                print(f"⚠️ Unexpected chunk size: {len(audio_bytes)} (expected {expected_bytes})")
                await websocket.send_text("✅ safe")   # safe default, don't crash
                continue
            

            # ── Decode int16 → float32 [-1.0, 1.0] ───────────────────
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # ── Run inference on thread pool (frees the async loop) ───
            loop       = asyncio.get_event_loop()
            risk_label = await loop.run_in_executor(
                executor, process_window, audio, state
            )

            # ── Send risk_label string back to Android ────────────────
            await websocket.send_text(risk_label)

            print(f"  → {risk_label}  (running_risk={state.running_risk:.3f})")

    except WebSocketDisconnect:
        density = state.keyword_hits / state.processed if state.processed > 0 else 0.0
        print(
            f"❌ Client disconnected | "
            f"Windows={state.processed} | "
            f"FinalRisk={state.running_risk:.3f} | "
            f"KeywordDensity={density:.3f} | "
            f"FinalStage={STAGE_LABELS[state.current_stage]}"
        )
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)