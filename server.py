"""
AI Wellness Assistant - FastAPI Backend
Run with:  uvicorn server:app --reload --port 8000
"""

import random
import time
import threading

import cv2
import pandas as pd
import pyttsx3
import speech_recognition as sr
from deepface import DeepFace
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ── app ──────────────────────────────────────
app = FastAPI(title="AI Wellness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model ────────────────────────────────────
df    = pd.read_csv("emotion_dataset.csv")
model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
model.fit(df["text"], df["emotion"])

# ── tts ──────────────────────────────────────
tts_engine = pyttsx3.init()

def speak(text: str):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception:
        pass

# ── helpers ──────────────────────────────────
def recommend(emotion: str) -> str:
    m = {
        "happy":  "Keep up the great energy — ideal state for productive studying!",
        "sad":    "Take some rest. Listen to calming music and be kind to yourself.",
        "angry":  "Step away and try slow deep breathing before returning.",
        "fear":   "You are safe. Write down worries to externalise them.",
        "stress": "Take a 5-minute break — drink water, stretch, breathe deeply.",
        "sleepy": "Splash water on your face and do a quick stretch.",
    }
    return m.get(emotion, "Stay focused and take care of your wellbeing.")

def motivation(emotion: str) -> str:
    quotes = {
        "sad":    ["Believe in yourself.", "Small progress is still progress.", "You are stronger than you think."],
        "stress": ["One step at a time.", "Stay calm and focused.", "You can do this!"],
        "fear":   ["Courage is not the absence of fear.", "You've got this.", "Breathe. You are ready."],
        "happy":  ["Great job!", "Keep learning and growing.", "Stay consistent!"],
        "angry":  ["Patience is power.", "This too shall pass.", "Channel this energy positively."],
        "sleepy": ["Recharge and come back stronger.", "Rest is productive."],
    }
    return random.choice(quotes.get(emotion, ["Stay positive and healthy."]))

def activities(emotion: str) -> list:
    m = {
        "stress": ["Meditation", "Deep breathing", "Drink water", "Short walk"],
        "fear":   ["Positive affirmations", "Talk to someone", "Grounding exercise"],
        "angry":  ["Deep breathing", "Cold water splash", "Write your feelings"],
        "sad":    ["Listen to music", "Talk with friends", "Watch motivational videos"],
        "sleepy": ["Face wash", "Stretching", "Drink coffee or tea"],
        "happy":  ["Stay productive", "Continue studying", "Set a new goal"],
    }
    return m.get(emotion, ["Stay productive", "Stay hydrated", "Plan your next task"])

def build_result(emotion: str) -> dict:
    return {
        "emotion":    emotion,
        "recommend":  recommend(emotion),
        "motivation": motivation(emotion),
        "activities": activities(emotion),
    }

# ── routes ───────────────────────────────────

class TextIn(BaseModel):
    text: str

@app.post("/analyze/text")
def analyze_text(body: TextIn):
    emotion = model.predict([body.text])[0]
    speak(f"Detected emotion: {emotion}. {recommend(emotion)}")
    return build_result(emotion)


@app.get("/analyze/voice")
def analyze_voice():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.5)
            audio = r.listen(src, timeout=10, phrase_time_limit=8)
        text    = r.recognize_google(audio)
        emotion = model.predict([text])[0]
        speak(f"I detected that you are feeling {emotion}.")
        return {"transcript": text, **build_result(emotion)}
    except sr.WaitTimeoutError:
        return {"error": "No speech detected. Please try again."}
    except sr.UnknownValueError:
        return {"error": "Could not understand audio. Speak more clearly."}
    except Exception as e:
        return {"error": str(e)}


# shared state for camera stream
cam_state: dict = {"running": False, "emotions": [], "stress": []}

def _camera_worker():
    cap = cv2.VideoCapture(0)
    while cam_state["running"]:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            res     = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)
            emotion = res[0]["dominant_emotion"]
        except Exception:
            emotion = "neutral"
        stress = (
            random.randint(70, 95) if emotion in ["sad", "fear", "angry"]
            else random.randint(20, 40) if emotion == "happy"
            else random.randint(40, 65)
        )
        cam_state["emotions"].append(emotion)
        cam_state["stress"].append(stress)
        time.sleep(2)
    cap.release()

@app.get("/camera/start")
def camera_start():
    cam_state["running"] = True
    cam_state["emotions"] = []
    cam_state["stress"]   = []
    threading.Thread(target=_camera_worker, daemon=True).start()
    return {"status": "started"}

@app.get("/camera/stop")
def camera_stop():
    cam_state["running"] = False
    emotions = cam_state["emotions"]
    stresses = cam_state["stress"]
    if not emotions:
        return {"error": "No data collected"}
    dominant = max(set(emotions), key=emotions.count)
    avg_stress = int(sum(stresses) / len(stresses))
    return {
        "avg_stress": avg_stress,
        "samples":    len(emotions),
        **build_result(dominant),
    }

@app.get("/camera/status")
def camera_status():
    emotions = cam_state["emotions"]
    stresses = cam_state["stress"]
    if not emotions:
        return {"running": cam_state["running"], "emotion": "—", "stress": 0}
    return {
        "running": cam_state["running"],
        "emotion": emotions[-1],
        "stress":  stresses[-1],
    }


def _meditation_gen():
    speak("Close your eyes and relax")
    for i in range(3):
        yield f"data: inhale|Round {i+1} of 3\n\n"
        speak("Inhale")
        time.sleep(3)
        yield f"data: exhale|Round {i+1} of 3\n\n"
        speak("Exhale")
        time.sleep(3)
    yield "data: done|Complete\n\n"
    speak("You are calm now. Great job.")

@app.get("/meditation/stream")
def meditation_stream():
    return StreamingResponse(_meditation_gen(), media_type="text/event-stream")