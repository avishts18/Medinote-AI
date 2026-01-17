from flask import Flask, request, jsonify, render_template, send_file
import os, json, uuid
from datetime import datetime
from groq import Groq
from config import Config
from google.cloud import speech
from pydub import AudioSegment, silence
from PIL import Image
import cv2, textwrap

# ---------------- APP SETUP ----------------

app = Flask(__name__)
client = Groq(api_key=Config.GROQ_API_KEY)

BASE_DIR = os.getcwd()
STATIC_DIR = os.path.join(BASE_DIR, "static")
PATIENT_DIR = os.path.join(STATIC_DIR, "patient_details")

os.makedirs(PATIENT_DIR, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    STATIC_DIR, "APIKEY", "google.json"
)

# ---------------- UI ROUTES ----------------

@app.route("/")
def analyse():
    return render_template("Analyse.html")

# ---------------- PATIENT LOOKUP ----------------

@app.route("/lookup-patient", methods=["POST"])
def lookup_patient():
    data = request.json
    phone = data.get("phone_number")

    for pid in os.listdir(PATIENT_DIR):
        info_path = os.path.join(PATIENT_DIR, pid, "patient_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                if info.get("contact_number") == phone:
                    return jsonify({
                        "patient_id": pid,
                        "name": info.get("name")
                    })

    return jsonify({"error": "Patient not found"}), 404

# ---------------- AUDIO UPLOAD ----------------

@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    patient_id = request.form.get("patient_id")
    language = request.form.get("language", "en-US")
    audio_file = request.files.get("audio")

    if not patient_id or not audio_file:
        return jsonify({"error": "Invalid request"}), 400

    date = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(PATIENT_DIR, patient_id, date)
    os.makedirs(folder, exist_ok=True)

    audio_path = os.path.join(folder, "recording.wav")
    audio_file.save(audio_path)

    transcript = transcribe_audio(audio_path, language)

    transcript_path = os.path.join(folder, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump({"transcript": transcript}, f, indent=4)

    return jsonify({"transcript": transcript})

def transcribe_audio(path, language):
    audio = AudioSegment.from_file(path)
    chunks = silence.split_on_silence(audio, 500, -40)

    client = speech.SpeechClient()
    full_text = []

    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        with open(chunk_path, "rb") as f:
            content = f.read()

        response = client.recognize(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code=language,
                enable_automatic_punctuation=True
            ),
            audio=speech.RecognitionAudio(content=content),
        )

        for r in response.results:
            full_text.append(r.alternatives[0].transcript)

        os.remove(chunk_path)

    return " ".join(full_text)

# ---------------- REPORT GENERATION ----------------

@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.json
    transcript = data.get("transcript")

    if not transcript:
        return jsonify({"error": "Transcript missing"}), 400

    patient = extract_llm(transcript, "patient")
    prescription = extract_llm(transcript, "prescription")

    return jsonify({
        "patient": patient,
        "prescription": prescription
    })

def extract_llm(text, mode):
    prompts = {
        "patient": """
Extract patient details as JSON:
Name, Age, Gender, Blood Group, Height, Weight, Contact Number
""",
        "prescription": """
Extract prescription details as JSON:
Tablet(Name, Timing, Before/After Food, Frequency), Revisiting
"""
    }

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"{prompts[mode]}\n{text}"
        }]
    )

    return json.loads(res.choices[0].message.content)

# ---------------- PDF DOWNLOAD (OPTIONAL) ----------------

@app.route("/download-latest/<patient_id>")
def download_latest(patient_id):
    base = os.path.join(PATIENT_DIR, patient_id)
    dates = sorted(os.listdir(base), reverse=True)

    pdf_path = os.path.join(base, dates[0], "combined_prescription.pdf")
    return send_file(pdf_path, as_attachment=True)

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
