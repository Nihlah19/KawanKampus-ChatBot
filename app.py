import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

# =========================
# LOAD ENV
# =========================
load_dotenv()

key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1"

# =========================
# APP
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# FOLDER MEMORY
# =========================
os.makedirs("memory", exist_ok=True)

# =========================
# INIT VERTEX AI
# =========================
vertexai.init(project=PROJECT_ID, location=LOCATION)

# =========================
# MODEL CONFIG
# =========================
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

model = None
selected_model = None

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
Kamu adalah KawanKampus, asisten AI khusus mahasiswa dan kehidupan kampus.

IDENTITAS:
- Nama: KawanKampus
- Persona: Teman kampus yang pintar, ramah, santai, suportif, dan cepat paham kebutuhan user.
- Gaya bicara: Natural, tidak kaku, mudah dipahami, seperti ngobrol dengan teman kampus yang helpful.

TUGAS:
- Bantu tugas kuliah
- Bantu coding & project
- Teman ngobrol
- Produktivitas mahasiswa
- Rekomendasi tempat

ATURAN:
- Singkat, jelas, membantu
- Jangan bertele-tele
- Gunakan bahasa natural
"""

# =========================
# INIT MODEL ANTI LOOP
# =========================
def init_model():
    global model, selected_model

    for model_name in AVAILABLE_MODELS:
        for attempt in range(2):  # retry max 2x
            try:
                temp = GenerativeModel(model_name)
                temp.generate_content("Halo")

                model = temp
                selected_model = model_name

                print("✅ MODEL AKTIF:", model_name)
                return True

            except Exception as e:
                print(f"❌ {model_name} gagal ({attempt+1}/2): {e}")
                time.sleep(2)

    model = None
    selected_model = None
    print("❌ Semua model gagal")
    return False

# Jalankan sekali saat startup
init_model()

# =========================
# WAKTU REALTIME
# =========================
def get_current_time_info():
    now = datetime.now(ZoneInfo("Asia/Jakarta"))

    hari_map = {
        "Monday": "Senin",
        "Tuesday": "Selasa",
        "Wednesday": "Rabu",
        "Thursday": "Kamis",
        "Friday": "Jumat",
        "Saturday": "Sabtu",
        "Sunday": "Minggu"
    }

    bulan_map = {
        "January": "Januari",
        "February": "Februari",
        "March": "Maret",
        "April": "April",
        "May": "Mei",
        "June": "Juni",
        "July": "Juli",
        "August": "Agustus",
        "September": "September",
        "October": "Oktober",
        "November": "November",
        "December": "Desember"
    }

    hari = hari_map[now.strftime("%A")]
    bulan = bulan_map[now.strftime("%B")]
    tanggal = f"{now.strftime('%d')} {bulan} {now.strftime('%Y')}"
    jam = now.strftime("%H:%M")

    return f"""
INFORMASI WAKTU:
Hari: {hari}
Tanggal: {tanggal}
Jam: {jam} WIB

Jika user bertanya waktu, gunakan data ini.
"""

# =========================
# JSON MEMORY
# =========================
def get_path(user_id):
    return f"memory/{user_id}.json"

def load_history(user_id):
    path = get_path(user_id)

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_history(user_id, role, message):
    history = load_history(user_id)

    history.append({
        "role": role,
        "message": message
    })

    with open(get_path(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    status = selected_model if selected_model else "No Model"
    return f"KawanKampus jalan 🚀 | Model: {status}"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_id = data.get("user_id", "guest")
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message kosong"}), 400

        # kalau model mati, coba init ulang
        if model is None:
            init_model()

        if model is None:
            return jsonify({
                "error": "AI sedang sibuk / quota habis. Coba lagi nanti."
            }), 503

        save_history(user_id, "user", user_message)

        history = load_history(user_id)

        history_text = ""
        for item in history[-10:]:
            history_text += f"{item['role']}: {item['message']}\n"

        prompt = f"""
{SYSTEM_PROMPT}

{get_current_time_info()}

Riwayat:
{history_text}

User: {user_message}
"""

        try:
            response = model.generate_content(prompt)
        except Exception as e:
            print("⚠️ Generate gagal:", e)

            # fallback ulang
            init_model()

            if model is None:
                return jsonify({
                    "error": "Semua model gagal. Coba lagi nanti."
                }), 503

            response = model.generate_content(prompt)

        ai_reply = response.text

        save_history(user_id, "assistant", ai_reply)

        return jsonify({
            "reply": ai_reply,
            "user_id": user_id,
            "model": selected_model
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    try:
        data = request.json
        user_id = data.get("user_id", "guest")

        path = get_path(user_id)

        if os.path.exists(path):
            os.remove(path)

        return jsonify({
            "message": "History dihapus",
            "user_id": user_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)