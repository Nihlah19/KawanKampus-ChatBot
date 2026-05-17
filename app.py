# app.py (VERSI FREEMIUM & RATE LIMITING)

import os
import re
import json
import pandas as pd
import math

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

app = Flask(__name__)
CORS(app)

# =====================================================
# SYSTEM STORAGE (Histori & Kuota)
# =====================================================
HISTORY_FOLDER = "chat_histories"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# File JSON buat nyatet kuota Vertex AI per user
QUOTA_FILE = "user_quotas.json"

if not os.path.exists(QUOTA_FILE):
    with open(QUOTA_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

DATASET_FILE = "cleaned_places.csv"

# --- LOGIKA REKOMENDASI LOKASI (DIPERTENGAHKAN, GW SKIP BIAR PENDEK, TETEP PAKE YANG LAMA YA) ---
# ... (Masukkan fungsi load_and_clean_dataset() dan get_nearest_recommendations() lu di sini) ...

# =====================================================
# INIT VERTEX AI 
# =====================================================
try:
    if PROJECT_ID:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"❌ GAGAL INIT VERTEX: {e}")

model = None
selected_model = "gemini-2.5-flash"

def init_model():
    global model
    try:
        if model is None:
            model = GenerativeModel(selected_model)
    except Exception as e:
        print(f"❌ ERROR MODEL: {e}")
init_model()

SYSTEM_PROMPT = """
Latar Belakang Persona:
Kamu adalah KawanKampus AI, tutor sebaya (peer tutor) dan mentor akademis virtual terkemuka untuk mahasiswa di Indonesia. Persona kamu adalah mahasiswa tingkat akhir yang jenius, berpengetahuan luas, metodis, namun sangat suportif, rendah hati, dan mudah didekati. Kamu bukan sekadar memberikan jawaban, tetapi mengajarkan cara berpikir.

Misi Utama:
Membantu user (mahasiswa) menyelesaikan tugas kuliah mereka dengan memberikan penjelasan yang mendalam, akurat secara akademis, dan komprehensif. Tujuan akhirmu adalah memastikan user memahami materi, bukan hanya menyalin jawaban.

Gaya Bicara & Nada:
- Gunakan bahasa Indonesia yang santai, natural, friendly, dan menggunakan jargon kampus umum sewajarnya (seperti "aku/kamu", "nih", "deh", "opspek", "skripsian").
- Hindari bahasa yang terlalu kaku/formal seperti surat dinas, tetapi tetap menjaga kesopanan akademis.
- Terdengar seperti teman kos yang pintar yang sedang mengajari temannya sebelum ujian.

Protokol Respon Wajib (Ikuti urutan ini dengan ketat):

Langkah 1: Analisis Masalah & Metodologi [Berpikir]
Sebelum memberikan jawaban apa pun, tuliskan analisis mendalam kamu terhadap pertanyaan user di dalam blok terpisah dengan judul "**[Analisis KawanKampus]**". Di sini kamu harus:
a. Identifikasi inti masalah dan tujuan pertanyaan.
b. Tentukan teori, rumus, konsep akademis, atau kerangka berpikir yang relevan yang akan digunakan untuk menjawab.
c. Uraikan langkah-langkah logis yang akan kamu tempuh untuk menyusun jawaban akhir.

Langkah 2: Penjelasan Teoretis & Kontekstual
Jelaskan konsep dasar atau teori pendukung yang relevan dengan pertanyaan tersebut secara detail. Gunakan analogi dunia nyata atau contoh kasus di Indonesia agar lebih mudah dipahami.

Langkah 3: Langkah-langkah Penyelesaian (Lakukan Perhitungan/Analisis)
Tunjukkan proses penyelesaian masalah secara bertahap (step-by-step).
- Jika bersifat hitungan, tunjukkan rumusnya, substitusi angkanya, dan proses perhitungannya dengan rapi menggunakan LaTeX untuk persamaan matematika.
- Jika bersifat analisis/esai, bangun argumen yang kuat, gunakan data (jika ada), dan referensikan konsep akademis yang relevan.

Langkah 4: Jawaban Akhir & Kesimpulan
Berikan jawaban akhir yang spesifik dan jelas dari pertanyaan user. Rangkum poin-poin penting dari penyelesaian yang telah dilakukan.

Langkah 5: Tips Tambahan/Pengayaan
Berikan satu tips tambahan, saran bacaan lanjutan, atau perangkap umum yang harus dihindari terkait materi tersebut untuk membantu user belajar lebih lanjut.

Aturan Tambahan & Larangan:
- HILANGKAN ATURAN "JAWABAN SINGKAT". Jawaban harus detail, mendalam, dan komprehensif. Jangan memotong penjelasan penting demi kesingkatan.
- Gunakan Markdown secara kaya: gunakan **bold** untuk penekanan, *italic* untuk istilah asing, `inline code` untuk variabel, dan blok kode untuk pemrograman.
- JANGAN BERTELE-TELE dengan ramah tamah yang tidak perlu (misalnya, "Halo, apa kabar? Semoga hari kamu menyenangkan, aku akan menjawab pertanyaamu..."). Langsung masuk ke protokol respon.
- Jangan mengulang pertanyaan user secara verbatim. Langsung interpretasikan masalahnya di bagian Analisis.
- JANGAN PERNAH menggunakan frasa "Sebagai AI", "Saya adalah model bahasa", atau sejenisnya. Tetaplah dalam persona teman kampus pintar.
- Jika user memberikan file atau codingan, berikan perbaikan yang clean, gunakan code block, dan jelaskan setiap perubahan logikanya.
"""

# =====================================================
# UTILITIES: HISTORI & LIMITASI KUOTA (NEW LOGIC)
# =====================================================
def get_time():
    return datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

def load_user_history(user_id: str):
    safe_filename = "".join([c for c in user_id if c.isalpha() or c.isdigit() or c=='@' or c=='.']).rstrip()
    filepath = os.path.join(HISTORY_FOLDER, f"{safe_filename}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_chat(user_id: str, role: str, message: str):
    safe_filename = "".join([c for c in user_id if c.isalpha() or c.isdigit() or c=='@' or c=='.']).rstrip()
    filepath = os.path.join(HISTORY_FOLDER, f"{safe_filename}.json")
    history = load_user_history(user_id)
    history.append({"role": role, "message": message, "timestamp": get_time()})
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# --- FUNGSI CEK KUOTA FREEMIUM ---
def get_vertex_usage(user_id: str):
    with open(QUOTA_FILE, "r", encoding="utf-8") as f:
        quotas = json.load(f)
    return quotas.get(user_id, 0) # Default 0 kalo user baru

def increment_vertex_usage(user_id: str):
    with open(QUOTA_FILE, "r", encoding="utf-8") as f:
        quotas = json.load(f)
    
    current_usage = quotas.get(user_id, 0)
    quotas[user_id] = current_usage + 1
    
    with open(QUOTA_FILE, "w", encoding="utf-8") as f:
        json.dump(quotas, f, indent=2)

# --- FUNGSI FILTER PERTANYAAN GAMPANG (Rule-Based NLP) ---
def is_simple_chat(text: str):
    text_lower = text.lower().strip()
    # Keyword dasar yang ga butuh AI
    easy_keywords = ["halo", "hai", "pagi", "siang", "sore", "malam", "siapa kamu", "makasih", "terima kasih", "ok", "oke", "thanks"]
    
    # Kalo textnya cuma 1-2 kata dan ada di keyword, anggep gampang
    if len(text_lower.split()) <= 3 and any(word in text_lower for word in easy_keywords):
        return True
    return False

def get_simple_reply(text: str):
    text_lower = text.lower()
    if "siapa" in text_lower:
        return "Aku KawanKampus AI, asisten virtual kamu buat ngerjain tugas kampus. Ada tugas yang bikin pusing?"
    elif "makasih" in text_lower or "thanks" in text_lower or "ok" in text_lower:
        return "Sama-sama bro! Santai aja, kalau ada tugas lagi kabarin gw ya."
    else:
        return "Halo bro! Gw KawanKampus AI. Ada materi kuliah atau tugas yang bisa gw bantu?"


# =====================================================
# ROUTES & CHAT MAIN LOGIC 
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "KawanKampus AI API Running", "version": "Freemium-v2"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    global model
    try:
        data = request.form
        user_id = data.get("user_id")
        
        if not user_id:
             return jsonify({"success": False, "error": "Wajib kirim user_id dari Frontend."}), 400

        special_action = data.get("special_action")
        user_message = data.get("message", "").strip()

        save_user_chat(user_id, "user", user_message)

        # =====================================================
        # MODE A: REKOMENDASI LOKASI (Diskip logikanya di sini biar lu fokus, tetep pake yang lama)
        # =====================================================
        if special_action == "recommendation_proximity":
            reply_text = "Logic rekomendasi kampus lu jalan di sini."
            save_user_chat(user_id, "assistant", reply_text)
            return jsonify({"success": True, "reply": reply_text})

        # =====================================================
        # MODE B: BANTU TUGAS (Dengan Filter & Limitasi)
        # =====================================================
        else:
            is_task_mode_active = data.get("task_mode") == "true"

            if not user_message:
                return jsonify({"success": False, "error": "Pesan kosong."}), 400
                
            if is_task_mode_active or "tugas" in user_message.lower():
                
                # 1. FILTER PERTANYAAN GAMPANG (Ga ngurangin Kuota)
                if is_simple_chat(user_message):
                    ai_reply = get_simple_reply(user_message)
                    save_user_chat(user_id, "assistant", ai_reply)
                    return jsonify({"success": True, "reply": ai_reply, "status": "local_reply"})

                # 2. CEK LIMIT KUOTA VERTEX AI (Maks 2)
                usage_count = get_vertex_usage(user_id)
                if usage_count >= 2:
                    # BLOCKING PAYWALL TRIGGERED!
                    paywall_msg = "Wah bro, sori banget nih. Kuota pertanyaan AI gratis kamu udah habis (Limit: 2 kali). Biar bisa nanya tugas sepuasnya, yuk **Upgrade ke KawanKampus Pro**! Hubungi admin kampus ya. 🚀"
                    save_user_chat(user_id, "assistant", paywall_msg)
                    return jsonify({"success": True, "reply": paywall_msg, "status": "quota_exceeded"})

                # 3. EKSEKUSI VERTEX AI (Pertanyaan Susah & Kuota Aman)
                if model is None: init_model()
                
                if model is None:
                    ai_reply = "AI offline, Bro. Bantu tugas ga bisa jalan."
                else:
                    prompt = f"""{SYSTEM_PROMPT}\n\nPertanyaan/Tugas User:\n{user_message}\n\nAssistant:"""
                    try:
                        response = model.generate_content(prompt)
                        ai_reply = getattr(response, "text", "Maaf, AI ga ngasih jawaban.")
                        
                        # BERHASIL JAWAB -> POTONG/TAMBAH KUOTA
                        increment_vertex_usage(user_id)
                        
                    except ResourceExhausted:
                        ai_reply = "Quota GCP gw yang padat, coba lagi semenit lagi."
                    except Exception as e:
                         ai_reply = f"Terjadi kesalahan AI: {str(e)}"

                save_user_chat(user_id, "assistant", ai_reply)

                return jsonify({
                    "success": True, 
                    "reply": ai_reply,
                    "usage_count": usage_count + 1, # Infoin ke frontend sisa berapa
                    "status": "ai_answered"
                })

            else:
                # FALLBACK
                reply_text = "Aku murni bekerja buat rekomendasi tempat dan bantu tugas. Gunakan tombol ya!"
                save_user_chat(user_id, "assistant", reply_text)
                return jsonify({"success": True, "reply": reply_text})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/history/<user_id>", methods=["GET"])
def get_history_endpoint(user_id):
    history = load_user_history(user_id)
    return jsonify({"user_id": user_id, "history": history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)