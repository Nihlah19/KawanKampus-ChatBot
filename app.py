# app.py (DEPLOY INI DI RENDER SENDERIAN)

import os
import re
import json
import pandas as pd
import math

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from flask_cors import CORS

# --- VERTEX AI DEPENDENCIES ---
import vertexai
from vertexai.generative_models import (
    GenerativeModel
)
from google.api_core.exceptions import (
    ResourceExhausted
)
# ----------------------------------------

from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# LOAD ENV & CONFIG 
# =====================================================
load_dotenv()

# vertexai sdk akan membaca GOOGLE_APPLICATION_CREDENTIALS secara otomatis
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Ambil dari env, default us-central1

app = Flask(__name__)
CORS(app) # CORS diaktifkan biar Frontend Fullstack bisa nembak langsung

# --- LOGIC HISTORI CHAT (JSON File-Based NoSQL) ---
# Folder buat nyimpen file JSON histori per user
HISTORY_FOLDER = "chat_histories"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

DATASET_FILE = "cleaned_places.csv"  # File data kamu

# =====================================================
# SINKRONISASI NAMA KOLOM DATASET
# =====================================================
COL_KAMPUS = "Kampus"
COL_NAMA = "Nama_Tempat"
COL_KATEGORI = "Kategori_Awal" 
COL_LAT = "Latitude"
COL_LON = "Longitude"

# =====================================================
# LOAD & CLEAN DATASET (Logika Rekomendasi Dipertahankan Total)
# =====================================================
places_df = None
cleaned_kampus_list = []
cleaned_category_list = []

def load_and_clean_dataset():
    global places_df, cleaned_kampus_list, cleaned_category_list
    try:
        if os.path.exists(DATASET_FILE):
            # 1. Load CSV
            df = pd.read_csv(DATASET_FILE, encoding='utf-8')
            
            # 2. Validasi Kolom Wajib Ada
            required_cols = [COL_KAMPUS, COL_NAMA, COL_KATEGORI, COL_LAT, COL_LON]
            if not all(col in df.columns for col in required_cols):
                print(f"❌ ERROR: Kolom dataset tidak sinkron. Pastikan ada: {required_cols}")
                places_df = pd.DataFrame()
                return

            # 3. Pastikan Koordinat & Kolom Vital tidak kosong & tipe data benar
            df = df.dropna(subset=[COL_LAT, COL_LON, COL_KATEGORI, COL_KAMPUS])
            df[COL_LAT] = pd.to_numeric(df[COL_LAT])
            df[COL_LON] = pd.to_numeric(df[COL_LON])

            # 4. Fungsi Pembersihan Teks
            def clean_text(text):
                if not isinstance(text, str): return ""
                text = text.strip() 
                # Hapus ekstensi .csv atau .Csv (case insensitive)
                text = re.sub(r'(?i)\.csv$', '', text)
                text = text.strip()
                return text

            # 5. Terapkan pembersihan dasar
            df['kampus_clean'] = df[COL_KAMPUS].apply(clean_text)
            df['jenis_clean'] = df[COL_KATEGORI].apply(clean_text)

            # 6. Map Perbaikan Typo Spesifik 
            category_typo_map = {
                'fotocopy': 'fotokopi',
                'reestoran padang': 'restoran padang',
                'restaurant': 'restoran',
                'toko eskrim': 'toko es krim',
            }

            def fix_typos(text):
                lower_text = text.lower()
                if lower_text in category_typo_map:
                    return category_typo_map[lower_text]
                return lower_text

            # 7. Terapkan perbaikan typo dan buat ID Lowercase untuk pencarian
            df['jenis_id'] = df['jenis_clean'].apply(fix_typos)
            df['kampus_id'] = df['kampus_clean'].str.lower()

            # 8. Simpan DataFrame yang sudah diproses
            places_df = df
            
            # 9. Buat Daftar untuk Tombol UI (Unik, Terurut, Title Case)
            raw_unis = df['kampus_clean'].unique()
            cleaned_kampus_list = sorted([u.title() for u in raw_unis if u])

            raw_categories_ids = df['jenis_id'].unique()
            cleaned_category_list = sorted([c.title() for c in raw_categories_ids if c])

            print(f"✅ DATASET SINKRON & DIBERSIHKAN (Pure Logic Rekomendasi Aktif).")
            
        else:
            print(f"⚠️ DATASET TIDAK DITEMUKAN: File '{DATASET_FILE}' diperlukan.")
            places_df = pd.DataFrame()

    except Exception as e:
        print(f"❌ GAGAL MEMUAT/MEMBERSIHKAN DATASET: {e}")
        places_df = pd.DataFrame()

load_and_clean_dataset()

# =====================================================
# GEOLOCATION LOGIC (Rumus Haversine)
# =====================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius bumi km
    l1, o1, l2, o2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = l2 - l1
    dlon = o2 - o1
    a = math.sin(dlat / 2)**2 + math.cos(l1) * math.cos(l2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# =====================================================
# LOGIKA PENCARIAN BERDASARKAN JARAK TERDEKAT (Dipertahankan)
# =====================================================
def get_nearest_recommendations(user_lat, user_lon, uni_name, category_name):
    global places_df
    if places_df is None or places_df.empty:
        return None, "Database tempat bermasalah atau kosong."

    uni_id = uni_name.lower().strip()
    cat_id = category_name.lower().strip()

    # Filter berdasarkan Kampus ID DAN Kategori ID
    results = places_df[
        (places_df['kampus_id'] == uni_id) & 
        (places_df['jenis_id'] == cat_id)
    ].copy()

    if results.empty:
        return None, f"Yah, aku belum punya data {category_name.upper()} di sekitar kampus {uni_name.upper()} di database."

    # Hitung jarak murni backend
    results['distance_km'] = results.apply(
        lambda row: haversine_distance(user_lat, user_lon, row[COL_LAT], row[COL_LON]),
        axis=1
    )

    # Urutkan terdekat dan ambil 5
    final_results = results.sort_values(by='distance_km').head(5)

    recommendations_data = []
    for _, row in final_results.iterrows():
        dist_str = f"{int(row['distance_km'] * 1000)} m" if row['distance_km'] < 1 else f"{row['distance_km']:.1f} km"

        recommendations_data.append({
            "name": row[COL_NAMA], # Nama asli dari CSV
            "distance": f"📍 Jarak: {dist_str}",
            "hours": f"{row[COL_KATEGORI].title()} - Sekitar {row[COL_KAMPUS].title()}", 
            "map_link": f"https://www.google.com/maps/search/?api=1&query={row[COL_LAT]},{row[COL_LON]}"
        })

    reply = f"Oke, berdasarkan lokasi kamu saat ini, ini 5 rekomendasi {category_name.upper()} paling dekat dari kampus {uni_name.upper()} yang aku temuin murni menggunakan database Full-Stack:"
    return recommendations_data, reply


# =====================================================
# INIT VERTEX AI 
# =====================================================
try:
    if not PROJECT_ID:
        raise Exception("GCP_PROJECT_ID tidak ditemukan di file .env")
    
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION
    )
    print(f"✅ VERTEX AI INISIALISASI BERHASIL.")
except Exception as e:
    print(f"❌ GAGAL INISIALISASI VERTEX AI: {e}")

# =====================================================
# AI MODELS (Stabil Gemini Version)
# =====================================================
model = None
# Gunakan Gemini 2.5 Flash agar tidak error 404
selected_model = "gemini-2.5-flash"

def init_model():
    global model
    try:
        if model is None:
            model = GenerativeModel(selected_model)
            print(f"✅ MODEL AI AKTIF: {selected_model}")
    except Exception as e:
        print(f"❌ GAGAL INIT MODEL AI: {e}")
init_model()

# =====================================================
# SYSTEM PROMPT (Updated to show thought process)
# =====================================================
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
# UTILITIES LOGIKAHISTORI (New)
# =====================================================
def get_time():
    return datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

def load_user_history(user_id: str):
    """Membaca file JSON histori chat berdasarkan ID User."""
    # Amankan filename, hapus karakter aneh
    safe_filename = "".join([c for c in user_id if c.isalpha() or c.isdigit() or c=='@' or c=='.']).rstrip()
    filepath = os.path.join(HISTORY_FOLDER, f"{safe_filename}.json")
    
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return [] # Kalo ga ada, balikin list kosong

def save_user_chat(user_id: str, role: str, message: str):
    """Menambahkan chat baru ke file JSON histori User."""
    safe_filename = "".join([c for c in user_id if c.isalpha() or c.isdigit() or c=='@' or c=='.']).rstrip()
    filepath = os.path.join(HISTORY_FOLDER, f"{safe_filename}.json")
    
    # Load data lama
    history = load_user_history(user_id)
    
    # Tambah data baru
    history.append({
        "role": role,
        "message": message,
        "timestamp": get_time()
    })
    
    # Simpan kembali
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# =====================================================
# ROUTES & CHAT MAIN LOGIC 
# =====================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "KawanKampus AI Server is Running", "version": "Stateful-NoSQL-v1"}), 200

@app.route("/data/config")
def get_config():
    """Memberikan daftar Kampus dan Kategori murni backend dari CSV."""
    global cleaned_kampus_list, cleaned_category_list
    return jsonify({
        "kampus": cleaned_kampus_list,
        "kategori": cleaned_category_list
    })

# --- PERBAIKAN TOTAL ENDPOINT CHAT (Updated for History) ---
@app.route("/chat", methods=["POST"])
def chat():
    global model
    try:
        # Flask ngebaca request.form (multpart/form-data)
        data = request.form
        
        # --- TAMBAHAN KRUSIAL: Ambil ID User dari Fullstack ---
        user_id = data.get("user_id")
        if not user_id:
             return jsonify({"success": False, "error": "Mana user_id nya, Bro? Wajib dikirim dari Fullstack."}), 400

        special_action = data.get("special_action")
        user_message = data.get("message", "").strip()

        # 1. Simpan Pesan User ke Histori NoSQL di Render
        save_user_chat(user_id, "user", user_message)

        # =====================================================
        # MODE A: REKOMENDASI LOKASI (Pure Logic - Dipertahankan)
        # =====================================================
        if special_action == "recommendation_proximity":
            user_lat = data.get("lat")
            user_lon = data.get("lon")
            selected_uni = data.get("selected_uni")
            selected_cat = data.get("selected_cat")

            if not all([user_lat, user_lon, selected_uni, selected_cat]):
                 return jsonify({"success": False, "error": "Data lokasi atau pilihan tidak lengkap."}), 400

            try:
                lat = float(user_lat)
                lon = float(user_lon)
            except ValueError:
                 return jsonify({"success": False, "error": "Koordinat lokasi tidak valid."}), 400

            # Gunakan logika pure Full-Stack murni
            rec_data, reply_text = get_nearest_recommendations(lat, lon, selected_uni, selected_cat)
            
            # Simpan balasan Bot murni logic ke Histori
            save_user_chat(user_id, "assistant", reply_text)

            return jsonify({
                "success": True,
                "reply": reply_text,
                "recommendations": rec_data 
            })

        # =====================================================
        # MODE B: BANTU TUGAS atau Chat Biasa (Route ke AI - Restored)
        # =====================================================
        else:
            # Cek flag task_mode dari Frontend untuk menghentikan loopingFallback
            is_task_mode_active = data.get("task_mode") == "true"

            if not user_message:
                return jsonify({"success": False, "error": "Pesan kosong."}), 400
                
            # Logika Percabangan dalam Mode Chat Biasa
            if is_task_mode_active or "tugas" in user_message.lower():
                # -------------------------------------------------
                # SUB-MODE: BANTU TUGAS (Minta Respon Vertex AI)
                # -------------------------------------------------
                if model is None: 
                    init_model()
                
                if model is None:
                    ai_reply = "AI offline, Bro. Bantu tugas ga bisa jalan."
                else:
                    # Prompt murni fokus pertanyaan saat ini (Tanpa context lama sesuai request)
                    prompt = f"""{SYSTEM_PROMPT}\n\nPertanyaan/Tugas User (Murni AI Tanpa CSV):\n{user_message}\n\nAssistant:"""

                    # GENERATE AI
                    try:
                        response = model.generate_content(prompt)
                        ai_reply = getattr(response, "text", "")
                    except ResourceExhausted:
                        ai_reply = "Quota AI padat, coba lagi nanti semenit lagi."
                    except Exception as e:
                         ai_reply = f"Terjadi kesalahan AI: {str(e)}"

                if not ai_reply: 
                    ai_reply = "Maaf, AI ga konek. Bisa ulangi pertanyaannya?"

                # Simpan balasan AI ke Histori NoSQL di Render
                save_user_chat(user_id, "assistant", ai_reply)

                return jsonify({
                    "success": True, 
                    "reply": ai_reply,
                    "model": selected_model
                })

            else:
                # -------------------------------------------------
                # SUB-MODE: FALLBACK (Pure Backend Response)
                # -------------------------------------------------
                # JIKA BUKAN dalam mode tugas
                reply_text = "Maaf, saat ini aku murni bekerja berdasarkan tombol workflow untuk rekomendasi tempat, atau menjawab tentang 'Bantu Tugas'. Gunakan tombol di halaman awal ya!"
                save_user_chat(user_id, "assistant", reply_text)

                return jsonify({
                    "success": True, 
                    "reply": reply_text
                })

    except Exception as e:
        print("ERROR HYBRID CHAT:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# --- NEW ENDPOINT: AMBIL HISTORI (Buat nampilin di profil user) ---
@app.route("/api/history/<user_id>", methods=["GET"])
def get_history_endpoint(user_id):
    """Endpoint buat Frontend Fullstack ngambil histori chat si user."""
    # Backend Node.js/PHP lu tinggal nembak GET ke sini pas user buka halaman chat
    history = load_user_history(user_id)
    return jsonify({
        "user_id": user_id,
        "history": history
    })

if __name__ == "__main__":
    # Load history dinonaktifkan di frontend awal, jadi route history dihapus
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )