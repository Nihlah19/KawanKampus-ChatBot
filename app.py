# app.py

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
LOCATION = "us-central1"

app = Flask(__name__)
CORS(app)

MEMORY_FOLDER = "memory"
DATASET_FILE = "cleaned_places.csv"  # File data kamu

os.makedirs(MEMORY_FOLDER, exist_ok=True)

# =====================================================
# SINKRONISASI NAMA KOLOM DATASET
# =====================================================
COL_KAMPUS = "Kampus"
COL_NAMA = "Nama_Tempat"
COL_KATEGORI = "Kategori_Awal" 
COL_LAT = "Latitude"
COL_LON = "Longitude"

# =====================================================
# LOAD & CLEAN DATASET (Logika Rekomendasi Dipertahankan)
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
# LOGIKA PENCARIAN BERDASARKAN JARAK TERDEKAT (Maks 5)
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
            "map_link": f"https://https://www.google.com/maps/search/?api=1&query=?q={row[COL_LAT]},{row[COL_LON]}"
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
Kamu adalah KawanKampus AI, asisten virtual mahasiswa Indonesia.
Tugas utama kamu adalah membantu mengerjakan tugas kuliah atau menjelaskan materi.

Gaya bicara:
- Santai, Natural, Friendly
- Seperti teman kampus pintar

Aturan respon:
- Tunjukkan langkah berpikirmu secara jelas dan logis sebelum memberikan jawaban akhir.
- Jawaban singkat, jelas, langsung ke inti.
- Jangan bertele-tele.
- Fokus membantu user menyelesaikan masalah tugasnya.
- Jangan mengulang pertanyaan user.
- Jangan pernah bilang "Sebagai AI..." atau sejenisnya.
"""

# =====================================================
# ROUTES & CHAT MAIN LOGIC 
# =====================================================
@app.route("/")
def home(): return render_template("index.html")

@app.route("/data/config")
def get_config():
    """Memberikan daftar Kampus dan Kategori murni backend dari CSV."""
    global cleaned_kampus_list, cleaned_category_list
    return jsonify({
        "kampus": cleaned_kampus_list,
        "kategori": cleaned_category_list
    })

@app.route("/chat", methods=["POST"])
def chat():
    global model
    try:
        data = request.form
        special_action = data.get("special_action")

        # =====================================================
        # MODE A: REKOMENDASI LOKASI (Pure Logic)
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
            
            return jsonify({
                "success": True,
                "reply": reply_text,
                "recommendations": rec_data 
            })

        # =====================================================
        # MODE B: BANTU TUGAS atau Chat Biasa (Route ke AI - Restored)
        # =====================================================
        else:
            user_message = data.get("message", "").strip()
            
            # --- TAMBAHAN KRUSIAL: Cek flag task_mode dari Frontend ---
            is_task_mode_active = data.get("task_mode") == "true"

            if not user_message:
                return jsonify({"success": False, "error": "Pesan kosong."}), 400
                
            # --- LOGIKA BARU: Prioritaskan Flag Mode Tugas ---
            if is_task_mode_active or "tugas" in user_message.lower():
                # -------------------------------------------------
                # SUB-MODE: BANTU TUGAS (Minta Respon Vertex AI)
                # -------------------------------------------------
                if model is None: 
                    init_model()
                
                if model is None:
                    return jsonify({"success": False, "error": "AI offline. Bantu tugas tidak tersedia."}), 500

                # Bangun Prompt Tanpa Riwayat Chat (murni fokus pertanyaan saat ini)
                prompt = f"""{SYSTEM_PROMPT}\n\nPertanyaan/Tugas User (Murni AI Tanpa CSV):\n{user_message}\n\nAssistant:"""

                # GENERATE AI
                try:
                    response = model.generate_content(prompt)
                    ai_reply = getattr(response, "text", "")
                except ResourceExhausted:
                    return jsonify({
                        "success": False,
                        "error": "LIMIT_VERTEX",
                        "message": "Quota AI sedang padat. Coba lagi nanti."
                    }), 429
                except Exception as e:
                     ai_reply = f"Terjadi kesalahan AI: {str(e)}"

                if not ai_reply: 
                    ai_reply = "Maaf, AI ga konek. Bisa ulangi pertanyaannya?"

                return jsonify({
                    "success": True, 
                    "reply": ai_reply,
                    "model": selected_model
                })

            else:
                # -------------------------------------------------
                # SUB-MODE: FALLBACK (Pure Backend Response)
                # -------------------------------------------------
                # Ini akan muncul jika user mengetik sembarang teks di input chat
                # JIKA BUKAN dalam mode tugas (Flag 'isTaskMode' di JS mati)
                return jsonify({
                    "success": True, 
                    "reply": "Maaf, saat ini aku murni bekerja berdasarkan tombol workflow untuk rekomendasi tempat, atau menjawab tentang 'Bantu Tugas'. Gunakan tombol di halaman awal ya!"
                })

    except Exception as e:
        print("ERROR HYBRID CHAT:", e)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )