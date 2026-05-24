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
import zoneinfo
from flask_sqlalchemy import SQLAlchemy

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

app = Flask(__name__)
CORS(app)

# =====================================================
# SYSTEM STORAGE & DATABASE CONFIGURATION 
# =====================================================
# Membaca DATABASE_URL dari environment variable (untuk Postgres di Cloud Run)
# Jika tidak ada, otomatis menggunakan SQLite lokal (kawankampus.db)

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL belum diset!")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

DATASET_FILE = "cleaned_places.csv"

# =====================================================
# DATABASE MODELS (PENGGANTI JSON LOKAL)
# =====================================================
class User(db.Model): #buat nampung user_id dan kuota, biar ga perlu file JSON lagi
    __tablename__ = 'users'
    id = db.Column(db.String(100), primary_key=True) # user_id dari frontend
    quota_used = db.Column(db.Integer, default=0)    # Pengganti user_quotas.json
    created_at = db.Column(db.DateTime, default=datetime.utcnow) # Timestamp pembuatan user
    sessions = db.relationship('ChatSession', backref='user', lazy=True) # Relasi ke sesi chat (1 user bisa punya banyak sesi)

class ChatSession(db.Model): #buat nampung tiap ruang chat, biar bisa multi-session
    __tablename__ = 'chat_sessions'
    id = db.Column(db.String(100), primary_key=True) # session_id ruang chat
    user_id = db.Column(db.String(100), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), default="Obrolan Baru")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='session', lazy=True, cascade="all, delete-orphan")

class Message(db.Model): #buat nampung histori pesan per sesi, biar ga perlu file JSON lagi
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), db.ForeignKey('chat_sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'user' atau 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)

# Buat tabel otomatis jika belum terbuat
with app.app_context():
    db.create_all()

# =====================================================
# LOGIKA REKOMENDASI LOKASI 
# =====================================================
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

    print("UNI:", uni_id)
    print("CAT:", cat_id)

    print("DATA KAMPUS:")
    print(places_df['kampus_id'].unique()[:10])

    print("DATA KATEGORI:")
    print(places_df['jenis_id'].unique()[:10])

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
            def get_model():
                return GenerativeModel(selected_model)    
    except Exception as e:
        print(f"❌ ERROR MODEL: {e}")
init_model()

SYSTEM_PROMPT = """
Latar Belakang Persona:
Kamu adalah KawanKampus AI, tutor sebaya (peer tutor) and mentor akademis virtual terkemuka untuk mahasiswa di Indonesia. Persona kamu adalah mahasiswa tingkat akhir yang jenius, berpengetahuan luas, metodis, namun sangat suportif, rendah hati, dan mudah didekati. Kamu bukan sekadar memberikan jawaban, tetapi mengajarkan cara berpikir.

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
# UTILITIES: HISTORI & LIMITASI KUOTA
# =====================================================
def get_time():
    return datetime.now(zoneinfo.ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

def save_db_chat(session_id: str, role: str, message: str):
    new_msg = Message(session_id=session_id, role=role, content=message, timestamp=get_time())
    db.session.add(new_msg)
    db.session.commit()

# --- FUNGSI FILTER PERTANYAAN GAMPANG (Rule-Based NLP) ---
# =====================================================
# FUNGSI FILTER & UTALITAS ROUTING PINTAR (ANTI-BOCOR)
# =====================================================
def analyze_local_routing(text: str):
    """
    Menganalisis teks secara lokal untuk menyaring sapaan basa-basi 
    dan mengotomatiskan pengalihan kata kunci rekomendasi tempat.
    """
    text_lower = text.lower().strip()
    words_count = len(text_lower.split())

    # 1. OTOMATISASI JALUR REKOMENDASI LOKASI (DETEKSI TEKS)
    # Jika mendeteksi kata berkaitan dengan rekomendasi, langsung potong komando ke sini
    rec_keywords = [
        "rekomendasi", "rekomendasiin", "carikan tempat", "kos", "kosan", "kontrakan",
        "warkop", "cafe", "kafe", "fotokopi", "warung makan", "restoran", "nongkrong",
        "tempat murah", "dekat kampus", "kuliner", "makan mana", "print", "tempat ngeprint"
    ]
    if any(keyword in text_lower for keyword in rec_keywords):
        return {
            "is_local": True,
            "status": "recommendation_triggered",
            "reply": "Wah, kamu lagi nyari info tempat di sekitar kampus ya? Kebetulan gw punya fitur khusus pencarian lokasi terdekat! Yuk, langsung cek dan isi filter kampus serta kategori di menu **Rekomendasi Tempat** biar hasilnya akurat sesuai database kampusmu! 📍"
        }

    # 2. FILTER CASUAL CHAT & BASA-BASI PENDEK (Maksimal 4 Kata)
    greetings = ["halo", "hai", "oi", "hey", "helo", "pagi", "siang", "sore", "malam", "permisi", "misi", "assalamualaikum", "p", "test", "tes", "ping"]
    identities = ["siapa kamu", "siapa lu", "lu siapa", "kamu siapa", "apa ini", "fitur apa", "bisa apa aja", "aplikasi apa"]
    thanks_ok = ["makasih", "terima kasih", "thanks", "thank you", "tengks", "ok", "oke", "woke", "siap", "sip", "nuhun", "suwun", "mantap", "gokil", "paham"]
    farewells = ["bye", "dadah", "duluan", "cabut", "dah", "good bye"]

    if words_count <= 4:
        if any(g == text_lower or text_lower.startswith(g) for g in greetings):
            return {
                "is_local": True,
                "status": "local_reply",
                "reply": "Halo bro! Gw KawanKampus AI. Ada materi kuliah atau tugas yang bikin pusing dan perlu gw bantu bedah? Lempar ke sini aja! 🎓"
            }
        if any(i in text_lower for i in identities):
            return {
                "is_local": True,
                "status": "local_reply",
                "reply": "Aku KawanKampus AI, tutor sebaya virtual kamu. Gw dilatih khusus buat bantu kamu analisis tugas kuliah secara mendalam, sekalian bisa ngasih rekomendasi tempat hits atau kosan di sekitar kampus! 🚀"
            }
        if any(t in text_lower for t in thanks_ok):
            return {
                "is_local": True,
                "status": "local_reply",
                "reply": "Sama-sama, Bro! Santai aja, itu udah jadi tugas gw selaku mentor sebaya lu. Kalau ada tugas lain yang bikin mentok, langsung chat gw lagi ya! 💪"
            }
        if any(f in text_lower for f in farewells):
            return {
                "is_local": True,
                "status": "local_reply",
                "reply": "Sip, sampai jumpa lagi! Semangat kuliahnya, jangan lupa istirahat dan ngopi ya! ☕"
            }

    return {"is_local": False}

# =====================================================
# ROUTES & CHAT MAIN LOGIC 
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "KawanKampus AI API Running", "version": "Production-v3"}), 200

@app.route("/data/config", methods=["GET"])
def get_config():
    try:
        # 1. DAFTAR MASTER KAMPUS (Disamakan persis dengan data lu)
        raw_kampus = [
            'Universitas Airlangga - B', 
            'Universitas Bina Nusantara @Anggrek',
            'Universitas Brawijaya', 
            'Universitas Gadjah Mada',
            'Universitas Institut Teknologi Bandung - Ganesha', 
            'STMIK IKMI CIREBON',
            'UNIVERSITAS MULTI DATA PALEMBANG', 
            'Universitas Indonesia',
            'Universitas Pendidikan Indonesia Bandung'
        ]
        
        # 2. DAFTAR MASTER KATEGORI (Data kotor disaring otomatis menjadi nama bersih)
        raw_kategori = [
            'Apotek', 'Cafe', 'Fotokopi', 'Kedai', 'Makanan', 'Makanan Siap Saji',
            'Minimarket', 'Perhentian Bus', 'Pizza', 'Print', 'Restoran',
            'Restoran Padang', 'Tempat Fitness', 'Toko Es Krim', 'Warteg', 'Kedai Kopi',
            'Apotek.Csv', 'Cafe.Csv', 'Fotocopy.Csv', 'Kedai.Csv',
            'Makanan Siap Saji.Csv', 'Makanan.Csv', 'Minimarket.Csv',
            'Perhentian Bus.Csv', 'Pizza.Csv', 'Print.Csv', 'Restoran Padang.Csv',
            'Restoran.Csv', 'Tempat Fitness.Csv', 'Toko Es Krim.Csv', 'Warteg.Csv',
            'Fotokopi.Csv', 'Reestoran Padang.Csv', 'Restaurant.Csv', 'Toko Eskrim.Csv',
            ' Toko Eskrim.Csv'
        ]

        # Proses pembersihan kategori secara otomatis
        cleaned_categories = set()
        for cat in raw_kategori:
            # Hapus ekstensi .csv / .Csv jika ada
            clean = cat.replace(".Csv", "").replace(".csv", "")
            # Bersihkan spasi di awal dan di akhir kata
            clean = clean.strip()
            
            # Normalisasi typo yang parah agar seragam di tombol HTML
            if clean in ["Fotocopy", "Fotokopi"]:
                clean = "Fotokopi"
            elif clean in ["Reestoran Padang", "Restoran Padang"]:
                clean = "Restoran Padang"
            elif clean in ["Restaurant", "Restoran"]:
                clean = "Restoran"
            elif clean in ["Toko Eskrim", "Toko Es Krim"]:
                clean = "Toko Es Krim"
                
            if clean:  # Pastikan tidak kosong
                cleaned_categories.add(clean)

        # Kembalikan hasil bersih dalam bentuk list yang berurutan secara alfabetis
        return jsonify({
            "kampus": sorted(list(set(raw_kampus))),
            "kategori": sorted(list(cleaned_categories))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# MAIN ROUTE /CHAT
# =====================================================
@app.route("/chat", methods=["POST"])
def chat():
    global model
    try:
        # =====================================================
        # 1. PENGAMAN PAYLOAD DATA: SUPPORT JSON & FORM DATA (ANTI-400)
        # =====================================================
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        if not data:
            return jsonify({"success": False, "error": "Tidak ada data payload yang dikirim."}), 400

        user_id = data.get("user_id")
        session_id = data.get("session_id")
        
        if not user_id:
            return jsonify({"success": False, "error": "Wajib kirim user_id dari Frontend."}), 400
        
        if not session_id:
            session_id = f"default_sess_{user_id}"

        special_action = data.get("special_action")
        user_message = str(data.get("message", "")).strip()

        # Pemutus arus jika chat biasa kosong (tapi kalau workflow rekomendasi tombol, message emang dikosongin frontend lu)
        if not user_message and not special_action:
            return jsonify({"success": False, "error": "Pesan kosong."}), 400

        text_lower = user_message.lower()
        words_count = len(user_message.split())

        # Sync User & Session ke DB
        user = User.query.get(user_id)
        if not user:
            user = User(id=user_id)
            db.session.add(user)
            db.session.commit()

        session = ChatSession.query.get(session_id)
        if not session:
            title_preview = user_message if len(user_message) <= 25 else user_message[:22] + "..."
            session = ChatSession(id=session_id, user_id=user_id, title=title_preview or "Obrolan Baru")
            db.session.add(session)
            db.session.commit()

        if user_message:
            save_db_chat(session_id, "user", user_message)

        # =====================================================
        # 2. ENGINE ROUTING LOKAL (Saring Basa-basi Ketikan User)
        # =====================================================
        if user_message:
            local_route = analyze_local_routing(user_message)
            if local_route["is_local"]:
                ai_reply = local_route["reply"]
                save_db_chat(session_id, "assistant", ai_reply)
                return jsonify({
                    "success": True, 
                    "reply": ai_reply, 
                    "status": local_route["status"],
                    "usage_count": user.quota_used
                })

        # =====================================================
        # MODE A: REKOMENDASI LOKASI (DI SINI TEMPAT EDITNYA! 🎯)
        # =====================================================
        if special_action == "recommendation_proximity":
            # Tangkap lemparan data tombol dari HTML lu
            selected_uni = data.get("selected_uni")
            selected_cat = data.get("selected_cat")
            user_lat = data.get("lat")
            user_lon = data.get("lon")

            # =================================================================
            # LOGIKA LOGIC REKOMENDASI LU (Silakan ganti/sesuaikan dengan logic hitung jarak lu)
            # =================================================================
            try:
                # CONTOH IMPLEMENTASI LOGIC (Sesuaikan dengan fungsi milik lu sendiri):
                # info_msg = f"Menampilkan hasil pencarian untuk {selected_cat} di sekitar {selected_uni}."
                # list_rekomendasi = hitung_jarak_terdekat(selected_uni, kategori_untuk_search, user_lat, user_lon)
                
                # Ini placeholder logic bawaan lama lu, silakan disematkan fungsi pemroses aslinya:
                try:
                    # validasi input
                    if not selected_uni or not selected_cat:
                        return jsonify({
                            "success": False,
                            "error": "Kampus atau kategori belum dipilih."
                        }), 400

                    if not user_lat or not user_lon:
                        return jsonify({
                            "success": False,
                            "error": "Lokasi user tidak ditemukan."
                        }), 400

                    # convert koordinat
                    try:
                        user_lat = float(user_lat)
                        user_lon = float(user_lon)
                    except ValueError:
                        return jsonify({
                            "success": False,
                            "error": "Format koordinat invalid."
                        }), 400
                    
                    # ambil rekomendasi 5 terdekat
                    list_rekomendasi, reply_text = get_nearest_recommendations(
                        user_lat=user_lat,
                        user_lon=user_lon,
                        uni_name=selected_uni,
                        category_name=selected_cat
                    )

                    # fallback kalau kosong
                    if list_rekomendasi is None:
                        list_rekomendasi = []

                except Exception as e:
                    print("ERROR REKOMENDASI:", str(e))

                    reply_text = f"Gagal memproses rekomendasi: {str(e)}"
                    list_rekomendasi = []

            except Exception as e:
                    print("ERROR REKOMENDASI:", str(e))

                    reply_text = f"Gagal memproses rekomendasi: {str(e)}"
                    list_rekomendasi = []

            save_db_chat(session_id, "assistant", reply_text)
            return jsonify({
                "success": True, 
                "reply": reply_text, 
                "recommendations": list_rekomendasi, # Ini nanti dibaca oleh fungsi `renderRecommendations()` di HTML lu
                "status": "recommendation_triggered"
            })

        # =====================================================
        # MODE B: BANTU TUGAS (Proses Masuk ke Vertex AI)
        # =====================================================
        else:
            is_task_mode_active = data.get("task_mode") == "true"
                
            if is_task_mode_active or "tugas" in text_lower or words_count > 4:
                
                usage_count = user.quota_used
                updated = User.query.filter(
                    User.id == user_id,
                    User.quota_used < 2
                ).update({
                    "quota_used": User.quota_used + 1
                })

                db.session.commit()

                if updated == 0:                                   
                    paywall_msg = "Wah bro, sori banget nih. Kuota pertanyaan AI gratis kamu udah habis (Limit: 2 kali). Biar bisa nanya tugas sepuasnya, yuk **Upgrade ke KawanKampus Pro**! 🚀"
                    save_db_chat(session_id, "assistant", paywall_msg)
                    return jsonify({"success": True, "reply": paywall_msg, "status": "quota_exceeded"})

                if model is None: init_model()
                
                if model is None:
                    ai_reply = "AI offline, Bro. Bantu tugas ga bisa jalan."
                else:
                    past_msgs = Message.query.filter_by(session_id=session_id)\
                        .order_by(Message.id.desc())\
                        .limit(20)\
                        .all()

                    past_msgs = list(reversed(past_msgs))                    
                    history_context = ""
                    for msg in past_msgs[:-1]: 
                        history_context += f"{msg.role.capitalize()}: {msg.content}\n"

                    prompt = f"""{SYSTEM_PROMPT}\n\nRiwayat Obrolan Sebelumnya Sesi Ini:\n{history_context}\nPertanyaan/Tugas User Saat Ini:\n{user_message}\n\nAssistant:"""
                    
                    try:
                        response = model.generate_content(prompt)
                        ai_reply = getattr(response, "text", "Maaf, AI ga ngasih jawaban.")
                        user.quota_used += 1
                        db.session.commit()
                    except ResourceExhausted:
                        ai_reply = "Quota GCP gw yang padat, coba lagi semenit lagi."
                    except Exception as e:
                        db.session.rollback()
                        ai_reply = f"Terjadi kesalahan AI: {str(e)}"

                save_db_chat(session_id, "assistant", ai_reply)
                return jsonify({
                    "success": True, 
                    "reply": ai_reply,
                    "usage_count": user.quota_used, 
                    "status": "ai_answered"
                })

            else:
                reply_text = "Aku murni bekerja buat rekomendasi tempat dan bantu tugas kuliah, Bro. Ada tugas spesifik yang mau dibahas?"
                save_db_chat(session_id, "assistant", reply_text)
                return jsonify({"success": True, "reply": reply_text, "status": "fallback_reply"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500
            
# =====================================================
# ENDPOINTS ENDPOINT HISTORY BARU (MULTI-SESSION)
# =====================================================
@app.route("/api/history/<user_id>", methods=["GET"])
def get_old_history(user_id):
    """Fallback endpoint lama biar web lama lu ga error tiba-tiba"""
    messages = Message.query.join(ChatSession).filter(ChatSession.user_id == user_id).order_by(Message.id.asc()).all()
    history_list = [{"role": m.role, "message": m.content, "timestamp": m.timestamp} for m in messages]
    return jsonify({"user_id": user_id, "history": history_list})

@app.route("/api/history/<user_id>/<session_id>", methods=["GET"])
def get_session_history(user_id, session_id):
    """Endpoint baru untuk meload chat per Ruangan/Sesi khusus"""
    messages = Message.query.filter_by(session_id=session_id).order_by(Message.id.asc()).all()
    history_list = [{"role": m.role, "message": m.content, "timestamp": m.timestamp} for m in messages]
    return jsonify({"success": True, "user_id": user_id, "session_id": session_id, "history": history_list})

@app.route("/api/sessions/<user_id>", methods=["GET"])
def get_user_sessions(user_id):
    """Endpoint untuk list daftar obrolan di sidebar kiri"""
    sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.created_at.desc()).all()
    sessions_list = [{"session_id": s.id, "title": s.title, "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S")} for s in sessions]
    return jsonify({"success": True, "sessions": sessions_list})

@app.route("/test-db")
def test_db():
    try:
        db.session.execute("SELECT 1")
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Menyesuaikan port dinamis Cloud Run (PORT), default ke 5000 untuk lokal
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)

