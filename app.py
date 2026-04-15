import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv


load_dotenv()
print(f"API KEY TERDETEKSI: {os.getenv('GEMINI_API_KEY')[:5]}***") # Cek 5 huruf pertama saja

app = Flask(__name__)
CORS(app)

# === CONFIG ===
PROJECT_ID = os.getenv("GEMINI_API_KEY")
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

# System Prompt
SYSTEM_PROMPT = """
Kamu adalah "KawanKampus", chatbot mahasiswa.
Gaya santai, ramah, kayak temen kampus.

Tugas:
1. Bantu tugas kuliah
2. Rekomendasi tempat
3. Cerita random

Gunakan markdown, singkat, jelas.
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    history = data.get("history", [])

    try:
        # Gabungkan history
        chat = model.start_chat(history=[])

        full_prompt = SYSTEM_PROMPT + "\n\nUser: " + user_message

        response = chat.send_message(full_prompt)

        return jsonify({
            "reply": response.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)