"""
Flask web application for Career Guidance Chatbot.
Run: python app.py
"""

import uuid
from flask import Flask, render_template, request, jsonify, session
from chatbot import get_chatbot
from database import init_db, save_chat, get_careers_by_stream, get_all_exams, get_colleges_by_stream

app = Flask(__name__)
app.secret_key = "career_guidance_secret_key_2024"

# Initialize DB and chatbot on startup
init_db()
chatbot = get_chatbot()


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    response = chatbot.respond(user_message)

    # Save to DB
    save_chat(session_id, user_message, response)

    return jsonify({"response": response})


@app.route("/api/careers/<stream>")
def careers(stream):
    results = get_careers_by_stream(stream)
    return jsonify(results)


@app.route("/api/exams")
def exams():
    results = get_all_exams()
    return jsonify(results)


@app.route("/api/colleges/<stream>")
def colleges(stream):
    results = get_colleges_by_stream(stream)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
