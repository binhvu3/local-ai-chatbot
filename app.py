# --- Imports ---
import os
import logging
from datetime import datetime
import base64
import re

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, session, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import requests

import torch
from torchvision import models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

# --- Application Setup ---

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.setdefault('TESTING', False)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # IMPORTANT: Change in production

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# --- Database Models ---

class User(db.Model):
    """Represents a user in the database."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Message(db.Model):
    """Represents a chat message in the database."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_text = db.Column(db.String(200))
    bot_response = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    has_image = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(200), nullable=True)

# --- Helper Functions ---

def allowed_file(filename: str) -> bool:
    """Checks if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Image Classification ---

# Load a pretrained ResNet-50 model
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()

def classify_image(image_path: str) -> str:
    """
    Takes the path to an image and returns a classification label.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
        idx = outputs.argmax(dim=1).item()
        return weights.meta["categories"][idx]
    except Exception as e:
        logger.error("Local image classification error: %s", e)
        return ""

# --- LLM Interaction ---

def clean_response(text: str) -> str:
    """
    Removes <think> tags from the beginning of the LLM's response.
    This is useful for models that output their reasoning process.
    """
    m = re.match(r"<think>(.*?)</think>\s*(.*)", text, flags=re.S)
    return m.group(2).strip() if m else text

def generate_response(user_input: str) -> str:
    """
    Sends a prompt to the Ollama API and returns the response.
    """
    text = (user_input or "").strip()
    if not text:
        return "Please enter a message."

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": text,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        logger.error("Ollama API error: %s", e)
        return "Sorry, could not process your request. Please try again"
    except ValueError as e:
        logger.error("Ollama response parse error: %s", e)
        return "Sorry, there seem to be an error parsing the LLM response 🥲"


# --- Routes ---

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles user registration."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already taken", "error")
            return redirect(url_for("register"))
        u = User(username=username)
        u.set_password(password)
        db.session.add(u); db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        u = User.query.filter_by(username=username).first()
        if u and u.check_password(password):
            session['user_id'] = u.id
            return redirect(url_for("index"))
        flash("Invalid credentials", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Handles user logout."""
    session.pop('user_id', None)
    return redirect(url_for("login"))


@app.route("/")
def index():
    """Renders the main chat page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Fetch the last 10 messages for the current user
    msgs = Message.query.filter_by(user_id=session['user_id'])\
                        .order_by(Message.timestamp.desc())\
                        .limit(10).all()
    return render_template("chat.html", messages=msgs[::-1])

@app.route("/send", methods=["POST"])
def send():
    """
    Handles sending a message and getting a response.
    This is the primary endpoint for the chat functionality.
    """
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session['user_id']
    text = request.form.get("message", "")
    has_image = False
    image_path = None
    image_label = ""

    # Process image if it exists
    if 'image' in request.files:
        f = request.files['image']
        if f and f.filename and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(fp)
            has_image = True
            image_path = fp
            image_label = classify_image(fp)
            # Append image label to prompt for context
            text = f"{text} [Image: {image_label}]"

    bot_raw = generate_response(text)
    bot_msg = clean_response(bot_raw)

    # Save message to database
    msg = Message(
        user_id=user_id,
        user_text=text,
        bot_response=bot_msg,
        has_image=has_image,
        image_path=image_path
    )
    db.session.add(msg)
    db.session.commit()

    # Prepare JSON response for the client
    out = {"response": bot_msg}
    if has_image:
        with open(image_path, "rb") as imf:
            out["image"] = base64.b64encode(imf.read()).decode('utf-8')
        out["image_classification"] = image_label
    return jsonify(out)


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    """A dedicated endpoint for image classification."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    f = request.files['image']
    if not (f and allowed_file(f.filename)):
        return jsonify({"error": "Invalid file"}), 400
    fname = secure_filename(f.filename)
    fp = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(fp)
    label = classify_image(fp)
    return jsonify({"label": label}), 200


@app.route("/toggle-theme", methods=["POST"])
def toggle_theme():
    """Toggles the UI theme between light and dark mode."""
    data = request.get_json(silent=True)
    if data is None or 'dark_mode' not in data:
        return jsonify({"error": "Missing dark_mode flag"}), 400
    theme = "dark" if data["dark_mode"] else "light"
    session['dark_mode'] = data["dark_mode"]
    return jsonify({"theme": theme}), 200


# --- CLI and Main Execution ---

@app.cli.command('init_db')
def init_db():
    """Initializes the database."""
    db.create_all()
    print('Database initialized.')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # Create a default admin user for convenience
        if not User.query.filter_by(username="admin").first():
            default_username = "admin"
            default_password = "admin"
            default = User(username=default_username)
            default.set_password(default_password)
            db.session.add(default)
            db.session.commit()
            
            app.logger.info("Default admin user created")
            print("╔" + "═" * 76 + "╗")
            print("║" + " " * 76 + "║")
            print("║" + "🎉 Default Login Created:".center(75) + "║")
            print("║" + f"Username: {default_username}".center(76) + "║")
            print("║" + f"Password: {default_password}".center(76) + "║")
            print("║" + " " * 76 + "║")
            print("╚" + "═" * 76 + "╝")
    
    app.logger.info("Starting Flask application on localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)