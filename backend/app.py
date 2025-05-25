from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os, pickle, json
import pandas as pd

# ------------------ Configuration ------------------
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# File upload
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Email
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=os.getenv("MAIL_USERNAME"),
)
mail = Mail(app)

# ------------------ Model ------------------
MODEL_PATH = 'model.pkl'
model = pickle.load(open(MODEL_PATH, 'rb')) if os.path.exists(MODEL_PATH) else None

# ------------------ History ------------------
HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

history = load_history()

# ------------------ Helpers ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('requirement_text', '').strip()
    if not text:
        flash("⚠️ Please enter a requirement.", "warning")
        return redirect(url_for('index'))

    if not model:
        flash("❌ Model not loaded.", "danger")
        return redirect(url_for('index'))

    prediction = model.predict([text])[0]
    history.append({'requirement': text, 'prediction': prediction})
    save_history(history)

    return render_template('index.html', prediction=prediction)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash("⚠️ No file selected.", "warning")
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash("❌ Unsupported file type.", "danger")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        if 'requirement' not in df.columns:
            flash("❌ Column 'requirement' not found.", "danger")
            return redirect(url_for('index'))

        df['prediction'] = model.predict(df['requirement'].astype(str))
        for _, row in df.iterrows():
            history.append({'requirement': row['requirement'], 'prediction': row['prediction']})
        save_history(history)

        df.to_csv('categorized_output.csv', index=False)
        flash("✅ Successfully categorized!", "success")

        return render_template(
            'index.html',
            functional=df[df['prediction'] == 'functional']['requirement'].tolist(),
            non_functional=df[df['prediction'] == 'non-functional']['requirement'].tolist()
        )
    except Exception as e:
        flash(f"❌ Error: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/categories')
def categories():
    return render_template('categories.html', history=history)

@app.route('/delete/<int:index>', methods=['POST'])
def delete_history_item(index):
    if 0 <= index < len(history):
        history.pop(index)
        save_history(history)
        flash("🗑️ Entry deleted.", "info")
    else:
        flash("❌ Invalid index.", "danger")
    return redirect(url_for('categories'))

@app.route('/download')
def download():
    path = 'categorized_output.csv'
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("❌ No file available to download.", "danger")
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        msg = Message(
            subject=f"New message from {name}",
            sender=email,
            recipients=[os.getenv("MAIL_USERNAME")],
            body=f"From: {name} <{email}>\n\n{message_body}"
        )

        try:
            mail.send(msg)
            flash("✅ Message sent!", "success")
        except Exception as e:
            print(f"Email error: {e}")
            flash("❌ Could not send message.", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html')

# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=False, port=5001)
