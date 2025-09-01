from flask import Flask, request, render_template_string, send_from_directory, url_for, redirect
import os, uuid, tempfile
from datetime import datetime

# ------- 可選：雲端 Whisper 轉錄（需 OPENAI_API_KEY） -------
USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "0") == "1"  # 你目前設 0，就是用雲端
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ------- 簡單音訊分析（不需 AI） -------
def basic_audio_metrics(wav_path):
    try:
        from pydub import AudioSegment, silence
        import numpy as np
    except Exception:
        return {"error": "Need pydub & numpy (already in requirements.txt)"}

    audio = AudioSegment.from_file(wav_path)
    duration_s = len(audio) / 1000.0
    silences = silence.detect_silence(audio, min_silence_len=350, silence_thresh=audio.dBFS - 16)
    total_sil = sum((e - s) for s, e in silences) / 1000.0
    silence_ratio = total_sil / max(duration_s, 1e-6)
    samples = np.array(audio.get_array_of_samples()).astype(float)
    if samples.std() > 0:
        samples = (samples - samples.mean()) / (samples.std() + 1e-9)
    import numpy as np
    peaks = np.where((samples[1:-1] > 1.5) & (samples[1:-1] > samples[:-2]) & (samples[1:-1] > samples[2:]))[0]
    peak_rate = float(len(peaks)) / max(duration_s, 1e-6)

    return {
        "duration_s": round(duration_s, 2),
        "silence_ratio": round(silence_ratio, 3),
        "num_silence_segments": len(silences),
        "approx_peak_rate": round(peak_rate, 2),
    }

def transcribe_cloud(wav_path):
    if not OPENAI_API_KEY:
        return ""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        with open(wav_path, "rb") as f:
            # Whisper v1 API（如未啟用，僅回空字串）
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
        return getattr(transcript, "text", "") or ""
    except Exception:
        return ""

def short_feedback(transcript, metrics):
    # 70% 鼓勵 + 30% 指導（3-5句）
    en = (
        "Great job! Your voice is clear and confident. "
        "Try slowing down a little so each word is easy to hear, and press ending sounds like -s and -t. "
        "Keep practicing—I’m proud of your effort!"
    )
    zh = (
        "做得很棒！你的聲音清楚又有自信。"
        "下次放慢一點，讓每個字更清楚，也加強 -s、-t 這類結尾音。"
        "持續練習，我為你的努力感到驕傲！"
    )
    # 簡單量表（1-5）
    sil = float(metrics.get("silence_ratio", 0.2) or 0.2)
    peak = float(metrics.get("approx_peak_rate", 1.2) or 1.2)
    duration = float(metrics.get("duration_s", 30) or 30)
    pronunciation = 4
    fluency = 4
    intonation = 3
    if sil > 0.45: fluency -= 1
    if peak > 3.0 or peak < 0.8: fluency -= 1
    if len((transcript or "").split()) < max(int(duration // 2), 8): pronunciation -= 1
    rubric = {
        "pronunciation": max(1, min(5, pronunciation)),
        "fluency": max(1, min(5, fluency)),
        "intonation": max(1, min(5, intonation)),
    }
    return en, zh, rubric

app = Flask(__name__)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "student_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED = {"wav", "mp3", "m4a", "aac", "ogg", "flac", "webm", "mp4"}

INDEX_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upload speaking audio</title>
<style>
body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Arial; padding:24px;background:#f7f7f8}
.card{max-width:760px;margin:0 auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.06)}
label{display:block;margin:10px 0 6px;font-weight:600}
input{width:100%;padding:10px 12px;border:1px solid #ddd;border-radius:10px}
button{margin-top:16px;padding:12px 16px;border:none;border-radius:999px;background:#111;color:#fff;font-weight:700;cursor:pointer}
.hint{font-size:12px;color:#666}
</style>
</head><body>
<div class="card">
  <h2>Upload your speaking audio</h2>
  <p>Parents & Students: upload MP3/WAV/M4A/MP4 etc. Max 50MB.</p>
  <form method="post" enctype="multipart/form-data" action="{{ url_for('upload') }}">
    <label>Student name</label><input name="student" required placeholder="e.g., Andy Chen">
    <label>Age</label><input name="age" type="number" min="3" max="18" value="8">
    <label>Notes to teacher (optional)</label><input name="notes" placeholder="Anything we should know">
    <label>Audio file</label><input name="audio" type="file" accept="audio/*,video/mp4,video/webm" required>
    <button type="submit">Get feedback</button>
  </form>
  <p class="hint">Privacy: files stored in temp folder; avoid sensitive data.</p>
</div>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaking feedback</title>
<style>
body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Arial; padding:24px;background:#f7f7f8}
.wrap{max-width:900px;margin:0 auto}
.card{background:#fff;border-radius:16px;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.06);margin-bottom:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
table{width:100%;border-collapse:collapse}
th,td{border-bottom:1px solid #eee;padding:10px;text-align:left}
.score{font-weight:700}
audio{width:100%;margin-top:10px}
a.btn{display:inline-block;margin-top:8px;text-decoration:none;background:#111;color:#fff;padding:10px 14px;border-radius:999px;font-weight:700}
</style>
</head><body>
<div class="wrap">
  <div class="card">
    <h2>{{ student }} (Age {{ age }})</h2>
    <p>Uploaded at {{ uploaded_at }} · File: {{ filename }}</p>
    <audio controls src="{{ file_url }}"></audio><br>
    <a class="btn" href="{{ url_for('index') }}">← Upload another</a>
  </div>
  <div class="grid">
    <div class="card"><h3>Short Feedback (EN)</h3><p>{{ en }}</p></div>
    <div class="card"><h3>短評（中文）</h3><p>{{ zh }}</p></div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Rubric</h3>
      <table>
        <tr><th>Pronunciation</th><td class="score">{{ rubric.pronunciation }}/5</td></tr>
        <tr><th>Fluency</th><td class="score">{{ rubric.fluency }}/5</td></tr>
        <tr><th>Intonation</th><td class="score">{{ rubric.intonation }}/5</td></tr>
      </table>
      <p style="font-size:12px;color:#666">Scores are heuristic; for higher accuracy, enable transcription.</p>
    </div>
    <div class="card">
      <h3>Basic Audio Metrics</h3>
      <table>
        <tr><th>Duration (s)</th><td>{{ metrics.duration_s }}</td></tr>
        <tr><th>Silence ratio</th><td>{{ metrics.silence_ratio }}</td></tr>
        <tr><th>Silence segments</th><td>{{ metrics.num_silence_segments }}</td></tr>
        <tr><th>Approx. peak rate</th><td>{{ metrics.approx_peak_rate }}/s</td></tr>
      </table>
    </div>
  </div>
</div>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files: return redirect(url_for("index"))
    f = request.files["audio"]
    if not f.filename or not allowed(f.filename): return "Unsupported file type.", 400
    ext = f.filename.rsplit(".", 1)[1].lower()
    uid = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_DIR, uid)
    f.save(path)

    # 轉 WAV（便於分析）
    wav_path = path
    if ext != "wav":
        try:
            from pydub import AudioSegment
            wav_path = path + ".wav"
            AudioSegment.from_file(path).export(wav_path, format="wav")
        except Exception:
            wav_path = path

    metrics = basic_audio_metrics(wav_path)
    transcript = ""
    if not USE_LOCAL_WHISPER:
        transcript = transcribe_cloud(wav_path)

    en, zh, rubric = short_feedback(transcript, metrics if isinstance(metrics, dict) else {})

    return render_template_string(
        RESULT_HTML,
        student=request.form.get("student", "Student"),
        age=request.form.get("age", "-"),
        uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        filename=f.filename,
        file_url=url_for("serve_file", name=uid),
        en=en, zh=zh, rubric=rubric,
        metrics=metrics if isinstance(metrics, dict) else {"error": str(metrics)},
    )

@app.route("/file/<name>")
def serve_file(name):
    return send_from_directory(UPLOAD_DIR, name)

# 本機測試用；Render 由 gunicorn 啟動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
