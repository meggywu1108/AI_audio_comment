from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ It works! Your Render service is running."

# （用 gunicorn 啟動時不需要下面這段，但本機測試用）
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
