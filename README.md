# 👁 RetinaAI — AI-Based Eye Disease Detection

An AI-powered web application for **Diabetic Retinopathy (DR)** detection using deep learning. Upload a fundus retinal image and get an instant severity prediction from **Grade 0 (No DR)** to **Grade 4 (Proliferative DR)**.

---

## 🚀 Features

- 🔬 **Deep Learning Model** — EfficientNetB3 with transfer learning
- 📊 **5-Class Classification** — Grade 0 to Grade 4 DR severity
- 📈 **Confidence Chart** — Visual probability breakdown for all grades
- 🕒 **Scan History** — Tracks all past analyzed images in session
- 🖥️ **Modern UI** — Dark medical-themed interface with drag & drop upload
- ⚡ **Real-time Prediction** — Instant results via Flask REST API

---

## 🩺 DR Severity Grades

| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild NPDR | Microaneurysms present |
| 2 | Moderate NPDR | Blocked blood vessels detected |
| 3 | Severe NPDR | Many blocked vessels, urgent referral needed |
| 4 | Proliferative DR | Abnormal new vessels, immediate intervention required |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML Model | TensorFlow, tf-Keras, EfficientNetB3 |
| Dataset | APTOS 2019 Blindness Detection (Kaggle) |

---

## 📁 Project Structure

```
RetinaAI/
├── index.html          # Main web UI
├── app.py              # Flask API server
├── train.py            # Model training script
├── create_csv.py       # Dataset label generator
├── requirements.txt    # Python dependencies
├── static/
│   ├── style.css       # UI styling
│   ├── script.js       # Frontend logic
│   └── favicon.svg     # App icon
├── dataset/            # APTOS 2019 images (not in repo)
│   ├── train/
│   └── train.csv
├── model/              # Trained model (not in repo)
│   └── retina_model.h5
└── uploads/            # Temp image storage
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/anshul-kushwahaa/RetinaAI.git
cd RetinaAI
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Download **APTOS 2019 Blindness Detection** from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection) and place images in `dataset/train/` with `train.csv`.

### 5. Train the model
```bash
python train.py
```

### 6. Run the application
```bash
# Terminal 1 — Flask API
python app.py

# Terminal 2 — Frontend
python -m http.server 8080
```

Open `http://localhost:8080` in your browser.

---

## 🌐 API Reference

### `POST /predict`
Upload a retinal image and get DR grade prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image>
```

**Response:**
```json
{
  "grade": 2,
  "confidence": 0.78,
  "all_probabilities": [0.05, 0.08, 0.78, 0.06, 0.03]
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNetB3 |
| Input Size | 300×300 px |
| Classes | 5 (Grade 0–4) |
| Training Epochs | 30 (early stopping) |
| Dataset | APTOS 2019 (3,662 images) |

---

## ⚕️ Disclaimer

RetinaAI is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a certified ophthalmologist for clinical decisions.

---

## 👨‍💻 Author

**Anshul Kushwaha**
- GitHub: [@anshul-kushwahaa](https://github.com/anshul-kushwahaa)

---

## 📄 License

This project is licensed under the MIT License.
