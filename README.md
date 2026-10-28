# SentiPulse AI - Emotion Predictor

An end-to-end Deep Learning Natural Language Processing (NLP) web application that analyzes text and predicts human emotion across 6 classes: **Sadness**, **Joy**, **Love**, **Anger**, **Fear**, and **Surprise**.

The project uses a **Bidirectional Gated Recurrent Unit (BiGRU)** neural network trained on the `dair-ai/emotion` dataset, achieving **91.95% test accuracy**.

---

## Live Deployments

- **Live Website (Vercel)**: https://emotionpredictor.vercel.app
- **Backend API (Render)**: https://emotion-predictor-9mn2.onrender.com
- **API Health Check**: https://emotion-predictor-9mn2.onrender.com/health
- **GitHub Repository**: https://github.com/The-Ghost23/Emotion-Predictor

---

## How It Works

1. **Text Preprocessing**: Normalizes text by lowercasing, expanding apostrophes, stripping punctuation, and trimming whitespace.
2. **Tokenization & Padding**: Converts text to numeric sequences using a 10,000-word vocabulary, padded to a maximum length of 50 tokens.
3. **BiGRU Model**: 
   - Embedding Layer (300 dimensions, 3M parameters)
   - Bidirectional GRU Layer 1 (128 units forward + 128 units backward) with 0.5 Dropout
   - Bidirectional GRU Layer 2 (64 units forward + 64 units backward) with 0.5 Dropout
   - Dense Output Layer with Softmax activation across 6 emotion classes
4. **FastAPI Backend**: Serves predictions with latency and confidence scores.
5. **Frontend UI**: Responsive web interface with real-time probability meters, emotion presets, and dark/light mode.

---

## Model Evaluation

Evaluated on 2,000 unseen holdout test samples:

- **Test Accuracy**: 91.95%
- **Test Loss**: 0.1960
- **Total Parameters**: 3,454,662

### Per-Class Metrics

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Sadness | 0.97 | 0.96 | 0.97 | 581 |
| Joy | 0.99 | 0.90 | 0.94 | 695 |
| Love | 0.73 | 0.99 | 0.84 | 159 |
| Anger | 0.96 | 0.89 | 0.92 | 275 |
| Fear | 0.84 | 0.86 | 0.85 | 224 |
| Surprise | 0.63 | 0.89 | 0.74 | 66 |
| **Weighted Average** | **0.93** | **0.92** | **0.92** | **2,000** |

---

## Tech Stack

- **Machine Learning**: TensorFlow, Keras, NumPy, Scikit-learn
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Cloud Deployment**: Vercel (Frontend static hosting) + Render (FastAPI Python web service)

---

## Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/The-Ghost23/Emotion-Predictor.git
cd Emotion-Predictor
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Open the application
Visit `http://127.0.0.1:8000` in your browser.

---

## API Usage

### Health Check
```bash
curl https://emotion-predictor-9mn2.onrender.com/health
```

**Response**:
```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

### Predict Emotion
```bash
curl -X POST "https://emotion-predictor-9mn2.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy and excited about this project!"}'
```

**Response**:
```json
{
  "text": "I feel so happy and excited about this project!",
  "predicted_emotion": "joy",
  "confidence": 0.996,
  "all_probabilities": {
    "sadness": 0.00027,
    "joy": 0.99603,
    "love": 0.00316,
    "anger": 0.00005,
    "fear": 0.00024,
    "surprise": 0.00023
  }
}
```

---

## Project Structure

```
├── Frontend/               # HTML, CSS, and JS web interface
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
├── Models/                 # Trained model weights and tokenizer
│   ├── BiGRU_model.keras
│   └── tokenizer.pkl
├── Emotion.ipynb           # Model training and evaluation notebook
├── main.py                 # FastAPI backend application
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development/notebook dependencies
├── vercel.json             # Vercel deployment config
├── render.yaml             # Render deployment config
└── README.md
```

---

## Author

- **Dipyaman Bhowmick** - [GitHub Profile](https://github.com/The-Ghost23)

---

## License

This project is licensed under the MIT License.
