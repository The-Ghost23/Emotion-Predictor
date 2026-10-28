import os
os.environ.setdefault("KERAS_HOME", "/tmp/keras")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel,Field
from keras.models import load_model
import numpy as np
import pickle
import re
import os
import uvicorn

model_path = "Models/BiGRU_model.keras"
tokenizer_path = "Models/tokenizer.pkl"

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

emotion_emojis = {
    "sadness": "😢",
    "joy": "😁",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
}

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'"," ",text)
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

class textinput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description='The Sentence to analyze the emotion',
        json_schema_extra={
            "example": "I am so anxious today"
        }
    )

class predictionresponse(BaseModel):
    text: str = ""
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str,float]

class healthresponse(BaseModel):
    status: str
    model_loaded: bool

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Model and Tokenizer...")
    ml_model["BiGRU"] = load_model(model_path)          #BiGru Model
    with open(tokenizer_path,'rb') as file:             #Tokenizer
        ml_model["tokenizer"] = pickle.load(file)
    print("Model Loaded Successfully!")

    yield

    ml_model.clear()

app = FastAPI(
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/Frontend',StaticFiles(directory='Frontend'),name='Frontend')
app.mount('/assets',StaticFiles(directory='Frontend/assets'),name='assets')

@app.get('/',include_in_schema=False)
def server_ui():
    return FileResponse('Frontend/index.html')

@app.get('/style.css', include_in_schema=False)
def serve_css():
    return FileResponse('Frontend/style.css', media_type='text/css')

@app.get('/script.js', include_in_schema=False)
def serve_js():
    return FileResponse('Frontend/script.js', media_type='application/javascript')

@app.get('/health',response_model=healthresponse)

def health_check():
    return healthresponse(status='Server is running',model_loaded=bool(ml_model))

@app.post('/predict',response_model=predictionresponse)
def predict_emotion(text_input: textinput):
    #1. Cleans the input sentences
    BiGRU_model = ml_model.get("BiGRU")
    tokenizer_model  = ml_model.get("tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503,detail="Model not loaded")

    cleaned_text = preprocess_text(text_input.text)
    
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(tokenized_text,maxlen=50,padding='post',truncating='post')

    probabilities = BiGRU_model.predict(padded_sequence)[0] 
    top_emotion_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, emotion_labels)
    }

    return predictionresponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]),
        all_probabilities = all_probabilities,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


    
    