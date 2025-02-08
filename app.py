from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFCLIPModel,
    CLIPProcessor,
    pipeline,
    BertTokenizer,
    BertForSequenceClassification
)
import cv2
import os
import subprocess
import torch
from PIL import Image
import numpy as np
import base64
import uuid
from ultralytics import YOLO
import tensorflow as tf
import logging
from Home import home
from ExtractText import extract_text
from ServeAudio import serve_audio
from UploadFile import upload_file
from ExtractFrames import extract_frames
from ExtractAudio import extract_audio
from ProcessAudio import process_audio
from AnalyzeText import analyze_text_content
from AnalyzeBatch import analyze_batch
from GetRecommendations import get_recommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

os.makedirs('save', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('unsafe_frames', exist_ok=True)
os.makedirs('audio', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('text_output', exist_ok=True)

print("Loading /Models...")
try:
    nm = YOLO("/Models/nudenet/320n.pt")

    bt = BertTokenizer.from_pretrained('bert-base-uncased')
    bm = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    pm = AutoModelForSequenceClassification.from_pretrained(
        "/Models/profanity_model_3.1",
        local_files_only=True
    )
    pt = AutoTokenizer.from_pretrained(
        "/Models/profanity_model_3.1",
        local_files_only=True
    )

    hm = AutoModelForSequenceClassification.from_pretrained(
        "/Models/dehatebert-mono-english",
        local_files_only=True
    )
    ht = AutoTokenizer.from_pretrained(
        "/Models/dehatebert-mono-english",
        local_files_only=True
    )

    cm = TFCLIPModel.from_pretrained(
        "/Models/clip-vit-base-patch32",
        local_files_only=True
    )
    cp = CLIPProcessor.from_pretrained(
        "/Models/clip-vit-base-patch32",
        local_files_only=True
    )

    wm = pipeline(
        "automatic-speech-recognition",
        model="/Models/whisper-tiny",
        chunk_length_s=30,
        batch_size=8
    )

    print("All /Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading /Models: {str(e)}")
    raise

@app.route("/")
def home_route():
    return home()

@app.route("/extract_text", methods=["POST"])
def extract_text_route():
    return extract_text()

@app.route('/audio/<path:filename>')
def serve_audio_route(filename):
    return serve_audio(filename)

@app.route("/upload", methods=["POST"])
def upload_file_route():
    return upload_file()

if __name__ == "__main__":
    app.run(port=5050, debug=True)