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