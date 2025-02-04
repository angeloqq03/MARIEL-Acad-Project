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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

os.makedirs('save', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('unsafe_frames', exist_ok=True)
os.makedirs('audio', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('text_output', exist_ok=True)

print("Loading models...")
try:
    # Load models
    nm = YOLO("Models/nudenet/320n.pt")

    bt = BertTokenizer.from_pretrained('bert-base-uncased')
    bm = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    pm = AutoModelForSequenceClassification.from_pretrained(
        "Models/profanity_model_3.1",
        local_files_only=True
    )
    pt = AutoTokenizer.from_pretrained(
        "Models/profanity_model_3.1",
        local_files_only=True
    )

    hm = AutoModelForSequenceClassification.from_pretrained(
        "Models/dehatebert-mono-english",
        local_files_only=True
    )
    ht = AutoTokenizer.from_pretrained(
        "Models/dehatebert-mono-english",
        local_files_only=True
    )

    cm = TFCLIPModel.from_pretrained(
        "Models/clip-vit-base-patch32",
        local_files_only=True
    )
    cp = CLIPProcessor.from_pretrained(
        "Models/clip-vit-base-patch32",
        local_files_only=True
    )

    wm = pipeline(
        "automatic-speech-recognition",
        model="Models/whisper-tiny",
        chunk_length_s=30,
        batch_size=8
    )

    print("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/extract_text", methods=["POST"])
def extract_text():
    try:
        af = request.form.get('audio_file')
        if not af:
            return jsonify({"error": "No audio file specified"}), 400

        ap = os.path.join('audio', af)
        if not os.path.exists(ap):
            return jsonify({"error": "Audio file not found"}), 404
        ar = process_audio(ap)
        if not ar['success']:
            return jsonify({"error": ar['error']}), 500
        tf = f"text_{uuid.uuid4().hex}.txt"
        tp = os.path.join('text_output', tf)
        with open(tp, 'w', encoding='utf-8') as f:
            f.write(ar['text'])
        ta = analyze_text_content(ar['text'])
        return jsonify({
            "success": True,
            "text": ar['text'],
            "text_file": tf,
            "confidence": ar['confidence'],
            "analysis": ta
        })
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('audio', filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        v = request.files['file']
        if v.filename == '':
            return jsonify({"error": "No file selected"}), 400
        vp = os.path.join('save', v.filename)
        v.save(vp)
        try:
            fs = extract_frames(vp)
            rs = []
            afn = f"audio_{uuid.uuid4().hex}.wav"
            ap = os.path.join('audio', afn)
            ar = extract_audio(vp, ap)
            if ar:
                at = process_audio(ap)
                tc = at.get('text', '')
                if tc:
                    tf = f"text_{uuid.uuid4().hex}.txt"
                    tp = os.path.join('text_output', tf)
                    with open(tp, 'w', encoding='utf-8') as f:
                        f.write(tc)
                    ta = analyze_text_content(tc)
                else:
                    tf = None
                    ta = None
            else:
                tc = ''
                tf = None
                ta = None
            bs = 15
            for i in range(0, len(fs), bs):
                bf = fs[i:i + bs]
                r = analyze_batch(bf, tc)
                if r is None:
                    continue
                rs.extend(r)
                for fd in bf:
                    if fd.get('is_inappropriate', False) or fd.get('is_harmful', False):
                        ufn = f'unsafe_{uuid.uuid4().hex}.png'
                        ufp = os.path.join('unsafe_frames', ufn)
                        os.rename(fd['frame'], ufp)
                    else:
                        os.remove(fd['frame'])
                    os.remove(fd['thumbnail'])
            if os.path.exists(vp):
                os.remove(vp)
            if rs:
                tms = sum(r['meta_standards']['score'] for r in rs) / len(rs)
                oa = {
                    "total_score": tms,
                    "risk_level": "High" if tms > 35 else "Medium" if tms > 30 else "Low",
                    "recommendation": get_recommendation(tms)
                }
            else:
                oa = {
                    "total_score": 0,
                    "risk_level": "Low",
                    "recommendation": "No issues detected"
                }
            return jsonify({
                "success": True,
                "results": rs,
                "audio_path": afn,
                "audio_text": tc,
                "text_file": tf,
                "text_analysis": ta,
                "overall_assessment": oa
            })
        except Exception as e:
            if os.path.exists(vp):
                os.remove(vp)
            logger.error(f"Error in content analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_frames(vp):
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    fs = []
    fc = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fc % fps == 0:
            fp = os.path.join('temp', f'frame_{fc}.jpg')
            tp = os.path.join('temp', f'thumb_{fc}.jpg')
            cv2.imwrite(fp, frame)
            thumbnail = cv2.resize(frame, (648, 648))
            cv2.imwrite(tp, thumbnail)
            fs.append({
                'frame': fp,
                'thumbnail': tp,
                'timestamp': fc // fps
            })
        fc += 1
    cap.release()
    return fs

def extract_audio(vp, op):
    try:
        cmd = [
            'ffmpeg',
            '-i', vp,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            op
        ]
        result = subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        if os.path.exists(op) and os.path.getsize(op) > 0:
            logger.info(f"Audio extracted successfully: {op}")
            return op
        else:
            raise Exception("Audio extraction failed - empty or missing file")

    except Exception as e:
        logger.error(f"Audio extraction error: {str(e)}")
        return None

def process_audio(ap):
    try:
        if not os.path.exists(ap):
            logger.error(f"Audio file not found: {ap}")
            return {
                'success': False,
                'text': "Audio file not found",
                'error': "File not found"
            }
        logger.info(f"Processing audio file: {ap}")
        wr = wm(
            ap,
            return_timestamps=True,
            generate_kwargs={'task': 'transcribe', 'language': 'english'}
        )
        logger.info(f"Whisper result: {wr}")
        if not wr.get('text'):
            logger.error("Whisper failed to extract text")
            return {
                'success': False,
                'text': "Whisper failed to extract text",
                'error': "No text found in Whisper output"
            }
        text = wr['text']
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        pc = []
        for chunk in chunks:
            inputs = bt(chunk, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(torch.device('cpu')) for key, value in inputs.items()}  
            with torch.no_grad():
                outputs = bm(**inputs)
            pc.append(bt.decode(
                inputs['input_ids'][0],
                skip_special_tokens=True
            ))
        final_text = " ".join(pc)
        return {
            'success': True,
            'text': final_text,
            'confidence': wr.get('confidence', 0)
        }
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return {
            'success': False,
            'text': "Audio processing failed",
            'error': str(e)
        }

def analyze_text_content(text):
    try:
        pi = pt(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            po = pm(**pi)
            ps = torch.nn.functional.softmax(po.logits, dim=-1)
        hi = ht(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            ho = hm(**hi)
            hs = torch.nn.functional.softmax(ho.logits, dim=-1)
        return {
            "profanity": {
                "score": float(ps[0][1]) * 100,
                "is_offensive": float(ps[0][1]) > 0.5
            },
            "hate_speech": {
                "score": float(hs[0][1]) * 100,
                "is_hateful": float(hs[0][1]) > 0.5
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return None

def analyze_batch(bf, text):
    try:
        rs = []
        imgs = []
        ts = []

        for fd in bf:
            img = Image.open(fd['frame'])
            img = img.resize((128, 128))
            imgs.append(img)
            ts.append(fd['timestamp'])
        ia = np.array([np.array(img) / 255.0 for img in imgs])
        it = torch.tensor(ia).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            nr = nm(it)
            np = [result.boxes for result in nr]
        if text:
            pi = pt(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                po = pm(**pi)
                ps = torch.nn.functional.softmax(po.logits, dim=-1)
            hi = ht(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                ho = hm(**hi)
                hs = torch.nn.functional.softmax(ho.logits, dim=-1)
        else:
            ps = torch.zeros(1, 2)
            hs = torch.zeros(1, 2)

        ci = cp(text=[text if text else ""] * len(imgs), images=imgs, return_tensors="tf", padding=True)
        co = cm(**ci)
        cs = tf.nn.softmax(co.logits_per_image, axis=-1).numpy()
        for i in range(len(bf)):
            with open(bf[i]['thumbnail'], 'rb') as img_file:
                tb64 = base64.b64encode(img_file.read()).decode('utf-8')
            ns = float(np[i].conf[0]) * 99 if np[i] else 0.0
            ps = float(ps[0][1]) * 99
            hs = float(hs[0][1]) * 99
            hs = float(cs[i][1]) * 99 if cs[i].size > 1 else 0.0
            ms = (
                (ns * 0.6) +
                (hs * 0.1) +
                (hs * 0.1) +
                (ps * 0.1)
            )
            rs.append({
                "nudity": {
                    "score": ns,
                    "is_inappropriate": ns > 65
                },
                "profanity": {
                    "score": ps,
                    "is_offensive": ps > 65
                },
                "hate_speech": {
                    "score": hs,
                    "is_hateful": hs > 40
                },
                "harm": {
                    "score": hs,
                    "is_harmful": hs > 40
                },
                "meta_standards": {
                    "score": ms,
                    "is_violating": ms > 30,
                    "risk_level": "High" if ms > 60 else "Medium" if ms > 25 else "Low",
                    "recommendation": get_recommendation(ms)
                },
                "thumbnail": tb64,
                "timestamp": ts[i]
            })
        return rs
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return None

def get_recommendation(score):
    if score > 70:
        return "Content likely violates Meta Community Standards. Major modifications needed."
    elif score > 30:
        return "Content may need modifications to comply with Meta Community Standards."
    else:
        return "Content likely complies with Meta Community Standards."

if __name__ == "__main__":
    app.run(port=5050, debug=True)