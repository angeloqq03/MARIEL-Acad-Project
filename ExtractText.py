from flask import request, jsonify
import os
import uuid
from ProcessAudio import process_audio
from AnalyzeText import analyze_text_content
import logging

logger = logging.getLogger(__name__)

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