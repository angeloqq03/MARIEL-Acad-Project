from flask import request, jsonify
import os
import uuid
from ExtractFrames import extract_frames
from ExtractAudio import extract_audio
from ProcessAudio import process_audio
from AnalyzeBatch import analyze_batch
from AnalyzeText import analyze_text_content
from GetRecommendations import get_recommendation
import logging

logger = logging.getLogger(__name__)

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