import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import logging
from transformers import CLIPProcessor, TFCLIPModel
from GetRecommendations import get_recommendation

logger = logging.getLogger(__name__)

nm = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
cm = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

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