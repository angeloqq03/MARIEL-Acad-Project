import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

pm = AutoModelForSequenceClassification.from_pretrained("Models/profanity_model_3.1", local_files_only=True)
pt = AutoTokenizer.from_pretrained("Models/profanity_model_3.1", local_files_only=True)
hm = AutoModelForSequenceClassification.from_pretrained("Models/dehatebert-mono-english", local_files_only=True)
ht = AutoTokenizer.from_pretrained("Models/dehatebert-mono-english", local_files_only=True)

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