import os
import torch
import logging
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

logger = logging.getLogger(__name__)

bt = BertTokenizer.from_pretrained('bert-base-uncased')
bm = BertForSequenceClassification.from_pretrained('bert-base-uncased')
wm = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

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