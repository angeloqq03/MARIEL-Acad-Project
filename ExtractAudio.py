import subprocess
import os
import logging

logger = logging.getLogger(__name__)

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