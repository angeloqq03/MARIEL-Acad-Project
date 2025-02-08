import cv2
import os

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