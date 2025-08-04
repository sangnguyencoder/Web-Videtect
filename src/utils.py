import cv2
import numpy as np

FRAMES_PER_CLIP = 16
IMG_SIZE = 112


def video_to_tensor(video_path, frames=FRAMES_PER_CLIP, size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        cap.release()
        raise ValueError(f"Không đọc được số frame từ video: {video_path}")
    frame_idxs = np.linspace(0, frame_count - 1, frames).astype(int)
    frames_list = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames_list.append(frame)
    cap.release()
    if len(frames_list) < frames:
        for _ in range(frames - len(frames_list)):
            frames_list.append(frames_list[-1])
    return np.array(frames_list)
