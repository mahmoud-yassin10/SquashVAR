import cv2, numpy as np
from src.geom.coords import CourtGeometry, speed_cm_per_s

def probe_video(path, fps_override=None):
    cap = cv2.VideoCapture(path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30
    cg = CourtGeometry("configs/calib.frontwall.yaml","configs/calib.floor.yaml")

    prev_floor = None; prev_wall = None
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        # TODO: replace with your detector; here we fake a point to prove plumbing:
        h,w = frame.shape[:2]
        fake_px = (w//2, h//2)   # replace with (x,y) from detector

        if cg.is_on_wall(fake_px):
            cm = cg.px_to_wall(fake_px)
            if prev_wall is not None:
                v = speed_cm_per_s(prev_wall, cm, fps)
                # print or log v
            prev_wall = cm; prev_floor = None
        elif cg.is_on_floor(fake_px):   
            cm = cg.px_to_floor(fake_px)
            if prev_floor is not None:
                v = speed_cm_per_s(prev_floor, cm, fps)
            prev_floor = cm; prev_wall = None
        else:
            prev_floor = prev_wall = None

        t += 1
    cap.release()
