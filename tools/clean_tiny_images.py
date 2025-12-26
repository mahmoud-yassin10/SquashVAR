import os, glob, cv2, sys

def purge(root, min_side=10):
    # root should be like /workspace/data/roboflow
    splits = ["train", "valid", "val", "test"]
    removed = 0
    for sp in splits:
        imgd = os.path.join(root, sp, "images")
        lbld = os.path.join(root, sp, "labels")
        if not os.path.isdir(imgd): continue
        for p in glob.glob(os.path.join(imgd, "*.jpg")) + glob.glob(os.path.join(imgd, "*.png")):
            img = cv2.imread(p)
            if img is None: 
                continue    
            h, w = img.shape[:2]
            if h < min_side or w < min_side:
                lbl = os.path.join(lbld, os.path.splitext(os.path.basename(p))[0] + ".txt")
                try: os.remove(p); removed += 1
                except: pass
                try: os.remove(lbl)
                except: pass
    print(f"Removed {removed} tiny images/labels (<{min_side}px).")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/roboflow"
    purge(root)
