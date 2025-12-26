import argparse, yaml, cv2, numpy as np, os

POINTS_NEEDED = 4  # corners on a plane (e.g., front wall or floor)

def click_points(img):
    pts = []
    clone = img.copy()
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < POINTS_NEEDED:
            pts.append((x,y))
            cv2.circle(clone, (x,y), 5, (0,255,0), -1)
            cv2.imshow("click", clone)
    cv2.imshow("click", clone)
    cv2.setMouseCallback("click", cb)
    while len(pts) < POINTS_NEEDED:
        if cv2.waitKey(20) & 0xFF == 27: break
    cv2.destroyAllWindows()
    return np.array(pts, dtype=np.float32)

def main(args):
    img = cv2.imread(args.image)
    assert img is not None, f"Cannot read {args.image}"

    print("Click 4 corners in order (clockwise).")
    px_pts = click_points(img)

    # Real-world target plane coordinates (e.g., front wall rectangle in cm)
    # Replace with your court dimensions later:
    W, H = 640, 480  # placeholder target rectangle
    world_pts = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)

    Hmat, _ = cv2.findHomography(px_pts, world_pts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    yaml.safe_dump({"homography": Hmat.tolist(), "target_size":[int(W),int(H)]}, open(args.out, "w"))
    print("Saved:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="configs/calib.example.yaml")
    args = p.parse_args()
    main(args)
