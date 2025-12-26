# tools/plot_detector_operating_panel.py
# Option 1 (NO rerun): crop off outside legends to kill whitespace, then upscale + big A–D labels.
# Output: figs/detector_operating_panel.png

import os
from PIL import Image, ImageChops, ImageDraw, ImageFont

def autocrop_whitespace(im, bg=(255, 255, 255), tol=8):
    """Crop near-white borders around content."""
    im = im.convert("RGB")
    bg_im = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    diff = ImageChops.add(diff, diff, 2.0, -tol)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def upscale(im, factor=3):
    """High-quality upscale."""
    w, h = im.size
    return im.resize((w * factor, h * factor), resample=Image.Resampling.LANCZOS)

def crop_right_legend(im, keep_frac=0.84):
    """
    Remove right-side legend area (common in Ultralytics curves).
    keep_frac=0.84 keeps the left 84% of the image width.
    """
    w, h = im.size
    return im.crop((0, 0, int(w * keep_frac), h))

def add_panel_label(im, label, pad=18, font_px=72):
    """Add large (A) label with a white background box."""
    im = im.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
    except:
        font = ImageFont.load_default()

    text = f"({label})"
    x, y = pad, pad
    bb = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bb[0]-10, bb[1]-6, bb[2]+10, bb[3]+6], fill="white")
    draw.text((x, y), text, fill="black", font=font)
    return im

def pad_to(im, W, H, bg=(255, 255, 255)):
    canvas = Image.new("RGB", (W, H), bg)
    x = (W - im.size[0]) // 2
    y = (H - im.size[1]) // 2
    canvas.paste(im, (x, y))
    return canvas

def main():
    in_dir = "figs"
    src = [
        ("A", "PR_curve.png"),
        ("B", "F1_curve.png"),
        ("C", "P_curve.png"),
        ("D", "R_curve.png"),
    ]

    panels = []
    for lab, fn in src:
        p = os.path.join(in_dir, fn)
        im = Image.open(p).convert("RGB")

        # 1) remove outer whitespace
        im = autocrop_whitespace(im, tol=10)

        # 2) upscale to make tiny text larger
        im = upscale(im, factor=3)

        # 3) remove whitespace again after upscale
        im = autocrop_whitespace(im, tol=10)

        # 4) aggressively remove right-side legend area
        im = crop_right_legend(im, keep_frac=0.69)

        # 5) final trim (clean edges)
        im = autocrop_whitespace(im, tol=10)

        # 6) add big panel label
        im = add_panel_label(im, lab, font_px=72)

        panels.append(im)

    # Uniform sizes
    W = max(im.size[0] for im in panels)
    H = max(im.size[1] for im in panels)
    panels = [pad_to(im, W, H) for im in panels]

    # 2x2 grid
    gap = 26
    outW = 2 * W + gap
    outH = 2 * H + gap
    out = Image.new("RGB", (outW, outH), (255, 255, 255))

    out.paste(panels[0], (0, 0))
    out.paste(panels[1], (W + gap, 0))
    out.paste(panels[2], (0, H + gap))
    out.paste(panels[3], (W + gap, H + gap))

    out_path = os.path.join(in_dir, "detector_operating_panel.png")
    out.save(out_path, dpi=(300, 300))
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
