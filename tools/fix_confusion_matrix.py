# tools/fix_confusion_matrix.py
# Image-only fix (NO rerun): tight crop + upscale + remove right colorbar strip.
# Output: figs/confusion_matrix_tight.png

import os
from PIL import Image, ImageChops

def autocrop_whitespace(im, bg=(255, 255, 255), tol=8):
    im = im.convert("RGB")
    bg_im = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    diff = ImageChops.add(diff, diff, 2.0, -tol)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def upscale(im, factor=3):
    w, h = im.size
    return im.resize((w * factor, h * factor), resample=Image.Resampling.LANCZOS)

def crop_right_colorbar(im, keep_frac=0.88):
    # trims the right strip where the colorbar usually lives
    w, h = im.size
    return im.crop((0, 0, int(w * keep_frac), h))

def main():
    in_path = os.path.join("figs", "confusion_matrix.png")
    out_path = os.path.join("figs", "confusion_matrix_tight.png")

    im = Image.open(in_path).convert("RGB")

    im = autocrop_whitespace(im, tol=10)   # remove outer margins
    im = upscale(im, factor=3)             # make text larger in the final raster
    im = autocrop_whitespace(im, tol=10)   # trim again after upscale

    # remove colorbar region (adjust if needed: 0.86–0.92 typical)
    im = crop_right_colorbar(im, keep_frac=0.88)
    im = autocrop_whitespace(im, tol=10)

    im.save(out_path, dpi=(300, 300))
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
