import re
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import os

# -------------------------
# CONFIG
# -------------------------
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
GREEN = (0, 255, 0)

# -------------------------
# Image / PDF loader
# -------------------------
def load_image(path):
    ext = os.path.splitext(path.lower())[1]

    if ext == ".pdf":
        pages = convert_from_path(
            path,
            poppler_path=POPPLER_PATH,
            dpi=300
        )
        pil_img = pages[0]  # first page only
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

# -------------------------
# Preprocess
# -------------------------
def preprocess(gray_up):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_up = clahe.apply(gray_up)
    gray_up = cv2.bilateralFilter(gray_up, 9, 75, 75)
    bw = cv2.threshold(
        gray_up, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    return bw

def normalize_word(w: str) -> str:
    w = w.lower()
    return re.sub(r"[^a-z0-9]+", "", w)

# -------------------------
# OCR with bounding boxes
# -------------------------
def ocr_words_with_boxes(img, scale=3, psm=11, conf_thresh=10.0):

    orig_h, orig_w = img.shape[:2]

    draw_img_up = cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_up = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    bw = preprocess(gray_up)

    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        bw, config=config, output_type=pytesseract.Output.DICT
    )

    words = []
    for i in range(len(data["text"])):
        raw = data["text"][i].strip()
        if not raw:
            continue

        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue

        if conf < conf_thresh:
            continue

        norm = normalize_word(raw)
        if not norm:
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        words.append({
            "raw": raw,
            "norm": norm,
            "conf": conf,
            "box": (x, y, w, h)
        })

    return draw_img_up, (orig_h, orig_w), words

# -------------------------
# Name matching
# -------------------------
def match_names(words_source, words_target):
    source_set = {w["norm"] for w in words_source}
    return [w["norm"] in source_set for w in words_target]

# -------------------------
# Draw green boxes
# -------------------------
def draw_green_boxes(draw_img_up, words, matches):
    for w, match in zip(words, matches):
        if not match:
            continue

        x, y, ww, hh = w["box"]
        cv2.rectangle(
            draw_img_up,
            (x, y),
            (x + ww, y + hh),
            GREEN,
            2
        )
    return draw_img_up

# -------------------------
# Main
# -------------------------
def highlight_names(
    source_pdf,
    target_pdf,
    output_image="highlighted_output.png",
    scale=3,
    psm=11,
    conf_thresh=10.0
):

    # OCR source (names only)
    img_src = load_image(source_pdf)
    _, _, words_src = ocr_words_with_boxes(
        img_src, scale, psm, conf_thresh
    )

    # OCR target (draw here)
    img_tgt = load_image(target_pdf)
    draw_tgt, shape_tgt, words_tgt = ocr_words_with_boxes(
        img_tgt, scale, psm, conf_thresh
    )

    matches = match_names(words_src, words_tgt)
    draw_tgt = draw_green_boxes(draw_tgt, words_tgt, matches)

    h, w = shape_tgt
    final_img = cv2.resize(draw_tgt, (w, h))

    cv2.imwrite(output_image, final_img)
    print(f"Saved: {output_image}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    highlight_names(
        source_pdf="baylay_test.pdf",
        target_pdf="ballot_test.pdf",
        output_image="dead_pres_highlighted.png"
    )
