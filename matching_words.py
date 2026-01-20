import re
import cv2
import pytesseract
from difflib import SequenceMatcher


#File paths
image_path = "input file path"
out_path = "output file path"

# Colors for boxes
GREEN = (0, 255, 0)
RED   = (0, 0, 255)



# Preprocess image for better OCR
def preprocess(gray_up):
    # Contrast boost for faint text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_up = clahe.apply(gray_up)

    # Light denoise
    gray_up = cv2.bilateralFilter(gray_up, 9, 75, 75)

    # Binarize
    bw = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return bw

def normalize_word(w: str) -> str:
    # Lowercase + strip punctuation; keep letters/numbers
    w = w.lower()
    w = re.sub(r"[^a-z0-9]+", "", w)
    return w



 # OCR with bounding boxes
def ocr_words_with_boxes(image_path: str, scale: int = 3, psm: int = 11, conf_thresh: float = 10.0):
    """
    Returns:
      draw_img_up: upscaled color image (for drawing)
      orig_shape:  original image shape (h, w)
      words:       list of dicts: {raw, norm, conf, box=(x,y,w,h)}
    """
    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = orig.shape[:2]

    # Upscale for consistent OCR + drawing coordinates
    draw_img_up = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    bw = preprocess(gray_up)

    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        bw,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    out = []
    n = len(data["text"])
    for i in range(n):
        raw = data["text"][i].strip()
        if not raw:
            continue

        conf_str = data["conf"][i]
        try:
            conf = float(conf_str)
        except ValueError:
            continue

        if conf < conf_thresh:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        norm = normalize_word(raw)
        if not norm:
            continue

        out.append({
            "raw": raw,
            "norm": norm,
            "conf": conf,
            "box": (x, y, w, h)
        })

    return draw_img_up, (orig_h, orig_w), out

def label_matches(words_a, words_b):
    """
    Uses sequence alignment on normalized words.
    Returns:
      labels_a: list of "same" or "diff" for each word token in A
      labels_b: list of "same" or "diff" for each word token in B
    """
    seq_a = [w["norm"] for w in words_a]
    seq_b = [w["norm"] for w in words_b]

    labels_a = ["diff"] * len(seq_a)
    labels_b = ["diff"] * len(seq_b)

    sm = SequenceMatcher(a=seq_a, b=seq_b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                labels_a[i] = "same"
            for j in range(j1, j2):
                labels_b[j] = "same"
        # replace/insert/delete remain "diff"

    return labels_a, labels_b

def draw_labeled_boxes(draw_img_up, words, labels):
    for w, lab in zip(words, labels):
        x, y, ww, hh = w["box"]
        color = GREEN if lab == "same" else RED
        cv2.rectangle(draw_img_up, (x, y), (x + ww, y + hh), color, 2)
    return draw_img_up

def compare_and_annotate(img_a_path, img_b_path,
                         out_a="annotated_A.png", out_b="annotated_B.png",
                         scale=3, psm=11, conf_thresh=10.0):
    draw_a, shape_a, words_a = ocr_words_with_boxes(img_a_path, scale=scale, psm=psm, conf_thresh=conf_thresh)
    draw_b, shape_b, words_b = ocr_words_with_boxes(img_b_path, scale=scale, psm=psm, conf_thresh=conf_thresh)

    labels_a, labels_b = label_matches(words_a, words_b)

    draw_a = draw_labeled_boxes(draw_a, words_a, labels_a)
    draw_b = draw_labeled_boxes(draw_b, words_b, labels_b)

    # Downscale back to original sizes
    a_h, a_w = shape_a
    b_h, b_w = shape_b
    final_a = cv2.resize(draw_a, (a_w, a_h), interpolation=cv2.INTER_AREA)
    final_b = cv2.resize(draw_b, (b_w, b_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(out_a, final_a)
    cv2.imwrite(out_b, final_b)

    same_a = sum(1 for x in labels_a if x == "same")
    diff_a = len(labels_a) - same_a
    same_b = sum(1 for x in labels_b if x == "same")
    diff_b = len(labels_b) - same_b

    print(f"A: {same_a} same (green), {diff_a} different (red) -> {out_a}")
    print(f"B: {same_b} same (green), {diff_b} different (red) -> {out_b}")

if __name__ == "__main__":
    compare_and_annotate(
        "test.png",
        "test2.png",
        out_a="image1_compared.png",
        out_b="image2_compared.png",
        scale=3,
        psm=11,
        conf_thresh=10.0
    )
