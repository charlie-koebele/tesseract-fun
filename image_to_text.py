import cv2
import pytesseract

path = "path of image"

def preprocess_for_ocr(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Upscale (helps small/thin text)
    scale = 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2) Boost local contrast (excellent for faint gray text)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) Light denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4) Binarize (try Otsu)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return bw

def ocr_text(path: str) -> str:
    bw = preprocess_for_ocr(path)

    # psm 11 often works best for separated lines
    config = "--oem 3 --psm 11 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(bw, lang="eng", config=config)

    return text

if __name__ == "__main__":
    print(ocr_text("your_image.png"))
