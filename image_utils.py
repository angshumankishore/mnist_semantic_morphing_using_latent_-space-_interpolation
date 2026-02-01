import io
import cv2
import numpy as np
import torch

# Use same device as model file
DEVICE = torch.device("cpu")


def _imdecode_filebytes(filebytes):
    arr = np.frombuffer(filebytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _split_digits_from_image(img):
    """Return a list of cropped grayscale images (numpy arrays) representing
    individual digit regions. Tries contours first; if only one contour is
    found and the image is wide, it attempts a vertical split by a valley in
    the vertical projection."""
    if img is None:
        return []

    # Preprocess: blur and binarize. We expect dark-on-light scans; adapt by
    # inverting as needed later.
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so digits are white on black for contouring
    inv = 255 - th

    # Morphological open to remove small specks, then close to join broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    min_area = max(30, int(0.001 * h * w))

    # keep only reasonably sized contours
    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area:
            continue
        boxes.append((x, y, cw, ch))

    if len(boxes) >= 2:
        boxes = sorted(boxes, key=lambda b: b[0])
        crops = []
        for (x, y, cw, ch) in boxes:
            pad = int(0.1 * max(cw, ch))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + cw + pad)
            y1 = min(h, y + ch + pad)
            crops.append(img[y0:y1, x0:x1])
        return crops

    # fallback: if image is significantly wider than tall, attempt a vertical
    # split by finding a low-sum column (valley) in the cleaned image
    if w >= max(2 * h // 3, h * 1.2):
        colsum = np.sum(clean, axis=0)
        left = w // 8
        right = w - left
        if right - left <= 0:
            mid = w // 2
        else:
            valley_idx = left + int(np.argmin(colsum[left:right]))
            mid = int(valley_idx)
        left_img = img[:, :mid]
        right_img = img[:, mid:]
        return [left_img, right_img]

    # otherwise return the whole image as single piece
    return [img]


def _to_vae_tensor(img):
    # invert so digit ink is bright, resize to 28x28 and normalize
    img = 255 - img
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return t.to(DEVICE)


def load_user_image(path_or_file_or_digit):
    """Accepts one of:
    - single digit character ('0'–'9') as str -> loads from digits/<d>.png
    - path to an image file (str)
    - a file-like object with .read() (e.g., Streamlit uploaded file)

    Returns a list of 1-or-more torch tensors shaped [1,1,28,28]. If the
    provided image contains multiple digits (e.g., "12" combined), this
    attempts to split it and returns tensors for each digit in left-to-right
    order."""
    img = None

    if isinstance(path_or_file_or_digit, str):
        s = path_or_file_or_digit
        if len(s) == 1 and s.isdigit():
            path = f"digits/{s}.png"
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    else:
        # assume file-like object (Streamlit's UploadedFile)
        try:
            path_or_file_or_digit.seek(0)
        except Exception:
            pass
        filebytes = path_or_file_or_digit.read()
        img = _imdecode_filebytes(filebytes)

    if img is None:
        raise ValueError("Invalid image or path provided to load_user_image")

    pieces = _split_digits_from_image(img)
    tensors = [_to_vae_tensor(p) for p in pieces if p is not None and p.size > 0]
    return tensors


def load_images_from_file_list(file_list):
    """Given a list of file-like objects or paths, return a flat list of
    tensors for all detected digits (keeps order of files and left-to-right
    within files)."""
    out = []
    for f in file_list:
        if f is None:
            continue
        tensors = load_user_image(f)
        out.extend(tensors)
    return out
