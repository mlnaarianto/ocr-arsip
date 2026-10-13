"""
ocr_engine.py  --  TAHAP 1 + 2: halaman -> baris -> teks

Pakai:
    python ocr_engine.py contoh_surat\\pages\\surat01.png
    python ocr_engine.py contoh_surat\\pages\\surat01.pdf
    python ocr_engine.py file.png --model crnn_id_best.pth

Hasil:
    teks hasil OCR dicetak di layar (dan disimpan di debug\\<nama>.ocr.txt)
    Kalau ada file <nama>.txt di folder yang sama (teks yang benar), CER dihitung otomatis.
    debug\\<nama>_p1_boxes.png   halaman dengan kotak hijau di tiap baris yang ditemukan
    debug\\<nama>_p1_lines\\      potongan tiap baris

Fungsi untuk dipanggil FastAPI nanti:  run_ocr(path) -> {"text": ..., "pages": [...]}
"""
import os
import sys
import numpy as np
import cv2

DPI = 200   # resolusi saat PDF diubah jadi gambar (sama dengan surat contoh)


# ------------------------------------------------------------------ 1. buka file
def load_pages(path):
    """Kembalikan list gambar grayscale (satu per halaman)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pymupdf
        pages = []
        with pymupdf.open(path) as doc:
            for pg in doc:
                pix = pg.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                pages.append(arr.copy())
        return pages
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # aman untuk path Windows
    if img is None:
        raise ValueError("Tidak bisa membuka gambar: " + path)
    return [img]


# ------------------------------------------------------------------ 2. binarisasi
def binarize(gray):
    """Tulisan = 255 (putih), latar = 0. Tahan terhadap cahaya tidak rata."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 18)
    # buang bintik kecil (noise fotokopi)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    return (keep[lab] * 255).astype(np.uint8)


# ------------------------------------------------------------------ 3. luruskan
def estimate_skew(bw, max_deg=3.0, step=0.1):
    """Cari sudut putar yang membuat baris teks paling 'tajam' (proyeksi horizontal)."""
    small = cv2.resize(bw, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        s = np.var(r.sum(axis=1, dtype=np.float64))
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def deskew(gray, angle):
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ 4. potong baris
def remove_rules(bw):
    """Hapus garis horizontal panjang (garis kop, garis tabel) supaya tidak dianggap baris."""
    w = bw.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 6, 50), 1))
    rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    return cv2.subtract(bw, rules)


def remove_blobs(bw):
    """Hapus gumpalan yang jauh lebih tinggi dari huruf biasa (stempel, coretan tanda tangan)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return bw
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    limit = max(64, 1.9 * np.percentile(hs, 90))
    drop = np.zeros(n, dtype=bool)
    drop[1:] = hs > limit
    out = bw.copy()
    out[drop[lab]] = 0
    return out


def text_extent(strip, max_gap=160):
    """Batas kiri-kanan baris. Kelompok tulisan yang terpencil dan sangat sedikit (bintik) diabaikan."""
    ink = strip.sum(axis=0, dtype=np.float64) / 255.0
    cols = np.where(ink > 0)[0]
    if len(cols) == 0:
        return None
    groups, g0, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > max_gap:
            groups.append((g0, prev)); g0 = c
        prev = c
    groups.append((g0, prev))
    total = ink.sum()
    kept = [g for g in groups if ink[g[0]:g[1] + 1].sum() >= 0.08 * total]
    if not kept:
        return None
    return int(kept[0][0]), int(kept[-1][1]) + 1


# Perluasan batas baris ke arah huruf yang menempel stempel. DIMATIKAN (0): di tes surat contoh
# justru membawa goresan stempel ke dalam potongan dan menaikkan CER. Isi angka (mis. 70) untuk mencoba.
EXPAND_STAMP_PX = 0


def expand_extent(strip_raw, x0, x1, max_expand=70, gap=22):
    """Lebarkan batas baris ke arah tulisan yang masih menyambung (huruf yang tadi ikut terhapus
    karena menempel stempel). Dibatasi max_expand supaya coretan stempel tidak ikut terbawa."""
    ink = strip_raw.sum(axis=0) > 0
    n = len(ink)
    last, x = x0, x0 - 1
    while x >= max(x0 - max_expand, 0):
        if ink[x]:
            last = x
        elif last - x > gap:
            break
        x -= 1
    new0 = last
    last, x = x1 - 1, x1
    while x < min(x1 + max_expand, n):
        if ink[x]:
            last = x
        elif x - last > gap:
            break
        x += 1
    return int(new0), int(last) + 1


def find_lines(bw, min_h=10):
    """Kembalikan list (x0, y0, x1, y1) untuk tiap baris teks, berurutan atas ke bawah."""
    ruleless = remove_rules(bw)
    clean = remove_blobs(ruleless)
    ruleless[:, :8] = 0; ruleless[:, -8:] = 0
    clean[:, :8] = 0; clean[:, -8:] = 0; clean[:8, :] = 0; clean[-8:, :] = 0   # abaikan bayangan tepi kertas
    prof = clean.sum(axis=1, dtype=np.float64) / 255.0           # jumlah piksel tulisan per baris piksel
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    thr = max(2.0, 0.004 * clean.shape[1])                       # ambang: minimal beberapa piksel tulisan
    on = prof > thr

    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y)); start = None
    if start is not None:
        runs.append((start, len(on)))

    boxes = []
    for y0, y1 in runs:
        if y1 - y0 < min_h:
            continue
        ext = text_extent(clean[y0:y1])
        if ext is None:
            continue
        x0, x1 = ext
        if EXPAND_STAMP_PX > 0:
            x0, x1 = expand_extent(ruleless[y0:y1], x0, x1, max_expand=EXPAND_STAMP_PX)
        boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


# ------------------------------------------------------------------ 5. jalankan & simpan debug
def process_page(gray):
    bw0 = binarize(gray)
    angle = estimate_skew(bw0)
    gray2 = deskew(gray, angle)
    bw = binarize(gray2)
    return gray2, bw, angle, find_lines(bw)


def save_debug(name, page_no, gray, boxes, out_dir="debug"):
    os.makedirs(out_dir, exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lines_dir = os.path.join(out_dir, "%s_p%d_lines" % (name, page_no))
    os.makedirs(lines_dir, exist_ok=True)
    for f in os.listdir(lines_dir):                               # bersihkan hasil lama
        os.remove(os.path.join(lines_dir, f))
    pad = 6
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 170, 0), 2)
        cv2.putText(vis, str(i), (max(x0 - 60, 0), y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        crop = gray[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
        cv2.imencode(".png", crop)[1].tofile(os.path.join(lines_dir, "line_%02d.png" % i))
    path = os.path.join(out_dir, "%s_p%d_boxes.png" % (name, page_no))
    cv2.imencode(".png", vis)[1].tofile(path)
    return path


# ------------------------------------------------------------------ 6. model (TAHAP 2)
import string

CHARS = string.ascii_letters + string.digits + ".,:;-/()'\"&%#@!?+ "   # HARUS sama dengan saat training
BLANK = 0
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
IMG_H = 32
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crnn_id_best.pth")
_MODEL = {}


def _build_crnn(num_classes, hidden=256):
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            def block(i, o):
                return [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
            self.cnn = nn.Sequential(
                *block(1, 64),    nn.MaxPool2d(2, 2),
                *block(64, 128),  nn.MaxPool2d(2, 2),
                *block(128, 256), *block(256, 256), nn.MaxPool2d((2, 1)),
                *block(256, 512), *block(512, 512), nn.MaxPool2d((2, 1)),
                nn.Conv2d(512, 512, (2, 1), bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            )
            self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden * 2, num_classes)

        def forward(self, x):
            f = self.cnn(x).squeeze(2).permute(0, 2, 1)
            f, _ = self.rnn(f)
            return self.fc(f)

    return CRNN()


def load_model(path=DEFAULT_MODEL):
    """Muat model sekali saja (disimpan di memori untuk panggilan berikutnya)."""
    if path in _MODEL:
        return _MODEL[path]
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError("File model tidak ditemukan: " + path)
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["chars"] != CHARS:
        raise ValueError("Charset model berbeda dengan CHARS di ocr_engine.py")
    model = _build_crnn(len(CHARS) + 1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _MODEL[path] = model
    return model


def _prep_line(gray_crop):
    """Persis seperti prep_image di notebook (Pillow): autocontrast, tinggi 32 px, normalisasi."""
    from PIL import Image, ImageOps
    img = ImageOps.autocontrast(Image.fromarray(gray_crop).convert("L"))
    w, h = img.size
    new_w = min(max(int(w * IMG_H / h), 32), 2000)
    img = img.resize((new_w, IMG_H), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return arr[None, None]


def _greedy_decode(logits):
    p = logits.argmax(2)[0].cpu().numpy()
    out, prev = [], 0
    for k in p:
        if k != prev and k != BLANK:
            out.append(IDX2CHAR[int(k)])
        prev = k
    return "".join(out)


def read_line(model, gray_crop):
    """Baca satu potongan baris -> teks."""
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(_prep_line(gray_crop)))
    return _greedy_decode(logits)


# ------------------------------------------------------------------ 7. pipeline utuh
def crop_line(gray, box, pad=6):
    x0, y0, x1, y1 = box
    return gray[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]


def run_ocr(path, model_path=DEFAULT_MODEL, reader=None, debug_name=None):
    """
    Satu fungsi untuk semuanya: file (gambar/PDF) -> teks.
    reader(gray_crop) -> str  boleh diganti untuk pengujian; default memakai model CRNN.
    """
    if reader is None:
        model = load_model(model_path)
        reader = lambda crop: read_line(model, crop)
    pages = []
    for no, page in enumerate(load_pages(path), 1):
        gray, bw, angle, boxes = process_page(page)
        lines = [reader(crop_line(gray, b)) for b in boxes]
        if debug_name:
            save_debug(debug_name, no, gray, boxes)
        pages.append({"page": no, "angle": angle, "n_lines": len(lines),
                      "lines": lines, "text": "\n".join(lines)})
    return {"text": "\n\n".join(p["text"] for p in pages), "pages": pages}


# ------------------------------------------------------------------ 8. hitung CER
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    return edit_distance(pred, gt) / max(len(gt), 1)


def compare_lines(pred_lines, gt_lines):
    """Kalau jumlah baris sama, bandingkan baris per baris dan tampilkan baris yang salah."""
    bad = []
    for i, (p, g) in enumerate(zip(pred_lines, gt_lines), 1):
        if p != g:
            bad.append((i, g, p))
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.file))[0]
    res = run_ocr(a.file, a.model, debug_name=name)
    os.makedirs("debug", exist_ok=True)
    with open(os.path.join("debug", name + ".ocr.txt"), "w", encoding="utf-8") as fh:
        fh.write(res["text"])

    for pg in res["pages"]:
        print("Halaman %d: kemiringan %.1f derajat | %d baris" % (pg["page"], pg["angle"], pg["n_lines"]))
    print("\n" + "=" * 20 + " HASIL OCR " + "=" * 20)
    print(res["text"])
    print("=" * 51)

    gt_path = os.path.splitext(a.file)[0] + ".txt"
    if os.path.exists(gt_path) and len(res["pages"]) == 1:
        gt = open(gt_path, encoding="utf-8").read()
        pred = res["text"]
        norm = lambda t: "".join(t.split())          # abaikan spasi dan baris baru
        print("\nCER halaman (apa adanya)          : %.2f%%" % (cer(pred, gt) * 100))
        print("CER halaman (abaikan spasi/baris) : %.2f%%" % (cer(norm(pred), norm(gt)) * 100))
        pl, gl = res["pages"][0]["lines"], gt.split("\n")
        if len(pl) == len(gl):
            bad = compare_lines(pl, gl)
            print("Baris persis benar: %d dari %d" % (len(gl) - len(bad), len(gl)))
            for i, g, p in bad[:10]:
                print("  baris %d\n     GT  : %s\n     Pred: %s" % (i, g, p))
        else:
            print("Jumlah baris berbeda (hasil %d, sebenarnya %d): perlu dicek di gambar debug."
                  % (len(pl), len(gl)))