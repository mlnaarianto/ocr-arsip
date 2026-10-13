"""
main.py  --  FastAPI tipis di atas ocr_engine.run_ocr

Jalankan (dari folder ocr-arsip, venv aktif):
    uvicorn main:app --port 8001

Cek di browser:
    http://127.0.0.1:8001/docs      (halaman uji: upload file, klik Execute)
    http://127.0.0.1:8001/health
"""
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

import ocr_engine

MODEL_PATH = os.environ.get("OCR_MODEL", ocr_engine.DEFAULT_MODEL)
MAX_MB = 20
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}

_lock = threading.Lock()   # satu proses OCR pada satu waktu (CPU laptop tidak kuat paralel)


@asynccontextmanager
async def lifespan(app):
    ocr_engine.load_model(MODEL_PATH)   # muat model sekali saat server menyala
    yield


app = FastAPI(title="OCR Arsip RRI Batam", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(415, "Format tidak didukung: %s. Gunakan %s" % (ext or "-", ", ".join(sorted(ALLOWED))))

    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, "File lebih dari %d MB" % MAX_MB)
    if not data:
        raise HTTPException(400, "File kosong")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + ext)
        with open(path, "wb") as fh:
            fh.write(data)
        t0 = time.time()
        try:
            with _lock:
                res = ocr_engine.run_ocr(path, MODEL_PATH)
        except ValueError as e:                      # file rusak / bukan gambar
            raise HTTPException(422, "Gagal memproses file: %s" % e)

    return {
        "filename": file.filename,
        "seconds": round(time.time() - t0, 2),
        "n_pages": len(res["pages"]),
        "text": res["text"],
        "pages": [{"page": p["page"], "n_lines": p["n_lines"], "text": p["text"]} for p in res["pages"]],
    }