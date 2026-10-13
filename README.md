# OCR Arsip RRI Batam

Sistem OCR (Optical Character Recognition) untuk mendigitalkan arsip surat berbahasa Indonesia. Pengguna mengunggah gambar atau PDF surat, lalu sistem mengembalikan teks hasil pembacaan lewat REST API.

Model yang dipakai adalah **CRNN (CNN + BiLSTM + CTC)** yang dilatih dari nol dengan data sintetis, kemudian di-*fine-tune* ke teks surat bahasa Indonesia. Model dilatih di Kaggle (GPU), sedangkan inference berjalan di CPU laptop.

---

## Daftar Isi

1. [Gambaran Umum](#gambaran-umum)
2. [Alur Kerja (Pipeline)](#alur-kerja-pipeline)
3. [Struktur Folder](#struktur-folder)
4. [Instalasi](#instalasi)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Penjelasan Kode](#penjelasan-kode)
7. [Training Model](#training-model)
8. [Dokumentasi API](#dokumentasi-api)
9. [Evaluasi (CER)](#evaluasi-cer)
10. [Keterbatasan](#keterbatasan)
11. [Catatan Git](#catatan-git)

---

## Gambaran Umum

OCR di project ini dibagi menjadi dua tahap besar:

| Tahap | Tugas | Teknologi |
|-------|-------|-----------|
| **Tahap 1: Deteksi baris** | Halaman surat dipecah menjadi potongan gambar per baris teks | OpenCV (binarisasi, deskew, proyeksi horizontal) |
| **Tahap 2: Pengenalan teks** | Tiap potongan baris dibaca menjadi string | PyTorch CRNN + CTC |

Hasilnya dibungkus dengan **FastAPI** supaya bisa dipanggil dari aplikasi lain (misalnya sistem arsip berbasis web).

---

## Alur Kerja (Pipeline)

```
File (PNG/JPG/PDF)
      │
      ▼
 load_pages()        PDF dirender 200 DPI per halaman (PyMuPDF), gambar dibuka grayscale
      │
      ▼
 binarize()          Adaptive threshold: tulisan = putih, latar = hitam
      │
      ▼
 estimate_skew()     Cari sudut kemiringan (-3° s/d +3°) lalu deskew()
      │
      ▼
 find_lines()        Hapus garis & gumpalan (stempel), proyeksi horizontal → kotak tiap baris
      │
      ▼
 crop_line()         Potong tiap baris (dengan margin 6 px)
      │
      ▼
 read_line()         Resize tinggi 32 px → CRNN → greedy CTC decode
      │
      ▼
 Teks per halaman → digabung menjadi teks akhir
```

---

## Struktur Folder

```
ocr-arsip/
├── main.py                  # FastAPI (endpoint /health dan /ocr)
├── ocr_engine.py            # Pipeline OCR: deteksi baris + model CRNN + hitung CER
├── crnn_best.pth            # Model tahap 1 (training bahasa Inggris)
├── crnn_id_best.pth         # Model tahap 2 (fine-tune bahasa Indonesia) ← dipakai default
├── training_ai/
│   ├── ai-ocr-arsip-aris.ipynb                              # Training awal (data sintetis Inggris)
│   ├── fine-tunning-ai-crnn-ocr-indonesia-arsip-aris.ipynb  # Fine-tuning ke surat Indonesia
│   └── ocr-testing-surat.ipynb                              # Tes model pada potongan baris asli
├── contoh_surat/            # (tidak di-push) data uji: pages, real_lines, real_lines_full
├── debug/                   # (tidak di-push) hasil visualisasi kotak baris
└── .gitignore
```

> Folder `venv/`, `debug/`, `contoh_surat/`, dan `__pycache__/` tidak ikut di Git.

---

## Instalasi

**Kebutuhan:** Python 3.10 (versi yang dipakai saat pengembangan).

```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependensi
pip install torch numpy opencv-python pillow pymupdf fastapi uvicorn python-multipart
```

Opsional, simpan daftar versi yang terpasang supaya mudah direproduksi:

```powershell
pip freeze > requirements.txt
```

Pastikan file model `crnn_id_best.pth` ada di folder yang sama dengan `ocr_engine.py`.

---

## Cara Menjalankan

### A. Lewat command line (uji satu file)

```powershell
python ocr_engine.py contoh_surat\pages\surat01.png
python ocr_engine.py contoh_surat\pages\surat01.pdf
python ocr_engine.py file.png --model crnn_id_best.pth
```

Hasil:

- Teks OCR dicetak di layar dan disimpan di `debug\<nama>.ocr.txt`.
- `debug\<nama>_p1_boxes.png`: halaman dengan kotak hijau di tiap baris yang terdeteksi.
- `debug\<nama>_p1_lines\`: potongan gambar tiap baris.
- Jika ada file `<nama>.txt` (teks yang benar) di folder yang sama, **CER dihitung otomatis**.

### B. Lewat API (FastAPI)

```powershell
uvicorn main:app --port 8001
```

- Dokumentasi interaktif (Swagger): http://127.0.0.1:8001/docs
- Cek status: http://127.0.0.1:8001/health

Untuk memakai model lain tanpa mengubah kode:

```powershell
$env:OCR_MODEL = "C:\path\ke\model.pth"
uvicorn main:app --port 8001
```

---

## Penjelasan Kode

### `ocr_engine.py`

| Bagian | Fungsi | Keterangan |
|--------|--------|------------|
| 1. Buka file | `load_pages()` | PDF dirender dengan PyMuPDF pada 200 DPI (grayscale). Gambar dibaca dengan `cv2.imdecode` supaya aman untuk path Windows. |
| 2. Binarisasi | `binarize()` | `adaptiveThreshold` (tahan cahaya tidak rata), lalu bintik kecil (< 12 piksel) dibuang sebagai noise fotokopi. |
| 3. Luruskan | `estimate_skew()`, `deskew()` | Mencoba sudut -3° sampai +3° (langkah 0,1°) dan memilih sudut dengan variansi proyeksi horizontal tertinggi (baris paling "tajam"). |
| 4. Potong baris | `remove_rules()`, `remove_blobs()`, `text_extent()`, `find_lines()` | Hapus garis horizontal panjang (kop/tabel), hapus gumpalan besar (stempel/tanda tangan), lalu cari baris dari profil proyeksi horizontal. Batas kiri-kanan ditentukan oleh `text_extent()`. |
| 5. Debug | `save_debug()` | Menyimpan gambar kotak dan potongan tiap baris ke folder `debug/`. |
| 6. Model | `_build_crnn()`, `load_model()`, `_prep_line()`, `_greedy_decode()`, `read_line()` | Arsitektur CRNN, pemuatan model (di-cache di memori), preprocessing, dan decoding. |
| 7. Pipeline | `run_ocr()` | Satu fungsi dari file → teks. Parameter `reader` bisa diganti untuk pengujian. |
| 8. Evaluasi | `edit_distance()`, `cer()`, `compare_lines()` | Menghitung CER dan menampilkan baris yang salah. |

**Pengaturan penting:**

| Konstanta | Nilai | Fungsi |
|-----------|-------|--------|
| `DPI` | 200 | Resolusi render PDF |
| `IMG_H` | 32 | Tinggi gambar input model (piksel) |
| `CHARS` | huruf + angka + `.,:;-/()'"&%#@!?+` dan spasi | Karakter yang dikenali (80 karakter + 1 blank CTC) |
| `EXPAND_STAMP_PX` | 0 | Perluasan batas baris ke arah stempel. Dimatikan karena pada tes surat contoh justru membawa goresan stempel dan menaikkan CER. |

> **Penting:** `CHARS` di `ocr_engine.py` harus **identik** dengan charset saat training. Jika berbeda, `load_model()` akan menolak model dengan pesan error.

### `main.py`

API tipis di atas `ocr_engine.run_ocr`:

- Model dimuat **sekali** saat server menyala (`lifespan`).
- Validasi: format file, ukuran maksimal **20 MB**, dan file tidak boleh kosong.
- `threading.Lock` memastikan **satu proses OCR pada satu waktu**, karena CPU laptop tidak kuat paralel.
- File upload disimpan di folder sementara dan otomatis terhapus setelah selesai.

### Arsitektur model (CRNN)

```
Input (1 × 32 × W)
  │
  ├─ CNN: 7 blok Conv-BN-ReLU + MaxPool  →  fitur (512 × 1 × W/4)
  │       (tinggi dipadatkan 32 → 1, lebar diperkecil 4×)
  │
  ├─ BiLSTM 2 lapis, hidden 256 (dropout 0,2)  →  urutan kiri ke kanan
  │
  └─ Linear (512 → 81 kelas)  →  CTC decode
```

Karena lebar diperkecil 4×, panjang urutan CTC (`ilens`) = `lebar_gambar // 4`.

---

## Training Model

Seluruh training dilakukan di **Kaggle (GPU)**. Notebook ada di folder `training_ai/`.

### Tahap 1: `ai-ocr-arsip-aris.ipynb` → `crnn_best.pth`

Melatih CRNN dari nol dengan **data sintetis yang digenerate on-the-fly** (tidak butuh dataset gambar).

- **Teks:** kata dari docstring modul Python (tanpa internet), angka, pola nomor dokumen, tanggal, huruf besar, dan kapital di awal kata.
- **Font:** otomatis dicari di sistem, lalu difilter (`font_is_good`) untuk membuang font simbol/matematika atau font yang tidak lengkap.
- **Augmentasi mirip hasil scan:** rotasi kecil, blur, noise, kontras rendah.
- **Konfigurasi:** 30 epoch, batch 128, 50.000 sampel per epoch, `AdamW` + `OneCycleLR` (LR maks 1e-3), mixed precision (AMP), `CTCLoss`.
- **Charset sengaja dibuat lengkap dari awal** (termasuk `/` dan tanda baca surat) agar tahap 2 tidak perlu mengganti layer output.

### Tahap 2: `fine-tunning-ai-crnn-ocr-indonesia-arsip-aris.ipynb` → `crnn_id_best.pth`

Memulai dari bobot `crnn_best.pth`, lalu dilatih dengan teks surat bahasa Indonesia.

- **Generator teks surat:** nomor surat, perihal, tanggal, nama dengan gelar, NIP, nominal rupiah, jabatan, alamat, kalimat isi surat, dan heading (misalnya "SURAT KEPUTUSAN", "MEMUTUSKAN:").
- **Augmentasi tambahan:** shear (miring), lingkaran mirip stempel, garis tabel, ketebalan tulisan, artefak JPEG, bintik fotokopi, pencahayaan tidak rata.
- **Campuran data Inggris 15%** (`EN_RATIO`) supaya model tidak "lupa" hasil tahap 1.
- **Pemisahan font:** sebagian keluarga font disisihkan khusus untuk validasi dan tidak pernah dilihat saat training. Model terbaik dipilih berdasarkan **CER pada font baru**, sehingga penilaiannya lebih jujur.
- **Konfigurasi:** 20 epoch, batch 96, LR 3e-4 (lebih kecil karena fine-tuning), `MAX_LEN` 48 karakter per baris.
- Notebook juga mencetak perbandingan CER **sebelum vs sesudah** fine-tuning.

### Pengujian pada gambar asli: `ocr-testing-surat.ipynb`

Menguji `crnn_id_best.pth` pada potongan baris dari surat asli, dengan label di `labels.csv` (kolom `filename,text`). Dua folder diuji: `real_lines` dan `real_lines_full`. Notebook menampilkan CER, persentase baris yang persis benar, dan baris terburuk.

---

## Dokumentasi API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /ocr`

Unggah file sebagai `multipart/form-data` dengan field `file`.

**Format didukung:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.pdf`
**Ukuran maksimal:** 20 MB

**Contoh (curl):**

```bash
curl -X POST "http://127.0.0.1:8001/ocr" -F "file=@surat01.pdf"
```

**Contoh respons:**

```json
{
  "filename": "surat01.pdf",
  "seconds": 4.21,
  "n_pages": 1,
  "text": "Nomor : 123/RRI/BTM/IV/2024\nPerihal : Undangan Rapat Koordinasi\n...",
  "pages": [
    { "page": 1, "n_lines": 32, "text": "Nomor : 123/RRI/BTM/IV/2024\n..." }
  ]
}
```

**Kode error:**

| Kode | Penyebab |
|------|----------|
| 400 | File kosong |
| 413 | File lebih dari 20 MB |
| 415 | Format file tidak didukung |
| 422 | File rusak atau tidak bisa diproses |

---

## Evaluasi (CER)

Metrik utama adalah **CER (Character Error Rate)**:

```
CER = jarak edit (Levenshtein) antara hasil dan teks benar / jumlah karakter teks benar
```

`ocr_engine.py` menghitung dua versi CER untuk satu halaman: apa adanya, dan dengan mengabaikan spasi/baris baru. Untuk memakainya, letakkan teks yang benar dalam file `.txt` dengan nama sama dengan gambarnya.

Isi tabel berikut dengan hasil dari notebook Anda:

| Pengujian | CER |
|-----------|-----|
| Data sintetis, font baru (sebelum fine-tune) | _isi_ |
| Data sintetis, font baru (sesudah fine-tune) | _isi_ |
| Baris asli (`real_lines`) | _isi_ |
| Baris asli (`real_lines_full`) | _isi_ |

---

## Keterbatasan

- Hanya untuk **teks cetak**; tulisan tangan tidak didukung.
- Charset terbatas pada 80 karakter; huruf beraksen dan simbol lain tidak dikenali.
- Deteksi baris berbasis proyeksi horizontal sehingga paling cocok untuk surat **satu kolom**. Tata letak kompleks (multi-kolom, tabel rumit) perlu penanganan tambahan.
- Stempel dan tanda tangan yang menimpa teks dapat menurunkan akurasi.
- Decoding memakai *greedy*, tanpa kamus atau *language model*.
- Inference berjalan di CPU dan diproses satu per satu.

---

## Catatan Git

- File model (`*.pth`, masing-masing ± 31 MB) ikut di repo karena masih di bawah batas 100 MB GitHub.
- `contoh_surat/` tidak di-push karena dapat berisi dokumen asli yang bersifat pribadi.
- Untuk menjalankan di komputer lain, clone repo, buat venv baru, lalu install dependensi (lihat [Instalasi](#instalasi)).