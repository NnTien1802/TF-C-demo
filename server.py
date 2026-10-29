"""GestureLab: FastAPI backend for the original TF-C + SGR checkpoint.

HTML/CSS/JavaScript frontend only. No Streamlit dependency.
File uploads are parsed only as CSV, NPZ (allow_pickle=False), or Sensor Logger ZIP.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from data_io import MAX_BYTES, SensorData, csv_bytes, npz_bytes, zip_bytes
from model_runtime import LABELS, TRANSLATIONS, load_model, predict

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
CHECKPOINT = BASE / "models" / "tfc_sgr_demo.pt"
EXAMPLES = BASE / "samples" / "sgr_examples.npz"
MAX_WINDOWS_TO_ANALYZE = 24
logger = logging.getLogger("gesturelab")

app = FastAPI(title="GestureLab TF-C API", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def get_model():
    # Sanctioned local checkpoint, tensor-only deserialization in model_runtime.
    return load_model(CHECKPOINT)


@lru_cache(maxsize=1)
def example_data():
    if not EXAMPLES.is_file():
        raise FileNotFoundError("Không tìm thấy samples/sgr_examples.npz.")
    return npz_bytes(EXAMPLES.read_bytes(), is_raw=False)


def _read_input(source: str, blob: bytes | None, filename: str | None, raw: bool, stride: int) -> SensorData:
    if source == "example":
        return example_data()
    if source != "upload":
        raise ValueError("Nguồn dữ liệu không hợp lệ.")
    if blob is None or not filename:
        raise ValueError("Bạn cần chọn file CSV, NPZ hoặc ZIP trước khi phân tích.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return csv_bytes(blob, is_raw=raw, stride=stride, name=filename)
    if suffix == ".npz":
        return npz_bytes(blob, is_raw=raw)
    if suffix == ".zip":
        return zip_bytes(blob, is_raw=raw, stride=stride)
    raise ValueError("Chỉ hỗ trợ .csv, .npz và .zip Sensor Logger.")


def _stats(signal: np.ndarray):
    """Descriptive statistics of vector magnitude in uploaded/preview units."""
    arr = np.asarray(signal, dtype=np.float64)
    magnitudes = np.linalg.norm(arr, axis=1)
    std = float(np.std(magnitudes))
    peak = float(np.max(magnitudes))
    rms = float(np.sqrt(np.mean(magnitudes ** 2)))
    middle = len(magnitudes) // 2
    first = float(np.std(magnitudes[:middle]))
    second = float(np.std(magnitudes[middle:]))
    if second > first * 1.18 and second > 0.02:
        trend = "Biến động tăng về cuối cửa sổ."
    elif first > second * 1.18 and first > 0.02:
        trend = "Biến động mạnh hơn ở đầu cửa sổ."
    else:
        trend = "Mức biến động tương đối đều giữa hai nửa cửa sổ."
    peak_index = int(np.argmax(magnitudes)) + 1
    return {
        "std": round(std, 4), "peak": round(peak, 4), "rms": round(rms, 4),
        "comment": f"{trend} Đỉnh độ lớn gia tốc nằm tại điểm {peak_index}/{len(magnitudes)}."
    }


def _serialize(data: SensorData, *, source: str, raw: bool, with_prediction: bool):
    count = min(len(data.samples), MAX_WINDOWS_TO_ANALYZE)
    notices = list(data.notices)
    if len(data.samples) > count:
        notices.append(f"Giới hạn phân tích {count} cửa sổ đầu để tiết kiệm tài nguyên trên Cloud.")
    probabilities = None
    if with_prediction:
        if not CHECKPOINT.exists():
            raise HTTPException(status_code=503, detail="Chưa có checkpoint tại models/tfc_sgr_demo.pt. Hãy copy file đã export từ TF-C.")
        try:
            encoder, classifier, _config = get_model()
            probabilities = predict(encoder, classifier, data.samples[:count], batch_size=6)
        except Exception as exc:
            logger.exception("TF-C inference failed")
            raise HTTPException(status_code=503, detail="Không thể nạp hoặc chạy checkpoint TF-C. Kiểm tra Render Logs và file model.") from exc

    windows = []
    for i in range(count):
        current = np.asarray(data.plot[i], dtype=np.float64)
        times = np.asarray(data.time[i], dtype=np.float64)
        times = times - times[0]
        truth = None
        if data.labels is not None and len(data.labels) > i:
            index = int(data.labels[i])
            if 0 <= index < len(LABELS):
                truth = {"id": index, "name": LABELS[index], "label": TRANSLATIONS[LABELS[index]]}
        result = None
        if probabilities is not None:
            scores = np.asarray(probabilities[i], dtype=float)
            index = int(np.argmax(scores))
            result = {
                "id": index, "name": LABELS[index], "label": TRANSLATIONS[LABELS[index]],
                "score": round(float(scores[index]), 6),
                "probabilities": [round(float(x), 6) for x in scores],
                "matches_truth": (index == truth["id"]) if truth is not None else None,
            }
        windows.append({
            "index": i, "name": str(data.names[i]) if i < len(data.names) else f"Cửa sổ {i+1}",
            "time": np.round(times, 5).tolist(),
            "signal": np.round(current, 6).tolist(),
            "statistics": _stats(current), "truth": truth, "prediction": result,
        })
    return {
        "info": {
            "source": "Mẫu SGR test" if source == "example" else "File người dùng",
            "window_count": count,
            "raw_rows": int(data.raw_rows),
            "sampling_hz": round(float(data.sampling_hz), 3) if data.sampling_hz is not None else None,
            "signal_units": "Giá trị gia tốc đầu vào" if raw else "Giá trị đầu vào đã chuẩn hóa",
            "notices": notices,
            "model_used": "TF-C Transformer gốc · HAR → SGR · 6 lớp" if with_prediction else None,
            "ood_note": "Suy luận chỉ sử dụng kênh X (kênh 0) từ ba trục hiển thị, đúng với lần huấn luyện. Mô hình được fine-tune cho 6 cử chỉ SGR. Hoạt động HAR như ngồi/đứng/đi bộ nằm ngoài 6 nhãn mục tiêu; điểm Softmax không phải độ tin cậy đã hiệu chuẩn.",
        },
        "labels": [{"id": i, "name": k, "label": TRANSLATIONS[k]} for i, k in enumerate(LABELS)],
        "windows": windows,
    }


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/api/health")
def health():
    return {
        "service": "ok", "model_file_present": CHECKPOINT.is_file(),
        "examples_available": EXAMPLES.is_file(),
        "model_kind": "Original TF-C Transformer, 1-channel checkpoint (not DINOv3)",
        "classes": len(LABELS),
    }


@app.get("/api/labels")
def labels():
    return [{"id": i, "name": k, "label": TRANSLATIONS[k]} for i, k in enumerate(LABELS)]


@app.post("/api/analyze")
async def analyze(
    request: Request,
    source: str = Form("example"),
    is_raw: bool = Form(False),
    stride: int = Form(100),
    do_predict: bool = Form(False),
    file: UploadFile | None = File(None),
):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BYTES + 1024 * 1024:
                raise HTTPException(status_code=413, detail="Upload quá 20 MB.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length không hợp lệ.")
    if not 1 <= stride <= 206:
        raise HTTPException(status_code=422, detail="Stride cần nằm trong khoảng 1–206.")
    blob = None
    filename = None
    if source == "upload":
        if file is None:
            raise HTTPException(status_code=422, detail="Hãy chọn file để upload.")
        filename = Path(file.filename or "").name
        blob = await file.read(MAX_BYTES + 1)
        await file.close()
        if len(blob) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Upload quá 20 MB.")
    try:
        data = _read_input(source, blob, filename, raw=is_raw, stride=stride)
        # For built-in samples, these are already normalized; ignore raw flag.
        return _serialize(data, source=source, raw=(is_raw if source == "upload" else False), with_prediction=do_predict)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unable to process uploaded data")
        raise HTTPException(status_code=422, detail="Dữ liệu không hợp lệ hoặc không thể đọc an toàn.") from exc


@app.exception_handler(413)
async def too_large(_request, exc):
    return JSONResponse(status_code=413, content={"detail": exc.detail})
