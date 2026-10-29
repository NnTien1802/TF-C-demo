"""Safe CSV/NPZ/ZIP sensor IO, with SGR training-statistics normalization."""
from dataclasses import dataclass, field
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd

# From data/SGR/README.md in the user's TF-C training package.
SGR_MEAN = np.array([-2.00637245, -4.74350595, 3.29379106], dtype=np.float32)
SGR_STD = np.array([5.04262161, 4.82831144, 4.80061388], dtype=np.float32)
LENGTH = 206
STEP = 100
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 120000
MAX_WINDOWS = 100


@dataclass
class SensorData:
    samples: np.ndarray                # [N,3,206]
    plot: np.ndarray                   # [N,206,3] (values as uploaded)
    time: np.ndarray                   # [N,206] seconds or point indices
    names: list = field(default_factory=list)
    labels: np.ndarray = None
    notices: list = field(default_factory=list)
    raw_rows: int = 0
    sampling_hz: float = None


def _normalize(arr):
    return (arr - SGR_MEAN[None, :]) / SGR_STD[None, :]


def _check_size(blob):
    if len(blob) > MAX_BYTES:
        raise ValueError("File quá lớn (>20 MB). Hãy xuất một phiên ghi ngắn hơn.")


def _csv_frame(blob):
    _check_size(blob)
    df = pd.read_csv(BytesIO(blob), low_memory=False)
    df.columns = [str(c).strip().lower().lstrip('\ufeff') for c in df.columns]
    for c in ("x", "y", "z"):
        if c not in df.columns:
            raise ValueError("CSV phải chứa ba cột x, y, z (không phân biệt chữ hoa/chữ thường).")
    if len(df) < LENGTH:
        raise ValueError(f"CSV có {len(df)} dòng, cần tối thiểu {LENGTH} điểm đo.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"CSV có quá nhiều dòng, tối đa {MAX_ROWS} dòng.")
    arr = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Các cột x/y/z có giá trị rỗng hoặc không phải số. Hãy làm sạch file trước.")
    if "seconds_elapsed" in df.columns:
        t = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=np.float64)
    elif "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
        # Sensor Logger uses nanosecond Unix timestamps; convert based on magnitude.
        if len(t) and np.isfinite(t[0]):
            t = t - t[0]
            if np.nanmax(np.abs(t)) > 1e7:
                t = t / 1e9
    else:
        t = np.arange(len(df), dtype=np.float64)
    if not np.isfinite(t).all():
        raise ValueError("Timestamp chứa giá trị không hợp lệ.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Timestamp cần tăng nghiêm ngặt theo thời gian; hãy kiểm tra các dòng trùng hoặc đảo thứ tự.")
    has_seconds = "seconds_elapsed" in df.columns or "time" in df.columns
    hz = None
    if has_seconds and len(t) > 3:
        median_dt = np.median(np.diff(t))
        if median_dt > 0:
            hz = float(1.0 / median_dt)
    return arr, t, hz


def csv_bytes(blob, is_raw=True, stride=STEP, name="CSV"):
    arr, t, hz = _csv_frame(blob)
    starts = list(range(0, len(arr) - LENGTH + 1, int(stride)))
    if len(starts) > MAX_WINDOWS:
        starts = starts[:MAX_WINDOWS]
    plot = np.stack([arr[i:i + LENGTH] for i in starts], axis=0)
    times = np.stack([t[i:i + LENGTH] for i in starts], axis=0)
    processed = _normalize(plot) if is_raw else plot.copy()
    samples = np.transpose(processed, (0, 2, 1)).astype(np.float32)
    notes = []
    if len(arr) > LENGTH:
        notes.append(f"Đã tách {len(starts)} cửa sổ, mỗi cửa sổ 206 điểm; bước trượt {stride} điểm.")
    if len(arr) - LENGTH + 1 > 0 and ((len(arr) - LENGTH)//int(stride) + 1) > MAX_WINDOWS:
        notes.append("Chỉ xử lý 100 cửa sổ đầu để đảm bảo tài nguyên Cloud.")
    if is_raw:
        notes.append("Đã chuẩn hóa theo mean/std của tập SGR train. Chỉ có ý nghĩa nếu cảm biến có cùng đơn vị và điều kiện đo tương tự dữ liệu SGR.")
    else:
        notes.append("Đã coi CSV là dữ liệu đã chuẩn hóa theo SGR; không chuẩn hóa lại.")
    if hz is not None:
        notes.append(f"Tần số suy ra từ timestamp: khoảng {hz:.2f} Hz. Không tự nội suy lên tần số của SGR.")
    return SensorData(samples, plot, times, [f"Cửa sổ {j + 1}" for j in range(len(starts))],
                      notices=notes, raw_rows=len(arr), sampling_hz=hz)


def zip_bytes(blob, is_raw=True, stride=STEP):
    _check_size(blob)
    with zipfile.ZipFile(BytesIO(blob)) as arc:
        candidates = [i for i in arc.infolist()
                      if not i.is_dir() and i.filename.lower().split("/")[-1] == "accelerometer.csv"]
        if not candidates:
            raise ValueError("ZIP không chứa Accelerometer.csv từ Sensor Logger.")
        if len(candidates) > 1:
            raise ValueError("ZIP chứa nhiều Accelerometer.csv; hãy chọn một phiên ghi.")
        info = candidates[0]
        if info.file_size > MAX_BYTES:
            raise ValueError("Accelerometer.csv giải nén vượt 20 MB.")
        csv_content = arc.read(info)
    return csv_bytes(csv_content, is_raw=is_raw, stride=stride, name="Sensor Logger")


def npz_bytes(blob, is_raw=False):
    _check_size(blob)
    # NPZ is a ZIP container: cap the decompressed size as well as upload bytes.
    try:
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            total_uncompressed = sum(info.file_size for info in archive.infolist())
            if total_uncompressed > 40 * 1024 * 1024 or len(archive.infolist()) > 24:
                raise ValueError("NPZ giải nén vượt 40 MB hoặc chứa quá nhiều tệp.")
    except zipfile.BadZipFile:
        raise ValueError("File NPZ không hợp lệ.")
    with np.load(BytesIO(blob), allow_pickle=False) as pack:
        selected = next((k for k in ("samples", "data", "x") if k in pack), None)
        if selected is None:
            raise ValueError("NPZ phải có khóa 'samples' (hoặc 'data'/'x').")
        arr = np.asarray(pack[selected], dtype=np.float32)
        y = np.asarray(pack["labels"]) if "labels" in pack else None
        names = [str(s) for s in np.asarray(pack["indices"]).tolist()] if "indices" in pack else None
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("NPZ yêu cầu [N,3,206] hoặc [N,206,3].")
    if arr.shape[1:] == (206, 3):
        arr = arr.transpose(0, 2, 1)
    if arr.shape[1:] != (3, LENGTH):
        raise ValueError(f"NPZ cần [N,3,206]; hiện có {arr.shape}.")
    if arr.shape[0] > MAX_WINDOWS:
        arr = arr[:MAX_WINDOWS]
        if y is not None:
            y = y[:MAX_WINDOWS]
        if names is not None:
            names = names[:MAX_WINDOWS]
    if names is not None and len(names) != len(arr):
        raise ValueError("NPZ có indices không khớp số mẫu.")
    if y is not None and len(y) != len(arr):
        raise ValueError("NPZ có labels không khớp số mẫu.")
    if not np.isfinite(arr).all():
        raise ValueError("NPZ chứa NaN hoặc Inf.")
    if not len(arr):
        raise ValueError("NPZ không có mẫu.")
    plot = np.transpose(arr, (0, 2, 1))
    processed = _normalize(plot) if is_raw else plot.copy()
    samples = processed.transpose(0, 2, 1).astype(np.float32)
    times = np.tile(np.arange(206, dtype=float), (len(samples), 1))
    return SensorData(samples, plot, times,
                      names or [f"Mẫu {j + 1}" for j in range(len(samples))],
                      labels=y,
                      notices=["NPZ đang được coi là gia tốc thô và chuẩn hóa theo SGR train." if is_raw
                               else "NPZ được coi là dữ liệu SGR đã chuẩn hóa; giữ nguyên giá trị."],
                      raw_rows=len(samples) * 206)
