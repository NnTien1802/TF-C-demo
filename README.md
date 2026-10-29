# GestureLab — TF-C Gesture Recognition (HTML / CSS / JavaScript + FastAPI)

> **Bản demo web độc lập, không sử dụng Streamlit.** Giao diện HTML/CSS/JavaScript giao tiếp với FastAPI; backend PyTorch chạy mô hình TF-C **thật** đã huấn luyện HAR → SGR và trả xác suất cho 6 cử chỉ.

## Giao diện mới — 3 bước theo mẫu tham khảo

Đã chuyển giao diện thành **ba cột đơn giản, sáng màu**, thay cho thiết kế dashboard có sidebar:

1. **Input data**: chọn mẫu SGR hoặc tải CSV/NPZ/ZIP, chọn chế độ chuẩn hóa, xem số điểm đo/cửa sổ.
2. **Preview**: biểu đồ gia tốc X/Y/Z trong miền thời gian; **FFT magnitude** của độ lớn vector gia tốc trong miền tần số; Std/Peak/RMS và nút chạy mô hình.
3. **Prediction result**: nhãn dự đoán, điểm Softmax lớn nhất, điểm sáu lớp, đối chiếu nhãn thực và tải kết quả CSV.

Giao diện tự chuyển thành bố cục một cột trên điện thoại. Phổ FFT tại frontend **chỉ phục vụ xem trước**, không thay thế bước FFT và tiền xử lý dùng bên trong TF-C để suy luận. Nếu file có tần số lấy mẫu, trục phổ có đơn vị Hz; nếu không, trục dùng **chu kỳ/mẫu**, tuyệt đối không giả định Hz.

> Mẫu ảnh tham khảo ghi **Speech Deepfake Detection / Real / Fake**, nhưng bài toán thực tế của GestureLab là **6 cử chỉ cảm biến SGR**; vì thế các nhãn và biểu đồ đã được sửa theo đúng đầu vào/đầu ra của checkpoint. Không có chế độ dự đoán Real/Fake trong mô hình này.

## Tính năng

- Mẫu **SGR test** có nhãn, chọn từng mẫu và đối chiếu dự đoán.
- Upload **CSV** (`x,y,z`), **NPZ** (`samples: [N,3,206]`) hoặc **ZIP Sensor Logger** (`Accelerometer.csv`).
- Biểu đồ X/Y/Z và FFT magnitude của độ lớn gia tốc bằng **Canvas thuần**, không phụ thuộc CDN JavaScript.
- Thống kê **Std**, **Peak**, **RMS** trên độ lớn gia tốc từng cửa sổ; nhận xét xu hướng biến động.
- Kết quả dự đoán thật từ TF-C, điểm **Softmax cho 6 lớp**, so sánh nhãn và tải CSV dự đoán.
- Responsive desktop/mobile; trạng thái backend và checkpoint; deploy được bằng **Render Web Service**.

## Kiến trúc

```
Web browser
  ├── static/index.html       (UI)
  ├── static/styles.css       (giao diện)
  └── static/app.js           (Fetch API, Canvas, bảng xác suất)
         ↓ POST /api/analyze
FastAPI: server.py
         ├── data_io.py       (CSV/NPZ/ZIP, cửa sổ 206 điểm, chuẩn hóa SGR)
         └── model_runtime.py (PyTorch TF-C gốc, 2 Transformer encoders)
                  ↓
         models/tfc_har_to_sgr_safe.pt
                  ↓
         6 lớp: typing, teeth, drinking, writing, clapping, folding
```

**Không sử dụng DINOv3** ở website này. Phần thay encoder bằng DINOv3 là bài nộp kiến trúc riêng.

## Quan trọng: thêm checkpoint đã huấn luyện

**Bản ZIP không đính kèm checkpoint vì file trọng số chưa có trong những tệp được gửi trực tiếp ở cuộc trò chuyện này.** Không có checkpoint thì website **vẫn xem được đồ thị và thống kê**, nhưng endpoint dự đoán sẽ báo thiếu model. Tuyệt đối không có kết quả dự đoán giả.

Trên máy Windows, sao chép checkpoint đã được chuẩn hóa (`weights_only=True`) từ GestureLab v2:

```cmd
copy /Y "C:\AI_projects\TFC_SGR_Streamlit_v2\models\tfc_har_to_sgr_safe.pt" "C:\AI_projects\GestureLab_HTML_FastAPI_Render\models\tfc_har_to_sgr_safe.pt"
```

File này khoảng **7,1 MB**, phải nằm đúng đường dẫn `models/tfc_har_to_sgr_safe.pt`. Khi đưa lên GitHub, chỉ đăng checkpoint nếu bạn có quyền phân phối; **không đăng checkpoint chưa rõ nguồn hoặc dữ liệu cá nhân**. `torch.load` trong runtime dùng `weights_only=True` để tránh tải pickle tùy ý từ người dùng.

## Chạy trên Windows

Yêu cầu Python **3.11**. Tạo môi trường Conda riêng để không ảnh hưởng phiên bản cũ:

```cmd
cd /d C:\AI_projects\GestureLab_HTML_FastAPI_Render
conda create -n gesturelab-web python=3.11 -y
conda activate gesturelab-web
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Mở **http://127.0.0.1:8000**. Kiểm tra `/api/health` (file checkpoint đã có hay chưa) và `/api/docs` (OpenAPI). Sau khi chọn mẫu SGR, nhấn **Phân loại bằng TF-C**.

## Deploy lên Render (không dùng Streamlit)

1. Đưa toàn bộ dự án này, bao gồm thư mục `static/`, `server.py`, `data_io.py`, `model_runtime.py`, `samples/`, `requirements.txt`, `.python-version`, `render.yaml` và **checkpoint đã được phép công khai**, lên **GitHub repo mới** (khuyên dùng `GestureLab-HTML-Render` để giữ nguyên repo Streamlit cũ).
2. Truy cập [Render Dashboard](https://dashboard.render.com/) → **New → Blueprint** → chọn repo GitHub vừa tạo. Render đọc `render.yaml` và tạo **Python Web Service, plan Free**.
3. Nếu cấu hình thủ công bằng **New → Web Service**, chọn `Python 3`, **Build Command** `pip install -r requirements.txt`, **Start Command** `uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1`, Python `3.11.11`, plan Free.
4. Sau khi Render báo **Live**, mở URL được Render cấp (`*.onrender.com`), chọn mẫu SGR và thử dự đoán; kiểm tra cửa sổ ẩn danh.

> Render Free có thể ngủ khi không hoạt động, khiến lượt truy cập đầu tiên chậm. CPU/RAM miễn phí có thể giới hạn tốc độ infer PyTorch. Website có endpoint `/api/health` để xác định có tìm thấy checkpoint, nhưng health check không thay cho bài kiểm tra dự đoán.

**Tài liệu chính thức:** [Render FastAPI](https://render.com/docs/deploy-fastapi) · [Render Python version](https://render.com/docs/python-version)

## API

| Endpoint | Ý nghĩa |
|---|---|
| `GET /` | Website HTML |
| `GET /api/health` | Trạng thái API, checkpoint và mẫu |
| `GET /api/labels` | Danh sách 6 nhãn |
| `POST /api/analyze` | Form `source=example|upload`, `file` (nếu upload), `is_raw`, `stride`, `do_predict` |
| `GET /api/docs` | Swagger/OpenAPI |

`/api/analyze` với `do_predict=false` vẫn trả tín hiệu và thống kê để xem trước. Với `do_predict=true`, **backend nạp checkpoint và chạy TF-C thật**; nếu chưa có checkpoint, trả HTTP 503. File tối đa 20 MB; tối đa 24 cửa sổ/1 lượt trên website để tránh quá tải CPU. Không hỗ trợ upload `.pt` hoặc giải mã pickle từ người dùng.

## Định nghĩa dữ liệu

- **Đầu vào:** mẫu chuẩn hóa `[N, 3, 206]`.
- **CSV/ZIP thô:** chuẩn hóa theo mean/std SGR train trong `data_io.py`. Nếu điều kiện đo/đơn vị khác tập SGR, kết quả có thể bị lệch.
- **NPZ mẫu:** đã chuẩn hóa; không chuẩn hóa lại.
- **Đầu ra:** phân bố Softmax `[N, 6]` (chưa được hiệu chuẩn).
- **Lưu ý:** file Sensor Logger điện thoại có thể tạo ra dự đoán trong 6 lớp SGR dù thuộc miền ngoài huấn luyện (OOD). Điều đó **không có nghĩa** là mô hình đã học bốn hoạt động HAR.

## Nguồn và ghi công

- Mô hình gốc: [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining) (TF-C, NeurIPS 2022).
- Model triển khai: Transformer TF-C gốc, được fine-tune với SGR; không tuyên bố đã đánh giá mô hình DINOv3.
- Pipeline giao diện web: HTML, CSS, JavaScript, FastAPI và PyTorch; bản chuyển đổi từ demo Streamlit trước đó.

**Dự án demo học thuật; không dành cho quyết định có tính an toàn.**
