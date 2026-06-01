# Đồ án 2: Data Fitting và Phương pháp OLS
**Môn học:** Toán ứng dụng và Thống kê (MTH00051) 
**Nhóm:** 10\
**Thành viên:** 
- 24120126 - Khúc Minh Quân
- 24120132 - Phạm Nguyễn Quang Sáng
- 24120196 - Nguyễn Lê Anh Kiên
- 24120211 - Lê Công Minh Nhựt
- 24120218 - Nguyễn Đức Quân

## 1. Cấu trúc thư mục
- `part1/`: Chứa các file cài đặt thuật toán OLS, ma trận Hat và mô phỏng lý thuyết.
    - `part1_notebook.ipynb`: Demo `Phần 1`.
    - `part1_benchmark.ipynb`: Kiểm chứng các hàm của part1/ với module NumPy/scikit-learn.
    - `linalg.py`: module toán học cơ bản, chứa các hàm xử lý ma trận, vector.
    - `ols_implementation.py`: module chứa các hàm cài đặt thuật toán OLS.
- `part2/`: Chứa pipeline tiền xử lý dữ liệu thực, so sánh các mô hình và kỹ thuật nâng cao.
    - `part2_notebook.ipynb`: Demo `Phần 2`.
    - `part2_benchmark.ipynb`: Kiểm chứng các hàm của part2/ với module NumPy/scikit-learn.
    - `data_pipeline.py`: module chứa các bước pipeline tiền xử lý dữ liệu.
    - `preprocess.py`: module chứa các hàm tiền xử lý dữ liệu.

## 2. Hướng dẫn cài đặt và chạy [[xem chi tiết bằng link này](https://docs.python.org/3/library/venv.html)]
Để chạy các chương trình trong đây cần có môi trường ảo (virtual environment), có thể dùng `venv` hoặc `conda`, ở file `README.md` này chỉ hướng dẫn sử dụng `venv`.
### Khởi tạo môi trường ảo
- **Mở terminal ở thư mục hiện tại**, từ giờ ta sẽ làm việc với terminal này.

- Trong terminal chạy
```bash 
python -m venv .venv
```
### Kích hoạt môi trường (Activate)
- Đối với hệ điều hành `Linux`, trong `bash/zsh` terminal chạy
```bash
source .venv/bin/activate
```
- Đồi với hệ điều hành `Windows`, trong `cmd.exe` terminal chạy
```
.venv\Scripts\activate.bat
```

### Sau khi đã kích hoạt môi trường, cài đặt các thư viện cần thiết
Trong terminal chạy
```bash
pip install -r requirements.txt
```

> Nếu *người sử dụng* muốn chạy các file `*.ipynb` bằng môi trường ảo có được từ hướng dẫn trong file này thì trong terminal:
> Chọn kernel theo `Python Environment` `.venv` được tạo theo chỉ dẫn của file này.