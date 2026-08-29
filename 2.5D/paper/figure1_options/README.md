# Các phương án bố cục Hình 1

Tất cả phương án được ghép trực tiếp từ 15 ảnh địa hình chỉ số 001 trong
`dataset_5010_v1/images/terrain/`. Không có ảnh GenAI hoặc nội dung địa hình mới.

- `option_a_compact_15_panels.png`: giữ đủ năm họ và ba mức độ khó trong một
  hình, bỏ khoảng trắng và các tiêu đề ngoài bị lặp của contact sheet hiện tại.
- `option_b1_split_first_9_panels.png` và
  `option_b2_split_last_6_panels.png`: giữ đủ 15 ảnh nhưng tách thành hai phần,
  phù hợp khi ưu tiên khả năng đọc hơn số lượng figure.
- `option_c_easy_hard_10_panels.png`: giữ đủ năm họ nhưng chỉ đặt Dễ cạnh Khó;
  mức Trung bình vẫn được mô tả trong Methods và có trong bộ dữ liệu.
- `option_d_clean_15_panels.png`: giữ đủ 15 địa hình, cắt phần lề và tiêu đề lặp
  trong từng ảnh rồi dùng một bộ nhãn hàng/cột chung. Giá trị địa hình, vùng bị
  chặn và Start--Goal vẫn lấy nguyên từ ảnh do pipeline kết xuất.
- `option_d_clean_15_panels_en.png`: cùng bố cục và dữ liệu với phương án D,
  nhưng dùng nhãn cột và chú giải tiếng Anh cho bản thảo nộp chính thức.

Chạy lại bằng:

```bash
sh make_figure1_options.sh
```
