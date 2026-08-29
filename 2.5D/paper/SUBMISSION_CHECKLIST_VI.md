# Checklist nộp Scientific Data

## Đã có trong dự án

- 5.010 tệp JSON bản đồ, 50.100 bản ghi bản đồ--nhiệm vụ và 90.180 lần chạy benchmark.
- README, lược đồ, bộ đọc mẫu, môi trường ghim phiên bản và kiểm thử hồi quy.
- Báo cáo kiểm định đọc được bằng máy, danh mục tệp và mã kiểm tra SHA-256.
- Bản thảo đủ các heading bắt buộc của loại bài Data Descriptor.
- Tài liệu tham khảo nhúng trực tiếp trong một tệp `.tex`.
- Mười sáu tài liệu học thuật đã được đối chiếu, gồm nguyên tắc FAIR, phần mềm,
  nguồn gốc planner và tám bộ dữ liệu/benchmark off-road liên quan gần nhất,
  cùng trích dẫn dữ liệu Zenodo chính thức.
- Bảng so sánh được giới hạn theo modality thu thập, quy mô phát hành, điều kiện
  khả năng di chuyển, độ khó ghép cặp, thành phần lập kế hoạch và hỗ trợ phát
  hành; không coi frame nhận thức là tương đương với bản đồ điều hướng.
- Methods đã nêu đầy đủ phân phối từng họ, mọi tham số trường chi tiết, lưới 34
  giá trị độ khó chính xác, metric và điểm chấp nhận, footprint số hóa và xử lý
  biên, fallback nhiệm vụ xác định cùng thuật toán sinh bộ ba đầu cuối.
- Bốn hình được nhúng trong PDF từ mười một PNG, đều là sản phẩm thật của quy trình dữ liệu hoặc
  benchmark; không dùng ảnh GenAI.
- Giấy phép dữ liệu đã chốt là CC BY 4.0; mã nguồn dùng giấy phép MIT.
- Nhãn schema lịch sử `3.1-preview` được chủ ý giữ lại và đã được mô tả cho bản
  phát hành dữ liệu bất biến này.
- Cấu hình tạo release đã được ghi trong Methods: MacBook Pro (Mac16,1), Apple
  M4 10 lõi CPU, bộ nhớ hợp nhất 16 GB, macOS 26.5.2; sinh dữ liệu dùng tối đa
  sáu tiến trình xử lý và benchmark dùng ba tiến trình.

## Bắt buộc xử lý trước khi nộp vòng 1

- Publish bản ghi Zenodo đã tải lên với DOI dự phòng
  `10.5281/zenodo.22074838`; kho dữ liệu, DOI và trích dẫn dữ liệu chính thức đã
  được thêm vào cả hai bản thảo.
- Đẩy tag mã nguồn `v1.0.0` đã chuẩn bị và có thể lưu thêm trên Zenodo để nhận
  DOI phần mềm trước khi nộp.
- Xác nhận tên khoa/bộ môn, tên tác giả, ORCID và email liên hệ.
- Tuyên bố Author Contributions đã ghi nhận T.Q.M. thực hiện toàn bộ công việc;
  Funding đã ghi không có tài trợ bên ngoài. Tuyên bố Competing Interests hiện
  ghi không có xung đột; cần đổi nếu có lợi ích tài chính, quan hệ nghề nghiệp
  hoặc quan hệ cá nhân có thể được xem là ảnh hưởng đến nghiên cứu.
- Có thể lưu mã trên Zenodo để nhận DOI phần mềm sau khi đã chốt đúng phiên bản.
- Nếu muốn so sánh runtime như độ trễ đơn luồng giữa các máy, chạy thêm timing
  một worker; số liệu hiện tại chỉ đại diện cho thông lượng ba worker trên máy
  đã ghi trong Methods.

## Khi nộp

- Vòng đầu có thể nộp một PDF chính đã nhúng đủ hình và bảng; dữ liệu phải tải
  được qua URL ẩn danh hoặc kho chính thức.
- Bản sửa bằng LaTeX phải là một `.tex` độc lập, không phụ thuộc `.bib`, `.bbl`
  hoặc style riêng. Hệ thống nộp bài yêu cầu upload hình riêng ở vòng sửa.
- Cover letter là tệp kỹ thuật bắt buộc nhưng không cần trình bày độ mới/ảnh
  hưởng; hướng dẫn truy cập dữ liệu phải nằm trong bài, không chỉ ở cover letter.
- Bản nộp chính thức phải là tiếng Anh; bản tiếng Việt này dùng để duyệt nội dung.
