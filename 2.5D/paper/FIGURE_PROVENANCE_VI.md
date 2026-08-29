# Nguồn gốc hình trong bản LaTeX tiếng Việt

Gói Overleaf tiếng Việt chỉ chứa hình được kết xuất trực tiếp từ dữ liệu và kết
quả benchmark thật của dự án. Không hình nào được tạo bằng GenAI.

| Tệp trong ZIP | Nguồn | Cách tạo |
|---|---|---|
| `figures/terrain_triplet_001.png` | 15 tệp `dataset_5010_v1/images/terrain/<family>/terrain_<family>_001_<difficulty>.png` | Mỗi ô là ảnh do pipeline kết xuất trực tiếp từ map JSON tương ứng. `make_figure1_options.sh` và `make_figure1_options.m` chỉ cắt lề, bỏ tiêu đề/chú giải lặp rồi ghép thành lưới 5 × 3 với nhãn chung; không sửa dữ liệu địa hình, vùng bị chặn hoặc Start--Goal. |
| `figures/dataset_difficulty_overview.png` | `dataset_5010_v1/metadata/benchmark_results/figures/dataset_difficulty_overview.png` | Biểu đồ do `analyze_25d_benchmark_results.py` tổng hợp từ metadata của toàn bộ 5.010 map. |
| Sáu tệp `figures/*_mountain_hard.png` | `dataset_5010_v1/images/paths/mountain/terrain_mountain_001_hard_<planner>.png` | Sáu ảnh đường đi do BFS, Dijkstra, A*, PRM, RRT-Connect và RRT* cùng renderer của dự án tạo trên một map và Start--Goal. |
| Ba tệp `figures/analysis_*_hard.png` | `dataset_5010_v1/images/analysis/<family>/terrain_<family>_001_hard_analysis.png` | Ảnh analysis thật của smooth-obstacles, rugged và plateau, gồm elevation, traversability cost và nguyên nhân blocking. |

Mười một tệp PNG đều đã có trong bản phát hành dữ liệu được kiểm định và được
dùng để dựng bốn figure trong bản thảo. Cấu trúc thư
mục được trình bày bằng môi trường `verbatim` trong LaTeX, không chuyển thành
ảnh minh họa.
