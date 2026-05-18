# Ứng dụng PSO trong tối ưu hệ thống Scalable Cell-Free Massive MIMO

## 1. Giới thiệu

Báo cáo nghiên cứu việc ứng dụng thuật toán **Particle Swarm Optimization (PSO)** để tối ưu tài nguyên trong hệ thống **Scalable Cell-Free Massive MIMO**. Mục tiêu chính là tối ưu phân bổ công suất cho người dùng nhằm cải thiện hiệu năng hệ thống, đặc biệt theo tiêu chí **max-min fairness** — tức là tối đa hóa tốc độ nhỏ nhất trong số các người dùng.

Trong hệ thống Cell-Free Massive MIMO, nhiều trạm thu phát nhỏ (**Access Points - APs**) được phân bố trên diện rộng và phối hợp phục vụ các thiết bị người dùng (**User Equipments - UEs**) mà không dựa trên ranh giới cell truyền thống. Kiến trúc scalable giúp hệ thống dễ mở rộng, giảm tải xử lý tại CPU và giảm yêu cầu băng thông fronthaul.

## 2. Mục tiêu của báo cáo

Báo cáo tập trung vào các mục tiêu sau:

- Trình bày nguyên lý hoạt động của thuật toán PSO.
- Phân tích một số biến thể của PSO như APSO, CPSO, QPSO và lbest-PSO.
- Mô tả hệ thống Scalable Cell-Free Massive MIMO.
- Xây dựng bài toán tối ưu phân bổ công suất cho các UE.
- Áp dụng PSO để tối ưu tốc độ nhỏ nhất của người dùng.
- Mô phỏng, so sánh PSO với phương pháp phân bổ công suất gốc.
- Đánh giá và so sánh hiệu quả của các biến thể PSO.

## 3. Nội dung chính

### 3.1 Thuật toán Particle Swarm Optimization

PSO là thuật toán tối ưu metaheuristic lấy cảm hứng từ hành vi tìm kiếm thức ăn của đàn chim hoặc đàn cá. Mỗi nghiệm ứng viên được biểu diễn bởi một **particle**. Mỗi particle có:

- Vị trí hiện tại `x`
- Vận tốc `v`
- Nghiệm tốt nhất cá nhân `pbest`
- Nghiệm tốt nhất toàn cục `gbest`

Công thức cập nhật cơ bản:

```text
v(t+1) = wv(t) + c1r1(pbest - x(t)) + c2r2(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

Trong đó:

- `w`: hệ số quán tính
- `c1`: hệ số học hỏi cá nhân
- `c2`: hệ số học hỏi xã hội
- `r1`, `r2`: các số ngẫu nhiên trong khoảng `[0, 1]`

PSO phù hợp với các bài toán tối ưu phi tuyến, không lồi và không yêu cầu đạo hàm.

### 3.2 Các biến thể PSO được trình bày

Báo cáo trình bày một số biến thể tiêu biểu:

| Biến thể | Đặc điểm chính |
|---|---|
| APSO | Điều chỉnh trọng số quán tính theo vòng lặp để cân bằng exploration và exploitation |
| CPSO | Sử dụng hệ số co giãn để ổn định quá trình hội tụ |
| Velocity-Clamped PSO | Giới hạn vận tốc để tránh particle di chuyển quá xa |
| QPSO | Cập nhật vị trí dựa trên mô hình xác suất lượng tử, không cần vận tốc |
| lbest-PSO | Mỗi particle học từ nghiệm tốt nhất cục bộ thay vì toàn cục |

## 4. Hệ thống Scalable Cell-Free Massive MIMO

Hệ thống được mô tả gồm:

- `M` Access Points phân bố trên khu vực phủ sóng.
- `K` User Equipments, mỗi UE sử dụng một anten.
- Các AP kết nối về CPU thông qua mạng fronthaul.
- Trong kiến trúc scalable, các AP có thể được gom thành các cụm xử lý cục bộ để giảm độ phức tạp.

### 4.1 Mô hình tín hiệu uplink

Tín hiệu thu tại AP thứ `m`:

```text
y_m = Σ g_mk √p_k s_k + n_m
```

Trong đó:

- `g_mk`: hệ số kênh giữa UE `k` và AP `m`
- `p_k`: công suất phát của UE `k`
- `s_k`: tín hiệu truyền của UE `k`
- `n_m`: nhiễu Gauss trắng cộng

### 4.2 Hiệu suất phổ

Hiệu suất phổ của UE `k`:

```text
SE_k = (1 - τp/τc) log2(1 + SINR_k)
```

Hiệu suất phổ phụ thuộc mạnh vào công suất phát và mức giao thoa giữa các UE.

## 5. Bài toán tối ưu

Bài toán chính là tối ưu phân bổ công suất cho các UE theo tiêu chí **max-min fairness**:

```text
maximize    min_k R_k(p)
subject to  0 ≤ p_k ≤ Pmax
```

Trong đó:

- `p = [p1, p2, ..., pK]`: vector công suất của `K` người dùng
- `R_k`: tốc độ achievable của UE thứ `k`
- `Pmax`: công suất phát cực đại

Đây là bài toán phi tuyến, không lồi, có nhiều cực trị địa phương nên phù hợp để áp dụng PSO.

## 6. Ứng dụng PSO vào bài toán

Trong PSO, mỗi particle biểu diễn một vector phân bổ công suất:

```text
x_i = [p1, p2, ..., pK]
```

Hàm mục tiêu:

```text
f(x_i) = min_k R_k(x_i)
```

Quy trình xử lý:

1. Khởi tạo quần thể particle.
2. Mỗi particle biểu diễn một phương án phân bổ công suất.
3. Tính tốc độ của từng UE.
4. Tính fitness bằng tốc độ nhỏ nhất trong hệ thống.
5. Cập nhật `pbest` và `gbest`.
6. Cập nhật vận tốc và vị trí particle.
7. Chiếu nghiệm về miền hợp lệ `[0, Pmax]`.
8. Lặp lại đến khi đạt số vòng lặp tối đa hoặc hội tụ.

## 7. Cấu hình mô phỏng

Một cấu hình mô phỏng chính trong báo cáo:

| Tham số | Giá trị |
|---|---|
| Số AP | `M = 100` |
| Số UE | `K = 40` |
| Số anten mỗi AP | `1` |
| Công suất phát tối đa | `Pmax = 20 dBm` |
| Nhiễu nền | `σ² = -92 dBm` |
| Khu vực phủ sóng | `1000 × 1000 m²` |
| Kích thước quần thể PSO | `Npop = 30` |
| Số vòng lặp tối đa | `Tmax = 200` |
| Hệ số quán tính | `w = 0.7` |
| Hệ số cá nhân | `c1 = 1.5` |
| Hệ số xã hội | `c2 = 1.5` |

## 8. Kết quả và nhận xét

Kết quả mô phỏng cho thấy:

- PSO giúp giá trị **min-rate** tăng dần theo số vòng lặp.
- Thuật toán hội tụ về giá trị ổn định sau một số vòng lặp.
- PSO cho hiệu năng tốt hơn phương pháp baseline, trong đó tất cả UE phát với công suất cực đại.
- Việc tối ưu công suất bằng PSO cải thiện fairness, đặc biệt cho các UE có điều kiện kênh kém.
- PSO phù hợp với bài toán tối ưu phi tuyến, không lồi và không yêu cầu thông tin gradient.

## 9. So sánh các biến thể PSO

Báo cáo cũng so sánh APSO, CPSO và QPSO trong bài toán tối ưu công suất.

Nhận xét chính:

| Biến thể | Nhận xét |
|---|---|
| APSO | Hội tụ nhanh ở giai đoạn đầu nhờ điều chỉnh trọng số quán tính |
| CPSO | Hội tụ ổn định, ít dao động |
| QPSO | Đạt fitness cao hơn ở giai đoạn cuối, có khả năng tìm nghiệm tốt hơn trong không gian lớn |

Kết luận: các biến thể PSO đều có thể áp dụng cho bài toán Scalable Cell-Free Massive MIMO, nhưng cần lựa chọn tùy theo yêu cầu về tốc độ hội tụ, độ ổn định và chất lượng nghiệm.

## 10. Kết luận

Báo cáo cho thấy thuật toán PSO là một hướng tiếp cận phù hợp cho bài toán tối ưu phân bổ công suất trong hệ thống Scalable Cell-Free Massive MIMO. Nhờ khả năng tìm kiếm toàn cục, không yêu cầu đạo hàm và dễ triển khai, PSO có thể cải thiện tốc độ nhỏ nhất của người dùng và nâng cao tính công bằng trong hệ thống.

Các biến thể nâng cao như QPSO có tiềm năng cải thiện chất lượng nghiệm trong các hệ thống quy mô lớn, trong khi APSO và CPSO có ưu điểm về tốc độ hội tụ và độ ổn định.

## 11. Từ khóa

`Particle Swarm Optimization`, `PSO`, `Scalable Cell-Free Massive MIMO`, `Power Allocation`, `Max-Min Fairness`, `APSO`, `CPSO`, `QPSO`, `Wireless Communication`, `Metaheuristic Optimization`

## 12. Tài liệu tham khảo chính

- Emil Björnson, Luca Sanguinetti, *Scalable Cell-Free Massive MIMO Systems*, 2020.
- Papazafeiropoulos et al., *Scalable Cell-Free Massive MIMO Systems: Impact of Hardware Impairments*, 2021.
- L. Xing et al., *Scalable Statistical Channel Estimation and Its Applications in User-Centric Cell-Free Massive MIMO Systems*, 2025.
- E. Björnson, J. Hoydis, L. Sanguinetti, *Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency*, 2017.
- Q. Sun, *Optimal Power Allocation Based on Metaheuristic Algorithms in Wireless Network*, 2022.
