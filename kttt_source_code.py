import numpy as np
import matplotlib.pyplot as plt

# --- 1. Cấu hình thông số hệ thống (Chương 6.1) ---
M = 100          # Số lượng Access Points (APs)
K = 40           # Số lượng Users (UEs)
P_MAX_DBM = 20   # Công suất phát cực đại (dBm)
P_MAX = 10**(P_MAX_DBM / 10) / 1000 # Chuyển sang Watt
NOISE_VAR_DBM = -92    #Nhiễu nền AWGN tại mỗi AP(dBm)
NOISE_VAR = 10**(NOISE_VAR_DBM / 10) / 1000
AREA_SIZE = 1000 # Kích thước khu vực (m x m)

# Thông số PSO (Chương 5.4)
N_POP = 30       # Kích thước quần thể
T_MAX = 200       # Số vòng lặp tối đa
W = 0.7          # Hệ số quán tính
C1 = 1.5         # Hệ số nhận thức
C2 = 1.5         # Hệ số xã hội

# --- 2. Mô hình kênh và tính toán tốc độ
def generate_system():
    # Tọa độ ngẫu nhiên APs và UEs
    ap_pos = np.random.uniform(0, AREA_SIZE, (M, 2))
    ue_pos = np.random.uniform(0, AREA_SIZE, (K, 2))
    
    # Tính Large-scale fading (Path loss đơn giản)
    beta = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            dist = np.linalg.norm(ap_pos[m] - ue_pos[k])
            dist = max(dist, 10) # Tránh khoảng cách quá gần gây lỗi log
            # Model: PL = -136.7 - 35 * log10(d)
            path_loss_db = -136.7 - 35 * np.log10(dist/1000)
            beta[m, k] = 10**(path_loss_db / 10)
    return beta

def calculate_rates(powers, beta):
    """
    Tính Achievable Rate Rk cho mỗi UE (Uplink MRC)
    powers: vector công suất [p1, p2, ..., pK]
    """
    K_users = len(powers)
    rates = np.zeros(K_users)
    
    # Giả sử sử dụng bộ thu MRC (Maximum Ratio Combining) đơn giản
    # Công thức rút gọn cho SE trong Cell-Free Massive MIMO
    for k in range(K_users):
        # Tín hiệu mong muốn
        signal = powers[k] * (np.sum(beta[:, k]))**2
        
        # Nhiễu cộng và nhiễu liên thuê bao
        interference = 0
        for i in range(K_users):
            if i != k:
                interference += powers[i] * np.sum(beta[:, k] * beta[:, i])
        
        noise = NOISE_VAR * np.sum(beta[:, k])
        
        sinr = signal / (interference + noise)
        rates[k] = np.log2(1 + sinr) # bit/s/Hz
        
    return rates

def fitness_function(powers, beta):
    # Mục tiêu: Max-min Fairness (Chương 5.3)
    # Tối đa hóa giá trị nhỏ nhất của Rk
    rates = calculate_rates(powers, beta)
    return np.min(rates)

# --- 3. Thuật toán PSO (Chương 5.5) ---
def pso_optimization(beta):
    # Khởi tạo vị trí (công suất) và vận tốc
    particles_pos = np.random.uniform(0, P_MAX, (N_POP, K))
    particles_vel = np.zeros((N_POP, K))
    
    # Khởi tạo cá nhân tốt nhất và toàn cục tốt nhất
    pbest_pos = np.copy(particles_pos)
    pbest_fit = np.array([fitness_function(p, beta) for p in pbest_pos])
    
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fit = pbest_fit[gbest_idx]
    
    history = []

    for t in range(T_MAX):
        for i in range(N_POP):
            # Cập nhật vận tốc (Công thức 2.1)
            r1, r2 = np.random.rand(K), np.random.rand(K)
            particles_vel[i] = (W * particles_vel[i] + 
                                C1 * r1 * (pbest_pos[i] - particles_pos[i]) + 
                                C2 * r2 * (gbest_pos - particles_pos[i]))
            
            # Cập nhật vị trí (Công thức 2.2)
            particles_pos[i] += particles_vel[i]
            
            # Chiếu vào miền hợp lệ [0, P_MAX] (Chương 5.4)
            particles_pos[i] = np.clip(particles_pos[i], 0, P_MAX)
            
            # Đánh giá Fitness
            current_fit = fitness_function(particles_pos[i], beta)
            
            # Cập nhật pbest
            if current_fit > pbest_fit[i]:
                pbest_fit[i] = current_fit
                pbest_pos[i] = np.copy(particles_pos[i])
                
        # Cập nhật gbest
        if np.max(pbest_fit) > gbest_fit:
            gbest_idx = np.argmax(pbest_fit)
            gbest_pos = np.copy(pbest_pos[gbest_idx])
            gbest_fit = pbest_fit[gbest_idx]
            
        history.append(gbest_fit)
        print(f"Iteration {t+1}/{T_MAX}, Max-Min Rate: {gbest_fit:.4f}")

    return gbest_pos, gbest_fit, history

# --- 4. Chạy mô phỏng và Hiển thị kết quả ---
print("Đang khởi tạo hệ thống...")
beta_matrix = generate_system()

print("\n--- Bắt đầu tối ưu hóa PSO ---")
best_powers, best_min_rate, convergence = pso_optimization(beta_matrix)

# Tính kết quả gốc (Phân bổ công suất tối đa - Baseline)
baseline_powers = np.ones(K) * P_MAX
baseline_rates = calculate_rates(baseline_powers, beta_matrix)
baseline_min_rate = np.min(baseline_rates)

print(f"\nKẾT QUẢ:")
print(f"Max-min Rate (Gốc - Full Power): {baseline_min_rate:.4f} bit/s/Hz")
print(f"Max-min Rate (Sau PSO tối ưu): {best_min_rate:.4f} bit/s/Hz")
print(f"Cải thiện: {((best_min_rate - baseline_min_rate)/baseline_min_rate)*100:.2f}%")

# Vẽ đồ thị hội tụ
plt.figure(figsize=(10, 5))
plt.plot(convergence, label='PSO Optimized Min-Rate', color='blue', linewidth=2)
plt.axhline(y=baseline_min_rate, color='red', linestyle='--', label='Baseline (Full Power)')
plt.xlabel('Số vòng lặp (Iteration)')
plt.ylabel('Min-Rate (bit/s/Hz)')
plt.title('Đồ thị hội tụ của thuật toán PSO trong hệ thống Scalable Cell-Free')
plt.legend()
plt.grid(True)
plt.show()