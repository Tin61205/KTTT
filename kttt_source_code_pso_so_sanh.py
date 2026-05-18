import numpy as np
import matplotlib.pyplot as plt

# --- 1. Cấu hình Hệ thống (Giữ nguyên từ mô hình MIMO) ---
M, K = 100, 40
P_MAX_DBM = 20
P_MAX = 10**(P_MAX_DBM / 10) / 1000
NOISE_VAR = 10**(-92 / 10) / 1000
T_MAX = 60
N_POP = 30

def generate_system():
    beta = np.random.exponential(1e-10, (M, K)) # Giả lập nhanh ma trận kênh
    return beta

def calculate_fitness(powers, beta):
    # Max-min Fairness (Tối ưu người dùng yếu nhất)
    K_users = len(powers)
    rates = np.zeros(K_users)
    for k in range(K_users):
        signal = powers[k] * (np.sum(beta[:, k]))**2
        interf = sum(powers[i] * np.sum(beta[:, k] * beta[:, i]) for i in range(K_users) if i != k)
        noise = NOISE_VAR * np.sum(beta[:, k])
        rates[k] = np.log2(1 + signal / (interf + noise))
    return np.min(rates)

# --- 2. Các biến thể PSO ---

def run_pso_variants(beta):
    # Khởi tạo quần thể dùng chung cho tất cả biến thể để công bằng
    initial_pos = np.random.uniform(0, P_MAX, (N_POP, K))
    
    results = {}

    #ADAPTIVE PSO (APSO)
    def apso():
        pos = np.copy(initial_pos)
        vel = np.zeros((N_POP, K))
        pbest_pos = np.copy(pos)
        pbest_fit = np.array([calculate_fitness(p, beta) for p in pos])
        gbest_pos = pbest_pos[np.argmax(pbest_fit)]
        gbest_fit = np.max(pbest_fit)
        
        history = []
        w_max, w_min = 0.9, 0.4
        c1, c2 = 1.5, 1.5
        v_max = 0.1 * P_MAX # Velocity Clamping - Chương 3.3

        for t in range(T_MAX):
            w = w_max - (w_max - w_min) * (t / T_MAX) # w giảm dần
            for i in range(N_POP):
                r1, r2 = np.random.rand(K), np.random.rand(K)
                vel[i] = w * vel[i] + c1*r1*(pbest_pos[i] - pos[i]) + c2*r2*(gbest_pos - pos[i])
                vel[i] = np.clip(vel[i], -v_max, v_max) # Clamping
                pos[i] = np.clip(pos[i] + vel[i], 0, P_MAX)
                
                fit = calculate_fitness(pos[i], beta)
                if fit > pbest_fit[i]:
                    pbest_fit[i], pbest_pos[i] = fit, np.copy(pos[i])
            
            if np.max(pbest_fit) > gbest_fit:
                gbest_fit = np.max(pbest_fit)
                gbest_pos = np.copy(pbest_pos[np.argmax(pbest_fit)])
            history.append(gbest_fit)
        return history

    #CPSO
    def cpso():
        pos = np.copy(initial_pos)
        vel = np.zeros((N_POP, K))
        c1, c2 = 2.05, 2.05
        phi = c1 + c2
        chi = 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi)) # Hệ số co giãn
        
        pbest_pos = np.copy(pos)
        pbest_fit = np.array([calculate_fitness(p, beta) for p in pos])
        gbest_fit = np.max(pbest_fit)
        gbest_pos = pbest_pos[np.argmax(pbest_fit)]
        
        history = []
        for t in range(T_MAX):
            for i in range(N_POP):
                r1, r2 = np.random.rand(K), np.random.rand(K)
                vel[i] = chi * (vel[i] + c1*r1*(pbest_pos[i] - pos[i]) + c2*r2*(gbest_pos - pos[i]))
                pos[i] = np.clip(pos[i] + vel[i], 0, P_MAX)
                
                fit = calculate_fitness(pos[i], beta)
                if fit > pbest_fit[i]:
                    pbest_fit[i], pbest_pos[i] = fit, np.copy(pos[i])
            
            if np.max(pbest_fit) > gbest_fit:
                gbest_fit = np.max(pbest_fit)
                gbest_pos = np.copy(pbest_pos[np.argmax(pbest_fit)])
            history.append(gbest_fit)
        return history

    #QPSO
    def qpso():
        pos = np.copy(initial_pos)
        pbest_pos = np.copy(pos)
        pbest_fit = np.array([calculate_fitness(p, beta) for p in pos])
        gbest_fit = np.max(pbest_fit)
        gbest_pos = pbest_pos[np.argmax(pbest_fit)]
        
        history = []
        for t in range(T_MAX):
            alpha = 0.8 - 0.3 * (t / T_MAX) # Hệ số điều khiển alpha
            mbest = np.mean(pbest_pos, axis=0) # Trọng tâm quần thể
            
            for i in range(N_POP):
                phi = np.random.rand(K)
                p_local = phi * pbest_pos[i] + (1 - phi) * gbest_pos
                u = np.random.rand(K)
                
                # Cập nhật vị trí theo hàm delta lượng tử
                if np.random.rand() > 0.5:
                    pos[i] = p_local + alpha * np.abs(mbest - pos[i]) * np.log(1/u)
                else:
                    pos[i] = p_local - alpha * np.abs(mbest - pos[i]) * np.log(1/u)
                
                pos[i] = np.clip(pos[i], 0, P_MAX)
                fit = calculate_fitness(pos[i], beta)
                if fit > pbest_fit[i]:
                    pbest_fit[i], pbest_pos[i] = fit, np.copy(pos[i])

            if np.max(pbest_fit) > gbest_fit:
                gbest_fit = np.max(pbest_fit)
                gbest_pos = np.copy(pbest_pos[np.argmax(pbest_fit)])
            history.append(gbest_fit)
        return history

    print("Đang chạy APSO...")
    results['APSO (Adaptive)'] = apso()
    print("Đang chạy CPSO...")
    results['CPSO (Constriction)'] = cpso()
    print("Đang chạy QPSO...")
    results['QPSO (Quantum)'] = qpso()
    
    return results

# --- 3. Thực thi và Vẽ đồ thị so sánh ---
beta_mat = generate_system()
all_results = run_pso_variants(beta_mat)

plt.figure(figsize=(12, 6))
for label, hist in all_results.items():
    plt.plot(hist, label=label, linewidth=2)

plt.title('So sánh các biến thể PSO trong tối ưu Scalable Cell-Free MIMO', fontsize=14)
plt.xlabel('Vòng lặp (Iteration)', fontsize=12)
plt.ylabel('Min-Rate (bit/s/Hz)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()