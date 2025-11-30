import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1) Bài toán tối ưu hoá
# =============================
# Ta sẽ thử dùng PSO để: 
# Tìm giá trị x,y để f(x,y) nhỏ nhất


#f(x,y)=20+x^2−10cos(2πx)+y^2−10cos(2πy)
#nghiem dung la f(0,0) = 0 -> PSO se co gang tien gan ve (0,0)
def rastrigin(x): 
    A = 10
    return 2 * A + (x[0]**2 - A * np.cos(2 * np.pi * x[0])) \
                 + (x[1]**2 - A * np.cos(2 * np.pi * x[1]))


# =============================
# 2) Cài đặt PSO
# =============================

class PSO:
    def __init__(self, func, dim, bounds, pop_size=30,
                 w=0.7, c1=1.5, c2=1.5, max_iter=200):
        
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        self.lb = np.array([bounds[0]] * dim)
        self.ub = np.array([bounds[1]] * dim)

        # Khởi tạo
        self.pos = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.vel = np.zeros((pop_size, dim))

        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.array([self.func(p) for p in self.pos])

        g_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[g_idx].copy()
        self.gbest_val = self.pbest_val[g_idx]

        self.loss_curve = []

    def step(self):

        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)

        cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
        social    = self.c2 * r2 * (self.gbest_pos - self.pos)

        self.vel = self.w * self.vel + cognitive + social
        self.pos = self.pos + self.vel

        self.pos = np.clip(self.pos, self.lb, self.ub)

        # cập nhật pbest
        for i in range(self.pop_size):
            val = self.func(self.pos[i])
            if val < self.pbest_val[i]:
                self.pbest_val[i] = val
                self.pbest_pos[i] = self.pos[i].copy()

        # cập nhật gbest
        g_idx = np.argmin(self.pbest_val)
        if self.pbest_val[g_idx] < self.gbest_val:
            self.gbest_val = self.pbest_val[g_idx]
            self.gbest_pos = self.pbest_pos[g_idx].copy()

        self.loss_curve.append(self.gbest_val)

    def run(self):
        for t in range(self.max_iter):
            self.step()
            if t % 20 == 0:
                print(f"Iter {t:4d}  ->  Best = {self.gbest_val:.6f}")
        return self.gbest_pos, self.gbest_val


# =============================
# 3) Chạy thử với bài toán
# =============================

if __name__ == "__main__":

    pso = PSO(
        func=rastrigin,
        dim=2,
        bounds=(-5.12, 5.12),
        pop_size=40,
        max_iter=300
    )

    best_x, best_val = pso.run()

    print("\n===== KẾT QUẢ CUỐI =====")
    print("Điểm tối ưu tìm được:", best_x)
    print("Giá trị hàm:", best_val)

    # vẽ đường hội tụ
    plt.plot(pso.loss_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("PSO optimization on Rastrigin function")
    plt.grid(True)
    plt.show()
