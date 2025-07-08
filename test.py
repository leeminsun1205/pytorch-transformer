import numpy as np
import matplotlib.pyplot as plt

# Tạo đầu vào ngẫu nhiên
x = np.linspace(-1, 1, 100)

# Một lớp đơn giản: hàm phi tuyến
def layer(x):
    return np.tanh(3 * x)

# Mạng không residual: x → layer → layer → layer
def no_residual(x):
    out = layer(x)
    out = layer(out)
    out = layer(out)
    return out

# Mạng có residual: x → layer + x → layer + x → layer + x
def with_residual(x):
    out = layer(x) + x
    out = layer(out) + x
    out = layer(out) + x
    return out

# Tính giá trị
y1 = no_residual(x)
y2 = with_residual(x)

# Vẽ đồ thị
plt.plot(x, y1, label='No Residual', linestyle='--')
plt.plot(x, y2, label='With Residual', linestyle='-')
plt.plot(x, x, label='Input x', color='gray', alpha=0.5)
plt.legend()
plt.title("Residual vs No Residual")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
