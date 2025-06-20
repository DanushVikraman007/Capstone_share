import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from scipy.ndimage import sobel

# Load and normalize grayscale image
img_path = r"C:\Users\user\Desktop\CAPSTONE\brain\1a8106f3-e4ca-4e18-a275-065d020704f4.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Black‐pixel threshold
black_threshold = 0.3

# RGB version of the background for plotting
gray_background = np.stack([img] * 3, axis=-1)

# HMC parameters
patch_size = 90
alpha, beta = 0.6, 0.4
eps = 0.4
leapfrog_steps = 30
n_samples = 100000
burn_in = int(0.2 * n_samples)

# Build smoothed intensity and gradient fields
kernel = np.ones((patch_size, patch_size)) / (patch_size ** 2)
I_smooth = cv2.filter2D(img, -1, kernel)
Gx = sobel(img, axis=0)
Gy = sobel(img, axis=1)
G_mag = np.hypot(Gx, Gy)
G_smooth = cv2.filter2D(G_mag, -1, kernel)

# Potential and its gradient
U = -(alpha * G_smooth + beta * I_smooth)
grad_U_y, grad_U_x = np.gradient(U)

H, W = img.shape

def reflect(val, max_val):
    return max(0, min(val, max_val - 1))

def Hval(xp, yp, p_x, p_y):
    Uloc = np.interp(xp, np.arange(W), U[int(yp), :])
    return Uloc + 0.5 * (p_x ** 2 + p_y ** 2)

# Initialize chain
x, y = np.random.uniform(0, W), np.random.uniform(0, H)

accepted = []
rejected = []

# --- HMC Sampling over the entire image domain ---
for i in range(n_samples):
    p_x, p_y = np.random.randn(), np.random.randn()
    x_new, y_new = x, y
    p_x_new, p_y_new = p_x, p_y

    gx = np.interp(x_new, np.arange(W), grad_U_x[int(y_new), :])
    gy = np.interp(y_new, np.arange(H), grad_U_y[:, int(x_new)])
    p_x_new -= 0.5 * eps * gx
    p_y_new -= 0.5 * eps * gy

    for lf in range(leapfrog_steps):
        x_new += eps * p_x_new
        y_new += eps * p_y_new
        x_new = reflect(x_new, W)
        y_new = reflect(y_new, H)

        gx = np.interp(x_new, np.arange(W), grad_U_x[int(y_new), :])
        gy = np.interp(y_new, np.arange(H), grad_U_y[:, int(x_new)])
        if lf != leapfrog_steps - 1:
            p_x_new -= eps * gx
            p_y_new -= eps * gy

    p_x_new -= 0.5 * eps * gx
    p_y_new -= 0.5 * eps * gy

    xi = int(np.clip(round(x_new), 0, W - 1))
    yi = int(np.clip(round(y_new), 0, H - 1))

    H_curr = Hval(x, y, p_x, p_y)
    H_prop = Hval(x_new, y_new, p_x_new, p_y_new)
    accept = (np.random.rand() < np.exp(H_curr - H_prop))
    if accept:
        x, y = x_new, y_new

    if i >= burn_in:
        if img[yi, xi] >= black_threshold:
            rejected.append((yi, xi))
        elif accept:
            accepted.append((yi, xi))
        else:
            rejected.append((yi, xi))

# === Static Heatmap with Controlled Green Intensity ===
green_intensity_scale = 2  # Adjust green brightness here

heatmap = np.zeros((*img.shape, 3), dtype=np.float32)
for yi, xi in accepted:
    intensity = np.clip(green_intensity_scale * img[yi, xi], 0, 1)
    heatmap[yi, xi] = [0, intensity, 0]  # Green with intensity scaling


plt.figure(figsize=(12, 6))
plt.title("Accepted (Green) vs. Rejected (White) Samples")
plt.imshow(gray_background)
plt.imshow(heatmap, alpha=0.8)
plt.axis('off')
plt.tight_layout()
plt.show()

# === Animation with Intensity Scaling ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("HMC Sampling Animation\n(energy rejections)")
ax.imshow(gray_background)
scat = ax.scatter([], [], s=2, alpha=0.9)
ax.axis('off')

all_pts = []
all_cols = []

def update(frame):
    if frame < len(accepted):
        yi, xi = accepted[frame]
        intensity = np.clip(green_intensity_scale , 0, 1)
        all_pts.append((xi, yi))
        all_cols.append((0, intensity, 0))
    else:
        yi, xi = rejected[frame - len(accepted)]
        all_pts.append((xi, yi))
        all_cols.append((1, 1, 1))
    scat.set_offsets(np.array(all_pts))
    scat.set_color(all_cols)
    return scat,

ani = animation.FuncAnimation(
    fig, update,
    frames=len(accepted) + len(rejected),
    interval=1, blit=True
)
plt.show()
