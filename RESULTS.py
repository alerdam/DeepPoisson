import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# --- Configuration ---
SIZE = 50
MODEL_PATH = "poisson_model.tflite"

# Dark Mode Styling
plt.style.use('dark_background')
ACCENT_COLOR = '#00FFCC' # Neon Cyan

# Initialize Interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def solve_poisson_python(v_top, v_bot, src_x, src_y, charge, iterations=2000):
    V = np.zeros((SIZE, SIZE), dtype=np.float32)
    f = np.zeros((SIZE, SIZE), dtype=np.float32)
    V[0, :] = v_top
    V[SIZE-1, :] = v_bot
    f[int(src_x), int(src_y)] = charge
    
    start_time = time.time()
    for _ in range(iterations):
        V_new = V.copy()
        # Vectorized Jacobi update
        V_new[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] + 
                                   V[1:-1, 2:] + V[1:-1, :-2] - f[1:-1, 1:-1])
        V = V_new
    duration = (time.time() - start_time) * 1000
    return V, duration

# --- UI Setup ---
fig, axes = plt.subplots(1, 3, figsize=(16, 8), facecolor='#121212')
plt.subplots_adjust(bottom=0.3, top=0.85, wspace=0.3)

# Initial setup for images
v_t, v_b, s_x, s_y, chg = 100.0, 0.0, 25, 25, -25.0
dummy_data = np.zeros((SIZE, SIZE))

ims = []
titles = ["Boundary Conditions", "Jacobi Solver (Ground Truth)", "Neural Network Prediction"]
for i in range(3):
    im = axes[i].imshow(dummy_data, cmap='magma', vmin=0, vmax=100, interpolation='bilinear')
    axes[i].set_title(titles[i], color='white', fontsize=12, pad=15)
    axes[i].axis('off')
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    ims.append(im)

# Error Text Setup
info_text = fig.text(0.5, 0.26, "", ha='center', color=ACCENT_COLOR, fontsize=11, fontweight='bold')

def update(val):
    vt, vb = s_vtop.val, s_vbot.val
    sx, sy = int(s_srcx.val), int(s_srcy.val)
    c = s_chg.val
    
    # 1. Input Prep
    input_grid = np.zeros((1, SIZE, SIZE, 1), dtype=np.float32)
    input_grid[0, 0, :, 0], input_grid[0, SIZE-1, :, 0] = vt, vb
    input_grid[0, sx, sy, 0] = c
    
    # 2. Numerical Physics
    target_data, target_ms = solve_poisson_python(vt, vb, sx, sy, c)
    
    # 3. AI Inference
    ai_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_grid)
    interpreter.invoke()
    ai_prediction = interpreter.get_tensor(output_details[0]['index'])[0].reshape(SIZE, SIZE)
    ai_ms = (time.time() - ai_start) * 1000

    # 4. Error Metrics
    mae = np.mean(np.abs(target_data - ai_prediction))
    max_err = np.max(np.abs(target_data - ai_prediction))
    speedup = target_ms / ai_ms if ai_ms > 0 else 0

    # 5. Update Visuals
    ims[0].set_data(input_grid[0].reshape(SIZE, SIZE))
    ims[1].set_data(target_data)
    ims[2].set_data(ai_prediction)

    # Update dynamic labels
    axes[1].set_title(f"Jacobi Solver\n{target_ms:.1f} ms", color='white')
    axes[2].set_title(f"Neural Network\n{ai_ms:.2f} ms", color=ACCENT_COLOR)
    
    fig.suptitle(f"AI Speedup: {speedup:.1f}x Faster", fontsize=20, color=ACCENT_COLOR)
    info_text.set_text(f"Mean Absolute Error: {mae:.4f}  |  Max Error: {max_err:.4f}")
    
    fig.canvas.draw_idle()

# --- Sliders Design ---
slider_color = '#333333'
ax_vtop = plt.axes([0.2, 0.18, 0.6, 0.02], facecolor=slider_color)
ax_vbot = plt.axes([0.2, 0.15, 0.6, 0.02], facecolor=slider_color)
ax_srcx = plt.axes([0.2, 0.12, 0.6, 0.02], facecolor=slider_color)
ax_srcy = plt.axes([0.2, 0.09, 0.6, 0.02], facecolor=slider_color)
ax_chg  = plt.axes([0.2, 0.06, 0.6, 0.02], facecolor=slider_color)

s_vtop = Slider(ax_vtop, 'V-Top ', 0, 100, valinit=v_t, color=ACCENT_COLOR)
s_vbot = Slider(ax_vbot, 'V-Bot ', 0, 100, valinit=v_b, color=ACCENT_COLOR)
s_srcx = Slider(ax_srcx, 'Src X ', 1, 48, valinit=s_x, valfmt='%0.0f', color=ACCENT_COLOR)
s_srcy = Slider(ax_srcy, 'Src Y ', 1, 48, valinit=s_y, valfmt='%0.0f', color=ACCENT_COLOR)
s_chg  = Slider(ax_chg,  'Charge ', -50, 50, valinit=chg, color=ACCENT_COLOR)

for s in [s_vtop, s_vbot, s_srcx, s_srcy, s_chg]:
    s.on_changed(update)

update(None)
plt.show()