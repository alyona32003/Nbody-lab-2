import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import random

from nbody import generate_cloud, run_leapfrog

# Параметры
SEED = 21
G = 1.0
M_total = 1000.0
N_particles = 2000
R_CLOUD = 11.0
DISTRIBUTION = 0
FLUCTUATION = 0.009
OMEGA = 1.01
softening = 0.5
dt = 0.005
STEPS_PER_FRAME = 5
N_FRAMES = 500
FPS = 40
THETA = 0.5

def simulate_and_save_gif(path):
    random.seed(SEED)
    np.random.seed(SEED)

    pos, vel = generate_cloud(N_particles, R_CLOUD, OMEGA, FLUCTUATION, DISTRIBUTION)
    m_array = np.full(N_particles, M_total / N_particles)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, facecolor='black')
    ax.set_facecolor('black')

    lim = R_CLOUD * 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)

    scatter = ax.scatter(pos[:, 0], pos[:, 1], s=1.5, c='white', alpha=0.7, lw=0)

    p, v = pos.copy(), vel.copy()
    pil_frames = []

    print(f"{N_particles} particles, dist={DISTRIBUTION}, fluct={FLUCTUATION}, omega={OMEGA}")
    start_time = time.time()

    for f in range(N_FRAMES):
        frame_start = time.time()
        p, v = run_leapfrog(p, v, m_array, dt, G, softening, STEPS_PER_FRAME, THETA, verbose=(f == 0))

        if not np.all(np.isfinite(p)):
            print(f"\nNaN at frame {f}, stopping")
            break

        scatter.set_offsets(p[:, :2])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pil_frames.append(Image.fromarray(buf))

        elapsed = time.time() - start_time
        frame_time = time.time() - frame_start
        eta = frame_time * (N_FRAMES - f - 1)
        sys.stdout.write(
            f"\r  frame {f+1:3d}/{N_FRAMES} | "
            f"{frame_time:4.2f}s/frame | "
            f"elapsed {elapsed:5.1f}s | "
            f"eta ~{eta:5.1f}s"
        )
        sys.stdout.flush()

    print("\ndone")
    plt.close(fig)

    duration_ms = int(round(1000.0 / FPS))
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, disposal=2,
    )
    print(f"saved: {path}")

def show_gif(path, fps):
    gif = Image.open(path)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)

    img = ax.imshow(np.array(gif.convert('RGB')))

    def update(i):
        gif.seek(i % gif.n_frames)
        img.set_data(np.array(gif.convert('RGB')))
        return img,

    ani = animation.FuncAnimation(
        fig, update, frames=gif.n_frames,
        interval=1000.0 / fps, blit=True,
    )
    plt.show()

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    out_path = "results/nbody_collapse.gif"
    simulate_and_save_gif(out_path)
    show_gif(out_path, FPS)