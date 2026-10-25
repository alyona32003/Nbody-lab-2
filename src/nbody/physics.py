import numpy as np
from numba import njit

def generate_cloud(N, R, omega, fluctuation, dist_type):
    if dist_type == 0:
        r = R * np.random.rand(N)**(1/3)
        theta = np.arccos(2.0 * np.random.rand(N) - 1.0)
        phi = 2.0 * np.pi * np.random.rand(N)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    else:
        lim = R * 2.5
        x = np.random.uniform(-lim, lim, N)
        y = np.random.uniform(-lim, lim, N)
        z = np.random.uniform(-lim, lim, N)

    pos = np.column_stack((x, y, z))
    vx = -omega * y
    vy = omega * x
    vz = np.zeros(N)
    vel = np.column_stack((vx, vy, vz))

    if fluctuation > 0:
        pos += fluctuation * np.random.randn(N, 3) * R
        vel += fluctuation * np.random.randn(N, 3) * (omega * R + 0.1)

    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    return pos, vel

def build_flat_tree(pos, m_array):
    max_nodes = len(m_array) * 20
    mass = np.zeros(max_nodes)
    cm = np.zeros((max_nodes, 3))
    size = np.zeros(max_nodes)
    children = np.full((max_nodes, 8), -1, dtype=np.int32)
    is_leaf = np.zeros(max_nodes, dtype=np.bool_)
    body_idx = np.full(max_nodes, -1, dtype=np.int32)

    node_count = 1
    stack = [(0, np.arange(len(m_array)))]

    while stack:
        node, indices = stack.pop()
        mass[node] = np.sum(m_array[indices])
        cm[node] = np.average(pos[indices], axis=0, weights=m_array[indices])

        min_b = np.min(pos[indices], axis=0)
        max_b = np.max(pos[indices], axis=0)
        size[node] = np.max(max_b - min_b)

        if len(indices) == 1:
            is_leaf[node] = True
            body_idx[node] = indices[0]
        else:
            center = (min_b + max_b) / 2.0
            octants = [[] for _ in range(8)]
            for idx in indices:
                oct_idx = 0
                if pos[idx][0] > center[0]: oct_idx |= 1
                if pos[idx][1] > center[1]: oct_idx |= 2
                if pos[idx][2] > center[2]: oct_idx |= 4
                octants[oct_idx].append(idx)

            for oct_idx, oct_indices in enumerate(octants):
                if oct_indices:
                    children[node, oct_idx] = node_count
                    stack.append((node_count, np.array(oct_indices)))
                    node_count += 1

    return mass, cm, size, children, is_leaf, body_idx

@njit(cache=True, fastmath=True)
def compute_accelerations_numba(pos, mass, cm, size, children, is_leaf, body_idx, G, softening, THETA):
    N = len(pos)
    acc = np.zeros((N, 3))
    for i in range(N):
        stack = [0]
        while stack:
            node = stack.pop()
            if is_leaf[node] and body_idx[node] == i:
                continue
            dr = cm[node] - pos[i]
            dist = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2) + 1e-9
            if (size[node] / dist < THETA) or is_leaf[node]:
                dist2 = dist**2 + softening**2
                acc[i] += G * mass[node] * dr / (dist2 * np.sqrt(dist2))
            else:
                for c in range(8):
                    if children[node, c] != -1:
                        stack.append(children[node, c])
    return acc

def run_leapfrog(pos, vel, m_array, dt, G, softening, n_steps, theta=0.5, verbose=False):
    args = build_flat_tree(pos, m_array)
    a = compute_accelerations_numba(pos, *args, G, softening, theta)

    for s in range(n_steps):
        if verbose:
            print(f"  step {s+1}/{n_steps}", end="", flush=True)
        v_half = vel + 0.5 * dt * a
        pos = pos + dt * v_half
        args = build_flat_tree(pos, m_array)
        a = compute_accelerations_numba(pos, *args, G, softening, theta)
        vel = v_half + 0.5 * dt * a
        if verbose:
            print(" ok", flush=True)
    return pos, vel