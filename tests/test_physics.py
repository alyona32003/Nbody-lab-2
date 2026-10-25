import numpy as np
import pytest
from nbody.physics import generate_cloud, build_flat_tree, compute_accelerations_numba

def test_generate_cloud_shape():
    pos, vel = generate_cloud(N=100, R=10.0, omega=1.0, fluctuation=0.0, dist_type=0)
    assert pos.shape == (100, 3)
    assert vel.shape == (100, 3)
    # Центр масс должен быть в нуле (с учётом вычитания среднего)
    assert np.allclose(np.mean(pos, axis=0), np.zeros(3), atol=1e-8)

def test_tree_mass_conservation():
    N = 50
    np.random.seed(42)
    pos = np.random.randn(N, 3)
    m_array = np.ones(N)
    mass, cm, size, children, is_leaf, body_idx = build_flat_tree(pos, m_array)
    # Корень дерева (индекс 0) должен содержать массу всех частиц
    assert np.isclose(mass[0], np.sum(m_array))

def test_acceleration_two_body():
    # Два тела массами по 1
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m_array = np.array([1.0, 1.0])
    args = build_flat_tree(pos, m_array)
    
    G = 1.0
    softening = 0.0 # Без смягчения для аналитики
    THETA = 0.0     # THETA=0 заставляет алгоритм считать попарно (точно)
    
    acc = compute_accelerations_numba(pos, *args, G, softening, THETA)
    
    # Аналитически: a1 = G * m2 / r^2 = 1 * 1 / 1^2 = 1 (вправо)
    # a2 = G * m1 / r^2 = 1 (влево)
    assert np.isclose(acc[0, 0], 1.0, atol=1e-4)
    assert np.isclose(acc[1, 0], -1.0, atol=1e-4)