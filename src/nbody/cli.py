import click
import numpy as np
from .physics import generate_cloud, run_leapfrog

@click.command()
@click.option("--n-particles", default=2000, show_default=True, type=int, help="Число частиц")
@click.option("--steps", default=100, show_default=True, type=int, help="Число шагов интегрирования")
@click.option("--omega", default=1.01, show_default=True, type=float, help="Угловая скорость")
@click.option("--fluct", default=0.009, show_default=True, type=float, help="Амплитуда флуктуаций")
@click.option("--dist", default=0, show_default=True, type=int, help="0 - сфера, 1 - куб")
@click.option("--seed", default=21, show_default=True, type=int, help="Зерно ГСЧ")
def main(n_particles, steps, omega, fluct, dist, seed):
    """Запуск N-body симуляции (вывод только в консоль)."""
    np.random.seed(seed)
    
    # Параметры
    G = 1.0
    M_total = 1000.0
    R_CLOUD = 11.0
    softening = 0.5
    dt = 0.005
    THETA = 0.5
    
    pos, vel = generate_cloud(n_particles, R_CLOUD, omega, fluct, dist)
    m_array = np.full(n_particles, M_total / n_particles)
    
    click.echo(f"Запуск: N={n_particles}, steps={steps}, omega={omega}, fluct={fluct}, dist={dist}")
    
    pos, vel = run_leapfrog(pos, vel, m_array, dt, G, softening, steps, THETA, verbose=False)
    
    click.echo("Симуляция завершена.")
    click.echo(f"Средняя координата X: {np.mean(pos[:, 0]):.4f}")
    click.echo(f"Средняя координата Y: {np.mean(pos[:, 1]):.4f}")
    click.echo(f"Средняя скорость (по модулю): {np.mean(np.linalg.norm(vel, axis=1)):.4f}")

if __name__ == "__main__":
    main()