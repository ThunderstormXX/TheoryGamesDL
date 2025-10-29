"""
Визуализация результатов эксперимента exp2
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from theorygamesdl.agents.market_qlearning import MarketAgent
from theorygamesdl.models.market_game import MarketGame


def visualize_learning_dynamics(game, history, window_size=100):
    """
    Визуализировать динамику обучения
    
    Args:
        game: Объект игры
        history: История симуляции
        window_size: Размер окна для скользящего среднего
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Цены во времени
    ax = axes[0, 0]
    ax.plot(history["p1"], alpha=0.3, label='Продавец A (сырые)', linewidth=0.5)
    ax.plot(history["p2"], alpha=0.3, label='Продавец B (сырые)', linewidth=0.5)
    
    # Скользящее среднее
    p1_smooth = np.convolve(history["p1"], np.ones(window_size)/window_size, mode='valid')
    p2_smooth = np.convolve(history["p2"], np.ones(window_size)/window_size, mode='valid')
    ax.plot(p1_smooth, label='Продавец A (среднее)', linewidth=2)
    ax.plot(p2_smooth, label='Продавец B (среднее)', linewidth=2)
    
    # Равновесие Нэша
    nash_price = game.get_nash_equilibrium_theory()
    if nash_price is not None:
        ax.axhline(nash_price, color='red', linestyle='--', label=f'Равновесие Нэша (p*={nash_price:.3f})')
    
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Цена')
    ax.set_title('Динамика цен')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Прибыль во времени
    ax = axes[0, 1]
    r1_smooth = np.convolve(history["r1"], np.ones(window_size)/window_size, mode='valid')
    r2_smooth = np.convolve(history["r2"], np.ones(window_size)/window_size, mode='valid')
    ax.plot(r1_smooth, label='Продавец A')
    ax.plot(r2_smooth, label='Продавец B')
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Прибыль')
    ax.set_title('Динамика прибыли (скользящее среднее)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Ожидаемые цены (если отслеживались)
    ax = axes[1, 0]
    if "ep1" in history:
        ax.plot(history["ep1"], label='E[p1]')
        ax.plot(history["ep2"], label='E[p2]')
        if nash_price is not None:
            ax.axhline(nash_price, color='red', linestyle='--', label=f'Nash p*={nash_price:.3f}')
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Ожидаемая цена')
        ax.set_title('Сходимость ожидаемых цен')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Данные не отслеживались', 
                ha='center', va='center', transform=ax.transAxes)
    
    # 4. Энтропия политики
    ax = axes[1, 1]
    if "entropy1" in history:
        ax.plot(history["entropy1"], label='Продавец A')
        ax.plot(history["entropy2"], label='Продавец B')
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Энтропия')
        ax.set_title('Энтропия политики (разнообразие действий)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Данные не отслеживались', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('experiments/exp2/results/learning_dynamics.png', dpi=150)
    print("💾 График сохранен: experiments/exp2/results/learning_dynamics.png")
    plt.show()


def visualize_policy_distribution(agent, title="Распределение политики"):
    """
    Визуализировать распределение вероятностей в политике агента
    
    Args:
        agent: Агент с обученной политикой
        title: Заголовок графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Распределение вероятностей
    ax = axes[0]
    ax.bar(agent.p_grid, agent.pi, width=0.01, alpha=0.7)
    ax.set_xlabel('Цена')
    ax.set_ylabel('Вероятность')
    ax.set_title(f'{title}: Распределение вероятностей')
    ax.axvline(agent.expected_price(), color='red', linestyle='--', 
               label=f'E[p]={agent.expected_price():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-значения
    ax = axes[1]
    ax.plot(agent.p_grid, agent.Q)
    ax.set_xlabel('Цена')
    ax.set_ylabel('Q-значение')
    ax.set_title(f'{title}: Q-функция')
    ax.axvline(agent.expected_price(), color='red', linestyle='--',
               label=f'E[p]={agent.expected_price():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = title.replace(' ', '_').replace('[', '').replace(']', '')
    plt.savefig(f'experiments/exp2/results/policy_{safe_name}.png', dpi=150)
    print(f"💾 График сохранен: experiments/exp2/results/policy_{safe_name}.png")
    plt.show()


def visualize_price_heatmap(history, bins=50):
    """
    Тепловая карта распределения цен
    
    Args:
        history: История симуляции
        bins: Количество бинов для гистограммы
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Создаем 2D гистограмму
    H, xedges, yedges = np.histogram2d(history["p1"], history["p2"], bins=bins)
    
    # Отображаем тепловую карту
    im = ax.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    ax.set_xlabel('Цена продавца A')
    ax.set_ylabel('Цена продавца B')
    ax.set_title('Совместное распределение цен')
    
    # Диагональ (равные цены)
    ax.plot([0, 1], [0, 1], 'b--', alpha=0.5, label='p1 = p2')
    
    plt.colorbar(im, ax=ax, label='Частота')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/exp2/results/price_heatmap.png', dpi=150)
    print("💾 График сохранен: experiments/exp2/results/price_heatmap.png")
    plt.show()


def run_and_visualize():
    """
    Запустить эксперимент и создать все визуализации
    """
    print("=" * 60)
    print("Запуск эксперимента и визуализация")
    print("=" * 60)
    
    # Создаем папку для результатов
    os.makedirs('experiments/exp2/results', exist_ok=True)
    
    # Параметры
    c = 0.2
    eta = 0.7
    beta = 3.0
    alpha = 0.01
    gamma = 0.9
    
    # Создаём агентов
    agent1 = MarketAgent(name="Продавец A", c=c, eta=eta, beta=beta, alpha=alpha, gamma=gamma)
    agent2 = MarketAgent(name="Продавец B", c=c, eta=eta, beta=beta, alpha=alpha, gamma=gamma)
    
    # Создаём игру
    game = MarketGame(agent1, agent2, T=20000, track_convergence=True)
    
    print("\n🚀 Запуск симуляции...")
    history = game.simulate(verbose=True, log_interval=2000)
    
    print("\n📊 Создание визуализаций...")
    
    # 1. Динамика обучения
    visualize_learning_dynamics(game, history, window_size=200)
    
    # 2. Распределение политик
    visualize_policy_distribution(agent1, title="Продавец A")
    visualize_policy_distribution(agent2, title="Продавец B")
    
    # 3. Тепловая карта
    visualize_price_heatmap(history, bins=50)
    
    # Статистика
    burn_in = int(0.8 * game.T)
    stats = game.compute_statistics(burn_in=burn_in)
    
    print("\n" + "=" * 60)
    print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Ожидаемая цена A: {stats['expected_p1']:.3f}")
    print(f"Ожидаемая цена B: {stats['expected_p2']:.3f}")
    print(f"Равновесие Нэша: {stats.get('nash_equilibrium', 'N/A')}")
    print(f"Средняя прибыль A: {stats['mean_r1']:.4f}")
    print(f"Средняя прибыль B: {stats['mean_r2']:.4f}")
    print("=" * 60)
    
    print("\n✅ Все визуализации созданы!")


if __name__ == "__main__":
    run_and_visualize()

