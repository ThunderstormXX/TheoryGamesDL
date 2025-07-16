# TheoryGamesDL

Библиотека для моделирования теоретико-игровых задач с использованием методов глубокого обучения.

## Описание

TheoryGamesDL предоставляет инструменты для моделирования различных типов социальных дилемм и применения методов обучения с подкреплением (классическое Q-обучение и нейросетевые подходы) для анализа стратегий агентов.

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/TheoryGamesDL.git
cd TheoryGamesDL

# Установка библиотеки
pip install -e .
```

## Структура библиотеки

```
theorygamesdl/
├── models/
│   └── social_dilemma.py  # Классы для моделирования социальных дилемм
├── agents/
│   ├── qlearning.py       # Реализация классического Q-обучения
│   └── neural_agent.py    # Реализация нейросетевых агентов (DQN, A2C)
└── utils/
    ├── analysis.py        # Функции для анализа результатов
    ├── simulation.py      # Функции для запуска симуляций
    └── encoders.py        # Вспомогательные классы для сериализации
```

## Поддерживаемые типы дилемм

- `pd`: Дилемма заключенного (Prisoner's Dilemma)
- `sh`: Охота на оленя (Stag Hunt)
- `ch`: Игра в курицу (Chicken Game)
- `bs`: Битва полов (Battle of Sexes)
- `hs`: Hawk-Dove
- `sp`: Snowdrift Problem
- `wmp`: War of Attrition

## Поддерживаемые типы агентов

- Классическое Q-обучение
- Глубокое Q-обучение (DQN)
- Advantage Actor-Critic (A2C)

## Пример использования

### Классическое Q-обучение

```python
from theorygamesdl.models import SocialDilemma
from theorygamesdl.agents import sd_qlearning
from theorygamesdl.utils.simulation import simulate

# Запуск симуляции дилеммы заключенного
pd_payoffs = [3, 1, 0, 4]  # Выплаты для дилеммы заключенного [CC, DD, DC, CD]
pol1_y1, pol2_y1, q1c, q1d, q2c, q2d, history, h_rew = simulate(
    pd=pd_payoffs, 
    time=10000,  # Количество эпизодов
    gamma=0.9,   # Коэффициент дисконтирования
    alpha=0.01,  # Скорость обучения
    beta=1       # Параметр температуры для softmax-стратегии
)
```

### Нейросетевые агенты

```python
from theorygamesdl.utils.simulation import simulate_neural

# Запуск симуляции с DQN агентами
coop_probs1, coop_probs2, history, rewards = simulate_neural(
    dilemma_type="pd",
    payoffs=[3, 1, 0, 4],
    agent_type="dqn",
    episodes=1000,
    eval_episodes=100
)

# Запуск симуляции с A2C агентами
coop_probs1, coop_probs2, history, rewards = simulate_neural(
    dilemma_type="pd",
    payoffs=[3, 1, 0, 4],
    agent_type="a2c",
    episodes=1000,
    eval_episodes=100
)
```

Более подробные примеры можно найти в файле `example_notebook.ipynb`.

## Требования

- Python 3.6+
- NumPy
- Matplotlib
- Pandas
- StatsModels
- tqdm
- PyTorch >= 1.7.0