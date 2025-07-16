# TheoryGamesDL

Библиотека для моделирования теоретико-игровых задач с использованием методов глубокого обучения.

## Описание

TheoryGamesDL предоставляет инструменты для моделирования различных типов социальных дилемм и применения методов обучения с подкреплением (в частности, Q-обучения) для анализа стратегий агентов.

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
│   └── qlearning.py       # Реализация Q-обучения
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

## Пример использования

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

Более подробные примеры можно найти в файле `example_notebook.ipynb`.

## Требования

- Python 3.6+
- NumPy
- Matplotlib
- Pandas
- StatsModels
- tqdm