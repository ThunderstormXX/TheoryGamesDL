# Инструкция по установке и запуску эксперимента exp2

## Требования

Для запуска эксперимента необходимы следующие Python-пакеты:
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0

## Установка зависимостей

### Вариант 1: Использование conda (рекомендуется)

Если у вас активно conda-окружение (base):

```bash
conda install numpy matplotlib tqdm
```

### Вариант 2: Использование pip

```bash
pip install numpy matplotlib tqdm
```

или

```bash
python3 -m pip install numpy matplotlib tqdm
```

### Вариант 3: Установка из requirements.txt

Из корня проекта:

```bash
pip install -r requirements.txt
```

## Запуск экспериментов

### Быстрый тест (1000 итераций)

```bash
cd experiments/exp2
python test_quick.py
```

### Полный эксперимент (20000 итераций)

```bash
cd experiments/exp2
python run_experiment.py
```

Это запустит три сценария:
1. Симметричные продавцы
2. Асимметричные продавцы (разная себестоимость)
3. Сравнение разных значений эластичности

### Визуализация результатов

```bash
cd experiments/exp2
python visualize.py
```

Графики будут сохранены в `experiments/exp2/results/`

## Структура файлов

```
exp2/
├── __init__.py              # Инициализация модуля
├── run_experiment.py        # Основной скрипт эксперимента
├── test_quick.py            # Быстрый тест
├── visualize.py             # Визуализация результатов
├── results/                 # Папка для результатов
├── README.md               # Описание эксперимента
└── INSTALL.md              # Этот файл
```

## Импорт модулей в Python-скриптах

После установки зависимостей вы можете импортировать модули так:

```python
from theorygamesdl.agents.market_qlearning import MarketAgent
from theorygamesdl.models.market_game import MarketGame

# Создать агента
agent = MarketAgent(name="Продавец A", c=0.2, eta=0.7, beta=3.0, alpha=0.01, gamma=0.9)

# Создать игру
game = MarketGame(agent1, agent2, T=1000)

# Запустить симуляцию
history = game.simulate()
```

## Решение проблем

### ModuleNotFoundError: No module named 'numpy'

Установите numpy:
```bash
pip install numpy
```

### Виртуальная среда не работает

Если `.venv` не работает, деактивируйте её и используйте системный Python:
```bash
deactivate
python3 experiments/exp2/test_quick.py
```

Или активируйте conda-окружение:
```bash
conda activate base
python experiments/exp2/test_quick.py
```

## Ожидаемый вывод

После успешного запуска `test_quick.py` вы должны увидеть:

```
============================================================
Быстрый тест exp2
============================================================

📊 Параметры:
   Себестоимость (c): 0.2
   Эластичность (eta): 0.7
   Теоретическое равновесие Нэша: p* = 0.923

🚀 Запуск симуляции (1000 итераций)...
------------------------------------------------------------
Iter    200: E[p1]=0.512, E[p2]=0.508, Avg_R1=0.0896, Avg_R2=0.0898
Iter    400: E[p1]=0.745, E[p2]=0.742, Avg_R1=0.2041, Avg_R2=0.2038
...
```



