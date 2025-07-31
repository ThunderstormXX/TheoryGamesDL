# Experiments Workspace

Рабочее пространство для экспериментов с библиотекой TheoryGamesDL.

## Структура

```
experiments/
├── exp1/                    # AlphaRank-based RL
│   ├── alpharank_agent.py
│   ├── alpharank_simulation.py
│   ├── run_experiment.py
│   └── README.md
└── README.md               # Этот файл
```

## Эксперименты

### exp1: AlphaRank-based Reinforcement Learning

Обучение нейросетевых агентов с использованием AlphaRank для вычисления reward в матричных играх.

**Запуск:**
```bash
cd exp1
python run_experiment.py
```

## Общие принципы

- Каждый эксперимент находится в отдельной папке
- Результаты сохраняются в JSON формате
- Графики генерируются автоматически
- Все эксперименты используют базовую библиотеку theorygamesdl