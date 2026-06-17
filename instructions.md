# Навигация по проекту TheoryGamesDL

## Общая структура

```
TheoryGamesDL/
├── theorygamesdl/          # Основная библиотека (базовые классы, агенты)
├── experiments/
│   └── exp8/
│       └── gpu_version/   # Текущий активный эксперимент (см. ниже)
├── results/                # Общие результаты и графики
├── notebooks/              # Jupyter-ноутбуки
├── scripts/                # Вспомогательные скрипты
└── main.tex                # LaTeX-статья с теорией
```

**Основная библиотека** (`theorygamesdl/`) — базовые модели социальных дилемм (PD, Stag Hunt, Chicken и др.) и агенты (Q-learning, DQN, A2C). Используется как фундамент, но активная работа ведётся в `exp8/gpu_version/`.

---

## experiments/exp8/gpu_version — главный рабочий модуль

Реализует батчевое GPU-обучение агентов в сетевых играх (теория кооперации / дилемма заключённого на графе). Основная идея: параллельный запуск сотен симуляций на GPU через PyTorch тензоры.

### Ядро (`core/`)

| Файл | Что делает |
|------|-----------|
| [`core/batched_gpu.py`](experiments/exp8/gpu_version/core/batched_gpu.py) | Главный класс `BatchedGPUQLearner` — Q-таблица для `batch_size × n_agents` агентов на GPU; методы выбора действия (ε-greedy / Boltzmann) и обновления Q-значений |
| [`core/batched_sarsa.py`](experiments/exp8/gpu_version/core/batched_sarsa.py) | Аналог выше, но SARSA (on-policy): обновление идёт по реально выбранному следующему действию, а не по `max Q` |
| [`core/graph_structure.py`](experiments/exp8/gpu_version/core/graph_structure.py) | Топологии сети агентов: `SmallWorldGraph` (Watts-Strogatz), `StarGraph`, `WheelGraph`, `RingGraph`, циркулянтные графы, смешанные `Mixed23/34/45/56Graph` (k-регулярный + 1 хорда → 2 вершины степени k+1). Возвращает тензор матрицы смежности |
| [`core/reward_models.py`](experiments/exp8/gpu_version/core/reward_models.py) | `RewardManager` — две модели вознаграждения: `'pf'` (парная игра) и `'pp'` (пул); принимает тензоры кооператоров и матрицу смежности |
| [`core/bonus_reward_manager.py`](experiments/exp8/gpu_version/core/bonus_reward_manager.py) | Расширение RewardManager с бонусными наградами для задачи 2 |

### Утилиты (`utils/`)

| Файл | Что делает |
|------|-----------|
| [`utils/gpu_utils.py`](experiments/exp8/gpu_version/utils/gpu_utils.py) | `gpu_config` — определяет доступное устройство (CUDA / MPS / CPU) и экспортирует его для всех модулей |

### Визуализация (`visualization/`)

| Файл | Что делает |
|------|-----------|
| [`visualization/plotting.py`](experiments/exp8/gpu_version/visualization/plotting.py) | Функции построения графиков: динамика кооперации со стандартным отклонением, фазовые диаграммы |
| [`visualization/cluster_plotting.py`](experiments/exp8/gpu_version/visualization/cluster_plotting.py) | **(новое)** `plot_convergence_clusters` — граф NetworkX с раскраской вершин по кластеру сходимости и подписью `id / d=степень / c=кластер`; `plot_q_curves_by_cluster` — траектории Q(C)/Q(D) с раскраской по кластерам |

### Анализ сходимости и топологии (`analysis/`) — новый модуль

Аддитивный пакет: переиспользует существующее ядро обучения и генераторы графов, **не меняя их**. Выделяет классы сходимости вершин (вершины одной степени могут сходиться к разным Q) и связывает их с топологией.

| Файл | Что делает |
|------|-----------|
| [`analysis/simulation.py`](experiments/exp8/gpu_version/analysis/simulation.py) | `run_convergence_simulation(adjacency, *, gamma, beta, learner_type, iters, reps, seed, ...)` → `SimulationResult`. Seedable обёртка над `BatchedGPUQLearner`/`BatchedGPUSARSALearner`; принимает матрицу смежности напрямую. **A100-оптимизации:** награда через один общий matmul `(reps,N)@(N,N)` вместо `bmm` по `(reps,N,N)` (память ∝ `reps·N`, не `reps·N²`); запись только усреднённых mean/std `(T_out,N)` (память не растёт с `reps`); TF32 + `inference_mode`; `suggest_reps()` подбирает `reps` под объём VRAM; флаг `progress` — внутренний tqdm по шагам (steps/s) |
| [`analysis/convergence_clustering.py`](experiments/exp8/gpu_version/analysis/convergence_clustering.py) | **Задача 1.** `compute_convergence_features` (признаки `[Q_C_final, Q_D_final, Q_C−Q_D]` по последним N шагам), `cluster_vertices_by_convergence` (auto: DBSCAN → HDBSCAN → KMeans+silhouette; масштаб `shared`), `save_cluster_table` (CSV) |
| [`analysis/topology_features.py`](experiments/exp8/gpu_version/analysis/topology_features.py) | Структурные признаки вершин: `degree, clustering_coefficient, betweenness_centrality, eigenvector_centrality` (NetworkX) — для корреляции «тип сходимости ↔ положение в графе» |
| [`analysis/interpolation.py`](experiments/exp8/gpu_version/analysis/interpolation.py) | **Задача 3.** `generate_interpolated_regular_graph(n, k, temperature, seed=None, *, mode)` — непрерывное семейство `t=0` (k-регулярный) ↔ `t=1` ((k+1)-регулярный) через паросочетание `E_add`; режимы `deterministic` (доля `t·\|E_add\|`) и `stochastic` (каждое ребро с вероятностью `t`) |
| [`analysis/pipeline.py`](experiments/exp8/gpu_version/analysis/pipeline.py) | `analyze_topology(...)` — связка sim → признаки → кластеры → топология → артефакты (`q_curves.png`, `convergence_clusters.png`, `cluster_table.csv`, `summary.json`, `artifacts.npz`, `run_params.json`) + корреляция кластер↔структура (η) |
| [`analysis/artifacts.py`](experiments/exp8/gpu_version/analysis/artifacts.py) | **Переиспользуемые артефакты.** `save_run_artifacts`/`load_run_artifacts` — `artifacts.npz` (матрица смежности, усреднённые по репликам траектории mean/std, метки кластеров, финальные Q, структурные признаки) + `run_params.json` (**все** параметры запуска вплоть до топологии: edge list графа, гиперпараметры обучения, настройки кластеризации). `replot_from_artifacts` — перерисовка графиков без пересчёта симуляции (с опциональной рекластеризацией) |

---

### Скрипты запуска экспериментов

#### Основные

| Файл | Что запускает |
|------|--------------|
| [`run_experiment.py`](experiments/exp8/gpu_version/run_experiment.py) | Базовый запуск: 100 батчей × 50 агентов × 1000 раундов на Small World графе |
| [`sweep_experiments.py`](experiments/exp8/gpu_version/sweep_experiments.py) | Перебор параметров (b, γ, топология, тип состояния) — основной скрипт для получения графиков статьи |
| [`convergence_experiment.py`](experiments/exp8/gpu_version/convergence_experiment.py) | Эксперимент на сходимость (запускался на A100), пишет логи в `experiments/exp8/logs/` |
| [`bernoulli_experiment.py`](experiments/exp8/gpu_version/bernoulli_experiment.py) | Параллельный CPU-эксперимент через `multiprocessing` для сравнения с Бернулли-моделью |

#### Задания супервайзера

| Файл | Что запускает |
|------|--------------|
| [`supervisor_k_regular_tasks.py`](experiments/exp8/gpu_version/supervisor_k_regular_tasks.py) | Задачи по k-регулярным графам (кольцо, кубический, смешанный); результаты → `supervisor_results/` |
| [`task1_b4_c1_experiment.py`](experiments/exp8/gpu_version/task1_b4_c1_experiment.py) | Задача 1: фиксированные параметры b=4, c=1 |
| [`task2_bonus_experiment.py`](experiments/exp8/gpu_version/task2_bonus_experiment.py) | Задача 2: бонусные вознаграждения |

#### Анализ сходимости и топологии (новые скрипты)

| Файл | Что запускает |
|------|--------------|
| [`run_all_convergence_topology_experiments.py`](experiments/exp8/gpu_version/run_all_convergence_topology_experiments.py) | **Задача 2.** Массовый прогон по всем k-регулярным семействам (вкл. `mixed45/56`), размерам и богатой сетке gamma×beta; параллелизм `--workers` (ProcessPoolExecutor/spawn), `--auto-reps` под VRAM A100, вложенный tqdm; на каждый запуск → `results/convergence_topology/<topology_name>/{q_curves.png, convergence_clusters.png, cluster_table.csv, summary.json, artifacts.npz, run_params.json}` + общий `index_summary.json` |
| [`run_topology_phase_transition.py`](experiments/exp8/gpu_version/run_topology_phase_transition.py) | **Задача 4.** Свип `temperature∈{0.00..1.00}` (усреднение по реализациям); параллелизм `--workers` по (temperature, realization), `--auto-reps`; → `phase_*.png`, серия `temp_0.00.png … temp_1.00.png`, `phase_summary.json/.csv` |
| [`replot_from_artifacts.py`](experiments/exp8/gpu_version/replot_from_artifacts.py) | Перерисовка `q_curves.png`/`convergence_clusters.png` из `artifacts.npz` **без пересчёта симуляции**; опции `--recluster`, `--cluster-method`, `--n-final-steps`, `--layout`, `--recursive` |
| [`run_server_all.sh`](experiments/exp8/gpu_version/run_server_all.sh) | Запуск всего на сервере одной командой (Задачи 2+4), A100-настройки по умолчанию; env-переменные (`STAGE, SMOKE, ITERS, AUTO_REPS, VRAM_FRACTION, WORKERS, GRAPHS, SIZES, GAMMAS, BETAS, PHASE_PAIRS, …`); логи в `results/logs/` |
| [`replot_from_artifacts.py`](experiments/exp8/gpu_version/replot_from_artifacts.py) | Перерисовка `q_curves.png` / `convergence_clusters.png` из сохранённых `artifacts.npz` **без пересчёта симуляции**; флаги `--recursive`, `--recluster`, `--cluster-method`, `--n-final-steps`, `--layout` |

#### Верификация теории

| Файл | Что проверяет |
|------|--------------|
| [`verify_drift_proof.py`](experiments/exp8/gpu_version/verify_drift_proof.py) | Проверяет доказательство времени выхода из ловушки EV-SARSA: леммы о Q-фиксированных точках, вероятность кооперации p₀, AR(1) дрейф, формула E[T] |
| [`verify_trap_theory.py`](experiments/exp8/gpu_version/verify_trap_theory.py) | Проверяет теорию ловушек через Q-learning (в отличие от verify_drift_proof — там SARSA) |
| [`verify_stateless.py`](experiments/exp8/gpu_version/verify_stateless.py) | Верификация безсостоянийного случая |
| [`stateless_convergence_experiment.py`](experiments/exp8/gpu_version/stateless_convergence_experiment.py) | Сходимость в безсостоянийном режиме |

#### Генерация графиков для статьи

| Файл | Что генерирует |
|------|---------------|
| [`generate_plots.py`](experiments/exp8/gpu_version/generate_plots.py) | Основные графики (CPU) |
| [`generate_plots_v2.py`](experiments/exp8/gpu_version/generate_plots_v2.py) | Обновлённая версия с новыми параметрами |
| [`generate_phase_diagram.py`](experiments/exp8/gpu_version/generate_phase_diagram.py) | Фазовая диаграмма кооперации |
| [`generate_section4_graphics.py`](experiments/exp8/gpu_version/generate_section4_graphics.py) | Графики для раздела 4 статьи |
| [`generate_large_scale.py`](experiments/exp8/gpu_version/generate_large_scale.py) | Эксперименты с большим числом агентов |
| [`generate_high_gamma.py`](experiments/exp8/gpu_version/generate_high_gamma.py) | Эксперименты при высоком γ |
| [`generate_single_gamma.py`](experiments/exp8/gpu_version/generate_single_gamma.py) | Один фиксированный γ |
| [`generate_beta_experiments.py`](experiments/exp8/gpu_version/generate_beta_experiments.py) | Зависимость от параметра β (температура Больцмана) |
| [`generate_igor_report.py`](experiments/exp8/gpu_version/generate_igor_report.py) | Отчёт для научного руководителя |
| [`compare_n2_n3_from_summary.py`](experiments/exp8/gpu_version/compare_n2_n3_from_summary.py) | Сравнение систем из N=2 и N=3 агентов |

#### Тесты и отладка

| Файл | Что делает |
|------|-----------|
| [`trap_effect_experiment.py`](experiments/exp8/gpu_version/trap_effect_experiment.py) | Детектирует и анализирует «ловушки» (Neighbor-Gap ≥ 0.1) для топологий Edge, Triangle, Star |
| [`test_and_compare.py`](experiments/exp8/gpu_version/test_and_compare.py) | Старый тест моделей вознаграждения (использует устаревшие импорты) |
| [`test_k_regular.py`](experiments/exp8/gpu_version/test_k_regular.py) | Тест k-регулярных топологий |
| [`test_custom_topology.py`](experiments/exp8/gpu_version/test_custom_topology.py) | Тест произвольных топологий |
| [`test_custom_topology_stateful.py`](experiments/exp8/gpu_version/test_custom_topology_stateful.py) | То же, но с состоянием |
| [`quick_triangle.py`](experiments/exp8/gpu_version/quick_triangle.py) | Быстрая проверка на треугольнике (3 агента) |
| [`reorganized_experiment.py`](experiments/exp8/gpu_version/reorganized_experiment.py) | Рефакторинговая версия run_experiment |
| [`sarsa_experiments.py`](experiments/exp8/gpu_version/sarsa_experiments.py) | Эксперименты с SARSA-агентами |
| [`igor_answers_script.py`](experiments/exp8/gpu_version/igor_answers_script.py) | Скрипт ответов на вопросы руководителя |

---

### Результаты

| Папка | Содержимое |
|-------|-----------|
| [`proof_verification_results/`](experiments/exp8/gpu_version/proof_verification_results/) | PNG-графики верификации теоремы: E[T] vs β, масштабирование, зависимость от α |
| [`supervisor_results/`](experiments/exp8/gpu_version/supervisor_results/) | Результаты задач k-регулярных графов (ring, cubic, mixed, sarsa) |
| `results/convergence_topology/` | **(новое)** Результаты массового прогона (Задача 2): папка на топологию `{q_curves.png, convergence_clusters.png, cluster_table.csv, summary.json, artifacts.npz, run_params.json}` + `index_summary.json` |
| `results/phase_transition/` | **(новое)** Результаты фазовых переходов (Задача 4): `phase_*.png`, `temp_*.png`, `phase_summary.json/.csv`; полные артефакты репрезентативной реализации — в `runs/t<temp>/` |

---

### Как запускать (из корня проекта)

```bash
# Активировать окружение
source ../.venv/bin/activate

# Базовый запуск
python experiments/exp8/gpu_version/run_experiment.py

# Эксперимент с ловушками
python experiments/exp8/gpu_version/trap_effect_experiment.py

# Перебор параметров (основной для статьи)
python experiments/exp8/gpu_version/sweep_experiments.py

# Верификация теоремы о времени выхода из ловушки
python experiments/exp8/gpu_version/verify_drift_proof.py

# Задачи супервайзера
python experiments/exp8/gpu_version/supervisor_k_regular_tasks.py
```

#### Анализ сходимости и топологии (новый модуль)

```bash
# Всё на сервере одной командой (Задачи 2 + 4)
bash experiments/exp8/gpu_version/run_server_all.sh

# A100: заполнить VRAM авто-батчем + параллельные процессы
AUTO_REPS=1 VRAM_FRACTION=0.85 WORKERS=4 ITERS=1000000 \
  bash experiments/exp8/gpu_version/run_server_all.sh

# Фиксированный батч вместо авто
AUTO_REPS=0 REPS=8192 SIZES="10 20 50 100" \
  bash experiments/exp8/gpu_version/run_server_all.sh

# Только фазовые переходы k=2→3 и k=3→4
STAGE=phase PHASE_PAIRS="2:20 3:20" \
  bash experiments/exp8/gpu_version/run_server_all.sh

# Быстрый sanity-check (секунды)
SMOKE=1 bash experiments/exp8/gpu_version/run_server_all.sh

# Скрипты по отдельности (A100: --auto-reps заполняет VRAM, --workers — параллелизм)
python -m experiments.exp8.gpu_version.run_all_convergence_topology_experiments \
    --graphs cubic mixed45 mixed56 --sizes 10 20 50 100 \
    --gammas 0.0 0.8 0.9 0.95 --betas 0.5 1.0 2.0 \
    --iters 500000 --auto-reps --workers 4
python -m experiments.exp8.gpu_version.run_topology_phase_transition \
    --n 20 --k 3 --realizations 5 --step 0.05 --auto-reps --workers 4

# Перерисовка графиков из артефактов (БЕЗ пересчёта симуляции)
python -m experiments.exp8.gpu_version.replot_from_artifacts \
    experiments/exp8/gpu_version/results/convergence_topology --recursive
# Перекластеризация и рестайл из сохранённых данных:
python -m experiments.exp8.gpu_version.replot_from_artifacts <run_dir> \
    --recluster --cluster-method kmeans --n-final-steps 5000 --layout spring
```

> **Артефакты.** Каждый запуск сохраняет `artifacts.npz` (матрица смежности, усреднённые траектории Q(C)/Q(D)/P(C) mean+std, метки кластеров, финальные Q, структурные признаки) и `run_params.json` (все гиперпараметры + полный edge list графа). Это позволяет перерисовывать любые графики и менять параметры кластеризации **без повторного запуска экспериментов**. Чтобы дополнительно сохранять полные `(T_out, reps, N)` истории — флаг `--save-full-histories`.

Подробная инструкция по запуску trap-эксперимента: [`trap_effect_runner.md`](experiments/exp8/gpu_version/trap_effect_runner.md)
