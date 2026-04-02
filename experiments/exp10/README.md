# exp10: Trap Search for Boltzmann Q-learning in 2x2 PD

`exp10` автоматически ищет режим ловушки в траекториях вероятности кооперации:

1. длительное зависание около нуля,
2. резкий скачок вверх на большом горизонте,
3. устойчивое удержание выше окрестности нуля без возврата назад.

## Что делает эксперимент
- Запускает сетку параметров для `simulate(...)` из `theorygamesdl.utils.simulation`.
- Проверяет ловушку по обоим игрокам (`player1`, `player2`).
- Сохраняет метрики и score в `CSV/JSONL`.
- Сохраняет графики для найденных ловушек.
- Поддерживает `resume` по `run_key`.

## Структура
- `run_trap_search.py` — CLI-запуск поиска.
- `utils/simulate_wrapper.py` — обёртка над `simulate` + seed + metadata.
- `utils/param_grid.py` — генератор donation/baseline/full grid.
- `utils/trap_detection.py` — прозрачный detector + trap score.
- `utils/io_utils.py` — сохранение/чтение результатов.
- `utils/plotting.py` — визуализация траекторий и Q-значений.

## Donation/Константинов
Для donation-режима используется форма:

`pd = [B - C, 0, -C, B]`

где:
- `R = B - C`,
- `P = 0`,
- `S = -C`,
- `T = B`.

Gaps:
- `g1 = P - S = C`,
- `g2 = R - P = B - C`,
- `g3 = T - R = C`.

## Detector (как работает)
Для каждого игрока:
1. сглаживание `p(t)` moving-average (`smooth_window`, адаптивно по длине),
2. поиск длинного раннего low-сегмента (`p <= near_zero_thr`),
3. поиск максимального прироста на окне `jump_window`,
4. проверка jump-условий (`min_jump`, `high_thr`),
5. проверка post-stability (`post_stable_frac`, `rel_drop_tol`),
6. расчёт score:

`score = w1*low + w2*jump + w3*stability + w4*tail_gain`.

Эксперимент считает ловушкой конфигурацию, если detector проходит у хотя бы одного игрока.

## Параметры по умолчанию
### Сетка
- `gamma`: `[0.85, 0.9, 0.95]`
- `beta`: `[1.0, 1.5, 2.0, 3.0]`
- `alpha`: `[0.005, 0.01, 0.02]`
- `time`: `[100000, 200000]`
- `C`: `[1, 2, 3, 4]`
- `B`: `[C+1, C+2, C+3, 2C+1, 2C+2]` (дубликаты автоматически убираются)

### Baseline PD
- `[3, 1, 0, 4]`
- `[2, 0, -1, 3]`
- `[3, 0, -2, 5]`
- `[4, 0, -3, 7]`
- `[6, 0, -4, 10]`

### Detector
- `near_zero_thr=0.05`
- `high_thr=0.20`
- `min_low_len_frac=0.10`
- `jump_window=2000`
- `min_jump=0.10`
- `post_stable_frac=0.20`
- `rel_drop_tol=0.5`
- `smooth_window=501`

## Запуск
Из корня репозитория:

```bash
python experiments/exp10/run_trap_search.py --mode donation
python experiments/exp10/run_trap_search.py --mode donation --time 200000
python experiments/exp10/run_trap_search.py --mode full --top-k 20
python experiments/exp10/run_trap_search.py --mode donation --beta 1.0 1.5 2.0 --gamma 0.9 0.95
```

Полезные опции:
- `--jobs N` — multiprocessing,
- `--resume` / `--no-resume`,
- `--plot-best-only`,
- `--max-runs N`.

## Выходные файлы
- `experiments/exp10/results/trap_search_results.csv`
- `experiments/exp10/results/trap_search_results.jsonl`
- `experiments/exp10/results/best_traps.json`
- `experiments/exp10/artifacts/*.png`

## Как интерпретировать ловушку
Сильный кандидат:
- большой `trap_low_segment_len`,
- заметный `trap_jump_size`,
- высокий `trap_post_jump_mean`,
- `trap_post_jump_min` не возвращается к near-zero,
- высокий `trap_score`.
