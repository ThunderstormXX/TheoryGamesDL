# Контекст проекта для Claude Code

## Что это за проект

Исследовательский проект: **сравнение Q-learning и LLM (in-context learning) в итеративной дилемме заключённого на графах**.

Агенты сидят в вершинах графа и каждый раунд выбирают: кооперировать (1) или дефектить (0). Награда зависит от действий соседей по графу. Q-learning агенты учатся через обновление Q-таблицы, LLM-агенты — через накопление истории в промпте (in-context learning).

Связанные статьи:
- `papers/Granular-q-learning.pdf` — Lee & Weng: II vs ID агенты с Q-learning на графах
- `papers/Konstantinov.pdf` — магистерская диссертация (Константинов)
- `papers/results_02_23.pdf` — наши текущие результаты с Q-learning, включая вывод теоретического ρ̂

---

## Структура проекта

```
context_learning/
├── graph_structure.py          # Графы: StarGraph, WheelGraph, SmallWorldGraph
├── rules.md                    # Описание моделей игры и наград
├── cpu_version/
│   ├── learner.py              # QLearner, SARSALearner (Boltzmann / ε-greedy)
│   ├── reward_model.py         # PPReward, PFReward, FPReward, FFReward
│   ├── game_launcher.py        # PairGame, MonteKarloPairGame, MonteKarloNotPairGame
│   ├── llm_agent_openrouter.py # ★ LLM-агент через OpenRouter API (4 режима промптов)
│   ├── llm_game_launcher.py    # ★ LLMPairGame — передаёт LLM-агентам доп. контекст
│   ├── run_llm_openrouter_exp.py # ★ Главный скрипт эксперимента Q-learning vs LLM
│   ├── run_cooperation_exp.py  # Эксперименты только с Q-learning
│   ├── llm_agent.py            # Старый LLM-агент через OpenAI (не используется)
│   ├── run_llm_exp.py          # Старый скрипт (не используется)
│   └── sample_exp.ipynb        # Jupyter-ноутбук с примерами
├── gpu_version/                # GPU-реализация (CUDA) — не трогаем сейчас
└── papers/                     # PDF статей
```

★ — файлы, созданные/изменённые в текущей сессии.

---

## Как работает LLM-эксперимент

### Архитектура

Для каждой вершины графа создаётся отдельный `LLMAgentOpenRouter`. Каждый ход — это **отдельный POST-запрос** к OpenRouter API. Никакого сохранения чата между ходами нет. Вся "память" — это `self.history`, который форматируется в текст промпта.

### 4 режима промптов (`--modes`)

| Режим | Что видит агент | Аналог |
|---|---|---|
| `blind` | Только свои прошлые action (0/1) и reward. Не знает про игру, соседей, дилемму | Чистый бандит |
| `history_only` | Свою историю + сколько соседей кооперировалось (агрегированное число) | Interactive Identity (II) |
| `history_and_global` | То же + глобальная доля кооперации по всей сети | II + соц. норма |
| `neighbors_detail` | Свою историю + конкретные действия каждого соседа | Interactive Diversity (ID) |

### Запуск

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Полный запуск
python3 run_llm_openrouter_exp.py --modes blind history_only --ep_llm 50 --api_delay 0

# Все флаги
python3 run_llm_openrouter_exp.py \
  --n_nodes 8 \
  --graph_type small_world \    # small_world | star
  --b 3.0 --c 1.0 \
  --reward_type pp \            # pp | pf | fp | ff
  --ep_q 500 --q_n_runs 10 \
  --q_lr 0.2 --q_gamma 0.0 --q_temp 1.0 \
  --llm_model "meta-llama/llama-3.2-3b-instruct" \
  --llm_temperature 0.2 \
  --ep_llm 100 \
  --modes blind history_only history_and_global neighbors_detail \
  --api_delay 0 \
  --verbose_llm \               # печатать каждый промпт и ответ
  --output_dir results/llm_exp
```

### Выходные файлы

Каждый запуск создаёт файлы с уникальным именем (таймстамп + параметры), старые НЕ перезаписываются:
```
results/llm_exp/
  cooperation_20250308_123004_N8_small_world_pp_b3.0_c1.0_g0.0_T1.0_blind+history_only.png
  reward_20250308_123004_...png
  data_20250308_123004_...npz
  report_20250308_123004_...json
```

На графиках отображаются все гиперпараметры: в заголовке, легенде и подписи внизу.

---

## Теоретическое значение ρ̂

Из `papers/article_v3.pdf`, Теорема 4.5. При γ=0 и Boltzmann-политике:

```
p_i = 1 / (1 + exp(C_i / T))
ρ̂ = (1/n) Σ p_i
```

где n — число вершин, C_i — эффективная цена кооперации:
- pp и fp: C_i = c·k_i  →  ρ̂ = (1/n) Σ 1/(1 + exp(c·k_i / T))
- pf и ff: C_i = c      →  ρ̂ = 1/(1 + exp(c / T))

Не зависит от b (при γ=0). Для звезды S_n (Теорема 4.7, pp/fp):
```
ρ̂ = 1/(n+1) · [n/(1 + exp(c/T)) + 1/(1 + exp(c·n/T))]
```
(n листьев со степенью 1, 1 центр со степенью n — нормировка на n+1 вершин).

В коде `np.mean([1/(1+exp(c*k/T)) for k in degrees])` — корректно, т.к. `degrees` содержит степени всех n вершин.

---

## Что уже сделано в этой сессии

1. Создан `llm_agent_openrouter.py` — LLM-агент через OpenRouter с 4 режимами промптов
2. Создан `llm_game_launcher.py` — LLMPairGame, передающий доп. контекст LLM-агентам
3. Создан `run_llm_openrouter_exp.py` — скрипт сравнения Q-learning vs LLM
4. Добавлен режим `blind` (агент не знает про игру)
5. Добавлен `--verbose_llm` для просмотра промптов и ответов
6. Все гиперпараметры отображаются на графиках
7. Файлы не перезаписываются (уникальные имена с таймстампом)
8. Все Q-learning параметры вынесены во флаги (`--q_lr`, `--q_gamma`, `--q_temp`)
9. Добавлен `--api_delay` для rate limiting (по умолчанию 1 сек)
10. Исправлена обработка пустых ответов API (NoneType)

## Что можно делать дальше

- Запустить полные эксперименты и сравнить все 4 режима
- Сравнить разные LLM-модели (Mistral vs Llama vs Gemma)
- Варьировать b/c и смотреть, как LLM реагирует на силу дилеммы
- Добавить смешанные популяции (часть агентов Q-learning, часть LLM)
- Добавить SARSA-агентов для сравнения
- Попробовать другие типы графов и наград (pf, fp, ff)
- Анализировать формирует ли LLM условные стратегии (tit-for-tat и т.д.)
- GPU-версия LLM-эксперимента (batch API calls)