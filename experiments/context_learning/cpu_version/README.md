# LLM vs Q-learning on Network Prisoner's Dilemma

Сравнение Q-learning и LLM-агентов (in-context learning) на сетевой дилемме заключённого.

## Быстрый старт

```bash
# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
python3 run_llm_openrouter_exp.py

# OpenAI
export OPENAI_API_KEY="sk-proj-..."
python3 run_llm_openrouter_exp.py --provider openai --llm_model gpt-4o-mini
```

## Флаги командной строки

### Сеть и игра

| Флаг | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `--n_nodes` | int | `8` | Количество узлов в сети |
| `--b` | float | `3.0` | Benefit (выигрыш за кооперацию соседа) |
| `--c` | float | `1.0` | Cost (цена кооперации) |
| `--reward_type` | str | `pp` | Тип награды: `pp`, `pf`, `ff`, `fp` |
| `--graph_type` | str | `small_world` | Тип графа: `small_world`, `star` |
| `--seed` | int | `42` | Random seed |

### Q-learning

| Флаг | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `--ep_q` | int | `500` | Количество эпизодов Q-learning |
| `--q_n_runs` | int | `10` | Количество прогонов Q-learning (для усреднения) |
| `--q_lr` | float | `0.2` | Learning rate (alpha) |
| `--q_gamma` | float | `0.0` | Discount factor (gamma) |
| `--q_temp` | float | `1.0` | Температура Boltzmann |

### LLM-агент

| Флаг | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `--provider` | str | `openrouter` | API-провайдер: `openrouter` или `openai` |
| `--llm_model` | str | `mistralai/mistral-7b-instruct-v0.1` | Идентификатор модели |
| `--llm_temperature` | float | `0.2` | Температура LLM |
| `--ep_llm` | int | `100` | Количество эпизодов LLM |
| `--llm_n_runs` | int | `1` | Количество прогонов LLM (для оценки дисперсии) |
| `--modes` | list | все 4 режима | Какие prompt-режимы запускать (см. ниже) |
| `--api_key` | str | из env | API-ключ (или через переменные окружения) |
| `--api_delay` | float | `1.0` | Пауза между API-вызовами (сек) |
| `--verbose_llm` | flag | `false` | Печатать каждый промпт и ответ в stdout |
| `--output_dir` | str | `results/llm_exp` | Папка для результатов |

### Reasoning

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--reasoning_effort` |str |   `None` | Встроенный reasoning для o-серии: `low`, `medium`, `high` |
| `--chain_of_thought` |flag|  `false` | Самодельный CoT через `<think>` теги (работает с любой моделью) |

> `--reasoning_effort` и `--chain_of_thought` взаимоисключающие.

## Имена моделей

Для `--provider openrouter` используется формат OpenRouter:
- `mistralai/mistral-7b-instruct-v0.1`
- `meta-llama/llama-3.2-3b-instruct`
- `openai/gpt-4o-mini`
- `openai/o4-mini` (с `--reasoning_effort`)

Для `--provider openai` используется формат OpenAI:
- `gpt-4o-mini`
- `gpt-4o`
- `o3-mini`, `o4-mini` (с `--reasoning_effort`)

---

## Prompt-режимы (`--modes`)

Каждый режим определяет, какую информацию получает LLM-агент. Все режимы можно запускать одновременно — эксперименты пойдут последовательно.

### `blind`

Агент не знает ни о дилемме, ни о соседях. Видит только свои действия и награды. Чистый бандитский режим.

**Системный промпт:**
```
You are making repeated decisions. Each round you pick action 0 or action 1.
After each round you receive a numerical reward. Your goal is to maximize
your total reward over many rounds. You will see your past actions and the
rewards you received. Respond with ONLY the single digit 0 or 1. No explanation.
```

**Что видит агент каждый раунд:**
```
Your past rounds:
  Round 1: action=1, reward=8.00
  Round 2: action=0, reward=6.00
Choose your next action (0 or 1):
```

---

### `history_only` (Interactive Identity)

Агент знает о дилемме и видит, сколько соседей кооперировались (агрегированно).

**Системный промпт:**
```
You are a player in a repeated Prisoner's Dilemma game on a network.
You have {degree} neighbors. Each round you choose: 1 = Cooperate, 0 = Defect.
For each neighbor who cooperates, you receive benefit b. If you cooperate,
you pay cost c for each of your neighbors. If you defect, you pay nothing
but also give nothing to your neighbors.
Your reward = b * (number of cooperating neighbors) - c * (your number of
neighbors if you cooperate, 0 if you defect).
Your goal is to maximize your own cumulative reward over many rounds.
Each round you will see how many of your neighbors cooperated and your own
past actions and rewards. Respond with ONLY the single digit 0 or 1. No explanation.
```

**Что видит агент каждый раунд:**
```
=== Your history (most recent last) ===
  Round 1: you=COOPERATE, reward=8.00, neighbors_cooperating=3/4
=== Current round ===
3 out of 4 neighbors are cooperating.
Your choice (0=Defect, 1=Cooperate):
```

---

### `history_and_global` (Interactive Identity + Social Norm)

То же что `history_only`, плюс общий уровень кооперации по всей сети.

**Системный промпт:**
```
... (то же что history_only) ...
Each round you will see how many of your neighbors cooperated, what fraction
of ALL players in the network cooperated, and your own past actions and rewards.
Respond with ONLY the single digit 0 or 1. No explanation.
```

**Что видит агент каждый раунд:**
```
=== Your history (most recent last) ===
  Round 1: you=COOPERATE, reward=8.00, neighbors_cooperating=3/4, network_cooperation=62.5%
=== Current round ===
3 out of 4 neighbors are cooperating.
Overall network cooperation rate: 62.5%
Your choice (0=Defect, 1=Cooperate):
```

---

### `neighbors_detail` (Interactive Diversity)

Агент видит действия каждого соседа по отдельности.

**Системный промпт:**
```
... (то же что history_only) ...
Each round you will see the individual actions of each of your neighbors
(who cooperated, who defected) and your own past actions and rewards.
Respond with ONLY the single digit 0 or 1. No explanation.
```

**Что видит агент каждый раунд:**
```
=== Your history (most recent last) ===
  Round 1: you=COOPERATE, reward=8.00, [n2=C, n5=D, n3=C, n7=C]
=== Current round ===
Your neighbors' current actions:
  Neighbor 2: COOPERATE
  Neighbor 5: DEFECT
  Neighbor 3: COOPERATE
  Neighbor 7: COOPERATE
Your choice (0=Defect, 1=Cooperate):
```

---

## Chain-of-thought (CoT)

При `--chain_of_thought` хвост системного промпта меняется для **всех** режимов:

**Было:**
```
Respond with ONLY the single digit 0 or 1. No explanation.
```

**Стало:**
```
First, reason step-by-step about your decision inside <think>...</think> tags.
Consider your history, your neighbors' behavior, and the payoff structure.
Then, after the closing </think> tag, output ONLY the single digit 0 or 1.
```

**Пример ответа модели:**
```
<think>
My neighbors mostly defected last round. If I cooperate, I pay cost 4
but only get benefit 3. Defecting is safer here.
</think>
0
```

Reasoning извлекается из `<think>` тегов и логируется в блок `[REASONING]` в лог-файле.

---

## Примеры запуска

### Базовый эксперимент (OpenRouter, все режимы)
```bash
export OPENROUTER_API_KEY="sk-or-..."
python3 run_llm_openrouter_exp.py \
  --llm_model mistralai/mistral-7b-instruct-v0.1 \
  --ep_llm 100 \
  --api_delay 1.0
```

### OpenAI, один режим, с логами
```bash
export OPENAI_API_KEY="sk-proj-..."
python3 run_llm_openrouter_exp.py \
  --provider openai \
  --llm_model gpt-4o-mini \
  --modes history_only \
  --ep_llm 50 \
  --verbose_llm
```

### O-серия с reasoning (OpenRouter)
```bash
python3 run_llm_openrouter_exp.py \
  --llm_model openai/o4-mini \
  --reasoning_effort low \
  --ep_llm 20 \
  --api_delay 0.5
```

### O-серия с reasoning (OpenAI напрямую)
```bash
python3 run_llm_openrouter_exp.py \
  --provider openai \
  --llm_model o4-mini \
  --reasoning_effort low \
  --ep_llm 20
```

### Самодельный CoT на любой модели
```bash
python3 run_llm_openrouter_exp.py \
  --llm_model mistralai/mistral-7b-instruct-v0.1 \
  --chain_of_thought \
  --modes history_only blind \
  --ep_llm 30 \
  --verbose_llm
```

### Быстрый тест (1 эпизод, без задержки)
```bash
python3 run_llm_openrouter_exp.py \
  --provider openai \
  --llm_model gpt-4o-mini \
  --ep_llm 1 --ep_q 10 \
  --api_delay 0.0 \
  --modes blind \
  --verbose_llm
```

### Несколько прогонов LLM для оценки дисперсии
```bash
python3 run_llm_openrouter_exp.py \
  --llm_model mistralai/mistral-7b-instruct-v0.1 \
  --llm_n_runs 5 \
  --modes history_only \
  --ep_llm 100
```

## Выходные файлы

Результаты сохраняются в `--output_dir` (по умолчанию `results/llm_exp/`):

```
results/llm_exp/
  cooperation/     -- графики кооперации (.png)
  rewards/         -- графики наград (.png)
  data/            -- сырые данные (.npz)
  reports/         -- JSON-отчёты с конфигом и итогами
  logs/            -- логи промптов и ответов LLM (.txt)
```
