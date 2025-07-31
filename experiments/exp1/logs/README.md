# Логи экспериментов

В этой папке сохраняются JSON файлы с результатами экспериментов.

## Структура файлов

- `pd_tournament_results.json` - Результаты турнира в дилемме заключенного
- `sh_tournament_results.json` - Результаты турнира в охоте на оленя  
- `large_tournament_results.json` - Результаты большого турнира с 8 агентами
- `tournament_results_YYYYMMDD_HHMMSS.json` - Результаты с временной меткой

## Формат данных

```json
{
  "config": {
    "n_agents": 6,
    "game_payoffs": [3, 1, 0, 4],
    "games_per_pair": 50,
    "rounds": 15
  },
  "history": {
    "payoff_matrices": [...],
    "transition_matrices": [...],
    "stationary_distributions": [...],
    "mean_rewards": [...],
    "round_results": [...]
  }
}
```