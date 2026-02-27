#!/bin/bash

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# Активируем виртуальное окружение (проверяем возможные пути)
VENV_PATH="../../../.venv/bin/activate"
ALT_VENV_PATH="../../../vev/bin/activate"

if [ -f "$VENV_PATH" ]; then
    echo "Активация виртуального окружения: $VENV_PATH"
    source "$VENV_PATH"
elif [ -f "$ALT_VENV_PATH" ]; then
    echo "Активация виртуального окружения: $ALT_VENV_PATH"
    source "$ALT_VENV_PATH"
else
    echo "Виртуальное окружение не найдено, используем системный Python"
fi

# Добавляем текущую директорию и корень проекта в PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/../../:$PYTHONPATH"

echo "Запуск GPU эксперимента..."
python run_gpu_experiment.py

echo "Эксперимент завершен."
