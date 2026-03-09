#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/all_project_text.txt"

if ! git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Ошибка: каталог $SCRIPT_DIR не находится внутри git-репозитория." >&2
  exit 1
fi

echo "Создаю $OUT ..."

{
  echo "# PROJECT TEXT FILES"
  echo

  # список файлов
  git -C "$SCRIPT_DIR" ls-files | grep -E '\.(py|md|tex|ipynb)$'

  # обработка файлов
  git -C "$SCRIPT_DIR" ls-files | grep -E '\.(py|md|tex|ipynb)$' | while IFS= read -r file; do
    echo
    echo "--------------------"
    echo "FILE: $file"
    echo "--------------------"

    case "$file" in
      *.ipynb)
        # извлекаем только текст ячеек
        if command -v jq >/dev/null 2>&1; then
          jq -r '.cells[] | select(.cell_type=="code" or .cell_type=="markdown") | .source[]?' "$SCRIPT_DIR/$file" 2>/dev/null
        else
          cat "$SCRIPT_DIR/$file"
        fi
        ;;
      *)
        cat "$SCRIPT_DIR/$file"
        ;;
    esac
  done

} > "$OUT"

echo "Готово."
echo "Файл: $OUT"
