#!/usr/bin/env bash

OUT="all_project_text.txt"

echo "Создаю $OUT ..."

{
  echo "# PROJECT TEXT FILES"
  echo

  # список файлов
  git ls-files | grep -E '\.(py|md|tex|ipynb)$'

  # обработка файлов
  git ls-files | grep -E '\.(py|md|tex|ipynb)$' | while IFS= read -r file; do
    echo
    echo "--------------------"
    echo "FILE: $file"
    echo "--------------------"

    case "$file" in
      *.ipynb)
        # извлекаем только текст ячеек
        if command -v jq >/dev/null 2>&1; then
          jq -r '.cells[] | select(.cell_type=="code" or .cell_type=="markdown") | .source[]?' "$file" 2>/dev/null
        else
          cat "$file"
        fi
        ;;
      *)
        cat "$file"
        ;;
    esac
  done

} > "$OUT"

echo "Готово."
echo "Файл: $OUT"