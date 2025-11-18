#!/bin/bash
# Grid search over beta and gamma parameters
# Сравнение с delta-инициализацией и без нее

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Grid Search: Beta x Gamma${NC}"
echo -e "${BLUE}======================================${NC}"

# Параметры сетки
BETA_VALUES=(0.01 0.1 0.3 1.0 5.0)
GAMMA_VALUES=(0.0 0.2 0.5 0.95)

# Фиксированные параметры
STEPS=3000
N=20
INIT_ACTION=0
SEED=42

# Директории для результатов
DIR_WITHOUT="./results/grid_beta_gamma/without_init_mode"
DIR_WITH="./results/grid_beta_gamma/with_init_mode"

# Создаем директории если их нет
mkdir -p "$DIR_WITHOUT"
mkdir -p "$DIR_WITH"

# Счетчики
TOTAL=$((${#BETA_VALUES[@]} * ${#GAMMA_VALUES[@]} * 2))
CURRENT=0

echo -e "\nВсего экспериментов: ${TOTAL}"
echo -e "Beta values: ${BETA_VALUES[@]}"
echo -e "Gamma values: ${GAMMA_VALUES[@]}"
echo -e "\n${YELLOW}Начинаем эксперименты...${NC}\n"

# Перебор без delta-инициализации (uniform)
echo -e "${GREEN}=== Режим 1/2: WITHOUT delta initialization (uniform) ===${NC}\n"
for beta in "${BETA_VALUES[@]}"; do
    for gamma in "${GAMMA_VALUES[@]}"; do
        CURRENT=$((CURRENT + 1))
        FILENAME="beta${beta}_gamma${gamma}.png"
        FILEPATH="${DIR_WITHOUT}/${FILENAME}"
        
        echo -e "${BLUE}[${CURRENT}/${TOTAL}]${NC} Running: beta=${beta}, gamma=${gamma} (uniform)"
        
        python3 run_experiment.py \
            --beta="${beta}" \
            --gamma="${gamma}" \
            --steps="${STEPS}" \
            --n="${N}" \
            --seed="${SEED}" \
            --init-mode=uniform \
            --save="${FILEPATH}" \
            > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} Saved: ${FILENAME}"
        else
            echo -e "  ${YELLOW}✗${NC} Failed: ${FILENAME}"
        fi
    done
done

echo -e "\n${GREEN}=== Режим 2/2: WITH delta initialization (init_action=${INIT_ACTION}) ===${NC}\n"
# Перебор с delta-инициализацией
for beta in "${BETA_VALUES[@]}"; do
    for gamma in "${GAMMA_VALUES[@]}"; do
        CURRENT=$((CURRENT + 1))
        FILENAME="beta${beta}_gamma${gamma}_init${INIT_ACTION}.png"
        FILEPATH="${DIR_WITH}/${FILENAME}"
        
        echo -e "${BLUE}[${CURRENT}/${TOTAL}]${NC} Running: beta=${beta}, gamma=${gamma} (delta, init=${INIT_ACTION})"
        
        python3 run_experiment.py \
            --beta="${beta}" \
            --gamma="${gamma}" \
            --steps="${STEPS}" \
            --n="${N}" \
            --seed="${SEED}" \
            --init-mode=delta \
            --init-action="${INIT_ACTION}" \
            --save="${FILEPATH}" \
            > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} Saved: ${FILENAME}"
        else
            echo -e "  ${YELLOW}✗${NC} Failed: ${FILENAME}"
        fi
    done
done

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Все эксперименты завершены!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\nРезультаты сохранены в:"
echo -e "  - ${DIR_WITHOUT}/"
echo -e "  - ${DIR_WITH}/"
echo -e "\nВсего файлов: $(ls ${DIR_WITHOUT}/*.png 2>/dev/null | wc -l) + $(ls ${DIR_WITH}/*.png 2>/dev/null | wc -l) = $(($(ls ${DIR_WITHOUT}/*.png 2>/dev/null | wc -l) + $(ls ${DIR_WITH}/*.png 2>/dev/null | wc -l)))"

