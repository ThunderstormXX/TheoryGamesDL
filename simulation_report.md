# Анализ влияния структуры сети на ловушки (Trap Effect)
В данном отчете представлены результаты симуляций, наглядно демонстрирующие процесс падения вероятности кооперации (P(C)) в зависимости от топологии сети. Мы построили сглаженные графики динамики за 200 000 итераций при `Beta=1.0`.

## 1. Звезда (=цепочка) (3 игрока)
**Вывод:** Топология типа «звезда» ставит центрального игрока в уязвимое положение. Поскольку он связан со всеми периферийными агентами, его действия усредняются, а попытки кооперироваться с одним агентом подвергаются риску дезертирства со стороны другого. В результате **центральный игрок жестко сваливается в ловушку** ($P(C)\to 0.1$), тогда как периферия стабилизируется на более высоком уровне ($\sim 0.25-0.30$). Этот эффект сохраняется при любой $\gamma$.

### Gamma = 0.0
![Star3 Gamma 0.0](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/star3_g0.0_enhanced.png)

### Gamma = 0.3
![Star3 Gamma 0.3](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/star3_g0.3_enhanced.png)

### Gamma = 0.5
![Star3 Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/star3_g0.5_enhanced.png)

### Gamma = 0.7
![Star3 Gamma 0.7](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/star3_g0.7_enhanced.png)

## 2. Треугольник (3 игрока)
**Вывод:** В полностью симметричном графе все игроки абсолютно равноправны. При умеренных $\gamma$ система испытывает симметричное затухание кооперации. Линии идут пучком, ловушки слабо выражены. Если $\gamma$ повысить (например, до 0.95, что мы видели в глубоких тестах), симметрия рушится, и двое проваливаются, однако на $\gamma \le 0.7$ система плавно деградирует в целом (P(C) падает до $\sim0.15$).

### Gamma = 0.0
![Triangle3 Gamma 0.0](TheoryGamesDL/experiments/exp8/results/custom_report_plots/triangle3_g0.0_enhanced.png)

### Gamma = 0.3
![Triangle3 Gamma 0.3](TheoryGamesDL/experiments/exp8/results/custom_report_plots/triangle3_g0.3_enhanced.png)

### Gamma = 0.5
![Triangle3 Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/triangle3_g0.5_enhanced.png)

### Gamma = 0.7
![Triangle3 Gamma 0.7](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/triangle3_g0.7_enhanced.png)

## 3. Топологии 4 игроков (Gamma=0.5)
Здесь мы наглядно видим топологические особенности на более сложных структурах.

### Полный граф
**Полный граф:** Все агенты связаны со всеми. Результат закономерен: абсолютно симметричное, резкое падение к зоне ловушки ($< 0.1$). Выделить конкретную «жертву» невозможно.
![Полный граф Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/complete4_g0.5_enhanced.png)

### Звезда
**Звезда:** Центральный игрок (Агент 0) имеет 3 связи, периферия — по 1. Ожидаемо, Центр падает прямо в красную зону ловушки, в то время как периферийные узлы успешно удерживают $P(C) \sim 0.25-0.30$.
![Звезда Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/star4_g0.5_enhanced.png)

### Кольцо
**Кольцо:** Каждый узел имеет ровно 2 связи. Идеальная локальная симметрия не позволяет кому-то одному стать слабейшим звеном. Все вероятности падают равномерно, но не так сильно, как в полном графе (остаются выше $\sim 0.10$).
![Кольцо Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/ring4_g0.5_enhanced.png)

### Цепочка
**Цепочка ($\dots-1-0-2-3\dots$):** Два внутренних агента выступают в роли локальных центров. Они испытывают давление с двух сторон и **сваливаются в ловушку** ($P(C)\to 0.1$). Крайние агенты (только 1 связь) остаются значительно выносливее ($P(C)\uparrow 0.25$).
![Цепочка Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/chain4_g0.5_enhanced.png)

### Колесо
**Колесо:** Синергия Кольца и Звезды. Как и в полном графе, высокая связность системы давит кооперацию у всех участников. Центральный игрок проиграет чуть быстрее, но по факту в зоне ловушки оказываются все агенты.
![Колесо Gamma 0.5](/Users/macbook/Documents/learning/rl-projects/TheoryGamesDL/experiments/exp8/results/custom_report_plots/wheel4_g0.5_enhanced.png)

