import re
import os

# Use absolute path based on the file location, assuming it's in TheoryGamesDL
filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments/exp8/papers/Q_Learning_on_graphs/main.tex"))

with open(filepath, "r") as f:
    text = f.read()

start_str = r"\subsection{Граф с 3 вершинами}"
end_str = r"\bibliographystyle{plain}"

start_idx = text.find(start_str)
end_idx = text.find(end_str)

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries")
    exit(1)

def gen_figure(graph_filename, gamma_val):
    if gamma_val == 0.0:
        g_str = "0.0"
    else:
        g_str = str(gamma_val)
        
    return f"""\\begin{{figure}} [H]
	\\centering
	\\includegraphics[width=\\linewidth]{{figures/{graph_filename}_gamma{g_str}.jpg}}
	\\caption{{Динамика вероятности выбора действия C $p_i^{{(t)}}(C)$ и Q-значений $Q_i^{{(t)}}(a)$ алгоритма Q-обучения при $\\gamma={g_str}$. $T=1000000$, $\\alpha = 0.01, \\beta = 0.5$.}}
\\end{{figure}}

"""

graphs = [
    ("Граф с 3 вершинами", [
        ("Клика", "triangle"),
        ("Линия", "chain3")
    ]),
    ("Граф с 4 вершинами", [
        ("Клика", "complete4"),
        ("Линия", "chain4"),
        ("Звезда", "star4"),
        ("Цикл", "ring4"),
        ("Колесо", "wheel4")
    ])
]

param_str = r"\textcolor{red}{ВЕЗДЕ ДАЛЕЕ НУЖНО УТОЧНИТЬ ПАРАМЕТР $\alpha$!!! (предлагаю выбрать $\alpha=0.01$)}"
new_param_str = r"Параметры экспериментов: $\alpha=0.01$, $\beta = 0.5$, $T_{it}=10^6$ итераций."
text = text.replace(param_str, new_param_str)

new_content = ""
for sec_title, subsecs in graphs:
    new_content += f"\\subsection{{{sec_title}}}\n\n"
    for subsec_title, file_prefix in subsecs:
        new_content += f"\\subsubsection{{{subsec_title}}}\n\n"
        for g in [0.0, 0.3, 0.5, 0.7, 0.9]:
            new_content += gen_figure(file_prefix, g)

new_text = text[:start_idx] + new_content + "\n" + text[end_idx:]

with open(filepath, "w") as f:
    f.write(new_text)

print("Updated main.tex")
