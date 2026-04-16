import re

with open("experiments/exp8/gpu_version/generate_section4_graphics.py", "r") as f:
    code = f.read()

# I will replace the main execution loop with a multiprocessing execution.

replacement = """
from multiprocessing import Pool, cpu_count
from functools import partial

def run_single_config(args):
    c_name, graph, g, graph_titles_c_name = args
    p_hist, qc_hist, qd_hist = run_simulation(graph, gamma=g)
    out_file = os.path.join(out_dir, f"{c_name}_gamma{g}.jpg")
    title = f"{graph_titles_c_name}, Gamma={g}"
    plot_dynamics(p_hist, qc_hist, qd_hist, title, out_file, c_name)
    return True

def main():
    print("Generating Section 4 Graphics...")
    configurations = {
        'triangle': CompleteGraph(3, DEVICE),
        'chain3': ChainGraph(3, DEVICE),
        'complete4': CompleteGraph(4, DEVICE),
        'chain4': ChainGraph(4, DEVICE),
        'star4': StarGraph(4, DEVICE),
        'ring4': RingGraph(4, DEVICE),
        'wheel4': WheelGraph(4, DEVICE)
    }
    
    graph_titles = {
        'triangle': 'Triangle (K3)',
        'chain3': 'Chain (3 nodes)',
        'complete4': 'Complete (K4)',
        'chain4': 'Chain (4 nodes)',
        'star4': 'Star (4 nodes)',
        'ring4': 'Ring (4 nodes)',
        'wheel4': 'Wheel (4 nodes)'
    }
    
    tasks = []
    for c_name, graph in configurations.items():
        for g in GAMMAS:
            tasks.append((c_name, graph, g, graph_titles[c_name]))
            
    total_runs = len(tasks)
    print(f"Total experiments to run: {total_runs}")
    
    # Run in parallel
    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} parallel workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(run_single_config, tasks), total=total_runs, desc="Overall Progress"))
        
    print("\\nAll experiments completed.")
"""

code = re.sub(r"def main\(\):.*", replacement.strip(), code, flags=re.DOTALL)

with open("experiments/exp8/gpu_version/generate_section4_graphics.py", "w") as f:
    f.write(code)

