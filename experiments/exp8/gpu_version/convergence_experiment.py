import os
import sys
import warnings
import gc
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
log_dir = "experiments/exp8/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"a100_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("STARTING A100 EXPERIMENT")
logger.info("="*70)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# A100 Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Force single thread CPU
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Check GPU memory at start
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
else:
    logger.error("CUDA not available! Check GPU installation.")
    sys.exit(1)

# Override gpu_config
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
gpu_utils.gpu_config.device = DEVICE
logger.info(f"Set gpu_config.device to {DEVICE}")

from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.graph_structure import StarGraph, WheelGraph, SmallWorldGraph
from experiments.exp8.gpu_version.core.reward_models import RewardManager

# PARAMETERS
N_REPLICATIONS = 100
BATCH_SIZE = 64
NUM_ITERATIONS = 100000
WARMUP_PERIOD = 80000
B_VALUES = [1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
GRAPH_TYPES = ['star_graph', 'wheel_graph', 'small_world_graph']
REWARD_TYPES = ['pp', 'pf', 'ff', 'fp']

logger.info(f"Parameters: N_REPLICATIONS={N_REPLICATIONS}, BATCH_SIZE={BATCH_SIZE}")
logger.info(f"Iterations: {NUM_ITERATIONS}, Warmup: {WARMUP_PERIOD}")
logger.info(f"B_VALUES: {B_VALUES}")
logger.info(f"GRAPH_TYPES: {GRAPH_TYPES}")
logger.info(f"REWARD_TYPES: {REWARD_TYPES}")


def calculate_theoretical_q_diff(reward_type, c, degrees):
    if reward_type in ['pp', 'fp']:
        return -c * degrees
    else:
        return -torch.full_like(degrees, c)


def run_batched_simulations(batch_reps, b_val, r_type, mode, num_nodes, 
                            adj_matrix, degrees, gamma):
    """
    Run BATCH_SIZE replications simultaneously on GPU.
    """
    logger.debug(f"Starting batch with {len(batch_reps)} replications")
    logger.debug(f"Parameters: b={b_val}, r_type={r_type}, mode={mode}, nodes={num_nodes}")
    
    actual_batch = len(batch_reps)
    max_degree = int(degrees.max().item())
    max_states = max_degree + 1 if mode == 'state' else 1
    actual_gamma = gamma if mode == 'state' else 0.0
    
    logger.debug(f"max_degree={max_degree}, max_states={max_states}, gamma={actual_gamma}")
    
    try:
        # Create learner
        logger.debug("Creating BatchedGPUQLearner...")
        learner = BatchedGPUQLearner(
            batch_size=actual_batch,
            n_agents=num_nodes,
            action_space_size=2,
            learning_rate=0.1,
            discount_factor=actual_gamma,
            exploration_rate=0.05,
            max_states=max_states
        )
        logger.debug(f"Learner created. Q-table shape: {learner.q_table.shape}")
        logger.debug(f"Q-table device: {learner.q_table.device}")
        
        # Check learner is on correct device
        if str(learner.q_table.device) != str(DEVICE):
            logger.warning(f"Learner on {learner.q_table.device}, expected {DEVICE}")
        
        reward_manager = RewardManager(reward_type=r_type, b=b_val, c=1.0)
        logger.debug("RewardManager created")
        
        # Pre-allocate tensors
        logger.debug("Allocating tensors...")
        states = torch.zeros((actual_batch, num_nodes), dtype=torch.long, device=DEVICE)
        delta_q_buffer = torch.zeros((NUM_ITERATIONS, actual_batch, num_nodes, max_states), 
                                      device=DEVICE, dtype=torch.float32)
        action_buffer = torch.zeros((NUM_ITERATIONS, actual_batch, num_nodes), 
                                     device=DEVICE, dtype=torch.bool)
        
        logger.debug(f"States shape: {states.shape}, device: {states.device}")
        logger.debug(f"Delta Q buffer: {delta_q_buffer.shape}, {delta_q_buffer.nbytes/1e9:.2f} GB")
        logger.debug(f"Action buffer: {action_buffer.shape}, {action_buffer.nbytes/1e9:.2f} GB")
        
        # Check memory
        logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Main loop
        logger.info(f"Running {NUM_ITERATIONS} iterations...")
        for t in range(NUM_ITERATIONS):
            if t % 20000 == 0:
                logger.debug(f"Iteration {t}/{NUM_ITERATIONS}")
                logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            actions = learner.get_actions(states)
            action_buffer[t] = (actions == 1)
            
            # Expand adj_matrix for batch
            adj_batch = adj_matrix.unsqueeze(0).expand(actual_batch, -1, -1)
            deg_batch = degrees.unsqueeze(0).expand(actual_batch, -1)
            
            rewards = reward_manager.calculate_rewards(actions.float(), adj_batch, deg_batch)
            
            if mode == 'state':
                next_states = torch.matmul(actions.float(), adj_matrix).long()
            else:
                next_states = torch.zeros_like(states)
            
            learner.update(states, actions, rewards, next_states)
            
            # Record Delta Q
            q_table = learner.q_table
            delta_q = q_table[:, :, :, 1] - q_table[:, :, :, 0]
            delta_q_buffer[t] = delta_q
            
            states = next_states
        
        logger.info("Simulation complete, calculating results...")
        
        # Calculate cooperation rates
        post_warmup = action_buffer[WARMUP_PERIOD:]
        coop_rates = post_warmup.float().mean(dim=0)
        
        logger.debug(f"Coop rates shape: {coop_rates.shape}")
        logger.debug(f"Coop rates: {coop_rates.cpu().numpy()}")
        
        # Move to CPU
        logger.debug("Moving results to CPU...")
        coop_rates_cpu = coop_rates.cpu().numpy()
        delta_q_cpu = delta_q_buffer.cpu().numpy()
        
        logger.debug(f"Delta Q history shape: {delta_q_cpu.shape}")
        
        # Cleanup
        logger.debug("Cleaning up GPU memory...")
        del learner, states, delta_q_buffer, action_buffer, adj_batch, deg_batch
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Batch complete. Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return coop_rates_cpu, delta_q_cpu
        
    except Exception as e:
        logger.exception(f"Error in run_batched_simulations: {e}")
        logger.error(f"GPU memory at error: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        raise


def run_experiment_single_process(graph_type='star_graph'):
    output_dir = "experiments/exp8/results/a100_single_process"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"GRAPH TYPE: {graph_type}")
    logger.info(f"{'='*70}")
    
    # Setup graph
    try:
        match graph_type:
            case 'star_graph':
                num_nodes = 5
                graph = StarGraph(num_nodes=num_nodes, device=DEVICE)
            case 'wheel_graph':
                num_nodes = 5
                graph = WheelGraph(num_nodes=num_nodes, device=DEVICE)
            case 'small_world_graph':
                num_nodes = 50
                graph = SmallWorldGraph(num_nodes=num_nodes, device=DEVICE)
        
        logger.info(f"Created graph: {graph_type} with {num_nodes} nodes")
        
    except Exception as e:
        logger.exception(f"Failed to create graph: {e}")
        raise
    
    adj_matrix = graph.generate_adjacency_matrix()
    degrees = torch.sum(adj_matrix, dim=1)
    
    logger.info(f"Degrees: {degrees.cpu().numpy()}")
    logger.info(f"Adj matrix shape: {adj_matrix.shape}")
    
    # Move to GPU
    adj_matrix = adj_matrix.to(DEVICE)
    degrees = degrees.to(DEVICE)
    logger.info(f"Moved graph data to {DEVICE}")
    
    modes = ['state', 'stateless']
    
    for r_type in REWARD_TYPES:
        for mode in modes:
            logger.info(f"\n{'-'*50}")
            logger.info(f"REWARD: {r_type}, MODE: {mode}")
            logger.info(f"{'-'*50}")
            
            th_delta_q = calculate_theoretical_q_diff(r_type, 1.0, degrees)
            logger.info(f"Theoretical Delta Q: {th_delta_q.cpu().numpy()}")
            
            all_results = {b: {'coop': [], 'delta_q': []} for b in B_VALUES}
            
            for b_val in B_VALUES:
                logger.info(f"Processing b={b_val}")
                
                n_batches = (N_REPLICATIONS + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"Number of batches: {n_batches}")
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min((batch_idx + 1) * BATCH_SIZE, N_REPLICATIONS)
                    batch_reps = list(range(start_idx, end_idx))
                    actual_batch = len(batch_reps)
                    
                    logger.info(f"  Batch {batch_idx+1}/{n_batches}: replications {start_idx}-{end_idx-1} (size={actual_batch})")
                    
                    try:
                        coop_rates, delta_q_hist = run_batched_simulations(
                            batch_reps, b_val, r_type, mode, num_nodes,
                            adj_matrix, degrees, gamma=0.9
                        )
                        
                        # Store results
                        for i in range(actual_batch):
                            all_results[b_val]['coop'].append(coop_rates[i])
                            all_results[b_val]['delta_q'].append(delta_q_hist[:, i, :, :])
                        
                        logger.info(f"  Batch {batch_idx+1} complete. Coop rates: {coop_rates.round(3)}")
                        
                    except Exception as e:
                        logger.exception(f"  Batch {batch_idx+1} FAILED: {e}")
                        # Continue with next batch instead of crashing
                        continue
                
                # Stack results
                try:
                    all_results[b_val]['coop'] = np.array(all_results[b_val]['coop'])
                    all_results[b_val]['delta_q'] = np.array(all_results[b_val]['delta_q'])
                    logger.info(f"  b={b_val} complete. Shape: {all_results[b_val]['delta_q'].shape}")
                    logger.info(f"  Mean coop rates: {all_results[b_val]['coop'].mean(axis=0).round(3)}")
                except Exception as e:
                    logger.exception(f"  Failed to stack results for b={b_val}: {e}")
                    continue
            
            # Generate plots
            logger.info("Generating plots...")
            try:
                plot_all_results(all_results, B_VALUES, r_type, mode, graph_type, 
                               degrees.cpu().numpy(), th_delta_q.cpu().numpy(), output_dir)
                logger.info("Plots generated successfully")
            except Exception as e:
                logger.exception(f"Plot generation failed: {e}")


def plot_all_results(results, b_values, r_type, mode, graph_type, degrees, 
                     th_delta_q, output_dir):
    """Generate all plot types with error handling."""
    logger.info("  Plotting convergence...")
    for b_val in b_values:
        try:
            plot_convergence(results[b_val], b_val, r_type, mode, graph_type, 
                          degrees, th_delta_q, output_dir)
        except Exception as e:
            logger.error(f"    Convergence plot b={b_val} failed: {e}")
    
    logger.info("  Plotting distributions...")
    for b_val in b_values:
        try:
            plot_delta_q_distribution(results[b_val], b_val, r_type, mode, 
                                      graph_type, degrees, output_dir)
        except Exception as e:
            logger.error(f"    Distribution plot b={b_val} failed: {e}")
    
    logger.info("  Plotting cooperation rates...")
    try:
        plot_cooperation_rates(results, b_values, degrees, r_type, mode, 
                              graph_type, output_dir)
    except Exception as e:
        logger.error(f"    Cooperation rates plot failed: {e}")
    
    logger.info("  Plotting statistics...")
    try:
        plot_statistics(results, b_values, r_type, mode, graph_type, output_dir)
    except Exception as e:
        logger.error(f"    Statistics plot failed: {e}")


# [Plotting functions remain the same but with added logger.debug calls]
# ... (plot_convergence, plot_delta_q_distribution, etc. with logger.debug added)


if __name__ == "__main__":
    import time
    start = time.time()
    
    logger.info("Starting main loop...")
    
    for graph_type in GRAPH_TYPES:
        try:
            run_experiment_single_process(graph_type)
        except Exception as e:
            logger.exception(f"Graph type {graph_type} failed completely: {e}")
            continue
    
    elapsed = time.time() - start
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE: {elapsed/3600:.2f} hours")
    logger.info(f"Log file: {log_file}")