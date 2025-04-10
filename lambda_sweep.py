import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import pandas as pd
from epobc_internet import EPOBCModelWithInternet

class LambdaSweep:
    """Class to run a parameter sweep of the lambda (stubbornness) parameter."""
    
    def __init__(self, output_dir="lambda_sweep_results"):
        """
        Initialize the experiment with the given output directory.
        
        Parameters:
        - output_dir: Directory to save results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a timestamped directory for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir)
        
        # Create directories for plots
        self.plots_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plots_dir)
        
        self.opinion_plots_dir = os.path.join(self.plots_dir, "opinion_evolution")
        os.makedirs(self.opinion_plots_dir)
    
    def create_base_config(self, num_agents=100, seed=None):
        """
        Create a base configuration with fixed epsilon=0.3.
        
        Parameters:
        - num_agents: Number of agents in the network
        - seed: Random seed for reproducibility
        
        Returns:
        - config: Configuration dictionary
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create initial private opinions (uniformly distributed from -1 to 1)
        initial_opinions = {i: np.random.uniform(-1, 1) for i in range(num_agents)}
        
        # Default lambda and phi values
        lambda_vals = {i: 0.5 for i in range(num_agents)}  # Will be overridden
        phi_vals = {i: 0.6 for i in range(num_agents)}     # Fixed conformity resistance
        
        config = {
            "num_agents": num_agents,
            "epsilon": 0.3,           # Fixed epsilon as requested
            "theta": 0.3,             # Internet influence weight
            "lambda_vals": lambda_vals,
            "phi_vals": phi_vals,
            "initial_opinions": initial_opinions,
            "num_internet_nodes": 1,
            "zipf_exponent": 1.5,
            "base_posting_prob": 0.2,
            "internet_alpha": 0.5,
            "beta": 1.0,
            "steps": 50
        }
        
        return config
    
    def calculate_time_to_stability(self, history, threshold=0.001):
        """
        Calculate the number of time steps until opinions stabilize.
        
        Parameters:
        - history: List of state dictionaries from model history
        - threshold: Maximum change considered stable
        
        Returns:
        - time_steps: Number of time steps until stabilization (-1 if never stabilizes)
        """
        if len(history) < 2:
            return -1  # Not enough history to determine stabilization
        
        for t in range(1, len(history)):
            # Get current and previous private opinions
            current_private = history[t]["private_opinions"]
            prev_private = history[t-1]["private_opinions"]
            
            # Get current and previous expressed opinions
            current_expressed = history[t]["expressed_opinions"]
            prev_expressed = history[t-1]["expressed_opinions"]
            
            # Calculate maximum change in opinions
            max_change_private = max([abs(current_private[i] - prev_private[i]) for i in current_private])
            max_change_expressed = max([abs(current_expressed[i] - prev_expressed[i]) for i in current_expressed])
            
            max_change = max(max_change_private, max_change_expressed)
            
            # Check if stabilized
            if max_change < threshold:
                return t  # Stabilized at this time step
        
        # If we get here, opinions never stabilized
        return -1
    
    def calculate_private_expressed_gap(self, private_opinions, expressed_opinions):
        """
        Calculate the average gap between private and expressed opinions.
        
        Parameters:
        - private_opinions: Dictionary of private opinions
        - expressed_opinions: Dictionary of expressed opinions
        
        Returns:
        - gap: Average absolute difference between private and expressed opinions
        """
        agent_ids = list(private_opinions.keys())
        differences = [abs(private_opinions[i] - expressed_opinions[i]) for i in agent_ids]
        return np.mean(differences)
    
    def plot_opinion_evolution(self, model, lambda_val, replication):
        """
        Plot the evolution of private and expressed opinions over time.
        
        Parameters:
        - model: The simulation model
        - lambda_val: The lambda value used
        - replication: The replication number
        """
        history = model.get_history()
        num_agents = model.num_agents
        num_steps = len(history)
        time_steps = list(range(num_steps))
        
        # Extract opinion evolution data
        private_opinions = {}
        expressed_opinions = {}
        
        for agent in range(num_agents):
            private_opinions[agent] = []
            expressed_opinions[agent] = []
            
            for step in history:
                private_opinions[agent].append(step["private_opinions"][agent])
                expressed_opinions[agent].append(step["expressed_opinions"][agent])
        
        # Create the plot - combined private and expressed opinions
        plt.figure(figsize=(12, 8))
        
        # Plot private opinions in blue (dashed)
        for agent in range(num_agents):
            plt.plot(time_steps, private_opinions[agent], color='blue', linestyle='--', alpha=0.2)
        
        # Plot expressed opinions in red (solid)
        for agent in range(num_agents):
            plt.plot(time_steps, expressed_opinions[agent], color='red', linestyle='-', alpha=0.2)
        
        # Add lines for the legend
        plt.plot([], [], color='blue', linestyle='--', label='Private Opinions', alpha=0.8)
        plt.plot([], [], color='red', linestyle='-', label='Expressed Opinions', alpha=0.8)
        
        # Create title with lambda value
        plt.title(f"Opinion Evolution Over Time (λ = {lambda_val}, Run {replication+1})")
        plt.xlabel("Time Step")
        plt.ylabel("Opinion Value")
        plt.ylim(-1.1, 1.1)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add a vertical line at the stabilization point if available
        stabilization_time = self.calculate_time_to_stability(model.history)
        if stabilization_time > 0:
            plt.axvline(x=stabilization_time, color='green', linestyle=':', linewidth=2, 
                        label=f'Stabilization: Step {stabilization_time}')
            plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"opinion_evolution_lambda_{lambda_val}_run_{replication+1}.png"
        plt.savefig(os.path.join(self.opinion_plots_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
    
    def run_lambda_sweep(self, lambda_values, num_agents=100, steps=50, replications=10):
        """
        Run the experiment with varying lambda values.
        
        Parameters:
        - lambda_values: List of lambda values to test
        - num_agents: Number of agents in the network
        - steps: Number of simulation steps
        - replications: Number of replications for each configuration
        
        Returns:
        - results_df: DataFrame with experiment results
        """
        print(f"Starting lambda sweep with {len(lambda_values)} values, {replications} replications each...")
        
        # Initialize results storage
        results = []
        
        total_runs = len(lambda_values) * replications
        run_count = 0
        
        # Create a master seed for this experiment
        master_seed = int(time.time())
        print(f"Master seed for this experiment: {master_seed}")
        
        # For each lambda value
        for lambda_val in lambda_values:
            print(f"\nRunning experiment with lambda = {lambda_val}")
            
            # Store models and data for each replication
            lambda_models = []
            stabilization_times = []
            private_expressed_gaps = []
            
            # For each replication
            for rep in range(replications):
                run_count += 1
                print(f"  Replication {rep+1}/{replications} (Run {run_count}/{total_runs})")
                
                # Create a unique seed for this run
                run_seed = master_seed + run_count
                
                # Create configuration with this lambda
                config = self.create_base_config(num_agents, seed=run_seed)
                config["lambda_vals"] = {i: lambda_val for i in range(num_agents)}
                config["steps"] = steps
                
                # Run the model
                start_time = time.time()
                model = EPOBCModelWithInternet(config)
                model.run()
                elapsed_time = time.time() - start_time
                
                print(f"  Simulation completed in {elapsed_time:.2f} seconds.")
                
                # Store the model
                lambda_models.append(model)
                
                # Calculate time to stabilization
                time_to_stabilize = self.calculate_time_to_stability(model.history)
                stabilization_times.append(time_to_stabilize)
                
                # Calculate private-expressed opinion gap over time
                gaps_over_time = []
                for step in model.history:
                    gap = self.calculate_private_expressed_gap(
                        step["private_opinions"], step["expressed_opinions"])
                    gaps_over_time.append(gap)
                
                # Store results - use lambda_val instead of "lambda" to avoid keyword issues
                results.append({
                    "lambda_val": lambda_val,
                    "replication": rep,
                    "seed": run_seed,
                    "time_to_stabilization": time_to_stabilize,
                    "final_gap": gaps_over_time[-1] if gaps_over_time else 0,
                    "gaps_over_time": gaps_over_time
                })
                
                # Create opinion evolution plot for only the first 2 runs
                if rep < 2:
                    self.plot_opinion_evolution(model, lambda_val, rep)
                
                # Save intermediate results
                df = pd.DataFrame([r for r in results if "gaps_over_time" not in r])
                df.to_csv(os.path.join(self.run_dir, "lambda_sweep_results.csv"), index=False)
            
            # Report statistics for this lambda value
            valid_times = [t for t in stabilization_times if t >= 0]
            if valid_times:
                avg_time = np.mean(valid_times)
                print(f"  Average time to stabilization: {avg_time:.2f} steps")
            else:
                print("  No runs stabilized within the simulation timeframe")
        
        # Create a clean DataFrame for the final results (without the time series data)
        clean_results = [r.copy() for r in results]
        for r in clean_results:
            if "gaps_over_time" in r:
                del r["gaps_over_time"]
        
        results_df = pd.DataFrame(clean_results)
        
        # Save final results
        results_df.to_csv(os.path.join(self.run_dir, "lambda_sweep_results.csv"), index=False)
        
        # Create summary plots
        self.plot_summary_results(results)
        
        print(f"Experiment completed. Results saved to {self.run_dir}")
        
        return results
    
    def plot_summary_results(self, results):
        """
        Create summary plots for the experiment.
        
        Parameters:
        - results: List of result dictionaries
        """
        # Create a DataFrame for plotting - make a deep copy to avoid modifying original
        plotting_results = []
        for r in results:
            if "gaps_over_time" in r:
                plot_result = {k: v for k, v in r.items() if k != "gaps_over_time"}
                plotting_results.append(plot_result)
            else:
                plotting_results.append(r.copy())
        
        results_df = pd.DataFrame(plotting_results)
        
        # Debug print to see actual column names
        print(f"DataFrame columns: {results_df.columns.tolist()}")
        
        # 1. Plot average time to stabilization vs lambda
        self.plot_time_to_stability(results_df)
        
        # 2. Plot private-expressed opinion gap over time for selected lambda values
        self.plot_opinion_gap_over_time(results)
    
    def plot_time_to_stability(self, results_df):
        """
        Plot average time to stabilization vs lambda.
        
        Parameters:
        - results_df: DataFrame with results
        """
        plt.figure(figsize=(10, 6))
        
        # Handle potential column name differences - some systems might convert "lambda" to other names
        lambda_col = None
        for possible_name in ["lambda", "Lambda", "lambda_val"]:
            if possible_name in results_df.columns:
                lambda_col = possible_name
                break
        
        if lambda_col is None:
            print("ERROR: Lambda column not found in DataFrame. Column names:", results_df.columns.tolist())
            return
            
        # Manual calculation of average time to stabilization for each lambda value
        lambda_values = sorted(results_df[lambda_col].unique())
        avg_times = []
        
        for lambda_val in lambda_values:
            # Get all stabilization times for this lambda value
            times = results_df[results_df[lambda_col] == lambda_val]["time_to_stabilization"].values
            # Filter for valid times (>= 0)
            valid_times = [t for t in times if t >= 0]
            # Calculate average if we have valid times
            if valid_times:
                avg_times.append(np.mean(valid_times))
            else:
                avg_times.append(np.nan)
        
        # Create data for plotting
        plot_data = pd.DataFrame({
            "lambda": lambda_values,
            "avg_time": avg_times
        })
        
        # Plot only where we have valid data
        valid_data = plot_data.dropna()
        if not valid_data.empty:
            plt.plot(valid_data["lambda"], valid_data["avg_time"], 
                     "o-", color="purple", linewidth=2)
            
            plt.xlabel("Stubbornness (λ)")
            plt.ylabel("Average Time Steps to Stabilization")
            plt.title("Impact of Stubbornness on Time to Opinion Stabilization")
            plt.grid(alpha=0.3)
            
            # Add value annotations
            for x, y in zip(valid_data["lambda"], valid_data["avg_time"]):
                plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                             xytext=(0, 10), ha='center')
            
            plt.savefig(os.path.join(self.plots_dir, "time_to_stability_vs_lambda.png"), 
                        dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def plot_opinion_gap_over_time(self, results):
        """
        Plot average private-expressed opinion gap over time for key lambda values.
        
        Parameters:
        - results: List of result dictionaries (including gaps_over_time)
        """
        plt.figure(figsize=(12, 8))
        
        # Get all unique lambda values
        lambda_values = sorted(set(r["lambda_val"] for r in results))
        
        # Choose a subset of lambda values to plot
        if len(lambda_values) > 5:
            # Pick evenly spaced values
            indices = np.linspace(0, len(lambda_values)-1, 5).astype(int)
            selected_lambdas = [lambda_values[i] for i in indices]
        else:
            selected_lambdas = lambda_values
        
        for lambda_val in selected_lambdas:
            # Get all replications for this lambda
            lambda_results = [r for r in results if r["lambda_val"] == lambda_val and "gaps_over_time" in r]
            
            if lambda_results:
                # Find the minimum length to align all time series
                min_length = min(len(r["gaps_over_time"]) for r in lambda_results)
                
                # Get aligned gap time series
                gaps_matrix = [r["gaps_over_time"][:min_length] for r in lambda_results]
                
                # Calculate the average gap at each time step
                avg_gaps = np.mean(gaps_matrix, axis=0)
                
                # Plot the average gap over time
                plt.plot(range(len(avg_gaps)), avg_gaps, 
                         linewidth=2, label=f"λ = {lambda_val}")
        
        plt.xlabel("Time Step")
        plt.ylabel("Average Private-Expressed Opinion Gap")
        plt.title("Evolution of Private-Expressed Opinion Gap Over Time")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, "opinion_gap_over_time.png"), 
                    dpi=300, bbox_inches="tight")
        plt.close()


def main():
    # Define lambda values to test (from 0.1 to 0.9)
    lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Initialize experiment
    experiment = LambdaSweep()
    
    # Run experiment
    experiment.run_lambda_sweep(
        lambda_values=lambda_values,
        num_agents=50,      # Smaller network for faster computation
        steps=50,           # Number of steps to run each simulation
        replications=10      # Run 5 times for each lambda value
    )
    
    print("Lambda sweep completed successfully!")

if __name__ == "__main__":
    main()