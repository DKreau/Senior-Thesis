import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import pandas as pd
import sys
import json
import networkx as nx

# Import the EPOBCModelWithInternet class from the module
# Assuming epobc_internet.py is in the same directory
from epobc_internet import EPOBCModelWithInternet

class MetricsCalculator:
    """
    Class to calculate various opinion dynamics metrics from the model.
    """
    @staticmethod
    def opinion_variance(opinions):
        """Calculate the variance of opinions"""
        return np.var(list(opinions.values()))
    
    @staticmethod
    def polarization_index(opinions):
        """
        Calculate polarization index as the difference between means of top and bottom quartiles,
        normalized to the opinion range.
        """
        values = np.array(list(opinions.values()))
        q1_threshold = np.percentile(values, 25)
        q4_threshold = np.percentile(values, 75)
        
        q1_values = values[values <= q1_threshold]
        q4_values = values[values >= q4_threshold]
        
        if len(q1_values) == 0 or len(q4_values) == 0:
            return 0.0
        
        mean_q1 = np.mean(q1_values)
        mean_q4 = np.mean(q4_values)
        
        # Normalize by dividing by 2 (since opinion range is [-1, 1])
        return (mean_q4 - mean_q1) / 2
    
    @staticmethod
    def count_opinion_clusters(opinions, epsilon):
        """
        Count the number of distinct opinion clusters.
        
        A cluster is defined as a group of opinions where each opinion is within
        epsilon distance of at least one other opinion in the cluster.
        
        Parameters:
        - opinions: Dictionary mapping agent IDs to opinion values
        - epsilon: The confidence threshold
        
        Returns:
        - num_clusters: Number of distinct opinion clusters
        """
        # Convert opinions to a list of values
        values = list(opinions.values())
        N = len(values)
        
        # If there are no opinions, return 0 clusters
        if N == 0:
            return 0
        
        # Create a distance matrix
        distance_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = abs(values[i] - values[j])
        
        # Create a graph where nodes are agents and edges connect agents
        # whose opinions are within epsilon distance
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        for i in range(N):
            for j in range(i+1, N):  # Only need to check one direction
                if distance_matrix[i, j] <= epsilon:
                    G.add_edge(i, j)
        
        # Count the number of connected components in the graph
        # Each connected component represents a cluster
        return nx.number_connected_components(G)
    
    @staticmethod
    def pluralistic_ignorance_index(private_opinions, expressed_opinions):
        """
        Calculate the Pluralistic Ignorance Index (PII).
        PII measures the average discrepancy between private and expressed opinions.
        """
        agent_ids = list(private_opinions.keys())
        
        # Calculate absolute differences for each agent
        differences = [abs(private_opinions[i] - expressed_opinions[i]) for i in agent_ids]
        
        return np.mean(differences)

class EpsilonExperiment:
    def __init__(self, output_dir="epsilon_experiment_results"):
        """Initialize the experiment with the given output directory."""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a timestamped directory for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
    
    def create_base_config(self, num_agents=100, seed=None):
        """
        Create a base configuration for the experiment.
        
        Parameters:
        - num_agents: Number of agents in the network
        - seed: Random seed for reproducibility (different for each run)
        
        Returns:
        - config: Configuration dictionary
        """
        # Initialize lambda and phi values
        lambda_vals = {i: 0.6 for i in range(num_agents)}  # Moderate stubbornness
        phi_vals = {i: 0.6 for i in range(num_agents)}     # Moderate conformity resistance
        
        # Set random seed if provided (will be different for each run)
        if seed is not None:
            np.random.seed(seed)
        
        # Create initial private opinions (uniformly distributed from -1 to 1)
        initial_opinions = {i: np.random.uniform(-1, 1) for i in range(num_agents)}
        
        config = {
            "num_agents": num_agents,
            "epsilon": 0.3,  # Will be overridden
            "theta": 0.0,    # No internet influence for baseline
            "lambda_vals": lambda_vals,
            "phi_vals": phi_vals,
            "initial_opinions": initial_opinions,
            "num_internet_nodes": 0,  # No internet nodes for baseline
            "zipf_exponent": 1.5,
            "base_posting_prob": 0.2,
            "internet_alpha": 0.5,
            "beta": 1.0,
            "steps": 50
        }
        
        return config
    
    def run_epsilon_experiment(self, epsilon_values, num_agents=100, steps=50, replications=10):
        """
        Run the experiment with varying epsilon values.
        
        Parameters:
        - epsilon_values: List of epsilon values to test
        - num_agents: Number of agents in the network
        - steps: Number of simulation steps
        - replications: Number of replications for each configuration (default: 10)
        
        Returns:
        - results_df: DataFrame with experiment results
        """
        print(f"Starting epsilon experiment with {len(epsilon_values)} values, {replications} replications each...")
        
        # Initialize results storage
        results = []
        
        total_runs = len(epsilon_values) * replications
        run_count = 0
        
        # Create plots directory
        plots_dir = os.path.join(self.run_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        time_plots_dir = os.path.join(plots_dir, "time_evolution")
        if not os.path.exists(time_plots_dir):
            os.makedirs(time_plots_dir)
        
        # Create a master seed for this experiment
        master_seed = int(time.time())
        print(f"Master seed for this experiment: {master_seed}")
        
        for epsilon in epsilon_values:
            print(f"\nRunning experiment with epsilon = {epsilon}")
            
            # Lists to track models and metrics for this epsilon value
            epsilon_models = []
            epsilon_metrics = []
            
            for rep in range(replications):
                run_count += 1
                print(f"  Replication {rep+1}/{replications} (Run {run_count}/{total_runs})")
                
                # Create a unique seed for this run
                run_seed = master_seed + run_count
                print(f"  Using seed: {run_seed}")
                
                # Create configuration for this run with a unique seed
                config = self.create_base_config(num_agents, seed=run_seed)
                config["epsilon"] = epsilon
                config["steps"] = steps
                
                # Run the model
                start_time = time.time()
                model = EPOBCModelWithInternet(config)
                model.run()
                elapsed_time = time.time() - start_time
                
                print(f"  Simulation completed in {elapsed_time:.2f} seconds. Calculating metrics...")
                
                # Extract final state
                final_state = model.history[-1]
                private_opinions = final_state["private_opinions"]
                expressed_opinions = final_state["expressed_opinions"]
                
                # Count the number of clusters
                num_clusters_private = self.metrics_calculator.count_opinion_clusters(private_opinions, epsilon)
                num_clusters_expressed = self.metrics_calculator.count_opinion_clusters(expressed_opinions, epsilon)
                
                # Calculate metrics at final state
                metrics = {
                    "epsilon": epsilon,
                    "replication": rep,
                    "seed": run_seed,
                    "opinion_variance_private": self.metrics_calculator.opinion_variance(private_opinions),
                    "opinion_variance_expressed": self.metrics_calculator.opinion_variance(expressed_opinions),
                    "polarization_index_private": self.metrics_calculator.polarization_index(private_opinions),
                    "polarization_index_expressed": self.metrics_calculator.polarization_index(expressed_opinions),
                    "num_clusters_private": num_clusters_private,
                    "num_clusters_expressed": num_clusters_expressed,
                    "pluralistic_ignorance_index": self.metrics_calculator.pluralistic_ignorance_index(private_opinions, expressed_opinions)
                }
                
                # Store model and metrics
                epsilon_models.append(model)
                epsilon_metrics.append(metrics)
                
                # Add to overall results
                results.append(metrics)
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(self.run_dir, "epsilon_experiment_results.csv"), index=False)
            
            # After all replications for this epsilon, find runs with most and fewest clusters
            cluster_counts = [m["num_clusters_private"] for m in epsilon_metrics]
            most_clusters_idx = np.argmax(cluster_counts)
            fewest_clusters_idx = np.argmin(cluster_counts)
            
            print(f"  For epsilon={epsilon}:")
            print(f"    Most clusters: {cluster_counts[most_clusters_idx]} (Rep {epsilon_metrics[most_clusters_idx]['replication']+1})")
            print(f"    Fewest clusters: {cluster_counts[fewest_clusters_idx]} (Rep {epsilon_metrics[fewest_clusters_idx]['replication']+1})")
            
            # Save plots only for the runs with most and fewest clusters
            self._plot_opinion_evolution(
                epsilon_models[most_clusters_idx], 
                epsilon, 
                epsilon_metrics[most_clusters_idx]["replication"], 
                time_plots_dir,
                label=f"most_clusters_{cluster_counts[most_clusters_idx]}"
            )
            
            self._plot_opinion_evolution(
                epsilon_models[fewest_clusters_idx], 
                epsilon, 
                epsilon_metrics[fewest_clusters_idx]["replication"], 
                time_plots_dir,
                label=f"fewest_clusters_{cluster_counts[fewest_clusters_idx]}"
            )
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save final results
        results_df.to_csv(os.path.join(self.run_dir, "epsilon_experiment_results.csv"), index=False)
        
        print(f"Experiment completed. Results saved to {self.run_dir}")
        
        return results_df
    
    def _plot_opinion_evolution(self, model, epsilon, replication, save_dir, label=None):
        """
        Plot the evolution of private and expressed opinions over time.
        
        Parameters:
        - model: The simulation model
        - epsilon: The epsilon value used
        - replication: The replication number
        - save_dir: Directory to save the plot
        - label: Optional label to add to the filename (e.g., "most_clusters_5")
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
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot private opinions
        plt.subplot(2, 1, 1)
        for agent in range(num_agents):
            plt.plot(time_steps, private_opinions[agent], color='blue', alpha=0.2)
        
        # Create title with run number clearly indicated
        title_prefix = f"Private Opinions Over Time (ε = {epsilon}, Run {replication+1})"
        if label:
            # Extract descriptive part and value from label (e.g., "most_clusters_5" -> "Most Clusters: 5")
            if "most_clusters" in label:
                description = "Most Clusters"
                value = label.split('_')[-1]
                title_suffix = f"{description}: {value}"
            elif "fewest_clusters" in label:
                description = "Fewest Clusters"
                value = label.split('_')[-1]
                title_suffix = f"{description}: {value}"
            else:
                title_suffix = label.replace('_', ' ').title()
            
            plt.title(f"{title_prefix}, {title_suffix}")
        else:
            plt.title(title_prefix)
            
        plt.xlabel("Time Step")
        plt.ylabel("Opinion Value")
        plt.ylim(-1.1, 1.1)
        plt.grid(alpha=0.3)
        
        # Plot expressed opinions
        plt.subplot(2, 1, 2)
        for agent in range(num_agents):
            plt.plot(time_steps, expressed_opinions[agent], color='red', alpha=0.2)
        
        # Create title with run number clearly indicated
        title_prefix = f"Expressed Opinions Over Time (ε = {epsilon}, Run {replication+1})"
        if label:
            if "most_clusters" in label or "fewest_clusters" in label:
                plt.title(f"{title_prefix}, {title_suffix}")
            else:
                plt.title(f"{title_prefix}, {label.replace('_', ' ').title()}")
        else:
            plt.title(title_prefix)
            
        plt.xlabel("Time Step")
        plt.ylabel("Opinion Value")
        plt.ylim(-1.1, 1.1)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"opinion_evolution_epsilon_{epsilon}_run_{replication+1}"
        if label:
            filename = f"opinion_evolution_epsilon_{epsilon}_{label}_run_{replication+1}"
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_results(self, results_df):
        """
        Plot the experiment results.
        
        Parameters:
        - results_df: DataFrame with experiment results
        """
        # Group results by epsilon and calculate mean across replications
        grouped_results = results_df.groupby("epsilon").mean().reset_index()
        
        # Create directory for plots
        plots_dir = os.path.join(self.run_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot 1: Opinion Variance vs Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_results["epsilon"], grouped_results["opinion_variance_private"], "o-", label="Private Opinions")
        plt.plot(grouped_results["epsilon"], grouped_results["opinion_variance_expressed"], "o-", label="Expressed Opinions")
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Opinion Variance")
        plt.title("Impact of Confidence Threshold on Opinion Variance")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "opinion_variance_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        # Plot 2: Polarization Index vs Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_results["epsilon"], grouped_results["polarization_index_private"], "o-", label="Private Opinions")
        plt.plot(grouped_results["epsilon"], grouped_results["polarization_index_expressed"], "o-", label="Expressed Opinions")
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Polarization Index")
        plt.title("Impact of Confidence Threshold on Polarization")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "polarization_index_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        # Plot 3: Number of Clusters vs Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_results["epsilon"], grouped_results["num_clusters_private"], "o-")
        plt.plot(grouped_results["epsilon"], grouped_results["num_clusters_expressed"], "o-")
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Number of Clusters")
        plt.title("Impact of Confidence Threshold on Number of Opinion Clusters")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "num_clusters_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        # Plot 4: Pluralistic Ignorance Index vs Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_results["epsilon"], grouped_results["pluralistic_ignorance_index"], "o-")
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Pluralistic Ignorance Index")
        plt.title("Impact of Confidence Threshold on Pluralistic Ignorance")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "pluralistic_ignorance_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        # Combined plot
        plt.figure(figsize=(12, 10))
        
        # Normalize values for better comparison
        metrics = [
            "opinion_variance_private", 
            "polarization_index_private", 
            "num_clusters_private",
            "pluralistic_ignorance_index"
        ]
        
        normalized_results = grouped_results.copy()
        for metric in metrics:
            max_val = normalized_results[metric].max()
            if max_val > 0:  # Avoid division by zero
                normalized_results[metric] = normalized_results[metric] / max_val
        
        plt.plot(normalized_results["epsilon"], normalized_results["opinion_variance_private"], "o-", label="Opinion Variance")
        plt.plot(normalized_results["epsilon"], normalized_results["polarization_index_private"], "o-", label="Polarization Index")
        plt.plot(normalized_results["epsilon"], normalized_results["num_clusters_private"], "o-", label="Number of Clusters")
        plt.plot(normalized_results["epsilon"], normalized_results["pluralistic_ignorance_index"], "o-", label="Pluralistic Ignorance")
        
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Normalized Metric Value")
        plt.title("Impact of Confidence Threshold on Key Opinion Dynamics Metrics")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "combined_metrics_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        # Plot 5: Box plot of number of clusters by epsilon
        plt.figure(figsize=(12, 8))
        data = []
        labels = []
        
        for epsilon in sorted(results_df["epsilon"].unique()):
            sub_df = results_df[results_df["epsilon"] == epsilon]
            data.append(sub_df["num_clusters_private"].values)
            labels.append(f"{epsilon}")
        
        plt.boxplot(data, labels=labels)
        plt.xlabel("Confidence Threshold (ε)")
        plt.ylabel("Number of Clusters")
        plt.title("Distribution of Number of Clusters across Replications")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "clusters_boxplot_vs_epsilon.png"), dpi=300, bbox_inches="tight")
        
        print(f"Plots saved to {plots_dir}")

def main():
    # Define epsilon values to test
    epsilon_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 
                 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 
                 0.8, 0.9]
    
    # Initialize experiment
    experiment = EpsilonExperiment()
    
    # Run experiment with 10 replications for each epsilon value
    results = experiment.run_epsilon_experiment(
        epsilon_values=epsilon_values,
        num_agents=50,  # Smaller network for faster computation
        steps=50,       # Number of steps to run each simulation
        replications=50 # Run 10 times for each epsilon value
    )
    
    # Plot results
    experiment.plot_results(results)
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()