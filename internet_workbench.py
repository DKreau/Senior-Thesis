import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import random
from datetime import datetime

class OpinionDynamicsModel:
    def __init__(self, num_agents: int, epsilon: float, lambda_vals: Dict[int, float], 
                 phi_vals: Dict[int, float], initial_opinions: Dict[int, float], 
                 use_internet: bool = False, theta: float = 0.3, internet_posters: Dict[int, bool] = None):
        """
        Initialize the opinion dynamics model with a dynamic network and optional internet node.

        Parameters:
        - num_agents: Number of agents in the network
        - epsilon: Confidence threshold for connecting nodes based on opinion similarity
        - lambda_vals: Stubbornness values for each agent (0 <= lambda <= 1)
        - phi_vals: Conformity resistance values for each agent (0 <= phi <= 1)
        - initial_opinions: Initial private opinions for each agent
        - use_internet: Whether to include an internet node
        - theta: Weight of internet influence (0 <= theta <= 1)
        - internet_posters: Dictionary specifying which agents can post to the internet
        """
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.lambda_vals = lambda_vals
        self.phi_vals = phi_vals
        self.private_opinions = initial_opinions.copy()
        self.expressed_opinions = initial_opinions.copy()
        
        # Internet node parameters
        self.use_internet = use_internet
        self.theta = theta if use_internet else 0.0
        self.internet_posters = internet_posters or {i: True for i in range(num_agents)}
        self.internet_opinion = sum(initial_opinions.values()) / len(initial_opinions)  # Initial internet opinion
        self.current_posters = []    # Agents who posted in the current step
        
        # Network structure
        self.network = nx.Graph()
        self.network.add_nodes_from(range(num_agents))
        self._update_network()
        
        # History to store model state over time
        self.history = []
        self._save_state()  # Save initial state
    
    def _update_network(self):
        """Update the network structure based on expressed opinions and bounded confidence."""
        self.network.clear_edges()  # Remove all edges but keep nodes
        
        # Add edges based on bounded confidence
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if abs(self.expressed_opinions[i] - self.expressed_opinions[j]) <= self.epsilon:
                    self.network.add_edge(i, j)
    
    def _update_internet_opinion(self):
        """Update the internet node's opinion based on posters."""
        if not self.use_internet:
            return
            
        # Identify agents who post (using the boolean flags)
        posting_agents = [i for i in range(self.num_agents) if self.internet_posters.get(i, False)]
        
        # Update internet opinion based only on posting agents
        if posting_agents:
            self.internet_opinion = np.mean([self.expressed_opinions[i] for i in posting_agents])
            self.current_posters = posting_agents
        else:
            # If nobody posts, internet opinion stays the same
            self.current_posters = []
    
    def _update_private_opinion(self, agent: int):
        """Update private opinion of a given agent."""
        # Get neighbors from the network
        neighbors = list(self.network.neighbors(agent))
        
        if neighbors:  # Ensure the agent has neighbors
            # Calculate neighbor influence as average of neighbors' expressed opinions
            neighbor_expressed_opinions = [self.expressed_opinions[neighbor] for neighbor in neighbors]
            neighbor_influence = np.mean(neighbor_expressed_opinions)
        else:
            # No neighbors, use current private opinion
            neighbor_influence = self.private_opinions[agent]
        
        # Calculate combined influence (internet + neighbors)
        if self.use_internet:
            combined_influence = (1 - self.theta) * neighbor_influence + self.theta * self.internet_opinion
        else:
            combined_influence = neighbor_influence
        
        # Update private opinion using the stubbornness parameter
        self.private_opinions[agent] = (
            self.lambda_vals[agent] * self.private_opinions[agent] +
            (1 - self.lambda_vals[agent]) * combined_influence
        )
    
    def _update_expressed_opinion(self, agent: int):
        """Update expressed opinion of a given agent."""
        # Get neighbors from the network
        neighbors = list(self.network.neighbors(agent))
        
        if neighbors:
            # Calculate neighbor influence as average of neighbors' expressed opinions
            neighbor_expressed_opinions = [self.expressed_opinions[neighbor] for neighbor in neighbors]
            neighbor_influence = np.mean(neighbor_expressed_opinions)
        else:
            # No neighbors, use current expressed opinion
            neighbor_influence = self.expressed_opinions[agent]
        
        # Calculate combined influence (internet + neighbors)
        if self.use_internet:
            combined_influence = (1 - self.theta) * neighbor_influence + self.theta * self.internet_opinion
        else:
            combined_influence = neighbor_influence
        
        # Update expressed opinion using the conformity resistance parameter
        self.expressed_opinions[agent] = (
            self.phi_vals[agent] * self.private_opinions[agent] +
            (1 - self.phi_vals[agent]) * combined_influence
        )
    
    def _save_state(self):
        """Save current state to history."""
        state = {
            "private_opinions": self.private_opinions.copy(),
            "expressed_opinions": self.expressed_opinions.copy(),
        }
        
        if self.use_internet:
            state["internet_opinion"] = self.internet_opinion
            state["posting_agents"] = self.current_posters.copy()
            
        self.history.append(state)
    
    def step(self):
        """Perform one time step of the simulation."""
        # First update the internet opinion based on current expressed opinions
        self._update_internet_opinion()
        
        # Then update network structure
        self._update_network()
        
        # Update private opinions for all agents
        new_private_opinions = {}
        for agent in range(self.num_agents):
            self._update_private_opinion(agent)
        
        # Update expressed opinions for all agents
        for agent in range(self.num_agents):
            self._update_expressed_opinion(agent)
        
        # Save current state to history
        self._save_state()
    
    def run(self, steps: int):
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()
        
        return self.history
    
    def get_history(self) -> List[Dict]:
        """Get the history of opinions over time."""
        return self.history


def plot_opinion_dynamics(model: OpinionDynamicsModel, run_name: str, save_dir: str = None, num_zealots: int = 0):
    """
    Plot the evolution of private and expressed opinions over time.
    
    Parameters:
    - model: The opinion dynamics model
    - run_name: Name for this run (used in the plot title and filename)
    - save_dir: Directory to save the plot (if None, plot is displayed but not saved)
    - num_zealots: Number of zealots to highlight differently (assuming they are the first n agents)
    """
    history = model.history
    num_agents = model.num_agents
    num_steps = len(history)
    time_steps = list(range(num_steps))
    
    # Extract opinion data
    private_opinions = {agent: [] for agent in range(num_agents)}
    expressed_opinions = {agent: [] for agent in range(num_agents)}
    internet_opinions = []
    
    for step in history:
        for agent in range(num_agents):
            private_opinions[agent].append(step["private_opinions"][agent])
            expressed_opinions[agent].append(step["expressed_opinions"][agent])
        
        if "internet_opinion" in step:
            internet_opinions.append(step["internet_opinion"])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot private opinions
    for agent in range(num_agents):
        if agent < num_zealots:  # Zealot
            plt.plot(time_steps, private_opinions[agent], color="purple", linestyle="dashed", alpha=0.7)
        else:  # Regular agent
            plt.plot(time_steps, private_opinions[agent], color="blue", linestyle="dashed", alpha=0.7)
    
    # Plot expressed opinions
    for agent in range(num_agents):
        if agent < num_zealots:  # Zealot
            plt.plot(time_steps, expressed_opinions[agent], color="purple", linestyle="solid", alpha=0.7)
        else:  # Regular agent
            plt.plot(time_steps, expressed_opinions[agent], color="red", linestyle="solid", alpha=0.7)
    
    # Plot internet opinion if available
    if internet_opinions:
        plt.plot(time_steps[1:], internet_opinions[1:], color="green", linewidth=3, label="Internet Opinion")
    
    # Add neutral opinion line
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    
    # Add legend
    plt.plot([], [], color="blue", linestyle="dashed", label="Private Opinions (Regular)")
    plt.plot([], [], color="red", linestyle="solid", label="Expressed Opinions (Regular)")
    if num_zealots > 0:
        plt.plot([], [], color="purple", linestyle="dashed", label="Private Opinions (Zealots)")
        plt.plot([], [], color="purple", linestyle="solid", label="Expressed Opinions (Zealots)")
    
    # Set labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Opinion Value")
    plt.title("Dynamics With Posting Zealots")
    plt.ylim(-1.1, 1.1)
    
    # Add parameter info to the plot
    
        
    internet_posters_count = sum(1 for i, posts in model.internet_posters.items() if posts)
    
    params_text = (
        f"Parameters:\n"
        f"Agents: {num_agents}\n"
        f"Zealots: {num_zealots}\n"
        f"ε (epsilon): {model.epsilon}\n"
        f"λ (lambda): {model.lambda_vals[num_agents-num_zealots-1]}\n"
        f"φ (phi): {model.phi_vals[num_agents-num_zealots-1]}\n"
        f"Internet: {'Yes' if model.use_internet else 'No'}\n"
    )
    if model.use_internet:
        params_text += f"θ (theta): {model.theta}\n"
        params_text += f"Internet Posters: {internet_posters_count}/{num_agents}"
    
    # Position in right side, a bit higher than bottom
    plt.figtext(0.97, 0.15, params_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7),
               horizontalalignment='right', verticalalignment='bottom')
    
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    # Save or display the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"{run_name}_dynamics.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()




def run_simulation():
    """Run an opinion dynamics simulation with user input for all parameters."""
    # Set the seed for reproducibility
    np.random.seed(42)
    random.seed(42)  # Optional, for Python's random module

    # Create a timestamp-based directory for saving outputs
    run_name = "internet_and_zealots_post"
    save_dir = f"opinion_dynamics"

    steps = 50
    num_zealots = 5
    num_agents = 55
    epsilon = 0.3

    lambda_vals = {i: 0.85 for i in range(num_agents)}
    phi_vals = {i: 0.3 for i in range(num_agents)}

    opinions = np.linspace(-1, 1, num_agents - num_zealots)
    initial_opinions = {i + num_zealots: opinions[i] for i in range(num_agents - num_zealots)}


    use_internet = True
    theta = 0.1

    internet_posters = {i: np.random.rand() < 0.15 for i in range(num_agents)}

    #include zealots
    for k in range(num_zealots):
        lambda_vals[k] = 1
        phi_vals[k] = 1
        initial_opinions[k] = 1
        internet_posters[k] = True

    # Create and run the model
    model = OpinionDynamicsModel(
        num_agents=num_agents,
        epsilon=epsilon,
        lambda_vals=lambda_vals,
        phi_vals=phi_vals,
        initial_opinions=initial_opinions,
        use_internet=use_internet,
        theta=theta,
        internet_posters=internet_posters
    )
    
    print(f"Running simulation for {steps} steps...")
    model.run(steps)
    
    # Generate and save the plot
    plot_opinion_dynamics(model, run_name, save_dir, num_zealots)
    
    print(f"Simulation completed. Plot saved to {save_dir}/{run_name}_dynamics.png")



if __name__ == "__main__":
    run_simulation()