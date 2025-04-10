import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict


class DynamicOpinionDynamics:
    def __init__(self, num_agents: int, epsilon: float, lambda_vals: Dict[int, float], phi_vals: Dict[int, float], initial_opinions: Dict[int, float]):
        """
        Initialize the opinion dynamics model with a dynamic network.

        Parameters:
        - num_agents: int
            Number of agents in the network.
        - epsilon: float
            Confidence threshold for connecting nodes based on opinion similarity.
        - lambda_vals: Dict[int, float]
            Stubbornness values for each agent (0 <= lambda <= 1).
        - phi_vals: Dict[int, float]
            Conformity resistance values for each agent (0 <= phi <= 1).
        - initial_opinions: Dict[int, float]
            Initial private opinions for each agent.
        """
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.lambda_vals = lambda_vals
        self.phi_vals = phi_vals
        self.private_opinions = initial_opinions.copy()
        self.expressed_opinions = initial_opinions.copy()
        self.network = nx.Graph()  # Dynamic network graph
        self.history = []  # To store opinions over time

        # Initialize the network with all nodes
        self.network.add_nodes_from(range(num_agents))
        self._update_network()

    def _update_network(self):
        """Update the network structure based on expressed opinions and bounded confidence."""
        self.network.clear()  # Remove all edges
        self.network.add_nodes_from(range(self.num_agents))

        # Add edges based on bounded confidence
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if abs(self.expressed_opinions[i] - self.expressed_opinions[j]) <= self.epsilon:
                    self.network.add_edge(i, j)

    def _update_private_opinion(self, agent: int):
        """Update private opinion of a given agent."""
        neighbors = list(self.network.neighbors(agent))
        if neighbors:  # Ensure the agent has neighbors
            neighbor_expressed_opinions = [self.expressed_opinions[neighbor] for neighbor in neighbors]
            avg_opinion = np.mean(neighbor_expressed_opinions)
        else:
            avg_opinion = self.private_opinions[agent]  # No neighbors, keep current opinion

        self.private_opinions[agent] = (
            self.lambda_vals[agent] * self.private_opinions[agent] +
            (1 - self.lambda_vals[agent]) * avg_opinion
        )

    def _update_expressed_opinion(self, agent: int):
        """Update expressed opinion of a given agent."""
        neighbors = list(self.network.neighbors(agent))
        if neighbors:
            neighbor_expressed_opinions = [self.expressed_opinions[neighbor] for neighbor in neighbors]
            avg_expressed_opinion = np.mean(neighbor_expressed_opinions)
        else:
            avg_expressed_opinion = self.expressed_opinions[agent]

        self.expressed_opinions[agent] = (
            self.phi_vals[agent] * self.private_opinions[agent] +
            (1 - self.phi_vals[agent]) * avg_expressed_opinion
        )

    def step(self):
        """Perform one time step of the simulation."""
        # Update network structure based on current expressed opinions
        self._update_network()

        # Update private opinions
        for agent in range(self.num_agents):
            self._update_private_opinion(agent)

        # Update expressed opinions
        for agent in range(self.num_agents):
            self._update_expressed_opinion(agent)

        # Save current state to history
        self.history.append({
            "private_opinions": self.private_opinions.copy(),
            "expressed_opinions": self.expressed_opinions.copy(),
        })

    def run(self, steps: int):
        """
        Run the simulation for a given number of steps.

        Parameters:
        - steps: int
            The number of time steps to simulate.
        """
        for _ in range(steps):
            self.step()

    def get_history(self) -> List[Dict[str, Dict[int, float]]]:
        """
        Get the history of opinions over time.

        Returns:
        - history: List[Dict[str, Dict[int, float]]]
            A list of dictionaries containing private and expressed opinions at each time step.
        """
        return self.history
    
def plot_dynamic_opinion_evolution(history: List[Dict[str, Dict[int, float]]]):
    """
    Plot the evolution of private and expressed opinions for all agents over time.

    Parameters:
    - history: List[Dict[str, Dict[int, float]]]
        A list of dictionaries containing private and expressed opinions at each time step.
    """
    num_steps = len(history)
    num_agents = len(history[0]["private_opinions"])
    time = range(num_steps)

    # Prepare data
    private_opinions = {agent: [] for agent in range(num_agents)}
    expressed_opinions = {agent: [] for agent in range(num_agents)}

    for step in history:
        for agent in range(num_agents):
            private_opinions[agent].append(step["private_opinions"][agent])
            expressed_opinions[agent].append(step["expressed_opinions"][agent])

    # Plot private opinions
    plt.figure(figsize=(12, 6))
    for agent in range(num_agents):
        plt.plot(time, private_opinions[agent], color="blue", linestyle="dashed", alpha=0.7)

    # Plot expressed opinions
    for agent in range(num_agents):
        plt.plot(time, expressed_opinions[agent], color="red", linestyle="solid", alpha=0.7)

    # Labels and formatting
    plt.xlabel("Time Step")
    plt.ylabel("Opinion Value")
    plt.title("Evolution of Private and Expressed Opinions Over Time")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)  # Neutral opinion line
    plt.plot([], [], color="blue", linestyle="dashed", label="Private Opinions (Agents)")
    plt.plot([], [], color="red", linestyle="solid", label="Expressed Opinions (Agents)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



# Example Usage
if __name__ == "__main__":
    # Number of agents and confidence threshold
    num_agents = 20
    epsilon = 0.4

    # Initialize parameters
    lambda_vals = {i: 0.9 for i in range(num_agents)}
    phi_vals = {i: 0.1 for i in range(num_agents)}
    initial_opinions = {i: np.random.uniform(0, 1) for i in range(num_agents)}

    for i in range(5):
        lambda_vals[i] = 1
        phi_vals[i] = 0.7
        initial_opinions[i] = 1

    # Run the simulation
    model = DynamicOpinionDynamics(
        num_agents=num_agents,
        epsilon=epsilon,
        lambda_vals=lambda_vals,
        phi_vals=phi_vals,
        initial_opinions=initial_opinions
    )
    model.run(steps=20)

    # Print final opinions
    print("Final Private Opinions:", model.private_opinions)
    print("Final Expressed Opinions:", model.expressed_opinions)
    plot_dynamic_opinion_evolution(model.get_history())
