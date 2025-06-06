# %% [markdown]
# TODO:
# - update node simulations to make it so that each "timeepoch" is an epoch (360 blocks, with each block being 12 seconds)
# - multiple audits can take place during an epoch

# %%
import numpy as np
import matplotlib.pyplot as plt

# by default make matplotlib use dark mode
plt.style.use("dark_background")


class NodeReputationSimulator:
    def __init__(self):
        # Configuration from satellite/reputation/config.go
        self.audit_lambda = 0.997
        self.audit_weight = 1.0
        self.audit_dq = 0.96
        self.initial_alpha = 1000.0
        self.initial_beta = 0.0

        # Initialize reputation
        self.audit_alpha = self.initial_alpha
        self.audit_beta = self.initial_beta

        self.history = []

    def update_reputation_multiple(self, beta, alpha, lambda_val, weight, success):
        """Implementation of UpdateReputationMultiple from reputation package"""
        v = 1.0 if success else 0.0
        alpha = lambda_val * alpha + weight * (1 + v) / 2
        beta = lambda_val * beta + weight * (1 - v) / 2
        return beta, alpha

    def apply_audit_result(self, result_type):
        """Apply audit results: 'success', 'failure', 'offline'"""
        if result_type == "failure":
            self.audit_beta, self.audit_alpha = self.update_reputation_multiple(
                self.audit_beta,
                self.audit_alpha,
                self.audit_lambda,
                self.audit_weight,
                False,
            )
        else:
            # Success improves reputation
            self.audit_beta, self.audit_alpha = self.update_reputation_multiple(
                self.audit_beta,
                self.audit_alpha,
                self.audit_lambda,
                self.audit_weight,
                True,
            )

        # Record current state
        audit_score = self.audit_alpha / (self.audit_alpha + self.audit_beta)

        self.history.append(
            {
                "audit_score": audit_score,
                "audit_alpha": self.audit_alpha,
                "audit_beta": self.audit_beta,
                "disqualified": audit_score <= self.audit_dq,
            }
        )

    def simulate_scenario(self, scenario_name, events):
        """Simulate a sequence of audit events"""
        # print(f"\n=== {scenario_name} ===")
        self.__init__()  # Reset

        for event in events:
            self.apply_audit_result(event["type"])
            if len(self.history) % 50 == 0:  # Print every 50 audits
                last = self.history[-1]
                # print(f"Audit {len(self.history)}: Score={last['audit_score']:.4f}, "
                #       f"DQ={last['disqualified']}")

        return self.history


# Scenario 1: Reliable node that suddenly starts failing
def scenario_reliable_then_failing():
    events = []
    # 500 successful audits (building good reputation)
    events.extend([{"type": "success"} for _ in range(10000)])
    # Sudden failure period - 50 failures
    events.extend([{"type": "failure"} for _ in range(100)])
    # Try to recover with successes
    events.extend([{"type": "success"} for _ in range(100)])
    return events


# Scenario 3: Gradual degradation
def scenario_gradual_degradation():
    events = []
    # Start reliable
    events.extend([{"type": "success"} for _ in range(10000)])

    # Gradually increase failure rate
    for phase in range(10):
        failure_rate = 0.05 + (phase * 0.05)  # 5% to 50% failure rate
        for _ in range(50):
            if np.random.random() < failure_rate:
                events.append({"type": "failure"})
            else:
                events.append({"type": "success"})
    return events


# Run simulations
simulator = NodeReputationSimulator()

scenarios = [
    ("Reliable Node Then Failing", scenario_reliable_then_failing()),
    ("Gradual Degradation", scenario_gradual_degradation()),
]

plt.figure(figsize=(15, 10))

for i, (name, events) in enumerate(scenarios):
    history = simulator.simulate_scenario(name, events)

    plt.subplot(2, 2, i + 1)
    audit_scores = [h["audit_score"] for h in history]

    plt.plot(audit_scores, label="Audit Score", linewidth=2)
    # plt.axhline(y=0.96, color='red', linestyle='--', label='Audit DQ Threshold')

    plt.title(name)
    plt.xlabel("Audit Number")
    plt.ylabel("Reputation Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Print final statistics
# for name, events in scenarios:
#     history = simulator.simulate_scenario(name, events)
#     final = history[-1]
#     dq_point = next((i for i, h in enumerate(history) if h['disqualified']), None)

#     print(f"\n{name} Final Results:")
#     print(f"  Final Audit Score: {final['audit_score']:.4f}")
#     print(f"  Disqualified: {final['disqualified']}")
#     if dq_point:
#         print(f"  Disqualified at audit: {dq_point + 1}")

# %%

from enum import Enum


class NodeReliablity(Enum):
    VERY_RELIABLE = 1
    RELIABLE = 2
    MODERATELY_UNRELIABLE = 3
    DEGRADING = 4
    GARBAGE = 5


def run_simulation_with_node_churn():
    """Run a simulation with node churn, where nodes are replaced based on their reputation scores."""
    num_nodes = 192
    epochs = 10000
    interval = 500

    # Initialize nodes with different reliability levels
    nodes = [NodeReputationSimulator() for _ in range(num_nodes)]
    # Assign reliability levels to nodes
    for i, node in enumerate(nodes):
        if i < num_nodes // 4:
            node.reliability = NodeReliablity.VERY_RELIABLE
        elif i < num_nodes // 2:
            node.reliability = NodeReliablity.RELIABLE
        elif i < int(num_nodes * 0.8):
            node.reliability = NodeReliablity.MODERATELY_UNRELIABLE
        elif i < int(num_nodes * 0.9):
            node.reliability = NodeReliablity.DEGRADING
        else:
            node.reliability = NodeReliablity.GARBAGE

    # Calculate number of plots needed
    num_plots = epochs // interval + 1
    # Calculate grid dimensions
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create a single figure for all plots
    fig = plt.figure(figsize=(15, 5 * num_rows))

    # Track churn metrics
    churn_history = []

    for epoch in range(1, epochs + 1):
        # Apply audit results to ALL nodes each epoch
        for i, node in enumerate(nodes):
            # Simulate audit results based on node reliability
            if node.reliability == NodeReliablity.VERY_RELIABLE:
                result_type = "success" if np.random.random() < 0.999 else "failure"
            elif node.reliability == NodeReliablity.RELIABLE:
                result_type = "success" if np.random.random() < 0.99 else "failure"
            elif node.reliability == NodeReliablity.MODERATELY_UNRELIABLE:
                result_type = "success" if np.random.random() < 0.9 else "failure"
            elif node.reliability == NodeReliablity.DEGRADING:
                if epoch < epochs // 2:
                    result_type = "success" if np.random.random() < 0.99 else "failure"
                else:
                    failure_rate = 0.05 + (epoch / epochs) * 0.45
                    result_type = (
                        "success" if np.random.random() > failure_rate else "failure"
                    )
            else:
                # GARBAGE nodes fail most of the time
                result_type = "success" if np.random.random() < 0.5 else "failure"

            node.apply_audit_result(result_type)

        # Node churn logic: every 360 epochs, replace the lowest scoring node with a new one
        # Only apply churn after nodes have some history
        if epoch % 2 == 0:
            # Only consider nodes that have history
            nodes_with_history = [
                (i, node) for i, node in enumerate(nodes) if len(node.history) > 0
            ]
            if nodes_with_history:
                # replace the 3 lowerst nodes with new nodes
                nodes_with_history.sort(key=lambda x: x[1].history[-1]["audit_score"])
                churn_indices = [index for index, _ in nodes_with_history[:3]]
                churn_history.extend(churn_indices)

        # Plot every 'interval' epochs
        # Only plot after nodes have some history
        if epoch % interval == 0:
            plot_index = epoch // interval
            ax = fig.add_subplot(num_rows, num_cols, plot_index)

            # get the reliability levels for color mapping
            reliability_colors = [
                node.reliability.value for _, node in nodes_with_history
            ]

            # Only include nodes with history
            nodes_with_history = [node for node in nodes if len(node.history) > 0]
            if nodes_with_history:
                scores = [
                    node.history[-1]["audit_score"] for node in nodes_with_history
                ]
                if len(set(scores)) > 1:  # Avoid division by zero
                    scores = [
                        (score - min(scores)) / (max(scores) - min(scores))
                        for score in scores
                    ]
                    # sorted indices of the uids for color mapping
                    sorted_indices = np.argsort(scores)
                    scores = sorted(scores)
                    scatter = ax.scatter(
                        range(len(nodes_with_history)),
                        scores,
                        c=reliability_colors,
                        cmap="viridis",
                        s=10,
                    )
                else:
                    # All scores are the same
                    scores = [0.0] * len(nodes_with_history)
                    scatter = ax.scatter(
                        range(len(nodes_with_history)),
                        scores,
                        c=reliability_colors,
                        s=10,
                    )

                ax.set_title(f"epoch {epoch}")
                ax.set_xlim(0, len(nodes_with_history) - 1)
                ax.set_xlabel("nodes")
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
                ax.text(
                    0.5,
                    0.9,
                    f"Churn: {len(churn_history)} nodes replaced",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=10,
                    color="white",
                )
    plt.tight_layout()
    plt.show()

    # also plot the number of times a node was replaced (by their sorted index of score)
    churn_counts = np.zeros(num_nodes)
    for index in churn_history:
        churn_counts[index] += 1
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_nodes), churn_counts, color="orange")
    plt.title("Node Churn Counts")
    plt.xlabel("Node Index")
    plt.ylabel("Churn Count")
    plt.grid(True, alpha=0.3)


run_simulation_with_node_churn()

# %%
