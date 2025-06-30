"""
Shared Node Reputation Simulator

This module contains the core simulation logic that can be used by both
the web application and the notebook analysis scripts.
"""

import numpy as np
from enum import Enum


class NodeReliability(Enum):
    VERY_RELIABLE = 1
    RELIABLE = 2
    MODERATELY_UNRELIABLE = 3
    DEGRADING = 4
    GARBAGE = 5


class NodeReputationSimulator:
    def __init__(self, node_id=None):
        # Configuration from satellite/reputation/config.go
        self.audit_lambda = 0.99
        self.audit_weight = 1.0
        self.initial_alpha = 10.0
        self.initial_beta = 20.0

        # Initialize reputation
        self.audit_alpha = self.initial_alpha
        self.audit_beta = self.initial_beta

        self.history = []
        self.node_id = node_id
        self.epochs_alive = 0
        self.reliability = NodeReliability.RELIABLE

    def update_reputation_multiple(self, beta, alpha, lambda_val, weight, success):
        """Implementation of UpdateReputationMultiple from reputation package"""
        v = 1.0 if success else 0.0
        alpha = lambda_val * alpha + weight * (1 + v) / 2
        beta = lambda_val * beta + weight * (1 - v) / 2
        return beta, alpha

    def apply_audit_result(self, result_type, epoch=None):
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

        history_entry = {
            "audit_score": audit_score,
            "audit_alpha": self.audit_alpha,
            "audit_beta": self.audit_beta,
            "reliability": self.reliability.name,
            "epochs_alive": self.epochs_alive,
            "result_type": result_type,
        }

        if epoch is not None:
            history_entry["epoch"] = epoch

        self.history.append(history_entry)

    def get_audit_result_for_reliability(self, epochs_total=None):
        """Generate audit result based on node reliability"""
        if self.reliability == NodeReliability.VERY_RELIABLE:
            return "success" if np.random.random() < 0.999999 else "failure"
        elif self.reliability == NodeReliability.RELIABLE:
            return "success" if np.random.random() < 0.99 else "failure"
        elif self.reliability == NodeReliability.MODERATELY_UNRELIABLE:
            return "success" if np.random.random() < 0.9 else "failure"
        elif self.reliability == NodeReliability.DEGRADING:
            if epochs_total is None or self.epochs_alive < epochs_total // 8:
                return "success" if np.random.random() < 0.99 else "failure"
            else:
                failure_rate = 0.05 + (self.epochs_alive / 10) * 0.45
                return "success" if np.random.random() > failure_rate else "failure"
        else:
            # GARBAGE nodes fail most of the time
            return "success" if np.random.random() < 0.25 else "failure"


class NetworkSimulator:
    def __init__(self, config=None):
        """Initialize network simulator with configuration"""
        if config is None:
            config = {}

        self.config = config
        self.num_nodes_target = config.get("num_nodes_target", 192)
        self.epochs = config.get("epochs", 1000)
        self.nodes_per_epoch_add = config.get("nodes_per_epoch_add", 3)
        self.nodes_per_epoch_churn = config.get("nodes_per_epoch_churn", 3)
        self.min_epochs_before_churn = config.get("min_epochs_before_churn", 10)

        self.num_nodes_per_piece_upload = config.get("num_nodes_per_piece_upload", 10)
        self.num_piece_per_upload = config.get("num_piece_per_upload", 4)
        self.num_pieces_download_per_audit = config.get(
            "num_pieces_download_per_audit", 5
        )
        self.num_nodes_per_piece_download = config.get(
            "num_nodes_per_piece_download", 4
        )

        # Start with empty network
        self.nodes = []
        self.epoch_data = []
        self.churn_events = []
        self.next_node_id = 0

    def create_new_node(self, node_id=None):
        """Create a new node with random reliability"""
        if node_id is None:
            node_id = self.next_node_id
            self.next_node_id += 1

        node = NodeReputationSimulator(node_id=node_id)

        # Assign reliability based on distribution
        rand = np.random.random()
        if rand < 0.2:
            node.reliability = NodeReliability.VERY_RELIABLE
        elif rand < 0.4:
            node.reliability = NodeReliability.RELIABLE
        elif rand < 0.6:
            node.reliability = NodeReliability.MODERATELY_UNRELIABLE
        elif rand < 0.8:
            node.reliability = NodeReliability.DEGRADING
        else:
            node.reliability = NodeReliability.GARBAGE

        # if rand < 0.2:
        #     node.reliability = NodeReliability.VERY_RELIABLE
        # elif rand < 0.5:
        #     node.reliability = NodeReliability.RELIABLE
        # # elif rand < 0.8:
        # #     node.reliability = NodeReliability.MODERATELY_UNRELIABLE
        # elif rand < 0.9:
        #     node.reliability = NodeReliability.DEGRADING
        # else:
        #     node.reliability = NodeReliability.GARBAGE

        # node.epochs_alive = 0
        return node

    def add_nodes_to_network(self, epoch):
        """Add new nodes to the network"""
        # Add nodes until we reach target, but don't exceed nodes_per_epoch_add per epoch
        nodes_to_add = min(
            self.nodes_per_epoch_add, max(0, self.num_nodes_target - len(self.nodes))
        )

        for _ in range(nodes_to_add):
            new_node = self.create_new_node()
            self.nodes.append(new_node)

    def churn_nodes(self, epoch):
        """Remove lowest performing nodes and replace them"""
        # Only churn if we have nodes that are eligible (lived long enough)
        eligible_nodes = [
            (i, node)
            for i, node in enumerate(self.nodes)
            if len(node.history) > 0
            and node.epochs_alive >= self.min_epochs_before_churn
        ]

        if len(eligible_nodes) >= self.nodes_per_epoch_churn:
            # Sort by audit score (lowest first)
            eligible_nodes.sort(key=lambda x: x[1].history[-1]["audit_score"])
            churn_indices = [i for i, _ in eligible_nodes[: self.nodes_per_epoch_churn]]

            for index in churn_indices:
                old_node = self.nodes[index]
                if len(old_node.history) > 0:
                    # Record churn event
                    self.churn_events.append(
                        {
                            "epoch": epoch,
                            "node_id": old_node.node_id,
                            "score": old_node.history[-1]["audit_score"],
                            "reliability": old_node.reliability.name,
                            "epochs_alive": old_node.epochs_alive,
                        }
                    )

                    # Replace with new node that inherits the old node's ID
                    self.nodes[index] = self.create_new_node(node_id=old_node.node_id)

    def run_epoch_audits(self, epoch):
        """Run all audits for a single epoch"""
        # Uploads
        for _ in range(self.num_piece_per_upload):
            to_audit = np.random.choice(
                self.nodes,
                size=min(self.num_nodes_per_piece_upload, len(self.nodes)),
                # if node.epochs_alive < self.min_epochs_before_churn, then it is more likely to be selected
                # p=p,
                replace=False,
            )
            for node in to_audit:
                result_type = node.get_audit_result_for_reliability(self.epochs)
                node.apply_audit_result(result_type, epoch)

        # Downloads
        for _ in range(self.num_pieces_download_per_audit):
            to_audit = np.random.choice(
                self.nodes,
                size=min(self.num_pieces_download_per_audit, len(self.nodes)),
                replace=False,
            )
            for node in to_audit:
                result_type = node.get_audit_result_for_reliability(self.epochs)
                node.apply_audit_result(result_type, epoch)

    def collect_epoch_data(self, epoch):
        """Collect data for visualization"""
        current_epoch_data = []
        for i, node in enumerate(self.nodes):
            if len(node.history) > 0:
                latest = node.history[-1]
                current_epoch_data.append(
                    {
                        "node_id": node.node_id,
                        "x": i,
                        "y": latest["audit_score"],
                        "reliability": latest["reliability"],
                        "epochs_alive": latest["epochs_alive"],
                        "audit_alpha": latest["audit_alpha"],
                        "audit_beta": latest["audit_beta"],
                        "total_audits": len(node.history),
                    }
                )

        self.epoch_data.append(
            {
                "epoch": epoch,
                "nodes": current_epoch_data,
                "total_nodes": len(self.nodes),
            }
        )

    def run_simulation(self):
        """Run the complete simulation"""
        for epoch in range(1, self.epochs + 1):
            # Add new nodes to network
            self.add_nodes_to_network(epoch)

            # Run audits for all nodes
            if self.nodes:  # Only if we have nodes
                self.run_epoch_audits(epoch)

                # Increment epochs alive for each node
                for node in self.nodes:
                    node.epochs_alive += 1

                # Churn nodes (replace worst performers)
                self.churn_nodes(epoch)

            # Collect data for this epoch
            self.collect_epoch_data(epoch)

        return {
            "epoch_data": self.epoch_data,
            "churn_events": self.churn_events,
            "config": {
                "num_nodes_target": self.num_nodes_target,
                "epochs": self.epochs,
                "nodes_per_epoch_add": self.nodes_per_epoch_add,
                "nodes_per_epoch_churn": self.nodes_per_epoch_churn,
            },
        }


def run_interactive_simulation(config=None):
    """Run simulation for interactive visualization (web app)"""
    if config is None:
        config = {
            "num_nodes_target": 192,
            "epochs": 1000,
            "nodes_per_epoch_add": 3,
            "nodes_per_epoch_churn": 3,
            "min_epochs_before_churn": 20,
            # "min_epochs_before_churn": 15,
            "num_nodes_per_piece_upload": 25,
            # "num_nodes_per_piece_upload": 10,
            "num_piece_per_upload": 4,
            "num_pieces_download_per_audit": 5,
            "num_nodes_per_piece_download": 4,
        }

    simulator = NetworkSimulator(config)
    return simulator.run_simulation()


def run_notebook_simulation(config=None):
    """Run simulation for notebook analysis with more detailed data"""
    if config is None:
        config = {
            "num_nodes_target": 192,
            "epochs": 1000,
            "nodes_per_epoch_add": 3,
            "nodes_per_epoch_churn": 3,
            "min_epochs_before_churn": 20,
            # "min_epochs_before_churn": 15,
            "num_nodes_per_piece_upload": 25,
            # "num_nodes_per_piece_upload": 10,
            "num_piece_per_upload": 4,
            "num_pieces_download_per_audit": 5,
            "num_nodes_per_piece_download": 4,
        }

    simulator = NetworkSimulator(config)
    result = simulator.run_simulation()

    # Add the simulator instance for detailed analysis
    result["simulator"] = simulator
    return result


# Legacy functions for backward compatibility
def scenario_reliable_then_failing():
    """Legacy function for scenario testing"""
    events = []
    # 500 successful audits (building good reputation)
    events.extend([{"type": "success"} for _ in range(10000)])
    # Sudden failure period - 50 failures
    events.extend([{"type": "failure"} for _ in range(100)])
    # Try to recover with successes
    events.extend([{"type": "success"} for _ in range(100)])
    return events


def scenario_gradual_degradation():
    """Legacy function for scenario testing"""
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
