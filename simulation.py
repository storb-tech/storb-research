import numpy as np
from enum import Enum

# Set reproducible seed
seed = "storb"
seed = sum(ord(c) for c in seed) % (2**32 - 1)
np.random.seed(seed)

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
        self.audit_dq = 0.96
        self.initial_alpha = 500.0
        self.initial_beta = 1000.0

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

    def apply_audit_result(self, result_type, epoch):
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

        self.history.append({
            "epoch": epoch,
            "audit_score": audit_score,
            "audit_alpha": self.audit_alpha,
            "audit_beta": self.audit_beta,
            "disqualified": audit_score <= self.audit_dq,
            "reliability": self.reliability.name,
            "epochs_alive": self.epochs_alive,
            "result_type": result_type
        })

def run_interactive_simulation():
    """Run simulation and return data structure for interactive visualization"""
    num_nodes = 192
    epochs = 1000  # Reduced for better performance
    audits_per_epoch = 30
    
    # Initialize nodes with different reliability levels
    nodes = [NodeReputationSimulator(node_id=i) for i in range(num_nodes)]
    
    # Assign reliability levels to nodes
    for i, node in enumerate(nodes):
        node.epochs_alive = 0  # Start at 0
        if i < num_nodes // 4:
            node.reliability = NodeReliability.VERY_RELIABLE
        elif i < num_nodes // 2:
            node.reliability = NodeReliability.RELIABLE
        elif i < int(num_nodes * 0.8):
            node.reliability = NodeReliability.MODERATELY_UNRELIABLE
        elif i < int(num_nodes * 0.9):
            node.reliability = NodeReliability.DEGRADING
        else:
            node.reliability = NodeReliability.GARBAGE

    # Store all epoch data for animation
    epoch_data = []
    churn_events = []
    
    for epoch in range(1, epochs + 1):
        # Run multiple audits per epoch
        for audit_round in range(audits_per_epoch):
            # Apply audit results to ALL nodes each audit round
            for i, node in enumerate(nodes):
                # Simulate audit results based on node reliability
                if node.reliability == NodeReliability.VERY_RELIABLE:
                    result_type = "success" if np.random.random() < 0.999999 else "failure"
                elif node.reliability == NodeReliability.RELIABLE:
                    result_type = "success" if np.random.random() < 0.99 else "failure"
                elif node.reliability == NodeReliability.MODERATELY_UNRELIABLE:
                    result_type = "success" if np.random.random() < 0.9 else "failure"
                elif node.reliability == NodeReliability.DEGRADING:
                    if node.epochs_alive < epochs // 4:
                        result_type = "success" if np.random.random() < 0.99 else "failure"
                    else:
                        failure_rate = 0.05 + (node.epochs_alive / 10) * 0.45
                        result_type = "success" if np.random.random() > failure_rate else "failure"
                else:
                    # GARBAGE nodes fail most of the time
                    result_type = "success" if np.random.random() < 0.25 else "failure"

                node.apply_audit_result(result_type, epoch)

        # Increment epochs alive for each node
        for node in nodes:
            node.epochs_alive += 1

        # Node churn logic
        if epoch % 1 == 0:
            nodes_with_history = [
                (i, node) for i, node in enumerate(nodes)
                if len(node.history) > 0 and node.epochs_alive >= 10
            ]
            
            if nodes_with_history:
                # Replace the 3 lowest nodes with new nodes
                nodes_with_history.sort(key=lambda x: x[1].history[-1]["audit_score"])
                churn_indices = [i for i, _ in nodes_with_history[:3]]

                for index in churn_indices:
                    old_node = nodes[index]
                    if len(old_node.history) > 0:
                        # Record churn event
                        churn_events.append({
                            "epoch": epoch,
                            "node_id": index,
                            "score": old_node.history[-1]["audit_score"],
                            "reliability": old_node.reliability.name
                        })
                        
                        # Create new node
                        nodes[index] = NodeReputationSimulator(node_id=index)
                        
                        # Assign new reliability
                        rand = np.random.random()
                        if rand < 0.2:
                            nodes[index].reliability = NodeReliability.VERY_RELIABLE
                        elif rand < 0.5:
                            nodes[index].reliability = NodeReliability.RELIABLE
                        elif rand < 0.8:
                            nodes[index].reliability = NodeReliability.MODERATELY_UNRELIABLE
                        elif rand < 0.9:
                            nodes[index].reliability = NodeReliability.DEGRADING
                        else:
                            nodes[index].reliability = NodeReliability.GARBAGE
                            
                        nodes[index].epochs_alive = 0

        # Collect data for this epoch
        current_epoch_data = []
        for i, node in enumerate(nodes):
            if len(node.history) > 0:
                latest = node.history[-1]
                current_epoch_data.append({
                    "node_id": i,
                    "x": i,
                    "y": latest["audit_score"],
                    "reliability": latest["reliability"],
                    "epochs_alive": latest["epochs_alive"],
                    "audit_alpha": latest["audit_alpha"],
                    "audit_beta": latest["audit_beta"],
                    "disqualified": latest["disqualified"],
                    "total_audits": len(node.history)
                })
        
        epoch_data.append({
            "epoch": epoch,
            "nodes": current_epoch_data
        })

    return {
        "epoch_data": epoch_data,
        "churn_events": churn_events,
        "config": {
            "num_nodes": num_nodes,
            "epochs": epochs,
            "audits_per_epoch": audits_per_epoch,
            "audit_dq": 0.00
        }
    }
