from flask import Flask, jsonify
import numpy as np
import random

app = Flask(__name__)

class NodeReliability:
    VERY_RELIABLE = 1
    RELIABLE = 2
    MODERATELY_UNRELIABLE = 3
    DEGRADING = 4
    GARBAGE = 5

class NodeReputationSimulator:
    def __init__(self):
        self.audit_lambda = 0.99
        self.audit_weight = 1.0
        self.audit_dq = 0.96
        self.initial_alpha = 0.0
        self.initial_beta = 50.0
        self.audit_alpha = self.initial_alpha
        self.audit_beta = self.initial_beta
        self.history = []
        self.reliability = random.choice([
            NodeReliability.VERY_RELIABLE,
            NodeReliability.RELIABLE,
            NodeReliability.MODERATELY_UNRELIABLE,
            NodeReliability.DEGRADING,
            NodeReliability.GARBAGE
        ])

    def update_reputation_multiple(self, beta, alpha, lambda_val, weight, success):
        v = 1.0 if success else 0.0
        alpha = lambda_val * alpha + weight * (1 + v) / 2
        beta = lambda_val * beta + weight * (1 - v) / 2
        return beta, alpha

    def apply_audit_result(self, result_type):
        if result_type == "failure":
            self.audit_beta, self.audit_alpha = self.update_reputation_multiple(
                self.audit_beta,
                self.audit_alpha,
                self.audit_lambda,
                self.audit_weight,
                False,
            )
        else:
            self.audit_beta, self.audit_alpha = self.update_reputation_multiple(
                self.audit_beta,
                self.audit_alpha,
                self.audit_lambda,
                self.audit_weight,
                True,
            )
        audit_score = self.audit_alpha / (self.audit_alpha + self.audit_beta)
        self.history.append({
            "audit_score": audit_score,
            "audit_alpha": self.audit_alpha,
            "audit_beta": self.audit_beta,
            "disqualified": audit_score <= self.audit_dq,
            "reliability": self.reliability
        })

def run_simulation(num_nodes=192, epochs=100, audits_per_epoch=100):
    nodes = [NodeReputationSimulator() for _ in range(num_nodes)]
    for epoch in range(epochs):
        for _ in range(audits_per_epoch):
            for node in nodes:
                if node.reliability == NodeReliability.VERY_RELIABLE:
                    result_type = "success" if random.random() < 0.999999 else "failure"
                elif node.reliability == NodeReliability.RELIABLE:
                    result_type = "success" if random.random() < 0.99 else "failure"
                elif node.reliability == NodeReliability.MODERATELY_UNRELIABLE:
                    result_type = "success" if random.random() < 0.9 else "failure"
                elif node.reliability == NodeReliability.DEGRADING:
                    result_type = "success" if random.random() < 0.99 else "failure"
                else:
                    result_type = "success" if random.random() < 0.25 else "failure"
                node.apply_audit_result(result_type)
    return [[node.history for node in nodes]]

@app.route('/simulate')
def simulate():
    data = run_simulation()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)