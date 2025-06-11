#!/usr/bin/env python3
"""
Test script to verify the reorganized simulation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test basic import functionality"""
    try:
        from shared_simulator import NodeReputationSimulator, NetworkSimulator
        print("✓ Successfully imported core classes")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_individual_node():
    """Test individual node simulation"""
    try:
        from shared_simulator import NodeReputationSimulator
        
        node = NodeReputationSimulator(node_id=1)
        
        # Apply some audit results
        node.apply_audit_result("success", epoch=1)
        node.apply_audit_result("success", epoch=1)
        node.apply_audit_result("failure", epoch=1)
        
        if len(node.history) == 3:
            score = node.history[-1]["audit_score"]
            print(f"✓ Individual node test passed (final score: {score:.4f})")
            return True
        else:
            print(f"✗ Individual node test failed (history length: {len(node.history)})")
            return False
            
    except Exception as e:
        print(f"✗ Individual node test failed: {e}")
        return False

def test_network_simulation():
    """Test network-level simulation"""
    try:
        from shared_simulator import NetworkSimulator
        
        config = {
            "num_nodes_target": 5,
            "epochs": 10,
            "audits_per_epoch": 3,
            "nodes_per_epoch_add": 2,
            "nodes_per_epoch_churn": 1,
            "min_epochs_before_churn": 3
        }
        
        sim = NetworkSimulator(config)
        result = sim.run_simulation()
        
        if len(result["epoch_data"]) == 10:
            final_nodes = result["epoch_data"][-1]["total_nodes"]
            print(f"✓ Network simulation test passed (final nodes: {final_nodes})")
            return True
        else:
            print(f"✗ Network simulation test failed (epoch count: {len(result['epoch_data'])})")
            return False
            
    except Exception as e:
        print(f"✗ Network simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_interface():
    """Test web interface functions"""
    try:
        from shared_simulator import run_interactive_simulation
        
        config = {
            "num_nodes_target": 3,
            "epochs": 5,
            "audits_per_epoch": 2,
        }
        
        result = run_interactive_simulation(config)
        
        if "epoch_data" in result and "churn_events" in result and "config" in result:
            print("✓ Web interface test passed")
            return True
        else:
            print(f"✗ Web interface test failed (missing keys)")
            return False
            
    except Exception as e:
        print(f"✗ Web interface test failed: {e}")
        return False

def test_zero_start_behavior():
    """Test that simulation starts with zero nodes"""
    try:
        from shared_simulator import NetworkSimulator
        
        config = {
            "num_nodes_target": 10,
            "epochs": 5,
            "nodes_per_epoch_add": 3,
        }
        
        sim = NetworkSimulator(config)
        result = sim.run_simulation()
        
        # Check that we start with zero nodes
        if result["epoch_data"][0]["total_nodes"] == 0:
            print("✗ First epoch should have nodes added")
            return False
        
        # Check that nodes are added progressively
        node_counts = [epoch["total_nodes"] for epoch in result["epoch_data"]]
        if node_counts[0] > 0 and node_counts[-1] >= node_counts[0]:
            print(f"✓ Zero-start behavior test passed (growth: {node_counts[0]} -> {node_counts[-1]})")
            return True
        else:
            print(f"✗ Zero-start behavior test failed (counts: {node_counts})")
            return False
            
    except Exception as e:
        print(f"✗ Zero-start behavior test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running simulation tests...\n")
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Individual Node", test_individual_node),
        ("Network Simulation", test_network_simulation),
        ("Web Interface", test_web_interface),
        ("Zero-Start Behavior", test_zero_start_behavior),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"Running {name} test...")
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The reorganized simulation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
