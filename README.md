# Node Reputation Simulation Research

## Overview

This project simulates node reputation systems with network churn, starting from zero nodes and growing organically to a target network size. The system models how nodes build reputation through audit results and how poor-performing nodes are replaced over time.

## Key Features

- **Zero-start network growth**: Network begins empty and grows to target size
- **Realistic node behavior**: Different reliability types (Very Reliable, Reliable, Moderately Unreliable, Degrading, Garbage)
- **Intelligent churn**: Worst-performing nodes are replaced, with immunity periods for new nodes
- **Interactive visualization**: Web-based interface for real-time simulation viewing
- **Detailed analysis**: Jupyter notebook tools for in-depth study

## Quick Start

### Web Application
```bash
pip install -r requirements.txt
python3 app.py
# Visit http://127.0.0.1:5000
```

### Notebook Analysis
```bash
# Open node_rep_sim.py in Jupyter or run directly
python3 node_rep_sim.py
```

## Files

- `shared_simulator.py`: Core simulation engine (shared by web and notebook)
- `app.py`: Flask web application
- `node_rep_sim.py`: Jupyter notebook analysis and plotting
- `simulation.py`: Web app interface (uses shared simulator)
- `templates/index.html`: Web interface
- `REORGANIZATION_SUMMARY.md`: Detailed technical documentation

## Network Simulation Demo

https://github.com/user-attachments/assets/93e1f272-6889-4a7a-b966-b08db26fd3f8

## Architecture

The simulation uses a shared architecture where both the web application and notebook analysis use the same underlying simulation engine, ensuring consistency and eliminating code duplication.
