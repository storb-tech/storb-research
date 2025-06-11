# Storb Research Monorepo

## Bayesian Scoring System Simulation
### Overview
This project simulates node reputation systems with network churn, starting from zero nodes and growing organically to a target network size. The system models how nodes build reputation through audit results and how poor-performing nodes are replaced over time. The audit process is adapted from [Storj's Bayesian scoring system](http://storj.io/whitepaper)
### Quick Start

#### Web Application
```bash
pip install -r requirements.txt
python3 app.py
# Visit http://127.0.0.1:5000
```
#### Demo

https://github.com/user-attachments/assets/c1ec186c-d395-460b-9f2e-91fa4ef57bf3

#### Notebook Analysis
```bash
# Open node_rep_sim.py in Jupyter or run directly
python3 node_rep_sim.py
```
#### Preview
![data](https://github.com/user-attachments/assets/4259aaf0-05e9-4d77-9f8b-9cd0e0133c5a)

### Files

- `shared_simulator.py`: Core simulation engine (shared by web and notebook)
- `app.py`: Flask web application
- `node_rep_sim.py`: Jupyter notebook analysis and plotting
- `simulation.py`: Web app interface (uses shared simulator)
- `templates/index.html`: Web interface

