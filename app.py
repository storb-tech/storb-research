from flask import Flask, render_template, jsonify
from simulation import run_interactive_simulation

app = Flask(__name__)

# Global variable to store simulation data
simulation_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulation-data')
def get_simulation_data():
    global simulation_data
    if simulation_data is None:
        simulation_data = run_interactive_simulation()
    return jsonify(simulation_data)

@app.route('/api/reset-simulation')
def reset_simulation():
    global simulation_data
    simulation_data = None
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
