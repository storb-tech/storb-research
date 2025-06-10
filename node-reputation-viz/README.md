# Node Reputation Visualization

This project provides an interactive web-based visualization of node reputation scores over time. Users can step through epochs to observe changes in node scores and inspect individual nodes for detailed information.

## Project Structure

```
node-reputation-viz
├── static
│   ├── css
│   │   └── style.css          # CSS styles for the web application
│   ├── js
│   │   ├── main.js            # Main JavaScript code for interactivity
│   │   └── plotly-config.js   # Configuration for Plotly charts
│   └── favicon.ico            # Favicon for the web application
├── templates
│   └── index.html             # Main HTML template for the web application
├── app.py                     # Entry point for the Flask server
├── simulation.py              # Simulation logic and data generation
├── requirements.txt           # Python dependencies for the project
└── README.md                  # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd node-reputation-viz
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the Flask server by running:
   ```
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.

## Usage

- Navigate to the web application in your browser.
- Use the interactive controls to step through epochs and observe the changes in node scores.
- Hover over points on the chart to inspect details such as node reliability and score at that specific time.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.