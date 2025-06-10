const data = []; // Array to hold node data over epochs
let currentEpoch = 0; // Current epoch index
let totalEpochs = 100; // Total number of epochs (to be set dynamically)

// Function to initialize the visualization
function initVisualization() {
    // Fetch the simulation data from the server
    fetch('/api/simulation-data')
        .then(response => response.json())
        .then(jsonData => {
            data.push(...jsonData);
            totalEpochs = data.length; // Set total epochs based on fetched data
            updateChart(); // Initial chart update
        });
}

// Function to update the chart based on the current epoch
function updateChart() {
    const epochData = data[currentEpoch];
    const scores = epochData.map(node => node.audit_score);
    const reliabilities = epochData.map(node => node.reliability);

    const trace = {
        x: Array.from({ length: scores.length }, (_, i) => i + 1), // Node indices
        y: scores,
        mode: 'markers',
        marker: {
            color: reliabilities,
            colorscale: 'Viridis',
            size: 10,
            colorbar: {
                title: 'Reliability',
                tickvals: [1, 2, 3, 4, 5],
                ticktext: ['Very Reliable', 'Reliable', 'Moderately Unreliable', 'Degrading', 'Garbage']
            }
        },
        type: 'scatter'
    };

    const layout = {
        title: `Node Scores at Epoch ${currentEpoch + 1}`,
        xaxis: {
            title: 'Node Index'
        },
        yaxis: {
            title: 'Audit Score',
            range: [0, 1]
        }
    };

    Plotly.newPlot('chart', [trace], layout);
}

// Function to step through epochs
function stepForward() {
    if (currentEpoch < totalEpochs - 1) {
        currentEpoch++;
        updateChart();
    }
}

function stepBackward() {
    if (currentEpoch > 0) {
        currentEpoch--;
        updateChart();
    }
}

// Function to display node details on click
function displayNodeDetails(event) {
    const pointIndex = event.points[0].pointIndex;
    const nodeData = data[currentEpoch][pointIndex];

    const details = `
        Node Index: ${pointIndex + 1}<br>
        Reliability: ${nodeData.reliability}<br>
        Audit Score: ${nodeData.audit_score.toFixed(4)}<br>
        Disqualified: ${nodeData.disqualified ? 'Yes' : 'No'}
    `;

    document.getElementById('node-details').innerHTML = details;
}

// Event listeners for stepping through epochs
document.getElementById('step-forward').addEventListener('click', stepForward);
document.getElementById('step-backward').addEventListener('click', stepBackward);

// Event listener for chart click
document.getElementById('chart').addEventListener('plotly_click', displayNodeDetails);

// Initialize the visualization on page load
window.onload = initVisualization;