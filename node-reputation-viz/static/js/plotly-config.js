const plotlyConfig = {
    layout: {
        title: 'Node Reputation Scores Over Time',
        xaxis: {
            title: 'Epochs',
            showgrid: true,
            zeroline: false,
        },
        yaxis: {
            title: 'Reputation Score',
            range: [0, 1],
            showline: true,
        },
        hovermode: 'closest',
        plot_bgcolor: '#1e1e1e',
        paper_bgcolor: '#1e1e1e',
        font: {
            color: '#ffffff',
        },
    },
    config: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'resetScale2d'],
    },
};

export default plotlyConfig;