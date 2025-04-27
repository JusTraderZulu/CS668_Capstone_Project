#!/bin/bash
# docker-run.sh - Simplify running DQL experiments in Docker

# Default values
AGENT_TYPE="dql"
DATA_FILE="test_small.csv"
EXPERIMENT_NAME="docker_experiment_$(date +%Y%m%d_%H%M%S)"
EPISODES=100
EXTRA_ARGS=""

# Function to display usage information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --agent_type TYPE       Agent type (dql or custom), default: $AGENT_TYPE"
    echo "  --data_file FILE        Data file in data/ directory, default: $DATA_FILE"
    echo "  --experiment_name NAME  Name for this experiment, default: auto-generated timestamp"
    echo "  --episodes NUM          Number of episodes to run, default: $EPISODES"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --agent_type custom --data_file full_dataset.csv --episodes 500"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent_type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Source Telegram credentials if file exists
if [ -f telegram_config.sh ]; then
    source telegram_config.sh
fi

echo "🚀 Starting DQL Trading experiment in Docker"
echo "Agent type: $AGENT_TYPE"
echo "Data file: $DATA_FILE"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Episodes: $EPISODES"

# Build the Docker image if needed
docker-compose build

# Run the experiment
docker-compose run --rm dql-trading \
    full-workflow \
    --agent_type $AGENT_TYPE \
    --data_file $DATA_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --episodes $EPISODES \
    $EXTRA_ARGS

echo "✅ Experiment completed"
echo "Results saved in: ./results/$EXPERIMENT_NAME/" 