#!/bin/bash
echo "🔧 Setting up AI-Driven QoS System..."

# Create project structure
mkdir -p {src,models,scripts,config,data/{incoming,tmp},qos-logs}

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install system packages
sudo apt-get update
sudo apt-get install -y tshark wireshark-common

# Set up traffic control
sudo ./scripts/qos_tc_setup.sh

# Set environment variables
echo "export QOS_SNS_TOPIC_ARN='$QOS_SNS_TOPIC_ARN'" >> ~/.bashrc
echo "export AWS_DEFAULT_REGION='us-east-1'" >> ~/.bashrc

echo "✅ Setup complete! Run './scripts/capture_everything.sh 15 5000' to test."