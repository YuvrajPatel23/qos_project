# QoS Network Monitoring System - Complete Deployment Guide

> **Project Repository**: [https://github.com/YuvrajPatel23/qos_project](https://github.com/YuvrajPatel23/qos_project)

## Prerequisites

- Ubuntu/Debian-based Linux system
- Root/sudo access
- AWS account with appropriate permissions
- Network interface for traffic monitoring

## Phase 1: System Setup and Dependencies

### 1.1 Update System

```bash

sudo apt-get update && sudo apt-get upgrade -y
```

### 1.2 Install Core Dependencies

```bash
sudo apt-get install -y python3 python3-pip python3-venv git curl wget unzip


sudo apt-get install -y tshark wireshark-common tcpdump net-tools


sudo apt-get install -y build-essential libpcap-dev
```

### 1.3 Configure Network Monitoring Permissions

```bash

sudo usermod -a -G wireshark $USER
sudo chmod +x /usr/bin/dumpcap

# Note: You may need to log out and back in for group changes to take effect
```

## Phase 2: Project Structure Setup

### 2.1 Create Project Directory

```bash

mkdir -p /home/ubuntu/qos-project
cd /home/ubuntu/qos-project


mkdir -p {src,models,scripts,config,data/{incoming,tmp},qos-logs}
```

### 2.2 Download Project Files

```bash
# Clone your repository
git clone https://github.com/YuvrajPatel23/qos_project.git temp-repo
mv temp-repo/* .
rm -rf temp-repo

```

### 2.3 Organize Files

Ensure files are in correct locations:

- `qos_pipeline.py` → `src/`
- Training script → `src/`
- All `.sh` scripts → `scripts/`
- CloudWatch config → `config/`
- Dataset → `data/`

## Phase 3: Python Environment Setup

### 3.1 Create Virtual Environment

```bash
cd /home/ubuntu/qos-project
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install Python Dependencies

Create `requirements.txt` if not present:

```txt
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
scapy>=2.5.0
psutil>=5.9.0
boto3>=1.26.0
joblib>=1.3.0
pytz>=2023.3
```

Install packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Phase 4: Machine Learning Model Training

### 4.1 Prepare Dataset

```bash
ls -la sdn_dataset_2022.csv
```

### 4.2 Train Models

```bash
python src/train_models.py


ls -la models/
# Expected files: traffic_classifier.pkl, label_encoder.pkl, feature_columns.pkl
```

## Phase 5: Network Configuration

### 5.1 Identify Network Interface

```bash
ip route | grep default
# Common names: ens5 (AWS), eth0 (general), enp0s3 (VirtualBox)

# Note the interface name - you'll need to update scripts if it's not 'ens5'
```

### 5.2 Setup Traffic Control

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Setup QoS traffic control
sudo ./scripts/qos_tc_setup.sh

# Verify TC setup (replace ens5 with your interface)
sudo tc class show dev ens5
```

## Phase 6: AWS Integration

### 6.1 Install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws --version
```

### 6.2 Configure AWS Credentials

```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)
```

### 6.3 Setup SNS for Alerts

````bash
aws sns create-topic --name qos-alerts


```bash
echo "export QOS_SNS_TOPIC_ARN='arn:aws:sns:us-east-1:YOUR-ACCOUNT:qos-alerts'" >> ~/.bashrc
echo "export AWS_DEFAULT_REGION='us-east-1'" >> ~/.bashrc
source ~/.bashrc

aws sns subscribe --topic-arn $QOS_SNS_TOPIC_ARN --protocol email --notification-endpoint your-email@example.com

# Check your email and confirm the subscription
````

## Phase 7: CloudWatch Setup

### 7.1 Install CloudWatch Agent

```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
```

### 7.2 Configure CloudWatch Agent

```bash
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -c file:config/amazon-cloudwatch-agent.json -s
```

## Phase 8: System Testing

### 8.1 Activate Environment

```bash
cd /home/ubuntu/qos-project
source .venv/bin/activate
```

### 8.2 Basic Tests

```bash
sudo ./scripts/capture_only.sh 30

./scripts/capture_everything.sh 30 5000
```

### 8.3 Verify Results

```bash
# Check results
cat qos_results.csv
tail -f qos-logs/qos_main.log


```

## Phase 9: Production Deployment(for it to run continously)

### 9.1 Create Systemd Service

Create `/etc/systemd/system/qos-monitor.service`:

```ini
[Unit]
Description=QoS Network Monitor
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/qos-project
Environment=PATH=/home/ubuntu/qos-project/.venv/bin
ExecStart=/home/ubuntu/qos-project/.venv/bin/python src/qos_pipeline.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable qos-monitor
sudo systemctl start qos-monitor
```

### 9.2 Setup Monitoring

```bash
sudo systemctl status qos-monitor

# View logs
#you can check the cloudwatch in the app for logs and results.
```
