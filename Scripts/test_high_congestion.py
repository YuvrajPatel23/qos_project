
cat > test_congestion_direct.py << 'EOF'
import boto3
import os
from datetime import datetime

SNS_TOPIC_ARN = os.environ.get('QOS_SNS_TOPIC_ARN', '')
AWS_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

print("🚨 Testing HIGH CONGESTION SNS Alert...")
print(f"Topic: {SNS_TOPIC_ARN}")

try:
    sns = boto3.client('sns', region_name=AWS_REGION)
    
    
    response = sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="🚨 QoS Alert: High Congestion Detected",
        Message=f"""
QoS System Alert
================
Severity: HIGH
Time: {datetime.now().isoformat()}
Host: ip-172-31-44-131

SIMULATED HIGH CONGESTION EVENT
Congestion level: 90,000,000 bps (90 Mbps)
High priority traffic: 100.0%
Recommendation: Scale UP VMs (+2) - High congestion

This is a test of the high congestion alerting system.
Your QoS monitoring is working correctly!

---
QoS Monitoring System
""")
    
    print("✅ HIGH CONGESTION alert sent!")
    print(f"Message ID: {response['MessageId']}")
    print("📧 Check your email for the alert!")
    
except Exception as e:
    print(f"❌ Failed: {e}")
EOF
