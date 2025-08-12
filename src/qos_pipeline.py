import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib
import logging
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError, BotoCoreError
import json

warnings.filterwarnings("ignore")

# ✅ Enhanced logging setup for CloudWatch
LOG_FILE_PATH = "/home/ubuntu/qos-project/qos-logs/qos_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# ======================= Config =======================
PROJECT_DIR = "/home/ubuntu/qos-project"
CLASSIFIER_PATH = f"{PROJECT_DIR}/models/traffic_classifier.pkl"
FEATURE_SCHEMA_PATH = f"{PROJECT_DIR}/models/feature_columns.pkl"
LABEL_ENCODER_PATH = f"{PROJECT_DIR}/models/label_encoder.pkl"
OUTPUT_CSV = f"{PROJECT_DIR}/qos_results.csv"

# CloudWatch and SNS configuration
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
SNS_TOPIC_ARN = os.environ.get("QOS_SNS_TOPIC_ARN", "")
CLOUDWATCH_NAMESPACE = "QoS/NetworkAnalysis"

# Performance tuning
CHUNK_SIZE = int(os.environ.get("PCAP_CHUNK_SIZE", 50000))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 2))
USE_TSHARK = os.environ.get("USE_TSHARK", "true").lower() == "true"

# ======================= AWS Clients =======================
try:
    cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)
    sns = boto3.client('sns', region_name=AWS_REGION)
    logging.info("✅ AWS clients initialized successfully")
except NoCredentialsError:
    logging.warning("⚠️ AWS credentials not found - CloudWatch integration disabled")
    cloudwatch = None
    sns = None
except Exception as e:
    logging.error(f"❌ AWS client initialization failed: {e}")
    cloudwatch = None
    sns = None

# ======================= Model Load =======================
try:
    clf = joblib.load(CLASSIFIER_PATH)
    feature_columns = joblib.load(FEATURE_SCHEMA_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    
    # Get actual traffic classes from the model
    TRAFFIC_CLASSES = list(le.classes_)
    logging.info(f"✅ Models loaded. Traffic classes: {TRAFFIC_CLASSES}")
    
    # Identify high-priority traffic types (adjust based on your actual classes)
    HIGH_PRIORITY_CLASSES = []
    for cls in TRAFFIC_CLASSES:
        cls_lower = cls.lower()
        if any(keyword in cls_lower for keyword in ['streaming', 'video', 'audio', 'voip', 'real', 'interactive']):
            HIGH_PRIORITY_CLASSES.append(cls)
    
    if not HIGH_PRIORITY_CLASSES and 'Streaming' in TRAFFIC_CLASSES:
        HIGH_PRIORITY_CLASSES = ['Streaming']
    
    logging.info(f"High priority classes identified: {HIGH_PRIORITY_CLASSES}")
    
except Exception as e:
    logging.error(f"❌ Failed to load models: {e}")
    sys.exit(1)

# ======================= CloudWatch Functions =======================
def send_custom_metric(metric_name, value, unit="Count", dimensions=None):
    """Send custom metric to CloudWatch."""
    if not cloudwatch:
        return False
        
    try:
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = dimensions
            
        cloudwatch.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=[metric_data]
        )
        return True
    except Exception as e:
        logging.error(f"Failed to send CloudWatch metric {metric_name}: {e}")
        return False

def send_sns_alert(subject, message, severity="INFO"):
    """Send SNS alert notification."""
    if not sns or not SNS_TOPIC_ARN:
        return False
        
    try:
        # Add severity emoji
        emoji_map = {
            "HIGH": "🚨",
            "MODERATE": "⚠️", 
            "INFO": "ℹ️",
            "SUCCESS": "✅"
        }
        
        formatted_subject = f"{emoji_map.get(severity, '📊')} QoS Alert: {subject}"
        
        # Create detailed message
        detailed_message = f"""
QoS System Alert
================
Severity: {severity}
Time: {datetime.now().isoformat()}
Host: {os.uname().nodename}

{message}

---
QoS Monitoring System
"""
        
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=formatted_subject,
            Message=detailed_message
        )
        logging.info(f"📧 SNS alert sent: {subject}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send SNS alert: {e}")
        return False

# ======================= Enhanced Feature Extraction =======================
def extract_features_tshark_streaming(pcap_file, bin_size=1, max_packets=None):
    """Fast streaming feature extraction with CloudWatch metrics."""
    logging.info(f"Extracting features from {pcap_file} using tshark")
    
    file_size_mb = os.path.getsize(pcap_file) / (1024 * 1024)
    logging.info(f"PCAP file size: {file_size_mb:.1f} MB")
    
    # Send file size metric
    send_custom_metric("PcapFileSizeMB", file_size_mb, "None")
    
    cmd = [
        'tshark', '-r', pcap_file, '-T', 'fields',
        '-e', 'frame.time_epoch',
        '-e', 'frame.len',
        '-E', 'header=y', '-E', 'separator=,', '-E', 'quote=d', '-E', 'occurrence=f'
    ]
    
    if max_packets:
        cmd.extend(['-c', str(max_packets)])
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        packets_data = []
        start_time = None
        packet_count = 0
        
        for line_num, line in enumerate(proc.stdout):
            if line_num == 0:
                continue
                
            try:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                    
                timestamp = float(parts[0]) if parts[0] else 0
                length = int(parts[1]) if parts[1] else 0
                
                if timestamp == 0 or length == 0:
                    continue
                    
                if start_time is None:
                    start_time = timestamp
                    
                norm_timestamp = timestamp - start_time
                time_bin = int(norm_timestamp // bin_size)
                
                packets_data.append({
                    'time_bin': time_bin,
                    'length': length,
                    'timestamp': norm_timestamp
                })
                
                packet_count += 1
                
                if len(packets_data) >= CHUNK_SIZE:
                    yield process_packet_chunk(packets_data)
                    packets_data = []
                    
            except (ValueError, IndexError):
                continue
        
        if packets_data:
            yield process_packet_chunk(packets_data)
            
        proc.wait()
        
        # Send packet count metric
        send_custom_metric("PacketsProcessed", packet_count, "Count")
        logging.info(f"Processed {packet_count} packets from {pcap_file}")
        
    except Exception as e:
        send_custom_metric("ProcessingErrors", 1, "Count")
        logging.error(f"tshark failed: {e}")
        raise

def process_packet_chunk(packets_data):
    """Process a chunk of packets and return aggregated features."""
    if not packets_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(packets_data)
    
    grouped = df.groupby('time_bin').agg({
        'length': ['mean', 'var', 'sum', 'count'],
        'timestamp': lambda x: x.diff().mean()
    }).fillna(0)
    
    grouped.columns = [
        'forward_pl_mean', 'forward_pl_var', 'forward_bps_mean', 
        'forward_pps_mean', 'forward_piat_mean'
    ]
    
    return grouped.reset_index()

def extract_features_pyshark_fallback(pcap_file, bin_size=1):
    """Pyshark fallback with metrics."""
    logging.info(f"Using pyshark fallback for {pcap_file}")
    send_custom_metric("PysharkFallbackUsed", 1, "Count")
    
    try:
        import pyshark
    except ImportError:
        logging.error("❌ pyshark is not installed.")
        raise

    cap = pyshark.FileCapture(pcap_file, only_summaries=True)
    packets = []
    for pkt in cap:
        try:
            packets.append({
                "timestamp": float(pkt.time),
                "length": int(pkt.length),
            })
        except:
            continue
    cap.close()

    df = pd.DataFrame(packets)
    if df.empty:
        raise ValueError("No packets extracted from PCAP.")

    df['timestamp'] -= df['timestamp'].min()
    df['time_bin'] = (df['timestamp'] // bin_size).astype(int)

    grouped = df.groupby('time_bin').agg(
        forward_pl_mean=('length', 'mean'),
        forward_pl_var=('length', 'var'),
        forward_piat_mean=('timestamp', lambda x: x.diff().mean()),
        forward_pps_mean=('length', 'count'),
        forward_bps_mean=('length', 'sum')
    ).fillna(0).reset_index()

    return grouped

def extract_features_from_pcap(pcap_file, bin_size=1, max_packets=None):
    """Smart feature extraction with CloudWatch integration."""
    file_size_mb = os.path.getsize(pcap_file) / (1024 * 1024)
    
    if file_size_mb > 10 or USE_TSHARK:
        logging.info(f"Large file ({file_size_mb:.1f}MB) - using tshark")
        
        all_features = []
        for chunk_features in extract_features_tshark_streaming(pcap_file, bin_size, max_packets):
            if not chunk_features.empty:
                all_features.append(chunk_features)
        
        if not all_features:
            logging.warning("No features from tshark, trying pyshark fallback")
            return extract_features_pyshark_fallback(pcap_file, bin_size)
            
        combined = pd.concat(all_features, ignore_index=True)
        final_features = combined.groupby('time_bin').agg({
            'forward_pl_mean': 'mean',
            'forward_pl_var': 'mean', 
            'forward_bps_mean': 'sum',
            'forward_pps_mean': 'sum',
            'forward_piat_mean': 'mean'
        }).fillna(0).reset_index()
        
        return final_features
    else:
        return extract_features_pyshark_fallback(pcap_file, bin_size)

# ======================= Enhanced ML Processing =======================
def align_features(df, feature_columns):
    """Align features with expected schema."""
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns].fillna(0)

def classify_traffic(df):
    """Classify traffic types."""
    preds = clf.predict(df)
    return le.inverse_transform(preds)

def predict_congestion_simple(features_df):
    """Enhanced congestion prediction with metrics."""
    loads = features_df['forward_bps_mean'].values
    
    if len(loads) < 5:
        avg_load = float(np.mean(loads))
        send_custom_metric("InsufficientData", 1, "Count")
        return {
            "status": "insufficient_data",
            "predicted_congestion": avg_load,
            "severity": "Normal"
        }
    
    recent_avg = float(np.mean(loads[-10:]))
    overall_avg = float(np.mean(loads))
    trend = recent_avg - overall_avg
    
    # Enhanced thresholds with CloudWatch metrics
    if recent_avg > 80000000:  # 80 Mbps
        severity = "High"
        send_custom_metric("HighCongestionDetected", 1, "Count")
    elif recent_avg > 50000000:  # 50 Mbps
        severity = "Moderate"
        send_custom_metric("ModerateCongestionDetected", 1, "Count")
    else:
        severity = "Normal"
        send_custom_metric("NormalTrafficDetected", 1, "Count")
    
    # Send bandwidth metric
    send_custom_metric("CurrentBandwidthBps", recent_avg, "Bytes/Second")
    
    return {
        "status": "simple_prediction",
        "predicted_congestion": recent_avg,
        "severity": severity,
        "trend": trend
    }

def suggest_scaling(streaming_ratio, congestion_result, traffic_classes):
    """Enhanced scaling suggestions with alerts - using actual traffic classes."""
    congestion_level = congestion_result["predicted_congestion"]
    severity = congestion_result.get("severity", "Normal")
    
    # Calculate high-priority traffic ratio using actual classes
    high_priority_ratio = 0
    if len(traffic_classes) > 0:
        for hp_class in HIGH_PRIORITY_CLASSES:
            high_priority_ratio += (traffic_classes == hp_class).mean()
    
    # Send traffic composition metrics
    send_custom_metric("StreamingRatio", streaming_ratio, "Percent")
    send_custom_metric("HighPriorityRatio", high_priority_ratio, "Percent")
    
    if severity == "High":
        if high_priority_ratio > 0.3:  # If >30% high-priority traffic
            suggestion = f"Scale UP VMs (+3) - High priority traffic ({high_priority_ratio:.1%})"
        else:
            suggestion = "Scale UP VMs (+2) - High congestion"
        
        send_sns_alert(
            "High Congestion Detected",
            f"Congestion level: {congestion_level:.0f} bps\nHigh priority traffic: {high_priority_ratio:.1%}\nRecommendation: {suggestion}",
            "HIGH"
        )
        send_custom_metric("ScaleUpRecommendations", 1, "Count")
        
    elif severity == "Moderate":
        suggestion = "Scale UP VMs (+1) - Moderate congestion"
        send_sns_alert(
            "Moderate Congestion Alert",
            f"Congestion level: {congestion_level:.0f} bps\nRecommendation: {suggestion}",
            "MODERATE"
        )
        send_custom_metric("ScaleUpRecommendations", 1, "Count")
        
    elif streaming_ratio < 0.2 and congestion_level < 20000000:
        suggestion = "Scale DOWN VMs (-1) - Low traffic"
        send_custom_metric("ScaleDownRecommendations", 1, "Count")
        
    else:
        suggestion = "Maintain current allocation"
        send_custom_metric("MaintainRecommendations", 1, "Count")
    
    return suggestion

def check_ood(df):
    """Enhanced anomaly detection with metrics."""
    if len(df) < 5:
        return pd.DataFrame()
        
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(df)
    anomalies = df[preds == -1]
    
    if len(anomalies) > 0:
        send_custom_metric("AnomaliesDetected", len(anomalies), "Count")
        send_sns_alert(
            "Traffic Anomalies Detected",
            f"Detected {len(anomalies)} anomalous traffic patterns. Investigation recommended.",
            "MODERATE"
        )
    
    return anomalies

# ======================= Enhanced Main Pipeline =======================
def run_pipeline(pcap_file, max_packets=None):
    """Enhanced QoS pipeline with full CloudWatch integration."""
    start_time = datetime.now()
    logging.info(f"Starting enhanced QoS pipeline for {pcap_file}")
    
    try:
        # Extract features
        features = extract_features_from_pcap(pcap_file, bin_size=1, max_packets=max_packets)
        
        if features.empty:
            send_custom_metric("EmptyFeatures", 1, "Count")
            raise ValueError("No features extracted from PCAP")
        
        logging.info(f"Extracted {len(features)} time bins")
        send_custom_metric("TimeBinsExtracted", len(features), "Count")
        
        # ML processing
        ml_features = features.drop(columns=['time_bin'])
        aligned = align_features(ml_features, feature_columns)
        
        # Classify traffic
        preds = classify_traffic(aligned)
        features['traffic_class'] = preds
        
        # Calculate ratios using actual classes
        streaming_ratio = (features['traffic_class'] == "Streaming").mean() if "Streaming" in TRAFFIC_CLASSES else 0
        
        # Calculate high-priority ratio
        high_priority_ratio = 0
        for hp_class in HIGH_PRIORITY_CLASSES:
            if hp_class in features['traffic_class'].values:
                high_priority_ratio += (features['traffic_class'] == hp_class).mean()
        
        # Predict congestion
        congestion_result = predict_congestion_simple(features)
        
        # Anomaly detection
        anomalies = check_ood(aligned)
        
        # Scaling suggestion
        scaling = suggest_scaling(streaming_ratio, congestion_result, features['traffic_class'])
        
        # Processing time metric
        processing_time = (datetime.now() - start_time).total_seconds()
        send_custom_metric("ProcessingTimeSeconds", processing_time, "Seconds")
        
        # Results
        results = pd.DataFrame({
            "timestamp": [datetime.now().isoformat()],
            "pcap_file": [pcap_file],
            "processing_time_seconds": [processing_time],
            "total_time_bins": [len(features)],
            "streaming_ratio": [streaming_ratio],
            "high_priority_ratio": [high_priority_ratio],
            "congestion_level": [congestion_result["predicted_congestion"]],
            "congestion_severity": [congestion_result["severity"]],
            "scaling_suggestion": [scaling],
            "ood_count": [len(anomalies)],
            "detected_classes": ["|".join(features['traffic_class'].unique())]
        })
        
        # Save results
        if os.path.exists(OUTPUT_CSV):
            results.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        else:
            results.to_csv(OUTPUT_CSV, index=False)

        # Enhanced logging with CloudWatch context
        severity = congestion_result["severity"]
        if severity == "High":
            logging.error(f"🚨 HIGH CONGESTION: {congestion_result['predicted_congestion']:.0f} bps")
        elif severity == "Moderate":
            logging.warning(f"⚠️ MODERATE CONGESTION: {congestion_result['predicted_congestion']:.0f} bps")
        else:
            logging.info(f"✅ Normal traffic: {congestion_result['predicted_congestion']:.0f} bps")

        # Log detected traffic classes
        logging.info(f"Detected traffic classes: {list(features['traffic_class'].unique())}")
        if HIGH_PRIORITY_CLASSES:
            logging.info(f"High priority ratio: {high_priority_ratio:.1%} (classes: {HIGH_PRIORITY_CLASSES})")

        # Send success metrics
        send_custom_metric("SuccessfulRuns", 1, "Count")
        logging.info(f"✅ Enhanced pipeline completed in {processing_time:.2f}s")
        print(results.to_string(index=False))
        
        return results
        
    except Exception as e:
        send_custom_metric("FailedRuns", 1, "Count")
        send_sns_alert(
            "QoS Pipeline Failure",
            f"Pipeline failed for {pcap_file}\nError: {str(e)}\nCheck logs for details.",
            "HIGH"
        )
        logging.error(f"❌ Pipeline failed: {e}")
        raise

# ======================= Entry Point =======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qos_pipeline.py <pcap_file> [max_packets]")
        print("Environment variables:")
        print("  QOS_SNS_TOPIC_ARN=arn:aws:sns:region:account:qos-alerts")
        print("  AWS_DEFAULT_REGION=us-east-1")
        print("  PCAP_CHUNK_SIZE=50000")
        print("  MAX_WORKERS=2") 
        print("  USE_TSHARK=true")
        print(f"Detected traffic classes: {TRAFFIC_CLASSES}")
        print(f"High priority classes: {HIGH_PRIORITY_CLASSES}")
        sys.exit(1)
        
    pcap_file = sys.argv[1]
    max_packets = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    run_pipeline(pcap_file, max_packets)