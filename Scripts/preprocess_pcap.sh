#!/bin/bash

PCAP_FILE="$1"
MAX_PKTS="${2:-40000}"
[ -z "$PCAP_FILE" ] && { echo "Usage: $0 <pcap_file> [MAX_PKTS]"; exit 1; }

PROJECT_DIR="/home/ubuntu/qos-project"
LOG_DIR="$PROJECT_DIR/qos-logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
MAIN_LOG="$LOG_DIR/qos_main.log"

mkdir -p "$LOG_DIR"

echo "$(date '+%F %T') [process] start pcap=$PCAP_FILE (limit: $MAX_PKTS packets)" | tee -a "$MAIN_LOG"

cd "$PROJECT_DIR"
source .venv/bin/activate

python src/qos_pipeline.py "$PCAP_FILE" "$MAX_PKTS" 2>&1 | tee -a "$PIPELINE_LOG" "$MAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%F %T') [process] ✅ SUCCESS - Results in qos_results.csv" | tee -a "$MAIN_LOG"
else
    echo "$(date '+%F %T') [process] ❌ FAILED with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
fi

echo "$(date '+%F %T') [process] done pcap=$PCAP_FILE" | tee -a "$MAIN_LOG"