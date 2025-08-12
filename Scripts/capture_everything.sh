
#!/bin/bash
PROJECT_DIR="/home/ubuntu/qos-project"
MAIN_LOG="$PROJECT_DIR/qos-logs/qos_main.log"
CAPTURE_DURATION="${1:-20}"
MAX_PACKETS="${2:-10000}"

mkdir -p "$PROJECT_DIR/qos-logs"

echo "$(date '+%F %T') [loop] ===== QoS Automation Loop START =====" | tee -a "$MAIN_LOG"
echo "$(date '+%F %T') [loop] Duration: ${CAPTURE_DURATION}s, Max packets: ${MAX_PACKETS}" | tee -a "$MAIN_LOG"

# Step 1: Capture traffic
echo "$(date '+%F %T') [loop] Step 1: Capturing traffic..." | tee -a "$MAIN_LOG"
PCAP_FILE=$("$PROJECT_DIR/scripts/capture_only.sh" "$CAPTURE_DURATION")
echo "$(date '+%F %T') [loop] Captured: $PCAP_FILE" | tee -a "$MAIN_LOG"

# Check if capture was successful
if [ ! -s "$PCAP_FILE" ]; then
    echo "$(date '+%F %T') [loop] ❌ Capture failed or empty file" | tee -a "$MAIN_LOG"
    exit 1
fi

# Step 2: Process with ML pipeline
echo "$(date '+%F %T') [loop] Step 2: Processing with ML pipeline..." | tee -a "$MAIN_LOG"
"$PROJECT_DIR/scripts/preprocess_pcap.sh" "$PCAP_FILE" "$MAX_PACKETS"
PROCESS_EXIT=$?

if [ $PROCESS_EXIT -ne 0 ]; then
    echo "$(date '+%F %T') [loop] ❌ Processing failed" | tee -a "$MAIN_LOG"
    exit 1
fi

# Step 3: Apply QoS
echo "$(date '+%F %T') [loop] Step 3: Applying QoS..." | tee -a "$MAIN_LOG"
"$PROJECT_DIR/scripts/apply_qos.sh"
QOS_EXIT=$?

if [ $QOS_EXIT -eq 0 ]; then
    echo "$(date '+%F %T') [loop] ✅ QoS automation completed successfully" | tee -a "$MAIN_LOG"
else
    echo "$(date '+%F %T') [loop] ⚠️ QoS application had issues" | tee -a "$MAIN_LOG"
fi

# Show current status
echo "$(date '+%F %T') [loop] Current results:" | tee -a "$MAIN_LOG"
tail -n 1 "$PROJECT_DIR/qos_results.csv" | cut -d, -f7-9 | tee -a "$MAIN_LOG"

echo "$(date '+%F %T') [loop] ===== QoS Automation Loop END =====" | tee -a "$MAIN_LOG"