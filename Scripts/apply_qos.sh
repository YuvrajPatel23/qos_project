#!/bin/bash
# Enhanced QoS application with severity-based control
INTERFACE="ens5"
CSV_FILE="/home/ubuntu/qos-project/qos_results.csv"
MAIN_LOG="/home/ubuntu/qos-project/qos-logs/qos_main.log"

# Check if results file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "$(date '+%F %T') [qos] ERROR: No results file found at $CSV_FILE" | tee -a "$MAIN_LOG"
    exit 1
fi

# Get the most recent result
LAST_LINE=$(tail -n 1 "$CSV_FILE")
RAW_SUGG=$(echo "$LAST_LINE" | awk -F, '{print $9}')   # scaling_suggestion column
SEVERITY=$(echo "$LAST_LINE" | awk -F, '{print $8}')   # congestion_severity column
CONGESTION_LEVEL=$(echo "$LAST_LINE" | awk -F, '{print $7}')  # congestion_level column

# Parse scaling decision
if echo "$RAW_SUGG" | grep -qi "Scale UP.*+3"; then
    DECISION="scale_up_high"
elif echo "$RAW_SUGG" | grep -qi "Scale UP.*+2"; then
    DECISION="scale_up_medium"  
elif echo "$RAW_SUGG" | grep -qi "Scale UP.*+1"; then
    DECISION="scale_up_low"
elif echo "$RAW_SUGG" | grep -qi "Scale DOWN"; then
    DECISION="scale_down"
else
    DECISION="normal"
fi

# Set traffic control rates based on decision
case "$DECISION" in
    "scale_up_high")
        HP="90mbit"; LP="10mbit"
        ;;
    "scale_up_medium")
        HP="80mbit"; LP="20mbit"
        ;;
    "scale_up_low")
        HP="70mbit"; LP="30mbit"
        ;;
    "scale_down")
        HP="50mbit"; LP="50mbit"
        ;;
    *)
        HP="60mbit"; LP="40mbit"
        ;;
esac

# Apply severity adjustments
if echo "$SEVERITY" | grep -qi "High"; then
    case "$DECISION" in
        "normal"|"scale_down") HP="75mbit"; LP="25mbit" ;;
    esac
fi

# Apply traffic control
echo "$(date '+%F %T') [qos] Applying: HP=$HP LP=$LP (Decision: $DECISION, Severity: $SEVERITY)" | tee -a "$MAIN_LOG"

sudo tc class change dev "$INTERFACE" parent 1:1 classid 1:10 htb rate "$HP" ceil "$HP" 2>/dev/null || \
    echo "$(date '+%F %T') [qos] WARNING: Failed to update high-priority class" | tee -a "$MAIN_LOG"

sudo tc class change dev "$INTERFACE" parent 1:1 classid 1:20 htb rate "$LP" ceil "$LP" 2>/dev/null || \
    echo "$(date '+%F %T') [qos] WARNING: Failed to update low-priority class" | tee -a "$MAIN_LOG"

MSG="$(date '+%F %T') [qos] Applied: $RAW_SUGG -> HP=$HP LP=$LP"
echo "$MSG" | tee -a "$MAIN_LOG"