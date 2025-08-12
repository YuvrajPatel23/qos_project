#!/bin/bash
# Enhanced capture script with better logging
DURATION="${1:-20}"
INTERFACE="ens5"
PROJECT_DIR="/home/ubuntu/qos-project"
OUT_DIR="$PROJECT_DIR/data/incoming"
LOGDIR="$PROJECT_DIR/qos-logs"
MAIN_LOG="$LOGDIR/qos_main.log"

mkdir -p "$OUT_DIR" "$LOGDIR"

TS=$(date +"%Y%m%d_%H%M%S")
FILEPATH="$OUT_DIR/capture_${TS}.pcap"

SNAPLEN=96
FILTER='tcp or udp'

echo "$(date '+%F %T') [capture] start dur=${DURATION}s iface=${INTERFACE} -> $FILEPATH" >> "$MAIN_LOG"
sudo timeout "$DURATION" tcpdump -i "$INTERFACE" -s "$SNAPLEN" $FILTER -w "$FILEPATH" -U 2>/dev/null
echo "$(date '+%F %T') [capture] done file=$FILEPATH size=$(du -h $FILEPATH | cut -f1)" >> "$MAIN_LOG"

echo "$FILEPATH"