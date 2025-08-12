#!/bin/bash
echo "🎯 COMPLETE QoS DEMO SEQUENCE"
echo "============================="

echo "📊 Step 1: Baseline measurement (normal traffic)..."
./scripts/capture_everything.sh 15 5000

echo ""
echo "🔥 Step 2: Starting HIGH traffic generation..."
./scripts/interface_traffic_gen.sh &
sleep 10  # Let traffic build up

echo ""
echo "📈 Step 3: Measuring under HIGH load..."
./scripts/capture_everything.sh 20 8000

echo ""
echo "🛑 Step 4: Stopping traffic generation..."
/tmp/stop_interface_traffic.sh

echo ""
echo "📊 Step 5: Results comparison..."
echo "Last 3 QoS measurements:"
tail -3 qos_results.csv | awk -F, 'BEGIN{print "Time | Traffic | Congestion | Severity | Action"} {print $1 " | " $11 " | " $7 " bps | " $8 " | " $9}' | column -t -s "|"

echo ""
echo "✅ Demo complete! Check for SNS alerts if congestion was detected."