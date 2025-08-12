#!/bin/bash
# Generate traffic directly on the monitored interface

INTERFACE="ens5"  # Your monitored interface
echo "🚀 Generating traffic directly on interface: $INTERFACE"

# Method 1: Generate traffic using ping flood (will be captured)
generate_ping_traffic() {
    echo "📡 Starting ping flood traffic..."
    while true; do
        # Ping flood to generate lots of packets
        ping -f -c 100 8.8.8.8 >/dev/null 2>&1 &
        ping -f -c 100 1.1.1.1 >/dev/null 2>&1 &
        sleep 1
    done
}

# Method 2: Generate large file downloads (HTTP traffic)
generate_download_traffic() {
    echo "📥 Starting download traffic..."
    while true; do
        # Multiple large downloads simultaneously
        wget -q -O /dev/null "http://speedtest.ftp.otenet.gr/files/test100Mb.db" &
        wget -q -O /dev/null "http://speedtest.ftp.otenet.gr/files/test100Mb.db" &
        wget -q -O /dev/null "http://speedtest.ftp.otenet.gr/files/test100Mb.db" &
        sleep 10
        killall wget 2>/dev/null
    done
}

# Method 3: Generate continuous data streams
generate_data_streams() {
    echo "🌊 Starting data streams..."
    while true; do
        # Create multiple data streams
        curl -s "http://httpbin.org/drip?duration=30&numbytes=50000000" > /dev/null &
        curl -s "http://httpbin.org/drip?duration=30&numbytes=50000000" > /dev/null &
        curl -s "http://httpbin.org/drip?duration=30&numbytes=50000000" > /dev/null &
        sleep 15
    done
}

# Start all traffic generators
generate_ping_traffic &
PING_PID=$!

generate_download_traffic &
DOWNLOAD_PID=$!

generate_data_streams &
STREAM_PID=$!

echo "✅ Interface traffic generators started:"
echo "  Ping traffic PID: $PING_PID"  
echo "  Download traffic PID: $DOWNLOAD_PID"
echo "  Stream traffic PID: $STREAM_PID"

# Create stop script
cat > /tmp/stop_interface_traffic.sh << 'STOP'
#!/bin/bash
echo "🛑 Stopping interface traffic generators..."
kill PING_PID DOWNLOAD_PID STREAM_PID 2>/dev/null
killall ping wget curl 2>/dev/null
echo "✅ Interface traffic generation stopped"
STOP

sed -i "s/PING_PID/$PING_PID/g" /tmp/stop_interface_traffic.sh
sed -i "s/DOWNLOAD_PID/$DOWNLOAD_PID/g" /tmp/stop_interface_traffic.sh  
sed -i "s/STREAM_PID/$STREAM_PID/g" /tmp/stop_interface_traffic.sh
chmod +x /tmp/stop_interface_traffic.sh

echo "⏹️  To stop: /tmp/stop_interface_traffic.sh"
echo "📊 Monitor interface with: watch -n1 'cat /proc/net/dev | grep $INTERFACE'"