#!/bin/bash
INTERFACE="ens5"
HP="60mbit"
LP="40mbit"

echo "Setting up TC on $INTERFACE..."
sudo tc qdisc del dev "$INTERFACE" root 2>/dev/null || echo "No existing qdisc"
sudo tc qdisc add dev "$INTERFACE" root handle 1: htb default 20 r2q 100
sudo tc class add dev "$INTERFACE" parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev "$INTERFACE" parent 1:1 classid 1:10 htb rate "$HP" ceil "$HP"
sudo tc class add dev "$INTERFACE" parent 1:1 classid 1:20 htb rate "$LP" ceil "$LP"
sudo tc qdisc add dev "$INTERFACE" parent 1:10 handle 110: fq_codel
sudo tc qdisc add dev "$INTERFACE" parent 1:20 handle 120: fq_codel

echo "✅ TC setup complete on $INTERFACE (HP=$HP, LP=$LP)"
sudo tc class show dev "$INTERFACE"