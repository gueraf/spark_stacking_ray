#!/bin/bash
# Script to tear down the Ray cluster manually provisioned

echo "Stopping Ray Cluster..."

# 1. Discover Nodes (Simple approach: look at ray_info.txt or avahi)
# We need to find all nodes to stop containers on them.
# Let's use the same avahi-browse logic as start.py or just use ray_info.txt if it had all IPs?
# ray_info.txt only has head IP.
# So we must scan again.

echo "Scanning for nodes..."
NODES=$(avahi-browse -p -r -f -t _ssh._tcp | grep ";IPv4;" | grep -v "127.0.0.1" | awk -F';' '{print $8}' | sort | uniq)

if [ -z "$NODES" ]; then
    echo "No nodes found via Avahi. Trying local cleanup only."
    NODES="127.0.0.1"
else
    echo "Found nodes: $NODES"
fi

# 2. Loop and Stop
for IP in $NODES; do
    echo ">>> Stopping container on $IP..."
    ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$IP" "docker rm -f ray_cluster_node 2>/dev/null || true"
    
    # Also stop any 'ray start' process that might be running on host (unlikely with docker, but good measure)
    # ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$IP" "pkill -f 'ray start' || true"
done

# 3. Clean up local files
rm -f ray_info.txt
echo "Removed ray_info.txt"

# 4. Aggressive Local Cleanup (Just in case)
echo "Performing aggressive cleanup of local Ray processes and containers..."
ray stop 2>/dev/null || true
pkill -f "ray" || true
docker rm -f ray_cluster_node 2>/dev/null || true

echo "Cleanup complete."
