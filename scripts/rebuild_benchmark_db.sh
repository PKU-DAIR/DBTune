#!/bin/bash
set -e

echo "=== Rebuilding Benchmark Database ==="

# Ensure KUBECONFIG is set (default to local file if exists)
if [ -f "./kubeconfig" ]; then
    export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
fi

# 1. Get Primary Pod
echo "Detecting Primary Pod..."
POD=$(kubectl get pods -n pg -l cnpg.io/instanceRole=primary -o name | head -n 1)
if [ -z "$POD" ]; then
    POD=$(kubectl get pods -n pg -l cnpg.io/cluster=pg-cluster -o name | head -n 1)
fi
if [ -z "$POD" ]; then
    echo "Error: Primary pod not found"
    exit 1
fi
echo "Found Primary Pod: $POD"

# The pod's PVC (where the actual PG data lives) is on a node-local disk
# (/storage, local-path storage class) on whichever worker currently hosts
# the pod — NOT on this orchestrator host. The pod can land on a different
# worker each time it's recreated, so resolve the node dynamically.
trim_pg_node() {
    local NODE_NAME NODE_IP
    NODE_NAME=$(kubectl get "$POD" -n pg -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    if [ -z "$NODE_NAME" ]; then
        echo "WARN: could not resolve node for $POD, skipping SSD trim."
        return
    fi
    NODE_IP=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    if [ -z "$NODE_IP" ]; then
        echo "WARN: could not resolve IP for node $NODE_NAME, skipping SSD trim."
        return
    fi
    echo "Trimming /storage on $NODE_NAME ($NODE_IP)..."
    ssh -o ConnectTimeout=10 -o BatchMode=yes ${FSTRIM_SSH_USER:?set FSTRIM_SSH_USER}@"$NODE_IP" \
        "sync && sudo fstrim -v /storage" 2>&1 || \
        echo "WARN: remote fstrim failed on $NODE_NAME, continuing anyway."
}

trim_pg_node

# 2. Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if kubectl exec -n pg $POD -- pg_isready -U postgres; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "Waiting for PostgreSQL... ($i/30)"
    sleep 2
done

# 3. Drop Database
echo "Dropping database 'tpcc' if exists..."
kubectl exec -n pg $POD -- psql -U postgres -c "DROP DATABASE IF EXISTS tpcc WITH (FORCE);"

# 3. Apply Environment
echo "Applying Benchmark ConfigMaps and Secrets..."
if [ -f "k8s/setup_benchmark_env.yaml" ]; then
    kubectl apply -f k8s/setup_benchmark_env.yaml
else
    echo "Warning: setup_benchmark_env.yaml not found, assuming env is already set."
fi

# 4. Trigger Build Job
echo "Triggering Build Job 'tprocc-buildschema' (80 Warehouses)..."
kubectl delete job tprocc-buildschema -n benchmark --ignore-not-found=true
# Esperar a que la eliminación propague antes de apply (evita race con job en Terminating)
kubectl wait --for=delete job/tprocc-buildschema -n benchmark --timeout=60s 2>/dev/null || true

# Patch the standard yaml on the fly to increase Warehouses to 80
# We use sed to replace the default 20 with 80
cat k8s/tprocc_rebuild.yaml | sed 's/value: "20"/value: "80"/' | kubectl apply -f -

# 5. Wait — polling loop (kubectl wait --for=condition=complete falla en K3s 1.33
#    cuando el job recién arrancado no tiene conditions todavía)
echo "Waiting for job completion (Timeout: 60m)..."
BUILD_DEADLINE=$(($(date +%s) + 3600))
while true; do
    if [ $(date +%s) -gt $BUILD_DEADLINE ]; then
        echo "ERROR: tprocc-buildschema timed out after 60m"
        kubectl describe job tprocc-buildschema -n benchmark 2>/dev/null | tail -20
        exit 1
    fi
    SUCCEEDED=$(kubectl get job tprocc-buildschema -n benchmark \
        -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
    FAILED=$(kubectl get job tprocc-buildschema -n benchmark \
        -o jsonpath='{.status.failed}' 2>/dev/null || echo "0")
    if [ "${SUCCEEDED:-0}" -ge 1 ]; then
        echo "buildschema completed successfully"
        break
    fi
    if [ "${FAILED:-0}" -ge 1 ]; then
        echo "ERROR: tprocc-buildschema job failed"
        kubectl describe job tprocc-buildschema -n benchmark 2>/dev/null | tail -20
        kubectl logs -n benchmark -l bench=tprocc-build --tail=30 2>/dev/null || true
        exit 1
    fi
    sleep 15
done

# 6. Fix tpcc password AFTER HammerDB build (HammerDB may overwrite it during build)
echo "Fixing tpcc role password for VUser connections..."
kubectl exec -n pg $POD -- psql -U postgres -c "ALTER ROLE tpcc PASSWORD 'tpcc';"

# 7. VACUUM ANALYZE + CHECKPOINT: HammerDB's bulk load leaves dead tuples and
# stale planner stats; settle the DB into steady state before baseline measurement.
echo "Running VACUUM ANALYZE on tpcc database..."
kubectl exec -n pg $POD -- psql -U postgres -d tpcc -c "VACUUM ANALYZE;" || echo "WARN: VACUUM ANALYZE failed (pod restart?), continuing."
echo "Running CHECKPOINT..."
kubectl exec -n pg $POD -- psql -U postgres -c "CHECKPOINT;" || echo "WARN: CHECKPOINT failed, continuing."

# 8. Trim SSD after the bulk write so the controller can GC before benchmarking
trim_pg_node

echo "=== Database Rebuild Complete ==="
