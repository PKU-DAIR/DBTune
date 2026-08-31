# Prometheus Setup for DML Metrics Collection

## Prerequisites
- Docker and Docker Compose installed
- MySQL running on the host (port 3306)

## Quick Start

### 1. Grant MySQL exporter access

Connect to MySQL and create an exporter user (or allow root TCP access):

```sql
-- Option A: Create a dedicated exporter user
CREATE USER 'exporter'@'%' IDENTIFIED BY 'exporterpass';
GRANT PROCESS, REPLICATION CLIENT, SELECT ON *.* TO 'exporter'@'%';
FLUSH PRIVILEGES;

-- Option B: If using root, ensure root can connect via TCP
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '';
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY '';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%';
FLUSH PRIVILEGES;
```

If using a dedicated user, update `.my.cnf`:
```ini
[client]
user=exporter
password=exporterpass
host=host.docker.internal
port=3306
```

### 2. Start the stack

```bash
cd scripts/prometheus
docker compose up -d
```

### 3. Verify

- Prometheus UI: http://localhost:9090
- Check targets: http://localhost:9090/targets (all 3 should be UP)
- Test a query: http://localhost:9090/graph?g0.expr=mysql_global_status_queries

### 4. Test the DML metrics collection

```python
from autotune.optimizer.dml_metrics import collect_metrics_from_prometheus
metrics = collect_metrics_from_prometheus(
    prometheus_url='http://localhost:9090',
    tps=100.0,  # from your benchmark
    mysql_instance='mysqld-exporter:9104',
    node_instance='node-exporter:9100'
)
print(metrics)
```

### 5. Use with DBTune

In `config_performance.ini`:
```ini
optimize_method = DML
transfer_framework = dml
prometheus_url = http://localhost:9090
mysql_instance = mysqld-exporter:9104
node_instance = node-exporter:9100
```

## Stop / Clean up

```bash
docker compose down        # stop containers
docker compose down -v     # stop and remove data volume
```

## Ports

| Service         | Port |
|----------------|------|
| Prometheus     | 9090 |
| MySQL Exporter | 9104 |
| Node Exporter  | 9100 |
