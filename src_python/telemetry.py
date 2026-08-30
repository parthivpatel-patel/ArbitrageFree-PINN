import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SystemTelemetryMonitor:
    """
    Real-time telemetry and metrics exporter for tracking GPU memory, 
    model inference latency percentiles, and loss convergence rates.
    """
    def __init__(self):
        logging.info("Initializing institutional system telemetry and metrics monitor...")
        self.metrics_log = []

    def record_metric(self, metric_name, value):
        timestamp = time.time()
        record = {"timestamp": timestamp, "metric": metric_name, "value": value}
        self.metrics_log.append(record)
        logging.info(f"Telemetry Recorded | {metric_name}: {value}")

    def export_summary(self):
        logging.info(f"Exporting telemetry summary: {len(self.metrics_log)} data points recorded.")
        return self.metrics_log
