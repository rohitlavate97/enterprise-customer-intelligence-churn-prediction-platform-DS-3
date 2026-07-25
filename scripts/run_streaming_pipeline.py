"""CLI script to run simulated real-time event producer and stream processor consumer."""

import json
from config.settings import settings
from streaming.consumer import StreamProcessorConsumer
from streaming.producer import CustomerEventProducer
from utils.logger import get_logger

logger = get_logger("scripts.run_streaming_pipeline")


def main() -> None:
    logger.info("Starting Real-Time Event Producer and Stream Processor Consumer pipeline...")

    producer = CustomerEventProducer(seed=42)
    consumer = StreamProcessorConsumer(alert_threshold=0.80)

    event_count = 0
    alert_count = 0

    for event in producer.stream_events(max_events=30, delay_sec=0.0):
        event_count += 1
        alert = consumer.process_event(event)
        if alert is not None:
            alert_count += 1

    alerts_path = settings.artifacts_dir / "streaming_alerts.json"
    with open(alerts_path, "w", encoding="utf-8") as f:
        json.dump(consumer.dispatched_alerts, f, indent=2)

    logger.info(f"\n==================== STREAMING PIPELINE SUMMARY ====================")
    logger.info(f"Processed Events: {event_count}")
    logger.info(f"High-Risk Real-Time Alerts Emitted: {alert_count}")
    logger.info(f"Saved Real-Time Alert Log JSON to {alerts_path}")


if __name__ == "__main__":
    main()
