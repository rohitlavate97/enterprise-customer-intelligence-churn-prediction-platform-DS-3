"""Simulated Real-Time Customer Telemetry Event Producer."""

import random
import time
from typing import Any, Generator
from utils.logger import get_logger

logger = get_logger("streaming.producer")

EVENT_TYPES = [
    "support_ticket_opened",
    "usage_drop_detected",
    "payment_failed",
    "competitor_viewed",
    "price_increase_notified",
    "app_login",
]


class CustomerEventProducer:
    """Simulates real-time stream of customer telemetry and behavior events."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def generate_single_event(self, customer_id: str | None = None) -> dict[str, Any]:
        """Generate a single customer activity event."""
        cust_id = customer_id or f"CUST-{self.random.randint(1000, 9999)}"
        event_type = self.random.choice(EVENT_TYPES)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payload = {
            "event_id": f"EVT-{self.random.randint(100000, 999999)}",
            "customer_id": cust_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "attributes": {},
        }

        if event_type == "support_ticket_opened":
            payload["attributes"] = {"ticket_category": "Billing Dispute", "severity": "High"}
        elif event_type == "usage_drop_detected":
            payload["attributes"] = {"drop_percentage": self.random.uniform(-0.60, -0.20)}
        elif event_type == "payment_failed":
            payload["attributes"] = {"reason": "Insufficient Funds", "retry_count": 2}
        elif event_type == "competitor_viewed":
            payload["attributes"] = {"competitor_name": "CloudFlex", "offer_discount": "20%"}

        return payload

    def stream_events(self, max_events: int = 20, delay_sec: float = 0.01) -> Generator[dict[str, Any], None, None]:
        """Generator yielding simulated event stream."""
        logger.info(f"CustomerEventProducer starting stream generation ({max_events} events)...")
        for _ in range(max_events):
            event = self.generate_single_event()
            yield event
            if delay_sec > 0:
                time.sleep(delay_sec)
