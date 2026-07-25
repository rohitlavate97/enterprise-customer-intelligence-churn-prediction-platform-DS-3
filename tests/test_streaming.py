"""Unit tests for CustomerEventProducer, StreamProcessorConsumer, and AlertDispatcher."""

import pytest
from streaming.consumer import AlertDispatcher, StreamProcessorConsumer
from streaming.producer import CustomerEventProducer


def test_event_producer():
    """Assert CustomerEventProducer generates valid event JSON structure."""
    producer = CustomerEventProducer(seed=42)
    event = producer.generate_single_event("CUST-1001")

    assert event["customer_id"] == "CUST-1001"
    assert "event_id" in event
    assert "event_type" in event
    assert "timestamp" in event
    assert isinstance(event["attributes"], dict)


def test_stream_consumer_processing():
    """Assert StreamProcessorConsumer updates sliding window and returns alert or None."""
    producer = CustomerEventProducer(seed=42)
    consumer = StreamProcessorConsumer(alert_threshold=0.80)

    events = list(producer.stream_events(max_events=10, delay_sec=0.0))
    for evt in events:
        res = consumer.process_event(evt)
        if res is not None:
            assert res["churn_probability"] >= 0.80
            assert "channel" in res


def test_alert_dispatcher():
    """Assert AlertDispatcher generates Slack/Webhook alert payload format."""
    alert = AlertDispatcher.dispatch_alert(
        customer_id="CUST-HIGH",
        churn_prob=0.88,
        clv=2400.0,
        reasons=["Support ticket surge (+30% risk)"],
        recommended_action="VIP Callback",
    )

    assert alert["customer_id"] == "CUST-HIGH"
    assert alert["churn_probability"] == 0.88
    assert alert["revenue_at_risk"] == 0.88 * 2400.0
    assert alert["severity"] == "CRITICAL"
    assert alert["channel"] == "#retention-urgent-alerts"
