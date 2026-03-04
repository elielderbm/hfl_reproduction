import asyncio
import base64
import json
import os
import threading
import time

import paho.mqtt.client as mqtt

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


class MQTTBus:
    def __init__(
        self,
        client_id: str,
        host: str | None = None,
        port: int | None = None,
        keepalive: int | None = None,
        qos: int | None = None,
    ):
        self.host = host or os.getenv("MQTT_HOST", "mosquitto")
        self.port = int(port or os.getenv("MQTT_PORT", "1883"))
        self.keepalive = int(keepalive or os.getenv("MQTT_KEEPALIVE", "60"))
        self.qos = int(qos or os.getenv("MQTT_QOS", "1"))
        self._loop = None
        self._queue: asyncio.Queue[tuple[str, str, dict]] = asyncio.Queue()
        self._connected = threading.Event()

        self.client = mqtt.Client(client_id=client_id, clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected.set()
        else:
            self._connected.clear()

    def _on_disconnect(self, client, userdata, rc):
        self._connected.clear()

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore")
        if self._loop is None:
            return
        payload_len = len(msg.payload or b"")
        packet_len = self._packet_size(msg.topic, payload_len, qos=msg.qos)
        meta = {
            "mqtt_payload_bytes": payload_len,
            "mqtt_packet_bytes": packet_len,
            "mqtt_qos": int(msg.qos),
        }
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (msg.topic, payload, meta))

    async def connect(self, timeout: float | None = 10.0):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self.client.connect(self.host, self.port, keepalive=self.keepalive)
        self.client.loop_start()
        # wait for connect with timeout
        ok = await asyncio.get_running_loop().run_in_executor(None, self._connected.wait, timeout)
        if not ok:
            raise TimeoutError(f"MQTT connect timeout ({self.host}:{self.port})")

    def subscribe(self, topic: str):
        self.client.subscribe(topic, qos=self.qos)

    @staticmethod
    def _remaining_length_bytes(n: int) -> int:
        # MQTT Remaining Length uses variable-length encoding (1-4 bytes)
        count = 0
        x = int(n)
        while True:
            count += 1
            x //= 128
            if x == 0:
                break
        return count

    def _packet_size(self, topic: str, payload_len: int, qos: int | None = None) -> int:
        # MQTT v3.1.1 PUBLISH packet size (fixed + variable header + payload)
        topic_len = len(topic.encode("utf-8"))
        qos_val = self.qos if qos is None else int(qos)
        var_header = 2 + topic_len + (2 if qos_val > 0 else 0)  # topic length + topic + packet id (QoS>0)
        remaining = var_header + int(payload_len)
        fixed = 1 + self._remaining_length_bytes(remaining)
        return fixed + remaining

    def publish_json(self, topic: str, payload: dict):
        msg = json.dumps(payload, ensure_ascii=False)
        payload_bytes = msg.encode("utf-8")
        payload_len = len(payload_bytes)
        packet_len = self._packet_size(topic, payload_len)
        self.client.publish(topic, msg, qos=self.qos)
        return {
            "mqtt_payload_bytes": payload_len,
            "mqtt_packet_bytes": packet_len,
            "mqtt_qos": int(self.qos),
        }

    async def recv_json(self, timeout: float | None = None):
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if len(item) == 2:
            topic, payload = item
            meta = {}
        else:
            topic, payload, meta = item
        try:
            data = json.loads(payload)
        except Exception as e:
            meta = dict(meta or {})
            meta["mqtt_parse_error"] = True
            meta["mqtt_parse_error_msg"] = str(e)
            data = {}
        return topic, data, meta

    async def recv_match(self, predicate, timeout: float | None = None):
        while True:
            topic, data, meta = await self.recv_json(timeout=timeout)
            if predicate(topic, data):
                return topic, data, meta

def now_ms(): return int(time.time()*1000)
