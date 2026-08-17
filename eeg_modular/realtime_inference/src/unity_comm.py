from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class UnityMessage:
    emotion: str
    confidence: float
    transition_progress: float
    probabilities: Dict[str, float]
    timestamp: float
    device_connected: bool = True
    stream_connected: bool = True
    data_age_seconds: float = 0.0
    packet_age_seconds: float = 0.0
    poor_signal: int = 0
    attention: int = 0
    meditation: int = 0
    source: str = "unknown"
    raw_emotion: str = "unknown"
    esense_age_seconds: float = 999999.0
    power_age_seconds: float = 999999.0
    raw_packet_count: int = 0
    esense_packet_count: int = 0
    power_packet_count: int = 0


@dataclass
class UnityConfig:
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 5
    ping_interval: float = 30.0
    ping_timeout: float = 10.0


class UnityWebSocketServer:
    """Unity WebSocket通信服务器"""

    def __init__(self, cfg: UnityConfig):
        self.cfg = cfg
        self.clients: Set[Any] = set()
        self.server = None
        self._running = False
        self._message_callbacks: list[Callable[[Dict], None]] = []

    async def start(self):
        """启动WebSocket服务器"""
        try:
            import websockets
        except ImportError:
            logger.error("websockets library not found. Please install: pip install websockets")
            raise

        self._running = True
        
        # 使用兼容的方式创建服务器
        self.server = await websockets.serve(
            self._handle_client,
            self.cfg.host,
            self.cfg.port,
            max_size=None,
            max_queue=self.cfg.max_connections,
            ping_interval=self.cfg.ping_interval,
            ping_timeout=self.cfg.ping_timeout,
        )
        
        logger.info(f"Unity WebSocket server started on {self.cfg.host}:{self.cfg.port}")

    async def stop(self):
        """停止服务器"""
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Unity WebSocket server stopped")

    async def _handle_client(self, websocket, path=None):
        """处理客户端连接 - 兼容新旧版本websockets库"""
        client_id = id(websocket)
        logger.info(f"[WS] ✅ New Unity client connected (id={client_id})")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                await self._on_message(websocket, message)
        except Exception as e:
            logger.error(f"[WS] ❌ Client {client_id} error: {e}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            logger.info(f"[WS] 🔌 Unity client disconnected (id={client_id})")

    async def _on_message(self, websocket, message: str):
        """处理来自Unity的消息"""
        try:
            data = json.loads(message)
            logger.debug(f"[WS] Received from Unity: {data}")
            
            for callback in self._message_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"[WS] Callback error: {e}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"[WS] Invalid JSON from Unity: {e}")

    async def send_emotion_update(self, message: UnityMessage):
        """发送情绪更新到所有连接的Unity客户端"""
        if not self.clients:
            return
        
        try:
            payload = json.dumps(asdict(message), ensure_ascii=False)
            await asyncio.gather(
                *[client.send(payload) for client in self.clients],
                return_exceptions=True
            )
            logger.debug(f"[WS] Sent emotion update: {message.emotion} (conf={message.confidence:.2f})")
        except Exception as e:
            logger.error(f"[WS] Failed to send emotion update: {e}")

    def add_message_callback(self, callback: Callable[[Dict], None]):
        """添加Unity消息回调"""
        self._message_callbacks.append(callback)

    def remove_message_callback(self, callback: Callable[[Dict], None]):
        """移除Unity消息回调"""
        if callback in self._message_callbacks:
            self._message_callbacks.remove(callback)

    def is_connected(self) -> bool:
        """检查是否有Unity客户端连接"""
        return len(self.clients) > 0

    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self.clients)


class UnityEmotionSender:
    """简化的Unity情绪发送器（同步版本）"""

    def __init__(self, cfg: UnityConfig):
        self.cfg = cfg
        self._loop = None
        self._server: Optional[UnityWebSocketServer] = None
        self._task = None

    def start(self):
        """启动发送器（在后台线程中运行）"""
        import threading
        
        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._server = UnityWebSocketServer(self.cfg)
            self._task = self._loop.create_task(self._server.start())
            try:
                self._loop.run_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self._loop.close()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info("UnityEmotionSender started in background thread")

    def stop(self):
        """停止发送器"""
        if self._loop and self._server:
            asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("UnityEmotionSender stopped")

    def send(self, emotion: str, confidence: float, transition_progress: float, 
             probabilities: Dict[str, float], timestamp: float, device_connected: bool = True,
             stream_connected: bool = True, data_age_seconds: float = 0.0,
             packet_age_seconds: float = 0.0, poor_signal: int = 0,
             attention: int = 0, meditation: int = 0, source: str = "unknown",
             raw_emotion: str = "unknown", esense_age_seconds: float = 999999.0,
             power_age_seconds: float = 999999.0, raw_packet_count: int = 0,
             esense_packet_count: int = 0, power_packet_count: int = 0):
        """发送情绪更新（非阻塞）"""
        if self._loop and self._server:
            message = UnityMessage(
                emotion=emotion,
                confidence=confidence,
                transition_progress=transition_progress,
                probabilities=probabilities,
                timestamp=timestamp,
                device_connected=device_connected,
                stream_connected=stream_connected,
                data_age_seconds=data_age_seconds,
                packet_age_seconds=packet_age_seconds,
                poor_signal=poor_signal,
                attention=attention,
                meditation=meditation,
                source=source,
                raw_emotion=raw_emotion,
                esense_age_seconds=esense_age_seconds,
                power_age_seconds=power_age_seconds,
                raw_packet_count=raw_packet_count,
                esense_packet_count=esense_packet_count,
                power_packet_count=power_packet_count,
            )
            asyncio.run_coroutine_threadsafe(
                self._server.send_emotion_update(message),
                self._loop
            )

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._server is not None and self._server.is_connected()
