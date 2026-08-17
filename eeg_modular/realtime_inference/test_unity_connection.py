#!/usr/bin/env python
"""
Unity 连接测试脚本 - 单独测试 WebSocket 连接
"""
import asyncio
import json
import websockets
import time
from dataclasses import dataclass, asdict


@dataclass
class UnityMessage:
    emotion: str
    confidence: float
    transition_progress: float
    probabilities: dict
    timestamp: float


async def test_server():
    """测试 WebSocket 服务器"""
    print("=" * 60)
    print("🧪 Unity WebSocket 测试服务器")
    print("=" * 60)
    print("\n📋 测试说明：")
    print("1. 启动 Unity 场景")
    print("2. 确认 Unity 中的 EmotionReceiver 服务器地址是 ws://localhost:8765")
    print("3. 观察下方连接日志\n")
    
    emotions = ["happy", "sad", "normal"]
    emotion_idx = 0
    
    async def handle_client(websocket):
        nonlocal emotion_idx
        print(f"\n✅ Unity 客户端已连接!")
        print(f"   开始发送测试数据...\n")
        
        try:
            count = 0
            while True:
                # 循环切换情绪
                emotion = emotions[emotion_idx % 3]
                emotion_idx += 1
                
                # 生成随机概率
                probs = [0.0, 0.0, 0.0]
                probs[emotions.index(emotion)] = 0.7 + (0.3 * (count % 10) / 10)
                other_prob = (1 - probs[emotions.index(emotion)]) / 2
                for i in range(3):
                    if i != emotions.index(emotion):
                        probs[i] = other_prob
                
                msg = UnityMessage(
                    emotion=emotion,
                    confidence=probs[emotions.index(emotion)],
                    transition_progress=0.0,
                    probabilities={
                        "happy": probs[0],
                        "sad": probs[1],
                        "normal": probs[2]
                    },
                    timestamp=time.time()
                )
                
                payload = json.dumps(asdict(msg), ensure_ascii=False)
                await websocket.send(payload)
                
                print(f"📤 发送 #{count}: emotion={msg.emotion:8s}, confidence={msg.confidence:.4f}")
                print(f"   原始 JSON: {payload}")
                
                count += 1
                await asyncio.sleep(1.0)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n❌ Unity 客户端断开连接")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    print("🚀 服务器启动中... (ws://localhost:8765)")
    print("⏳ 等待 Unity 连接...\n")
    
    server = await websockets.serve(
        handle_client,
        "localhost",
        8765
    )
    
    print("✅ 服务器已启动!")
    print("   现在去 Unity 中运行场景\n")
    
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
