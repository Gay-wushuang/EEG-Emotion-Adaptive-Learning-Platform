from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VotingConfig:
    window_size: int = 10
    vote_threshold: float = 0.6
    transition_duration: float = 1.0
    min_stability_frames: int = 3


class SlidingWindowVoter:
    """滑动窗口投票算法，用于情绪结果防抖动处理"""

    def __init__(self, cfg: VotingConfig):
        self.cfg = cfg
        self.window: Deque[Tuple[str, np.ndarray]] = deque(maxlen=cfg.window_size)
        self.current_emotion: Optional[str] = None
        self.stability_counter: int = 0
        self.is_transitioning: bool = False
        self.transition_start_time: float = 0.0
        self.transition_from: Optional[str] = None
        self.transition_to: Optional[str] = None
        self.transition_progress: float = 0.0
        
        logger.info(f"SlidingWindowVoter initialized: window_size={cfg.window_size}, "
                   f"vote_threshold={cfg.vote_threshold}, transition_duration={cfg.transition_duration}s")

    def update(self, emotion: str, probabilities: np.ndarray, current_time: float) -> Tuple[str, float]:
        """
        更新投票窗口并返回当前情绪状态
        
        Args:
            emotion: 当前推理出的情绪
            probabilities: 各类别的概率分布
            current_time: 当前时间戳（秒）
            
        Returns:
            (current_emotion, transition_progress)
        """
        self.window.append((emotion, probabilities))
        
        if len(self.window) < self.cfg.min_stability_frames:
            return emotion, 0.0
        
        vote_result = self._count_votes()
        dominant_emotion = vote_result["dominant_emotion"]
        confidence = vote_result["confidence"]
        
        if self.is_transitioning:
            progress = min(1.0, (current_time - self.transition_start_time) / self.cfg.transition_duration)
            self.transition_progress = progress
            
            if progress >= 1.0:
                self.is_transitioning = False
                self.current_emotion = self.transition_to
                self.stability_counter = 0
                logger.debug(f"Transition completed: {self.transition_from} -> {self.transition_to}")
            
            return self.current_emotion, progress
        
        if dominant_emotion == self.current_emotion:
            self.stability_counter += 1
            return self.current_emotion, 0.0
        
        if confidence >= self.cfg.vote_threshold:
            if self.current_emotion is not None:
                self.is_transitioning = True
                self.transition_start_time = current_time
                self.transition_from = self.current_emotion
                self.transition_to = dominant_emotion
                logger.info(f"Starting transition: {self.current_emotion} -> {dominant_emotion}")
                return self.current_emotion, 0.0
            else:
                self.current_emotion = dominant_emotion
                return self.current_emotion, 0.0
        
        return self.current_emotion or emotion, 0.0

    def _count_votes(self) -> Dict:
        """统计窗口内的投票结果"""
        votes: Dict[str, int] = {}
        avg_probs = np.zeros(len(self.window[0][1])) if self.window else None
        
        for emotion, probs in self.window:
            votes[emotion] = votes.get(emotion, 0) + 1
            if avg_probs is not None:
                avg_probs += probs
        
        if avg_probs is not None:
            avg_probs /= len(self.window)
        
        total_votes = len(self.window)
        dominant_emotion = max(votes.items(), key=lambda x: x[1])[0]
        max_votes = votes.get(dominant_emotion, 0)
        confidence = max_votes / total_votes
        
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "votes": votes,
            "total_votes": total_votes,
            "avg_probabilities": avg_probs,
        }

    def get_window_stats(self) -> Dict:
        """获取当前窗口的统计信息"""
        if not self.window:
            return {"window_size": 0, "votes": {}}
        
        vote_result = self._count_votes()
        return {
            "window_size": len(self.window),
            "max_window_size": self.cfg.window_size,
            "current_emotion": self.current_emotion,
            "stability_counter": self.stability_counter,
            "is_transitioning": self.is_transitioning,
            "transition_progress": self.transition_progress,
            **vote_result,
        }

    def reset(self):
        """重置投票器状态"""
        self.window.clear()
        self.current_emotion = None
        self.stability_counter = 0
        self.is_transitioning = False
        self.transition_start_time = 0.0
        self.transition_from = None
        self.transition_to = None
        self.transition_progress = 0.0
        logger.info("SlidingWindowVoter reset")


class ProbabilityAggregator:
    """概率聚合器，结合投票和概率平滑"""

    def __init__(self, window_size: int = 10, alpha: float = 0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.prob_window: Deque[np.ndarray] = deque(maxlen=window_size)
        self.ema_probs: Optional[np.ndarray] = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.prob_window.append(probs)
        
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs
        
        return self.ema_probs

    def get_smoothed_probs(self) -> np.ndarray:
        if not self.prob_window:
            raise ValueError("No probabilities in window")
        
        return np.mean(np.array(self.prob_window), axis=0)

    def reset(self):
        self.prob_window.clear()
        self.ema_probs = None
