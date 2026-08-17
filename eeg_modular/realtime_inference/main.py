from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
import datetime
import traceback
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EmotionInferenceModel, InferenceConfig
from src.voting import SlidingWindowVoter, VotingConfig, ProbabilityAggregator
from src.unity_comm import UnityEmotionSender, UnityConfig
from src.thinkgear import ThinkGearCollector, ThinkGearConfig
from src.decision import EWMASustainedNegativeDecision


@dataclass
class SystemConfig:
    model: InferenceConfig
    voting: VotingConfig
    unity: UnityConfig
    thinkgear: ThinkGearConfig
    inference: Dict[str, Any]
    logging: Dict[str, Any]


def setup_logging(cfg: Dict[str, Any]):
    log_level = getattr(logging, cfg.get("level", "INFO"))
    log_file = cfg.get("log_file", "realtime_inference.log")
    console_output = cfg.get("console_output", True)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handlers = []
    if console_output:
        handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str) -> SystemConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg["model"]
    voting_cfg = cfg["voting"]
    unity_cfg = cfg["unity"]
    thinkgear_cfg = cfg["thinkgear"]
    inference_cfg = cfg["inference"]
    logging_cfg = cfg["logging"]
    
    return SystemConfig(
            model=InferenceConfig(
                model_path=Path(model_cfg["path"]),
                model_type=model_cfg.get("type", "multimodal_cnn"),
                device=model_cfg.get("device", "auto"),
                num_classes=model_cfg.get("num_classes", 3),
                modalities=tuple(model_cfg.get("modalities", ["filtered", "powerspec", "att", "med"])),
                time_steps=model_cfg.get("time_steps", 10),
                feat_dim=model_cfg.get("feat_dim", 4),
                use_cvae=model_cfg.get("use_cvae", True),
                cvae_latent_dim=model_cfg.get("cvae_latent_dim", 64),
                cvae_input_dim=model_cfg.get("cvae_input_dim", 160),
                dropout=model_cfg.get("dropout", 0.5),
                scalers_dir=Path(model_cfg["scalers_dir"]) if "scalers_dir" in model_cfg and model_cfg["scalers_dir"] is not None else None,
                skip_scaling=model_cfg.get("skip_scaling", False),
            ),
        voting=VotingConfig(
            window_size=voting_cfg.get("window_size", 10),
            vote_threshold=voting_cfg.get("vote_threshold", 0.6),
            transition_duration=voting_cfg.get("transition_duration", 1.0),
            min_stability_frames=voting_cfg.get("min_stability_frames", 3),
        ),
        unity=UnityConfig(
            host=unity_cfg.get("host", "localhost"),
            port=unity_cfg.get("port", 8765),
            max_connections=unity_cfg.get("max_connections", 5),
            ping_interval=unity_cfg.get("ping_interval", 30.0),
            ping_timeout=unity_cfg.get("ping_timeout", 10.0),
        ),
        thinkgear=ThinkGearConfig(
            connection_mode=thinkgear_cfg.get("connection_mode", "tcp"),
            com_port=thinkgear_cfg.get("com_port", "COM3"),
            baud_rate=thinkgear_cfg.get("baud_rate", 57600),
            tcp_host=thinkgear_cfg.get("tcp_host", "127.0.0.1"),
            tcp_port=thinkgear_cfg.get("tcp_port", 13854),
            sample_rate=thinkgear_cfg.get("sample_rate", 512),
            buffer_size=thinkgear_cfg.get("buffer_size", 1024),
            analysis_window_seconds=thinkgear_cfg.get("analysis_window_seconds", 30.0),
            use_mock=thinkgear_cfg.get("use_mock", False),
            use_training_data=thinkgear_cfg.get("use_training_data", True),
            features_dir=thinkgear_cfg.get("features_dir", "../features"),
            stale_timeout_seconds=thinkgear_cfg.get("stale_timeout_seconds", 2.5),
        ),
        inference=inference_cfg,
        logging=logging_cfg,
    )


class EmotionInferenceSystem:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("EmotionInferenceSystem")
        
        self.model: Optional[EmotionInferenceModel] = None
        self.voter: Optional[SlidingWindowVoter] = None
        self.prob_agg: Optional[ProbabilityAggregator] = None
        self.unity_sender: Optional[UnityEmotionSender] = None
        self.collector: Optional[ThinkGearCollector] = None
        
        self._running = False
        self._start_time: float = 0.0
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_offline_log_time = 0.0
        self._decision_policy: Optional[EWMASustainedNegativeDecision] = None
        self.results_file = None
        self.results_writer = None
        
        # log.txt 文件处理
        self.log_file_path = Path("log.txt")
        self.log_file = None

    def initialize(self):
        self.logger.info("Initializing EmotionInferenceSystem...")
        
        # 打开 log.txt 文件
        try:
            self.log_file = open(self.log_file_path, "a", encoding="utf-8")
            self._write_to_log("=" * 80 + "\n")
            self._write_to_log(f"System started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._write_to_log("=" * 80 + "\n")
        except Exception as e:
            self.logger.error(f"Failed to open log.txt: {e}")
            self.log_file = None
        
        self.model = EmotionInferenceModel(self.cfg.model)
        self.voter = SlidingWindowVoter(self.cfg.voting)
        self.prob_agg = ProbabilityAggregator(window_size=self.cfg.voting.window_size)
        self.unity_sender = UnityEmotionSender(self.cfg.unity)
        self.collector = ThinkGearCollector(self.cfg.thinkgear)
        negative_index = self.model.class_names.index("sad")
        self._decision_policy = EWMASustainedNegativeDecision(
            negative_index=negative_index,
            alpha=float(self.cfg.inference.get("decision_ewma_alpha", 0.20)),
            negative_threshold=float(
                self.cfg.inference.get("intervention_negative_probability", 0.60)
            ),
            sustain_seconds=float(
                self.cfg.inference.get("intervention_sustain_seconds", 20)
            ),
            cooldown_seconds=float(
                self.cfg.inference.get("intervention_cooldown_seconds", 90)
            ),
        )
        results_path = Path(self.cfg.inference.get("results_csv", "logs/inference_results.csv"))
        results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_file = results_path.open("a", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp", "predicted_class", "prob_positive", "prob_neutral",
            "prob_negative", "confidence", "attention", "meditation",
            "poor_signal", "inference_latency_ms", "raw_class",
            "signal_good", "intervention_triggered",
        ]
        self.results_writer = csv.DictWriter(self.results_file, fieldnames=fieldnames)
        if self.results_file.tell() == 0:
            self.results_writer.writeheader()
            self.results_file.flush()
        
        self.logger.info("System initialized successfully")

    def start(self):
        if self._running:
            self.logger.warning("System already running")
            return
        
        self._running = True
        self._start_time = time.time()
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_offline_log_time = 0.0
        
        self.logger.info("Starting system...")
        
        self.collector.start()
        self.unity_sender.start()
        
        time.sleep(0.5)
        
        self.logger.info("System started. Press Ctrl+C to stop.")
        self._main_loop()

    def stop(self):
        self.logger.info("Stopping system...")
        self._running = False
        
        if self.collector:
            self.collector.stop()
        if self.unity_sender:
            self.unity_sender.stop()
        
        self._log_performance_stats()
        
        # 关闭 log.txt 文件
        if self.log_file:
            try:
                self._write_to_log("=" * 80 + "\n")
                self._write_to_log(f"System stopped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._write_to_log("=" * 80 + "\n\n")
                self.log_file.close()
            except Exception as e:
                self.logger.error(f"Error closing log.txt: {e}")
            finally:
                self.log_file = None
        if self.results_file:
            self.results_file.close()
            self.results_file = None
        
        self.logger.info("System stopped")

    def _main_loop(self):
        inference_interval = self.cfg.inference.get("inference_interval", 0.1)
        
        try:
            while self._running:
                loop_start = time.time()
                
                self._do_inference()
                
                elapsed = time.time() - loop_start
                if elapsed < inference_interval:
                    time.sleep(inference_interval - elapsed)
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.stop()

    def _write_to_log(self, log_line: str):
        """安全地写入 log.txt"""
        try:
            if not self.log_file:
                # 尝试重新打开文件
                self.log_file = open(self.log_file_path, "a", encoding="utf-8")
            self.log_file.write(log_line)
            self.log_file.flush()
        except Exception as e:
            self.logger.error(f"Failed to write to log.txt: {e}")
    
    def _do_inference(self):
        try:
            inference_start = time.time()
            current_time = time.time()
            status = self.collector.get_status()

            if not status["device_connected"] or not status["buffer_ready"]:
                self.unity_sender.send(
                    emotion="offline",
                    confidence=0.0,
                    transition_progress=0.0,
                    probabilities={"happy": 0.0, "sad": 0.0, "normal": 0.0},
                    timestamp=current_time,
                    device_connected=False,
                    stream_connected=bool(status["stream_connected"]),
                    data_age_seconds=float(status["data_age_seconds"]),
                    packet_age_seconds=float(status["packet_age_seconds"]),
                    poor_signal=int(status["poor_signal"]),
                    attention=int(status["attention"]),
                    meditation=int(status["meditation"]),
                    source=str(status["source"]),
                    raw_emotion="offline",
                    esense_age_seconds=float(status["esense_age_seconds"]),
                    power_age_seconds=float(status["power_age_seconds"]),
                    raw_packet_count=int(status["raw_packet_count"]),
                    esense_packet_count=int(status["esense_packet_count"]),
                    power_packet_count=int(status["power_packet_count"]),
                )
                if current_time - self._last_offline_log_time >= 2.0:
                    self._last_offline_log_time = current_time
                    self.logger.info(
                        "[MAIN] Device offline/stale: "
                        f"stream={status['stream_connected']}, "
                        f"source={status['source']}, "
                        f"poor_signal={status['poor_signal']}, "
                        f"att={status['attention']}, med={status['meditation']}, "
                        f"data_age={status['data_age_seconds']:.1f}s, "
                        f"esense_age={status['esense_age_seconds']:.1f}s, "
                        f"raw/esense/power={status['raw_packet_count']}/"
                        f"{status['esense_packet_count']}/{status['power_packet_count']}, "
                        f"buffer={status['buffer_fill_seconds']:.1f}/"
                        f"{status['analysis_window_seconds']:.1f}s"
                    )
                return
            
            self.logger.debug("[MAIN] Step 1: Collecting multimodal features...")
            multimodal_data = self.collector.get_multimodal_features(
                time_steps=self.cfg.model.time_steps,
                feat_dim=self.cfg.model.feat_dim,
            )
            
            self.logger.debug("[MAIN] Step 2: Running model prediction...")
            emotion, probs = self.model.predict(multimodal_data)
            
            self.logger.debug("[MAIN] Step 3: Applying probability smoothing...")
            smoothed_probs = self.prob_agg.update(probs)
            prob_dict = {
                name: float(smoothed_probs[index])
                for index, name in enumerate(self.model.class_names)
            }
            self.logger.debug("[MAIN] Smoothed probabilities - %s", prob_dict)
            
            self.logger.debug("[MAIN] Step 4: Applying voting...")
            final_emotion, transition_progress = self.voter.update(
                emotion=emotion,
                probabilities=smoothed_probs,
                current_time=current_time,
            )
            
            confidence = float(smoothed_probs.max())
            confidence_threshold = float(self.cfg.inference.get("confidence_threshold", 0.60))
            max_poor_signal = int(self.cfg.inference.get("max_poor_signal", 50))
            signal_good = int(status["poor_signal"]) <= max_poor_signal
            output_class = final_emotion
            if not signal_good:
                output_class = "poor_signal"
            elif confidence < confidence_threshold:
                output_class = "uncertain"

            decision = self._decision_policy.update(
                probabilities=np.asarray(probs),
                timestamp=current_time,
                eligible=signal_good and confidence >= confidence_threshold,
            )
            intervention_triggered = decision.intervention_triggered
            if intervention_triggered:
                self.logger.warning(
                    "[FEEDBACK] EWMA negative probability remained above threshold; "
                    "learning intervention requested"
                )
            
            self.logger.debug("[MAIN] Step 5: Sending to Unity...")
            self.unity_sender.send(
                emotion=output_class,
                confidence=confidence,
                transition_progress=transition_progress,
                probabilities=prob_dict,
                timestamp=current_time,
                device_connected=bool(status["device_connected"]),
                stream_connected=bool(status["stream_connected"]),
                data_age_seconds=float(status["data_age_seconds"]),
                packet_age_seconds=float(status["packet_age_seconds"]),
                poor_signal=int(status["poor_signal"]),
                attention=int(status["attention"]),
                meditation=int(status["meditation"]),
                source=str(status["source"]),
                raw_emotion=emotion,
                esense_age_seconds=float(status["esense_age_seconds"]),
                power_age_seconds=float(status["power_age_seconds"]),
                raw_packet_count=int(status["raw_packet_count"]),
                esense_packet_count=int(status["esense_packet_count"]),
                power_packet_count=int(status["power_packet_count"]),
            )
            
            inference_time = (time.time() - inference_start) * 1000
            self._inference_count += 1
            self._total_inference_time += inference_time
            self.results_writer.writerow(
                {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "predicted_class": output_class,
                    "prob_positive": f"{prob_dict['happy']:.6f}",
                    "prob_neutral": f"{prob_dict['normal']:.6f}",
                    "prob_negative": f"{prob_dict['sad']:.6f}",
                    "confidence": f"{confidence:.6f}",
                    "attention": int(status["attention"]),
                    "meditation": int(status["meditation"]),
                    "poor_signal": int(status["poor_signal"]),
                    "inference_latency_ms": f"{inference_time:.3f}",
                    "raw_class": emotion,
                    "signal_good": int(signal_good),
                    "intervention_triggered": int(intervention_triggered),
                }
            )
            self.results_file.flush()
            
            if self._inference_count % 100 == 0:
                self._log_performance_stats()
            
            # 获取投票窗口统计信息
            window_stats = self.voter.get_window_stats() if self.voter else {}
            
            self.logger.info(
                f"[MAIN] Inference #{self._inference_count}: "
                f"Final={final_emotion} (conf={confidence:.2f}, "
                f"transition={transition_progress:.2f}, "
                f"latency={inference_time:.1f}ms), "
                f"Raw={emotion}, EEG att={status['attention']}, med={status['meditation']}, "
                f"poor={status['poor_signal']}, source={status['source']}, "
                f"Votes={window_stats}"
            )
            
            self.logger.debug(
                f"[MAIN] Raw model output: {emotion}, probs=[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]"
            )
            
            # 写入 log.txt
            log_line = (
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
                f"Inference #{self._inference_count:6d} | "
                f"Final={final_emotion:8s} | "
                f"Conf={confidence:.4f} | "
                f"Transition={transition_progress:.4f} | "
                f"Latency={inference_time:6.1f}ms | "
                f"Raw={emotion:8s} | "
                f"Att={int(status['attention']):3d} | "
                f"Med={int(status['meditation']):3d} | "
                f"Poor={int(status['poor_signal']):3d} | "
                f"Source={status['source']} | "
                f"Happy={prob_dict['happy']:.4f} | "
                f"Sad={prob_dict['sad']:.4f} | "
                f"Normal={prob_dict['normal']:.4f}\n"
            )
            self._write_to_log(log_line)
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            self.logger.error(traceback.format_exc())
            # 记录错误到 log.txt
            error_log_line = (
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
                f"ERROR: {str(e)}\n"
            )
            self._write_to_log(error_log_line)

    def _log_performance_stats(self):
        if self._inference_count == 0:
            return
        
        avg_latency = self._total_inference_time / self._inference_count
        uptime = time.time() - self._start_time
        
        stats = {
            "uptime_seconds": uptime,
            "inference_count": self._inference_count,
            "avg_latency_ms": avg_latency,
            "unity_connected": self.unity_sender.is_connected() if self.unity_sender else False,
            "window_stats": self.voter.get_window_stats() if self.voter else {},
        }
        
        self.logger.info(f"Performance stats: {stats}")


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Emotion Real-time Inference System")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        cfg = load_config(args.config)
        setup_logging(cfg.logging)
        
        logger = logging.getLogger("main")
        logger.info("=" * 60)
        logger.info("EEG Emotion Real-time Inference System")
        logger.info("=" * 60)
        
        system = EmotionInferenceSystem(cfg)
        system.initialize()
        system.start()
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
