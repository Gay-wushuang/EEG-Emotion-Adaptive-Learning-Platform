"""测试修复后的实时推理系统"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.training_data_sampler import get_sampler
from src.model import EmotionInferenceModel, InferenceConfig
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("test_fix")

def test_sampler():
    """测试训练数据采样器"""
    logger.info("=" * 60)
    logger.info("测试 1: 训练数据采样器")
    logger.info("=" * 60)
    
    try:
        sampler = get_sampler("../features")
        logger.info("✅ 采样器加载成功")
        
        # 获取几个样本
        for i in range(3):
            sample, label = sampler.get_sample()
            class_names = ["happy", "sad", "normal"]
            logger.info(f"样本 {i+1}: 标签={label} ({class_names[label] if label < len(class_names) else 'unknown'})")
            
            for mod, arr in sample.items():
                logger.info(f"  {mod}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 采样器测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model():
    """测试模型推理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 模型推理（使用训练数据）")
    logger.info("=" * 60)
    
    try:
        # 加载采样器
        sampler = get_sampler("../features")
        
        # 加载模型（跳过归一化）
        cfg = InferenceConfig(
            model_path=Path("../outputs/20260316_004402/models/best_fold4.pt"),
            skip_scaling=True,
            scalers_dir=Path("../features")
        )
        
        model = EmotionInferenceModel(cfg)
        logger.info("✅ 模型加载成功")
        
        # 测试几个样本
        class_names = ["happy", "sad", "normal"]
        for i in range(5):
            sample, true_label = sampler.get_sample()
            emotion, probs = model.predict(sample)
            
            true_label_name = class_names[true_label] if true_label < len(class_names) else "unknown"
            logger.info(f"推理 {i+1}: 真实标签={true_label_name}, 预测={emotion}, "
                       f"置信度={probs.max():.4f}")
            logger.info(f"  概率分布: happy={probs[0]:.4f}, sad={probs[1]:.4f}, normal={probs[2]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 模型测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("开始测试修复后的实时推理系统...")
    
    success1 = test_sampler()
    success2 = test_model()
    
    logger.info("\n" + "=" * 60)
    if success1 and success2:
        logger.info("✅ 所有测试通过！修复成功！")
    else:
        logger.error("❌ 部分测试失败，请检查错误信息")
    logger.info("=" * 60)
