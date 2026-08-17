
import re
import datetime
from collections import defaultdict

def parse_python_log(log_path):
    """解析 Python 日志"""
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| Inference #\s*(\d+) \| Final=(\w+)\s+\| Conf=([\d\.]+)', line)
            if match:
                time_str = match.group(1)
                inference_num = int(match.group(2))
                emotion = match.group(3)
                confidence = float(match.group(4))
                
                # 解析概率
                prob_match = re.search(r'Happy=([\d\.]+) \| Sad=([\d\.]+) \| Normal=([\d\.]+)', line)
                if prob_match:
                    happy = float(prob_match.group(1))
                    sad = float(prob_match.group(2))
                    normal = float(prob_match.group(3))
                else:
                    happy = sad = normal = 0.0
                
                dt = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
                timestamp = dt.timestamp()
                
                data.append({
                    'timestamp': timestamp,
                    'inference_num': inference_num,
                    'emotion': emotion,
                    'confidence': confidence,
                    'happy': happy,
                    'sad': sad,
                    'normal': normal
                })
    return data

def parse_unity_log(log_path):
    """解析 Unity 日志"""
    data = []
    import json
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '收到数据' in line:
                match = re.search(r'\{.*\}', line)
                if match:
                    try:
                        json_str = match.group(0)
                        msg = json.loads(json_str)
                        data.append({
                            'timestamp': msg['timestamp'],
                            'emotion': msg['emotion'],
                            'confidence': msg['confidence'],
                            'happy': msg['probabilities'].get('happy', 0),
                            'sad': msg['probabilities'].get('sad', 0),
                            'normal': msg['probabilities'].get('normal', 0)
                        })
                    except:
                        pass
    return data

def find_matches(python_data, unity_data, threshold=0.1):
    """寻找匹配的数据"""
    matches = []
    mismatches = []
    
    # 按时间排序
    python_data.sort(key=lambda x: x['timestamp'])
    unity_data.sort(key=lambda x: x['timestamp'])
    
    print(f"Python 数据点: {len(python_data)}")
    print(f"Unity 数据点: {len(unity_data)}")
    print("=" * 80)
    
    for unity_msg in unity_data:
        ut = unity_msg['timestamp']
        
        # 寻找最接近的 Python 数据点
        best_match = None
        best_diff = float('inf')
        
        for py_msg in python_data:
            pt = py_msg['timestamp']
            diff = abs(ut - pt)
            if diff < best_diff and diff < threshold:
                best_diff = diff
                best_match = py_msg
        
        if best_match:
            # 检查情绪是否一致
            emotion_match = (best_match['emotion'] == unity_msg['emotion'])
            confidence_match = abs(best_match['confidence'] - unity_msg['confidence']) < 0.01
            happy_match = abs(best_match['happy'] - unity_msg['happy']) < 0.001
            sad_match = abs(best_match['sad'] - unity_msg['sad']) < 0.001
            normal_match = abs(best_match['normal'] - unity_msg['normal']) < 0.001
            
            all_match = emotion_match and confidence_match and happy_match and sad_match and normal_match
            
            result = {
                'time_diff': best_diff,
                'python': best_match,
                'unity': unity_msg,
                'emotion_match': emotion_match,
                'confidence_match': confidence_match,
                'happy_match': happy_match,
                'sad_match': sad_match,
                'normal_match': normal_match,
                'all_match': all_match
            }
            
            if all_match:
                matches.append(result)
            else:
                mismatches.append(result)
    
    return matches, mismatches

def main():
    python_path = r'd:\proegg\eeg_modular\realtime_inference\log.txt'
    unity_path = r'd:\proegg\eeg_modular\aa.txt'
    
    print("正在解析 Python 日志...")
    python_data = parse_python_log(python_path)
    
    print("正在解析 Unity 日志...")
    unity_data = parse_unity_log(unity_path)
    
    print("正在匹配数据...")
    matches, mismatches = find_matches(python_data, unity_data)
    
    print("\n" + "=" * 80)
    print(f"完全匹配: {len(matches)} 条")
    print(f"不匹配: {len(mismatches)} 条")
    print("=" * 80)
    
    if mismatches:
        print("\n不匹配的记录:")
        for i, mm in enumerate(mismatches[:5], 1):
            py = mm['python']
            un = mm['unity']
            print(f"\n--- 不匹配 #{i} (时间差: {mm['time_diff']:.3f}s) ---")
            print(f"Python: emotion={py['emotion']}, conf={py['confidence']:.4f}, H={py['happy']:.4f}, S={py['sad']:.4f}, N={py['normal']:.4f}")
            print(f"Unity:  emotion={un['emotion']}, conf={un['confidence']:.4f}, H={un['happy']:.4f}, S={un['sad']:.4f}, N={un['normal']:.4f}")
            if not mm['emotion_match']:
                print("  情绪不一致！")
            if not mm['confidence_match']:
                print("  置信度不一致！")
    
    if matches and not mismatches:
        print("\n所有数据完全一致！系统工作正常！")

if __name__ == '__main__':
    main()
