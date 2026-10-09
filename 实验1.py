# -*- coding: utf-8 -*-
"""
数字信号处理实验一：基于时域分析技术的语音识别

数据目录结构：
data/
  0/*.wav
  1/*.wav
  ...
  9/*.wav

运行示例：
python experiment1.py --data_dir data --plot
python experiment1.py --analyze data/0/0_01.wav --plot
python experiment1.py --record data/0/0_11.wav --duration 1.0
"""

import os
import glob
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. 读取 WAV 文件
# ============================================================
def read_wav(path):
    """
    读取 wav 文件，返回采样率和单声道 float32 信号。
    自动处理 int16/int32/uint8 等常见格式，并去直流。
    """
    sr, data = wavfile.read(path)

    # 多声道转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 归一化到 [-1, 1] 附近
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # 去直流
    data = data - np.mean(data)
    return sr, data


# ============================================================
# 2. 分帧与加窗
# ============================================================
def get_window(window, N):
    """返回指定类型的窗函数。"""
    if window == "rect":
        return np.ones(N, dtype=np.float32)
    elif window == "hamming":
        return np.hamming(N).astype(np.float32)
    elif window == "hann":
        return np.hanning(N).astype(np.float32)
    else:
        raise ValueError("window 必须是 'rect'、'hamming' 或 'hann'")


def enframe(x, frame_len, hop_len, window="hamming"):
    """
    分帧加窗。

    参数：
        x          : 一维语音信号
        frame_len  : 帧长，单位样点
        hop_len    : 帧移，单位样点
        window     : 'rect' / 'hamming' / 'hann'

    返回：
        frames     : 二维数组，shape = (帧数, frame_len)
        w          : 窗函数
        num_frames : 帧数
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))

    num_frames = 1 + (len(x) - frame_len) // hop_len
    if num_frames <= 0:
        num_frames = 1

    frames = np.empty((num_frames, frame_len), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        frames[i] = x[start:start + frame_len]

    w = get_window(window, frame_len)
    return frames * w, w, num_frames


# ============================================================
# 3. 短时时域特征
# ============================================================
def short_time_energy(frames):
    """短时能量：E_n = sum x_n^2(m)"""
    return np.sum(frames ** 2, axis=1)


def short_time_amplitude(frames):
    """短时平均幅度：M_n = sum |x_n(m)|"""
    return np.sum(np.abs(frames), axis=1)


def short_time_zcr(frames):
    """
    短时过零率。
    先对每帧去直流，再统计相邻样点符号变化次数，并归一化。
    """
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    signs = np.sign(frames)
    signs[signs == 0] = 1.0

    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float32)

    zcr = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2.0 * (frames.shape[1] - 1))
    return zcr


# ============================================================
# 4. 双门限端点检测
# ============================================================
def double_threshold_endpoint(x, sr,
                              frame_len=0.025, hop_len=0.010,
                              energy_high_ratio=0.15,
                              energy_low_ratio=0.03,
                              zcr_ratio=0.15,
                              min_speech_dur=0.10):
    """
    基于短时能量和短时过零率的双门限端点检测。

    返回：
        start_sample, end_sample : 语音段起止样点
        E, Z                     : 全部帧的短时能量、过零率
        T1, T2, T3               : 低能量门限、高能量门限、过零率门限
    """
    frame_len_samples = int(sr * frame_len)
    hop_len_samples = int(sr * hop_len)

    if frame_len_samples < 2:
        frame_len_samples = 2
    if hop_len_samples < 1:
        hop_len_samples = 1

    frames, _, _ = enframe(x, frame_len_samples, hop_len_samples, window="hamming")
    E = short_time_energy(frames)
    Z = short_time_zcr(frames)

    # 用能量最低的 15% 帧估计背景噪声
    n_noise = max(1, int(0.15 * len(E)))
    noise_idx = np.argsort(E)[:n_noise]
    noise_E = float(np.mean(E[noise_idx]))
    noise_Z = float(np.mean(Z[noise_idx]))

    max_E = float(np.max(E))
    max_Z = float(np.max(Z))

    # 双门限
    T2 = noise_E + energy_high_ratio * (max_E - noise_E)   # 高能量门限
    T1 = noise_E + energy_low_ratio * (max_E - noise_E)    # 低能量门限
    T3 = noise_Z + zcr_ratio * (max_Z - noise_Z)           # 过零率门限

    # 高门限初判
    high = E > T2
    segments = []
    in_seg = False
    start = 0

    for i, h in enumerate(high):
        if h and not in_seg:
            start = i
            in_seg = True
        elif (not h) and in_seg:
            segments.append([start, i - 1])
            in_seg = False

    if in_seg:
        segments.append([start, len(high) - 1])

    # 合并间隔小于 50ms 的段
    max_gap = max(1, int(0.05 * sr / hop_len_samples))
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            if seg[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)

    final_segments = []
    min_frames = max(1, int(min_speech_dur * sr / hop_len_samples))
    max_extend = max(1, int(0.20 * sr / hop_len_samples))  # 过零率扩展最多 200ms

    for s, e in merged:
        # 用低能量门限向左右扩展
        while s > 0 and E[s - 1] > T1:
            s -= 1
        while e < len(E) - 1 and E[e + 1] > T1:
            e += 1

        # 用过零率门限继续扩展，但限制最大扩展帧数
        cnt = 0
        while s > 0 and Z[s - 1] > T3 and cnt < max_extend:
            s -= 1
            cnt += 1

        cnt = 0
        while e < len(Z) - 1 and Z[e + 1] > T3 and cnt < max_extend:
            e += 1
            cnt += 1

        if e - s + 1 >= min_frames:
            final_segments.append([s, e])

    if not final_segments:
        return 0, len(x), E, Z, T1, T2, T3

    # 孤立字一般只有一个主要语音段，取最长段
    s, e = max(final_segments, key=lambda seg: seg[1] - seg[0])
    start_sample = s * hop_len_samples
    end_sample = min(len(x), e * hop_len_samples + frame_len_samples)

    return start_sample, end_sample, E, Z, T1, T2, T3


# ============================================================
# 5. 时域特征提取
# ============================================================
def extract_time_domain_features(x, sr, return_info=False):
    """
    对单个语音文件提取时域特征。

    流程：
        端点检测 -> 裁剪语音段 -> 分帧加窗 -> 短时能量/平均幅度/过零率
        -> 统计均值、标准差、最大、最小、中位数、25%分位、75%分位
    """
    start, end, E_det, Z_det, T1, T2, T3 = double_threshold_endpoint(x, sr)

    x_voiced = x[start:end]
    if len(x_voiced) < int(0.01 * sr):
        x_voiced = x

    frame_len_samples = int(sr * 0.025)
    hop_len_samples = int(sr * 0.010)

    frames, _, _ = enframe(x_voiced, frame_len_samples, hop_len_samples, window="hamming")

    E = short_time_energy(frames)
    M = short_time_amplitude(frames)
    Z = short_time_zcr(frames)

    feats = []
    for seq in (E, M, Z):
        seq = np.asarray(seq, dtype=np.float32)
        feats.extend([
            float(np.mean(seq)),
            float(np.std(seq)),
            float(np.max(seq)),
            float(np.min(seq)),
            float(np.median(seq)),
            float(np.percentile(seq, 25)),
            float(np.percentile(seq, 75)),
        ])

    feats.append(float(len(E)))                  # 帧数
    feats.append(float(end - start) / float(sr)) # 语音段时长
    feats = np.array(feats, dtype=np.float32)

    if return_info:
        info = {
            "start": start,
            "end": end,
            "E_det": E_det,
            "Z_det": Z_det,
            "T1": T1,
            "T2": T2,
            "T3": T3,
            "E": E,
            "M": M,
            "Z": Z,
            "frames": frames,
        }
        return feats, info

    return feats


# ============================================================
# 6. 加载数据集
# ============================================================
def load_dataset(data_dir):
    """加载 data/0/*.wav ... data/9/*.wav，并提取特征。"""
    X, y, files = [], [], []

    for label in range(10):
        folder = os.path.join(data_dir, str(label))
        if not os.path.isdir(folder):
            print(f"警告：目录不存在 {folder}")
            continue

        wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        for f in wav_files:
            try:
                sr, x = read_wav(f)
                feat = extract_time_domain_features(x, sr)
                X.append(feat)
                y.append(label)
                files.append(f)
            except Exception as e:
                print(f"读取/处理失败 {f}: {e}")

    if len(X) == 0:
        raise RuntimeError("未找到任何 wav 文件，请按 data/0/*.wav ... data/9/*.wav 准备数据。")

    return np.vstack(X), np.array(y), files


# ============================================================
# 7. 训练与评估
# ============================================================
def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """划分训练集/测试集，对比多个分类器。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifiers = {
        "kNN(k=3)": KNeighborsClassifier(n_neighbors=3),
        "SVM(RBF)": SVC(kernel="rbf", C=10.0, gamma="scale"),
        "决策树": DecisionTreeClassifier(random_state=random_state),
        "朴素贝叶斯": GaussianNB(),
        "Fisher线性判别": LinearDiscriminantAnalysis(),
    }

    results = {}
    best_name, best_acc, best_clf, best_cm = None, -1.0, None, None

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))

        results[name] = acc

        print(f"\n分类器：{name}")
        print(f"准确率：{acc:.4f}")
        print("混淆矩阵：")
        print(cm)
        print("分类报告：")
        print(classification_report(y_test, y_pred, labels=np.arange(10), zero_division=0))

        if acc > best_acc:
            best_name, best_acc, best_clf, best_cm = name, acc, clf, cm

    print("\n===== 分类器对比 =====")
    for name, acc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name}: {acc:.4f}")

    print(f"\n最佳分类器：{best_name}，准确率：{best_acc:.4f}")

    return best_name, best_clf, scaler, best_cm, results


# ============================================================
# 8. 可视化
# ============================================================
def plot_windows(frame_len=400):
    """绘制三种窗函数。"""
    n = np.arange(frame_len)
    plt.figure(figsize=(10, 4))
    plt.plot(n, np.ones(frame_len), label="矩形窗")
    plt.plot(n, np.hamming(frame_len), label="汉明窗")
    plt.plot(n, np.hanning(frame_len), label="海宁窗")
    plt.title("三种窗函数对比")
    plt.xlabel("样点")
    plt.ylabel("幅度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_framing(x, sr, frame_len=0.025, hop_len=0.010, n_show=5):
    """绘制分帧示意图。"""
    fl = int(sr * frame_len)
    hl = int(sr * hop_len)
    t = np.arange(len(x)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(t, x, color="k", linewidth=0.8)

    for i in range(n_show):
        s = i * hl / sr
        e = s + fl / sr
        plt.axvspan(s, e, alpha=0.15)

    plt.title("语音分帧示意图（交叠分帧）")
    plt.xlabel("时间/s")
    plt.ylabel("幅度")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_analysis(x, sr, info, title="语音时域分析"):
    """绘制波形、端点检测、短时能量、过零率和短时特征。"""
    start, end = info["start"], info["end"]
    E_det, Z_det = info["E_det"], info["Z_det"]
    T1, T2, T3 = info["T1"], info["T2"], info["T3"]

    hop_len = 0.010
    t = np.arange(len(x)) / sr
    t_frame = np.arange(len(E_det)) * hop_len

    plt.figure(figsize=(12, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t, x, linewidth=0.8)
    plt.axvline(start / sr, color="r", linestyle="--", label="起点")
    plt.axvline(end / sr, color="r", linestyle="--", label="终点")
    plt.title(title + "：波形与端点检测")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(t_frame, E_det, label="短时能量")
    plt.axhline(T2, color="r", linestyle="--", label="高门限 T2")
    plt.axhline(T1, color="g", linestyle="--", label="低门限 T1")
    plt.title("短时能量与双门限")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t_frame, Z_det, label="短时过零率")
    plt.axhline(T3, color="r", linestyle="--", label="过零率门限 T3")
    plt.title("短时过零率")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    E, M, Z = info["E"], info["M"], info["Z"]
    t2 = np.arange(len(E)) * hop_len
    plt.plot(t2, E / (np.max(E) + 1e-12), label="短时能量(归一化)")
    plt.plot(t2, M / (np.max(M) + 1e-12), label="短时平均幅度(归一化)")
    plt.plot(t2, Z / (np.max(Z) + 1e-12), label="短时过零率(归一化)")
    plt.title("裁剪语音段的短时时域特征")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, title="混淆矩阵"):
    """绘制混淆矩阵。"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    labels = np.arange(10)
    plt.xticks(labels, labels)
    plt.yticks(labels, labels)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. 录音（可选）
# ============================================================
def record_wav(filename, duration=1.0, sr=16000):
    """使用 sounddevice 录音并保存为 wav。"""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError:
        print("录音需要 sounddevice 和 scipy：pip install sounddevice scipy")
        return

    print(f"开始录音 {duration}s ...")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    write(filename, sr, rec)
    print(f"已保存：{filename}")


# ============================================================
# 10. 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数字信号处理实验一：基于时域分析技术的语音识别")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录，子目录为 0-9")
    parser.add_argument("--analyze", type=str, default=None, help="分析单个 wav 文件并绘图")
    parser.add_argument("--plot", action="store_true", help="是否绘图")
    parser.add_argument("--record", type=str, default=None, help="录音保存路径")
    parser.add_argument("--duration", type=float, default=1.0, help="录音时长/秒")

    args = parser.parse_args()

    # 录音模式
    if args.record:
        record_wav(args.record, duration=args.duration)
        return

    # 单文件分析模式
    if args.analyze:
        sr, x = read_wav(args.analyze)
        feat, info = extract_time_domain_features(x, sr, return_info=True)

        print(f"文件：{args.analyze}")
        print(f"采样率：{sr} Hz")
        print(f"语音段起点/终点：{info['start']} / {info['end']} 样点")
        print(f"特征向量长度：{len(feat)}")

        if args.plot:
            plot_windows()
            plot_framing(x, sr)
            plot_analysis(x, sr, info, title=os.path.basename(args.analyze))
        return

    # 训练与识别模式
    X, y, files = load_dataset(args.data_dir)
    print(f"共加载 {len(y)} 个样本，特征维度：{X.shape[1]}")

    best_name, best_clf, scaler, best_cm, results = train_and_evaluate(X, y)

    if args.plot:
        plot_confusion_matrix(best_cm, title=f"最佳分类器 {best_name} 混淆矩阵")


if __name__ == "__main__":
    main()