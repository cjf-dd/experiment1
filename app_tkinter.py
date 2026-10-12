import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import sys
import os


# 👇 新增：兼容 PyInstaller 打包的资源路径函数
def resource_path(relative_path):
    """ 获取资源绝对路径，兼容 PyInstaller 打包 """
    if hasattr(sys, '_MEIPASS'):
        # 打包后，返回临时解压目录
        return os.path.join(sys._MEIPASS, relative_path)
    # 未打包时，返回当前目录
    return os.path.join(os.path.abspath("."), relative_path)


# 使用 resource_path 加载模型
try:
    # 👇 修改：使用 resource_path("best_model.pkl")
    with open(resource_path("best_model.pkl"), "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]
except FileNotFoundError:
    tk.messagebox.showerror("错误", "找不到 best_model.pkl，请先运行训练脚本生成模型！")
    exit()


def select_and_recognize():
    file_path = filedialog.askopenfilename(title="选择音频文件", filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return

    try:
        # 从 experiment1.py 导入函数
        from experiment1 import read_wav, extract_time_domain_features
        sr, x = read_wav(file_path)
        feat = extract_time_domain_features(x, sr)
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        prediction = model.predict(feat_scaled)[0]

        result_label.config(text=f"识别结果：数字 【{prediction}】", fg="green")
    except Exception as e:
        messagebox.showerror("错误", f"识别失败：{str(e)}")


# 创建窗口
root = tk.Tk()
root.title("语音识别 App")
root.geometry("400x250")

tk.Label(root, text="🎤 孤立字语音识别系统", font=("微软雅黑", 16)).pack(pady=20)
tk.Button(root, text="📂 选择音频文件并识别", font=("微软雅黑", 12), command=select_and_recognize).pack(pady=10)
result_label = tk.Label(root, text="等待识别...", font=("微软雅黑", 14), fg="gray")
result_label.pack(pady=20)

root.mainloop()