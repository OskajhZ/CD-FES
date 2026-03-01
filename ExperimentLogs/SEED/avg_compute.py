import numpy as np

# ====================== 1. 配置核心参数 ======================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
n_clf = 3  # 每个Feature下3个分类器（LSTM、GRU、EEGNet）

# ====================== 2. 整理表格数据（严格对应表格，层级：Session→Feature→[aa, bb]） ======================
data = {
    "Session 1": {
        "Raw Sub-frequency Bands": {
            "aa": np.array([
                [51.35, 57.77, 51.35, 48.36],  # LSTM
                [52.42, 50.01, 52.42, 49.05],  # GRU
                [51.35, 57.77, 51.35, 48.36]   # EEGNet
            ]),
            "bb": np.array([
                [5.16, 4.08, 5.16, 5.67],  # LSTM
                [4.77, 5.57, 4.77, 5.15],  # GRU
                [5.16, 4.08, 5.16, 5.67]   # EEGNet
            ])
        },
        "FES": {
            "aa": np.array([
                [52.92, 52.80, 52.92, 51.69],  # LSTM
                [50.82, 50.23, 50.82, 49.50],  # GRU
                [53.81, 56.36, 53.81, 51.10]   # EEGNet
            ]),
            "bb": np.array([
                [6.94, 7.11, 6.94, 7.16],  # LSTM
                [6.51, 6.88, 6.51, 6.67],  # GRU
                [7.37, 6.41, 7.37, 7.93]   # EEGNet
            ])
        }
    },
    "Session 2": {
        "Raw Sub-frequency Bands": {
            "aa": np.array([
                [58.99, 61.16, 58.99, 57.64],  # LSTM
                [56.04, 57.04, 56.04, 53.41],  # GRU
                [51.16, 58.71, 51.16, 48.25]   # EEGNet
            ]),
            "bb": np.array([
                [4.23, 4.43, 4.23, 4.32],  # LSTM
                [4.64, 5.10, 4.64, 4.89],  # GRU
                [4.94, 4.48, 4.94, 5.34]   # EEGNet
            ])
        },
        "FES": {
            "aa": np.array([
                [55.89, 59.10, 55.89, 54.13],  # LSTM
                [58.38, 60.30, 58.38, 56.99],  # GRU
                [57.30, 61.99, 57.30, 52.98]   # EEGNet
            ]),
            "bb": np.array([
                [4.27, 3.77, 4.27, 4.41],  # LSTM
                [4.93, 4.77, 4.93, 5.06],  # GRU
                [4.78, 5.01, 4.78, 5.42]   # EEGNet
            ])
        }
    },
    "Session 3": {
        "Raw Sub-frequency Bands": {
            "aa": np.array([
                [56.73, 56.25, 56.73, 55.10],  # LSTM
                [53.80, 53.69, 53.80, 51.88],  # GRU
                [55.18, 60.09, 55.18, 52.36]   # EEGNet
            ]),
            "bb": np.array([
                [4.20, 4.62, 4.20, 4.55],  # LSTM
                [3.87, 4.06, 3.87, 4.20],  # GRU
                [4.81, 4.72, 4.81, 5.65]   # EEGNet
            ])
        },
        "FES": {
            "aa": np.array([
                [59.06, 59.37, 59.06, 57.85],  # LSTM
                [60.05, 60.30, 60.05, 58.71],  # GRU
                [62.08, 61.83, 62.08, 59.76]   # EEGNet
            ]),
            "bb": np.array([
                [4.92, 4.90, 4.92, 5.06],  # LSTM
                [4.57, 4.47, 4.57, 4.72],  # GRU
                [4.36, 4.72, 4.36, 4.68]   # EEGNet
            ])
        }
    }
}

# ====================== 3. 核心计算函数（沿用之前的逻辑，适配n_clf=3） ======================
def compute_mean_se(aa_array, bb_array, n_clf):
    """
    计算综合均值和综合标准误：
    - 综合均值：n_clf个分类器aa的算术平均
    - 综合标准误：(1/n_clf) * √(sum(bb_i²))
    """
    # 计算每列综合均值
    mean_total = aa_array.mean(axis=0)
    # 计算每列综合标准误
    se_total = np.sqrt((bb_array ** 2).sum(axis=0)) / n_clf
    return mean_total, se_total

# ====================== 4. 遍历计算所有Average行并输出LaTeX格式结果 ======================
print("=" * 100)
print("LaTeX格式结果（可直接复制粘贴到表格中）")
print("=" * 100)

for session_name, session_data in data.items():
    print(f"\n🔹 {session_name}")
    for feature_name, feature_data in session_data.items():
        print(f"  └── {feature_name} - Average行（LaTeX可直接粘贴）")
        # 获取当前Feature的aa和bb数组
        aa = feature_data["aa"]
        bb = feature_data["bb"]
        # 计算结果
        avg_mean, avg_se = compute_mean_se(aa, bb, n_clf)
        # 格式化输出：LaTeX格式 均值$\pm$标准误，无多余空格
        for metric, m, s in zip(metrics, avg_mean, avg_se):
            latex_format = f"{m:.2f}$\\pm${s:.2f}"  # 关键修改：LaTeX兼容的±格式
            print(f"      {metric:12s}: {latex_format}")
    print("-" * 80)
