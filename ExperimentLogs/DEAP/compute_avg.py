import numpy as np

# ====================== 1. 配置核心参数 ======================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
n_clf = 3  # 每个Feature下3个分类器（LSTM、GRU、EEGNet）

# ====================== 2. 整理表格数据（层级：Dimension→Feature→[aa(均值), bb(标准误)]） ======================
data = {
    "Valence": {
        "Raw Sub-frequency Bands": {
            # aa: 各分类器的均值列 [Accuracy, Precision, Recall, F1]
            "aa": np.array([
                [59.33, 73.09, 59.33, 61.05],  # LSTM
                [61.59, 74.51, 61.59, 62.63],  # GRU
                [71.67, 80.34, 71.67, 63.41]   # EEGNet
            ]),
            # bb: 各分类器的标准误列 [Accuracy, Precision, Recall, F1]
            "bb": np.array([
                [5.32, 7.28, 5.32, 6.39],  # LSTM
                [6.22, 6.95, 6.22, 6.93],  # GRU
                [7.54, 6.19, 7.54, 9.52]   # EEGNet
            ])
        },
        "FES": {
            "aa": np.array([
                [55.12, 73.39, 55.12, 59.61],  # LSTM
                [73.77, 76.22, 73.77, 68.30],  # GRU
                [70.23, 74.87, 70.23, 66.68]   # EEGNet
            ]),
            "bb": np.array([
                [3.01, 6.82, 3.01, 4.52],  # LSTM
                [6.87, 6.51, 6.87, 8.28],  # GRU
                [7.14, 6.94, 7.14, 8.42]   # EEGNet
            ])
        }
    },
    "Arousal": {
        "Raw Sub-frequency Bands": {
            "aa": np.array([
                [59.88, 75.74, 59.88, 62.41],  # LSTM
                [61.55, 77.48, 61.55, 63.02],  # GRU
                [74.38, 82.74, 74.38, 66.99]   # EEGNet
            ]),
            "bb": np.array([
                [6.15, 7.01, 6.15, 6.78],  # LSTM
                [6.11, 6.58, 6.11, 7.01],  # GRU
                [7.74, 6.00, 7.74, 9.51]   # EEGNet
            ])
        },
        "FES": {
            "aa": np.array([
                [56.07, 77.33, 56.07, 61.38],  # LSTM
                [76.66, 77.90, 76.66, 73.38],  # GRU
                [72.89, 78.50, 72.89, 70.83]   # EEGNet
            ]),
            "bb": np.array([
                [3.97, 6.76, 3.97, 4.91],  # LSTM
                [6.78, 7.20, 6.78, 7.59],  # GRU
                [7.32, 6.61, 7.32, 8.07]   # EEGNet
            ])
        }
    }
}

# ====================== 3. 核心计算函数（与示例逻辑完全一致） ======================
def compute_mean_se(aa_array, bb_array, n_clf):
    """
    计算综合均值和综合标准误：
    - 综合均值：n_clf个分类器aa的算术平均
    - 综合标准误：(1/n_clf) * √(sum(bb_i²))
    """
    # 计算每列（指标）的综合均值
    mean_total = aa_array.mean(axis=0)
    # 计算每列（指标）的综合标准误
    se_total = np.sqrt((bb_array ** 2).sum(axis=0)) / n_clf
    return mean_total, se_total

# ====================== 4. 遍历计算并输出LaTeX格式结果 ======================
print("=" * 120)
print("LaTeX格式结果（可直接复制粘贴到表格的Average行）")
print("=" * 120)

for dim_name, dim_data in data.items():
    print(f"\n🔹 {dim_name}")
    for feature_name, feature_data in dim_data.items():
        print(f"  └── {feature_name} - Average行（LaTeX可直接粘贴）")
        # 获取当前Feature的均值(aa)和标准误(bb)数组
        aa = feature_data["aa"]
        bb = feature_data["bb"]
        # 计算综合均值和标准误
        avg_mean, avg_se = compute_mean_se(aa, bb, n_clf)
        # 格式化输出（保留2位小数，适配LaTeX的±格式）
        latex_results = []
        for metric, m, s in zip(metrics, avg_mean, avg_se):
            latex_format = f"{m:.2f}$\\pm${s:.2f}"
            latex_results.append(latex_format)
            print(f"      {metric:12s}: {latex_format}")
        # 输出整行拼接结果（方便直接复制到表格）
        print(f"      整行拼接: {' & '.join(latex_results)} \\\\")
    print("-" * 100)
