import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 全局参数设置（Nature风格核心）--------------------------
font_coef = 2

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10*font_coef,
    'axes.labelsize': 11*font_coef,
    'axes.titlesize': 12*font_coef,
    'xtick.labelsize': 9*font_coef,
    'ytick.labelsize': 9*font_coef,
    'legend.fontsize': 9*font_coef,
    
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    
    'figure.subplot.hspace': 0.35,
    'figure.subplot.wspace': 0.2,
})

# -------------------------- 数据准备（后续可直接替换此处数据）--------------------------
sessions = [
    "Session 1", 
    "Session 2", 
    "Session 3" 
]
raw = [48.59, 53.10, 53.11]
raw_err = [3.18, 2.81, 2.79]
FES = [50.76, 54.70, 58.77]
FES_err = [4.20, 2.88, 2.78]

# -------------------------- 风格参数优化 --------------------------
acc_color = '#A0C9E5' 
FES_color = '#256EA2'
bar_width = 0.35  # 分组柱状图柱宽,避免重叠
text_offset = 0.5  # 数值标注在标准误线上方的偏移量，可根据图表调整

# 误差棒配置
error_params = {
    'ecolor': 'black',
    'elinewidth': 1.0,
    'capsize': 4,
    'capthick': 1.0,
}

# -------------------------- 图表绘制 --------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 计算并列柱子的偏移x坐标
x = np.arange(len(sessions))
x_acc = x - bar_width / 2
x_FES = x + bar_width / 2

# 绘制Accuracy和F1 Score柱子
bars_acc = ax.bar(
    x=x_acc,
    height=raw,
    width=bar_width,
    color=acc_color,
    alpha=0.7,
    edgecolor='white',
    linewidth=0.6,
    yerr=raw_err,
    error_kw=error_params,
    label='Raw Sub-frequency Bands'
)

bars_FES = ax.bar(
    x=x_FES,
    height=FES,
    width=bar_width,
    color=FES_color,
    alpha=0.7,
    edgecolor='white',
    linewidth=0.6,
    yerr=FES_err,
    error_kw=error_params,
    label='FES'
)

# -------------------------- 新增：柱子数值标注（标准误线上方）--------------------------
# 标注 Raw 组的数值
for i, (x_pos, height, err) in enumerate(zip(x_acc, raw, raw_err)):
    # 标注纵坐标 = 柱子高度 + 标准误 + 偏移量（避免和标准误线重叠）
    y_pos = height + err + text_offset
    # 添加文本标注
    ax.text(
        x=x_pos,
        y=y_pos,
        s=f'{height:.2f}',  # 保留2位小数，和原始数据格式一致
        ha='center',  # 水平居中
        va='bottom',  # 垂直靠下（对齐标注底部和y_pos）
        fontsize=8*font_coef,  # 标注字体大小，适配全局风格
        family='Times New Roman'
    )

# 标注 FES 组的数值
for i, (x_pos, height, err) in enumerate(zip(x_FES, FES, FES_err)):
    y_pos = height + err + text_offset
    ax.text(
        x=x_pos,
        y=y_pos,
        s=f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=8*font_coef,
        family='Times New Roman'
    )

# -------------------------- 格式调整（核心：自定义规则自适应y轴）--------------------------
ax.set_ylabel("F1 Score (%)", labelpad=8)
#ax.set_title("F1 Score Comparision between Raw Sub-Bands and FES on SEED Dataset", pad=10, fontweight='normal')
ax.set_xticks(x)
ax.set_xticklabels(sessions, rotation=0, ha='center')
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=False)

# -------------------------- 核心：按指定规则计算y轴范围（适配任意替换数据）--------------------------
# 1. 收集所有「均值±标准误」的数据点,确保误差棒完整覆盖
all_lower = []  # 存储所有（均值-标准误）
all_upper = []  # 存储所有（均值+标准误）
all_lower.extend([a - ae for a, ae in zip(raw, raw_err)])
all_lower.extend([f - fe for f, fe in zip(FES, FES_err)])
all_upper.extend([a + ae for a, ae in zip(raw, raw_err)])
all_upper.extend([f + fe for f, fe in zip(FES, FES_err)])

# 2. 计算关键值：「均值-标准误」的最小值、「均值+标准误」的最大值
y_min_raw = min(all_lower)
y_max_raw = max(all_upper)

margin = 2
# 3. 按规则取整：向下取5（y_min）、向上取5（y_max）
y_min = np.floor(y_min_raw / margin) * margin  # 向下取最近的5的倍数（如40.2→40,38.9→35）
y_max = np.ceil(y_max_raw / margin) * margin   # 向上取最近的5的倍数（如58.3→60,55.1→60）

# 【微调】y轴上限增加少量空间，避免标注超出图表（适配数值标注）
y_max += text_offset + 1.5
# 4. 设置y轴范围
ax.set_ylim(y_min, y_max)

# -------------------------- 全局优化 --------------------------
plt.tight_layout()
fig.subplots_adjust(top=0.92)

# 保存图片（后续换数据后直接运行即可生成新图）
plt.savefig("F1_cmp.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig("F1_cmp.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
