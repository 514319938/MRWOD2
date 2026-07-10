import os
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import warnings

# 忽略警告，保持控制台整洁


def evaluation_outlier(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return fpr * 100, tpr * 100


# ======================= 基础路径设置 ============================
# 固定工作目录为你的实验目录
WORK_DIR = r"F:\代码练习"
data_dir = os.path.join(WORK_DIR, 'data')

# [关键修复] 修改为你的对比算法存放的真实路径
result_root_dir = r'F:\Experiental_results'

save_dir = os.path.join(WORK_DIR, 'results', '_figures', 'ROC')
os.makedirs(save_dir, exist_ok=True)

# ======================= 方法与数据集定义 ============================
# 目标算法 MRWOD 放在最后，这样画出来的线会在最上层，不会被遮挡
outlier_method = [
    'MIX' ,'LPOD' ,'ITB' ,'VARE' ,'ODGrCR', 'FGAS' ,'VOS' ,'WNINOD' ,'WFRDA'
]
outlier_method_name = outlier_method.copy()

# 线条样式与标记点
line_styles = ['r-', 'g-', 'c-', 'm-', 'b-', 'y-',
               'r-', 'g-', 'm-', 'b-', 'k-', 'y-']
marker_list = ["*", "x", "d", "+", "p", "^", "o", "o", ">", "s", "", ">", "h", "D", "v"]

# 替换为你实际参与实验的数据集名称
datasets = [
    'creditA_plus_42_variant1', 'german_1_14_variant1',
    'heart270_2_16_variant1', 'hepatitis_2_9_variant1',
    'horse_1_12_variant1', 'ionosphere_b_24_variant1',
   'vote_republican_29_variant1',
    'yeast_ERL_5_variant1'
]

# ======================= 主循环 ============================
for data_nameori in datasets:
    # 1. 动态加载真实标签 (支持 .csv 和 .mat)
    labels = None
    csv_path = os.path.join(data_dir, f'{data_nameori}.csv')
    mat_path = os.path.join(data_dir, f'{data_nameori}.mat')

    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            y_raw = df.iloc[:, -1]
            val_counts = y_raw.value_counts()
            if len(val_counts) >= 2:
                minority_class = val_counts.idxmin()
                labels = (y_raw == minority_class).astype(int).values
            else:
                labels = np.zeros(len(y_raw))
        elif os.path.exists(mat_path):
            data_mat = scio.loadmat(mat_path)
            valid_keys = [k for k in data_mat.keys() if not k.startswith('__')]
            if 'trandata' in valid_keys:
                labels = data_mat['trandata'][:, -1].flatten()
            else:
                for k in valid_keys:
                    if k.lower() in ['y', 'label', 'class']:
                        labels = data_mat[k].flatten()
                        break

        if labels is None:
            print(f"❌ 数据 {data_nameori} 缺少标签或文件不存在")
            continue
    except Exception as e:
        print(f"❌ 加载数据崩溃 {data_nameori}: {e}")
        continue

    # 开始绘图
    plt.figure(figsize=(6, 5))
    plot_success_count = 0

    for i, method in enumerate(outlier_method):
        method_file_path = None

        # 2. 特殊处理 MRWOD，直接从刚才跑出的 results 文件夹里读取 .xlsx
        if method == 'MRWOD':
            mrwod_path_xlsx = os.path.join(WORK_DIR, 'results', data_nameori, f'{data_nameori}.xlsx')
            if os.path.exists(mrwod_path_xlsx):
                method_file_path = mrwod_path_xlsx
            else:
                print(f"⚠️ 缺少方法结果文件:{data_nameori}_{method} (预期路径: {mrwod_path_xlsx})")
                continue
        else:
            # 遍历对比算法目录查找匹配文件
            for root, dirs, files in os.walk(result_root_dir):
                for file in files:
                    # 匹配规则容错处理 (不区分大小写匹配文件前缀)
                    if (file.lower() == f'{data_nameori.lower()}_{method.lower()}.xlsx'
                            or file.lower() == f'{data_nameori.lower()}_{method.lower()}.mat'
                            or file.lower() == f'{data_nameori.lower()}_{method.lower()}.xls'):
                        method_file_path = os.path.join(root, file)
                        break
                if method_file_path:
                    break

        if not method_file_path or not os.path.isfile(method_file_path):
            print(f"⚠️ 缺少对比方法文件:{data_nameori}_{method}")
            continue

        # 3. 读取异常分数
        try:
            if method_file_path.endswith('.mat'):
                result_mat = scio.loadmat(method_file_path)
                found_key = None
                for key in result_mat.keys():
                    if not key.startswith('__'):
                        found_key = key
                        break
                scores = result_mat[found_key][:, 0]

            elif method_file_path.endswith('.xlsx') or method_file_path.endswith('.xls'):
                result_df = pd.read_excel(method_file_path)
                # 针对 MRWOD，精确提取 opt_out_scores 列
                if method == 'MRWOD' and 'opt_out_scores' in result_df.columns:
                    scores = result_df['opt_out_scores'].values
                else:
                    scores = result_df.iloc[:, 0].values
        except Exception as e:
            print(f"❌ 读取文件崩溃 {method_file_path}: {e}")
            continue

        # 确保标签长度和分数长度一致（以防被其他算法截断）
        if len(scores) != len(labels):
            print(f"⚠️ 长度不匹配: {data_nameori}_{method} (分数长度:{len(scores)}, 标签长度:{len(labels)})")
            continue

        # 4. 计算并绘制 ROC
        fpr, tpr = evaluation_outlier(scores, labels)

        # 给予目标算法 MRWOD 更粗的线条，让它在对比图里更醒目
        line_w = 2.5 if method == 'MRWOD' else 1.2

        plt.plot(fpr, tpr, line_styles[i % len(line_styles)],
                 marker=marker_list[i % len(marker_list)],
                 markevery=0.1,
                 markersize=5,
                 linewidth=line_w,
                 label=outlier_method_name[i])

        plot_success_count += 1

    if plot_success_count == 0:
        print(f"⏭️ {data_nameori} 没有找到任何有效的算法结果，跳过画图。")
        plt.close()
        continue

    # 设置图表样式
    plt.xlabel('FPR (%)', fontsize=12)
    plt.ylabel('TPR (%)', fontsize=12)
    plt.xlim([0, 100])
    plt.ylim([0, 102])

    # [关键修复] framealpha=1 消除保存 eps 格式时的透明度警告
    plt.legend(loc='lower right', fontsize=9, framealpha=1.0, edgecolor='black')
    plt.grid(False)

    # 5. 保存图像
    filename = os.path.join(save_dir, f'{data_nameori}_ROC')
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f'{filename}.svg', format='svg', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f'{filename}.eps', format='eps', bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"✅ {data_nameori} ROC图绘制完成! 包含 {plot_success_count} 条曲线。")

print("\n🎉 所有 ROC 曲线画图执行完毕! 请去 F:\\代码练习\\results\\_figures\\ROC\\ 目录查看最终成图。")