import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings

# 忽略警告，保持控制台干净
warnings.filterwarnings('ignore')

# 动态添加 MRWOD 所在目录到环境变量
WORK_DIR = r"F:\代码练习"
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)

from mrwod import mrwod_outlier_detection

DATA_DIR = os.path.join(WORK_DIR, "data")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

# ==========================================
# 超参数设置 (完全锁定论文 4.1.3 节的设定)
# ==========================================
LAMBDA_GRID = np.round(np.arange(0.1, 1.6, 0.1), 2).tolist()
D_GRID = [0.1]

# ==========================================
# 离散分类特征映射表 (严格遵循论文 Table 1 维度拆分)
# ==========================================
CATEGORICAL_MAPPING = {
    # 纯分类数据集
    'vote_republican_29_variant1': list(range(16)),
    'mushroom_p_365_variant1': list(range(22)),
    'chess_nowin_145_variant1': list(range(36)),
    'chess_nowin_87_variant1': list(range(36)),

    # 混合属性数据集 (极其精准的索引校对，修复了之前的特征错位)
    'creditA_plus_42_variant1': [0, 3, 4, 5, 6, 8, 9, 11, 12],
    'german_1_14_variant1': [0,2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19],
    'hepatitis_2_9_variant1': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'horse_1_12_variant1': [0, 1,2,3,4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 20,21,22 ,23, 25,26],
    'sick_sick_35_variant1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28],
    'heart270_2_16_variant1': [1, 2, 5, 6, 8, 10, 12],
    'bands_band_42_variant1': [1, 2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20],
}
def clean_dataset_name(file_name):
    name = file_name
    for s in ['.csv', '.mat', 'ori.xls - Sheet1', 'ori.xls', 'ori']:
        name = name.replace(s, '')
    return name.strip()

def run_experiments():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    if not os.path.exists(DATA_DIR):
        print(f"未找到数据文件夹: {DATA_DIR}")
        return

    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat') or f.endswith('.csv')]
    if not data_files:
        print("数据文件夹为空！")
        return

    print(f"已找到 {len(data_files)} 个数据集，启动【MRWOD 论文级对标】实验集群...")
    print(f"底层引擎：智能表头识别 | 自动少数类提取 | NaN双轨插补 | 精准特征索引\n")

    for file_name in data_files:
        dataset_name = clean_dataset_name(file_name)
        file_path = os.path.join(DATA_DIR, file_name)

        print(f"[{dataset_name}] 数据载入中...")

        try:
            X, y_raw = None, None
            auto_cat_cols = []

            # ----------------------------------
            # 1. 智能数据读取模块
            # ----------------------------------
            if file_name.endswith('.csv'):
                df_temp = pd.read_csv(file_path, header=None)

                first_row_last_val = str(df_temp.iloc[0, -1]).lower().strip()
                first_row_first_val = str(df_temp.iloc[0, 0]).lower().strip()
                is_header = False

                if first_row_last_val in ['class', 'label', 'target', 'y'] or not first_row_first_val.replace('.', '', 1).isdigit():
                    is_header = True

                if is_header:
                    df = pd.read_csv(file_path, header=0)
                else:
                    df = df_temp

                y_raw = df.iloc[:, -1].values
                X_df = df.iloc[:, :-1]

                X_df = X_df.replace(['?', 'NaN', 'NA', 'null', ''], np.nan)

                # 提取字符串特征为分类索引
                for col in X_df.columns:
                    if X_df[col].dtype == 'object':
                        try:
                            X_df[col] = X_df[col].astype(float)
                        except ValueError:
                            X_df[col] = pd.factorize(X_df[col])[0]
                            auto_cat_cols.append(X_df.columns.get_loc(col))

                X = X_df.values.astype(float)

            elif file_name.endswith('.mat'):
                mat_data = loadmat(file_path)
                valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if 'trandata' in valid_keys:
                    X = mat_data['trandata'][:, :-1]
                    y_raw = mat_data['trandata'][:, -1]
                else:
                    for k in valid_keys:
                        if k.lower() in ['x', 'data', 'features']:
                            X = mat_data[k]
                        elif k.lower() in ['y', 'label', 'class']:
                            y_raw = mat_data[k]
                if isinstance(y_raw, np.ndarray):
                    y_raw = y_raw.ravel()

            if X is None or y_raw is None:
                print("  -> 跳过: 数据读取失败或格式不符。")
                continue

            # ----------------------------------
            # 2. 核心标签二值化
            # ----------------------------------
            val_counts = pd.Series(y_raw).value_counts()
            if len(val_counts) < 2:
                print("  -> 跳过: 数据集只有一种标签，无法计算 AUC。")
                continue

            minority_class = val_counts.idxmin()
            y = (pd.Series(y_raw) == minority_class).astype(int).values

            # ----------------------------------
            # 3. 特征分类与越界保护
            # ----------------------------------
            mapped_cat_cols = CATEGORICAL_MAPPING.get(dataset_name, [])
            mapped_cat_cols = [col_idx for col_idx in mapped_cat_cols if col_idx < X.shape[1]]

            final_cat_cols = sorted(list(set(mapped_cat_cols + auto_cat_cols)))
            num_cols = [i for i in range(X.shape[1]) if i not in final_cat_cols]

            # 缺失值处理: 论文指定数值用 mean，分类用 mode
            if len(num_cols) > 0:
                imp_num = SimpleImputer(strategy='mean')
                X[:, num_cols] = imp_num.fit_transform(X[:, num_cols])

            if len(final_cat_cols) > 0:
                imp_cat = SimpleImputer(strategy='most_frequent')
                X[:, final_cat_cols] = imp_cat.fit_transform(X[:, final_cat_cols])

            if final_cat_cols:
                print(f"  -> [特征对齐] {len(final_cat_cols)} 个分类属性, {len(num_cols)} 个数值属性。")
            else:
                print(f"  -> [特征对齐] 100% 连续数值属性。")

            # ----------------------------------
            # 4. 网格搜索求最优 AUC
            # ----------------------------------
            best_auc, best_lam, best_d, best_time, best_scores = -1.0, None, None, 0.0, None
            total_search_start = time.time()

            for current_lam in LAMBDA_GRID:
                for current_d in D_GRID:
                    run_start = time.time()

                    # 调用纯净版算法核心
                    is_outlier, mrwof_scores = mrwod_outlier_detection(
                        X,
                        categorical_cols=final_cat_cols if final_cat_cols else None,
                        lam=current_lam,
                        mu=0.5,
                        d=current_d,
                        handle_missing=False,
                    )

                    current_auc = roc_auc_score(y, mrwof_scores)

                    if current_auc > best_auc:
                        best_auc = current_auc
                        best_lam = current_lam
                        best_d = current_d
                        best_scores = mrwof_scores
                        best_time = time.time() - run_start

            total_search_time = time.time() - total_search_start

            # ----------------------------------
            # 5. 保存结果
            # ----------------------------------
            results_df = pd.DataFrame({
                'opt_out_scores': best_scores,
                'opt_ROC_AUC': [best_auc] * len(best_scores),
                'opt_time': [best_time] * len(best_scores),
                'opt_lam': [best_lam] * len(best_scores),
                'opt_d': [best_d] * len(best_scores),
                'ground_truth_y': y
            })

            dir_path = os.path.join(RESULTS_DIR, dataset_name)
            if not os.path.exists(dir_path): os.makedirs(dir_path)
            save_path = os.path.join(dir_path, f"{dataset_name}.xlsx")
            results_df.to_excel(save_path, index=False)

            print(f"  -> [寻优达成] 最优 AUC: {best_auc:.4f} (当 lam={best_lam}) | 耗时: {total_search_time:.2f}s\n")

        except Exception as e:
            print(f"  -> [报错阻断] 处理 {dataset_name} 时发生异常: {e}\n")

if __name__ == "__main__":
    run_experiments()