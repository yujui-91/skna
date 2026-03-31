import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_classif
from scipy.stats import skew, entropy
from scipy.signal import welch
from tqdm import tqdm

# ==========================================
# 1. 功能函式定義
# ==========================================
def get_metrics(y_true, y_pred):
    """計算並回傳 [Accuracy, Sensitivity, Specificity, MCC]"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    denom = tp + tn + fp + fn
    acc = (tp + tn) / denom if denom > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc_den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0
    return [acc, sens, spec, mcc]

def evaluate_predictions(y_true, y_prob, groups, threshold=0.5):
    """統一處理 Window-level 與 Subject-level 的指標計算"""
    win_metrics = get_metrics(y_true, (y_prob >= threshold).astype(int))
    
    # 計算 Subject-level
    df = pd.DataFrame({'ID': groups, 'Actual': y_true, 'Prob': y_prob}).groupby('ID').mean()
    sub_metrics = get_metrics(df['Actual'], (df['Prob'] >= threshold).astype(int))
    
    return win_metrics, sub_metrics, df

def load_data_with_groups(mace_path, non_mace_path):
    X, y, groups = [], [], []
    for path, label in [(mace_path, 1), (non_mace_path, 0)]:
        if not os.path.exists(path): continue
        for file in [f for f in os.listdir(path) if f.endswith('.npy')]:
            data = np.load(os.path.join(path, file))[:, 1:]
            X.append(data)
            y.append(np.full(len(data), label))
            groups.extend([file] * len(data))
    return np.vstack(X), np.concatenate(y), np.array(groups)

def extract_advanced_features(X_raw, fs=10000):
    rms = np.sqrt(np.mean(X_raw**2, axis=1))
    sk = skew(X_raw, axis=1)
    ptp = np.ptp(X_raw, axis=1)
    ent = np.apply_along_axis(lambda x: entropy(np.histogram(x, bins=10)[0] + 1e-10), 1, X_raw)
    
    diff_X = np.diff(X_raw, axis=1)
    n, n_delta = X_raw.shape[1], np.sum(diff_X[:, 1:] * diff_X[:, :-1] < 0, axis=1)
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))

    hj_act = np.var(X_raw, axis=1)
    d1, d2 = np.diff(X_raw, axis=1), np.diff(diff_X, axis=1)
    var_d1, var_d2 = np.var(d1, axis=1), np.var(d2, axis=1)
    hj_mob = np.sqrt(var_d1 / (hj_act + 1e-10))
    hj_comp = np.sqrt(var_d2 / (var_d1 + 1e-10)) / (hj_mob + 1e-10)

    freqs, psd = welch(X_raw, fs=fs, nperseg=256, axis=1)
    band_500_750 = np.mean(psd[:, (freqs >= 500) & (freqs <= 750)], axis=1)
    centroid = np.sum(psd * freqs, axis=1) / (np.sum(psd, axis=1) + 1e-10)
    
    return np.column_stack((rms, sk, ptp, ent, pfd, hj_act, hj_mob, band_500_750, centroid))

feature_names = ['RMS', 'Skewness', 'PTP', 'Entropy', 'PFD', 'Hjorth_Activity', 'Hjorth_Mobility', 'Band_500_750Hz', 'Centroid']

# ==========================================
# 2. 資料載入與特徵提取
# ==========================================
base_path = r"D:\M143020071\raw_data_result\iSKNA_signal\ch1\sr10000_500_1000_MI_1000pts_win60s_step1s"
X_raw, y, groups = load_data_with_groups(os.path.join(base_path, "mace"), os.path.join(base_path, "non_mace"))
X_features = extract_advanced_features(X_raw)

# 全域 F-Score
f_scores, _ = f_classif(X_features, y)
f_score_df = pd.DataFrame({'Feature': feature_names, 'F_Score': f_scores}).sort_values(by='F_Score', ascending=False)

# 建立共用模型 Pipeline (自動處理標準化)
clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced', C=0.01))

# ==========================================
# 3. 訓練模式 A: Group-based Split (70/30)
# ==========================================
train_ids, test_ids = train_test_split(np.unique(groups), test_size=0.3, random_state=42)
train_mask, test_mask = np.isin(groups, train_ids), np.isin(groups, test_ids)

clf.fit(X_features[train_mask], y[train_mask])
y_prob_split = clf.predict_proba(X_features[test_mask])[:, 1]
me_split_w, me_split_s, _ = evaluate_predictions(y[test_mask], y_prob_split, groups[test_mask])

# ==========================================
# 4. 訓練模式 B: Leave-One-Subject-Out (LOSO)
# ==========================================
logo = LeaveOneGroupOut()
loso_probs = np.zeros(len(y))
loso_weights = []

print(f"\n正在啟動 LOSO 交叉驗證 (總共 {logo.get_n_splits(groups=groups)} 個受試者)...")
for train_idx, test_idx in tqdm(logo.split(X_features, y, groups), total=logo.get_n_splits(groups=groups), desc="LOSO"):
    clf.fit(X_features[train_idx], y[train_idx])
    loso_probs[test_idx] = clf.predict_proba(X_features[test_idx])[:, 1]
    loso_weights.append(clf.named_steps['logisticregression'].coef_[0])

me_loso_w, me_loso_s, sub_loso_df = evaluate_predictions(y, loso_probs, groups)

# ==========================================
# 5. 最終報表顯示 
# ==========================================
print(f"\n{'='*65}\n{'Performance Comparison':^20} | {'Split (70/30)':^18} | {'LOSO (Cross)':^18}\n{'-'*65}")
print(f"{'Win Accuracy':<20} | {me_split_w[0]:^18.4f} | {me_loso_w[0]:^18.4f}")
print(f"{'Sub Accuracy':<20} | {me_split_s[0]:^18.4f} | {me_loso_s[0]:^18.4f}")
print(f"{'Sub MCC':<20} | {me_split_s[3]:^18.4f} | {me_loso_s[3]:^18.4f}\n{'='*65}")

print("\n=== [1] 特徵價值 (F-Score) ===\n", f_score_df.to_string(index=False))

print("\n=== [2] 特徵貢獻度比較 (Weights) ===")
weight_df = pd.DataFrame({'Feature': feature_names, 'Split_Weight': clf.named_steps['logisticregression'].coef_[0], 'LOSO_Avg_Weight': np.mean(loso_weights, axis=0)}).sort_values(by='LOSO_Avg_Weight', ascending=False)
print(weight_df.to_string(index=False))

print("\n=== [3] LOSO 混淆矩陣 (Subject-level) ===")
tn_ls, fp_ls, fn_ls, tp_ls = confusion_matrix(sub_loso_df['Actual'], (sub_loso_df['Prob'] >= 0.5).astype(int), labels=[0,1]).ravel()
print(pd.DataFrame({'predict\\actual': ['Positive', 'Negative'], 'Positive': [tp_ls, fn_ls], 'Negative': [fp_ls, tn_ls]}).to_string(index=False))

print("\n=== [4] 整體效能總結 ===")
perf_df = pd.DataFrame({'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'MCC'], 'Split_Win': me_split_w, 'Split_Sub': me_split_s, 'LOSO_Win': me_loso_w, 'LOSO_Sub': me_loso_s})
print(perf_df.to_string(index=False, float_format='{:.4f}'.format))