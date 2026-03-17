import pandas as pd
import matplotlib.pyplot as plt

# 讀取 .tsv 文件
file_path = "F:/junkai/SKNA/tsv_data/SKNA_features_beat_sym_0.5_150hz_l1.44s_1train6test/train/skna1.tsv"
data = pd.read_csv(file_path, delimiter='/t')

# 提取第一行的訊號
first_row = data.iloc[0]

# 獲取標籤和訊號
label = first_row[0]
signal = first_row[1:]

# 繪製信號
plt.figure(figsize=(10, 5))
plt.plot(signal)
plt.title(f'Signal from the first row (Label: {label})')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.grid(True)
plt.show()
