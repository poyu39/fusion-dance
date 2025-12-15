import os
import re
from pathlib import Path

# 指定資料夾路徑
folder_path = "data/Pokemon/original_data"  # 改成你的資料夾路徑

# 尋找所有 PNG 檔案並提取數字前綴
png_files = Path(folder_path).glob("*.png")
numbers = []

for file in png_files:
    match = re.match(r"(\d+)_", file.name)
    if match:
        numbers.append(int(match.group(1)))

if numbers:
    numbers.sort()
    print(f"總共找到 {len(numbers)} 個 PNG 檔案")
    print(f"數字範圍: {min(numbers):03d} ~ {max(numbers):03d}")
    print(f"缺少的編號: {set(range(min(numbers), max(numbers) + 1)) - set(numbers)}")
else:
    print("沒有找到符合格式的 PNG 檔案")