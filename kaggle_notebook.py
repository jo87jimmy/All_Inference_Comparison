# %%
# ===========================
# 0. GPU 與時間監控功能
# ===========================

import time, threading, torch, subprocess, os

SESSION_LIMIT_HOURS = 20  # Kaggle 預設 GPU 時間限制
start_time = time.time()

def monitor_gpu():
    while True:
        elapsed = time.time() - start_time
        remaining = SESSION_LIMIT_HOURS * 3600 - elapsed
        remaining_h = max(0, remaining // 3600)
        remaining_m = max(0, (remaining % 3600) // 60)

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        try:
            gpu_mem_info = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", shell=True
            ).decode().strip().split("\n")[0]
            mem_used, mem_total = gpu_mem_info.split(", ")
        except Exception:
            mem_used, mem_total = "?", "?"

        print(f"[GPU監控] GPU: {gpu_name} | 記憶體: {mem_used}/{mem_total} MB | "
              f"已運行: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m | "
              f"剩餘: {int(remaining_h)}h {int(remaining_m)}m")
        if remaining <= 1800:
            print("⚠️ 剩餘時間不足30分鐘，建議停止或儲存進度！")
        time.sleep(300)

threading.Thread(target=monitor_gpu, daemon=True).start()

# %%
# ===========================
# 1. 安裝環境 (含 anomalib)
# ===========================

# Kaggle 預裝 PyTorch + CUDA，通常不需要重裝
# 新增 anomalib 以支援 PatchCore / EfficientAD 載入
!pip install -q numpy scipy scikit-learn pillow pandas tqdm
!pip install -q imgaug
!pip install -q opencv-python-headless
!pip install -q anomalib

# %%
# ===========================
# 2. 環境變數設定
# ===========================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt

# %%
# ===========================
# 3. 下載 All_Inference_Comparison 專案
# ===========================

REPO_DIR = "All_Inference_Comparison"

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/jo87jimmy/All_Inference_Comparison.git
else:
    print(f"✅ '{REPO_DIR}' 已存在，跳過 clone。")

%cd {REPO_DIR}

# %%
# ===========================
# 4. 掛載所有資料集 & Checkpoints
# ===========================

def create_symlink(src, dst):
    """建立符號連結，若舊的已存在則先刪除"""
    if os.path.lexists(dst):
        os.remove(dst)
        print(f"🗑️ 已移除舊的連結: {dst}")

    # Kaggle 資料集有時會多包一層同名字料夾，自動檢測
    dir_name = os.path.basename(dst)
    nested_src1 = os.path.join(src, dir_name)
    nested_src2 = os.path.join(src, os.path.basename(src))
    
    # 判斷真實的路徑
    if os.path.isdir(nested_src1):
        actual_src = nested_src1
    elif os.path.isdir(nested_src2):
        actual_src = nested_src2
    else:
        # 有時候上傳時只含一個資料夾但名稱不同，例如 data-mvtec 裡面只有一個 mvtec
        contents = os.listdir(src) if os.path.exists(src) else []
        if len(contents) == 1 and os.path.isdir(os.path.join(src, contents[0])):
            potential_src = os.path.join(src, contents[0])
            # 但要注意：如果裡面唯一的是 'bottle' 等物件分類，就不能再往內退一層！
            # 這裡簡單判斷：如果 contents[0] 是 MVTec 的分類之一 (如 'bottle', 'cable' 等)，就不要當作被打包的資料夾
            # 我們假設若名稱不是 bottle 等物件類別，才往下退一層。
            if contents[0] not in ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]:
                actual_src = potential_src
            else:
                actual_src = src
        else:
            actual_src = src

    if os.path.exists(actual_src):
        os.symlink(actual_src, dst)
        print(f"✅ 已成功將 {actual_src} 連結至 {dst}")
    else:
        print(f"❌ 錯誤：找不到 {actual_src} (或 {src})，請確認 Kaggle 資料集是否已正確掛載。")

# --- MVTec Dataset ---
MVTEC_PATH = "/kaggle/input/datasets/jo87jimmy/data-mvtec"
create_symlink(MVTEC_PATH, "mvtec")

# --- DRAEM Teacher Checkpoints ---
DRAEM_CP_PATH = "/kaggle/input/datasets/jo87jimmy/draem-checkpoints"
create_symlink(DRAEM_CP_PATH, "DRAEM_checkpoints")

# --- Student Model Checkpoints (我們的模型) ---
STUDENT_CP_PATH = "/kaggle/input/datasets/jo87jimmy/student-model-checkpoints"
create_symlink(STUDENT_CP_PATH, "student_model_checkpoints")

# --- PatchCore Checkpoints ---
PATCHCORE_PATH = "/kaggle/input/datasets/jo87jimmy/patchcore"  # ← 請依實際 Kaggle dataset 名稱修改
create_symlink(PATCHCORE_PATH, "PatchCore")

# --- EfficientAD Checkpoints ---
EFFICIENTAD_PATH = "/kaggle/input/datasets/jo87jimmy/efficientad"  # ← 請依實際 Kaggle dataset 名稱修改
create_symlink(EFFICIENTAD_PATH, "EfficientAD")

# %%
# ===========================
# 5. 驗證所有路徑與環境
# ===========================

print("=" * 60)
print("環境驗證")
print("=" * 60)

# 檢查 GPU
print(f"\n🖥️  GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA 版本: {torch.version.cuda}")

# 檢查所有必要資料夾
dirs_to_check = {
    "MVTec 資料集": "mvtec",
    "DRAEM Teacher": "DRAEM_checkpoints",
    "Student Model": "student_model_checkpoints",
    "PatchCore": "PatchCore",
    "EfficientAD": "EfficientAD",
}

for label, path in dirs_to_check.items():
    exists = os.path.exists(path)
    status = "✅ 存在" if exists else "❌ 不存在"
    count = len(os.listdir(path)) if exists else 0
    print(f"📁 {label} ({path}): {status} ({count} 檔案)")

# 列出 MVTec 物件類別
if os.path.exists("mvtec"):
    objs = sorted([d for d in os.listdir("mvtec") if os.path.isdir(os.path.join("mvtec", d))])
    print(f"\n📋 MVTec 物件類別 ({len(objs)} 個): {objs}")

# 檢查 anomalib
try:
    import anomalib
    print(f"\n📦 anomalib 版本: {anomalib.__version__}")
except ImportError:
    print("\n❌ anomalib 未安裝！請執行: pip install anomalib")

print("\n" + "=" * 60)

# %%
# ===========================
# 6. 測試單一類別 (e.g. bottle)
# ===========================

# obj_id=1 → bottle；也可改為其他 ID (0~14) 或 -1 (全部)
!python main.py --obj_id 1 --mvtec_root ./mvtec --n_repeat 3

# %%
# ===========================
# 7. (選擇性) 測試全部 15 類
# ===========================

# ⚠️ 全部跑完需要較長時間（約 1-2 小時），
# 建議先用單一類別確認無誤後再執行
# !python main.py --obj_id -1 --mvtec_root ./mvtec --n_repeat 3

# %%
# ===========================
# 8. 顯示結果圖表
# ===========================

import glob
from IPython.display import display, Image as IPImage

results_dir = "./comparison_results"

if os.path.exists(results_dir):
    png_files = sorted(glob.glob(f"{results_dir}/**/*.png", recursive=True))
    print(f"找到 {len(png_files)} 張結果圖表：\n")
    for png_file in png_files:
        print(f"📊 {png_file}")
        display(IPImage(filename=png_file))
        print()
else:
    print("❌ 尚未產生結果，請先執行推論比較。")

# %%
# ===========================
# 9. 顯示 JSON 結果
# ===========================

import json

json_path = os.path.join(results_dir, "all_results.json")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2, ensure_ascii=False))
else:
    print("❌ JSON 結果尚未產生。")

# %%
# ===========================
# 10. 下載結果（可選）
# ===========================

if os.path.exists(results_dir):
    !zip -r /kaggle/working/comparison_results.zip {results_dir}
    print("✅ 結果已打包至 /kaggle/working/comparison_results.zip")
    print("   可從 Kaggle Output 頁面下載。")
