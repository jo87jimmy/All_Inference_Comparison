"""
==========================================================================
  全方法推論比較腳本 — All Baseline Inference Comparison
==========================================================================
  比較對象：
    1. DRAEM (Teacher, 原始大模型)
    2. DRAEM-Student (我們的蒸餾壓縮模型)
    3. PatchCore (anomalib checkpoint, WideResNet50 backbone)
    4. EfficientAD (anomalib checkpoint, Small variant)

  統一在相同硬體環境、相同 MVTec AD 資料集下，
  一次跑出所有方法的：
    • Image-level AUROC（影像異常偵測能力）
    • 平均推論時間 (ms/image)
    • FPS (frames per second)
    • 模型參數量 & 大小
==========================================================================
"""

import os
import sys
import time
import json

# ── 修正 Windows 主控台編碼問題 (cp950 無法輸出 emoji / 特殊字元) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

# 💡 針對 PyTorch 2.6+ 修改 torch.load 預設為 weights_only=True 導致 
# 載入 anomalib Lightning checkpoints (PatchCore, EfficientAD) 失敗的問題：
# 在此進行猴子補丁 (monkey-patch)，讓未指定 weights_only 的 load 回退使用 weights_only=False。
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    # 強制覆蓋為 False，因為 pytorch_lightning 內部會顯式傳入 weights_only=True
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import numpy as np
import random
import argparse
import cv2
import glob
from collections import OrderedDict

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 非互動式後端，避免彈出視窗
import matplotlib.pyplot as plt

# ── 本地模組：DRAEM 模型架構 ──
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEM_Test_Visual_Dataset


# ===========================================================================
# 0. 通用工具函數
# ===========================================================================

def setup_seed(seed):
    """固定隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_available_gpu():
    """自動選擇記憶體使用率最低的 GPU"""
    if not torch.cuda.is_available():
        return -1
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        gpu_memory.append((i, memory_allocated))
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


def count_parameters(model):
    """計算模型的總參數量與可訓練參數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """計算模型大小 (MB)"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb


def cuda_sync():
    """安全的 CUDA 同步：僅在 CUDA 可用時才呼叫"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_empty_cache():
    """安全的 GPU 記憶體釋放"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warm_up_gpu(func, *args, n_warmup=10):
    """GPU 暖機，避免首次推論偏慢。func 應是一個可呼叫的推論函數。"""
    with torch.no_grad():
        for _ in range(n_warmup):
            func(*args)
    cuda_sync()


# ===========================================================================
# 1. 通用 MVTec 測試資料集 (同時支援 DRAEM 系列 & anomalib 系列)
# ===========================================================================

class MVTecTestDataset(torch.utils.data.Dataset):
    """
    統一的 MVTec AD 測試資料集載入器。
    回傳:
      - image_256:  resize 到 256×256, 值域 [0,1], shape (3, 256, 256)  → DRAEM 系列
      - image_224:  resize 到 256→center crop 224, ImageNet 正規化     → PatchCore
      - image_ead:  resize 到 256×256, ImageNet 正規化                  → EfficientAD
      - label:      0=good, 1=anomaly
      - mask:       ground truth mask (256×256)
    """

    # ImageNet 正規化常數
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, resize=256, center_crop=224):
        """
        Args:
            root_dir: e.g. './mvtec/bottle/test/'
            resize: 初始 resize 大小
            center_crop: PatchCore 需要的 center crop 大小
        """
        self.root_dir = root_dir
        self.resize = resize
        self.center_crop = center_crop

        # 搜集所有 test 圖片
        self.images = sorted(glob.glob(os.path.join(root_dir, "*", "*.png")))
        if len(self.images) == 0:
            # 有些資料集用 jpg
            self.images = sorted(glob.glob(os.path.join(root_dir, "*", "*.jpg")))

        # 計算 ground truth mask 所在路徑的基底
        # MVTec 結構: {obj}/test/{defect_type}/{img}.png
        #             {obj}/ground_truth/{defect_type}/{img}_mask.png
        self.gt_root = os.path.join(os.path.dirname(root_dir.rstrip("/")), "ground_truth")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        defect_type = os.path.basename(dir_path)

        # ── 讀取原始影像 (BGR → RGB) ──
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── 讀取 mask ──
        if defect_type == "good":
            label = 0
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
        else:
            label = 1
            mask_fname = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(self.gt_root, defect_type, mask_fname)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)

        # ── 產生不同前處理版本 ──

        # (A) DRAEM 系列: resize → [0,1] → CHW (⚠️ 修正: DRAEM 原始訓練使用 cv2.imread 讀取 BGR，故此處餵入 BGR 而非 RGB)
        img_256 = cv2.resize(img_bgr, (self.resize, self.resize)).astype(np.float32) / 255.0
        img_256 = np.transpose(img_256, (2, 0, 1))  # (3, 256, 256)

        # (B) PatchCore: resize → center crop 224 → ImageNet normalize
        img_pc = cv2.resize(img_rgb, (self.resize, self.resize)).astype(np.float32) / 255.0
        # center crop
        offset = (self.resize - self.center_crop) // 2
        img_pc = img_pc[offset:offset + self.center_crop, offset:offset + self.center_crop, :]
        img_pc = np.transpose(img_pc, (2, 0, 1))  # (3, 224, 224)
        for c in range(3):
            img_pc[c] = (img_pc[c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]

        # (C) EfficientAD: resize 256 → RGB [0,1] (anomalib 模型在訓練時不依賴 ImageNet Norm，而是直接餵入 [0,1] 影像)
        img_ead = cv2.resize(img_rgb, (self.resize, self.resize)).astype(np.float32) / 255.0
        img_ead = np.transpose(img_ead, (2, 0, 1))  # (3, 256, 256)

        # mask resize to 256
        mask_256 = cv2.resize(mask, (self.resize, self.resize))

        return {
            "image_256": torch.tensor(img_256, dtype=torch.float32),
            "image_224": torch.tensor(img_pc, dtype=torch.float32),
            "image_ead": torch.tensor(img_ead, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.tensor(mask_256, dtype=torch.float32),
        }


# ===========================================================================
# 2. 各方法推論器 (Inference Runner)
# ===========================================================================
# 設計原則：每個方法實作相同介面 —— load(), infer_single_batch(), get_model_info()
# 如此便可在統一的 benchmark 迴圈中公平比較。
# ===========================================================================

class DRAEMRunner:
    """DRAEM (Teacher 或 Student) 推論器"""

    def __init__(self, name, recon_path, seg_path, recon_base_width, seg_base_channels, device):
        self.name = name
        self.recon_path = recon_path
        self.seg_path = seg_path
        self.recon_base_width = recon_base_width
        self.seg_base_channels = seg_base_channels
        self.device = device
        self.recon_model = None
        self.seg_model = None

    def load(self):
        """載入 DRAEM 重建 + 分割模型權重"""
        self.recon_model = ReconstructiveSubNetwork(
            in_channels=3, out_channels=3, base_width=self.recon_base_width
        )
        self.recon_model.load_state_dict(
            torch.load(self.recon_path, map_location=self.device, weights_only=True)
        )
        self.recon_model.to(self.device).eval()

        self.seg_model = DiscriminativeSubNetwork(
            in_channels=6, out_channels=2, base_channels=self.seg_base_channels
        )
        self.seg_model.load_state_dict(
            torch.load(self.seg_path, map_location=self.device, weights_only=True)
        )
        self.seg_model.to(self.device).eval()
        return True

    def infer_batch(self, batch):
        """
        完整推論: 重建 → 分割 → anomaly score
        回傳 image-level anomaly score (標量)
        """
        img = batch["image_256"].to(self.device)
        gray_rec = self.recon_model(img)
        joined = torch.cat((gray_rec, img), dim=1)
        out_mask = self.seg_model(joined)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        # ⚠️ 修正: 根據 DRAEM 官方與 ARAD_PIP_EVAL，需經過 21x21 的 avg_pool2d 平滑化後再取最大值
        anomaly_map = out_mask_sm[:, 1:, :, :]
        out_mask_averaged = torch.nn.functional.avg_pool2d(
            anomaly_map, 21, stride=1, padding=21 // 2
        )
        score = out_mask_averaged.reshape(out_mask_averaged.shape[0], -1).max(dim=1)[0]
        return score.cpu().numpy(), anomaly_map.detach().cpu().numpy()

    def warmup(self):
        """暖機"""
        dummy = torch.randn(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                rec = self.recon_model(dummy)
                joined = torch.cat((rec, dummy), dim=1)
                self.seg_model(joined)
        cuda_sync()

    def get_model_info(self):
        """回傳模型結構資訊 dict"""
        r_total, _ = count_parameters(self.recon_model)
        s_total, _ = count_parameters(self.seg_model)
        return {
            "total_params": r_total + s_total,
            "model_size_mb": get_model_size_mb(self.recon_model) + get_model_size_mb(self.seg_model),
        }

    def release(self):
        """釋放 GPU 記憶體"""
        del self.recon_model, self.seg_model
        self.recon_model = None
        self.seg_model = None
        cuda_empty_cache()


class PatchCoreRunner:
    """PatchCore (anomalib checkpoint) 推論器"""

    def __init__(self, name, ckpt_path, device):
        self.name = name
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = None  # anomalib PatchcoreModel (torch_model)

    def load(self):
        """
        從 anomalib Lightning checkpoint 載入 PatchCore 模型。
        僅載入底層 torch 子模組，不依賴完整 LightningModule（避免匯入衝突）。
        """
        from anomalib.models.image.patchcore.lightning_model import Patchcore

        # 利用 anomalib 的 Lightning 模型類別來還原完整模型
        self.lightning_model = Patchcore.load_from_checkpoint(
            self.ckpt_path, map_location=self.device
        )
        self.lightning_model.to(self.device)
        self.lightning_model.eval()
        self.model = self.lightning_model.model  # 底層 torch_model
        return True

    def infer_batch(self, batch):
        """
        PatchCore 推論:
          1. WideResNet50 提取 layer2+layer3 特徵
          2. 與 memory bank 做 kNN 距離
          3. 回傳 image-level anomaly score
        """
        img = batch["image_224"].to(self.device)
        # anomalib PatchcoreModel.forward 回傳 (anomaly_map, anomaly_score)
        # 但直接呼叫底層 model 需要用內部方法
        # 使用 lightning_model 的 forward 方式
        output = self.model(img)
        # output 是 InferenceBatch 或 tuple
        if hasattr(output, "anomaly_map"):
            score = output.pred_score
            if score is None:
                amap = output.anomaly_map
                score = amap.reshape(amap.shape[0], -1).max(dim=1)[0]
        elif isinstance(output, tuple):
            anomaly_map, score = output
            if score is None:
                score = anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1)[0]
        else:
            # 嘗試直接取最大值
            score = output.reshape(output.shape[0], -1).max(dim=1)[0]

        # 確保 score 為 1D (batch_size,)，避免 tolist() 產生 nested list
        score = score.view(-1)
        
        # 嘗試取得 amap
        if hasattr(output, "anomaly_map"):
            amap_np = output.anomaly_map.detach().cpu().numpy()
        elif isinstance(output, tuple):
            amap_np = output[0].detach().cpu().numpy()
        else:
            amap_np = None

        return score.detach().cpu().numpy(), amap_np

    def warmup(self):
        """暖機"""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy)
        cuda_sync()

    def get_model_info(self):
        total, _ = count_parameters(self.model)
        return {
            "total_params": total,
            "model_size_mb": get_model_size_mb(self.model),
        }

    def release(self):
        del self.model, self.lightning_model
        self.model = None
        self.lightning_model = None
        cuda_empty_cache()


class EfficientADRunner:
    """EfficientAD (anomalib checkpoint) 推論器"""

    def __init__(self, name, ckpt_path, device):
        self.name = name
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = None

    def load(self):
        """從 anomalib Lightning checkpoint 載入 EfficientAD 模型"""
        from anomalib.models.image.efficient_ad.lightning_model import EfficientAd

        self.lightning_model = EfficientAd.load_from_checkpoint(
            self.ckpt_path, map_location=self.device
        )
        self.lightning_model.to(self.device)
        self.lightning_model.eval()
        self.model = self.lightning_model.model
        return True

    def infer_batch(self, batch):
        """
        EfficientAD 推論:
          使用 anomalib 官方的 lightning_model 進行前向傳播，
          確保使用完整的 pre-processor 與 post-processor（包含 ImageNet norm 與 quantile norm），
          以獲得正確的 anomaly score。
        """
        img = batch["image_ead"].to(self.device)
        
        # ⚠️ 修正: 不要繞過 lightning_model！
        # 直接呼叫 self.lightning_model(img) 返回 InferenceBatch，
        # 其中已經包含經過完整後處理的 pred_score。
        output = self.lightning_model(img)
        
        score = output.pred_score

        # 確保 score 為 1D (batch_size,)，避免 tolist() 產生 nested list
        score = score.view(-1)
        amap_np = output.anomaly_map.detach().cpu().numpy()
        return score.detach().cpu().numpy(), amap_np

    def warmup(self):
        dummy = torch.randn(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                # 暖機也應一併跑完整個 lightning 模型包含的前後處理流程
                self.lightning_model(dummy)
        cuda_sync()

    def get_model_info(self):
        total, _ = count_parameters(self.model)
        return {
            "total_params": total,
            "model_size_mb": get_model_size_mb(self.model),
        }

    def release(self):
        del self.model, self.lightning_model
        self.model = None
        self.lightning_model = None
        cuda_empty_cache()


# ===========================================================================
# 3. 統一 Benchmark 引擎
# ===========================================================================

def benchmark_runner(runner, dataloader, n_repeat=3):
    """
    統一 benchmark 流程：
      1. 載入模型
      2. GPU 暖機
      3. 跑 n_repeat 輪推論，收集 anomaly score 與逐張推論時間
      4. 計算 AUROC, avg latency, FPS
      5. 回傳結果 dict

    設計理由：
      所有 Runner 共享相同介面 (infer_batch, warmup, get_model_info)，
      因此 benchmark 函數只需寫一次就能公平比較所有方法。
    """
    print(f"\n  ── {runner.name} ──")

    # 取得模型結構資訊
    model_info = runner.get_model_info()
    print(f"    參數量: {model_info['total_params']:,} ({model_info['total_params']/1e6:.2f}M)")
    print(f"    模型大小: {model_info['model_size_mb']:.2f} MB")

    # 暖機
    print(f"    🔥 暖機中...")
    runner.warmup()

    # 推論 + 計時
    print(f"    ⏱️  推論中 (重複 {n_repeat} 次)...")
    all_scores = []
    all_labels = []
    all_amaps = []
    all_masks = []
    all_times = []

    with torch.no_grad():
        for repeat_idx in range(n_repeat):
            for batch in dataloader:
                label = batch["label"].numpy()
                mask = batch["mask"].numpy()  # Extract GT mask

                cuda_sync()
                start = time.perf_counter()

                score, amap = runner.infer_batch(batch)

                cuda_sync()
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000.0
                all_times.append(elapsed_ms)

                # 只在第一輪收集 score/label/mask/amap（避免重複）
                if repeat_idx == 0:
                    all_scores.extend(score.tolist())
                    all_labels.extend(label.tolist())
                    if amap is not None:
                        # 確保 amap 大小與 mask 一致 (B, H, W)
                        if amap.ndim == 4:
                            amap = amap[:, 0, :, :]
                        if amap.ndim == 3 and (amap.shape[1] != mask.shape[1] or amap.shape[2] != mask.shape[2]):
                            import torch.nn.functional as F
                            amap_t = torch.tensor(amap).unsqueeze(1)
                            amap_t = F.interpolate(amap_t, size=(mask.shape[1], mask.shape[2]), mode="bilinear", align_corners=False)
                            amap = amap_t.squeeze(1).numpy()
                        all_amaps.append(amap)
                    all_masks.append(mask)

    # 計算 Image-AUROC
    if len(set(all_labels)) > 1:
        auroc = float(roc_auc_score(all_labels, all_scores))
    else:
        auroc = float("nan")
        print(f"    ⚠️ 只有單一類別的標籤，無法計算 Image-AUROC")

    # 計算 Pixel-AUROC & PRO-score
    pixel_auroc = float("nan")
    pro_score = float("nan")
    if len(all_masks) > 0 and len(all_amaps) > 0:
        # Concatenate all batches
        all_masks_np = np.concatenate(all_masks, axis=0)
        all_amaps_np = np.concatenate(all_amaps, axis=0)

        # Check if GT masks have both 0 and 1
        if len(np.unique(all_masks_np)) > 1:
            try:
                from torchmetrics.classification import BinaryAUROC
                from anomalib.metrics.aupro import _AUPRO
                
                # Pixel AUROC computation
                metric_auroc = BinaryAUROC().to(runner.device)
                preds_t = torch.tensor(all_amaps_np).flatten().to(runner.device)
                target_t = torch.tensor(all_masks_np).flatten().to(runner.device).long()
                pixel_auroc = metric_auroc(preds_t, target_t).item()
                
                print(f"    ⏳ 計算 PRO-score 中 (可能需要稍候)...")
                # PRO-score computation
                metric_pro = _AUPRO().to(runner.device)
                pro_preds = torch.tensor(all_amaps_np).unsqueeze(1).to(runner.device)
                pro_target = torch.tensor(all_masks_np).unsqueeze(1).to(runner.device)
                metric_pro.update(pro_preds, pro_target)
                pro_score = metric_pro.compute().item()
            except Exception as e:
                print(f"    ⚠️ 計算 Pixel-AUROC/PRO 失敗: {e}")
        else:
            print(f"    ⚠️ 所有像素標籤單一，無法計算 Pixel metrics")

    # 計算推論效率指標
    total_images = len(all_times)
    total_time_s = sum(all_times) / 1000.0
    avg_time_ms = np.mean(all_times)
    std_time_ms = np.std(all_times)
    fps = total_images / total_time_s if total_time_s > 0 else 0

    print(f"    ✅ Image-AUROC: {auroc:.4f}")
    if not np.isnan(pixel_auroc):
        print(f"    ✅ Pixel-AUROC: {pixel_auroc:.4f}")
        print(f"    ✅ PRO-score:   {pro_score:.4f}")
    print(f"    ✅ Avg Latency: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")
    print(f"    ✅ FPS: {fps:.1f}")

    return {
        "name": runner.name,
        "auroc": auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "avg_time_ms": avg_time_ms,
        "std_time_ms": std_time_ms,
        "fps": fps,
        "total_images": total_images,
        "total_time_s": total_time_s,
        "all_times": all_times,
        "model_params": model_info["total_params"],
        "model_size_mb": model_info["model_size_mb"],
    }


# ===========================================================================
# 4. 視覺化 — 多方法比較圖表
# ===========================================================================

# 配色方案：每個方法固定顏色，便於跨類別比較識別
METHOD_COLORS = {
    "DRAEM (Teacher)": "#e74c3c",       # 紅
    "DRAEM-Student (Ours)": "#2ecc71",  # 綠
    "PatchCore": "#3498db",              # 藍
    "EfficientAD": "#f39c12",            # 橘
}

def get_color(name):
    """依方法名稱回傳固定配色"""
    return METHOD_COLORS.get(name, "#9b59b6")


def plot_all_methods_comparison(results, save_dir, obj_name):
    """
    生成單一物件類別下所有方法的綜合比較圖表 (2x2 佈局)：
      左上: AUROC 長條圖
      右上: 推論時間 長條圖
      左下: FPS 長條圖
      右下: 摘要表格
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"All Methods Inference Comparison — {obj_name}",
        fontsize=18, fontweight="bold", y=0.98
    )

    names = [r["name"] for r in results]
    colors = [get_color(n) for n in names]

    # ── 左上: AUROC ──
    ax = axes[0, 0]
    aurocs = [r["auroc"] for r in results]
    bars = ax.bar(names, aurocs, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Image-level AUROC (Higher is Better)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, aurocs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=15)

    # ── 右上: 推論時間 ──
    ax = axes[0, 1]
    avg_times = [r["avg_time_ms"] for r in results]
    std_times = [r["std_time_ms"] for r in results]
    bars = ax.bar(names, avg_times, yerr=std_times, capsize=5,
                  color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg Inference Time (ms)", fontsize=12)
    ax.set_title("Avg Inference Time (Lower is Better)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f} ms", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=15)

    # ── 左下: FPS ──
    ax = axes[1, 0]
    fps_vals = [r["fps"] for r in results]
    bars = ax.bar(names, fps_vals, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Frames Per Second (FPS)", fontsize=12)
    ax.set_title("Inference Speed FPS (Higher is Better)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=15)

    # ── 右下: 摘要表格 ──
    ax = axes[1, 1]
    ax.axis("off")

    header = f"{'Method':<25}{'AUROC':>8}{'Latency(ms)':>14}{'FPS':>8}{'Params(M)':>12}{'Size(MB)':>10}\n"
    divider = "─" * 77 + "\n"
    table_text = header + divider
    for r in results:
        auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "N/A"
        table_text += (
            f"{r['name']:<25}{auroc_str:>8}{r['avg_time_ms']:>12.2f} ms"
            f"{r['fps']:>8.1f}{r['model_params']/1e6:>10.2f} M{r['model_size_mb']:>9.2f}\n"
        )

    ax.text(0.02, 0.95, table_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, f"{obj_name}_all_methods_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊 比較圖表已儲存: {save_path}")
    return save_path


def plot_inference_time_distribution(results, save_dir, obj_name):
    """生成所有方法的推論時間分佈直方圖"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Inference Time Distribution — {obj_name}",
        fontsize=14, fontweight="bold"
    )

    for r in results:
        ax.hist(r["all_times"], bins=30, alpha=0.4, label=r["name"], color=get_color(r["name"]))

    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{obj_name}_time_distribution.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊 時間分佈圖已儲存: {save_path}")


def plot_overall_summary(all_obj_results, save_dir):
    """
    生成跨物件類別總覽圖：
      每個方法在所有物件上的平均 AUROC / Latency / FPS
    """
    # 收集所有方法名稱（有序）
    method_names = []
    for obj_res in all_obj_results.values():
        for r in obj_res:
            if r["name"] not in method_names:
                method_names.append(r["name"])

    # 彙整每個方法在各物件上的指標
    method_aurocs = {m: [] for m in method_names}
    method_latency = {m: [] for m in method_names}
    method_fps = {m: [] for m in method_names}

    for obj_name, obj_results in all_obj_results.items():
        for r in obj_results:
            if not np.isnan(r["auroc"]):
                method_aurocs[r["name"]].append(r["auroc"])
            method_latency[r["name"]].append(r["avg_time_ms"])
            method_fps[r["name"]].append(r["fps"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "Overall Comparison Across All Object Categories",
        fontsize=16, fontweight="bold"
    )

    x = np.arange(len(method_names))
    colors = [get_color(m) for m in method_names]

    # 平均 AUROC
    ax = axes[0]
    avg_aurocs = [np.mean(method_aurocs[m]) if method_aurocs[m] else 0 for m in method_names]
    bars = ax.bar(x, avg_aurocs, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg AUROC", fontsize=12)
    ax.set_title("Average AUROC", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, avg_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 平均 Latency
    ax = axes[1]
    avg_lats = [np.mean(method_latency[m]) for m in method_names]
    bars = ax.bar(x, avg_lats, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg Latency (ms)", fontsize=12)
    ax.set_title("Average Latency (Lower is Better)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    for bar, val in zip(bars, avg_lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 平均 FPS
    ax = axes[2]
    avg_fps = [np.mean(method_fps[m]) for m in method_names]
    bars = ax.bar(x, avg_fps, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Avg FPS", fontsize=12)
    ax.set_title("Average FPS (Higher is Better)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    for bar, val in zip(bars, avg_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "overall_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 跨類別總覽圖已儲存: {save_path}")


def plot_per_category_auroc(all_obj_results, save_dir):
    """
    生成每個物件類別各方法 AUROC 的分組長條圖，
    讓讀者一目了然看出各方法在不同缺陷類型上的表現差異。
    """
    # 收集方法名稱
    method_names = []
    for obj_res in all_obj_results.values():
        for r in obj_res:
            if r["name"] not in method_names:
                method_names.append(r["name"])

    obj_names = list(all_obj_results.keys())
    n_methods = len(method_names)
    n_objs = len(obj_names)

    fig, ax = plt.subplots(figsize=(max(14, n_objs * 1.5), 7))
    fig.suptitle("AUROC per Category per Method", fontsize=16, fontweight="bold")

    x = np.arange(n_objs)
    width = 0.8 / n_methods

    for i, method in enumerate(method_names):
        aurocs = []
        for obj in obj_names:
            found = [r["auroc"] for r in all_obj_results[obj] if r["name"] == method]
            aurocs.append(found[0] if found else 0)
        bars = ax.bar(x + i * width, aurocs, width, label=method,
                      color=get_color(method), alpha=0.85, edgecolor="black")

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(obj_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "per_category_auroc.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊 各類別 AUROC 圖已儲存: {save_path}")


# ===========================================================================
# 5. 主執行程式
# ===========================================================================

def main(obj_names, args):
    setup_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_root = "./comparison_results"
    os.makedirs(save_root, exist_ok=True)

    n_repeat = args.n_repeat

    print("=" * 80)
    print("  🔬 全方法推論比較 — DRAEM / DRAEM-Student / PatchCore / EfficientAD")
    print("=" * 80)
    print(f"  重複推論次數: {n_repeat}")
    print(f"  物件類別數: {len(obj_names)}")
    print(f"  裝置: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 80)

    # 跨類別結果匯整
    all_obj_results = OrderedDict()
    summary_rows = []  # 用於最終摘要表

    for obj_name in obj_names:
        print(f"\n{'━' * 80}")
        print(f"  📁 物件類別: {obj_name}")
        print(f"{'━' * 80}")

        # ── 檢查資料集路徑 ──
        test_path = os.path.join(args.mvtec_root, obj_name, "test")
        if not os.path.exists(test_path):
            print(f"  ❌ 資料集路徑不存在: {test_path}, 跳過")
            continue

        # ── 建立統一資料集 & DataLoader ──
        dataset = MVTecTestDataset(test_path, resize=256, center_crop=224)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"  📂 資料集大小: {len(dataset)} 張圖片")

        # ── 準備所有 Runner ──
        runners = []

        # (1) DRAEM Teacher (原始)
        draem_recon_path = os.path.join(
            args.draem_root,
            f"DRAEM_seg_large_ae_large_0.0001_800_bs8_{obj_name}_.pckl"
        )
        draem_seg_path = os.path.join(
            args.draem_root,
            f"DRAEM_seg_large_ae_large_0.0001_800_bs8_{obj_name}__seg.pckl"
        )
        if os.path.exists(draem_recon_path) and os.path.exists(draem_seg_path):
            runners.append(DRAEMRunner(
                name="DRAEM (Teacher)",
                recon_path=draem_recon_path,
                seg_path=draem_seg_path,
                recon_base_width=128,
                seg_base_channels=64,
                device=device,
            ))
        else:
            print(f"  ⚠️ DRAEM Teacher 權重未找到，跳過")

        # (2) DRAEM-Student (我們的蒸餾模型)
        student_recon_path = os.path.join(args.student_root, f"{obj_name}_best_recon.pckl")
        student_seg_path = os.path.join(args.student_root, f"{obj_name}_best_seg.pckl")
        if os.path.exists(student_recon_path) and os.path.exists(student_seg_path):
            runners.append(DRAEMRunner(
                name="DRAEM-Student (Ours)",
                recon_path=student_recon_path,
                seg_path=student_seg_path,
                recon_base_width=64,
                seg_base_channels=32,
                device=device,
            ))
        else:
            print(f"  ⚠️ DRAEM-Student 權重未找到，跳過")

        # (3) PatchCore
        patchcore_path = os.path.join(args.patchcore_root, f"{obj_name}.ckpt")
        if os.path.exists(patchcore_path):
            runners.append(PatchCoreRunner(
                name="PatchCore",
                ckpt_path=patchcore_path,
                device=device,
            ))
        else:
            print(f"  ⚠️ PatchCore 權重未找到: {patchcore_path}，跳過")

        # (4) EfficientAD
        efficientad_path = os.path.join(args.efficientad_root, f"{obj_name}.ckpt")
        if os.path.exists(efficientad_path):
            runners.append(EfficientADRunner(
                name="EfficientAD",
                ckpt_path=efficientad_path,
                device=device,
            ))
        else:
            print(f"  ⚠️ EfficientAD 權重未找到: {efficientad_path}，跳過")

        if not runners:
            print(f"  ❌ 此類別無任何可用模型，跳過")
            continue

        # ── 依序載入、推論、釋放 ──
        obj_results = []
        for runner in runners:
            try:
                print(f"\n  📦 載入 {runner.name} ...")
                runner.load()
                result = benchmark_runner(runner, dataloader, n_repeat)
                obj_results.append(result)
            except Exception as e:
                print(f"  ❌ {runner.name} 推論失敗: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    runner.release()
                except Exception:
                    pass

        # ── 產生圖表 ──
        if obj_results:
            obj_save_dir = os.path.join(save_root, obj_name)
            os.makedirs(obj_save_dir, exist_ok=True)
            plot_all_methods_comparison(obj_results, obj_save_dir, obj_name)
            plot_inference_time_distribution(obj_results, obj_save_dir, obj_name)

            all_obj_results[obj_name] = obj_results

            # 為摘要表收集資料
            for r in obj_results:
                summary_rows.append({
                    "object": obj_name,
                    "method": r["name"],
                    "auroc": r["auroc"],
                    "avg_time_ms": r["avg_time_ms"],
                    "fps": r["fps"],
                    "params_m": r["model_params"] / 1e6,
                    "size_mb": r["model_size_mb"],
                })

    # ==========================
    # 跨類別總結報告
    # ==========================
    if all_obj_results:
        print(f"\n\n{'=' * 100}")
        print("  📋 全物件類別 — 總結報告")
        print(f"{'=' * 100}")

        # 文字表格
        header = (
            f"  {'Object':<15}{'Method':<25}"
            f"{'AUROC':>8}{'Latency(ms)':>14}{'FPS':>8}"
            f"{'Params(M)':>12}{'Size(MB)':>10}"
        )
        print(header)
        print(f"  {'─' * 92}")

        for obj_name, obj_results in all_obj_results.items():
            for r in obj_results:
                auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "N/A"
                print(
                    f"  {obj_name:<15}{r['name']:<25}"
                    f"{auroc_str:>8}{r['avg_time_ms']:>12.2f} ms{r['fps']:>8.1f}"
                    f"{r['model_params']/1e6:>10.2f} M{r['model_size_mb']:>9.2f}"
                )
            print(f"  {'─' * 92}")

        # 各方法平均值
        method_names = []
        for obj_res in all_obj_results.values():
            for r in obj_res:
                if r["name"] not in method_names:
                    method_names.append(r["name"])

        print(f"\n  {'--- 各方法跨類別平均指標 ---':^92}")
        print(f"  {'Method':<25}{'Avg AUROC':>12}{'Avg Latency(ms)':>18}{'Avg FPS':>10}")
        print(f"  {'─' * 65}")

        for method in method_names:
            aurocs = [r["auroc"] for obj_res in all_obj_results.values()
                      for r in obj_res if r["name"] == method and not np.isnan(r["auroc"])]
            latencies = [r["avg_time_ms"] for obj_res in all_obj_results.values()
                         for r in obj_res if r["name"] == method]
            fpss = [r["fps"] for obj_res in all_obj_results.values()
                    for r in obj_res if r["name"] == method]

            avg_auroc = np.mean(aurocs) if aurocs else float("nan")
            avg_lat = np.mean(latencies) if latencies else 0
            avg_fps = np.mean(fpss) if fpss else 0

            auroc_str = f"{avg_auroc:.4f}" if not np.isnan(avg_auroc) else "N/A"
            print(f"  {method:<25}{auroc_str:>12}{avg_lat:>16.2f} ms{avg_fps:>10.1f}")

        # 跨類別圖表
        plot_overall_summary(all_obj_results, save_root)
        if len(all_obj_results) > 1:
            plot_per_category_auroc(all_obj_results, save_root)

        # 儲存 JSON 結果
        json_results = {}
        for obj_name, obj_results in all_obj_results.items():
            json_results[obj_name] = []
            for r in obj_results:
                json_results[obj_name].append({
                    "name": r["name"],
                    "auroc": float(r["auroc"]) if not np.isnan(r["auroc"]) else None,
                    "pixel_auroc": float(r["pixel_auroc"]) if "pixel_auroc" in r and not np.isnan(r["pixel_auroc"]) else None,
                    "pro_score": float(r["pro_score"]) if "pro_score" in r and not np.isnan(r["pro_score"]) else None,
                    "avg_time_ms": float(r["avg_time_ms"]),
                    "std_time_ms": float(r["std_time_ms"]),
                    "fps": float(r["fps"]),
                    "model_params": int(r["model_params"]),
                    "model_size_mb": float(r["model_size_mb"]),
                })

        json_path = os.path.join(save_root, "all_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 JSON 結果已儲存: {json_path}")

        # ==========================
        # 生成 Markdown 表格
        # ==========================
        print(f"\n\n{'=' * 100}")
        print("  📝 Markdown 比較表格生成")
        print(f"{'=' * 100}")
        
        texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
        object_classes = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
        
        metrics = {"Image-AUROC": "auroc", "Pixel-AUROC": "pixel_auroc", "PRO-score": "pro_score"}
        md_tables = {}
        
        for metric_name, field_name in metrics.items():
            md = f"### {metric_name} Comparison\n"
            md += f"| Category | Class | " + " | ".join(method_names) + " |\n"
            md += f"| --- | --- | " + " | ".join(["---"] * len(method_names)) + " |\n"
            
            def get_row(cat, cls):
                row_str = f"| {cat} | {cls} | "
                vals = []
                for m in method_names:
                    # Find score
                    score = None
                    if cls in all_obj_results:
                        for r in all_obj_results[cls]:
                            if r["name"] == m and not np.isnan(r.get(field_name, float('nan'))):
                                score = r[field_name] * 100.0  # Convert to percentage
                                break
                    vals.append(score)
                
                # Format strong for best
                max_val = max([v for v in vals if v is not None], default=-1)
                for v in vals:
                    if v is None:
                        row_str += "- | "
                    elif v == max_val and v > 0:
                        row_str += f"**{v:.1f}** | "
                    else:
                        row_str += f"{v:.1f} | "
                return row_str + "\n", vals
            
            texture_vals_list = {m: [] for m in method_names}
            for cls in texture_classes:
                if cls in all_obj_results:
                    row_md, vals = get_row("Texture", cls)
                    md += row_md
                    for m_idx, m in enumerate(method_names):
                        if vals[m_idx] is not None:
                            texture_vals_list[m].append(vals[m_idx])
            
            # Texture Average
            t_avg_str = f"| Texture | average | "
            t_avgs = []
            for m in method_names:
                vs = texture_vals_list[m]
                avg = np.mean(vs) if vs else float('nan')
                t_avgs.append(avg)
            max_t = max([v for v in t_avgs if not np.isnan(v)], default=-1)
            for v in t_avgs:
                if np.isnan(v):
                    t_avg_str += "- | "
                elif v == max_t and v > 0:
                    t_avg_str += f"**{v:.2f}** | "
                else:
                    t_avg_str += f"{v:.2f} | "
            md += t_avg_str + "\n"
            
            object_vals_list = {m: [] for m in method_names}
            for cls in object_classes:
                if cls in all_obj_results:
                    row_md, vals = get_row("Object", cls)
                    md += row_md
                    for m_idx, m in enumerate(method_names):
                        if vals[m_idx] is not None:
                            object_vals_list[m].append(vals[m_idx])
                            
            # Object Average
            o_avg_str = f"| Object | average | "
            o_avgs = []
            for m in method_names:
                vs = object_vals_list[m]
                avg = np.mean(vs) if vs else float('nan')
                o_avgs.append(avg)
            max_o = max([v for v in o_avgs if not np.isnan(v)], default=-1)
            for v in o_avgs:
                if np.isnan(v):
                    o_avg_str += "- | "
                elif v == max_o and v > 0:
                    o_avg_str += f"**{v:.2f}** | "
                else:
                    o_avg_str += f"{v:.2f} | "
            md += o_avg_str + "\n"
            
            # Total Average
            tot_avg_str = f"| **Total Average** | | "
            tot_avgs = []
            for i, m in enumerate(method_names):
                tot = []
                tot.extend(texture_vals_list[m])
                tot.extend(object_vals_list[m])
                avg = np.mean(tot) if tot else float('nan')
                tot_avgs.append(avg)
            max_tot = max([v for v in tot_avgs if not np.isnan(v)], default=-1)
            for v in tot_avgs:
                if np.isnan(v):
                    tot_avg_str += "- | "
                elif v == max_tot and v > 0:
                    tot_avg_str += f"**{v:.2f}** | "
                else:
                    tot_avg_str += f"{v:.2f} | "
            md += tot_avg_str + "\n\n"
            
            md_tables[metric_name] = md

        print("\n".join(md_tables.values()))
        with open(os.path.join(save_root, "markdown_tables.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_tables.values()))
        print(f"\n  💾 Markdown 表格已產生並儲存至: {os.path.join(save_root, 'markdown_tables.md')}")

    print(f"\n{'=' * 80}")
    print("  🎉 所有比較測試已完成！")
    print(f"  結果儲存於: {os.path.abspath(save_root)}")
    print(f"{'=' * 80}")


# ===========================================================================
# 6. CLI 入口
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MVTec AD — 全方法推論比較 (DRAEM / Student / PatchCore / EfficientAD)"
    )
    parser.add_argument(
        "--obj_id", type=int, required=True,
        help="物件類別 ID (-1 = 全部 15 類)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-2,
        help="GPU ID (-2: auto-select, -1: CPU)"
    )
    parser.add_argument(
        "--mvtec_root", type=str, default="./mvtec",
        help="MVTec AD 資料集根目錄"
    )
    parser.add_argument(
        "--draem_root", type=str, default="./DRAEM_checkpoints",
        help="DRAEM Teacher checkpoint 目錄"
    )
    parser.add_argument(
        "--student_root", type=str, default="./student_model_checkpoints",
        help="DRAEM-Student checkpoint 目錄"
    )
    parser.add_argument(
        "--patchcore_root", type=str, default="./PatchCore",
        help="PatchCore checkpoint 目錄"
    )
    parser.add_argument(
        "--efficientad_root", type=str, default="./EfficientAD",
        help="EfficientAD checkpoint 目錄"
    )
    parser.add_argument(
        "--n_repeat", type=int, default=3,
        help="推論重複次數 (取平均，預設 3 次)"
    )

    args = parser.parse_args()

    # GPU 選擇
    if args.gpu_id == -2:
        args.gpu_id = get_available_gpu()
        if args.gpu_id >= 0:
            print(f"自動選擇 GPU: {args.gpu_id}")

    # 物件類別列表
    obj_batch = [
        ["capsule"], ["bottle"], ["carpet"], ["leather"], ["pill"],
        ["transistor"], ["tile"], ["cable"], ["zipper"], ["toothbrush"],
        ["metal_nut"], ["hazelnut"], ["screw"], ["grid"], ["wood"],
    ]

    all_classes = [
        "capsule", "bottle", "carpet", "leather", "pill",
        "transistor", "tile", "cable", "zipper", "toothbrush",
        "metal_nut", "hazelnut", "screw", "grid", "wood",
    ]

    if int(args.obj_id) == -1:
        picked_classes = all_classes
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    if args.gpu_id == -1:
        main(picked_classes, args)
    else:
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
