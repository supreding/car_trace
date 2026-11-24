# -*- coding: utf-8 -*-
"""园区车辆 FastReID 微调与评估闭环（基于人工标注 + 高置信车牌簇）。

解决问题：
  通用 VeRi 预训练权重（veri_sbs_R50-ibn.pth）在粮库同色同型大货车上区分度不足
  （同车均值 ~0.87，异车 P90 高达 ~0.81），导致无牌散图/大门图归类及外观聚类容易串车。

训练信号来源（人工优先，落库即沉淀）：
  1. 人工 ReID 真值组：capture_annotations.identity_id（同一工作台「同车对比」标出）
  2. 人工配对约束：pair_annotations
     - relation='same'      -> 正样本对（拉近跨视角/跨昼夜同车特征）
     - relation='different' -> 显式硬负样本对（强推开高相似的同款不同货车！）
  3. 人工合并记录：manual_vehicle_merge_pairs（同车正样本对）
  4. 人工确认车牌：capture_annotations.corrected_plate（同园区同牌归为同一真值车）
  5. 高置信车牌引导簇（可选，默认开启 --include-verified-plates）：
     OSD+LPR 一致或高置信(>=0.92)多单同牌车辆，避免初期人工标注量少时过拟合，
     让模型立即适应本园区机位、光照和货车外观分布。

安全上线机制：
  - 按车辆身份切分 train / val（杜绝同车跨帧泄漏），并在微调前后跑同一套留出评估；
  - 仅当验证集 ROC-AUC 与区分间隔（同车均值 - 异车 P90）严格优于基线（或显式
    传 --force-activate）时，才写入 models/fastreid/active_reid.json 激活新权重；
  - 激活后 reid_pipeline.py 下次运行（或传入 --reembed）自动加载新权重并刷新全库特征。

用法：
  # 仅评估当前激活模型在本库留出集上的区分度
  python app/finetune_reid.py --db app/granary.db --eval-only

  # 执行微调并按验证指标决定是否激活（GPU 约 1~3 分钟）
  python app/finetune_reid.py --db app/granary.db --epochs 12

  # 仅使用纯人工标注微调（当人工真值车辆 >= 30 时推荐）
  python app/finetune_reid.py --db app/granary.db --manual-only --epochs 15
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FASTREID_ROOT = PROJECT_ROOT / "third_party" / "fast-reid"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if FASTREID_ROOT.is_dir() and str(FASTREID_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTREID_ROOT))

# FastReID 1.3 Python 3.12 兼容垫片
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

import build_db


DEFAULT_CONFIG = FASTREID_ROOT / "configs" / "VeRi" / "sbs_R50-ibn.yml"
DEFAULT_BASE_WEIGHTS = PROJECT_ROOT / "models" / "fastreid" / "veri_sbs_R50-ibn.pth"
DEFAULT_OUT_WEIGHTS = PROJECT_ROOT / "models" / "fastreid" / "finetuned_sbs_R50-ibn.pth"
ACTIVE_META_PATH = PROJECT_ROOT / "models" / "fastreid" / "active_reid.json"
REPORT_PATH = PROJECT_ROOT / "models" / "fastreid" / "finetune_report.json"


def _resolve_crop(db_path: Path, stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else (db_path.resolve().parent / p)


def summarize_training_readiness(conn: sqlite3.Connection) -> dict[str, Any]:
    """统计数据库里各类人工标注与训练素材就绪度（供工作台看板与导出报告用）。"""
    build_db.ensure_schema(conn)
    build_db.ensure_label_schema(conn)

    def count(sql: str, params: tuple = ()) -> int:
        try:
            return int(conn.execute(sql, params).fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    reid_identities = count(
        "SELECT COUNT(DISTINCT identity_id) FROM capture_annotations "
        "WHERE identity_id IS NOT NULL AND usability='valid'"
    )
    reid_assigned = count(
        "SELECT COUNT(*) FROM capture_annotations "
        "WHERE identity_id IS NOT NULL AND usability='valid'"
    )
    same_pairs = count("SELECT COUNT(*) FROM pair_annotations WHERE relation='same'")
    diff_pairs = count("SELECT COUNT(*) FROM pair_annotations WHERE relation='different'")
    merge_pairs = count("SELECT COUNT(*) FROM manual_vehicle_merge_pairs")
    manual_plates = count(
        "SELECT COUNT(*) FROM capture_annotations "
        "WHERE coalesce(corrected_plate,'')!='' AND usability='valid'"
    )
    manual_boxes = count("SELECT COUNT(*) FROM manual_box_edits")
    manual_attrs = count(
        "SELECT COUNT(*) FROM capture_annotations "
        "WHERE coalesce(corrected_color,'')!='' OR coalesce(corrected_type,'')!=''"
    )
    manual_no_veh = count(
        "SELECT COUNT(*) FROM capture_annotations WHERE no_vehicle_manual=1"
    )
    verified_plate_groups = count(
        """SELECT COUNT(*) FROM (
             SELECT coalesce(park,''), substr(plate_final, 2) AS body
             FROM captures
             WHERE coalesce(capture_status,'vehicle')!='no_vehicle'
               AND vehicle_crop_path IS NOT NULL
               AND coalesce(plate_final,'')!=''
               AND length(plate_final)>=7
               AND (recognition_source IN ('OSD+车牌模型一致','人工标注','人工复核')
                    OR coalesce(lpr_conf,0)>=0.92)
               AND coalesce(park,'') NOT LIKE ?
             GROUP BY coalesce(park,''), body
             HAVING COUNT(*) >= 2
           )""",
        (build_db.QUARANTINE_PARK_LIKE,),
    )
    active_info: dict[str, Any] = {
        "active": False,
        "model_version": f"fastreid:{DEFAULT_BASE_WEIGHTS.name}",
        "weights": str(DEFAULT_BASE_WEIGHTS.name),
    }
    if ACTIVE_META_PATH.is_file():
        try:
            meta = json.loads(ACTIVE_META_PATH.read_text(encoding="utf-8"))
            active_info = {
                "active": True,
                "model_version": meta.get("model_version", ""),
                "weights": meta.get("weights", ""),
                "activated_at": meta.get("activated_at", ""),
                "metrics": meta.get("metrics", {}),
            }
        except Exception:
            pass
    last_report: dict[str, Any] | None = None
    if REPORT_PATH.is_file():
        try:
            last_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "reid": {
            "manual_identities": reid_identities,
            "manual_assigned_captures": reid_assigned,
            "same_pairs": same_pairs,
            "different_pairs": diff_pairs,
            "merge_pairs": merge_pairs,
            "verified_plate_vehicles": verified_plate_groups,
            "ready_with_bootstrap": verified_plate_groups >= 10 or (reid_identities >= 5 and reid_assigned >= 15),
            "ready_manual_only": reid_identities >= 15 and reid_assigned >= 40,
        },
        "detection": {
            "manual_boxes": manual_boxes,
            "manual_no_vehicle": manual_no_veh,
            "recommended_min": 200,
            "ready": manual_boxes >= 200,
        },
        "attr": {
            "manual_attrs": manual_attrs,
            "recommended_min": 300,
            "ready": manual_attrs >= 300,
        },
        "lpr": {
            "manual_plates": manual_plates,
            "recommended_min": 300,
            "ready": manual_plates >= 300,
        },
        "active_reid": active_info,
        "last_report": last_report,
    }


def collect_reid_dataset(
    db_path: Path,
    *,
    include_verified_plates: bool = True,
    max_per_id_day: int = 6,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """从数据库收集 ReID 训练/留出验证样本与显式人工配对约束。

    返回：
      - train_samples / val_samples: list[{capture_id, crop_path, pid, date, park, is_manual}]
      - train_same_pairs / train_diff_pairs: list[(cid1, cid2)]
      - val_manual_pairs: list[(cid1, cid2, label)]
      - stats: 数据统计摘要
    """
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    build_db.ensure_schema(conn)
    build_db.ensure_label_schema(conn)

    # 读取所有有效且裁剪图存在的抓拍
    cap_rows = conn.execute(
        """SELECT c.id, coalesce(c.park,'') AS park, coalesce(c.date,'') AS date,
                  coalesce(c.order_id,'') AS order_id, coalesce(c.camera_key,'') AS camera_key,
                  c.vehicle_crop_path, coalesce(c.plate_final,'') AS plate_final,
                  coalesce(c.lpr_conf, 0.0) AS lpr_conf,
                  coalesce(c.recognition_source,'') AS recognition_source,
                  ca.identity_id, coalesce(ca.usability,'valid') AS usability,
                  coalesce(ca.corrected_plate,'') AS corrected_plate
           FROM captures c
           LEFT JOIN capture_annotations ca ON ca.capture_id=c.id
           WHERE coalesce(c.capture_status,'vehicle')!='no_vehicle'
             AND c.vehicle_crop_path IS NOT NULL
             AND coalesce(c.park,'') NOT LIKE ?""",
        (build_db.QUARANTINE_PARK_LIKE,),
    ).fetchall()

    caps: dict[int, dict[str, Any]] = {}
    for r in cap_rows:
        if r["usability"] == "invalid":
            continue
        crop = _resolve_crop(db_path, r["vehicle_crop_path"])
        if not crop.is_file():
            continue
        cid = int(r["id"])
        caps[cid] = {
            "capture_id": cid,
            "park": r["park"],
            "date": r["date"],
            "order_id": r["order_id"],
            "camera_key": r["camera_key"],
            "crop_path": str(crop),
            "plate_final": r["plate_final"],
            "lpr_conf": float(r["lpr_conf"]),
            "recognition_source": r["recognition_source"],
            "identity_id": r["identity_id"],
            "corrected_plate": r["corrected_plate"],
        }

    # 读取人工配对与人工合并
    pair_same: list[tuple[int, int]] = []
    pair_diff: list[tuple[int, int]] = []
    diff_adj: dict[int, set[int]] = defaultdict(set)
    try:
        for lid, rid, rel in conn.execute(
            "SELECT left_capture_id, right_capture_id, relation FROM pair_annotations "
            "WHERE relation IN ('same', 'different')"
        ).fetchall():
            lid, rid = int(lid), int(rid)
            if lid not in caps or rid not in caps:
                continue
            if rel == "same":
                pair_same.append((lid, rid))
            else:
                pair_diff.append((lid, rid))
                diff_adj[lid].add(rid)
                diff_adj[rid].add(lid)
    except sqlite3.OperationalError:
        pass

    merge_same: list[tuple[int, int]] = []
    try:
        for cid, aid in conn.execute(
            "SELECT capture_id, anchor_capture_id FROM manual_vehicle_merge_pairs"
        ).fetchall():
            cid, aid = int(cid), int(aid)
            if cid in caps and aid in caps and cid != aid:
                merge_same.append((cid, aid))
    except sqlite3.OperationalError:
        pass
    conn.close()

    # 并查集构建车辆真值身份簇（人工真值优先，绝不跨越不同园区或显式 diff 对）
    parent: dict[int, int] = {}
    members: dict[int, set[int]] = {}
    manual_cids: set[int] = set()

    def ensure_node(cid: int) -> None:
        if cid not in parent:
            parent[cid] = cid
            members[cid] = {cid}

    def find(cid: int) -> int:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def union(a: int, b: int) -> bool:
        if caps[a]["park"] != caps[b]["park"]:
            return False
        ensure_node(a)
        ensure_node(b)
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # 若两组间存在人工标记的 different 硬负对，拒绝合并
        for ca in members[ra]:
            if ca in diff_adj and (diff_adj[ca] & members[rb]):
                return False
        low, high = (ra, rb) if ra < rb else (rb, ra)
        parent[high] = low
        members[low] |= members.pop(high)
        return True

    # 1) 人工 identity_id 分组
    by_identity: dict[int, list[int]] = defaultdict(list)
    for cid, c in caps.items():
        if c["identity_id"]:
            by_identity[int(c["identity_id"])].append(cid)
            manual_cids.add(cid)
    for cids in by_identity.values():
        ensure_node(cids[0])
        for other in cids[1:]:
            union(cids[0], other)

    # 2) 人工 same 配对与人工合并
    for a, b in pair_same + merge_same:
        manual_cids.add(a)
        manual_cids.add(b)
        union(a, b)
    for a, b in pair_diff:
        manual_cids.add(a)
        manual_cids.add(b)
        ensure_node(a)
        ensure_node(b)

    # 3) 人工确认车牌（corrected_plate）同园同牌归一
    by_manual_plate: dict[tuple[str, str], list[int]] = defaultdict(list)
    for cid, c in caps.items():
        if c["corrected_plate"]:
            body = build_db._plate_body(c["corrected_plate"])
            if len(body) >= 5:
                by_manual_plate[(c["park"], body)].append(cid)
                manual_cids.add(cid)
    for cids in by_manual_plate.values():
        ensure_node(cids[0])
        for other in cids[1:]:
            union(cids[0], other)

    # 4) 高置信车牌引导簇（防小样本过拟合，给微调提供丰富的园区货车底盘）
    if include_verified_plates:
        by_verified_plate: dict[tuple[str, str], list[int]] = defaultdict(list)
        for cid, c in caps.items():
            pf = c["plate_final"]
            if not pf or len(pf) < 7:
                continue
            trusted = (
                c["recognition_source"] in ("OSD+车牌模型一致", "人工标注", "人工复核")
                or c["lpr_conf"] >= 0.92
            )
            if not trusted:
                continue
            body = build_db._plate_body(pf)
            if len(body) >= 5:
                by_verified_plate[(c["park"], body)].append(cid)
        for cids in by_verified_plate.values():
            if len(cids) < 2:
                continue
            ensure_node(cids[0])
            for other in cids[1:]:
                union(cids[0], other)

    # 按簇汇总，应用单车单日上限（防止某辆高频车主导梯度）
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cid in parent:
        root = find(cid)
        c = dict(caps[cid])
        c["is_manual"] = cid in manual_cids
        clusters[root].append(c)

    rng = random.Random(seed)
    valid_clusters: list[tuple[int, list[dict[str, Any]], bool]] = []
    for root, items in sorted(clusters.items()):
        # 同车同日最多保留 max_per_id_day 张（人工标注帧优先保留）
        by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for it in items:
            by_day[it["date"] or "unknown"].append(it)
        kept: list[dict[str, Any]] = []
        for day_items in by_day.values():
            day_items.sort(key=lambda x: (0 if x["is_manual"] else 1, x["capture_id"]))
            kept.extend(day_items[:max_per_id_day])
        if len(kept) < 2:
            continue
        has_manual = any(x["is_manual"] for x in kept)
        valid_clusters.append((root, kept, has_manual))

    # 按车辆身份分层切分 train / val（防同车跨帧数据泄漏）
    rng.shuffle(valid_clusters)
    n_val = max(2, int(round(len(valid_clusters) * val_ratio))) if len(valid_clusters) >= 5 else max(1, len(valid_clusters) // 3)
    val_clusters = valid_clusters[:n_val]
    train_clusters = valid_clusters[n_val:]

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    train_cid_set: set[int] = set()
    val_cid_set: set[int] = set()

    for pid, (_root, items, _hm) in enumerate(train_clusters):
        for it in items:
            it["pid"] = pid
            train_samples.append(it)
            train_cid_set.add(it["capture_id"])
    for pid, (_root, items, _hm) in enumerate(val_clusters):
        for it in items:
            it["pid"] = pid
            val_samples.append(it)
            val_cid_set.add(it["capture_id"])

    # 配对样本分配：两端都在 val 的进验证；否则只要不跨进 val 集就用于训练（避免泄漏）
    train_same_pairs = [
        (caps[a]["crop_path"], caps[b]["crop_path"])
        for a, b in (pair_same + merge_same)
        if not (a in val_cid_set or b in val_cid_set)
    ]
    train_diff_pairs = [
        (caps[a]["crop_path"], caps[b]["crop_path"])
        for a, b in pair_diff
        if not (a in val_cid_set or b in val_cid_set)
    ]
    val_manual_pairs = [
        (caps[a]["crop_path"], caps[b]["crop_path"], 1)
        for a, b in (pair_same + merge_same)
        if (a in val_cid_set or b in val_cid_set)
    ] + [
        (caps[a]["crop_path"], caps[b]["crop_path"], 0)
        for a, b in pair_diff
        if (a in val_cid_set or b in val_cid_set)
    ]

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_same_pairs": train_same_pairs,
        "train_diff_pairs": train_diff_pairs,
        "val_manual_pairs": val_manual_pairs,
        "stats": {
            "total_identities": len(valid_clusters),
            "train_identities": len(train_clusters),
            "val_identities": len(val_clusters),
            "train_images": len(train_samples),
            "val_images": len(val_samples),
            "manual_train_same_pairs": len(train_same_pairs),
            "manual_train_diff_pairs": len(train_diff_pairs),
            "manual_val_pairs": len(val_manual_pairs),
            "include_verified_plates": include_verified_plates,
        },
    }


def _load_rgb_tensor(crop_path: str, size_hw: tuple[int, int], augment: bool = False):
    """按 FastReID DefaultPredictor 完全一致的预处理读取裁剪图（RGB, 0..255 float32 CHW）。"""
    import torch

    with Image.open(crop_path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    h, w = size_hw
    if augment:
        # 轻量车辆 ReID 增强：随机水平翻转 + 轻微亮/对比度抖动 + 随机擦除（模拟遮挡）
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12.0, 12.0)
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        if random.random() < 0.35:
            eh = random.randint(max(8, h // 10), max(16, h // 3))
            ew = random.randint(max(8, w // 10), max(16, w // 3))
            ey = random.randint(0, max(0, h - eh))
            ex = random.randint(0, max(0, w - ew))
            resized[ey:ey + eh, ex:ex + ew, :] = int(np.mean(resized))
    else:
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    return torch.as_tensor(resized.astype(np.float32).transpose(2, 0, 1))


def _build_fastreid_model(config_file: Path, weights_file: Path, device: str):
    """构建 FastReID SBS 模型并加载权重（100% 兼容 DefaultPredictor）。"""
    import torch
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer

    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(weights_file)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()

    model = build_model(cfg)
    Checkpointer(model).load(str(weights_file))
    model.to(torch.device(device))
    return model, cfg


def _extract_features_batch(model, images_tensor):
    """前向提取 L2 归一化 BNNeck 特征（训练与推理路径完全对齐）。"""
    import torch.nn.functional as F

    x = (images_tensor - model.pixel_mean) / model.pixel_std
    feat_map = model.backbone(x)
    pool_feat = model.heads.pool_layer(feat_map)
    neck_feat = model.heads.bottleneck(pool_feat)[..., 0, 0]
    return F.normalize(neck_feat, p=2, dim=1)


def _compute_roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """无 sklearn 依赖的精确 ROC-AUC（Mann-Whitney U 秩和计算）。"""
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # 处理并列分数（平均秩）
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = 0.5 * (i + 1 + j)
        i = j
    pos_rank_sum = ranks[sorted_labels == 1].sum()
    n_pos = float(pos_scores.size)
    n_neg = float(neg_scores.size)
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def evaluate_reid_model(
    model,
    cfg,
    val_samples: list[dict[str, Any]],
    val_manual_pairs: list[tuple[str, str, int]],
    device: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    """在留出验证集上评估车辆区分能力（ROC-AUC、同车均值、异车 P90、Margin、Rank-1、mAP）。"""
    import torch

    if len(val_samples) < 2:
        return {"val_images": len(val_samples), "roc_auc": 0.0, "margin": 0.0,
                "same_mean": 0.0, "diff_mean": 0.0, "diff_p90": 0.0,
                "rank1": 0.0, "map": 0.0}

    model.eval()
    size_hw = tuple(cfg.INPUT.SIZE_TEST)
    feats_list = []
    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i + batch_size]
            tensors = torch.stack([_load_rgb_tensor(s["crop_path"], size_hw, augment=False)
                                   for s in batch]).to(device)
            feats = _extract_features_batch(model, tensors)
            feats_list.append(feats.cpu().numpy())
    embs = np.concatenate(feats_list, axis=0)
    pids = np.array([s["pid"] for s in val_samples], dtype=np.int64)
    sim = embs @ embs.T

    # 上三角全对统计
    tri_i, tri_j = np.triu_indices(len(val_samples), k=1)
    pair_sims = sim[tri_i, tri_j]
    same_mask = pids[tri_i] == pids[tri_j]
    pos_scores = pair_sims[same_mask]
    neg_scores = pair_sims[~same_mask]

    same_mean = float(pos_scores.mean()) if pos_scores.size else 0.0
    diff_mean = float(neg_scores.mean()) if neg_scores.size else 0.0
    diff_p90 = float(np.percentile(neg_scores, 90)) if neg_scores.size else 0.0
    roc_auc = _compute_roc_auc(pos_scores, neg_scores)

    # Rank-1 与 mAP（留一检索）
    rank1_hits = 0
    aps: list[float] = []
    valid_queries = 0
    for idx in range(len(val_samples)):
        row_sim = sim[idx].copy()
        row_sim[idx] = -2.0
        order = np.argsort(-row_sim)
        matches = (pids[order] == pids[idx]).astype(np.int32)
        n_rel = int(matches.sum())
        if n_rel == 0:
            continue
        valid_queries += 1
        if matches[0] == 1:
            rank1_hits += 1
        cum = np.cumsum(matches)
        prec = cum / np.arange(1, len(matches) + 1)
        aps.append(float((prec * matches).sum() / n_rel))

    result: dict[str, Any] = {
        "val_images": len(val_samples),
        "pos_pairs": int(pos_scores.size),
        "neg_pairs": int(neg_scores.size),
        "same_mean": round(same_mean, 4),
        "diff_mean": round(diff_mean, 4),
        "diff_p90": round(diff_p90, 4),
        "margin": round(same_mean - diff_p90, 4),
        "roc_auc": round(roc_auc, 4),
        "rank1": round(rank1_hits / max(1, valid_queries), 4),
        "map": round(float(np.mean(aps)) if aps else 0.0, 4),
    }

    # 额外汇报显式人工配对验证指标（若有）
    if val_manual_pairs:
        m_pos, m_neg = [], []
        with torch.no_grad():
            for p1, p2, lbl in val_manual_pairs:
                t = torch.stack([
                    _load_rgb_tensor(p1, size_hw, augment=False),
                    _load_rgb_tensor(p2, size_hw, augment=False),
                ]).to(device)
                f = _extract_features_batch(model, t).cpu().numpy()
                s = float(np.dot(f[0], f[1]))
                (m_pos if lbl == 1 else m_neg).append(s)
        result["manual_pairs"] = {
            "same_count": len(m_pos),
            "diff_count": len(m_neg),
            "same_mean": round(float(np.mean(m_pos)), 4) if m_pos else None,
            "diff_mean": round(float(np.mean(m_neg)), 4) if m_neg else None,
        }
    return result


def _batch_hard_cosine_triplet_loss(feats, pids, margin: float = 0.25):
    """余弦距离 Batch-Hard Triplet Loss：每个样本找最难正样本 + 最难负样本。"""
    import torch
    import torch.nn.functional as F

    sim = feats @ feats.T  # [-1, 1]
    dist = 1.0 - sim       # [0, 2]
    same = pids.unsqueeze(0) == pids.unsqueeze(1)
    eye = torch.eye(len(pids), dtype=torch.bool, device=feats.device)
    pos_mask = same & (~eye)
    neg_mask = ~same

    # 最难正样本距离（同车最大距离）
    pos_dist = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values
    # 最难负样本距离（异车最小距离）
    neg_dist = dist.masked_fill(~neg_mask, 1e4).min(dim=1).values

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return feats.sum() * 0.0
    return F.relu(pos_dist[valid] - neg_dist[valid] + margin).mean()


def _supcon_loss(feats, pids, temperature: float = 0.08):
    """监督对比损失（Supervised Contrastive Loss），拉紧类内、推开类间。"""
    import torch

    sim = (feats @ feats.T) / temperature
    eye = torch.eye(len(pids), dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_mask = (pids.unsqueeze(0) == pids.unsqueeze(1)) & (~eye)
    pos_cnt = pos_mask.sum(dim=1)
    valid = pos_cnt > 0
    if not valid.any():
        return feats.sum() * 0.0
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_cnt[valid].float()
    return -mean_log_prob_pos.mean()


def run_finetune(
    db_path: Path,
    *,
    config_file: Path = DEFAULT_CONFIG,
    base_weights: Path = DEFAULT_BASE_WEIGHTS,
    out_weights: Path = DEFAULT_OUT_WEIGHTS,
    device: str = "cuda:0",
    epochs: int = 12,
    p_identities: int = 8,
    k_instances: int = 4,
    lr: float = 1e-4,
    include_verified_plates: bool = True,
    eval_only: bool = False,
    force_activate: bool = False,
    seed: int = 42,
    progress_cb=None,
) -> dict[str, Any]:
    """执行完整 ReID 微调与留出集对比评估。"""
    import torch
    import torch.nn.functional as F

    def report_progress(msg: str, pct: int = 0) -> None:
        print(f"[finetune-reid] {msg}", flush=True)
        if progress_cb:
            progress_cb(msg, pct)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        report_progress("未检测到 CUDA GPU，自动降级为 CPU 运行", 2)

    report_progress("收集训练与留出验证数据集...", 5)
    ds = collect_reid_dataset(
        db_path,
        include_verified_plates=include_verified_plates,
        seed=seed,
    )
    stats = ds["stats"]
    report_progress(
        f"数据就绪：训练 {stats['train_identities']} 辆车/{stats['train_images']} 张，"
        f"验证 {stats['val_identities']} 辆车/{stats['val_images']} 张，"
        f"人工配对 同车 {stats['manual_train_same_pairs']} / 异车 {stats['manual_train_diff_pairs']}",
        10,
    )
    if stats["val_images"] < 4 or (not eval_only and stats["train_identities"] < 4):
        raise RuntimeError(
            f"可用训练样本不足（训练车辆 {stats['train_identities']} 辆、验证图 {stats['val_images']} 张）。"
            "请先在「同车对比」或「车牌标注」中积累更多人工真值，或保持开启高置信车牌引导。"
        )

    # 评估基线模型（始终和原始预训练权重对比，也可看当前激活权重）
    report_progress(f"加载基线权重 {base_weights.name} 并跑留出验证集...", 15)
    model, cfg = _build_fastreid_model(config_file, base_weights, device)
    t0 = time.time()
    baseline_metrics = evaluate_reid_model(
        model, cfg, ds["val_samples"], ds["val_manual_pairs"], device
    )
    report_progress(
        f"基线指标：AUC={baseline_metrics['roc_auc']} | 同车均值={baseline_metrics['same_mean']} | "
        f"异车P90={baseline_metrics['diff_p90']} | Margin={baseline_metrics['margin']} | "
        f"Rank-1={baseline_metrics['rank1']} | mAP={baseline_metrics['map']}",
        25,
    )
    if eval_only:
        return {
            "mode": "eval_only",
            "device": device,
            "stats": stats,
            "baseline": baseline_metrics,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # 冻结底层特征（conv1/bn1/layer1/layer2），微调高层语义（layer3/layer4/BNNeck）防小样本过拟合
    for name, param in model.named_parameters():
        if name.startswith(("backbone.conv1", "backbone.bn1", "backbone.layer1", "backbone.layer2")):
            param.requires_grad = False
        else:
            param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr * 3.0},
        ],
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)

    # 按车辆分组构建 P×K 平衡采样池（含人工标注的车辆采样权重提高 3 倍）
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    manual_pids: set[int] = set()
    for s in ds["train_samples"]:
        by_pid[s["pid"]].append(s)
        if s.get("is_manual"):
            manual_pids.add(s["pid"])
    pid_pool = [pid for pid, items in by_pid.items() if len(items) >= 2]
    weighted_pool = pid_pool + [pid for pid in pid_pool if pid in manual_pids] * 2

    size_hw = tuple(cfg.INPUT.SIZE_TRAIN)
    p_eff = min(p_identities, len(pid_pool))
    steps_per_epoch = max(6, min(60, math.ceil(len(weighted_pool) / max(1, p_eff)) * 2))
    same_pairs = ds["train_same_pairs"]
    diff_pairs = ds["train_diff_pairs"]

    best_score = (baseline_metrics["roc_auc"], baseline_metrics["margin"])
    best_metrics = baseline_metrics
    best_state = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        # 冻结的底层 BN 保持 eval，避免小 batch 破坏统计量
        for mod_name in ("conv1", "bn1", "layer1", "layer2"):
            sub = getattr(model.backbone, mod_name, None)
            if sub is not None:
                sub.eval()

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            chosen_pids = random.sample(pid_pool, p_eff) if len(pid_pool) >= p_eff else list(pid_pool)
            # 若存在人工标注车辆，每 batch 至少掺入一部分人工车辆
            if manual_pids:
                m_list = list(manual_pids & set(pid_pool))
                if m_list:
                    inject = random.sample(m_list, min(len(m_list), max(1, p_eff // 3)))
                    chosen_pids = list(dict.fromkeys(inject + chosen_pids))[:p_eff]

            batch_imgs, batch_pids = [], []
            for pid in chosen_pids:
                items = by_pid[pid]
                picked = random.choices(items, k=k_instances) if len(items) < k_instances else random.sample(items, k=k_instances)
                for it in picked:
                    batch_imgs.append(_load_rgb_tensor(it["crop_path"], size_hw, augment=True))
                    batch_pids.append(pid)

            imgs_t = torch.stack(batch_imgs).to(device)
            pids_t = torch.tensor(batch_pids, dtype=torch.long, device=device)

            feats = _extract_features_batch(model, imgs_t)
            loss_tri = _batch_hard_cosine_triplet_loss(feats, pids_t, margin=0.25)
            loss_sup = _supcon_loss(feats, pids_t, temperature=0.08)
            loss = loss_tri + 0.5 * loss_sup

            # 显式人工配对损失：强拉近人工标"同车"、强推开人工标"不同车"（硬负样本！）
            pair_loss_terms = []
            if same_pairs:
                sub_same = random.sample(same_pairs, min(len(same_pairs), 4))
                t1 = torch.stack([_load_rgb_tensor(a, size_hw, augment=True) for a, _ in sub_same]).to(device)
                t2 = torch.stack([_load_rgb_tensor(b, size_hw, augment=True) for _, b in sub_same]).to(device)
                f1, f2 = _extract_features_batch(model, t1), _extract_features_batch(model, t2)
                cos_same = (f1 * f2).sum(dim=1)
                pair_loss_terms.append(F.relu(0.85 - cos_same).pow(2).mean() * 2.0)
            if diff_pairs:
                sub_diff = random.sample(diff_pairs, min(len(diff_pairs), 6))
                t1 = torch.stack([_load_rgb_tensor(a, size_hw, augment=True) for a, _ in sub_diff]).to(device)
                t2 = torch.stack([_load_rgb_tensor(b, size_hw, augment=True) for _, b in sub_diff]).to(device)
                f1, f2 = _extract_features_batch(model, t1), _extract_features_batch(model, t2)
                cos_diff = (f1 * f2).sum(dim=1)
                pair_loss_terms.append(F.relu(cos_diff - 0.45).pow(2).mean() * 2.5)
            if pair_loss_terms:
                loss = loss + sum(pair_loss_terms)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = round(epoch_loss / max(1, steps_per_epoch), 4)
        pct = 25 + int(65 * epoch / max(1, epochs))

        # 每 2 轮及最后一轮跑验证集
        if epoch % 2 == 0 or epoch == epochs:
            val_m = evaluate_reid_model(
                model, cfg, ds["val_samples"], ds["val_manual_pairs"], device
            )
            history.append({"epoch": epoch, "loss": avg_loss, **val_m})
            report_progress(
                f"Epoch {epoch}/{epochs} loss={avg_loss} | AUC={val_m['roc_auc']} | "
                f"Margin={val_m['margin']} (同车{val_m['same_mean']} vs 异车P90 {val_m['diff_p90']}) | "
                f"Rank-1={val_m['rank1']}",
                pct,
            )
            score_key = (val_m["roc_auc"], val_m["margin"])
            if score_key > best_score:
                best_score = score_key
                best_metrics = val_m
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            history.append({"epoch": epoch, "loss": avg_loss})
            report_progress(f"Epoch {epoch}/{epochs} loss={avg_loss}", pct)

    improved = (
        best_state is not None
        and (
            best_metrics["roc_auc"] > baseline_metrics["roc_auc"] + 1e-4
            or (best_metrics["roc_auc"] >= baseline_metrics["roc_auc"] - 1e-4
                and best_metrics["margin"] > baseline_metrics["margin"] + 0.005)
        )
    )
    activated = False
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"fastreid:{out_weights.name}@{version_tag}"

    if best_state is not None:
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state}, str(out_weights))
        if improved or force_activate:
            ACTIVE_META_PATH.write_text(
                json.dumps(
                    {
                        "weights": out_weights.name,
                        "model_version": model_version,
                        "activated_at": datetime.now().isoformat(timespec="seconds"),
                        "improved": improved,
                        "baseline": baseline_metrics,
                        "metrics": best_metrics,
                        "stats": stats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            activated = True

    report = {
        "mode": "finetune",
        "device": device,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
        "epochs": epochs,
        "stats": stats,
        "baseline": baseline_metrics,
        "finetuned": best_metrics,
        "improved": improved,
        "activated": activated,
        "model_version": model_version if activated else f"fastreid:{base_weights.name}",
        "out_weights": str(out_weights) if best_state is not None else "",
        "history": history,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_progress(
        f"微调结束：{'已验证提升并自动激活新权重' if activated else '验证指标未超过基线，保留原权重'}",
        100,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="园区车辆 FastReID 微调与留出集评估")
    parser.add_argument("--db", type=Path, default=APP_DIR / "granary.db")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--out-weights", type=Path, default=DEFAULT_OUT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p-identities", type=int, default=8)
    parser.add_argument("--k-instances", type=int, default=4)
    parser.add_argument("--manual-only", action="store_true",
                        help="只使用纯人工标注数据（不混入高置信车牌引导簇）")
    parser.add_argument("--eval-only", action="store_true",
                        help="只评估基线模型在留出集上的指标，不训练")
    parser.add_argument("--force-activate", action="store_true",
                        help="即使验证指标未超基线也激活微调权重")
    parser.add_argument("--reembed", action="store_true",
                        help="若激活了新权重，立即重刷全库 FastReID 特征并重聚车辆")
    args = parser.parse_args()

    report = run_finetune(
        args.db,
        config_file=args.config,
        base_weights=args.base_weights,
        out_weights=args.out_weights,
        device=args.device,
        epochs=args.epochs,
        p_identities=args.p_identities,
        k_instances=args.k_instances,
        lr=args.lr,
        include_verified_plates=not args.manual_only,
        eval_only=args.eval_only,
        force_activate=args.force_activate,
    )
    if args.reembed and report.get("activated"):
        import reid_pipeline
        print("[finetune-reid] 开始使用微调后的新权重重刷全库特征并重聚车辆...", flush=True)
        extractor = reid_pipeline.FastReIDExtractor(args.config, args.base_weights, args.device)
        embed_res = reid_pipeline.embed_pass(args.db, extractor, force=False, refresh_stale=True)
        conn = sqlite3.connect(args.db, timeout=60)
        assoc_res = build_db.associate_orders(conn)
        conn.close()
        report["reembed"] = embed_res
        report["associate"] = assoc_res
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
