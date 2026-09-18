#!/usr/bin/env bash
# 全量流水线编排：山高库(granary) -> 农易行库(nongyixing)，每库依次
# 同步照片入库(+OCR 时间戳+订单聚合) -> GPU 检测/属性/车牌/FastReID 特征。
# 用法: nohup bash run_full_pipeline.sh > /data/pipeline_full.log 2>&1 &
set -uo pipefail
cd /opt/car-trace

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

log "== 1/4 山高库（granary.db）同步照片+OCR =="
if ! .venv-reid/bin/python app/build_db.py --db app/granary.db --park 山高荃银丰麦田库; then
    log "失败：山高库同步"; exit 1
fi

log "== 2/4 山高库 GPU 流水线 =="
if ! bash run_reid_pipeline.sh --db app/granary.db; then
    log "失败：山高库流水线"; exit 1
fi

log "== 3/4 农易行库（nongyixing.db）同步照片+OCR =="
if ! .venv-reid/bin/python app/build_db.py --db app/nongyixing.db --park 农易行临泉辉煌库; then
    log "失败：农易行库同步"; exit 1
fi

log "== 4/4 农易行库 GPU 流水线 =="
if ! bash run_reid_pipeline.sh --db app/nongyixing.db --output-dir app/reid_crops_nongyixing; then
    log "失败：农易行库流水线"; exit 1
fi

log "== 全部完成 =="
