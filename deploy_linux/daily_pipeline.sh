#!/usr/bin/env bash
# 每日凌晨流水线：备份 -> 全园区同步(自动发现新园区)+OCR -> GPU 流水线（只处理新增/未处理）
# 由 car-trace-pipeline.timer 每天 03:00 触发；也可手动：systemctl start car-trace-pipeline
# 或直接: bash /opt/car-trace/deploy_linux/daily_pipeline.sh
set -uo pipefail
cd /opt/car-trace

# 防并发：上次没跑完时本次直接跳过（手动/定时冲突保护）
exec 9>/tmp/car-trace-pipeline.lock
flock -n 9 || { echo "[跳过] 上一次流水线仍在运行"; exit 0; }

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }
PY=.venv-reid/bin/python
log "== 每日流水线开始 =="

# ---- 0/3 数据库备份（sqlite 在线备份，保留近 7 份）----
log "== 0/3 备份数据库 =="
$PY - <<'PYEOF' || log "警告：备份失败（不阻断流水线）"
import glob, gzip, os, shutil, sqlite3, datetime
src = "/opt/car-trace/app/granary.db"
bdir = "/data/backups"
os.makedirs(bdir, exist_ok=True)
dst = os.path.join(bdir, f"granary-{datetime.date.today():%Y%m%d}.db.gz")
if not os.path.exists(dst):
    s = sqlite3.connect(src); d = sqlite3.connect("/tmp/_bak.db")
    s.backup(d); d.close(); s.close()
    with open("/tmp/_bak.db", "rb") as fin, gzip.open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove("/tmp/_bak.db")
for old in sorted(glob.glob(f"{bdir}/granary-*.db.gz"))[:-7]:
    os.remove(old)
print("备份完成:", dst)
PYEOF

# ---- 1/2 全园区同步（自动发现新园区）+ OCR 时间戳 ----
parks=$(ls -d data/*/ 2>/dev/null | sed 's:/$::' | xargs -n1 basename | tr '\n' ' ')
log "== 1/2 全园区同步+OCR（发现园区: ${parks:-无}）=="
if ! $PY app/build_db.py --db app/granary.db; then
    log "失败：同步/OCR 阶段"
    echo "FAILURE $(date '+%F %H:%M:%S') 同步/OCR" > /data/pipeline_status.txt
    exit 1
fi

# ---- 前置检查：无新增/待处理图片则跳过 GPU 流水线（备份和同步照常执行）----
pending=$($PY - <<'PYEOF'
import sqlite3
conn = sqlite3.connect("/opt/car-trace/app/granary.db")
# 与 detect_pass 的选择条件保持一致（含 vehicle_crop_path IS NULL）：
# 否则旧版已裁剪/照片已清理的帧会把待处理数永久顶在 >0，
# GPU 流水线每 30 分钟被定时器拉起重复空转。
detect = conn.execute(
    """SELECT COUNT(*) FROM captures
       WHERE coalesce(capture_status,'vehicle')!='no_vehicle'
         AND (status_reason IS NULL OR status_reason NOT LIKE '%与订单主车不符%')
         AND vehicle_crop_path IS NULL
         AND id NOT IN (SELECT DISTINCT capture_id FROM vehicle_instances)""").fetchone()[0]
embed = conn.execute(
    """SELECT COUNT(*) FROM captures
       WHERE coalesce(vehicle_crop_path,'')!='' AND reid_embedding IS NULL""").fetchone()[0]
conn.close()
print(detect + embed)
PYEOF
)
log "待处理抓拍（缺检测/缺特征）：${pending}"
if [ "$pending" = "0" ]; then
    log "无新增图片且无待处理项，跳过 GPU 流水线"
    echo "SKIPPED $(date '+%F %H:%M:%S') 无新增图片" > /data/pipeline_status.txt
    exit 0
fi

# ---- 2/2 GPU 流水线（只处理新增/未处理的抓拍）----
log "== 2/2 GPU 流水线 =="
if ! bash run_reid_pipeline.sh --db app/granary.db; then
    log "失败：GPU 流水线阶段"
    echo "FAILURE $(date '+%F %H:%M:%S') GPU流水线" > /data/pipeline_status.txt
    exit 1
fi

# ---- 2.5/2 GPU后收尾：新特征就位后重聚车辆 + 未关联货车归类 ----
# GPU 前的聚合看不到新抓拍的特征；归类依赖外观特征与时间窗，必须在
# 特征入库之后执行。归类默认试运行阈值已在 build_db 内置（0.55/0.03），
# 全部移动写 loose_assignments 日志，可 --undo-loose-assign 撤销。
log "== 2.5/2 车辆重聚 + 未关联货车归类 =="
if ! $PY app/build_db.py --db app/granary.db --assign-loose --apply; then
    log "警告：未关联货车归类失败（不阻断状态记录）"
fi
if ! $PY app/build_db.py --db app/granary.db --assign-gate --apply; then
    log "警告：大门照片重归类失败（不阻断状态记录）"
fi
# 车辆重聚放最后：让重聚类看到归类后的最新订单归属
if ! $PY app/build_db.py --db app/granary.db --associate-only; then
    log "警告：车辆重聚失败（不阻断状态记录）"
fi

echo "OK $(date '+%F %H:%M:%S')" > /data/pipeline_status.txt
log "== 每日流水线完成 =="
