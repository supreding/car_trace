#!/usr/bin/env bash
# 临时验证脚本：对两天子集(_nov试点_临时)按生产流水线顺序跑全流程，
# 最后输出未关联归类试运行报告。仅写 test_sub.db，不触碰生产库。
set -uo pipefail
cd /d/WorkSpace/car_trace/app
PY=../.venv-reid/Scripts/python.exe

echo "== [1/5] 同步入库 =="
$PY -X utf8 build_db.py --db test_sub.db --park _nov试点_临时 --sync-only || exit 1

echo "== [2/5] OCR 时间戳 =="
$PY -X utf8 build_db.py --db test_sub.db --park _nov试点_临时 --process-missing || exit 1

echo "== [3/5] GPU 流水线（detect/correct/plate/embed）=="
cd /d/WorkSpace/car_trace
powershell -NoProfile -ExecutionPolicy Bypass -File run_reid_pipeline.ps1 \
  -Db "D:\WorkSpace\car_trace\app\test_sub.db" \
  -OutputDir "D:\WorkSpace\car_trace\app\reid_crops_subnov" || exit 1

echo "== [4/5] 车辆聚合 =="
cd /d/WorkSpace/car_trace/app
$PY -X utf8 build_db.py --db test_sub.db --park _nov试点_临时 --associate-only || exit 1

echo "== [5/5] 未关联货车归类（试运行）=="
$PY -X utf8 build_db.py --db test_sub.db --assign-loose --park _nov试点_临时
echo "== 完成 =="
