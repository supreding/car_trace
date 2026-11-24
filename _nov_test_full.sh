#!/usr/bin/env bash
# 临时验证脚本：全量 11 月数据按生产流水线顺序跑到"未关联归类试运行"报告。
# 只写 test_nov.db，不触碰生产库；不做 --apply，不移动任何文件。
set -uo pipefail
cd /d/WorkSpace/car_trace/app
PY=../.venv-reid/Scripts/python.exe

echo "== [1/4] OCR 断点续跑 =="
$PY -X utf8 build_db.py --db test_nov.db --park 农易行临泉辉煌农事服务中心 --process-missing || exit 1

echo "== [2/4] GPU 流水线（detect/correct/plate/embed）=="
cd /d/WorkSpace/car_trace
powershell -NoProfile -ExecutionPolicy Bypass -File run_reid_pipeline.ps1 \
  -Db "D:\WorkSpace\car_trace\app\test_nov.db" \
  -OutputDir "D:\WorkSpace\car_trace\app\reid_crops_testnov" || exit 1

echo "== [3/4] 车辆聚合 =="
cd /d/WorkSpace/car_trace/app
$PY -X utf8 build_db.py --db test_nov.db --park 农易行临泉辉煌农事服务中心 --associate-only || exit 1

echo "== [4/4] 未关联货车归类（试运行）=="
$PY -X utf8 build_db.py --db test_nov.db --assign-loose --park 农易行临泉辉煌农事服务中心 2>&1 | tail -8
echo "== 完成 =="
