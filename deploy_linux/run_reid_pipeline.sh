#!/usr/bin/env bash
# 识别流水线环境封装（Linux 版，对应 Windows 的 run_reid_pipeline.ps1）
# 用法:
#   ./run_reid_pipeline.sh                                  # 处理默认库 app/granary.db
#   ./run_reid_pipeline.sh --db app/nongyixing.db \
#       --output-dir app/reid_crops_nongyixing              # 多园区：每库独立库与裁剪目录
#   可选: --limit N（试跑）  --force（重跑已处理）  --no-attr（跳过车辆属性）
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$project_root/.venv-reid/bin/python"
paddle_root="$project_root/third_party/PaddleDetection"
fastreid_root="$project_root/third_party/fast-reid"
runtime_home="$project_root/.reid-home"
mpl_config="$project_root/.mpl"

if [ ! -x "$python" ]; then
    echo "ReID Python 环境不存在：$python"
    echo "先运行 bash deploy_linux/install_server.sh 创建环境，再重试。"
    exit 1
fi
mkdir -p "$runtime_home" "$mpl_config"

# GPU 版 torch 与 paddlepaddle-gpu 的 cuDNN 库不能同进程共存；
# reid_pipeline.py --stage all 会自动拆成 detect / embed 两个子进程。
export PADDLE_DETECTION_ROOT="$paddle_root"
export PYTHONPATH="$fastreid_root:$paddle_root"
export HOME="$runtime_home"
export MPLCONFIGDIR="$mpl_config"

db=""; output_dir=""; limit=0; force=""; noattr=""
while [ $# -gt 0 ]; do
    case "$1" in
        --db)         db="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --limit)      limit="$2"; shift 2 ;;
        --force)      force="--force"; shift ;;
        --no-attr)    noattr="--no-attr"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
[ -z "$db" ] && db="$project_root/app/granary.db"
[ -z "$output_dir" ] && output_dir="$project_root/app/reid_crops"
if [ ! -f "$db" ]; then
    echo "数据库不存在：$db（先用 build_db.py 同步建库，或用 --db 指定）"
    exit 1
fi

cmd=("$python" "$project_root/app/reid_pipeline.py"
     "--db" "$db"
     "--detector-model" "$project_root/models/paddledetection/mot_ppyoloe_l_36e_ppvehicle"
     "--attr-model" "$project_root/models/paddledetection/vehicle_attribute"
     "--plate-det-model" "$project_root/models/paddledetection/ch_PP-OCRv3_det_infer"
     "--plate-rec-model" "$project_root/models/paddledetection/ch_PP-OCRv3_rec_infer"
     "--fastreid-config" "$fastreid_root/configs/VeRi/sbs_R50-ibn.yml"
     "--fastreid-weights" "$project_root/models/fastreid/veri_sbs_R50-ibn.pth"
     "--output-dir" "$output_dir"
     "--device" "cuda:0")
[ "$limit" -gt 0 ] && cmd+=("--limit" "$limit")
[ -n "$force" ] && cmd+=("$force")
[ -n "$noattr" ] && cmd+=("$noattr")

# 切到项目根执行：库里的相对路径（data/...、reid_crops/...）按项目根解析
cd "$project_root"
exec "${cmd[@]}"
