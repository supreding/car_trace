# 车辆轨迹查询系统 API 接口清单

更新时间：2026-09-10　对应代码：`app/server.py`　线上环境：阿里云 ECS（47.101.32.59）

接口分两组，**用途和防护等级不同**：

- **内部页面接口**（无前缀）：给本系统前端页面用，无鉴权、返回裸数组/裸对象。
  含写操作接口，**只允许在内网使用**，不要暴露给其他系统。
- **对外开放接口**（`/api/open/` 前缀）：给第三方系统对接用，只读、token 鉴权、统一响应格式。

---

## 〇、多园区隔离（2026-09 新增）

- 数据目录 `data/<园区名>/<月份照片根>/<日期>/<订单号>/`，一级子目录名即园区，
  记录在每条抓拍的 `park` 字段（老库无园区数据时为空串，展示为"未标注园区"）。
- **车辆身份（vehicle_id）按园区独立聚类**：同一车牌出现在多个园区会得到相互
  独立的车辆编号与轨迹，系统绝不会把一个园区的车追踪到另一个园区
  （聚类合并、以图搜车、人工移动/合并均在园区内进行）。
- 订单号在**园区内**唯一：不同园区可能使用同一套系统导致订单号撞号，
  订单聚合键为 (park, order_id)。按订单号查轨迹时若该订单号在多个园区存在，
  接口返回 400 并列出可选园区，需带 `park` 参数重试。
- 查询接口普遍支持可选 `park` 参数：缺省=全部园区；带 `park` 只看一个园区。
- `GET /api/open/parks`、`GET /api/parks` 返回园区清单及数量。

---

## 一、对外开放接口 `/api/open/`（对接方只看这一节）

### 1.1 通用约定

| 项目 | 约定 |
|---|---|
| Base URL | `http://47.101.32.59`（80 端口，无需带端口号） |
| 方法 | 全部只读，仅 `GET` |
| 鉴权 | 请求头 `X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d`（推荐）；图片类接口也支持 `?token=8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d` 查询参数，便于直接嵌入 `<img>` 标签。token 由服务端环境变量 `OPEN_API_TOKEN` 配置（systemd 单元内），**未配置时整组接口返回 503 停用**。当前线上 token：`8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d`（同存于服务器 `/data/car-trace/deploy_linux/open_api_token.txt`） |
| 响应格式（列表） | `{"code": 0, "message": "ok", "total": 总条数, "page": 页码, "page_size": 每页条数, "items": [...]}` |
| 响应格式（单对象） | `{"code": 0, "message": "ok", "data": {...}}` |
| 响应格式（错误） | `{"code": <HTTP状态码>, "message": "错误说明"}`，HTTP 状态码与 code 一致 |
| 分页 | `page` 从 1 开始；`page_size` 默认 50，上限 500 |
| 日期参数 | `date`（单日）、`date_from` / `date_to`（闭区间），格式 `YYYY-MM-DD`，三者可单独或组合使用，非法格式返回 400 |
| 园区参数 | `park`（可选，园区名）：只看指定园区；缺省=全部园区。车辆/订单轨迹按园区隔离，不会跨园区串联 |
| 缓存 | 所有接口响应带 `Cache-Control: no-store`，请勿做本地长缓存 |

**错误码**：`400` 参数错误（如日期格式）；`401` token 无效或缺失；`404` 资源不存在；`503` 服务端未配置 `OPEN_API_TOKEN`（接口停用）。

### 1.2 接口列表

| 接口 | 说明 | 专有参数 |
|---|---|---|
| `GET /api/open/ping` | 连通性与 token 自检，接入前先调它验证 token | — |
| `GET /api/open/parks` | 园区清单及各园区抓拍/订单/车辆数（park 为空串=未标注园区） | — |
| `GET /api/open/dates` | 有数据的日期清单及每日抓拍数、订单数 | `park` |
| `GET /api/open/orders` | 订单列表。订单号在园区内聚合（跨日期只出一行）；按 order_id 排序（翻页稳定）。items 含 `park` 字段 | `park`、`date` / `date_from` / `date_to`、`order_id`（模糊）、`stage`、`page`、`page_size` |
| `GET /api/open/order/<order_id>` | 单个订单完整轨迹（跨日期聚合，按时间+阶段排序），含订单属性 `attributes`。订单号跨园区撞号时返回 400 + 可选园区列表 | `park` |
| `GET /api/open/vehicles` | 车辆聚合列表（vehicle_id 分组，按园区隔离），`rep_id` 是代表抓拍 ID，`plate` 为出现最多的识别车牌，items 含 `park` | `park`、`date` / `date_from` / `date_to`、`page`、`page_size` |
| `GET /api/open/vehicle/<vehicle_id>` | 单车完整轨迹（车辆只属于一个园区） | — |
| `GET /api/open/captures` | 抓拍明细，多条件任意组合（全部为 AND） | `park`、`date` / `date_from` / `date_to`、`order_id`（精确）、`plate`（片段，匹配识别车牌和 OSD 车牌）、`vehicle_type`、`start_hour` / `end_hour`（0-23）、`page`、`page_size` |
| `GET /api/open/crop/<capture_id>` | 车辆裁剪图（JPEG 二进制），`capture_id` 取自 items 里的 `id` 或 `rep_id` | 支持 `?token=` |

### 1.3 字段说明（items / data.items 内）

订单字段：`order_id` 订单号、`capture_count` 抓拍数、`date_span` 跨越天数、`dates` 涉及日期、`first_seen` / `last_seen` 首末时间、`cropped` 有车辆裁剪的张数、`no_vehicle` 未检出车辆的张数、`vehicle_color` / `vehicle_type` 车辆颜色/车型（属性模型）、`rep_id` 可用于取图的抓拍 ID、`attr_conf` 属性置信度 `{color, type}`、`order_time` 订单大致发生时间（订单内最早的有效抓拍时间，可能为空）、`plate` 订单车牌号（整单唯一：识别车牌多数票，票数相同取置信度高者；整单无识别车牌时回退 OSD 参考车牌；"未关联图片"伪订单不产出车牌）。

车辆字段：`vehicle_id` 车辆聚合编号、`rep_id` 代表抓拍、`capture_count` 出现次数、`order_ids` 关联订单号（逗号分隔）、`plate` 主识别车牌、`plate_confidence` 车牌置信度、`color` / `vehicle_type` 颜色/车型、`camera_count` 涉及摄像头数、`first_seen` / `last_seen`。

抓拍明细字段（`/api/open/captures` 与订单轨迹的 items 相同）：`id` 抓拍 ID（用于取图）、`date` 日期、`order_id` 订单号、`stage` 环节（入磅/质检/出磅等）、`camera_key` 摄像头、`capture_time` OCR 识别时间（可能为空）、`hour` 小时、`plate_final` 最终车牌、`osd_plate` 屏幕 OSD 参考车牌、`plate_confidence` 车牌置信度、`vehicle_color` / `vehicle_type` 及对应 `_conf` 置信度、`vehicle_crop_path` 裁剪图相对路径、`vehicle_id` 聚合车辆编号、`quality_score` 质量分、`capture_status`（`no_vehicle` 表示该帧未检出车辆）。

### 1.4 调用示例

```bash
# 自检 token
curl -H "X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d" http://47.101.32.59/api/open/ping

# 查 2026-08-05 至 2026-08-08 的订单（第 1 页，每页 100 条）
curl -H "X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d" \
  "http://47.101.32.59/api/open/orders?date_from=2026-08-05&date_to=2026-08-08&page=1&page_size=100"

# 查某天的车辆，再取某辆车的完整轨迹
curl -H "X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d" "http://47.101.32.59/api/open/vehicles?date=2026-08-05"
curl -H "X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d" "http://47.101.32.59/api/open/vehicle/122"

# 按车牌片段 + 时段查抓拍明细
curl -H "X-API-Token: 8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d" \
  "http://47.101.32.59/api/open/captures?date=2026-08-05&plate=CE1929&start_hour=8&end_hour=18"

# 车辆裁剪图（查询参数方式鉴权，可直接嵌 <img src>）
<img src="http://47.101.32.59/api/open/crop/3088?token=8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d">
```

```python
import requests

BASE = "http://47.101.32.59"
HEADERS = {"X-API-Token": "8cc5318daf2faaeb93dbf21c030dcc053deae8afc5f4c10d"}

resp = requests.get(f"{BASE}/api/open/orders",
                    params={"date": "2026-08-10", "page": 1, "page_size": 100},
                    headers=HEADERS, timeout=10)
data = resp.json()
if data["code"] == 0:
    print(data["total"], len(data["items"]))
    # 取订单轨迹
    order = data["items"][0]["order_id"]
    detail = requests.get(f"{BASE}/api/open/order/{order}", headers=HEADERS, timeout=10).json()
```

---

## 二、内部页面接口（前端自用，无鉴权，勿对外暴露）

返回格式不统一（裸数组或裸对象），错误为 `{"error": "..."}`，仅供 `templates/` 页面消费。

### 2.1 页面与自检

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 查询主页 |
| GET | `/annotate` | 标注复核页 |
| GET | `/api/health` | 部署自检：数据库连通性、Python 版本 |

### 2.2 枚举与统计

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/parks` | 园区清单及各园区抓拍/订单/车辆数 |
| GET | `/api/dates` | 日期列表及每日抓拍数（可选 `park`） |
| GET | `/api/stats` | 全库统计（抓拍数、订单数、车辆数、摄像头数等，可选 `park`；含 `park_items` 各园区张数） |
| GET | `/api/stages` | 环节（stage）枚举及数量（可选 `park`） |
| GET | `/api/facets` | 属性模型产出的车型/颜色枚举（可选 `park`） |
| GET | `/api/color_type_codes` | 人工标注用的颜色/车型代码表 |

### 2.3 查询（读）

以下接口均支持可选 `park` 参数过滤园区：

| 方法 | 路径 | 参数 | 说明 |
|---|---|---|---|
| GET | `/api/orders` | `date`、`stage`、`q`（订单号模糊）、`park`、`limit`（≤1000，默认 200） | 订单聚合列表（按 park+order_id 分组），按可读性排序（无效抓拍沉底） |
| GET | `/api/order/<order_id>` | `park` | 订单完整轨迹 + 属性，不存在返回 404；订单号跨园区撞号且未带 park 时返回 400 + parks |
| GET | `/api/vehicles` | `date`、`stage`、`park` | 车辆聚合列表（vehicle_id 分组，园区隔离） |
| GET | `/api/captures` | `date`、`stage`、`order_id`、`park`、`limit`（≤500）、`offset` | 抓拍明细，offset 分页 |
| GET | `/api/search` | `plate`（片段）、`vehicle_type`、`start_hour`、`end_hour`、`order_id`（模糊）、`date`、`park`、`limit`（≤2000） | 组合查询，至少一个条件否则 400 |
| GET | `/api/track` | `vid`（车辆编号）或 `plate`（片段）+`date`+`park` | 按车辆/车牌查轨迹；同牌车跨园区时返回各园区独立轨迹段 |
| GET | `/api/track_by_img` | `id`（抓拍 ID） | 以图搜车：FastReID 余弦检索 + 属性重排，**候选只在同园区内** |
| GET | `/api/instances/<capture_id>` | — | 同帧检测到的其他车辆实例 |

### 2.4 图片（读）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/thumb/<capture_id>` | 抓拍缩略图（480×320 JPEG） |
| GET | `/api/image/<capture_id>` | 抓拍原图 |
| GET | `/api/crop/<capture_id>` | PaddleDetection 整车裁剪图 |
| GET | `/api/instance_crop/<instance_id>` | 同帧实例裁剪图 |

### 2.5 编辑与标注（写操作，会改数据库）

| 方法 | 路径 | 请求体（JSON） |
|---|---|---|
| POST | `/api/capture_update` | `{id, plate?, color?, type?, capture_time?, order_id?, note?, reviewed?}` 人工复核修改 |
| POST | `/api/vehicle_move` | `{capture_id, target_plate（null=拆出为独立车辆）, scope: "capture"|"order"}` 人工移动车辆归属 |
| POST | `/api/vehicle_move_delete` | `{capture_id}` 撤销人工移动 |
| POST | `/api/vehicle-merge` | `{vehicle_ids: [..], target_vehicle_id?}` 合并多个车辆编号 |
| GET | `/api/boxedit/<capture_id>` | 框选编辑器上下文：原图尺寸、当前主车框、同帧实例列表（含主车）、人工修正标记 |
| POST | `/api/boxedit/save` | `{capture_id, box: [l,t,r,b]}` 人工框选：按手绘框从原图重裁主车缩略图（框为原图像素坐标，自动收拢图幅，最小 24×24） |
| POST | `/api/boxedit/promote` | `{capture_id, instance_id}` 同帧多车切换主车：实例内容与 0 号记录对调，特征复用实例 embedding，归属立即重聚类 |
| POST | `/api/boxedit/reset` | `{capture_id}` 撤销人工框选标记（下次 detect --force 恢复自动检测） |
| POST | `/api/annotations/capture` | `{capture_id, action, usability, identity_id?, corrected_plate?, notes?, annotator}` 单张标注 |
| POST | `/api/annotations/pair` | `{capture_id, candidate_id, relation: "same"|"different"|"uncertain", notes?, annotator, confirm_merge?}` 成对判定 |
| GET | `/api/annotations/stats` | 标注进度统计 |
| GET | `/api/annotations/queue` | `date`、`stage`、`state`、`limit` 标注队列 |
| GET | `/api/annotations/item/<capture_id>` | `limit` 单张标注详情及候选 |
| GET | `/api/annotations/export.csv` | 导出人工真值 CSV（FastReID 微调用） |

**人工框选修正语义**：框变更后车牌/外观特征按断点续跑规则清空（`lpr_attempted_at`、
`reid_embedding`），由下轮流水线 plate/embed 阶段增量补算；决定记入 `manual_box_edits`
表，`detect_pass`（含 `--force`）与 `correct_pass` 跳过这些抓拍，人工修正不会被自动
覆盖。撤销标记请用 `/api/boxedit/reset`。

---

## 三、线上服务器运维（阿里云 ECS 47.101.32.59）

**部署布局**：项目实际位于 **`/data/car-trace`**（500G 数据盘，照片/数据库/裁剪图全在此盘），
`/opt/car-trace` 为指向它的软链接。数据库为**单库多园区**（`app/granary.db`，当前含
山高荃银丰麦田库 4908 张 + 农易行临泉辉煌库 3394 张），`app/nongyixing.db` 为合并前
的历史库（仅留档，未被服务使用）。

**查询服务**：systemd 服务 `car-trace`（waitress，监听 0.0.0.0:80），开机自启、崩溃自动拉起。

```bash
systemctl status  car-trace        # 状态
systemctl restart car-trace        # 重启（改代码/配置后执行）
journalctl -u car-trace -f         # 服务日志
```

**每日流水线**：systemd 定时器 `car-trace-pipeline.timer`，**每天 03:00（北京时间）自动执行**：
数据库备份（`/data/backups/`，gzip，保留 7 份）→ 全园区照片同步（**自动发现新园区**）+ OCR
→ GPU 流水线（**只处理新增/未处理**）。无新增图片时自动跳过 GPU（状态记为 SKIPPED）。

```bash
systemctl list-timers car-trace-pipeline.timer   # 下次触发时间
systemctl start car-trace-pipeline               # 手动立即触发（上传急用）
cat /data/pipeline_status.txt                    # 最近一次结果：OK / SKIPPED / FAILURE
journalctl -u car-trace-pipeline -f              # 运行日志
```

**新园区接入**：照片放 `/data/car-trace/data/<园区名>/<月份照片根>/<日期>/<订单号>/`，
等 03:00 自动处理或手动触发；页面园区下拉框自动出现新园区，无需改任何配置/代码。

**人工合并可靠性**：页面上的"合并车辆"决定按抓拍锚定持久化（`manual_vehicle_merge_pairs` 表），
每日重聚类后自动重放，不受车辆编号重排影响；人工"移动/拆出"同样持久生效。
注意：`vehicle_id` 编号本身每次重聚类会变化，对外系统持久引用请用 `order_id + park`。

**历史备份**：`app/granary.db.bak-20260907`（相对路径迁移前）、
`app/granary.db.bak-merge-20260909`（双库合并前），确认无用后可删。

**Windows 开发机**：查询服务 `启动查询服务.bat`、流水线 `启动识别流水线.bat`（需 GPU + Python 3.12，
依赖见 requirements-*.txt）；部署包生成脚本 `make_deploy_package.py`。

