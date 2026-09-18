# 园区车辆轨迹追踪项目：PaddleDetection + FastReID 16 步实施方案

本文档是园区车辆 ReID 项目的落地顺序。目标是精准各个监控图片中出现的每一辆车并把他们形成一个完整的轨迹，供用户查看，并把“车辆检测/裁剪”和“跨摄像头车辆外观检索”拆成两个可独立迭代的模型：

```text
原始抓拍
   │
   ▼
PaddleDetection（检测 vehicle，输出 bbox/score）
   │
   ▼
车辆裁剪（带少量边缘，保存 crop）
   │
   ├──► 车辆属性模型（颜色、车型 -> 结构化字段）
   │
   ▼
FastReID（输出归一化 embedding）
   │
   ├── 车牌、时间、摄像头等结构化信息
   ▼
向量检索 + 属性/时空过滤 + 人工复核
   │
   ▼
车辆轨迹、车辆身份库、持续微调数据
```

## 1. 明确业务目标和验收指标

先固定项目边界，避免把“检测不到车”“车牌识别错误”“同车匹配错误”混成一个指标。

建议首期验收指标：

- 车辆检测召回率：白天、夜间、雨天分别统计，目标先达到 95% 左右。
- 单摄像头同车匹配 Top-1 / Top-5。
- 跨摄像头同车匹配 Top-1 / Top-5。
- 误合并率（不同车辆被判成同车）和漏合并率（同车未被聚到一起）。
- 端到端延迟：从图片进入到可检索的时间。
- 每天处理量：按 20 路摄像头和峰值图片量压测，而不是只看平均值。

验收集必须按摄像头、时段、天气、车型、遮挡程度分层，不能只报一个总体准确率。

## 2. 固定目录、数据和模型版本

当前工作区目录结构如下（`app/` 与批处理脚本待从前一版项目迁移）：

```text
D:\WorkSpace\car_trace\
├─ 2026年8月份抓拍照片/          # 原始抓拍：日期/订单号/阶段_uuid.jpg
├─ app/                          # 待迁移
│  ├─ granary.db                 # SQLite 数据库（crop、属性、embedding、标注）
│  ├─ reid_pipeline.py           # 检测、裁剪、属性识别、FastReID 批处理
│  ├─ reid_crops/                # PaddleDetection 生成的车辆裁剪
│  ├─ server.py                  # Flask 查询和标注接口
│  └─ templates/annotate.html    # 人工标注页面
├─ models/
│  ├─ paddledetection/
│  │  ├─ mot_ppyoloe_l_36e_ppvehicle/  # 检测模型
│  │  └─ vehicle_attribute/            # 颜色/车型属性模型（PULC PPLCNet，待下载）
│  └─ fastreid/                  # ReID 配置和权重
├─ third_party/
│  ├─ PaddleDetection/
│  └─ fast-reid/
└─ run_reid_pipeline.ps1        # Windows 批处理入口（待迁移）
```

每次训练或替换模型都要记录：源码 commit、配置文件、权重文件 SHA256、训练数据版本和阈值。数据库中的 `reid_model_version` 用于追溯特征来源，属性模型对应 `attr_model_version`。

**迁移状态（2026-08-31 已完成）**：本项目已在本工作区落地。`app/`（build_db.py 扫描入库、reid_pipeline.py 检测/裁剪/属性/特征、server.py 查询标注、templates/、granary.db）、`run_reid_pipeline.ps1` 均已按本工作区的照片结构（日期/订单号/阶段_uuid.jpg）新建，旧项目（C:\Users\dell\WorkBuddy\2026-08-12-16-25-45）仅作参考，不再依赖。要点差异：

- Python 环境为 `.venv-reid`（Python 3.12 + torch 2.12.1+cu126 + paddlepaddle-gpu 3.3.1，RTX A1000 实测可用）。GPU 版 torch 与 paddle 的 cuDNN DLL 不能同进程共存，`reid_pipeline.py --stage all` 会自动拆成 detect（PaddleDetection）/ embed（FastReID）两个子进程顺序执行；
- 全量数据已处理完毕：4908 张全部入库，检测裁剪 4091 张（无车 817 张已标记），属性 4091 张，FastReID embedding 4079 张；OCR 时间戳 ok 85.6%、fallback 14.3%（如实标记"时间不确定"，按阶段顺序兜底）、conflict 0.1%；
- 订单内不再用 embedding 阈值聚类（VeRi 预训练模型跨视角区分度有限），一个订单=一辆车一个 vehicle 分组，"未关联图片"桶（985 张抓拍）不分组，留给以图搜车与人工复核；
- 模型权重 SHA256 登记见 `models/MODEL_MANIFEST.md`。

## 3. 采集和整理原始抓拍

原始图片至少保留：

- 原图文件；
- 拍摄时间（精确到秒或毫秒）通过图片左上角OCR文字；
- 过程阶段（文件名前缀）。实测共四种：入磅、出磅、抓拍、质检，当前 4908 张的分布为入磅 1503、出磅 1486、抓拍 1372、质检 547；
- 订单号（图片的文件夹名是订单号）；
- 车牌识别结果及置信度（如果已有）；
- 图片是否属于车辆抓拍。
- **重要，有的图片下方写了摄像头识别出来的车辆形象，应该OCR文字识别出来，但不能直接采用，因为摄像头算力有限经常出错，但应该作为参考**

文件名不能作为唯一车辆身份。相同车辆可能换时间、换摄像头，也可能因为补传而出现重复文件名。

### 3.1 订单号是聚合唯一键，不能按日期文件夹处理

实测 806 个订单目录中有 42 个订单号同时出现在两个日期文件夹里（订单跨午夜，按出磅日期归档）。例如 68002608090040 的入磅、抓拍、质检图在"8月9日"，出磅图在"8月10日"。因此：

- 聚合、入库、批处理都以订单号为唯一键，必须全盘扫描所有日期文件夹；
- 任何"以日期文件夹为单位"的处理都会把这 42 个订单拆成两半；
- 订单号中编码的日期（如 68002608090040 中的 0809）与所在日期文件夹可能不一致，两者不能互为校验依据。

### 3.2 时间戳 OCR 需要单独实现和验收

时间戳是轨迹排序的基石，不能只当作一个数据字段：

- 用 PaddleOCR 识别左上角时间文字，正则解析成时间戳；
- 与订单号中的日期、文件名阶段做一致性校验，矛盾时记录告警；
- OCR 失败或结果明显矛盾时降级：按阶段前缀顺序（入磅->质检->出磅）+ 订单号推断先后，记录 `ocr_status`，图片标记 `uncertain`。

## 4. 建立摄像头和时空约束表

不需要完整的摄像头标定表，但需要一个轻量的 `camera_key` 字段。实测各阶段分辨率不同（入磅/出磅 2560×1440、质检 1280×720、抓拍 1920×1264），说明本质上是多路不同摄像头：

- `camera_key` 由文件名阶段前缀 + 分辨率派生（如 `入磅@2560x1440`），入库保存；
- 它是第 13.1 步判同规则"摄像头和时间符合通行顺序"的数据基础；
- 订单内的先后顺序用阶段前缀（入磅->质检->出磅）判定，比 OCR 时间戳更可靠；抓拍没有固定位置，用 OCR 时间戳插入；
- 质检、抓拍两路分辨率低，crop 质量差一档，ReID 相似度阈值需要按 `camera_key` 分开调。

## 5. 安装并验证 PaddleDetection

使用官方 PaddleDetection 源码和 PPVehicle 模型，如果没有或无法运行，前往 github 下载。当前项目的检测模型目录为：

```text
models/paddledetection/mot_ppyoloe_l_36e_ppvehicle/
```

该目录应至少包含 `infer_cfg.yml`、模型文件和参数文件。启动前验证：

```powershell
Test-Path models\paddledetection\mot_ppyoloe_l_36e_ppvehicle\infer_cfg.yml
Test-Path third_party\PaddleDetection\deploy\python\infer.py
```

PaddleDetection 的职责是检测并输出车辆框，另提供车辆颜色/车型属性模型（见第 9.1 步），两者都不负责跨摄像头车辆身份。

**车牌识别（2026-09-02 追加）**：使用 PP-Vehicle 官方 VEHICLE_PLATE 组件（ch_PP-OCRv3_det + ch_PP-OCRv3_rec + 车牌专用字典），实现为 `reid_pipeline.py` 的 `--stage plate`：对每张车辆裁剪图跑车牌检测+识别，结果写入 `lpr_plate/lpr_conf`，并按"模型为主、OSD 为参考"融合出 `plate_final`（一致时来源记 `OSD+车牌模型一致` 并加成置信度）。注意两点：det 限边用 1280/min（官方默认 736 对远距离小车牌放大不足，会大面积漏检）；识别结果过质量闸（归一化后 ≥5 位且置信度 ≥0.55）才采信，否则回退 OSD 参考值。全量实测：4091 张裁剪识别出 1992 张车牌（置信度 0.9+ 占 91%），有车牌记录从 693 增至 2157；远距离小车牌（<60 像素）受分辨率限制无法识别，由多帧投票与人工复核兜底。

`models/paddledetection/` 下还留有 `mot_ppyoloe_l_36e_ppvehicle.zip`（约 190MB），确认解压目录可用后可删除。

## 6. 安装并验证 FastReID

当前使用 VeRi 数据集预训练的 SBS R50-IBN，如果没有或无法运行，前往 github 下载：

```text
third_party/fast-reid/configs/VeRi/sbs_R50-ibn.yml
models/fastreid/veri_sbs_R50-ibn.pth
```

FastReID 的输入是车辆裁剪图，输出归一化向量。本项目验证过输出维度为 2048，保存为 float32 BLOB；后续可替换为在园区数据上微调后的权重。 

## 7. 单图检测

选取白天、夜间、远距离、遮挡和不同摄像头各若干张图片，确认：

- 能检测到车辆；
- bbox 没有越界；
- 车辆类别和置信度合理；
- 多车图片中选择策略符合业务（首期可按 score 最高，后续扩展为每车一条记录）。

多车图片必须把 `detection_count`（检测到的车辆数）写入数据库，`detection_count > 1` 的图片列入人工复核清单。抓拍镜头拍到多辆车的概率不低，只取 score 最高会静默丢掉其余目标，必须可追溯。

主体车辆选框（2026-09-02 改进）：不用单纯最高分——低置信度时最高分常落在画面边缘的局部车辆上。改为 **检测置信度 × 面积 × 中心度** 加权（`choose_main`），并把选框分数存入 `vehicle_det_score`，页面 <80% 标黄提示。

属性一致性复核（`--stage correct`，2026-09-02 新增）：利用第 9.1 步的颜色/车型结果反查选框质量。订单级共识（≥2 张、置信度 ≥0.6）建立后，若某帧选框颜色与共识冲突、或车型置信度 <0.7，则以 0.2 阈值重检原图并对每个候选框跑属性，选与共识最匹配的替换；帧内确实没有主车时保留原框并标记"属性冲突待复核"（如订单 68002608010003 中入磅帧拍到的蓝色三轮车，与主车 皖KW1636 无关）。复核只影响选框与标记，不删除任何图片。

检测阈值建议从 0.35 开始，必须在验证集上调参。阈值越低召回越高，但会增加背景误检和后续人工量。

## 8. 生成车辆裁剪图

裁剪步骤：

1. 读取原图并处理 EXIF 方向；
2. 使用 PaddleDetection 输出的 bbox；
3. 在 bbox 四周增加约 8% 边缘；
4. 截断到图片边界；
5. 保存 JPEG 裁剪图；
6. 把 bbox、裁剪来源和相对路径写入数据库。

当前实现由 `app/reid_pipeline.py` 完成，写入字段：

```text
vehicle_crop_path = reid_crops/00000003.jpg
vehicle_box       = [x1, y1, x2, y2]
crop_source       = PaddleDetection
```

没有检测到车辆的图片会跳过，不会生成伪造 embedding。

## 9. 提取车辆属性和 ReID embedding

第 8 步的裁剪图有两个消费者：车辆属性模型（9.1）和 FastReID（9.2）。

### 9.1 车辆颜色和车型属性识别

第 12 步"按车型查询"和第 13.1 步判同规则"车身颜色、车型一致"都依赖颜色/车型字段，本节是这些字段唯一的数据来源。

- 模型：PaddleDetection PP-Vehicle 的车辆属性模型（PULC PPLCNet，几 MB，CPU 推理也很快），本地推理代码在 `third_party/PaddleDetection/deploy/pipeline/ppvehicle/vehicle_attr.py`（`VehicleAttr` 类），输入就是第 8 步的车辆裁剪图；
  - **预处理必须按 PaddleClas 官方训练配置**（`ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml`）：`DecodeImage to_rgb=True` → **输入是 RGB 通道顺序**，按 deploy 流程的 BGR 惯例喂图会让红/蓝完全互换（红色货车全被识别成 blue，2026-08-31 已踩坑并修复）；Resize 为 **宽 256 × 高 192**（infer_cfg.yml 的 `target_size [192,256]` 是 [h,w] 顺序）；归一化先乘 1/255 再减 ImageNet mean、除 std；
- 权重：放在 `models/paddledetection/vehicle_attribute/`，需要下载 `vehicle_attribute_model.zip`（下载地址见 `deploy/pipeline/config/infer_cfg_ppvehicle.yml` 的 `VEHICLE_ATTR.model_dir`，本地 download.py 目前只登记了人像属性模型），解压后按第 2 步要求登记 SHA256；
- 输出：10 种颜色 + 9 种车型（sedan/suv/van/hatchback/mpv/pickup/bus/truck/estate），默认阈值 0.5，低于阈值输出 Unknown。

落库为结构化字段，**不并入向量库**：

```text
vehicle_color       = white
vehicle_color_conf  = 0.83
vehicle_type        = truck
vehicle_type_conf   = 0.97
attr_model_version  = paddledet:vehicle_attribute_model
```

明确禁止把属性 one-hot 拼进 2048 维 embedding、或与 ReID 向量共用同一个向量索引：ReID 向量的语义是"整体外观相似度"，颜色/车型是低基数离散属性，拼接会破坏余弦相似度的含义，而且属性用 SQL 过滤比向量检索更高效、更可解释。属性字段的三个用途：

1. 检索前过滤/检索后重排：车型不一致的候选直接剔除或降权；
2. 判同证据：车型冲突是"不同车"的强证据；颜色在低置信度时只是软证据（夜间/逆光下白、灰、银、蓝经常互相混淆，车型判断比颜色稳得多）；
3. 第 12 步"按车型查询"直接变成 SQL 查询。

补充规则：

- 同一订单的多张 crop 各自出属性结果后，按多数投票聚合成订单级属性，进一步降噪；
- 上线前抽 100~200 张本场景 crop 人工标注，实测颜色/车型准确率后再定阈值，默认 0.5 未必适合夜间磅房；
- 9 类车型没有半挂/自卸/厢式等细分，重型卡车都归 truck，"按车型查询"的粒度到此为止；需要更细的车型要用 PaddleClas PULC 自行训练。

### 9.2 用 FastReID 提取 embedding

对每张车辆裁剪图：

1. RGB 转换（cv2 读图是 BGR）；
2. resize 到配置中的 `SIZE_TEST`（当前为 256×256，直接拉伸、不保持宽高比，与 fastreid 官方一致）；
3. 转成 NCHW float32 Tensor，数值保持 0~255，**不要除以 255**；
4. 调用完整的模型 forward（`model({'images': tensor})`），**不要绕过模型直接调 backbone/head**：像素归一化 `(x - mean) / std`（ImageNet 均值方差、0-255 尺度）只发生在 `Baseline.preprocess_image`（`fastreid/modeling/meta_arch/baseline.py`），跳过它输出仍是合法的 2048 维向量，"维度=2048"的检查发现不了，但相似度区分度会悄悄下降；
5. 取 heads 输出的 features（2048 维），L2 归一化；
6. 写入 `reid_embedding` BLOB。

同时写入：

```text
reid_source        = FastReID
reid_model_version = fastreid:veri_sbs_R50-ibn.pth
```

候选检索只允许使用 `reid_source=FastReID` 的真实特征，旧的通用分类模型特征不能冒充 ReID 特征。

embedding 正确性验收：同一张 crop 分别用本 pipeline 和 fastreid 官方 demo 各算一次 embedding，两者余弦相似度应接近 1；明显偏低时优先排查预处理（颜色通道错误、提前除以 255、绕过 forward 直接调 backbone，这三个错误都不会报错）。

## 10. 批处理和断点续跑

使用项目根目录的脚本：

```powershell
cd D:\WorkSpace\car_trace

# CPU 小批量验证
.\run_reid_pipeline.ps1 -Limit 10

# 本地 GPU （已安装一张英伟达RTX A1000，注意不要使用核显）正式处理
.\run_reid_pipeline.ps1 -PaddleDevice GPU -FastReIDDevice cuda:0

# 模型或裁剪策略变更后强制重算
.\run_reid_pipeline.ps1 -PaddleDevice GPU -FastReIDDevice cuda:0 -Force
```

脚本默认只处理 `vehicle_crop_path` 或 `reid_embedding` 为空的记录，可以中断后继续。正式运行前先备份 `app/granary.db`。

## 11. 建立向量检索库

- SQLite BLOB + NumPy 余弦相似度暴力检索，便于验证是否是同一车辆。当前量级（约 5000 张图、800 个订单）暴力检索完全够用，**明确不引入 FAISS**；数据增长到几十万级向量、单次全库检索明显变慢时再重新评估；
- 检索方式：以每张 crop 为 query，在全库（或先按车牌、车型等结构化字段过滤后的子集）中取 Top-K，再用属性、时间做重排。属性过滤走 SQL，不走向量。

## 12. 设计产品界面

确认车辆身份关系，并把该车辆一天的图片轨迹按时间显示出来，至少支持：

- 按照车牌查询；
- 按照车型查询；
- 按照出现时段查询；
- 按照订单号查询（订单号即是文件夹名称）；
- 没有检测到车辆的图片应注明；
- 可以人工进行修改某图片数据的操作（如修改时间，车牌，属于某一订单等）；

车型、颜色查询直接走第 9.1 步落库的结构化字段（SQL），不经过向量检索；跨日期文件夹的订单（第 3.1 步）要聚合成一条完整轨迹展示。

## 13. 判定规则

### 13.1 什么情况下判定“同车”

至少满足以下两项，且没有明显冲突：

- 车牌完全一致或只有可解释的 OCR 混淆（应该建立规则库，尤其是对于容易混淆的字符包括但不限于"0"和"O"，"2"和"Z"等等，对于车牌开头的文字，非常容易识别错误，直接从判定规则里排除，比对时只看后续的字母和数字）；
- 车身颜色、车型是否一致（来自第 9.1 步属性字段：车型冲突是"不同车"的强证据，颜色低置信度时只作软证据）；
- 摄像头和时间符合通行顺序（摄像头用第 4 步的 `camera_key`；订单内顺序用阶段前缀入磅->质检->出磅，抓拍用 OCR 时间戳插入）；
- ReID 候选图中的车辆主体确实可见。

### 13.2 什么情况下判定“不同车”

- 车牌明确不同；
- 车型明显不同（属性模型高置信度输出车型冲突时，可直接判"不同车"）；或车身颜色高置信度明显不同；
- 同一时间不可能出现在两个相距较远的位置；
- 候选相似仅来自背景、颜色或拍摄角度，车辆细节不一致。

### 13.3 “不确定”和“不可用”要分开

- `uncertain`：图片仍有价值，但当前无法确认身份；
- `invalid`：车辆严重遮挡、裁剪错误、图片损坏或根本不是车辆。

## 14. 导出 ReID 微调数据集

人工确认后导出 CSV，建议转换为如下目录：

```text
dataset/park_vehicle/
├─ train/
│  ├─ pid_0001/
│  └─ pid_0002/
├─ query/
├─ gallery/
└─ splits.json
```

## 15. 微调 FastReID 和评估

暂时不进行微调

## 16. 上线、监控和持续学习

### 16.1 线上服务分层

建议拆成三个进程或服务：

1. 图片接入/去重；
2. PaddleDetection（检测 + 车辆属性）+ FastReID GPU 推理队列；
3. 向量检索（SQLite + NumPy 暴力检索）、轨迹融合和 Web 服务。

这样检测高峰不会阻塞标注页面，模型升级也不会影响历史查询。

### 16.2 监控指标

- 每分钟接收量、处理量、队列长度；
- 检测无车比例和异常突增；
- FastReID 推理耗时、GPU 显存和失败数；
- 裁剪图缺失数、embedding 缺失数；
- 新摄像头或夜间场景的误合并率。

### 16.3 版本升级流程

1. 新模型只写入新的 `reid_model_version`；
2. 在离线验证集上比较旧模型；
3. 抽样查看候选图和误合并案例；
4. 通过后按新版本重算 embedding 并重建检索数据；
5. 保留旧 embedding，便于回滚和审计。

### 16.4 当前项目的最小验收清单

- [ ] `/annotate` 可以访问；

- [ ] `/api/crop/<id>` 返回真实车辆裁剪 JPEG；

- [ ] 数据库同时存在 `crop_source=PaddleDetection` 和 `reid_source=FastReID`；

- [ ] `reid_embedding` 长度与模型输出维度一致；

- [ ] 同一张 crop 的 pipeline embedding 与 FastReID 官方 demo 输出余弦相似度接近 1；

- [ ] `vehicle_type`/`vehicle_color` 及置信度已入库，`attr_model_version` 有值；

- [ ] `camera_key`、`detection_count`、`ocr_status` 字段有值；

- [ ] 候选排序不读取旧 MobileNetV2 embedding；

- [ ] 中断后重新运行不会重复处理已有记录；

- [ ] 跨日期文件夹的订单（如 68002608090040）聚合后图片完整；

- [ ] 10 项应用自动测试通过（测试代码需随 `app/` 一起迁移或新建，当前工作区还没有）；

  

