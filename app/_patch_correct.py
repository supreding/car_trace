# -*- coding: utf-8 -*-
"""一次性补丁：修正 correct_pass 的选择与替换契约（跑完可删）。"""
from pathlib import Path

p = Path(__file__).resolve().parent / "reid_pipeline.py"
t = p.read_text(encoding="utf-8")

# 1) 选择：不依赖 vehicle_id（拆出的行也要能被复核），通过订单兄弟行找组
old_sel = (
    '    # 可疑帧：所属车辆组有共识，且（颜色与共识冲突 或 车型置信度 <0.7 且\n'
    '    # 车型与共识不符）\n'
    '    suspicious = []\n'
    '    for row in db.execute(\n'
    '        """SELECT id, filepath, order_id, vehicle_crop_path, vehicle_color,\n'
    '                  vehicle_type, vehicle_type_conf, vehicle_id, status_reason\n'
    '           FROM captures\n'
    '           WHERE vehicle_crop_path IS NOT NULL AND vehicle_id IS NOT NULL\n'
    "             AND coalesce(capture_status,'vehicle')!='no_vehicle'\n"
    '           ORDER BY id"""\n'
    '    ).fetchall():\n'
    '        consensus = consensuses.get(row["vehicle_id"])\n'
    '        if not consensus:\n'
    '            continue\n'
    '        color_conflict = (row["vehicle_color"] and consensus["vehicle_color"]\n'
    '                          and row["vehicle_color"] != consensus["vehicle_color"])\n'
    '        type_suspect = (row["vehicle_type_conf"] is not None\n'
    '                        and row["vehicle_type_conf"] < 0.7)\n'
    '        if color_conflict or type_suspect:\n'
    '            suspicious.append(row)\n'
    '    rows = suspicious[::] if not limit else suspicious[:limit]'
)
new_sel = (
    '    # 可疑帧：所属车辆组有共识，且（颜色与共识冲突 或 车型置信度 <0.7 且\n'
    '    # 车型与共识不符）。注意不依赖 vehicle_id——被拆出的行（vehicle_id\n'
    '    # 为空）同样要能被复核修正；组共识通过订单兄弟行回溯。\n'
    '    order_group = {}\n'
    '    for oid, vid in db.execute(\n'
    '        "SELECT DISTINCT order_id, vehicle_id FROM captures\n'
    "           WHERE vehicle_id IS NOT NULL AND coalesce(order_id,'')!=''\"\n"
    '    ):\n'
    '        order_group.setdefault(oid, vid)\n'
    '    suspicious = []\n'
    '    for row in db.execute(\n'
    '        """SELECT id, filepath, order_id, vehicle_crop_path, vehicle_color,\n'
    '                  vehicle_type, vehicle_type_conf, vehicle_id, status_reason\n'
    '           FROM captures\n'
    '           WHERE vehicle_crop_path IS NOT NULL\n'
    "             AND coalesce(capture_status,'vehicle')!='no_vehicle'\n"
    "             AND coalesce(order_id,'')!='' AND order_id!='未关联图片'\n"
    '           ORDER BY id"""\n'
    '    ).fetchall():\n'
    '        vid = order_group.get(row["order_id"])\n'
    '        consensus = consensuses.get(vid) if vid else None\n'
    '        if not consensus:\n'
    '            continue\n'
    '        color_conflict = (row["vehicle_color"] and consensus["vehicle_color"]\n'
    '                          and row["vehicle_color"] != consensus["vehicle_color"])\n'
    '        type_suspect = (row["vehicle_type_conf"] is not None\n'
    '                        and row["vehicle_type_conf"] < 0.7)\n'
    '        if color_conflict or type_suspect:\n'
    '            suspicious.append(row)\n'
    '    rows = suspicious[::] if not limit else suspicious[:limit]'
)
assert old_sel in t, "selection not found"
t = t.replace(old_sel, new_sel)

# 2) 循环里的共识取用同步改（组共识按 vehicle_id 键）
old_loop = (
    '    for row in rows:\n'
    '        capture_id = int(row["id"])\n'
    '        consensus = consensuses.get(row["vehicle_id"])\n'
    '        if not consensus:\n'
    '            skipped += 1\n'
    '            continue'
)
new_loop = (
    '    for row in rows:\n'
    '        capture_id = int(row["id"])\n'
    '        consensus = consensuses.get(order_group.get(row["order_id"]))\n'
    '        if not consensus:\n'
    '            skipped += 1\n'
    '            continue'
)
assert old_loop in t, "loop lookup not found"
t = t.replace(old_loop, new_loop)

# 3) replace 分支：恢复 vehicle_id（修正后帧回到订单车辆轨迹），清理拆分标记
old_rep = (
    '            db.execute(\n'
    '                """UPDATE captures SET vehicle_box=?, vehicle_det_score=?,\n'
    '                          vehicle_color=?, vehicle_color_conf=?,\n'
    '                          vehicle_type=?, vehicle_type_conf=?,\n'
    '                          lpr_plate=NULL, lpr_conf=NULL, plate_final=NULL,\n'
    "                          plate_confidence=NULL, recognition_source='',\n"
    '                          reid_embedding=NULL, reid_source=NULL\n'
    '                   WHERE id=?""",\n'
    '                (json.dumps(det["box"]), float(det["score"]),\n'
    '                 a["color"] or None, a["color_conf"],\n'
    '                 a["type"] or None, a["type_conf"], capture_id),\n'
    '            )\n'
    '            replaced += 1'
)
new_rep = (
    '            sib = db.execute(\n'
    '                """SELECT vehicle_id FROM captures\n'
    '                   WHERE order_id=? AND vehicle_id IS NOT NULL LIMIT 1""",\n'
    '                (row["order_id"],),\n'
    '            ).fetchone()\n'
    '            db.execute(\n'
    '                """UPDATE captures SET vehicle_box=?, vehicle_det_score=?,\n'
    '                          vehicle_color=?, vehicle_color_conf=?,\n'
    '                          vehicle_type=?, vehicle_type_conf=?,\n'
    '                          lpr_plate=NULL, lpr_conf=NULL, plate_final=NULL,\n'
    "                          plate_confidence=NULL, recognition_source='',\n"
    '                          reid_embedding=NULL, reid_source=NULL,\n'
    "                          vehicle_id=?,\n"
    "                          status_reason=CASE WHEN status_reason LIKE '%订单主车%'\n"
    "                                             THEN '' ELSE status_reason END\n"
    '                   WHERE id=?""",\n'
    '                (json.dumps(det["box"]), float(det["score"]),\n'
    '                 a["color"] or None, a["color_conf"],\n'
    '                 a["type"] or None, a["type_conf"],\n'
    '                 sib["vehicle_id"] if sib else None, capture_id),\n'
    '            )\n'
    '            replaced += 1'
)
assert old_rep in t, "replace branch not found"
t = t.replace(old_rep, new_rep)

p.write_text(t, encoding="utf-8")
print("patched ok")
