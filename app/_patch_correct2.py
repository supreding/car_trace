# -*- coding: utf-8 -*-
"""一次性补丁：correct_pass 选择不依赖 vehicle_id（跑完可删）。"""
from pathlib import Path

p = Path(__file__).resolve().parent / "reid_pipeline.py"
t = p.read_text(encoding="utf-8")

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
)
assert old_sel in t, "selection not found"
t = t.replace(old_sel, new_sel, 1)

old_loop = (
    '        capture_id = int(row["id"])\n'
    '        consensus = consensuses.get(row["vehicle_id"])\n'
    '        if not consensus:\n'
    '            skipped += 1\n'
    '            continue'
)
new_loop = (
    '        capture_id = int(row["id"])\n'
    '        consensus = consensuses.get(order_group.get(row["order_id"]))\n'
    '        if not consensus:\n'
    '            skipped += 1\n'
    '            continue'
)
assert old_loop in t, "loop lookup not found"
t = t.replace(old_loop, new_loop, 1)

p.write_text(t, encoding="utf-8")
print("patched ok")
