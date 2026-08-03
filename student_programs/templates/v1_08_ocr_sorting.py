"""V1-08 学生模板：只选择主机已经批准的 OCR 条目顺序。

实验定义：config/experiments/V1-08.json
操作指南：docs/experiments/V1-08.md
"""


EXPECTED_ENTRY_IDS = ("entry_a", "entry_b", "entry_c", "entry_d")


def main(ctx):
    # 识别、训练和完整计划由主机一次性完成；学生不能提交 ROI、坐标或阈值。
    recognition = ctx.vision2d.ocr_sorting()
    if recognition.get("status") != "PASS":
        raise RuntimeError("OCR 识别或完整分拣计划未通过主机校验")

    entries = recognition.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("主机未返回可验证的白名单条目")
    approved = tuple(item.get("entry_id") for item in entries)
    if approved != EXPECTED_ENTRY_IDS:
        raise RuntimeError("OCR 计划必须包含按发布顺序排列的四个条目")
    if any(
        not isinstance(item, dict)
        or set(item) != {"entry_id", "part_id", "identifier", "route_id", "roi_px", "confidence", "status"}
        or item["status"] != "APPROVED"
        for item in entries
    ):
        raise RuntimeError("条目不是主机发布的只读白名单记录")

    # 只有严格的 entry_id 进入受控主机动作；动作内部的坐标、速度、吸盘和回零
    # 由 host runner 管理，学生模板不直接访问 robot/tool。
    for entry_id in EXPECTED_ENTRY_IDS:
        receipt = ctx.vision2d.sort_ocr_entry(entry_id)
        if receipt.get("status") != "PASS":
            raise RuntimeError(f"分拣条目 {entry_id} 未完成")

    ctx.log("V1-08 四个 OCR 条目已按主机安全策略完成")
    ctx.checkpoint("V1-08 OCR 分拣完成；详见同次运行证据")
