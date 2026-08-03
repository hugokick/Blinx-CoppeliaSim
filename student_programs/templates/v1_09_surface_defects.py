"""V1-09 student template: controlled surface-defect analysis and sorting."""


def main(ctx):
    analysis = ctx.vision2d.surface_defects()
    ctx.log(
        f"缺陷计划已冻结 plan_id={analysis.plan_id} "
        f"entries={len(analysis.entries)}"
    )
    for entry in analysis.entries:
        receipt = ctx.vision2d.defect_sort_entry(entry.entry_id)
        ctx.log(
            f"条目完成 entry_id={receipt.entry_id} "
            f"decision={receipt.decision} status={receipt.status}"
        )
