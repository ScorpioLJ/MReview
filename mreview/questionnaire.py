from __future__ import annotations

from .models import Question

QUESTIONS: list[Question] = [
    Question(
        "q1",
        "输入/输出契约",
        "该单元的输入与输出（dtype、shape、device、layout）预期是否明确？是否存在隐式类型转换或未言明的 shape 假设？",
    ),
    Question(
        "q2",
        "数值稳定性",
        "该单元是否存在产生 NaN/Inf 的路径（除零、log(0)、exp 溢出、未裁剪项）？混合精度下有无溢出风险？",
    ),
    Question(
        "q3",
        "梯度与反向传播",
        "该单元有无 in-place 修改、detach()/no_grad() 误用，或切断梯度流的隐患？",
    ),
    Question(
        "q4",
        "模式与状态",
        "train/eval 切换、dropout/BatchNorm 行为、随机种子/确定性开关是否正确处理？",
    ),
    Question(
        "q5",
        "设备与数据放置",
        "该单元中 .to(device)/.cuda() 有无遗漏或滥用？CPU/GPU 混合数据是否存在隐患？",
    ),
    Question(
        "q6",
        "资源与生命周期",
        "句柄、临时文件、显存、进程是否正确释放？长循环内有无泄漏？",
    ),
    Question(
        "q7",
        "错误处理",
        "失败路径（OOM、文件缺失、shape 不符）有明确处理，还是会静默吞错？",
    ),
    Question(
        "q8",
        "边界与索引",
        "循环边界、切片/索引、形状推断在边界输入下是否安全？",
    ),
    Question(
        "q9",
        "一致性与约定",
        "命名、返回值、副作用是否一致？有无重复实现导致漂移？",
    ),
    Question(
        "q10",
        "性能",
        "是否存在冗余计算、CPU-GPU 同步、逐元素循环或重复分配？",
    ),
]


def questions() -> list[Question]:
    return list(QUESTIONS)
