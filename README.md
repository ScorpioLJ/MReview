# MReview

模型设计 / 训练代码审查 CLI。借鉴 c-review 的审查逻辑——**按代码位置切分单元 → 固定问卷 → 账本证据 → 覆盖校验**——并集成 ruff / torchfix 等确定性工具与权重文件检查。以纯 CLI 形态交付，可被任意 agent harness 通过 shell 调用。

## 审查逻辑（源自 c-review，不搬其工程架构）

1. **位置切分**：用 AST 把目标代码切成单元（函数/方法、类体、模块级流程），每行恰好属于一个单元；超过 150 行在语法接缝处拆块。
2. **固定问卷**：每个单元针对同一套 10 题作答，每题要求落在单元行号范围内的证据。
3. **账本校验**：`verify` 重解析源码核对——①每单元 10 题全覆盖；②evidence 行号在单元范围内；③“ok”也必须带行号。缺口只把该单元标为“未验证”并列出 `(unit, question)`，不整份作废。
4. **外部工具合并**：ruff/torchfix 输出自动并入 findings（按 file:line 去重），缺失工具跳过并在报告标注。

## 审查模式

- **静态（默认）**：不真跑——跑外部工具 + 只读权重文件做数值检查。
- **动态（`--dynamic`）**：真正加载模型（import + 实例化 + 载入权重 + 一次 forward），检查输出有限性/幅值；失败降级为“未验证（原因）”。

## 安装

```bash
# 在目标 Python 环境（如 diffusion）
pip install -e .
```

无隔离 venv、无 uv；`doctor` 只确认审查工具是否存在。

## 用法

```bash
mreview doctor                    # 环境自检：确认 torch/ruff/torchfix 等是否存在
mreview split <target>            # 切分单元，生成 worksheet.json（单元 × 10 题账本模板）
mreview review <target> [--dynamic]  # 全自动：split + 外部工具 + 权重检查
mreview verify <ledger.json> <target> # 校验 agent 回填的账本，输出合并报告
```

退出码：`0` 通过 / `1` 环境或参数错误 / `2` 存在未核实单元。

`<target>` 可为目录（自动发现 `*.py` 与 `*.pt/.pth/.ckpt/.safetensors`）或单个文件。

## 落盘与追溯

每次运行新建 `.mreview-results/<UTC 时间戳>/`，写入：

| 文件 | 内容 |
|---|---|
| `units.json` | 单元清单（id、类型、行号范围） |
| `worksheet.json` | 问卷 + 账本模板 |
| `findings.json` | 结构化 findings（severity/category/file/line/cause/fix/evidence） |
| `REPORT.md` | 人读报告 |
| `ledger-gate.json` | verify 覆盖校验结果与缺口 |
| `tools/*.txt` | 各工具原始输出 |
| `weights/*.json` | 权重检查明细 |
| `meta.json` | 命令、参数、环境 |

## 10 题问卷（所有代码单元通用）

输入输出契约 / 数值稳定性 / 梯度与反向传播 / 模式与状态 / 设备与数据放置 / 资源与生命周期 / 错误处理 / 边界与索引 / 一致性与约定 / 性能。

## 权重检查

- **静态**：可加载性、state_dict key 与模型定义对齐、dtype/shape、NaN/Inf 计数与所在层、min/max/mean/std、全零/死区。
- **动态**（`--dynamic`）：实例化模型载入权重跑一次 forward，检查输出有限性与幅值。

每个 finding 都带**原因**与**解决方案**（如 NaN 追溯到 lr 过大 / AMP 溢出并给出修复步骤）。

## 账本格式（agent 回填）

```json
{
  "units": [
    {
      "unit_id": "bad_model.py:BadNet.forward",
      "answers": [
        { "question_id": "q1", "verdict": "ok", "evidence": "L30-L34" },
        { "question_id": "q2", "verdict": "issue",
          "evidence": "L33", "title": "...", "cause": "...", "fix": "...",
          "category": "correctness", "severity": "P2" }
      ]
    }
  ]
}
```

`verdict` 取值 `ok / issue / skip`；`ok` 与 `skip` 也必须带行号证据。

## 开发

```bash
pip install -e .[dev]
pytest tests/          # 单元切分 / 账本校验 / 权重检查测试
ruff check mreview tests
```

`fixtures/` 是已知缺陷样例与假账本，用于验证召回率与 verify 抓假能力。

## 目录结构

```
mreview/          # 包源码（cli / splitter / tools / weights / ledger / report / doctor / questionnaire）
tests/            # pytest
fixtures/         # 已知缺陷样例（bad_model.py、bad_train.py、bad_weights.pt、fake_ledger.json）
```

## 已知限制

- torchfix / hotpath 等未安装时自动跳过并在报告中标注，不影响运行。
- 动态检查的输入形状按模型首层启发式猜测（Conv2d → 64×64，Linear → in_features），失败会如实标注“未验证”。
- 无外网时无法验证 torchfix/hotpath-ai/torch-preflight 的存在性；`doctor` 会如实报告。
