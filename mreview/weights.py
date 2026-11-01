"""Checkpoint review: static (read-only) and dynamic (load and run) checks.

Static checks never execute the model: loadability, key/dtype/shape alignment
against the model definition, NaN/Inf and distribution statistics. Dynamic
checks (--dynamic) instantiate the model, load weights and run one forward
pass; any failure degrades to "unverified (reason)" instead of a hard error.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

from .models import Finding

WEIGHT_EXTS = {".pt", ".pth", ".ckpt", ".safetensors"}

NaN_CAUSE = "训练侧嫌疑：lr 过大、loss 含除零/log(0)、AMP 下 float16 溢出，或从损坏的 checkpoint 恢复。"
NaN_FIX = "降低学习率或加梯度裁剪；检查 loss 数值稳定性；从最近一个干净 checkpoint 恢复并复现。"
MISMATCH_CAUSE = "模型定义与权重来源不一致：权重来自旧结构、不同配置，或保存时混入了额外 state。"
MISMATCH_FIX = "用与权重来源相同的配置实例化模型；必要时删除多余 key 或用 strict=False 加载后重新保存。"


def discover_weight_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in WEIGHT_EXTS else []
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTS
    )


def _load_checkpoint(path: Path) -> tuple[dict | None, str | None]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(path)), None
        except Exception as exc:  # noqa: BLE001
            return None, f"safetensors 加载失败: {exc}"
    try:
        return torch.load(path, map_location="cpu", weights_only=True), None
    except Exception as exc1:  # noqa: BLE001
        try:
            return torch.load(path, map_location="cpu", weights_only=False), None
        except Exception as exc2:  # noqa: BLE001
            return None, f"torch.load 失败: {exc1}; 再试 weights_only=False 也失败: {exc2}"


def _iter_tensors(state: dict, prefix: str = ""):
    for key, value in state.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _iter_tensors(value, f"{full}.")
        elif isinstance(value, torch.Tensor):
            yield full, value


def _model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return dict(model.state_dict())


def _guess_input(model: nn.Module) -> torch.Tensor:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return torch.zeros(1, m.in_channels, 64, 64)
        if isinstance(m, nn.Linear):
            return torch.zeros(1, m.in_features)
        if isinstance(m, nn.Embedding):
            return torch.zeros(1, 8, dtype=torch.long)
    return torch.zeros(1, 3, 64, 64)


def _find_model_class(path: Path) -> type[nn.Module] | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:  # noqa: BLE001
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            else:
                base_name = None
            if base_name and ("Module" in base_name or "nn.Module" in base_name):
                return _import_class(path, node.name)
    return None


def _import_class(path: Path, class_name: str) -> type | None:
    try:
        mod_name = f"_mreview_dyn_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name, None)
    except Exception:  # noqa: BLE001
        return None


def _load_model(target: Path, model_file: Path) -> tuple[nn.Module | None, str | None]:
    cls = _find_model_class(model_file)
    if cls is None:
        return None, "未找到可实例化的 nn.Module 子类"
    try:
        model = cls()
    except Exception as exc:  # noqa: BLE001
        return None, f"实例化失败: {exc}"
    return model, None


def _stats_findings(path: Path, state: dict) -> list[Finding]:
    findings: list[Finding] = []
    nan_keys: list[str] = []
    dead_keys: list[str] = []
    details: dict[str, dict] = {}
    for key, tensor in _iter_tensors(state):
        if tensor.is_floating_point():
            numel = tensor.numel()
            nan_count = int(torch.isnan(tensor).sum().item()) if numel else 0
            inf_count = int(torch.isinf(tensor).sum().item()) if numel else 0
            if nan_count or inf_count:
                nan_keys.append(key)
            if numel:
                tmin, tmax = float(tensor.min()), float(tensor.max())
                tmean = float(tensor.mean())
                if tmax - tmin == 0.0:
                    dead_keys.append(key)
                details[key] = {
                    "nan": nan_count,
                    "inf": inf_count,
                    "min": tmin,
                    "max": tmax,
                    "mean": tmean,
                    "std": float(tensor.std().item()) if numel > 1 else 0.0,
                }
            else:
                details[key] = {"nan": 0, "inf": 0, "empty": True}
        else:
            details[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    if nan_keys:
        findings.append(
            Finding(
                source="weights-static",
                category="weights",
                severity="P0",
                file=str(path),
                line=None,
                title=f"权重包含 NaN/Inf：{len(nan_keys)} 个 tensor",
                cause=NaN_CAUSE,
                fix=NaN_FIX,
                evidence=f"涉及 key: {', '.join(nan_keys[:10])}",
            )
        )
    if dead_keys:
        findings.append(
            Finding(
                source="weights-static",
                category="weights",
                severity="P2",
                file=str(path),
                line=None,
                title=f"权重存在全零/常量 tensor：{len(dead_keys)} 个",
                cause="训练未有效更新该参数，或初始化错误导致梯度死亡。",
                fix="检查对应层的梯度流与初始化；确认 loss 对该参数有梯度。",
                evidence=f"涉及 key: {', '.join(dead_keys[:10])}",
            )
        )
    return findings, details


def _alignment_findings(path: Path, state: dict, model: nn.Module) -> list[Finding]:
    weight_keys = {key for key, _ in _iter_tensors(state)}
    model_keys = set(_model_state_dict(model))
    missing = sorted(model_keys - weight_keys)
    unexpected = sorted(weight_keys - model_keys)
    findings: list[Finding] = []
    if missing or unexpected:
        findings.append(
            Finding(
                source="weights-static",
                category="weights",
                severity="P1",
                file=str(path),
                line=None,
                title="state_dict 与模型定义不匹配",
                cause=MISMATCH_CAUSE,
                fix=MISMATCH_FIX,
                evidence=f"缺失 {len(missing)} 个 key；多余 {len(unexpected)} 个 key。"
                f" 缺失示例: {missing[:5]}；多余示例: {unexpected[:5]}",
            )
        )
    return findings


def _dynamic_findings(path: Path, model: nn.Module, state: dict) -> list[Finding]:
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:  # noqa: BLE001
        return [Finding(
            source="weights-dynamic",
            category="weights",
            severity="P2",
            file=str(path),
            line=None,
            title="权重加载到模型失败",
            cause="state_dict 结构与模型不兼容。",
            fix=MISMATCH_FIX,
            evidence=str(exc),
        )]
    try:
        model.eval()
        x = _guess_input(model)
        with torch.no_grad():
            out = model(x)
        tensor = out if isinstance(out, torch.Tensor) else torch.tensor(float("nan"))
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return [Finding(
                source="weights-dynamic",
                category="weights",
                severity="P0",
                file=str(path),
                line=None,
                title="加载后前向输出包含 NaN/Inf",
                cause=NaN_CAUSE,
                fix=NaN_FIX,
                evidence=f"输入 {tuple(x.shape)}；输出 {tuple(tensor.shape)}",
            )]
        peak = float(tensor.abs().max())
        if peak > 1e6:
            return [Finding(
                source="weights-dynamic",
                category="weights",
                severity="P2",
                file=str(path),
                line=None,
                title=f"前向输出幅值异常偏大（abs max={peak:.3e}）",
                cause="权重数值过大或存在数值不稳定路径（未归一化、初始化过强）。",
                fix="检查初始化与归一化层；验证输入范围假设。",
                evidence=f"输入 {tuple(x.shape)}；输出 {tuple(tensor.shape)}",
            )]
        return []
    except Exception as exc:  # noqa: BLE001
        return [Finding(
            source="weights-dynamic",
            category="weights",
            severity="P2",
            file=str(path),
            line=None,
            title="动态前向检查失败",
            cause="模型无法在该输入/环境下完成一次前向。",
            fix="提供可实例化且输入形状可推断的模型，或检查模型依赖的环境。",
            evidence=str(exc),
        )]


def check_weights_file(
    path: Path,
    target: Path,
    dynamic: bool = False,
    details_dir: Path | None = None,
) -> tuple[list[Finding], dict, str | None]:
    """Return (findings, detail dict, dynamic_status)."""
    findings: list[Finding] = []
    state, load_err = _load_checkpoint(path)
    if load_err:
        findings.append(
            Finding(
                source="weights-static",
                category="weights",
                severity="P1",
                file=str(path),
                line=None,
                title="权重文件无法加载",
                cause="文件损坏、格式不符，或依赖缺失。",
                fix="确认文件完整性与来源；用 torch.load / safetensors 工具修复或重新导出。",
                evidence=load_err,
            )
        )
        return findings, {"load_error": load_err}, "static-failed"

    model_files = sorted(target.rglob("*.py")) if target.is_dir() else []
    model: nn.Module | None = None
    model_err: str | None = None
    for mf in model_files:
        if any(part in {".git", "__pycache__", ".mreview-results"} for part in mf.relative_to(target).parts):
            continue
        candidate, err = _load_model(target, mf)
        if candidate is not None:
            model = candidate
            model_err = None
            break
        model_err = err

    stats, details = _stats_findings(path, state)
    findings.extend(stats)
    details["load_error"] = load_err
    details["model_loaded"] = model is not None
    details["model_error"] = model_err
    details["total_tensors"] = len(list(_iter_tensors(state)))

    if model is not None:
        findings.extend(_alignment_findings(path, state, model))
    else:
        details["key_alignment"] = "unverified"

    dynamic_status = "off"
    if dynamic:
        if model is None:
            dynamic_status = "unverified"
            details["dynamic"] = {"status": "unverified", "reason": model_err or "模型不可实例化"}
        else:
            dyn = _dynamic_findings(path, model, state)
            findings.extend(dyn)
            dynamic_status = "ok" if not dyn else "issues"
            details["dynamic"] = {"status": dynamic_status, "findings": [f.title for f in dyn]}

    if details_dir is not None:
        import json

        (details_dir / f"{path.name}.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    return findings, details, dynamic_status
