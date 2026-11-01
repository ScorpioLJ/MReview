
import torch
from torch import nn

from mreview.weights import check_weights_file


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _write_model(tmp_path):
    p = tmp_path / "model.py"
    p.write_text(
        "import torch\nfrom torch import nn\n\n"
        "class TinyNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    return p


def test_nan_weights_p0(tmp_path):
    w = tmp_path / "bad.pt"
    torch.save({"fc.weight": torch.tensor([[1.0, float("nan"), 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
                "fc.bias": torch.zeros(2)}, w)
    _write_model(tmp_path)
    findings, details, _ = check_weights_file(w, tmp_path, dynamic=True)
    titles = [f.title for f in findings]
    assert any("NaN" in t for t in titles)
    assert details["model_loaded"] is True


def test_key_mismatch_p1(tmp_path):
    w = tmp_path / "bad.pt"
    torch.save({"fc.weight": torch.zeros(2, 4), "fc.bias": torch.zeros(2), "extra": torch.zeros(3)}, w)
    _write_model(tmp_path)
    findings, _, _ = check_weights_file(w, tmp_path)
    assert any("不匹配" in f.title for f in findings)


def test_good_weights_no_static_findings(tmp_path):
    w = tmp_path / "good.pt"
    torch.save({"fc.weight": torch.randn(2, 4), "fc.bias": torch.randn(2)}, w)
    _write_model(tmp_path)
    findings, _, _ = check_weights_file(w, tmp_path)
    static = [f for f in findings if f.source == "weights-static"]
    assert not static
