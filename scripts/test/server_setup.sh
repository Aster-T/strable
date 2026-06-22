#!/usr/bin/env bash
# 一键搭建 RPT 服务器环境（Linux + GPU）。在 strable 项目里运行：
#   bash scripts/test/server_setup.sh
# 可选：若 PyTorch 默认 CUDA 版不匹配你的卡，用 TORCH_INDEX 指定 wheel，例如：
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/test/server_setup.sh   # Blackwell(50系/B200)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/test/server_setup.sh   # A100/H100/40系
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv"
echo "== repo: $REPO =="

echo "== 1/5 确保 uv 可用 =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

echo "== 2/5 建 Python 3.11 venv: $VENV =="
uv venv --python 3.11 "$VENV"
export VIRTUAL_ENV="$VENV"

echo "== 3/5 装 torch + sap-rpt-oss + harness 依赖 =="
if [ -n "${TORCH_INDEX:-}" ]; then
  uv pip install torch --index-url "$TORCH_INDEX"
else
  uv pip install torch          # Linux 上默认即 CUDA wheel
fi
uv pip install "git+https://github.com/SAP-samples/sap-rpt-1-oss"
uv pip install pandas scikit-learn pyarrow numpy   # bench_adapters 依赖（多数已随 sap-rpt-oss 装上）

echo "== 4/5 检查数据是否就位 =="
for d in data/data_processed data/CARTE_datasets data/TTB_datasets; do
  if [ -d "$REPO/$d" ]; then echo "  OK   $d"; else echo "  缺失 $d  (需从本机 rsync 过来)"; fi
done
if [ -f "$REPO/results/tabpfn_test/summary.csv" ]; then
  echo "  OK   results/tabpfn_test/summary.csv (横评基线)"
else
  echo "  注意 results/tabpfn_test/summary.csv 不在 —— 横评基线，建议带上"
fi

echo "== 5/5 验证 torch / GPU / sap_rpt_oss =="
"$VENV/bin/python" - <<'PY'
import torch
print("  torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
import sap_rpt_oss
print("  sap_rpt_oss OK ->", [n for n in dir(sap_rpt_oss) if "RPT" in n])
PY

cat <<EOF

环境就绪。下一步：
  1) 用你的 HF 账号同意门控条款：打开 https://huggingface.co/SAP/sap-rpt-1-oss 点 Agree
  2) 设置 token：       export HF_TOKEN=hf_xxxxx
  3) 跑全量 RPT 8192：   bash $HERE/run_rpt_8192.sh
EOF
