#!/usr/bin/env bash
# 跑全量 RPT（论文高配 ctx=8192 / bagging=8），覆盖 STRABLE+CARTE+TTB 共 1476 个 run。
# 断点续跑：已写入 results/rpt_test/summary.csv 的 (benchmark,dataset,test_size) 会自动跳过。
# 用法：
#   export HF_TOKEN=hf_xxxxx        # 已在 HF 网站同意门控条款后
#   bash scripts/test/run_rpt_8192.sh
# 可覆盖配置：CTX=4096 BAG=4 bash scripts/test/run_rpt_8192.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv"

export STRABLE_ROOT="$REPO"
export VIRTUAL_ENV="$VENV"
export PYTHONUTF8=1

if [ -z "${HF_TOKEN:-}" ] && [ "${HF_HUB_OFFLINE:-0}" != "1" ]; then
  echo "请先设置 HF token（并已在 https://huggingface.co/SAP/sap-rpt-1-oss 同意门控条款）："
  echo "  export HF_TOKEN=hf_xxxxx"
  exit 1
fi

CTX="${CTX:-8192}"
BAG="${BAG:-8}"
echo "== RPT: --benchmark all --max-context $CTX --bagging $BAG =="
"$VENV/bin/python" "$HERE/rpt_test.py" --benchmark all --max-context "$CTX" --bagging "$BAG"

echo
echo "跑完做横评（RPT vs TabPFN）："
echo "  \"$VENV/bin/python\" \"$HERE/compare_rpt_tabpfn.py\" \"$REPO/results/tabpfn_test/summary.csv\""
