#!/bin/bash
# Run the full TabPFN benchmark (all datasets x test_size 0.1..0.9), then the
# bad-case analysis. The Python runner loops over datasets internally, is
# resumable, and rotates the configured API tokens.
#
#   bash scripts/test/test.sh                 # everything, then analysis
#   bash scripts/test/test.sh --dry-run       # one dataset / one fold smoke test
#   bash scripts/test/test.sh --tasks classification
#
# Any extra args are forwarded to tabpfn_test.py.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PYTHON:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

"$PY" scripts/test/tabpfn_test.py "$@"
echo "==== analysing bad cases ===="
"$PY" scripts/test/extract_bad_cases.py
