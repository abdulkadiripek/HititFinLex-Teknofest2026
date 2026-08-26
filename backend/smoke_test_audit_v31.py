from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPORT = """\
Pipeline: historical_v3_1
[1/2] key=valid product=KART_KAMPANYASI
  ACCEPTED HARCAMA_ESIGI               0.9700 | 250 TL | spend_threshold
    EVIDENCE | Kartinizla 250 TL ve uzeri harcama yapin.
  ACCEPTED ODUL_TUTARI                  0.9600 | 250 TL | reward_amount
    EVIDENCE | Davet ettiginiz her kisi icin 250 TL nakit odul kazanin.
[2/2] key=invalid product=KART_KAMPANYASI
  ACCEPTED HARCAMA_ESIGI                0.9700 | 500 TL | spend_threshold
    EVIDENCE | Kampanyadan 500 TL bonus kazanin.
  ACCEPTED ODUL_TUTARI                  0.9600 | 500 TL | reward_amount
    EVIDENCE | Kampanyadan 500 TL bonus kazanin.
Summary: {"selected": 2, "facts_accepted": 4}
"""


def main() -> None:
    script = Path(__file__).with_name("audit_archive_ner_report_v31.py")
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "report.txt"
        report.write_text(REPORT, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script), str(report)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    result = json.loads(completed.stdout)
    assert result["accepted_valid_dual_role_amount_count"] == 1, result
    assert result["accepted_cross_role_amount_count"] == 1, result
    assert result["accepted_amount_role_mismatch_count"] == 1, result
    assert result["error_count"] == 0, result
    print("NER audit V3.1: OK (4 gates)")


if __name__ == "__main__":
    main()
