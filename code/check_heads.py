"""Report, for every registered model, whether it renders through the exact PEQ head.

peq_head=True means the model owns an ExactPEQResponse (or a subclass such as
baselines.DifferentiablePEQResponse); any module whose class name still mentions
"Gauss" is listed as a failure.
"""
import sys

import train_full as TF
from exact_peq_response import ExactPEQResponse

EXPECTED_PEQ = {"A0_Proposed", "A2_withPrefLoss", "E3_Nercessian", "E4_Pepe", "E5_Sequential"}

bad = []
for k, entry in TF.build_registry().items():
    m = entry["model"]
    heads = [(n, sub) for n, sub in m.named_modules() if isinstance(sub, ExactPEQResponse)]
    gauss = [n for n, sub in m.named_modules() if "Gauss" in type(sub).__name__]
    peq_head = bool(heads)
    st = ",".join(sorted({str(getattr(sub, "section_types", "?")) for _, sub in heads}))
    flag = ""
    if gauss or peq_head != (k in EXPECTED_PEQ):
        flag = "   <-- UNEXPECTED"
        bad.append(k)
    print(f"{k:24s} peq_head={str(peq_head):5s} {st:8s}{' gauss=' + ','.join(gauss) if gauss else ''}{flag}")

missing = EXPECTED_PEQ - set(TF.build_registry())
if missing:
    print("missing from registry:", sorted(missing))
    bad += sorted(missing)
print("OK" if not bad else f"FAIL: {bad}")
sys.exit(1 if bad else 0)
