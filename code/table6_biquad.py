"""
table6_biquad.py — Table 4 (tab:ac_fitting) 조립
================================================
계산하지 않는다. 이미 산출된 파일을 읽어 논문 Table 4 의 CSV 를 만든다.
모델도 데이터셋도 열지 않으므로 GPU 없이, 체크포인트 없이 실행된다.

입력과 각 열의 출처:

  results_json/ac_biquad_table.json        ← ac_biquad_table.py
      configs[*_g12]  : Option C 세 행의 LSD / CI / vs A0 / repr. penalty
      a0_reference    : A0 기준행의 LSD 와 CI. per-sample LSD 를 seed 축으로
                        평균한 N=3000 배열의 bootstrap(n_boot=2000, seed=42) 이며,
                        seed 를 표본처럼 합친 pooled(N=9000) 부트스트랩이 아니다.
  tables/table7_perceptual.csv             ← table7_perceptual.py
      % < JND. 이 값을 내는 곳은 여기 한 곳뿐이다.
  tables/table2_revision_synth.csv         ← table2_revision.py
      dense AC2 (raw) 기준행.
  results_json/ac_fitting_A_g12.csv        ← ac_fitting_A.py (--gain_max 12.0)
  results_json/ac_fitting_A_g6_orig.csv    ← ac_fitting_A.py (--gain_max 6.0)
  results_json/ac_fitting_D_naive7pt.csv   ← ac_fitting_C.py (Option D)

Option D 행이 개정 이전 산출물인 이유:
  naive 7-pt 는 목표 응답을 일곱 지점에서 그대로 표본화할 뿐 학습된 모델을 쓰지
  않는다. Stage-B 의 per-section gain bound 가 관여하지 않으므로 ±6 → ±12 완화에
  영향을 받지 않고, 따라서 ±12 로 재산출하지 않았다. Option A 의 ±6 행도 같은
  이유가 아니라 대조군으로서 의도적으로 개정 이전 값을 싣는다(라벨 'orig').

  python table6_biquad.py
  python table6_biquad.py --results_dir /path/to/results --out_dir /path/to/results
"""
import argparse
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_ap = argparse.ArgumentParser(description="assemble Table 4 (tab:ac_fitting)")
_ap.add_argument("--results_dir", default=str(HERE / "results"),
                 help="where the generators wrote their output (read first)")
_ap.add_argument("--out_dir", default=str(HERE / "results"),
                 help="destination directory (created if missing)")
_args = _ap.parse_args()

RES = Path(_args.results_dir)
OUT = Path(_args.out_dir) / "table6_biquad.csv"


def _find(name, *subdirs):
    """생성기가 방금 쓴 results/ 를 먼저 보고, 없으면 리포에 커밋된 위치를 쓴다."""
    for d in (RES, *(ROOT / s for s in subdirs)):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError(f"{name} not found in {RES} or {[str(ROOT / s) for s in subdirs]}")


def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    return float(str(s).strip().strip('"'))


def ci_of(text):
    """'[0.9081,0.9288]' -> (0.9081, 0.9288)"""
    lo, hi = str(text).strip().strip('"').strip("[]").split(",")
    return float(lo), float(hi)


# ── 입력 ────────────────────────────────────────────────────────────────────
bq = json.loads(_find("ac_biquad_table.json", "results_json").read_text(encoding="utf-8"))
cfg, a0 = bq["configs"], bq["a0_reference"]
A0_MEAN = a0["lsd_mean"]

perc = {r["Model"]: r for r in read_csv(_find("table7_perceptual.csv", "tables"))}
main = {r["model"]: r for r in read_csv(_find("table2_revision_synth.csv", "tables"))}
optA12 = {r["Model/Condition"]: r for r in read_csv(_find("ac_fitting_A_g12.csv", "results_json"))}
optA6 = {r["Model/Condition"]: r for r in read_csv(_find("ac_fitting_A_g6_orig.csv", "results_json"))}
optD = {r["Model/Condition"]: r for r in read_csv(_find("ac_fitting_D_naive7pt.csv", "results_json"))}


# ── 서식 (커밋된 표와 동일) ─────────────────────────────────────────────────
def f3(v):
    return f"{v:.3f}"


def fci(lo, hi):
    return f"[{lo:.3f},{hi:.3f}]"


def fdelta(v):
    """양수에만 부호를 붙이고, 정의상 0 인 기준행은 '0.000'."""
    return "0.000" if abs(v) < 5e-4 else f"{v:+.3f}"


rows = []

# Option C — AC{n}_Biquad ±12
for bname, disp, pkey in [("AC1_BiLSTM_Biquad", "AC1_BiLSTM", "AC1_Biquad"),
                          ("AC2_GRU_Biquad", "AC2_GRU", "AC2_Biquad"),
                          ("AC3_Conformer_Biquad", "AC3_Conformer", "AC3_Biquad")]:
    c = cfg[f"{bname}_g12"]
    rows.append([f"{disp} Biquad (Option C, ±12)", f3(c["lsd_mean"]), fci(*c["ci"]),
                 fdelta(c["vs_a0"]), f"{num(perc[pkey]['pct_below_JND']):.1f}",
                 fdelta(c["penalty"])])

# Option A — SciPy 사후 피팅
a = optA12["AC2_GRU biquad-fitted"]
rows.append(["Option A SciPy fit ±12 (AC2)", f3(num(a["LSD mean"])), fci(*ci_of(a["95% CI"])),
             fdelta(num(a["LSD mean"]) - A0_MEAN), "",
             fdelta(num(optA12["Representation penalty"]["LSD mean"]))])

a6 = optA6["AC2_GRU biquad-fitted"]
# ±6 대조군은 개정 이전 산출물이며 원고 Table 4 에 인쇄되지 않는다. 커밋된 표와
# 같이 CI 는 싣지 않는다(소스에는 존재).
rows.append(["Option A SciPy fit ±6 (AC2, orig)", f3(num(a6["LSD mean"])), "—",
             fdelta(num(a6["LSD mean"]) - A0_MEAN), "",
             fdelta(num(optA6["Representation penalty"]["LSD mean"]))])

# Option D — naive 7-pt 표본화 (학습 모델 미사용 → gain bound 무관)
d = optD["AC2_GRU 7-pt sampled (Q=1.41)"]
rows.append(["Option D naive 7-pt (AC2)", f3(num(d["LSD mean"])), fci(*ci_of(d["95% CI"])),
             fdelta(num(d["LSD mean"]) - A0_MEAN), "", ""])

# dense AC2 (raw) — Table 1 의 AC2 행과 같은 수치
m = main["AC2 TCN+GRU"]
rows.append(["dense AC2 (raw, reference)", f3(num(m["lsd"])),
             fci(num(m["ci_lo"]), num(m["ci_hi"])),
             fdelta(num(m["lsd"]) - A0_MEAN), "", ""])

# A0 기준행
rows.append(["A0 Proposed ±12 (reference)", f3(A0_MEAN), fci(*a0["ci"]), "0.000", "—", ""])

# ── 출력 ────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
header = ["Configuration", "LSD", "95% CI", f"vs A0({A0_MEAN:.3f})", "% < JND", "repr. penalty"]
with io.open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

width = [38, 7, 15, 9, 8, 8]
print("=" * (sum(width) + 6))
print("Table 4 (tab:ac_fitting) — 조립 결과, 새 계산 없음")
print("=" * (sum(width) + 6))
print("".join(h.ljust(x) for h, x in zip(header, width)))
print("-" * (sum(width) + 6))
for r in rows:
    print("".join(str(c).ljust(x) for c, x in zip(r, width)))
print(f"\nA0 기준 = {A0_MEAN:.4f}, CI {a0['ci'][0]:.4f}..{a0['ci'][1]:.4f} "
      f"(N={a0['n']}, n_boot={a0['n_boot']}, seed={a0['bootstrap_seed']})")
print("% < JND 출처 = table7_perceptual.csv / Option D = 개정 이전 산출물(gain bound 무관)")
print(f"저장: {OUT}")
