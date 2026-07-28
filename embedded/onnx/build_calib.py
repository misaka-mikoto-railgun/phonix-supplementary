"""build_calib.py — PTQ INT8 calibration set 구성 (train set 층화 추출).

스펙(확정):
  - 256 샘플, mode 4개 균등(64/mode), distinct room 최대화.
  - 입력 band_gains 극단(|bg|>9) 쿼터(16/mode) + feature x 꼬리 쿼터(2/mode).
  - 각 입력(feat/room/band)의 전역 min/max anchor 강제 포함(범위 양극단 보장).
  - 랜덤 슬라이스 금지(층화). seed=42 tie-break.

출력: A0/AC 입력명(feat,room,mode_onehot,band) 일치 npz (E 모델은 feat 만 사용).
onnxruntime.quantization static(QDQ) calibration → ST Edge AI Core import.
"""
import sys, glob, json
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent.parent
TRAIN = ORIG / "data" / "dataset_v3" / "train"
CHUNK = 1000

# ── load full train (global index = chunk*1000 + row) ───────────────────────
chunks = sorted(glob.glob(str(TRAIN / "chunk_*.npz")))
A = {k: [] for k in ["features", "room_response", "mode_id", "band_gains", "room_id", "track_id"]}
for c in chunks:
    d = np.load(c, allow_pickle=False)
    for k in A:
        A[k].append(d[k])
feats = np.concatenate(A["features"]).astype(np.float32)   # (N,32,10)
rooms = np.concatenate(A["room_response"]).astype(np.float32)  # (N,128)
modes = np.concatenate(A["mode_id"]).astype(np.int64)      # (N,)
bgains = np.concatenate(A["band_gains"]).astype(np.float32)  # (N,10)
room_ids = np.concatenate(A["room_id"]).astype(np.int64)
track_ids = np.concatenate(A["track_id"]).astype(np.int64)
N = len(modes)

bg_absmax = np.abs(bgains).max(1)
x_absmax = np.abs(feats).reshape(N, -1).max(1)
EXT = bg_absmax > 9.0

PER_MODE, EXT_QUOTA, XTAIL_QUOTA = 64, 16, 2
rng = np.random.default_rng(42)

# ── stratified selection ────────────────────────────────────────────────────
used_rooms, selected = set(), []
for m in range(4):
    pool = np.where(modes == m)[0]
    picks = []
    # (a) extreme input gain: 큰 |bg| 우선, 새 room 선호
    ext = pool[EXT[pool]]
    ext = ext[np.argsort(-bg_absmax[ext])]
    for i in ext:
        if len(picks) >= EXT_QUOTA: break
        if room_ids[i] in used_rooms: continue
        picks.append(int(i)); used_rooms.add(room_ids[i])
    for i in ext:                                   # 쿼터 못 채우면 room 중복 허용
        if len(picks) >= EXT_QUOTA: break
        if int(i) in picks: continue
        picks.append(int(i)); used_rooms.add(room_ids[i])
    # (b) feature x 꼬리: 큰 |x| 우선, 새 room 선호
    xt = pool[np.argsort(-x_absmax[pool])]; added = 0
    for i in xt:
        if added >= XTAIL_QUOTA: break
        if int(i) in picks or room_ids[i] in used_rooms: continue
        picks.append(int(i)); used_rooms.add(room_ids[i]); added += 1
    # (c) 나머지: distinct room 최대화
    order = rng.permutation(pool)
    for i in order:
        if len(picks) >= PER_MODE: break
        if int(i) in picks or room_ids[i] in used_rooms: continue
        picks.append(int(i)); used_rooms.add(room_ids[i])
    for i in order:                                 # room 소진 시 중복 허용
        if len(picks) >= PER_MODE: break
        if int(i) in picks: continue
        picks.append(int(i))
    selected += picks[:PER_MODE]

# ── per-input 전역 극단 anchor 강제 포함(범위 보장) ─────────────────────────
anchors = set()
for arr in (feats.reshape(N, -1), rooms, bgains):
    anchors.add(int(np.unravel_index(np.argmax(arr), arr.shape)[0]))
    anchors.add(int(np.unravel_index(np.argmin(arr), arr.shape)[0]))
sel = list(selected); sel_set = set(sel)
for a in sorted(anchors):
    if a in sel_set: continue
    m = modes[a]
    rc = Counter(room_ids[p] for p in sel)
    cand = [p for p in sel if modes[p] == m and p not in anchors]
    # 가장 중복된 room + 가장 안 극단인 pick 을 drop (room 다양성·극단 손실 최소)
    cand.sort(key=lambda p: (rc[room_ids[p]], -max(bg_absmax[p], x_absmax[p])), reverse=True)
    drop = cand[0]
    sel.remove(drop); sel_set.discard(drop); sel.append(a); sel_set.add(a)
selected = np.array(sorted(sel))
assert len(selected) == 256

# ── calib 배열 (onnx 입력명 일치) ───────────────────────────────────────────
n_modes = 4
mode_oh = np.eye(n_modes, dtype=np.float32)[modes[selected]]   # (256,4)
calib = {
    "feat": feats[selected],                       # (256,32,10)
    "room": rooms[selected],                        # (256,128)
    "mode_onehot": mode_oh,                          # (256,4)
    "band": bgains[selected],                        # (256,10)
}
np.savez(HERE / "calib.npz", **calib)

# ── manifest (재현용 인덱스) ────────────────────────────────────────────────
man = {"n": 256, "n_modes_each": 64, "source": "data/dataset_v3/train",
       "chunk_size": CHUNK, "seed": 42,
       "stratify": {"ext_gain_quota_per_mode": EXT_QUOTA, "ext_gain_thr_absbandgain": 9.0,
                    "xtail_quota_per_mode": XTAIL_QUOTA, "range_anchors": sorted(anchors)},
       "samples": []}
for gi in selected.tolist():
    man["samples"].append({
        "global_idx": gi, "chunk": gi // CHUNK, "row": gi % CHUNK,
        "mode": int(modes[gi]), "room_id": int(room_ids[gi]), "track_id": int(track_ids[gi]),
        "bandgain_absmax": round(float(bg_absmax[gi]), 3),
        "x_absmax": round(float(x_absmax[gi]), 3),
        "is_ext_gain": bool(EXT[gi]), "is_anchor": gi in anchors,
    })
(HERE / "calib_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

# ── coverage 요약 ───────────────────────────────────────────────────────────
sm, sr, sbg, sx = modes[selected], room_ids[selected], bgains[selected], feats[selected]
print("=" * 60); print("CALIBRATION SET COVERAGE (n=256)"); print("=" * 60)
print("mode counts:", {int(k): int(v) for k, v in zip(*np.unique(sm, return_counts=True))})
print(f"distinct rooms: {len(np.unique(sr))} / 500")
print(f"|band_gain|>9 samples: {(np.abs(sbg).max(1)>9).sum()}  (train rate 16.8%)")
print(f"x |.|>4 samples: {(np.abs(sx).reshape(256,-1).max(1)>4).sum()}")
print("\nper-input range  (calib  vs  train):")
for nm, cal, tr in [("feat", sx, feats), ("room", sr if False else calib['room'], rooms), ("band", sbg, bgains)]:
    print(f"  {nm:5s} calib[{cal.min():8.3f}, {cal.max():8.3f}]   train[{tr.min():8.3f}, {tr.max():8.3f}]")
print("\nsaved: calib.npz (feat/room/mode_onehot/band), calib_manifest.json")
