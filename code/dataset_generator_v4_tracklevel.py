
"""
Dataset Generation Pipeline v3
One-Shot Dual-Objective Adaptive PEQ

주요 수정사항 (v2 -> v3):
  1. RIR 적용 후 peak-normalize 제거 (room response 과장 방지)
  2. room_response 추정을 log-spaced axis + fractional-octave smoothing으로 변경
  3. 합성 방 파라미터를 더 현실적으로 조정
  4. extreme sample reject / clamp 옵션 추가
  5. dual target / room target의 과도한 보정량 제한

사용법:
  python dataset_generator_v3.py --fma_dir ./fma_audio --output_dir ./data/dataset_v3
"""

import os
import json
import random
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None
import librosa
from scipy.signal import fftconvolve
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from frequency_grid import make_frequency_grid_np


@dataclass
class DatasetConfig:
    # paths
    fma_audio_dir:   str = "./fma_audio"
    openair_dir:     str = "./openair_rir"
    but_dir:         str = "./but_reverb_rir"
    output_dir:      str = "./data/dataset_v3"

    # audio
    sample_rate:     int   = 48000
    clip_duration:   float = 4.0
    hop_duration:    float = 2.0

    # features
    n_mfcc:          int = 8
    n_fft:           int = 2048
    hop_length:      int = 480
    win_length:      int = 1200
    seq_len:         int = 32
    n_freqs:         int = 128
    freq_min:        float = 20.0
    freq_max:        float = 24000.0
    freq_spacing:    str = "log"    # "log" or "linear"

    # dataset size
    n_train:         int = 40000
    n_val:           int = 5000
    n_test_synth:    int = 3000
    n_test_real:     int = 2000

    # synthetic room ranges (tempered from v2)
    room_size_min:   float = 3.5
    room_size_max:   float = 8.0
    rt60_min:        float = 0.20
    rt60_max:        float = 0.75
    mic_dist_min:    float = 0.7
    mic_dist_max:    float = 2.2
    wall_margin:     float = 0.6
    mic_jitter_xy:   float = 0.35

    # microphone estimation noise
    mic_noise_min:   float = 0.4
    mic_noise_max:   float = 2.0
    mic_bias_max:    float = 1.25
    mic_smooth_prob: float = 0.4

    # response smoothing / clipping
    room_smooth_kernel: int = 7
    room_response_clip_db: float = 12.0
    room_target_clip_db:   float = 8.0
    pref_target_clip_db:   float = 10.0
    dual_target_clip_db:   float = 12.0

    # realistic filtering
    enable_realistic_filter: bool = True
    max_room_var:        float = 60.0
    max_low_ptp:         float = 30.0
    max_dual_abs_mean:   float = 10.0
    max_dual_abs_max:    float = 15.0
    low_band_max_hz:     float = 300.0

    # generation attempts
    max_attempt_factor:  int = 8

    seed: int = 42


MODE_PROFILES = {
    0: {"name": "Vocal",  "gains": np.array([-4.0, -3.0,  0.0,  2.0,  5.0,  6.0,  4.0,  2.0, -1.0, -2.0])},
    1: {"name": "Bass",   "gains": np.array([ 8.0,  6.0,  3.0,  0.0, -1.0, -2.0, -3.0, -2.0, -1.0,  0.0])},
    2: {"name": "Treble", "gains": np.array([-2.0, -1.0,  0.0,  0.0,  1.0,  3.0,  5.0,  7.0,  8.0,  6.0])},
    3: {"name": "Soft",   "gains": np.array([ 2.0,  1.0,  1.0,  0.0, -1.0, -3.0, -5.0, -6.0, -4.0, -3.0])},
}
BAND_FREQS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000], dtype=np.float32)


def get_target_freqs(cfg: DatasetConfig) -> np.ndarray:
    return make_frequency_grid_np(
        n_freqs=cfg.n_freqs,
        f_min=cfg.freq_min,
        f_max=cfg.freq_max,
        spacing=cfg.freq_spacing,
    )


def smooth_curve(curve: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k <= 1:
        return curve.astype(np.float32)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    padded = np.pad(curve, (k // 2, k // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def low_band_mask(cfg: DatasetConfig) -> np.ndarray:
    freqs = get_target_freqs(cfg)
    return freqs <= cfg.low_band_max_hz


def simulate_rir(room_dims, source_pos, mic_pos, rt60, sample_rate):
    import pyroomacoustics as pra
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(
        room_dims,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    impulse = np.zeros(sample_rate, dtype=np.float32)
    impulse[0] = 1.0
    room.add_source(source_pos, signal=impulse)
    mic_array = np.array(mic_pos).reshape(3, 1)
    room.add_microphone(mic_array)
    room.simulate()
    rir = room.rir[0][0]
    peak = np.max(np.abs(rir))
    if peak > 0:
        rir = rir / peak
    return rir.astype(np.float32)


def generate_random_room_params(cfg: DatasetConfig, rng: np.random.Generator) -> dict:
    w = rng.uniform(cfg.room_size_min, cfg.room_size_max)
    l = rng.uniform(cfg.room_size_min, cfg.room_size_max)
    h = rng.uniform(2.3, 3.2)
    room_dims = np.array([w, l, h], dtype=np.float32)

    mic_x = np.clip(w / 2 + rng.uniform(-cfg.mic_jitter_xy, cfg.mic_jitter_xy), cfg.wall_margin, w - cfg.wall_margin)
    mic_y = np.clip(l / 2 + rng.uniform(-cfg.mic_jitter_xy, cfg.mic_jitter_xy), cfg.wall_margin, l - cfg.wall_margin)
    mic_z = 1.2
    mic_pos = np.array([mic_x, mic_y, mic_z], dtype=np.float32)

    dist = rng.uniform(cfg.mic_dist_min, cfg.mic_dist_max)
    angle = rng.uniform(0, 2 * np.pi)
    src_x = np.clip(mic_x + dist * np.cos(angle), cfg.wall_margin, w - cfg.wall_margin)
    src_y = np.clip(mic_y + dist * np.sin(angle), cfg.wall_margin, l - cfg.wall_margin)
    src_z = rng.uniform(0.8, 1.4)
    source_pos = np.array([src_x, src_y, src_z], dtype=np.float32)

    rt60 = float(rng.uniform(cfg.rt60_min, cfg.rt60_max))
    return {
        "room_dims": room_dims.tolist(),
        "source_pos": source_pos.tolist(),
        "mic_pos": mic_pos.tolist(),
        "rt60": rt60,
    }


def load_audio_clip(filepath: str, sample_rate: int, duration: float, offset: float = 0.0) -> Optional[np.ndarray]:
    try:
        audio, _ = librosa.load(filepath, sr=sample_rate, offset=offset, duration=duration, mono=True)
        target_len = int(duration * sample_rate)
        if len(audio) < target_len * 0.8:
            return None
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        return audio.astype(np.float32)
    except Exception:
        return None


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    v2의 문제였던 post-convolution peak normalize 제거.
    clean/reverb의 상대적 스펙트럼 비를 보존하기 위해 gain calibration 유지.
    """
    reverb = fftconvolve(audio, rir, mode="full")[: len(audio)]
    peak = np.max(np.abs(reverb))
    if peak > 1.0:
        reverb = reverb / peak * 0.98
    return reverb.astype(np.float32)


def extract_audio_features(audio: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    sr = cfg.sample_rate
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
    )
    energy = librosa.feature.rms(y=audio, frame_length=cfg.win_length, hop_length=cfg.hop_length)
    log_energy = np.log(energy + 1e-8)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    centroid_norm = centroid / (sr / 2)
    features = np.concatenate([mfcc, log_energy, centroid_norm], axis=0)

    T = features.shape[1]
    if T < cfg.seq_len:
        features = np.pad(features, ((0, 0), (0, cfg.seq_len - T)), mode="edge")
    else:
        indices = np.linspace(0, T - 1, cfg.seq_len, dtype=int)
        features = features[:, indices]

    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std
    return features.T.astype(np.float32)


def compute_room_response(clean_audio: np.ndarray, reverb_audio: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    raw FFT ratio 대신 STFT 평균 magnitude 비 + log-spaced axis + smoothing 사용.
    peak normalize로 인한 인위적 gain shift를 피하고, 지나치게 톱니 모양인 응답을 줄인다.
    """
    n_fft = cfg.n_fft
    hop = cfg.hop_length

    S_clean = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop, win_length=cfg.win_length, center=True)
    S_reverb = librosa.stft(reverb_audio, n_fft=n_fft, hop_length=hop, win_length=cfg.win_length, center=True)

    mag_clean = np.maximum(np.abs(S_clean), 1e-6)
    mag_reverb = np.maximum(np.abs(S_reverb), 1e-6)

    ratio = np.median(mag_reverb / mag_clean, axis=1)
    H_room_db = 20.0 * np.log10(ratio + 1e-8)

    freqs_fft = librosa.fft_frequencies(sr=cfg.sample_rate, n_fft=n_fft)
    target_freqs = get_target_freqs(cfg)
    room_response = np.interp(target_freqs, freqs_fft, H_room_db, left=H_room_db[0], right=H_room_db[-1])
    room_response = smooth_curve(room_response, cfg.room_smooth_kernel)
    room_response = np.clip(room_response, -cfg.room_response_clip_db, cfg.room_response_clip_db)
    return room_response.astype(np.float32)


def add_mic_noise(room_response_clean: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    n = len(room_response_clean)
    noisy = room_response_clean.copy()

    noise_std = rng.uniform(cfg.mic_noise_min, cfg.mic_noise_max)
    noisy += rng.normal(0.0, noise_std, size=n).astype(np.float32)

    bias_amp = rng.uniform(0.0, cfg.mic_bias_max)
    if bias_amp > 0.05:
        n_ctrl = int(rng.integers(3, 6))
        ctrl_x = np.linspace(0, n - 1, n_ctrl)
        ctrl_y = rng.normal(0.0, bias_amp, size=n_ctrl)
        ctrl_y[0] *= 1.35
        ctrl_y[-1] *= 1.15
        bias = np.interp(np.arange(n), ctrl_x, ctrl_y)
        noisy += bias.astype(np.float32)

    if rng.random() < cfg.mic_smooth_prob:
        noisy = smooth_curve(noisy, int(rng.integers(3, 7)))

    noisy = np.clip(noisy, -cfg.room_response_clip_db, cfg.room_response_clip_db)
    return noisy.astype(np.float32)


def compute_room_correction_target(room_response: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    correction = -room_response
    correction = smooth_curve(correction, max(3, cfg.room_smooth_kernel // 2))
    return np.clip(correction, -cfg.room_target_clip_db, cfg.room_target_clip_db).astype(np.float32)


def compute_preference_target(mode_id: int, band_gains: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    base = MODE_PROFILES[mode_id]["gains"].copy()
    perturbation = rng.uniform(-1.0, 1.0, size=10)
    perturbed = base + perturbation + band_gains * 0.55
    target_freqs = get_target_freqs(cfg)
    pref_db = np.interp(target_freqs, BAND_FREQS, perturbed, left=perturbed[0], right=perturbed[-1])
    pref_db = smooth_curve(pref_db, max(3, cfg.room_smooth_kernel // 2))
    return np.clip(pref_db, -cfg.pref_target_clip_db, cfg.pref_target_clip_db).astype(np.float32)


def compute_dual_target(room_correction: np.ndarray, pref_target: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    dual = room_correction + pref_target
    dual = smooth_curve(dual, max(3, cfg.room_smooth_kernel // 2))
    return np.clip(dual, -cfg.dual_target_clip_db, cfg.dual_target_clip_db).astype(np.float32)


def generate_band_gains(mode_id: int, rng: np.random.Generator) -> np.ndarray:
    base = MODE_PROFILES[mode_id]["gains"].copy()
    noise = rng.uniform(-2.5, 2.5, size=10)
    gains = base + noise
    return np.clip(gains, -10.0, 10.0).astype(np.float32)


def measure_sample_difficulty(room_response: np.ndarray, dual_target: np.ndarray, cfg: DatasetConfig) -> dict:
    low_mask = low_band_mask(cfg)
    low_curve = room_response[low_mask]
    metrics = {
        "room_var": float(np.var(room_response)),
        "low_ptp": float(np.ptp(low_curve)) if len(low_curve) else 0.0,
        "dual_abs_mean": float(np.mean(np.abs(dual_target))),
        "dual_abs_max": float(np.max(np.abs(dual_target))),
    }
    return metrics


def is_realistic_sample(metrics: dict, cfg: DatasetConfig) -> bool:
    if not cfg.enable_realistic_filter:
        return True
    return (
        metrics["room_var"] <= cfg.max_room_var and
        metrics["low_ptp"] <= cfg.max_low_ptp and
        metrics["dual_abs_mean"] <= cfg.max_dual_abs_mean and
        metrics["dual_abs_max"] <= cfg.max_dual_abs_max
    )


def collect_fma_files(fma_dir: str) -> List[str]:
    audio_files = []
    fma_path = Path(fma_dir)
    for ext in ["*.mp3", "*.wav", "*.WAV", "*.flac"]:
        for f in fma_path.rglob(ext):
            audio_files.append(str(f))
    return sorted(audio_files)


def build_track_id_map(files: List[str]) -> Dict[str, int]:
    """Stable integer id for each audio file path inside a split."""
    return {str(path): idx for idx, path in enumerate(sorted(map(str, files)))}


def build_string_id_map(items: List[str]) -> Dict[str, int]:
    """Stable integer id for arbitrary string identifiers (e.g., RIR files)."""
    return {str(item): idx for idx, item in enumerate(sorted(map(str, items)))}


def save_id_map_json(path: Path, mapping: Dict[str, int], field_name: str) -> None:
    payload = {
        field_name: [
            {"id": int(idx), "value": value}
            for value, idx in sorted(mapping.items(), key=lambda kv: kv[1])
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def split_files(files: List[str], cfg: DatasetConfig, rng: np.random.Generator) -> Dict[str, List[str]]:
    rng.shuffle(files)
    n_total = len(files)
    n_test_real = max(200, int(n_total * 0.10))
    n_test_synth = max(200, int(n_total * 0.10))
    n_val = max(200, int(n_total * 0.08))

    test_real_files = files[:n_test_real]
    test_synth_files = files[n_test_real:n_test_real + n_test_synth]
    val_files = files[n_test_real + n_test_synth:n_test_real + n_test_synth + n_val]
    train_files = files[n_test_real + n_test_synth + n_val:]

    print("파일 분할:")
    print(f"  Train:      {len(train_files):,}개")
    print(f"  Val:        {len(val_files):,}개")
    print(f"  Test-Synth: {len(test_synth_files):,}개")
    print(f"  Test-Real:  {len(test_real_files):,}개")
    return {
        "train": train_files,
        "val": val_files,
        "test_synth": test_synth_files,
        "test_real": test_real_files,
    }


def collect_real_rir_files(openair_dir: str, but_dir: str) -> List[str]:
    result = {"openair": [], "but": []}
    openair_path = Path(openair_dir)
    if openair_path.exists():
        rooms_found = {}
        for wav in openair_path.rglob("*.wav"):
            parts = wav.relative_to(openair_path).parts
            if len(parts) < 1:
                continue
            room_name = parts[0]
            format_hint = parts[1].lower() if len(parts) > 1 else ""
            is_mono = "mono" in format_hint or "stereo" in format_hint
            if room_name not in rooms_found or is_mono:
                rooms_found[room_name] = str(wav)
        result["openair"] = list(rooms_found.values())
        print(f"  OpenAIR: {len(result['openair'])}개 방 ({openair_path})")
    else:
        print(f"  경고: {openair_dir} 없음")

    but_path = Path(but_dir)
    if but_path.exists():
        rooms_found = {}
        for wav in but_path.rglob("*.wav"):
            parts = wav.relative_to(but_path).parts
            if len(parts) < 1:
                continue
            room_name = parts[0]
            rooms_found.setdefault(room_name, []).append(str(wav))
        for room_name, files in rooms_found.items():
            result["but"].extend(files[:3] if len(files) > 3 else files)
        print(f"  BUT ReverbDB: {len(rooms_found)}개 방, {len(result['but'])}개 RIR ({but_path})")
    else:
        print(f"  경고: {but_dir} 없음")
    total = len(result["openair"]) + len(result["but"])
    print(f"  실측 RIR 총 {total}개")
    return result["openair"] + result["but"]


def load_rir_file(filepath: str, sample_rate: int) -> Optional[np.ndarray]:
    try:
        if sf is not None:
            info = sf.info(filepath)
            n_channels = info.channels
            if n_channels > 1:
                audio, sr = sf.read(filepath, dtype='float32')
                rir = audio[:, 0]
                if sr != sample_rate:
                    rir = librosa.resample(rir, orig_sr=sr, target_sr=sample_rate)
            else:
                rir, _ = librosa.load(filepath, sr=sample_rate, mono=True)
        else:
            rir, _ = librosa.load(filepath, sr=sample_rate, mono=True)

        min_len = int(0.01 * sample_rate)
        max_len = int(5.0 * sample_rate)
        if len(rir) < min_len:
            return None
        if len(rir) > max_len:
            rir = rir[:max_len]
        if np.max(np.abs(rir)) < 1e-6:
            return None
        rir = rir / np.max(np.abs(rir))
        return rir.astype(np.float32)
    except Exception:
        return None


def generate_single_sample(
    audio_file: str,
    room_params: dict,
    mode_id: int,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    rir: Optional[np.ndarray] = None,
    verbose: bool = False,
    preloaded_audio: Optional[np.ndarray] = None,
    clip_start_sample: int = 0,
) -> Optional[dict]:
    if preloaded_audio is not None:
        clean = preloaded_audio
    else:
        offset = rng.uniform(0, 30.0)
        clip_start_sample = int(offset * cfg.sample_rate)
        clean = load_audio_clip(audio_file, cfg.sample_rate, cfg.clip_duration, offset)
        if clean is None:
            if verbose:
                print(f"    [FAIL] load_audio_clip: {audio_file}")
            return None

    if rir is None:
        try:
            rir = simulate_rir(
                np.array(room_params["room_dims"]),
                np.array(room_params["source_pos"]),
                np.array(room_params["mic_pos"]),
                room_params["rt60"],
                cfg.sample_rate,
            )
        except Exception as e:
            if verbose:
                print(f"    [FAIL] simulate_rir: {e}")
            return None

    reverb = apply_rir(clean, rir)

    try:
        features = extract_audio_features(reverb, cfg)
        features_clean  = extract_audio_features(clean, cfg)
        room_response_clean = compute_room_response(clean, reverb, cfg)
    except Exception as e:
        if verbose:
            print(f"    [FAIL] feature/room_response: {e}")
        return None

    room_response_noisy = add_mic_noise(room_response_clean, cfg, rng)
    band_gains = generate_band_gains(mode_id, rng)
    room_target = compute_room_correction_target(room_response_clean, cfg)
    pref_target = compute_preference_target(mode_id, band_gains, cfg, rng)
    dual_target = compute_dual_target(room_target, pref_target, cfg)

    metrics = measure_sample_difficulty(room_response_clean, dual_target, cfg)
    if not is_realistic_sample(metrics, cfg):
        return None

    return {
        "features": features,
        "features_clean": features_clean,
        "room_response": room_response_noisy,
        "mode_id": int(mode_id),
        "band_gains": band_gains,
        "room_target": room_target,
        "pref_target": pref_target,
        "dual_target": dual_target,
        "room_params": room_params,
        "features_clean": features_clean,
        "difficulty": metrics,
        "audio_file": str(audio_file),
        "clip_start_sample": int(clip_start_sample),
    }


def _save_chunk(
    save_dir: Path,
    chunk_idx: int,
    features,
    room_responses,
    mode_ids,
    band_gains,
    room_targets,
    pref_targets,
    dual_targets,
    features_clean_list,
    track_ids,
    room_ids,
    clip_start_samples,
    target_freqs: np.ndarray,
    pair_ids: Optional[List[int]] = None,
):
    payload = {
        "features": np.array(features, dtype=np.float32),
        "room_response": np.array(room_responses, dtype=np.float32),
        "mode_id": np.array(mode_ids, dtype=np.int32),
        "band_gains": np.array(band_gains, dtype=np.float32),
        "room_target": np.array(room_targets, dtype=np.float32),
        "pref_target": np.array(pref_targets, dtype=np.float32),
        "dual_target": np.array(dual_targets, dtype=np.float32),
        "features_clean": np.array(features_clean_list, dtype=np.float32),
        "track_id": np.array(track_ids, dtype=np.int32),
        "room_id": np.array(room_ids, dtype=np.int32),
        "clip_start_sample": np.array(clip_start_samples, dtype=np.int32),
        "target_freqs": np.array(target_freqs, dtype=np.float32),
    }
    if pair_ids is not None:
        payload["pair_id"] = np.array(pair_ids, dtype=np.int32)
    np.savez_compressed(save_dir / f"chunk_{chunk_idx:04d}.npz", **payload)


def generate_dataset_split(split_name: str, audio_files: List[str], n_samples: int, cfg: DatasetConfig,
                           rng: np.random.Generator, real_rir_files: Optional[List[str]] = None) -> None:
    save_dir = Path(cfg.output_dir) / split_name
    save_dir.mkdir(parents=True, exist_ok=True)
    target_freqs = get_target_freqs(cfg)

    is_real = real_rir_files is not None and len(real_rir_files) > 0
    chunk_size = 1000
    track_id_map = build_track_id_map(audio_files)
    rir_id_map = build_string_id_map(real_rir_files) if is_real else {}

    n_preload = min(len(audio_files), n_samples * 2)
    preload_files = audio_files[:n_preload]
    print(f"  오디오 {n_preload}개 프리로드 중 (전체 {len(audio_files)}개 중)...")
    audio_cache: Dict[str, np.ndarray] = {}
    for path in tqdm(preload_files, desc="  preload"):
        if path not in audio_cache:
            clip = load_audio_clip(path, cfg.sample_rate, 34.0, 0.0)
            if clip is not None:
                audio_cache[path] = clip
    cache_keys = list(audio_cache.keys())
    print(f"  프리로드 완료: {len(cache_keys)}개")

    if not is_real:
        n_rir_pool = min(500, n_samples * 2)
        print(f"  RIR {n_rir_pool}개 미리 계산 중 (CPU)...")
        rir_pool, rir_params_pool = [], []
        for _ in tqdm(range(n_rir_pool), desc="  RIR pool"):
            params = generate_random_room_params(cfg, rng)
            try:
                rir = simulate_rir(
                    np.array(params["room_dims"]),
                    np.array(params["source_pos"]),
                    np.array(params["mic_pos"]),
                    params["rt60"],
                    cfg.sample_rate,
                )
                rir_pool.append(rir)
                rir_params_pool.append(params)
            except Exception:
                continue
        print(f"  RIR 풀 완료: {len(rir_pool)}개")
    else:
        rir_pool, rir_params_pool = None, None

    all_features = []
    all_room_responses = []
    all_mode_ids = []
    all_band_gains = []
    all_room_targets = []
    all_pref_targets = []
    all_dual_targets = []
    all_features_clean = []
    all_track_ids = []
    all_room_ids = []
    all_clip_start_samples = []

    chunk_idx = 0
    generated = 0
    attempted = 0
    rejected = 0
    pbar = tqdm(total=n_samples, desc=f"  {split_name}")

    while generated < n_samples:
        attempted += 1
        if attempted > n_samples * cfg.max_attempt_factor:
            print(f"  경고: {split_name} 샘플 생성 실패/거절 과다. 중단.")
            break

        audio_file = cache_keys[rng.integers(len(cache_keys))] if cache_keys else audio_files[rng.integers(len(audio_files))]
        audio_file = str(audio_file)
        mode_id = int(generated % 4)

        if is_real:
            rir_file = str(real_rir_files[rng.integers(len(real_rir_files))])
            rir = load_rir_file(rir_file, cfg.sample_rate)
            if rir is None:
                continue
            room_params = {"source": "real_rir", "file": str(rir_file)}
            room_id = int(rir_id_map[rir_file])
        else:
            if not rir_pool:
                continue
            pool_idx = int(rng.integers(len(rir_pool)))
            rir = rir_pool[pool_idx]
            room_params = rir_params_pool[pool_idx]
            room_id = pool_idx

        verbose = attempted <= 10

        if audio_file in audio_cache:
            clean = audio_cache[audio_file]
            clip_len = int(cfg.clip_duration * cfg.sample_rate)
            clip_start_sample = 0
            if len(clean) > clip_len:
                clip_start_sample = int(rng.integers(0, len(clean) - clip_len))
                clean = clean[clip_start_sample:clip_start_sample + clip_len]
            sample = generate_single_sample(
                audio_file, room_params, mode_id, cfg, rng, rir,
                verbose=verbose, preloaded_audio=clean, clip_start_sample=clip_start_sample
            )
        else:
            sample = generate_single_sample(audio_file, room_params, mode_id, cfg, rng, rir, verbose=verbose)

        if sample is None:
            rejected += 1
            continue

        all_features.append(sample["features"])
        all_room_responses.append(sample["room_response"])
        all_mode_ids.append(sample["mode_id"])
        all_band_gains.append(sample["band_gains"])
        all_room_targets.append(sample["room_target"])
        all_pref_targets.append(sample["pref_target"])
        all_dual_targets.append(sample["dual_target"])
        all_features_clean.append(sample["features_clean"])
        all_track_ids.append(int(track_id_map[sample["audio_file"]]))
        all_room_ids.append(int(room_id))
        all_clip_start_samples.append(int(sample["clip_start_sample"]))
        generated += 1
        pbar.update(1)

        if len(all_features) >= chunk_size:
            _save_chunk(
                save_dir, chunk_idx, all_features, all_room_responses, all_mode_ids,
                all_band_gains, all_room_targets, all_pref_targets, all_dual_targets,
                all_features_clean, all_track_ids, all_room_ids, all_clip_start_samples,
                target_freqs=target_freqs,
            )
            chunk_idx += 1
            all_features.clear()
            all_room_responses.clear()
            all_mode_ids.clear()
            all_band_gains.clear()
            all_room_targets.clear()
            all_pref_targets.clear()
            all_dual_targets.clear()
            all_features_clean.clear()
            all_track_ids.clear()
            all_room_ids.clear()
            all_clip_start_samples.clear()
    if all_features:
        _save_chunk(
            save_dir, chunk_idx, all_features, all_room_responses, all_mode_ids,
            all_band_gains, all_room_targets, all_pref_targets, all_dual_targets,
            all_features_clean, all_track_ids, all_room_ids, all_clip_start_samples,
            target_freqs=target_freqs,
        )
        chunk_idx += 1

    pbar.close()

    meta = {
        "split": split_name,
        "n_samples": generated,
        "n_chunks": chunk_idx,
        "chunk_size": chunk_size,
        "is_real_rir": is_real,
        "version": 4,
        "rejected_samples": rejected,
        "n_unique_tracks": len(track_id_map),
        "has_track_level_metadata": True,
        "freq_spacing": cfg.freq_spacing,
        "freq_min": cfg.freq_min,
        "freq_max": cfg.freq_max,
        "n_freqs": cfg.n_freqs,
        "target_freqs": target_freqs.tolist(),
        "config": asdict(cfg),
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_id_map_json(save_dir / "track_map.json", track_id_map, "tracks")
    if is_real:
        save_id_map_json(save_dir / "rir_map.json", rir_id_map, "rirs")
    print(f"  {split_name}: {generated:,}개 샘플, {chunk_idx}개 청크 저장 완료 (rejected={rejected:,})")

def generate_base_case(
    audio_file: str,
    room_params: dict,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    rir: Optional[np.ndarray] = None,
    preloaded_audio: Optional[np.ndarray] = None,
    clip_start_sample: int = 0,
):
    if preloaded_audio is not None:
        clean = preloaded_audio
    else:
        offset = rng.uniform(0, 30.0)
        clip_start_sample = int(offset * cfg.sample_rate)
        clean = load_audio_clip(audio_file, cfg.sample_rate, cfg.clip_duration, offset)
        if clean is None:
            return None

    if rir is None:
        try:
            rir = simulate_rir(
                np.array(room_params["room_dims"]),
                np.array(room_params["source_pos"]),
                np.array(room_params["mic_pos"]),
                room_params["rt60"],
                cfg.sample_rate,
            )
        except Exception:
            return None

    reverb = apply_rir(clean, rir)

    try:
        features = extract_audio_features(reverb, cfg)
        features_clean = extract_audio_features(clean, cfg)
        room_response_clean = compute_room_response(clean, reverb, cfg)
    except Exception:
        return None

    room_response_noisy = add_mic_noise(room_response_clean, cfg, rng)
    room_target = compute_room_correction_target(room_response_clean, cfg)

    return {
        "audio_file": audio_file,
        "clean": clean,
        "rir": rir,
        "room_params": room_params,
        "features": features,
        "features_clean": features_clean,
        "room_response": room_response_noisy,
        "room_target": room_target,
        "clip_start_sample": int(clip_start_sample),
    }

def expand_base_case_to_modes(base_case: dict, cfg: DatasetConfig, rng: np.random.Generator):
    samples = []
    for mode_id in range(4):
        band_gains = generate_band_gains(mode_id, rng)
        pref_target = compute_preference_target(mode_id, band_gains, cfg, rng)
        dual_target = compute_dual_target(base_case["room_target"], pref_target, cfg)

        metrics = measure_sample_difficulty(base_case["room_response"], dual_target, cfg)
        if not is_realistic_sample(metrics, cfg):
            return None

        samples.append({
            "features": base_case["features"],
            "room_response": base_case["room_response"],
            "features_clean": base_case["features_clean"],
            "mode_id": mode_id,
            "band_gains": band_gains,
            "room_target": base_case["room_target"],
            "pref_target": pref_target,
            "dual_target": dual_target,
            "room_params": base_case["room_params"],
            "audio_file": base_case["audio_file"],
            "clip_start_sample": int(base_case["clip_start_sample"]),
        })
    return samples

def generate_paired_mode_test_split(
    split_name: str,
    audio_files: List[str],
    n_pairs: int,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    real_rir_files: Optional[List[str]] = None,
):
    save_dir = Path(cfg.output_dir) / split_name
    save_dir.mkdir(parents=True, exist_ok=True)
    target_freqs = get_target_freqs(cfg)

    is_real = real_rir_files is not None and len(real_rir_files) > 0
    chunk_size = 1000
    track_id_map = build_track_id_map(audio_files)
    rir_id_map = build_string_id_map(real_rir_files) if is_real else {}

    all_features = []
    all_room_responses = []
    all_mode_ids = []
    all_band_gains = []
    all_room_targets = []
    all_pref_targets = []
    all_dual_targets = []
    all_features_clean = []
    all_pair_ids = []
    all_track_ids = []
    all_room_ids = []
    all_clip_start_samples = []

    chunk_idx = 0
    generated_pairs = 0
    attempted = 0

    pbar = tqdm(total=n_pairs, desc=f"  {split_name}")

    while generated_pairs < n_pairs:
        attempted += 1
        if attempted > n_pairs * cfg.max_attempt_factor:
            print(f"  경고: {split_name} pair 생성 실패/거절 과다. 중단.")
            break

        audio_file = str(audio_files[rng.integers(len(audio_files))])

        if is_real:
            rir_file = str(real_rir_files[rng.integers(len(real_rir_files))])
            rir = load_rir_file(rir_file, cfg.sample_rate)
            if rir is None:
                continue
            room_params = {"source": "real_rir", "file": str(rir_file)}
            room_id = int(rir_id_map[rir_file])
        else:
            room_params = generate_random_room_params(cfg, rng)
            rir = None
            room_id = int(generated_pairs)

        base_case = generate_base_case(audio_file, room_params, cfg, rng, rir=rir)
        if base_case is None:
            continue

        samples = expand_base_case_to_modes(base_case, cfg, rng)
        if samples is None:
            continue

        pair_id = generated_pairs
        for s in samples:
            all_features.append(s["features"])
            all_room_responses.append(s["room_response"])
            all_mode_ids.append(s["mode_id"])
            all_band_gains.append(s["band_gains"])
            all_room_targets.append(s["room_target"])
            all_pref_targets.append(s["pref_target"])
            all_dual_targets.append(s["dual_target"])
            all_features_clean.append(s["features_clean"])
            all_pair_ids.append(pair_id)
            all_track_ids.append(int(track_id_map[s["audio_file"]]))
            all_room_ids.append(int(room_id))
            all_clip_start_samples.append(int(s["clip_start_sample"]))

        generated_pairs += 1
        pbar.update(1)

        if len(all_features) >= chunk_size:
            _save_chunk(
                save_dir, chunk_idx, all_features, all_room_responses, all_mode_ids,
                all_band_gains, all_room_targets, all_pref_targets, all_dual_targets,
                all_features_clean, all_track_ids, all_room_ids, all_clip_start_samples,
                target_freqs=target_freqs,
                pair_ids=all_pair_ids,
            )
            chunk_idx += 1
            all_features.clear()
            all_room_responses.clear()
            all_mode_ids.clear()
            all_band_gains.clear()
            all_room_targets.clear()
            all_pref_targets.clear()
            all_dual_targets.clear()
            all_features_clean.clear()
            all_pair_ids.clear()
            all_track_ids.clear()
            all_room_ids.clear()
            all_clip_start_samples.clear()

    if all_features:
        _save_chunk(
            save_dir, chunk_idx, all_features, all_room_responses, all_mode_ids,
            all_band_gains, all_room_targets, all_pref_targets, all_dual_targets,
            all_features_clean, all_track_ids, all_room_ids, all_clip_start_samples,
            target_freqs=target_freqs,
            pair_ids=all_pair_ids,
        )
        chunk_idx += 1

    pbar.close()

    meta = {
        "split": split_name,
        "n_pairs": generated_pairs,
        "n_samples": generated_pairs * 4,
        "n_chunks": chunk_idx,
        "paired_mode_switch": True,
        "version": 4,
        "has_track_level_metadata": True,
        "n_unique_tracks": len(track_id_map),
        "freq_spacing": cfg.freq_spacing,
        "freq_min": cfg.freq_min,
        "freq_max": cfg.freq_max,
        "n_freqs": cfg.n_freqs,
        "target_freqs": target_freqs.tolist(),
        "config": asdict(cfg),
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    save_id_map_json(save_dir / "track_map.json", track_id_map, "tracks")
    if is_real:
        save_id_map_json(save_dir / "rir_map.json", rir_id_map, "rirs")

def run_pipeline(cfg: DatasetConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)

    print("=" * 60)
    print("One-Shot Dual-Objective PEQ 데이터셋 생성 v4 (track-level metadata)")
    print("=" * 60)

    print("\n[1/5] FMA 오디오 파일 수집 중...")
    fma_files = collect_fma_files(cfg.fma_audio_dir)
    if not fma_files:
        print(f"  오류: {cfg.fma_audio_dir}에서 파일을 찾을 수 없음")
        return
    print(f"  총 {len(fma_files):,}개 파일 발견")

    print("\n[2/5] 파일 분할 (곡 레벨)...")
    splits = split_files(fma_files, cfg, rng)

    print("\n[3/6] 실측 RIR 파일 수집 중...")
    real_rir_files = collect_real_rir_files(cfg.openair_dir, cfg.but_dir)
    if not real_rir_files:
        print("  경고: 실측 RIR 없음. Test-Real 세트 스킵.")
    else:
        print(f"  실측 RIR {len(real_rir_files)}개 발견")

    print("\n[4/6] 합성 데이터셋 생성 중...")
    for split_name, files, n_samples in [
        ("train", splits["train"], cfg.n_train),
        ("val", splits["val"], cfg.n_val),
        ("test_synth", splits["test_synth"], cfg.n_test_synth),
    ]:
        print(f"\n  [{split_name}] {n_samples:,}개 생성...")
        generate_dataset_split(split_name, files, n_samples, cfg, rng)

    if real_rir_files:
        print("\n[5/6] 실제 환경 테스트셋 생성 중...")
        generate_dataset_split("test_real", splits["test_real"], cfg.n_test_real, cfg, rng,
                               real_rir_files=real_rir_files)
    else:
        print("\n[5/6] 실측 RIR 없으므로 Test-Real 스킵")

    print("\n[6/6] Paired mode-switch test set 생성 중...")
    generate_paired_mode_test_split(
        "paired_mode_test",
        splits["val"],
        n_pairs=500,
        cfg=cfg,
        rng=rng,
        real_rir_files=None,
    )

    print("\n" + "=" * 60)
    print("데이터셋 생성 완료!")
    print(f"저장 위치: {cfg.output_dir}")
    print("=" * 60)

class PEQDataset:
    def __init__(self, split_dir: str, device: str = "cpu"):
        import torch
        self.split_dir = Path(split_dir)
        self.device = torch.device(device)

        with open(self.split_dir / "meta.json") as f:
            self.meta = json.load(f)

        chunk_files = sorted(self.split_dir.glob("chunk_*.npz"))
        base_keys = [
            "features", "room_response", "mode_id",
            "band_gains", "room_target", "pref_target", "dual_target"
        ]
        optional_keys = ["features_clean", "track_id", "room_id", "clip_start_sample", "pair_id"]

        arrays = {k: [] for k in base_keys + optional_keys}

        for chunk_path in chunk_files:
            data = np.load(chunk_path, allow_pickle=False)
            for k in base_keys:
                arrays[k].append(data[k])
            for k in optional_keys:
                if k in data:
                    arrays[k].append(data[k])

        self.data = {}
        for k, v in arrays.items():
            if not v:
                continue
            t = torch.from_numpy(np.concatenate(v))
            if k in {"mode_id", "track_id", "room_id", "clip_start_sample", "pair_id"}:
                t = t.long()
            self.data[k] = t.to(self.device)

        self.n_samples = len(self.data["features"])
        mem_mb = sum(t.nbytes for t in self.data.values()) / 1024 / 1024
        print(f"  [{split_dir}] {self.n_samples:,}개 샘플, {mem_mb:.1f}MB on {self.device}")

    def __len__(self):
        return self.n_samples

    def iter_batches(self, batch_size: int = 512, shuffle: bool = True):
        import torch
        if shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)
        for start in range(0, self.n_samples, batch_size):
            idx = indices[start:start + batch_size]
            yield {k: v[idx] for k, v in self.data.items()}

    def get_all(self):
        return self.data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fma_dir", default="./fma_audio")
    parser.add_argument("--openair_dir", default="./openair_rir")
    parser.add_argument("--but_dir", default="./but_reverb_rir")
    parser.add_argument("--output_dir", default="./data/dataset_v3")
    parser.add_argument("--n_train", type=int, default=40000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--n_mfcc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small", action="store_true", help="소규모 테스트 (각 100샘플)")
    parser.add_argument("--disable_realistic_filter", action="store_true")
    parser.add_argument("--freq_spacing", choices=["linear", "log"], default="log")
    args = parser.parse_args()

    cfg = DatasetConfig(
        fma_audio_dir=args.fma_dir,
        openair_dir=args.openair_dir,
        but_dir=args.but_dir,
        output_dir=args.output_dir,
        n_train=100 if args.small else args.n_train,
        n_val=20 if args.small else args.n_val,
        n_test_synth=20 if args.small else args.n_test,
        n_test_real=20 if args.small else 2000,
        n_mfcc=args.n_mfcc,
        seed=args.seed,
        enable_realistic_filter=not args.disable_realistic_filter,
        freq_spacing=args.freq_spacing,
    )
    run_pipeline(cfg)
