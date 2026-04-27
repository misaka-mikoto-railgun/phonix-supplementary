from __future__ import annotations


CANONICAL_MODEL_ALIASES = {
    "A0": "A0_Proposed",
    "A0_Proposed": "A0_Proposed",
    "A0_Full": "A0_Proposed",
    "A1": "A1_NoRoomInput",
    "A1_NoRoomInput": "A1_NoRoomInput",
    "A1_NoRoomInput_CleanFeat": "A1_NoRoomInput",
    "A2": "A2_withPrefLoss",
    "A2_withPrefLoss": "A2_withPrefLoss",
    "A2_NoPrefLoss": "A2_withPrefLoss",
    "A3": "A3_NoPrefInput",
    "A3_NoPrefInput": "A3_NoPrefInput",
    "E1": "E1_NoEQ",
    "E1_NoEQ": "E1_NoEQ",
    "E2": "E2_StaticEQ",
    "E2_StaticEQ": "E2_StaticEQ",
    "E3": "E3_Nercessian",
    "E3_Nercessian": "E3_Nercessian",
    "E4": "E4_Pepe",
    "E4_Pepe": "E4_Pepe",
    "E5": "E5_Sequential",
    "E5_Sequential": "E5_Sequential",
    "E6": "E6_DSP",
    "E6_DSP": "E6_DSP",
    "AC1": "AC1_BiLSTM",
    "AC1_BiLSTM": "AC1_BiLSTM",
    "AC2": "AC2_GRU",
    "AC2_GRU": "AC2_GRU",
    "AC3": "AC3_Conformer",
    "AC3_Conformer": "AC3_Conformer",
}


CHECKPOINT_NAME_CANDIDATES = {
    "A0_Proposed": ["A0_Proposed", "A2_NoPrefLoss"],
    "A2_withPrefLoss": ["A2_withPrefLoss", "A0_Full"],
}


def canonical_model_name(name: str) -> str:
    return CANONICAL_MODEL_ALIASES.get(name, name)


def checkpoint_name_candidates(name: str) -> list[str]:
    canonical = canonical_model_name(name)
    return CHECKPOINT_NAME_CANDIDATES.get(canonical, [canonical])
