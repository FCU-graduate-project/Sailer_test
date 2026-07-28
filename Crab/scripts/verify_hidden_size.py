"""
A.5: Verify hidden_size compatibility for bilingual backbone swap.

Resolves BILINGUAL_FINETUNE_PLAN.md TODO line 290-291.

Expected: all four models have hidden_size == 1024 → cross-attention 不用改架構。
"""
from transformers import AutoModel, AutoConfig

MODELS = [
    # current Crab backbones (EN-only)
    ("roberta-large",                       "text_current_en"),
    ("microsoft/wavlm-large",               "audio_current_en"),
    # bilingual replacements
    ("FacebookAI/xlm-roberta-large",        "text_bilingual"),
    ("facebook/wav2vec2-xls-r-300m",        "audio_bilingual"),
]

print(f"{'model':<45} {'role':<20} {'hidden_size':>11}")
print("-" * 80)

results = {}
for model_path, role in MODELS:
    cfg = AutoConfig.from_pretrained(model_path)
    hs = cfg.hidden_size
    results[role] = hs
    print(f"{model_path:<45} {role:<20} {hs:>11}")

print()
print("=" * 80)
expected = 1024
all_match = all(v == expected for v in results.values())
if all_match:
    print(f"✅ ALL hidden_size == {expected} → bilingual swap is architecture-compatible.")
    print("   Cross-attention dim (1024) requires no change.")
    print("   Plan §3 line 61 confirmed by execution.")
else:
    print(f"⚠ MISMATCH detected — bilingual swap will break cross-attention.")
    for role, hs in results.items():
        if hs != expected:
            print(f"   {role}: {hs} != {expected}")
