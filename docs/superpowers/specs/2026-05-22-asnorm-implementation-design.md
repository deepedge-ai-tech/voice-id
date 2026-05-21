# AS-Norm Implementation for Voice-ID Speaker Recognition

## Overview

Add AS-Norm (Adaptive Score Normalization) to the Voice-ID speaker recognition system. AS-Norm calibrates raw cosine similarity scores using a cohort of imposter speakers, reducing false positives from channel/accent/environment variation.

## Source of Truth

The existing general guide at `docs/AS-Norm.md` describes the theory. This spec defines the concrete implementation.

## Cohorts

### Data Sources

| Source | Speakers | Language | Access |
|--------|----------|----------|--------|
| VoxCeleb1 dev subset | ~300 | English | HuggingFace `voxceleb/voxceleb1` (requires user agreement) |
| CN-Celeb subset | ~200 | Chinese | HuggingFace `cn-celeb/cn-celeb` (requires user agreement) |
| **Total** | **~500** | **zh+en** | ~2-3GB streaming, ~512KB final embedding file |

### Download Strategy

Stream from HuggingFace `datasets` library, one utterance at a time:
1. Authenticate via `huggingface-cli login` (user needs to accept terms)
2. For each speaker in the subset, download 1 utterance
3. Run VBLINKF model to extract 256-dim embedding
4. Delete audio, keep embedding
5. Save accumulated embeddings as `(N, 256)` float32 `.npy`

### Fallback

If VoxCeleb1 or CN-Celeb are inaccessible, replace with freely available alternatives:

| Replaces | Fallback | Size | Access |
|----------|----------|------|--------|
| VoxCeleb1 | LibriSpeech dev-clean | ~100 speakers | Fully open on HF |
| CN-Celeb | AISHELL-1 | ~170 speakers | Fully open on HF |

## Architecture

### New File: `src/wespeaker_deep_edge/asnorm.py`

| Class / Function | Responsibility |
|-----------------|----------------|
| `build_cohort_embeddings(output_path, num_speakers=500)` | Download from HF streaming, extract embeddings, save `.npy` |
| `CohortCache` | Load/cache cohort embeddings in memory, precompute norms |
| `apply_asnorm(test_emb, enroll_matrix, cohort_embeds, top_k=300)` | Pure function: raw → AS-Norm calibrated score |

### `apply_asnorm()` Details

```
Input:
  test_emb:     (256,) float32 — test speaker embedding
  enroll_matrix: (7, 256) float32 — enrollment templates
  cohort_embeds: (500, 256) float32 — cohort embeddings

Step 1 — Test-side statistics:
  cohort_scores = cosine_similarity(test_emb, each cohort_embeds)
  top_k_idx = argsort(cohort_scores)[-top_k:]
  μ_test = mean(cohort_scores[top_k_idx])
  σ_test = std(cohort_scores[top_k_idx])

Step 2 — Enroll-side statistics (precomputed, cached):
  for each enroll template e_i:
    cohort_scores = cosine_similarity(e_i, each cohort_embeds)
    μ_enroll[i] = mean(top_k of cohort_scores)
    σ_enroll[i] = std(top_k of cohort_scores)

Step 3 — Normalize each enroll-template score:
  raw_i = cosine_similarity(test_emb, enroll_matrix[i])
  norm_i = 0.5 * (raw_i - μ_enroll[i]) / σ_enroll[i]
          + 0.5 * (raw_i - μ_test) / σ_test

Output:
  norm_scores: (7,) float32 — AS-Norm calibrated scores
  best_score: float32 — max of norm_scores
  best_name: str — name of highest-scoring enroll template
```

### Enroll-Side Precomputation

Enroll-side μ/σ only depends on enrollment templates + cohort — both are fixed after build. Precomputed once and cached:

```
Enrollment → Cohort similarity: 7 × 500 = 3,500 dot products
→ μ_enroll: (7,) array
→ σ_enroll: (7,) array
```

Stored alongside cohort embeddings.

### Modified File: `src/wespeaker_deep_edge/wespeaker_deep_dege.py`

**`DeepConfig` additions:**
```python
enable_asnorm: bool = False
asnorm_top_k: int = 300
asnorm_cohort_path: str = "asset/cohort/cohort_embeddings.npy"
```

**`WespeakerDeep` changes:**
- `__init__()`: conditionally precompute enroll-side cohort stats if `enable_asnorm`
- `_match_templates()`: swap raw scores for AS-Norm scores when enabled
- `load_templates()`: also compute enroll-side cohort stats when AS-Norm enabled

### File Layout

```
asset/cohort/
├── cohort_embeddings.npy       # (500, 256) float32
├── cohort_metadata.json        # speaker_ids, source info
└── enroll_cohort_stats.npy     # (2, 7) — μ_enroll[7], σ_enroll[7]
```

Total: < 1 MB. Can be git-tracked or packaged into whl.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Cohort file not found & AS-Norm enabled | Log warning, fall back to raw scores |
| HF streaming fails mid-download | Save progress, resume on retry |
| VoxCeleb1/CN-Celeb not accessible | Print clear instructions for fallback datasets |
| Empty enrollment matrix | Already handled by existing RuntimeError |

## Config & API

```python
# Enable AS-Norm
config = DeepConfig(enable_asnorm=True)
recognizer = WespeakerDeep(config=config)

# Build cohort (one-time)
recognizer.build_cohort(
    output_path="asset/cohort/cohort_embeddings.npy",
    num_voxceleb=300,
    num_cnceleb=200,
)

# Or load pre-built cohort
recognizer.load_cohort("asset/cohort/cohort_embeddings.npy")

# Recognize — transparently uses AS-Norm
result = recognizer.recognize_multi("test.wav")
```

CLI: `/asnorm` commands for cohort build/status.

## Testing

| Test | What it verifies |
|------|-----------------|
| `test_asnorm_pure_function` | `apply_asnorm()` with synthetic data produces z-scores centered at 0 |
| `test_cohort_cache` | Loading/saving `.npy` roundtrips correctly |
| `test_enroll_side_precompute` | Enroll μ/σ match manual calculation |
| `test_integration_recognize` | `recognize_multi()` with AS-Norm returns consistent results |
| `test_fallback_on_missing` | Missing cohort file degrades gracefully |

## Implementation Order

1. **Add `asnorm.py`** — `apply_asnorm()` pure function, `CohortCache` class
2. **Add cohort build script** — HF streaming download, embedding extraction
3. **Integrate into `WespeakerDeep`** — config, lifecycle, _match_templates hook
4. **Test with cross_test** — run existing cross_test with AS-Norm on/off, compare results
5. **Archive** — save final cohort embeddings to git

## Future Options (Not Implementing Now)

- Dynamic cohort selection (choose top-K nearest to test)
- Per-speaker threshold calibration
- AS-Norm in streaming/realtime mode
