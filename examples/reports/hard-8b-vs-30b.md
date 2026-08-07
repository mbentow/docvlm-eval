# Diff — `qwen3vl-8b` vs `qwen3vl-30b`

Corpus `synthetic-forms-hard` @ `d45cda24b564` (60 paired cases)

| field | baseline | candidate | Δ | Δ 95% CI | Δ hallucinated | Δ missing | sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| `patient_name` | 0.800 | 0.833 | +0.033 | -0.033, +0.100 | +0.0pp | +0.0pp | ~ |
| `doctor_crm` | 0.750 | 0.767 | +0.017 | -0.050, +0.100 | +0.0pp | +3.3pp | ~ |
| `insurer` | 0.733 | 0.733 | +0.000 | -0.083, +0.083 | +0.0pp | +13.3pp | ~ |
| `member_id` | 0.717 | 0.767 | +0.050 | -0.033, +0.133 | -3.3pp | +10.0pp | ~ |
| `exams` | 0.633 | 0.700 | +0.067 | -0.033, +0.167 | +0.0pp | +3.3pp | ~ |
| `request_date` | 0.717 | 0.700 | -0.017 | -0.083, +0.033 | +0.0pp | +3.3pp | ~ |
| `urgent` | 0.917 | 0.967 | +0.050 | -0.033, +0.133 | +0.0pp | +0.0pp | ~ |

**MACRO** 0.781 (+0.029)  
**ALL-FIELDS-CORRECT** 0.583 (+0.050)

| | baseline | candidate |
|---|---:|---:|
| hallucination rate | 0.48% | 0.00% |
| critical hallucination rate | 1.67% | 0.00% |
| latency p50 | 8,950 ms | 12,744 ms |

**Verdict:** quality change +0.029 is inside the noise (n=60) — not a result; latency p50 1.42x (12,744ms) — check the queue SLA

