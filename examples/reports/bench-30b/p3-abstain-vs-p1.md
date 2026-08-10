# Diff — `30b-p3-padrao` vs `30b-p1-padrao`

Corpus `synthetic-forms-hard` @ `d45cda24b564` (60 paired cases)

| field | baseline | candidate | Δ | Δ 95% CI | Δ hallucinated | Δ missing | sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| `patient_name` | 0.817 | 0.817 | +0.000 | +0.000, +0.000 | +0.0pp | +0.0pp | ~ |
| `doctor_crm` | 0.717 | 0.700 | -0.017 | -0.050, +0.000 | +0.0pp | +1.7pp | ~ |
| `insurer` | 0.717 | 0.733 | +0.017 | -0.033, +0.083 | +0.0pp | +11.7pp | ~ |
| `member_id` | 0.717 | 0.767 | +0.050 | -0.017, +0.117 | -6.7pp | +11.7pp | ~ |
| `exams` | 0.667 | 0.683 | +0.017 | +0.000, +0.050 | +0.0pp | +0.0pp | ~ |
| `request_date` | 0.700 | 0.717 | +0.017 | +0.000, +0.050 | +0.0pp | +0.0pp | ~ |
| `urgent` | 0.900 | 0.850 | -0.050 | -0.133, +0.017 | +0.0pp | +0.0pp | ~ |

**MACRO** 0.752 (+0.005)  
**ALL-FIELDS-CORRECT** 0.533 (-0.017)

| | baseline | candidate |
|---|---:|---:|
| hallucination rate | 0.95% | 0.00% |
| critical hallucination rate | 3.33% | 0.00% |
| latency p50 | 11,221 ms | 11,922 ms |

**Verdict:** quality change +0.005 is inside the noise (n=60) — not a result

