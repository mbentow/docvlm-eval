# Diff — `30b-p1-alta` vs `30b-p1-padrao`

Corpus `synthetic-forms-hard` @ `d45cda24b564` (60 paired cases)

| field | baseline | candidate | Δ | Δ 95% CI | Δ hallucinated | Δ missing | sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| `patient_name` | 0.817 | 0.767 | -0.050 | -0.133, +0.017 | +0.0pp | +0.0pp | ~ |
| `doctor_crm` | 0.717 | 0.700 | -0.017 | -0.067, +0.033 | +0.0pp | +0.0pp | ~ |
| `insurer` | 0.717 | 0.733 | +0.017 | -0.050, +0.100 | +0.0pp | -8.3pp | ~ |
| `member_id` | 0.717 | 0.683 | -0.033 | -0.100, +0.033 | +1.7pp | +0.0pp | ~ |
| `exams` | 0.667 | 0.667 | +0.000 | -0.100, +0.083 | +0.0pp | +0.0pp | ~ |
| `request_date` | 0.700 | 0.683 | -0.017 | -0.067, +0.033 | +0.0pp | +0.0pp | ~ |
| `urgent` | 0.900 | 0.850 | -0.050 | -0.117, +0.000 | +0.0pp | +0.0pp | ~ |

**MACRO** 0.726 (-0.021)  
**ALL-FIELDS-CORRECT** 0.533 (-0.017)

| | baseline | candidate |
|---|---:|---:|
| hallucination rate | 0.95% | 1.19% |
| critical hallucination rate | 3.33% | 4.17% |
| latency p50 | 11,221 ms | 22,924 ms |

**Verdict:** quality change -0.021 is inside the noise (n=60) — not a result; latency p50 2.04x (22,924ms) — check the queue SLA

