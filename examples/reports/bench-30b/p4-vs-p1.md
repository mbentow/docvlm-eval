# Diff — `30b-p4-padrao` vs `30b-p1-padrao`

Corpus `synthetic-forms-hard` @ `d45cda24b564` (60 paired cases)

| field | baseline | candidate | Δ | Δ 95% CI | Δ hallucinated | Δ missing | sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| `patient_name` | 0.817 | 0.850 | +0.033 | +0.000, +0.083 | +0.0pp | +0.0pp | ~ |
| `doctor_crm` | 0.717 | 0.733 | +0.017 | +0.000, +0.050 | +0.0pp | +0.0pp | ~ |
| `insurer` | 0.717 | 0.783 | +0.067 | +0.017, +0.133 | +0.0pp | -10.0pp | ▲ |
| `member_id` | 0.717 | 0.783 | +0.067 | +0.017, +0.133 | -5.0pp | -1.7pp | ▲ |
| `exams` | 0.667 | 0.650 | -0.017 | -0.083, +0.033 | +0.0pp | +0.0pp | ~ |
| `request_date` | 0.700 | 0.700 | +0.000 | +0.000, +0.000 | +0.0pp | +0.0pp | ~ |
| `urgent` | 0.900 | 0.967 | +0.067 | +0.000, +0.150 | +0.0pp | +0.0pp | ~ |

**MACRO** 0.781 (+0.033)  
**ALL-FIELDS-CORRECT** 0.517 (-0.033)

| | baseline | candidate |
|---|---:|---:|
| hallucination rate | 0.95% | 0.24% |
| critical hallucination rate | 3.33% | 0.83% |
| latency p50 | 11,221 ms | 11,566 ms |

**Verdict:** quality up +0.033 (significant)

