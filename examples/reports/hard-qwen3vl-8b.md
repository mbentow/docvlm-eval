# Run report — `hard-qwen3vl-8b`

- **Config**: `qwen3vl-8b`
- **Corpus**: `synthetic-forms-hard` @ `d45cda24b564` (60 cases)
- **Backend**: `ollama` · `qwen3-vl:8b-instruct` · digest `0533d74300e4f9bc` · Q4_K_M
- **Prompt hash**: `d59ccf92ae4c`
- **Schema hash**: `acd6aaa82321`

## Per field

| field | acc | 95% CI | missing | hallucinated | wrong | malformed | refused |
|---|---:|:---:|---:|---:|---:|---:|---:|
| `patient_name` | 0.833 | 0.733–0.917 | 0.0% | 0.0% | 16.7% | 0.0% | 0.0% |
| `doctor_crm` * | 0.767 | 0.667–0.867 | 3.3% | 0.0% | 20.0% | 0.0% | 0.0% |
| `insurer` | 0.733 | 0.617–0.833 | 20.0% | 0.0% | 6.7% | 0.0% | 0.0% |
| `member_id` * | 0.767 | 0.650–0.867 | 10.0% | 0.0% | 13.3% | 0.0% | 0.0% |
| `exams` | 0.700 | 0.583–0.817 | 3.3% | 0.0% | 26.7% | 0.0% | 0.0% |
| `request_date` | 0.700 | 0.583–0.817 | 3.3% | 0.0% | 26.7% | 0.0% | 0.0% |
| `urgent` | 0.967 | 0.917–1.000 | 0.0% | 0.0% | 3.3% | 0.0% | 0.0% |

**MACRO** 0.781 (0.702–0.855)  
**ALL-FIELDS-CORRECT** 0.583 (0.467–0.700) — the metric that maps to *this document skips human review*

## By tag

| tag | acc | all-fields-correct | n |
|---|---:|---:|---:|
| `motion_blur` | 0.361 | 0.000 | 17 |
| `low_light` | 0.610 | 0.182 | 11 |
| `stamped` | 0.643 | 0.500 | 2 |
| `phone_photo` | 0.758 | 0.538 | 39 |
| `handwritten` | 0.777 | 0.564 | 39 |
| `printed` | 0.789 | 0.619 | 21 |
| `skewed` | 0.816 | 0.571 | 14 |
| `scanned` | 0.823 | 0.667 | 21 |
| `partial` | 0.857 | 0.800 | 5 |
| `glare` | 0.941 | 0.824 | 17 |
| `faded` | 1.000 | 1.000 | 2 |
| `rotated` | 1.000 | 1.000 | 1 |

## Operational

- latency p50 **12,744 ms**, p95 **13,261 ms**
- tokens in/out (mean) 1,308 / 101
- cost total $0.0000
- cases with no usable output: 0
- 60/60 served from cache; the latency above is from the original call
