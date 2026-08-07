# Run report — `hard-qwen3vl-30b`

- **Config**: `qwen3vl-30b`
- **Corpus**: `synthetic-forms-hard` @ `d45cda24b564` (60 cases)
- **Backend**: `ollama` · `qwen3-vl:30b-a3b-instruct` · digest `c871fc73fabc5516` · Q4_K_M
- **Prompt hash**: `d59ccf92ae4c`
- **Schema hash**: `acd6aaa82321`

## Per field

| field | acc | 95% CI | missing | hallucinated | wrong | malformed | refused |
|---|---:|:---:|---:|---:|---:|---:|---:|
| `patient_name` | 0.800 | 0.700–0.900 | 0.0% | 0.0% | 20.0% | 0.0% | 0.0% |
| `doctor_crm` * | 0.750 | 0.633–0.850 | 0.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| `insurer` | 0.733 | 0.617–0.850 | 6.7% | 0.0% | 20.0% | 0.0% | 0.0% |
| `member_id` * | 0.717 | 0.600–0.833 | 0.0% | 3.3% ⚠️ | 25.0% | 0.0% | 0.0% |
| `exams` | 0.633 | 0.517–0.750 | 0.0% | 0.0% | 36.7% | 0.0% | 0.0% |
| `request_date` | 0.717 | 0.600–0.833 | 0.0% | 0.0% | 26.7% | 1.7% | 0.0% |
| `urgent` | 0.917 | 0.850–0.983 | 0.0% | 0.0% | 8.3% | 0.0% | 0.0% |

**MACRO** 0.752 (0.664–0.833)  
**ALL-FIELDS-CORRECT** 0.533 (0.417–0.667) — the metric that maps to *this document skips human review*

## By tag

| tag | acc | all-fields-correct | n |
|---|---:|---:|---:|
| `motion_blur` | 0.269 | 0.000 | 17 |
| `low_light` | 0.610 | 0.364 | 11 |
| `rotated` | 0.714 | 0.000 | 1 |
| `phone_photo` | 0.722 | 0.487 | 39 |
| `handwritten` | 0.747 | 0.513 | 39 |
| `printed` | 0.762 | 0.571 | 21 |
| `partial` | 0.771 | 0.600 | 5 |
| `stamped` | 0.786 | 0.500 | 2 |
| `scanned` | 0.810 | 0.619 | 21 |
| `skewed` | 0.816 | 0.571 | 14 |
| `faded` | 0.929 | 0.500 | 2 |
| `glare` | 0.941 | 0.765 | 17 |

## Operational

- latency p50 **8,950 ms**, p95 **9,296 ms**
- tokens in/out (mean) 1,308 / 90
- cost total $0.0000
- cases with no usable output: 0
- 60/60 served from cache; the latency above is from the original call
