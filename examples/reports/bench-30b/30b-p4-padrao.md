# Run report — `30b-p4-padrao`

- **Config**: `30b-p4-padrao`
- **Corpus**: `synthetic-forms-hard` @ `d45cda24b564` (60 cases)
- **Backend**: `ollama` · `qwen3-vl:30b-a3b-instruct` · digest `c871fc73fabc5516` · Q4_K_M
- **Prompt hash**: `2869e16f62bb`
- **Schema hash**: `acd6aaa82321`

## Per field

| field | acc | 95% CI | missing | hallucinated | wrong | malformed | refused |
|---|---:|:---:|---:|---:|---:|---:|---:|
| `patient_name` | 0.850 | 0.750–0.933 | 0.0% | 0.0% | 15.0% | 0.0% | 0.0% |
| `doctor_crm` * | 0.733 | 0.617–0.850 | 0.0% | 0.0% | 26.7% | 0.0% | 0.0% |
| `insurer` | 0.783 | 0.683–0.883 | 0.0% | 0.0% | 21.7% | 0.0% | 0.0% |
| `member_id` * | 0.783 | 0.667–0.883 | 0.0% | 1.7% ⚠️ | 20.0% | 0.0% | 0.0% |
| `exams` | 0.650 | 0.533–0.767 | 0.0% | 0.0% | 35.0% | 0.0% | 0.0% |
| `request_date` | 0.700 | 0.583–0.817 | 0.0% | 0.0% | 30.0% | 0.0% | 0.0% |
| `urgent` | 0.967 | 0.917–1.000 | 0.0% | 0.0% | 3.3% | 0.0% | 0.0% |

**MACRO** 0.781 (0.705–0.852)  
**ALL-FIELDS-CORRECT** 0.517 (0.400–0.650) — the metric that maps to *this document skips human review*

## By tag

| tag | acc | all-fields-correct | n |
|---|---:|---:|---:|
| `motion_blur` | 0.353 | 0.000 | 17 |
| `low_light` | 0.649 | 0.273 | 11 |
| `phone_photo` | 0.747 | 0.410 | 39 |
| `printed` | 0.776 | 0.571 | 21 |
| `handwritten` | 0.784 | 0.487 | 39 |
| `stamped` | 0.786 | 0.500 | 2 |
| `skewed` | 0.806 | 0.500 | 14 |
| `partial` | 0.829 | 0.600 | 5 |
| `scanned` | 0.844 | 0.714 | 21 |
| `rotated` | 0.857 | 0.000 | 1 |
| `glare` | 0.941 | 0.706 | 17 |
| `faded` | 1.000 | 1.000 | 2 |

## Operational

- latency p50 **11,566 ms**, p95 **11,916 ms**
- tokens in/out (mean) 1,802 / 92
- cost total $0.0000
- cases with no usable output: 0
- 60/60 served from cache; the latency above is from the original call
