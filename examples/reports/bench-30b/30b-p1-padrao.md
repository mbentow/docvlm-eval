# Run report — `30b-p1-padrao`

- **Config**: `30b-p1-padrao`
- **Corpus**: `synthetic-forms-hard` @ `d45cda24b564` (60 cases)
- **Backend**: `ollama` · `qwen3-vl:30b-a3b-instruct` · digest `c871fc73fabc5516` · Q4_K_M
- **Prompt hash**: `d59ccf92ae4c`
- **Schema hash**: `acd6aaa82321`

## Per field

| field | acc | 95% CI | missing | hallucinated | wrong | malformed | refused |
|---|---:|:---:|---:|---:|---:|---:|---:|
| `patient_name` | 0.817 | 0.717–0.917 | 0.0% | 0.0% | 18.3% | 0.0% | 0.0% |
| `doctor_crm` * | 0.717 | 0.600–0.833 | 0.0% | 0.0% | 28.3% | 0.0% | 0.0% |
| `insurer` | 0.717 | 0.600–0.817 | 10.0% | 0.0% | 18.3% | 0.0% | 0.0% |
| `member_id` * | 0.717 | 0.600–0.817 | 1.7% | 6.7% ⚠️ | 20.0% | 0.0% | 0.0% |
| `exams` | 0.667 | 0.550–0.783 | 0.0% | 0.0% | 33.3% | 0.0% | 0.0% |
| `request_date` | 0.700 | 0.583–0.817 | 0.0% | 0.0% | 30.0% | 0.0% | 0.0% |
| `urgent` | 0.900 | 0.817–0.967 | 0.0% | 0.0% | 10.0% | 0.0% | 0.0% |

**MACRO** 0.748 (0.657–0.831)  
**ALL-FIELDS-CORRECT** 0.550 (0.433–0.683) — the metric that maps to *this document skips human review*

## By tag

| tag | acc | all-fields-correct | n |
|---|---:|---:|---:|
| `motion_blur` | 0.252 | 0.000 | 17 |
| `low_light` | 0.584 | 0.364 | 11 |
| `stamped` | 0.643 | 0.500 | 2 |
| `phone_photo` | 0.711 | 0.462 | 39 |
| `printed` | 0.735 | 0.571 | 21 |
| `handwritten` | 0.755 | 0.538 | 39 |
| `skewed` | 0.806 | 0.643 | 14 |
| `scanned` | 0.816 | 0.714 | 21 |
| `partial` | 0.829 | 0.800 | 5 |
| `rotated` | 0.857 | 0.000 | 1 |
| `glare` | 0.941 | 0.824 | 17 |
| `faded` | 1.000 | 1.000 | 2 |

## Operational

- latency p50 **11,221 ms**, p95 **11,613 ms**
- tokens in/out (mean) 1,696 / 90
- cost total $0.0000
- cases with no usable output: 0
- 60/60 served from cache; the latency above is from the original call
