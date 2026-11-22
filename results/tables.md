### Wrong actions taken without asking (all 829 requests, thresholds cross-fitted)

| agent | condition | never confirm | confirm when any variant disagrees | utterance confidence, 20% budget | disagreement + argument confidence, 20% budget | difference [95% CI] | always confirm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | clean | 14.8% | **5.1%** (21.8% asked) | 10.0% (20.1%) | 6.4% (19.1%) | -3.6 [-5.4, -1.7] | 0% (77.6% asked) |
| Qwen2.5-7B | phone line | 15.6% | **5.8%** (20.1% asked) | 10.9% (21.2%) | 6.4% (18.7%) | -4.5 [-6.3, -2.5] | 0% (77.7% asked) |
| Qwen2.5-7B | phone + noise 10 dB | 16.6% | **5.8%** (20.7% asked) | 11.2% (20.3%) | 6.4% (19.8%) | -4.8 [-6.8, -3.0] | 0% (77.8% asked) |
| Qwen2.5-7B | phone + noise 0 dB | 20.7% | **6.0%** (25.3% asked) | 12.3% (20.4%) | 8.4% (20.4%) | -3.9 [-5.5, -2.1] | 0% (76.0% asked) |
| Qwen2.5-7B | phone + chatter 5 dB | 22.2% | **4.1%** (32.6% asked) | 13.1% (20.0%) | 9.7% (19.7%) | -3.5 [-5.5, -1.7] | 0% (76.2% asked) |
| Qwen2.5-7B | phone + chatter 0 dB | 32.0% | **2.2%** (40.7% asked) | 16.2% (20.0%) | 15.3% (20.4%) | -0.8 [-3.4, +1.8] | 0% (55.0% asked) |
| Llama-3.1-8B | clean | 26.8% | **12.2%** (27.9% asked) | 19.4% (19.8%) | 15.2% (20.1%) | -4.2 [-6.5, -1.8] | 0% (86.7% asked) |
| Llama-3.1-8B | phone line | 27.3% | **12.8%** (26.2% asked) | 19.2% (21.0%) | 15.9% (20.4%) | -3.3 [-5.5, -0.8] | 0% (86.9% asked) |
| Llama-3.1-8B | phone + noise 10 dB | 28.3% | **13.3%** (25.1% asked) | 19.5% (20.0%) | 16.3% (19.9%) | -3.3 [-5.5, -0.8] | 0% (86.7% asked) |
| Llama-3.1-8B | phone + noise 0 dB | 33.2% | **11.5%** (34.5% asked) | 22.3% (19.7%) | 19.2% (19.7%) | -3.1 [-5.2, -1.3] | 0% (86.6% asked) |
| Llama-3.1-8B | phone + chatter 5 dB | 35.0% | **9.9%** (39.9% asked) | 22.7% (20.9%) | 20.7% (19.9%) | -1.9 [-4.1, +0.1] | 0% (86.2% asked) |
| Llama-3.1-8B | phone + chatter 0 dB | 56.6% | **4.2%** (63.8% asked) | 39.2% (19.5%) | 37.9% (20.4%) | -1.3 [-4.0, +1.5] | 0% (80.1% asked) |

### Which wrong actions each policy catches (same number of confirmations for both policies)

| agent | condition | asked | mishearing-caused wrong actions | caught: any variant disagrees | caught: utterance confidence | agent-caused wrong actions | caught: any variant disagrees | caught: utterance confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | phone line | 20.1% | 9 | 9/9 | 3/9 | 120 | 72/120 | 36/120 |
| Qwen2.5-7B | phone + noise 10 dB | 20.7% | 17 | 16/17 | 9/17 | 121 | 74/121 | 36/121 |
| Qwen2.5-7B | phone + noise 0 dB | 25.3% | 55 | 50/55 | 41/55 | 117 | 72/117 | 43/117 |
| Qwen2.5-7B | phone + chatter 5 dB | 32.6% | 72 | 69/72 | 61/72 | 112 | 81/112 | 50/112 |
| Qwen2.5-7B | phone + chatter 0 dB | 40.7% | 184 | 175/184 | 163/184 | 81 | 72/81 | 65/81 |
| Llama-3.1-8B | phone line | 26.2% | 10 | 10/10 | 3/10 | 216 | 110/216 | 77/216 |
| Llama-3.1-8B | phone + noise 10 dB | 25.1% | 20 | 18/20 | 10/20 | 215 | 107/215 | 75/215 |
| Llama-3.1-8B | phone + noise 0 dB | 34.5% | 65 | 60/65 | 48/65 | 210 | 120/210 | 89/210 |
| Llama-3.1-8B | phone + chatter 5 dB | 39.9% | 85 | 81/85 | 70/85 | 205 | 127/205 | 96/205 |
| Llama-3.1-8B | phone + chatter 0 dB | 63.8% | 285 | 272/285 | 262/285 | 184 | 162/184 | 148/184 |

### Cost: extra agent calls per request vs wrong actions taken without asking (phone + noise 10 dB)

| agent | variants checked | extra agent calls | asked | wrong without asking |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | none | 0 | 0% | 16.6% |
| Qwen2.5-7B | second recogniser only | 1 | 5.8% | 12.1% |
| Qwen2.5-7B | 2nd-best beam alternative | 1 | 8.1% | 11.8% |
| Qwen2.5-7B | top-3 beam alternatives | 2 | 13.5% | 8.9% |
| Qwen2.5-7B | top-5 beam alternatives | 4 | 20.0% | 6.3% |
| Qwen2.5-7B | all 5-best + second recogniser | 5 | 20.7% | 5.8% |
| Llama-3.1-8B | none | 0 | 0% | 28.3% |
| Llama-3.1-8B | second recogniser only | 1 | 7.6% | 22.2% |
| Llama-3.1-8B | 2nd-best beam alternative | 1 | 8.6% | 22.2% |
| Llama-3.1-8B | top-3 beam alternatives | 2 | 16.8% | 16.9% |
| Llama-3.1-8B | top-5 beam alternatives | 4 | 24.6% | 13.5% |
| Llama-3.1-8B | all 5-best + second recogniser | 5 | 25.1% | 13.3% |

### Baseline: re-sample the agent on the same transcript (self-consistency, 4 samples at temperature 0.7)

| agent | condition | 4 agent samples disagree | beam alternatives 2-5 disagree (also 4 calls) | 20% budget: variants, then argument confidence | 20% budget: samples, then argument confidence | samples minus variants [95% CI] | variants or samples disagree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | clean | 10.1% (7.5% asked) | 5.4% (21.0%) | 6.4% (19.1%) | 7.4% (19.9%) | +1.0 [-0.6, +2.5] | 4.2% (23.5%) |
| Qwen2.5-7B | phone line | 10.4% (7.7% asked) | 6.2% (19.5%) | 6.4% (18.7%) | 7.6% (19.2%) | +1.2 [-0.5, +2.8] | 4.9% (22.0%) |
| Qwen2.5-7B | phone + noise 10 dB | 11.8% (8.2% asked) | 6.3% (20.0%) | 6.4% (19.8%) | 7.8% (20.9%) | +1.4 [-0.2, +3.0] | 4.9% (22.3%) |
| Qwen2.5-7B | phone + noise 0 dB | 15.3% (8.0% asked) | 7.4% (23.5%) | 8.4% (20.4%) | 9.0% (20.3%) | +0.6 [-1.2, +2.3] | 4.9% (27.3%) |
| Llama-3.1-8B | clean | 7.0% (35.2% asked) | 12.4% (27.1%) | 15.2% (20.1%) | 14.7% (20.1%) | -0.5 [-2.4, +1.3] | 3.9% (45.6%) |
| Llama-3.1-8B | phone line | 6.8% (35.7% asked) | 13.3% (25.2%) | 15.9% (20.4%) | 15.4% (19.4%) | -0.5 [-2.4, +1.3] | 4.3% (43.9%) |
| Llama-3.1-8B | phone + noise 10 dB | 8.1% (35.1% asked) | 13.5% (24.6%) | 16.3% (19.9%) | 15.6% (20.5%) | -0.7 [-2.8, +1.2] | 4.9% (42.9%) |
| Llama-3.1-8B | phone + noise 0 dB | 9.9% (36.9% asked) | 12.9% (32.4%) | 19.2% (19.7%) | 18.2% (21.1%) | -1.0 [-3.1, +1.1] | 4.3% (49.6%) |

### Accuracy of the agent's action

| agent | on what was said (clean transcript) | phone line | phone + noise 10 dB | phone + noise 0 dB | phone + chatter 5 dB | phone + chatter 0 dB |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 82.4% | 81.5% | 80.7% | 75.6% | 74.3% | 46.2% |
| Llama-3.1-8B | 73.2% | 72.7% | 71.7% | 66.6% | 64.5% | 36.8% |
