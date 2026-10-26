# VoxForge Stage A Editorial Certification Matrix

**10/10 niches certified** (Stage A threshold: score >= 88.0 and all six checks pass).

Generated offline via `tools/run_stage_a_matrix.py` — the editorial half of the pipeline
(intent parse → research dossier → showrunner script → audit/heal loop).

| Niche | Topic | Initial | Final | E1 | E2 | E3 | E4 | E5 | E6 | Certified |
|---|---|---|---|---|---|---|---|---|---|---|
| `01_science_education` | how volcanoes erupt and reshape the planet | 88.0 | 94.9 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `02_tech_ai` | the rise of artificial intelligence | 90.5 | 99.6 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `03_history` | the silk road trade networks of antiquity | 88.2 | 96.4 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `04_business_finance` | the history of money and banking | 89.7 | 98.0 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `05_health` | how the human heart works | 90.7 | 99.4 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `06_travel_geography` | the nile river and ancient egypt | 89.7 | 98.0 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `07_cooking` | the science of baking bread | 90.5 | 99.6 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `08_wildlife_nature` | the great whale migrations | 89.9 | 100.0 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `09_documentary_storytelling` | the lost libraries of the ancient world | 88.2 | 96.4 | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| `10_motivational` | stoic philosophy for modern life | 90.7 | 99.4 | PASS | PASS | PASS | PASS | PASS | PASS | YES |

## Lexical freshness detail (E2 · 6-gram overlap, spec <= 2.50%)

| Niche | Initial | Final | Measured (post) | Words / Target |
|---|---|---|---|---|
| `01_science_education` | 57.1 | 84.5 | 1.72% 6-gram overlap (2/116) | 145 / 157 |
| `02_tech_ai` | 70.2 | 100.0 | 0.00% 6-gram overlap (0/125) | 159 / 157 |
| `03_history` | 56.8 | 86.0 | 1.55% 6-gram overlap (2/129) | 163 / 157 |
| `04_business_finance` | 63.4 | 92.9 | 0.79% 6-gram overlap (1/127) | 161 / 157 |
| `05_health` | 70.5 | 100.0 | 0.00% 6-gram overlap (0/126) | 160 / 157 |
| `06_travel_geography` | 63.4 | 92.9 | 0.79% 6-gram overlap (1/127) | 161 / 157 |
| `07_cooking` | 70.2 | 100.0 | 0.00% 6-gram overlap (0/125) | 159 / 157 |
| `08_wildlife_nature` | 69.7 | 100.0 | 0.00% 6-gram overlap (0/123) | 157 / 157 |
| `09_documentary_storytelling` | 56.8 | 86.0 | 1.55% 6-gram overlap (2/129) | 163 / 157 |
| `10_motivational` | 70.5 | 100.0 | 0.00% 6-gram overlap (0/126) | 160 / 157 |
