# 📊 Presentation Guide — Student Performance Risk Monitoring System

Covers: 10-minute outline (17), slide-by-slide PowerPoint content (18), video
demo script (19), and the team responsibility template (21).

---

## 17. 10-minute presentation outline
| Segment | Time | Slides | Speaker focus |
|---|---|---|---|
| **Intro & Problem Statement** | 2 min | 1–3 | Who we are, the advisor problem, the dataset & target |
| **Dataflow** | 3 min | 4–7 | EDA findings → metric → AutoML training → evaluation |
| **Architecture** | 3 min | 8–11 | Pipeline diagram, deployment, monitoring dashboard, the data-change experiment |
| **Tools & Pillars** | 2 min | 12–13 | Stack + which MLOps pillar each tool covers; results & GitHub |
| **Q&A** | 1 min | 14 | Questions |

> Embed the **video demo** (steps 7 & 9) inside the Architecture segment, on
> slides 10–11.

---

## 18. Slide-by-slide content
**Slide 1 — Title**
- Student Performance Risk Monitoring System: Predicting & Monitoring Academic Performance with MLOps
- Team member names + course / date.

**Slide 2 — Problem statement (stakeholder language)**
- Advisors can't track every student's workload, sleep, stress, attendance.
- At-risk students are spotted *after* grades drop.
- Goal: predict performance early + monitor that the model stays reliable.

**Slide 3 — Dataset & target**
- 1,000 students, no missing values; target = `Average_Score` (0–100).
- 14 features: behaviour (study, sleep, stress, attendance), demographics, background.
- Key decision: exclude subject scores + Grade (target leakage) so the model
  learns from factors advisors can influence.
- At-Risk = predicted score < 60.

**Slide 4 — EDA findings** *(figures 01–06 in `reports/figures/`)*
- Performance roughly normal, mean ≈ 71; ~12.5% below the at-risk line.
- Strongest driver: **Study Hours (+0.70)**; strongest penalty: **Stress (−0.51)**.
- Part-time job lowers study hours → built-in trade-off in the data.

**Slide 5 — Evaluation metric**
- Primary: **RMSE** (penalises big misses, score-point units).
- Support: **MAE** (avg error in points), **R²** (variance explained).
- Why RMSE: a 30-point miss on an at-risk student is far worse than several 2-point misses.

**Slide 6 — Training pipeline (AutoML + MLflow)**
- FLAML searches LightGBM / XGBoost / Random Forest / Extra Trees, tunes hyperparameters.
- MLflow logs params, metrics, the full pipeline artifact; registers the best model.
- Preprocessing baked into the pipeline → no training/serving skew.

**Slide 7 — AutoML result & evaluation** *(figure 07)*
- Selected algorithm: **XGBoost**.
- Original test set: **RMSE ≈ 3.95, MAE ≈ 2.85, R² ≈ 0.86**.
- Predicted-vs-actual plot hugs the diagonal.

**Slide 8 — Architecture / dataflow diagram**
```
raw CSV ─► EDA ─► preprocess+split ─► FLAML AutoML ─► MLflow (track+registry)
                                                      │
                                              best_model.pkl
                                                      │
              ┌───────────────────────────┬──────────┴───────────┐
        FastAPI API                Streamlit dashboard      batch_inference
        (REST serving)          (inference + monitoring)   (orig vs modified)
                                          │
                                   PSI drift monitor ◄── test vs test_modified
```

**Slide 9 — Deployment**
- FastAPI REST endpoint `/predict` (Swagger UI) for programmatic scoring.
- Streamlit app for advisors: move sliders → instant score + risk flag.

**Slide 10 — Monitoring dashboard** *(VIDEO 1 — step 7)*
- Live PSI drift table (green/amber/red), prediction-distribution overlay, metric deltas.
- Explain PSI: <0.10 none · 0.10–0.25 moderate · >0.25 major.

**Slide 11 — The data-change experiment** *(VIDEO 2 — step 9)*
- "Academic stress" scenario: Study ×0.5, Stress +3, Sleep −1.5, Attendance −15 (4 features).
- Result table (original → modified): mean score ~71→~53, At-Risk ~14%→~70%,
  RMSE ~3.9→~19.6, prediction PSI ≈ 3.97 (major drift).
- Takeaway: degraded behaviour → lower predictions → monitor catches it early.

**Slide 12 — Tools & MLOps pillars**
| Pillar | Tool |
|---|---|
| EDA / visualisation | pandas, matplotlib, plotly |
| Reproducible split | scikit-learn |
| Experiment tracking & registry | MLflow |
| AutoML | FLAML |
| Deployment / serving | FastAPI + Streamlit |
| Monitoring / drift | PSI monitor (+ optional Evidently) |
| Orchestration | `run_all.sh` |

**Slide 13 — Results & GitHub**
- Headline metrics table; "the modified data produced an observable, monitored change."
- **GitHub link** + the team responsibility table.

**Slide 14 — Q&A / Thank you**

---

## 19. Video demo script (≈ 2–3 minutes, screen recording)
Record this and embed it on slides 10–11. Have the pipeline already run
(`bash run_all.sh`) and the dashboard open.

**Part A — Monitoring dashboard (covers step 7)**
1. *(Predict tab)* "This is our deployed model in Streamlit. I'll set a strong
   student — high study hours, low stress — predicted score ~86, **Not at risk**."
2. "Now a struggling student — low study, high stress — score drops to ~25,
   flagged **At Risk**. The same registered model is serving both."
3. *(Monitoring tab)* "This is our monitoring dashboard. On the original test
   data the model is healthy: RMSE ~3.9, R² ~0.86, ~14% at risk, and all PSI
   values are green — no drift."

**Part B — Change the test data & re-validate (covers step 9)**
4. *(terminal)* "Now I simulate an academic-stress semester. `make_modified_test.py`
   changes four behavioural features: study halved, stress up, sleep down,
   attendance down." → run `python -m src.make_modified_test`.
5. "I re-run inference and monitoring on the changed data." → run
   `python -m src.batch_inference` then `python -m src.monitoring`.
6. *(back to dashboard, refresh)* "The monitoring dashboard now tells the story:
   mean predicted score fell ~18 points, at-risk jumped from ~14% to ~70%, RMSE
   blew up to ~19, and the PSI table turns red — **major drift** detected,
   prediction PSI ~3.97."
7. "So the system not only predicts performance, it **detects** when incoming
   student behaviour has shifted enough to make the model unreliable — exactly
   what monitoring is for."

---

## 21. Team responsibility breakdown (template — fill in names)
| Member | Primary responsibility | Files / slides owned |
|---|---|---|
| **Member 1** | Data & EDA | `src/eda.py`, `src/preprocessing.py`, `src/config.py`; slides 3–4 |
| **Member 2** | Modeling & AutoML | `src/train_automl.py`, `src/evaluate.py`, MLflow setup; slides 5–7 |
| **Member 3** | Deployment | `app/inference_api.py`, `app/dashboard.py` (Predict tab); slides 8–9 |
| **Member 4** | Monitoring & data-change experiment | `src/monitoring.py`, `src/make_modified_test.py`, `src/batch_inference.py`, dashboard Monitoring tab; slides 10–11 |
| **All** | Presentation, video demo, GitHub README | `README.md`, `PRESENTATION.md`, recording; slides 1, 12–14 |

> For a smaller team, merge rows (e.g., 2 people: Member 1 takes Data+Modeling,
> Member 2 takes Deployment+Monitoring; both do the presentation).
