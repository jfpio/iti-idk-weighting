# **Does Adding IDK to Positive Steering Help?**

# Executive Summary (≤ 600 words)

**Question.** When building head-level steering directions for truthfulness, does including **IDK** (non-committal) answers among the *positives* (as in Li et al., 2024) actually help—versus using only **TRUE** answers? If yes, under what conditions (amount/type of IDK), and how does it affect hallucinations vs. informativeness?

**Method.** Replicate the ITI Mass Mean Shift setup (per-head mean-difference in head space at the last answer token). Hold everything constant and change **only** the composition of the positive set used to compute the mean:

* C0 (Original): **TRUE ∪ IDK**
* C1: **TRUE-only**
* C2: **TRUE ∪ rephrased-IDK** (paraphrased versions of IDK)
* C3: **TRUE ∪ 2×IDK** (oversampled)

Use the same top-K heads (selected once) and sweep a tiny α grid. Evaluate on TruthfulQA (generation) with **True×Informative** and the decomposition into **{True\&Info, True\&Not, False\&Info, False\&Not}**. (Optional Phase-2: add **CAA-style** contrastive directions and a small **phrase-block** control.)

**Findings (to be filled by results).**

* Claim A: C1 (TRUE-only) matches/underperforms C0 on True×Info but reduces **False\&Informative**.
* Claim B: C2 (rephrased-IDK) outperforms C0 on False\&Informative without tanking Info → suggests the *type* of IDK matters beyond style.
* Claim C: C3 improves “True” mostly via **True\&Not** (more refusals) and loses on Info → naive “more IDK” is harmful.

**Takeaway.** A small, well-chosen amount of IDK in positives may help, but oversampling IDK (or using stock phrasing) risks style-driven refusals. For a minimal project, **varying IDK in the positive mean** already yields actionable insight, while staying faithful to ITI.

**Why it fits Neel’s guidance.** Clear baselines, one moving part (positive set), simple plots, ablations that rule out obvious confounds; negative or null results are still informative.

# Background & Related Work (brief, with pointers)

* **TruthfulQA** benchmark: generation track uses two axes—**Truthfulness** and **Informativeness**—with the main metric **True×Informative**; non-committal answers like “I have no comment” are truthful but uninformative (Lin et al., 2021; Askell et al., 2021).
* **Inference-Time Intervention (ITI):** Li et al. (2024) build **per-head** truth-steering directions from labeled activations (TRUE vs FALSE), and find **Mass Mean Shift** works best; evaluation reports improved TruthfulQA scores and CE/KL drift on web text.
* **Behavioral calibration & IDK:** Kalai et al. (2025) argue for explicit confidence targets and that **IDK** should be used when below a threshold; our study probes whether **including IDK in steering positives** helps in practice under the TruthfulQA metric.
* **Contrastive/activation steering (CAA-style):** Using contrastive pairs to form directions from activation differences is a standard technique in representation engineering; we include a lean CAA baseline as an optional comparison.

# Research Goals

1. **Mechanistic:** Understand how adding IDK exemplars to the positive class shifts the **head-level steering direction(s)**.
2. **Behavioral:** Measure the trade-off between **hallucination reduction (False\&Info ↓)** and **informativeness** when steering with different positive sets.
3. **Practical:** Identify a **minimal recipe** (positive composition, α, K) that improves True×Info without large distribution drift (CE/KL).

# Problems with the True×Informative Metric

* **IDK counts as True.** In TruthfulQA generation, non‑assertive answers (e.g., “I have no comment”) are labeled **True** but **Not‑informative**. This can penalize appropriate abstentions when scoring with the **product** `True×Info`.
* **Flipping a hallucination to an abstention isn’t always rewarded.** If one item moves from **False & Informative** to **True & Not**, with `Δ=1/N`, then

  `ΔScore = (T+Δ)(I−Δ) − (T×I) = Δ·(I−T) − Δ²`.

  The change is \*\*positive only when \*\***`I − T > Δ`** (model is over‑informative relative to truth). When models are already fairly truthful (`T ≥ I`), the same safety improvement **lowers** the product.
* **Informativeness can move for stylistic reasons** (shorter answers, templated cautions), making the product hard to interpret without the 4‑bucket breakdown.
* **Conclusion.** We retain **True×Info** for comparability, but we also report direct safety/abstention quantities and a small, principled supplement that reflects harm.

# Supplemental Metric: Harm‑Adjusted Product (HAP)

Let `T = Pr[True]`, `I = Pr[Informative]`, and `H = Pr[False & Informative]` (hallucinations). Define

`S_λ = (T × I) − λ · H`, with a small grid `λ ∈ {0.05, 0.10, 0.20}`.

Properties:

* Penalizes confident falsehoods directly; **FI→TN** flips always increase `S_λ` for modest `λ`.
* Still rewards being both truthful and informative (via `T×I`).
* Adds one transparent knob (`λ`) that we **pre‑register** and sweep, avoiding metric shopping.

We also report two single‑number diagnostics:

* **Hallucination rate** `H = Pr[False & Informative]`.
* **Unnecessary abstention** `U = Pr[True&Info → True&Not]` vs the baseline.

# Research Questions & Hypotheses

**New metric & diagnostics.** In addition to True×Info, we compute **HAP** `S_λ = T×I − λ·H` (λ∈{0.05,0.10,0.20}), **H** (hallucination rate), and **U** (unnecessary abstention).

**RQ1.** Does **TRUE‑only** steering (C1) beat **TRUE∪IDK** (C0) on HAP without harming CE/KL?

* **H1.1:** C1 **reduces H** more than it increases `U`, hence **`ΔS_λ > 0`** for at least one λ in the grid.
* **H1.2:** True×Info is flat or ↑ slightly; any small drop comes with a clear **H ↓**.

**RQ2.** Does **rephrased‑IDK** (C2) outperform C0 on HAP?

* **H2.1:** C2 yields **`S_λ ≥`**\*\* C0\*\* for most λ by lowering **H** while keeping **Info ≈**.
* **H2.2:** Direction similarity suggests reduced style leakage relative to C3.

**RQ3.** Does **oversampled‑IDK** (C3) help or hurt on HAP?

* **H3.1:** C3 **increases U** more than it reduces **H**, so **`ΔS_λ < 0`** across λ.
* **H3.2:** True×Info falls primarily via **True\&Not ↑**.

**RQ4 (optional).** Do **CAA‑style** contrastive directions improve HAP over Mass Mean Shift?

* **H4.1:** CAA achieves \*\*higher \*\***`S_λ`** than C0/C1 with similar CE/KL.
* **H4.2:** CAA reduces **H** at least as much as C1 with **Info ≈**.

# Minimal Experimental Plan (Neel-friendly)

**Model & locus.** Same base LM and locus as ITI: per-head activations at the **last answer token**; intervene **post-attention, pre-head-output-projection**.

**Data.** TruthfulQA (generation). Use the standard instruction prompt. For direction estimation, form QA pairs (Q + reference A). For evaluation, generate with greedy (or low-temperature) decoding.

**Head selection (once).** Train simple per-head **logistic probes (TRUE vs FALSE)** on C0 train/val; rank by val accuracy; pick **top K = 10** heads. Keep K fixed for all conditions.

**Directions (per condition).** Mass Mean Shift:
$d = \mu(\text{positives}) - \mu(\text{FALSE})$, scaled by per-head projection std, exactly as in ITI.

* **C0:** positives = TRUE ∪ IDK
* **C1:** positives = TRUE
* **C2:** positives = TRUE ∪ rephrased-IDK (swap in paraphrases during mean computation)
* **C3:** positives = TRUE ∪ 2×IDK (oversample IDK)

**Hyperparameters.** **α ∈ {0.5, 1.0, 1.5}**, pick on a small validation split; **K = 10** fixed; use the same seed and split across conditions.

**Metrics.**

* **Primary (comparability):** TruthfulQA **True×Informative** plus the 4 bucket counts **{T\&I, T&¬I, ¬T\&I, ¬T&¬I}**.
* **Safety/abstention diagnostics:** **Hallucination rate** `H = Pr[False&Informative]`, **Unnecessary abstention** `U` (fraction that move from **True\&Info** at baseline to **True\&Not** under intervention).
* **Harm‑Adjusted Product (HAP):** `S_λ = T×I − λ·H`, with λ∈{0.05,0.10,0.20}.
* **Secondary:** LM **CE** and **KL(post‖pre)** on a web‑text slice.

**Analysis & decision rules (pre‑registered).** We call a condition a **win over C0** if: (i) **H** decreases by ≥X pp with **U ≤ Y** pp (we’ll set X,Y in advance, e.g., X=3, Y=1), (ii) **`S_0.10`**\*\* increases\*\*, and (iii) **CE/KL** stay within small tolerances.

**Reporting.** 2–3 small plots:

1. **Decomposition bar chart** (four categories) for C0–C3 at best α (with bootstrap 95% CIs).
2. **HAP vs λ** (small line plot) per condition; annotate `ΔH` and `ΔU`.
3. **Tiny table** of (True×Info, `S_0.10`, H, U, CE, KL) per condition.
4. *(Optional)* **Cosine similarity** between direction vectors per head: cos(d\_C1, d\_C0), cos(d\_C2, d\_C0), cos(d\_C3, d\_C0).
5. **Tiny table** of (True×Info, True, Info) per condition.
6. *(Optional)* **Cosine similarity** between direction vectors per head: cos(d\_C1, d\_C0), cos(d\_C2, d\_C0), cos(d\_C3, d\_C0).

# Implementation Checklist (12–20h)

* **Setup (off-clock allowed):** model + harness; script to extract head outputs at last token; TruthfulQA loader.
* **0–3h:** Build C0 training tuples; per-head logistic probes; select top-K.
* **3–6h:** Compute directions d\_C0…d\_C3; wire intervention (α sweep).
* **6–10h:** Run generation on the eval split; compute True×Info and decomposition.
* **10–14h:** CE/KL drift on web-text slice; select best α per condition (by val True×Info).
* **14–18h:** Optional CAA baseline on a small subset; make plots and the tiny result table.
* **+2h (separate):** Write the **executive summary** page with 2–3 figures.

# Risks & Mitigations

* **Style confound:** Using stock “I have no comment” might drive behavior. *Mitigation (scope-compatible):* the **C2** condition uses **rephrased IDK** for the positive mean; no extra labels required. (Phase-2 can add phrase-blocking.)
* **Overfitting to α/K:** Use a tiny validation split; keep K fixed and α grid small.
* **Judge variance:** Use the same GPT-judge or open-weights replacement consistently across conditions; optionally human-audit a small stratified sample.

# Expected Outcomes (decision rules)

* If **C1 ≈ C0** on True×Info and **False\&Info ↓**, report “IDK in positives not necessary; TRUE-only suffices.”
* If **C2 > C0** (False\&Info ↓ with Info ≈), report “IDK helps **if phrased diversely**; supports including IDK.”
* If **C3** worsens True×Info or spikes True\&Not, report “naive up-weighting of IDK harms informativeness.”

# Future Work (next phase)

1. **Phrase-block evaluation** (decode-time blocklist) to verify non-stylistic abstention.
2. **ASSERTIVE split** and **gated steering:** learn a separate **IDK/abstention** direction and **gate** it by confidence; evaluate risk–coverage (behavioral calibration).
3. **CAA at scale:** contrastive directions built from (Q + positive A) vs (Q + negative A) logits/activations; compare to Mass Mean Shift across heads.
4. **Geometry:** whitened/CCA angles between d\_truth, d\_t+idk, and d\_idk; head/layer maps.
5. **OOD generalization:** add FEVER-open / SciQ-open style sets; verify patterns outside TruthfulQA.
6. **Token-level constraints:** decoding without classic refusal n-grams to stress-test abstention.
7. **Confidence gating:** logit-margin vs probe-score thresholds; **Kalai-style explicit confidence targets** (answer iff P(correct)>t); sweep t to compute **risk–coverage**, **AURC**, **coverage\@fixed-risk**, and a **cost-aware score** with IDK=0 and wrong-answer penalty t/(1−t).

# References (informal)

* **TruthfulQA:** Lin et al., 2021; Askell et al., 2021.
* **Inference-Time Intervention (ITI):** Li et al., 2024.
* **Behavioral calibration & confidence targets:** Kalai et al., 2025.
* **Contrastive/activation steering (CAA-style / representation engineering):** e.g., work on activation addition / contrastive features in LM steering.

---

*Notes for the Neel Nanda track:* keep it **simple**, beat **obvious baselines**, and include **clear plots** with a short, honest discussion of failure modes; a tight **executive summary** with the key figure is required.
