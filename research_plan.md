# **Does Adding IDK to Positive Steering Help?**

## Executive Summary (≤ 600 words)

## What problem am I trying to solve?
**Question.** When building head-level steering directions for truthfulness, does including **I don't know (IDK)** answers among the *positives* (as in Li et al., 2024) actually help—versus using only **TRUE** answers? If yes, under what **dose** (weight) of IDK, and how does it affect **hallucinations** vs **informativeness**?

**Method.** Replicate ITI’s Mass Mean Shift (per-head mean-difference at the last answer token). Hold everything constant and vary **only** the positive-class composition by assigning a **weight β** to IDK examples when computing the positive mean:

- β = 0 → TRUE-only (numerically use β=1e-6 if the code requires non-empty weight).
- β = 0.5 → “light IDK”.
- β = 1 → original (TRUE ∪ IDK).
- β = 2 → oversampled IDK without duplicating rows.

Use one fixed set of top-K heads (selected once at β=1) and a tiny α grid. Evaluate on TruthfulQA (generation) with **True×Informative**, the **4-bucket decomposition** {True&Info, True&Not, False&Info, False&Not}, plus new safety diagnostics.

**Findings (to be filled by results).**
- *Claim A:* β≈0–0.5 reduces **False&Informative** (hallucinations) with small or no increase in unnecessary abstention; **HAP(0.10)** rises; CE/KL drift small.
- *Claim B:* Using **rephrased** IDK (instead of stock phrases) in the positive mean yields better trade-offs than stock IDK.
- *Claim C:* β=2 increases True&Not (refusals) and harms True×Info; HAP falls—naively adding IDK is harmful.

**Takeaway.** A *modest* IDK weight in the positive mean can improve safety with limited cost; large IDK weight encourages unhelpful refusals. The β **dose–response** makes the trade-off explicit and reproducible.

**Why this fits Neel’s guidance.** One moving part (β), clear baselines, compact plots, strong sanity checks, and honest failure analysis.

---

## Background & Related Work (brief pointers)

- **TruthfulQA**: generation track labels **Truthfulness** and **Informativeness**; the main metric is **True×Informative**; non-assertive answers (e.g., “I have no comment”) are typically **True & Not-informative** (Lin et al., 2021; Askell et al., 2021).
- **Inference-Time Intervention (ITI)**: Li et al. (2024) form per-head directions; **Mass Mean Shift** outperforms probe-weight and CCS; they report TruthfulQA gains and CE/KL drift on web text.
- **Behavioral calibration**: Kalai et al. (2025) advocate explicit confidence targets and appropriate **IDK**; our study investigates whether including IDK in the **steering positives** helps under TruthfulQA-style evaluation.

---

## Research Goals

1. **Mechanistic**: Characterize how IDK weight β changes the learned **head-space direction(s)**.
2. **Behavioral**: Quantify the trade-off between **hallucination reduction** and **informativeness** across β.
3. **Practical**: Identify a **minimal recipe** (β, α, K) that improves safety (and, ideally, True×Info) with small CE/KL drift.

---

## Problems with the True×Informative Metric

- **IDK counts as True.** Non-assertive answers are labeled **True** but **Not-informative**, so the product can **penalize appropriate abstention**.
- **FI→TN isn’t always rewarded.** For one item flip from **False&Info** to **True&Not**, with Δ=1/N:
  
  \[
  \Delta (T\!\times\!I)=(T+\Delta)(I-\Delta)-TI=\Delta\,(I-T)-\Delta^2
  \]
  
  The change is **positive only if \(I - T > \Delta\)** (over-informative regime). When \(T\!\ge\!I\), this safety improvement **lowers** the metric.
- **Style confounds** (shorter/templated answers) can move “Info” independent of truth.

**Conclusion.** Keep **True×Info** for comparability, but add **direct safety diagnostics** and a small, principled supplement that reflects harm.

---

## Supplemental Metric: Harm-Adjusted Product (HAP)

Let \(T=\Pr[\text{True}],\, I=\Pr[\text{Informative}],\, H=\Pr[\text{False \& Informative}]\).
Define a harm-aware score:

\[
\boxed{S_\lambda = (T \times I) - \lambda \, H}, \quad \lambda \in \{0.05, 0.10, 0.20\}.
\]

- Explicitly penalizes confident falsehoods; **FI→TN** flips **increase** \(S_\lambda\) for modest \(\lambda\).
- Still rewards joint truth + informativeness via \(T\times I\).
- We **pre-register** the λ grid to avoid metric shopping.

**Diagnostics to always report**
- **Hallucination rate** \(H = \Pr[\text{False \& Informative}]\).
- **Unnecessary abstention** \(U = \Pr[\text{items that are True\&Info at β=1 become True\&Not at β}] \).

---

## Research Questions & Hypotheses (with HAP)

- **RQ1.** Does **TRUE-only** (β≈0) beat **β=1** on HAP without harming CE/KL?  
  **H1.1**: \(H\) drops more than \(U\) rises ⇒ \(S_\lambda\) ↑ for ≥1 λ.  
  **H1.2**: True×Info flat/↑; any small drop coincides with clear \(H\) ↓.

- **RQ2.** Does **rephrased-IDK** (β=1, paraphrased) outperform stock β=1 on HAP?  
  **H2.1**: \(H\) ↓ with Info ≈ ⇒ \(S_\lambda\) ≥ baseline for most λ.

- **RQ3.** Does **oversampled IDK** (β=2) help or hurt?  
  **H3.1**: \(U\) rises more than \(H\) falls ⇒ \(S_\lambda\) ↓ across λ; True×Info falls due to **True&Not** ↑.

- **RQ4 (optional).** Do **CAA-style** contrastive directions improve HAP vs Mass Mean Shift?  
  **H4.1**: \(S_\lambda\) ↑ with similar CE/KL; \(H\) ↓ at least as much as β≈0.

---

## Minimal Experimental Plan (Neel-friendly)

### Model & locus
Same as ITI: per-head outputs at the **last answer token**; intervene **post-attention, pre head-output-projection**.

### Data
TruthfulQA (generation). Use the standard instruction prompt. For direction estimation, use (Q + reference A). For evaluation, generate deterministically (e.g., greedy/low-T).

### Head selection (once)
Train per-head **logistic probes** (TRUE vs FALSE) on β=1 (train/val). Rank by validation accuracy; pick **top-K=10**. Keep K **fixed** for all β.

### β-weighted Mass Mean Shift
Positive mean per head in head-space at last token:

\[
\mu_+(\beta)=\frac{\sum_i w_i(\beta)\,x_i}{\sum_i w_i(\beta)},\quad
w_i(\beta)=\begin{cases}
1 & \text{if TRUE}\\
\beta & \text{if IDK}
\end{cases}
\]
\[
d_\beta = \mu_+(\beta) - \mu_- \quad\text{(negatives = FALSE)}
\]

Scale by per-head projection std estimated with the same weights (or reuse ITI’s σ if preferred).

**β grid**: {0, 0.5, 1, 2}. If the repo disallows β=0, use **β=1e-6**.

**Rephrased-IDK variant (C2)**: For direction-estimation only, replace stock IDK strings with paraphrases; β still applies.

### Hyperparameters
β ∈ {0, 0.5, 1, 2}; **α ∈ {0.5, 1.0, 1.5}** (picked on a small validation split); **K=10** fixed; same seed/split across β.

### Cross-validation & α selection
Use `validate2fold`: compute \(d_\beta\) on the **train fold**; evaluate on the **held-out fold**.  
Select α per β to **maximize HAP(0.10)** on validation, then **freeze** α for test.  
Bootstrap per question (e.g., 10k resamples) to obtain **95% CIs**.

---

## Metrics

**Primary (comparability):** TruthfulQA **True×Informative** + the 4 buckets {T&I, T&¬I, ¬T&I, ¬T&¬I}.  
**Safety/abstention diagnostics:** **H** (False&Info), **U** (T&Info@β=1 → T&Not@β).  
**Harm-Adjusted Product:** \(S_\lambda=T\times I - \lambda H\) for λ∈{0.05, 0.10, 0.20}.  
**Secondary:** LM **CE** and **KL(post‖pre)** on a small web-text slice.

**Decision rules (pre-registered):** **Win over β=1** if  
(i) \(H\) decreases by ≥**X** pp with \(U \le \)**Y** pp (e.g., X=3, Y=1),  
(ii) \(S_{0.10}\) increases, and  
(iii) CE/KL remain within small tolerances.

---

## Core Analyses & Plots

1) **Dose–response lines (best α per β)**  
   - Panel A: **H vs β** (bootstrap 95% CIs).  
   - Panel B: **U vs β** (95% CIs).  
   *Read:* good β lowers H with small U.

2) **HAP vs β (λ grid)**  
   - y = \(S_\lambda\), x = β; lines for λ∈{0.05, 0.10, 0.20}.  
   *Read:* look for a robust peak (often β≈0–0.5).

3) **Four-bucket stacked bars**  
   - One bar per β (best α), labeled **F&I** and **T&I** segments.  
   *Read:* wins shrink F&I while keeping T&I near baseline.

4) **Tiny results table** (one row per β)  
   - Columns: True×Info, \(S_{0.10}\), T, I, **H**, **U**, CE, KL.

**Optional (choose ≤2 if time allows):**  
A) **Trade-off scatter** (ΔInfo vs ΔTruth or U vs H) with iso-HAP contours (λ=0.1).  
B) **Cosine(d_β, d_1)** across K heads (interpretability snack).  
C) **α-sensitivity mini-plot**: HAP(0.10) vs α for each β.  
D) **Subcategory bars**: ΔF&I by TruthfulQA category.  
E) **CE/KL drift bars** per β.  
F) **Sankey** of transitions from β=1 to β=best.

---

## Sanity-Check Toy Example (debug your pipeline)

Assume **N=100**; illustrate expected directions of change.

- **β=1 (baseline)**: T&I=62, T&N=8, F&I=20, F&N=10 → T=0.70, I=0.82, **H=0.20**, True×Info=0.574, \(S_{0.10}=0.554\).
- **β=0**: T&I=64, T&N=11, F&I=15, F&N=10 → **H=0.15**, **U=1%**, True×Info=0.5925, \(S_{0.10}=0.5775\).
- **β=0.5**: T&I=64, T&N=10, F&I=16, F&N=10 → **H=0.16**, **U=0%**, True×Info=0.5920, \(S_{0.10}=0.5760\).
- **β=2**: T&I=58, T&N=15, F&I=17, F&N=10 → **H=0.17**, **U=4%**, True×Info=0.5475, \(S_{0.10}=0.5305\).

*Use these to sanity-check your plotting code; replace with real outputs later.*

---

## Implementation Checklist (12–20 h)

- **Setup (off-clock allowed):** model/harness; extraction of head outputs at last token; TruthfulQA loader.
- **0–3 h:** β=1 dataset; per-head probes; select top-K once.
- **3–6 h:** Implement β-weighted mean/σ; compute \(d_\beta\) for β∈{0,0.5,1,2}; wire intervention; α grid.
- **6–10 h:** For each β, pick α on val; run generation on test; compute buckets, H, U, True×Info, HAP(λ).
- **10–14 h:** CE/KL drift; bootstrap CIs; finalize α per β.
- **14–18 h:** Make dose–response plots + tiny table; (optional) cosine figure.
- **+2 h (separate):** Executive summary with the key figure set (H/U vs β, HAP vs β, stacked bars).

---

## Risks & Mitigations

- **Style confound:** Stock “I have no comment” may drive behavior. *Mitigation (scope-compatible):* **rephrased-IDK** variant for the positive mean; Phase-2 adds decode-time **phrase blocking**.
- **Overfitting to α/K:** Use a tiny validation split; **K fixed**, small α grid.
- **Judge variance:** Keep judge constant across conditions; optionally human-audit a small stratified sample.

---

## Expected Outcomes (decision rules)

- **TRUE-only viable:** β≈0 reduces **H** with small **U**; \(S_{0.10}\) ↑; True×Info flat; CE/KL stable.  
- **Rephrased-IDK helps:** β=1 (paraphrased) **H ↓** with Info ≈, **HAP ↑**.  
- **Oversampling IDK harms:** β=2 **U ↑** dominates; **HAP ↓**; True×Info falls via **T&N ↑**.

---

## Future Work (Phase-2)

1. **Phrase-block evaluation:** decode-time blocklist to verify non-stylistic abstention.  
2. **ASSERTIVE split + gating:** learn an abstention direction; **gate** by confidence; evaluate **risk–coverage** (behavioral calibration).  
3. **CAA at scale:** contrastive directions from (Q+pos A) vs (Q+neg A); compare to Mass Mean Shift.  
4. **Geometry:** whitened/CCA angles among \(d_{\text{truth}}, d_{\text{t+idk}}, d_{\text{idk}}\); head/layer maps.  
5. **OOD generalization:** FEVER-open / SciQ-open-style sets.  
6. **Token-level constraints:** decode without classic refusal n-grams.  
7. **Confidence targets (Kalai):** sweep **t**, compute **risk–coverage**, **AURC**, **coverage@fixed-risk**, and a **cost-aware score** (IDK=0; wrong-answer penalty \(t/(1-t)\)).

---

## References (informal)
- **TruthfulQA:** Lin et al., 2021; Askell et al., 2021.  
- **Inference-Time Intervention:** Li et al., 2024.  
- **Behavioral calibration:** Kalai et al., 2025.  
- **Contrastive/activation steering:** standard representation-engineering / activation-addition literature.

---

*Neel-track notes:* prioritize **clarity over breadth**; beat **simple baselines**; show **error bars**; pre-register **λ, α, K** and decision rules; present **one killer figure** that tells the story at a glance.
