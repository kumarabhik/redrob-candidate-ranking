# Mathematical Specification

Formal definitions for every scoring component. Symbols are shared across sections.
Companion to [design_doc.md](design_doc.md) (prose) and [system_design.md](system_design.md) (serving).

## Notation
- Candidate pool $C = \{c_1,\dots,c_N\}$, $N = 100{,}000$.
- A job description $J$ decomposed into aspects $A = A^{\text{must}} \cup A^{\text{nice}}$, each aspect $a$ with weight $w_a \ge 0$, a skill vocabulary $V_a$, and a natural-language query $q_a$.
- Embedding map $\phi:\text{text}\to\mathbb{R}^d$ (L2-normalized, $\lVert\phi(\cdot)\rVert_2=1$), so cosine similarity is the dot product.
- Profile text $t_i$ for candidate $c_i$; profile embedding $\mathbf{e}_i=\phi(t_i)$. Aspect query embedding $\mathbf{q}_a=\phi(q_a)$.

## 1. Recall (Stage A)

### 1.1 Sparse — BM25
For query terms $Q$ (role keywords ∪ $\bigcup_a V_a$) and document $t_i$ with length $|t_i|$ and average length $\overline{dl}$:
$$\text{BM25}(Q,i)=\sum_{w\in Q}\text{IDF}(w)\cdot\frac{f(w,i)\,(k_1+1)}{f(w,i)+k_1\bigl(1-b+b\frac{|t_i|}{\overline{dl}}\bigr)},\quad \text{IDF}(w)=\ln\frac{N-n_w+0.5}{n_w+0.5}+1$$
with $f(w,i)$ the term frequency, $n_w$ the document frequency, and defaults $k_1=1.5,\ b=0.75$ (Okapi).

### 1.2 Dense — semantic aspect similarity
$$s_{i,a}=\mathbf{e}_i^\top \mathbf{q}_a\in[-1,1],\qquad \text{dense}(i)=\frac{\sum_{a\in A^{\text{must}}}w_a\,s_{i,a}}{\sum_{a\in A^{\text{must}}}w_a}.$$
The full matrix $S=[s_{i,a}]\in\mathbb{R}^{N\times|A|}$ is computed once as $S=E\,Q^\top$ ($E$ stacks $\mathbf{e}_i$, $Q$ stacks $\mathbf{q}_a$) and reused as features (§2).

### 1.3 Fusion — Reciprocal Rank Fusion
Let $r_d(i)$ and $r_b(i)$ be the 1-based ranks of $i$ under dense and BM25 (best = 1). With $k=60$:
$$\text{RRF}(i)=\frac{1}{k+r_d(i)}+\frac{1}{k+r_b(i)}.$$
The shortlist $\mathcal{S}$ is the top $M{=}8000$ by RRF. RRF is scale-free (uses ranks, not scores), so the incomparable ranges of BM25 ($[0,\infty)$) and cosine ($[-1,1]$) never need calibration.

## 2. Feature map (Stage B)
Each candidate maps to $\mathbf{x}_i=\Psi(c_i,S_{i,\cdot})\in\mathbb{R}^{F}$, $F=76$. Representative components:
- **Aspect fit:** semantic $s_{i,a}$; lexical hit count $h_{i,a}=\lvert\{v\in V_a: v\subseteq t_i^{\text{skills}}\}\rvert$; career-evidence indicator $\mathbb{1}[\exists v\in V_a: v\subseteq t_i^{\text{career}}]$.
- **Seniority band** (years $y_i$, ideal $[\,\underline{y},\overline{y}\,]=[6,8]$, broad $[5,9]$):
$$\beta_i=\begin{cases}1.0 & \underline{y}\le y_i\le\overline{y}\\ 0.8 & y_i\in[5,9]\setminus[\underline y,\overline y]\\ 0.1 & y_i<3\\ \max(0.2,\,0.8-0.1\,\delta_i) & \text{else}\end{cases},\quad \delta_i=\min(|y_i-5|,|y_i-9|).$$
- **Behavioral availability** (response rate $\rho_i$, days-inactive $\tau_i$, open flag $o_i$, notice $n_i$):
$$\alpha_i=\underbrace{g_\rho(\rho_i)}_{[0.6,1]}\cdot\underbrace{g_\tau(\tau_i)}_{[0.4,1]}\cdot\underbrace{(o_i?1:0.8)}_{}\cdot\underbrace{g_n(n_i)}_{[0.7,1]},\quad g_\tau(\tau)=\min\!\Bigl(1,\max\bigl(0.4,1-\tfrac{\tau-120}{365}\bigr)\Bigr).$$
- Plus product-vs-services fraction, location fit, verified-competence (assessment/endorsement-trust/systems-tenure), and binary trap detectors (wrong-role, consulting-only, keyword-stuffer, job-hopper, research-only, managerial).

## 3. Honeypot gate (Stage C)
A hard predicate $H(c_i)\in\{0,1\}$; $H=1\Rightarrow$ excluded from the result set. With snapshot date $T_0$, role span $\Delta_r=\text{months}(T_0\ \text{or}\ \text{end}_r - \text{start}_r)$:
$$H(c)=\bigvee\Bigl[\ \text{dur}_r>\Delta_r+\max(2,0.15\Delta_r),\ \ (\text{is\_current}\wedge \text{end}\neq\varnothing),\ \ \text{end}<\text{start},\ \ \lvert\{s:\text{expert},\,\text{dur}_s{=}0\}\rvert\ge3,\ \ \text{edu.start}>\text{edu.end}\ \Bigr].$$
Noisy-but-legal signals (inverted salary, skill-dur > career) are **soft** features, not in $H$ — empirically they fire on 18.9% / 13.4% of the pool, so gating on them would destroy precision. Measured: $\sum_i H(c_i)=40$ ($0.04\%$).

## 4. Weak-label rubric (training target)
No ground truth ships, so define a graded relevance $\text{rel}(c)\in[0,1]$ from the JD rubric. With normalized semantic $\hat s_{a}=\mathrm{clip}\!\bigl(\tfrac{s_a-0.05}{0.35},0,1\bigr)$ and per-aspect fit $\text{fit}_a=\mathrm{clip}(0.5\hat s_a+0.3\min(1,h_a/2)+0.2\,\text{cev}_a)$:
$$\text{core}=\frac{\sum_{a\in A^{\text{must}}}w_a\,\text{fit}_a}{\sum_{a\in A^{\text{must}}}w_a},\qquad
\text{rel}(c)=\Bigl[\text{core}\cdot\beta'\cdot\ell\cdot\alpha'\cdot\pi\cdot\kappa\cdot\!\!\prod_{p\in\text{traps}}\!\!\lambda_p\Bigr]+\text{bonus}_{\text{nice}},$$
where $\beta'=0.1+0.9\beta$ (seniority), $\ell$ location fit, $\alpha'=0.3+0.7\alpha$ (availability), $\pi$ product factor, $\kappa=0.85+0.15\,\text{cred}$ (verified competence), and penalties $\lambda_p\in\{0.2,0.3,0.35,0.4,0.6,0.7\}$ for {wrong-role, wrong-domain, keyword-stuffer, research-only, managerial, job-hopper}. Honeypots: $\text{rel}=0$. Tiers $y_i\in\{0,1,2,3,4\}$ by thresholds $\{0.24,0.38,0.52,0.68\}$.

## 5. Reranker (Stage D) — LambdaMART
Gradient-boosted trees $f:\mathbb{R}^F\to\mathbb{R}$ minimizing a pairwise loss whose gradients are weighted by the NDCG change from swapping a pair $(i,j)$:
$$\lambda_{ij}=\frac{-\sigma}{1+e^{\sigma(f(\mathbf{x}_i)-f(\mathbf{x}_j))}}\,\bigl|\Delta\text{NDCG}_{ij}\bigr|,\qquad \lambda_i=\sum_{j:\,y_i>y_j}\lambda_{ij}-\sum_{j:\,y_j>y_i}\lambda_{ji}.$$
Trees are fit stage-wise to $\{\lambda_i\}$ (XGBoost `rank:ndcg`). **Training set balance:** a single group of $N$ is $99.5\%$ tier-0, collapsing $\Delta\text{NDCG}$; we train on $C_{\text{train}}=\{i:y_i\ge1\}\cup\text{hardneg}\cup\text{rand}$ (~6k, hard negatives = top-rubric tier-0).

## 6. Ensemble (late fusion)
With min-max scaler $u(\mathbf{z})_i=\frac{z_i-\min z}{\max z-\min z}$, model scores $f_i$, rubric $\text{rel}_i$, and optional cross-encoder $\text{ce}_i$ (precomputed; missing → median):
$$\text{score}_i=\begin{cases}0.5\,u(f)_i+0.2\,u(\text{rel})_i+0.3\,u(\text{ce})_i & \text{CE available}\\[2pt] 0.6\,u(f)_i+0.4\,u(\text{rel})_i & \text{otherwise.}\end{cases}$$
The cross-encoder $g_\theta(J,c)$ is a teacher run **offline** over the top recall set; it is semi-independent of the rubric, reducing the weak-label circularity.

## 7. Selection, de-dup, calibration (Stages D→F)
- **Twin de-dup:** greedily take candidates best-first; skip $c_i$ if $\max_{j\in\text{selected}}\mathbf{e}_i^\top\mathbf{e}_j>0.985$ (near-duplicate "behavioral twin"); backfill if pruned below 100.
- **Order:** sort by $(\,-\text{score}_i,\ \text{candidate\_id}_i)$ (deterministic tie-break, FR-002).
- **Calibration (CSV score):** strictly monotone, top-heavy $\;\hat\sigma_k=0.06+0.93\,u(\text{score})_k^{0.65}$ over the chosen top-100, preserving order while spreading the head.

## 8. Evaluation metrics (the competition objective)
For a ranking with graded relevances $\text{rel}_1,\dots$ (rank order):
$$\text{DCG@}k=\sum_{i=1}^{k}\frac{\text{rel}_i}{\log_2(i+1)},\quad \text{NDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k},$$
$$\text{P@}k=\frac{1}{k}\sum_{i=1}^{k}\mathbb{1}[\text{rel}_i\ge3],\quad \text{AP}=\frac{\sum_i \mathbb{1}[\text{rel}_i\ge3]\cdot\text{P@}i}{\sum_i \mathbb{1}[\text{rel}_i\ge3]},$$
$$\boxed{\ \text{Composite}=0.50\,\text{NDCG@}10+0.30\,\text{NDCG@}50+0.15\,\text{MAP}+0.05\,\text{P@}10\ }$$
This is exactly what `src/evaluate.py` computes; the reranker (§5) optimizes the dominant NDCG terms directly.

## 9. Complexity (per JD, serve time)
Let $M$ = shortlist size, $F$ = features, $|Q|$ = query terms.
- Dense scoring $S_{\cdot,a}$: $O(Nd)$ (one matvec per aspect, vectorized).
- BM25 scoring: $O(N\,|Q|)$ (index prebuilt).
- Features + rerank: $O(M F)$ + tree inference $O(M\cdot\text{trees}\cdot\text{depth})$.
Candidate embeddings $E$ and the BM25 index are **JD-independent**, so they are built once and shared across all users/JDs; only $Q$, $S_{\cdot,A}$, features and rerank are per-JD — the basis for the multi-tenant design in [system_design.md](system_design.md).
