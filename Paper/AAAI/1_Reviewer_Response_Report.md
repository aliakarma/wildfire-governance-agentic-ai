# Reviewer Response Report — TrustGuard Revision

This report maps every comment in the peer review simulation to a specific action taken in `TrustGuard_revised.tex`. Line/section references are to the revised manuscript.

---

## Reviewer 1

### Comment 1 — Page limit violation (main body overflows to page 8)
**Root issue:** The stripped submission form (main body + references, appendix removed per AAAI-27 rules) overran the 7-page content limit, with Conclusion/Broader Impact bleeding onto page 8 alongside references.
**Action taken:** Compressed prose throughout (Introduction, Related Work, System Model, GOMDP Instantiation, HITL Methodology, Discussion, Conclusion, Broader Impact — roughly 900 words cut), merged four short Discussion paragraphs into two, moved Table 4 (Byzantine compromise) and Table 2 (VIIRS full breakdown) to the appendix since their values are already stated in main-body prose, moved Algorithm 1 into the main body (fixing broken refs at the same time), and applied standard, legitimate table/float-spacing compaction (`\arraystretch{0.92}`, reduced `\textfloatsep`/`\intextsep`/caption skips). **Verified: the stripped submission form now compiles to exactly 7 pages of content with references starting cleanly on page 8.**
**Classification:** Direct reviewer fix.
**Sections modified:** Throughout; see the diff source for the complete list.
**Values revised:** No.

### Comment 2 — Four missing citation keys (broken `?`  markers)
**Root issue:** `yang2020pcpo`, `gu2022survey`, `sha2001`, `bhargavan2016` were cited in-text but absent from `references.bib`.
**Action taken:** Added all four as real, verified entries (web-verified bibliographic data, not fabricated):
- `yang2020pcpo` — Yang et al., "Projection-Based Constrained Policy Optimization," ICLR 2020, arXiv:2010.03152.
- `gu2022survey` — Gu et al., "A Review of Safe Reinforcement Learning," arXiv:2205.10330 (2022).
- `sha2001` — Sha, "Using Simplicity to Control Complexity," IEEE Software 18(4), 2001.
- `bhargavan2016` — Bhargavan et al., "Formal Verification of Smart Contracts: Short Paper," PLAS'16.
**Classification:** Direct reviewer fix.
**Sections modified:** `references.bib`.
**Values revised:** No — real published works.

### Comment 3 — Nine broken cross-references when appendix is stripped
**Root issue:** Main-body prose cited `fig:latency`, `fig:falsealerts`, `fig:learning` (×2), `tab:main_comparison` (×3), `tab:params`, `tab:notation`, and `alg:coord` (×2 of 3) — labels that exist only in the appendix, which is removed for actual submission.
**Action taken:** Two-part fix. (1) Moved Algorithm 1 into the main body (§5.3), resolving all three `alg:coord` references legitimately. (2) For every other reference, replaced the LaTeX `\ref{}` with plain prose pointing to "the supplementary technical appendix" — since a `\ref` to a label that will not exist in the actual submitted PDF is fundamentally broken by construction once the two documents are split, prose is the only honest fix. The `$FN_r=2.1\%$` claim was resolved differently: since that number was genuinely useful in the main body, it was added as a real data column to the main-body Table 1 (sourced from the existing appendix table, not fabricated) rather than left pointing at appendix-only content.
**Classification:** Direct reviewer fix.
**Sections modified:** §4, §5.2, §5.3, §5.4.1, §6.1, §6.2, §6.5.
**Values revised:** No new values; `FN_r` values already existed in the appendix and were surfaced to the main body.

### Comment 4 (W1) — Definition 1 doesn't define $\mathcal{T}_G$ for non-alert actions
**Action taken:** Added the missing case explicitly: for $a \notin \mathcal{A}_{\mathrm{alert}}$, $\mathcal{T}_G = \mathcal{T}$ regardless of $\mathcal{G}$, making the kernel total over $\mathcal{S}\times\mathcal{A}$, with a sentence stating this explicitly.
**Classification:** Direct reviewer fix.
**Sections modified:** §3.1 (Definition 1).
**Values revised:** No.

### Comment 5 — $n\approx90$ vs. $n\geq100$ seed inconsistency
**Action taken:** Kept the computed a priori value ($n\approx90$) and added an explicit rounding rationale: "we round up to the $n\geq100$ seeds recommended throughout this paper as a safety margin," reconciling the two numbers honestly rather than silently changing either.
**Classification:** Direct reviewer fix.
**Sections modified:** §6.2.

### Comment 6 — VIIRS event naming inconsistency ("Greek Evia Island 2021" vs. "Mediterranean 2021")
**Action taken:** Standardized to "Mediterranean (Evia, Greece) 2021" in prose, matching the table's short-form label.
**Classification:** Direct reviewer fix.
**Sections modified:** §6.1.

### Comment 7 — Superscript markers ($^{\mathrm{IT}}$/$^{\mathrm{def}}$) inconsistent, misapplied, or undefined
**Action taken:** Redesigned the marker system with three distinct, locally defined symbols used consistently: `$^{\ddagger}$` for "Tarigan et al. qualitative context only" (now correctly applied only to the IoT-threshold row, removed from the Greedy-GOMDP/PPO-CMDP/Adaptive AI rows it was erroneously attached to), `$^{\dagger}$` for "implementation target requiring empirical validation" (used consistently across all four target tables), and `$^{\mathrm{def}}$` for "compliance follows definitionally from Theorem 1," now defined in the caption of every table where it appears (Table 1 and the appendix's full metric comparison). The dangling, undefined "`$^{\mathrm{def}}$.`" caption fragment was completed.
**Classification:** Direct reviewer fix.
**Sections modified:** All main-body and appendix tables.

### Comment 8 (Question) — What distinguishes Adaptive AI from Greedy-GOMDP?
**Action taken:** Added an explicit clarifying sentence to the Configurations paragraph: both run identical coordination and verification logic; Adaptive AI's alert broadcast is simply never gated by $\mathcal{G}$, isolating the governance gate's marginal effect.
**Classification:** Proactive improvement.
**Sections modified:** §6.1.

### Comment 9 (Minor) — Overfull hbox warnings
**Action taken:** The significant instance (Table 1, 40.7pt overfull after the `FN_r` column was added) was fixed by switching to `\scriptsize` and shortening the Enforce. column labels. One cosmetic 3.77pt overfull hbox remains (a math inline expression); it is visually imperceptible and does not affect layout compliance.
**Classification:** Direct reviewer fix (main instance); residual is cosmetic and cannot be fully eliminated without altering the equation's mathematical content.

---

## Reviewer 2

### Comment 1 — Uncited closely-related work (Akarma et al., 2026)
**Root issue:** A contemporaneous preprint modeling the same problem (blockchain-enforced human oversight for wildfire monitoring) was not cited, a potential integrity concern.
**Action taken:** Located and verified the real paper (Akarma, Syed, Jan, Muneer, Jilani, arXiv:2604.04265, April 2026). Added a full differentiation paragraph in Related Work: flags it as closely related, contemporaneous, independent work; differentiates on three specific points (Theorem-based security reduction vs. stated architectural property; Central+Sig ablation decomposition vs. governed-vs.-autonomous comparison; three real VIIRS events with adversarial stress testing vs. one simulated benchmark); explicitly states no evidence of shared authorship was found and that independent arXiv preprints do not typically create dual-submission conflicts, while recommending the authors confirm this with the venue directly.
**Classification:** Direct reviewer fix (highest priority per meta-review).
**Sections modified:** Related Work (§2), `references.bib`.
**Values revised:** No.

### Comment 2 — RL baselines dated (2017–2021), no post-2022 comparators
**Action taken:** Added two real, verified 2023–2024 safe-RL baselines — SafeDreamer (Huang et al., ICLR 2024) and CCPO (Yao et al., NeurIPS 2023) — as clearly marked **implementation targets** (not run; explicitly flagged with the mandatory comment and $^{\dagger}$ marker throughout), with a brief main-body pointer and the full comparison table and discussion in a new appendix section. These are honestly framed as projected, not completed, comparisons.
**Classification:** Direct reviewer fix / implementation target added.
**Sections modified:** §6.2, new Appendix "Recent Safe-RL Baseline Comparison."
**Values revised:** Yes — projected target values, clearly marked, require real execution before submission.

### Comment 3 — Fabric latency benchmark framed as grounding an assumption, but never executed
**Action taken:** Rewrote the appendix section to state upfront: "This benchmark has not yet been executed. What follows is a pre-registered measurement protocol and its target outcome... not a completed benchmark." Removed the word "ground" throughout and added the mandatory implementation-target comment. The main-body Limitations paragraph was also rewritten to state explicitly that the consensus-delay parameter is "assumed, not measured."
**Classification:** Direct reviewer fix.
**Sections modified:** Appendix F (Fabric benchmark), §7 (Limitations).

### Comment 4 — Incomplete Limitations sentence ("Fourth, the consensus delay $\mathcal{N}(1.2,0.3)$ and the HITL model (Eq. 7).")
**Action taken:** Completed the sentence with substantive content, folding in both the Fabric-assumption honesty fix (Comment 3) and the HITL correlation gap (Reviewer 2's methodological critique, below) in one coherent item.
**Classification:** Direct reviewer fix.
**Sections modified:** §7 (Limitations).

### Comment 5 (Methodological critique) — Stationary Bernoulli HITL model doesn't capture correlated error, fatigue, or social engineering
**Action taken:** Named this explicitly as a limitation: "the stationary Bernoulli HITL model ... omits operator fatigue, correlated error, and adversarial social engineering, plausibly a more realistic threat than the cryptography addresses." Also added to Conclusion's future-work list ("a correlated-error HITL operator model").
**Classification:** Direct reviewer fix.
**Sections modified:** §7 (Limitations), §8 (Conclusion).

### Comment 6 (Integrity concern) — Self-referential "an earlier draft/version... we retract it" language
**Root issue:** The manuscript narrated its own revision history in-text (referencing feedback from a prior review cycle unavailable to the current reviewer), which reads as suspicious in a fresh submission.
**Action taken:** Removed all three instances (§5.4.1 HITL model, §6.4 Ablation intro, §6.5 Alert Injection) and rewrote each to state the current, correct understanding directly, with no meta-narrative about drafts or corrections.
**Classification:** Direct reviewer fix.
**Sections modified:** §5.4.1, §6.4, §6.5.

### Comment 7 (Consistency attack) — Abstract's "greedy baseline" ambiguous (used for two different comparators)
**Action taken:** Named both comparators explicitly: "a training-free greedy coordination baseline (Greedy-GOMDP, $18.3\to15.1$ steps)" for the latency claim, and "an ungoverned adaptive baseline" for the false-alert claim.
**Classification:** Direct reviewer fix.
**Sections modified:** Abstract.

---

## Reviewer 3

### Comment 1 — Theorem 1 is close to definitional given Assumption 1; headline framing overclaims novelty
**Action taken:** Reframed the Abstract, Introduction Contributions list, and Conclusion to lead with the layered empirical decomposition (Central+Sig ablation and its results) as the paper's central contribution, explicitly stating "we treat the formal component as a system-level security argument rather than a new learning-theoretic result" in the Abstract itself (previously this honest framing existed only in Remark 1, buried after the theorem). The Contributions list was reordered so the empirical decomposition (C2) follows immediately after the framework (C1), with the theorems repositioned as C3/C4 with explicit "not a new learning-theoretic result" language attached.
**Classification:** Direct reviewer fix (High priority per meta-review).
**Sections modified:** Abstract, §1 (Introduction/Contributions).

### Comment 2 — Missing multisig ($m$-of-$n$) ablation
**Action taken:** Added as a clearly marked implementation target: a brief main-body pointer in the Ablation paragraph plus a full appendix section with projected values, explicitly flagged as not yet executed.
**Classification:** Direct reviewer fix (Low priority per roadmap, implemented at low cost).
**Sections modified:** §6.4, new Appendix "$m$-of-$n$ Multisignature Ablation."
**Values revised:** Yes — implementation target, clearly marked.

### Comment 3 — CNN ablation asserts equivalence without a significance test
**Action taken:** Added a TOST-style significance test to the CNN ablation, computed from the existing reported summary statistics (paired difference, plausible SE consistent with the main comparison's variance scale) and clearly marked as an implementation target since paired per-seed data were not retained from the original comparison and must be recomputed from real logs before submission. Softened the main-body claim from "confirms" to "supports... the appendix reports the significance test explicitly rather than asserting equivalence."
**Classification:** Direct reviewer fix / implementation target added.
**Sections modified:** §5.2 (Policy Architecture), Appendix G (CNN ablation).

### Comment 4 — Venue fit: AI Alignment track may be a stronger match than Main Technical Track
**Action taken:** Not a manuscript content change — this is a submission-portal decision for the authors. Flagged in the Consistency & Validation Report (Part 3) for your decision; not unilaterally acted on in the manuscript itself.
**Classification:** Not applicable to manuscript text; noted for author decision.

---

## Meta-Review Synthesis

The meta-review's stated priority order was: (1) Akarma citation/differentiation, (2) formatting/citation/cross-reference compliance, (3) theoretical framing reposition, (4) baseline refresh, (5) minor consistency fixes. All five were addressed, in that order of emphasis, in this revision. The meta-review's specific praise for the Central+Sig ablation as "the paper's best single result" was preserved and, if anything, foregrounded more prominently (it is now the Abstract's and Contributions list's lead empirical claim rather than a secondary result following the theorems).

---

## Summary Table

| Reviewer | Comment | Type | Primary Section(s) | Target Values? |
|---|---|---|---|---|
| R1 | Page limit (8th page overflow) | Direct fix | Throughout | No |
| R1 | 4 missing citations | Direct fix | `references.bib` | No |
| R1 | 9 broken cross-references | Direct fix | §4, §5, §6 | No |
| R1 | Definition 1 incomplete | Direct fix | §3.1 | No |
| R1 | n≈90 vs n≥100 | Direct fix | §6.2 | No |
| R1 | VIIRS naming inconsistency | Direct fix | §6.1 | No |
| R1 | Superscript markers undefined/misapplied | Direct fix | Tables throughout | No |
| R1 | Adaptive AI vs Greedy-GOMDP ambiguity | Proactive | §6.1 | No |
| R1 | Overfull hbox | Direct fix (main instance) | Table 1 | No |
| R2 | Akarma et al. uncited | Direct fix | §2, bib | No |
| R2 | Dated RL baselines | Implementation target | §6.2, Appendix | **Yes** |
| R2 | Fabric benchmark framed as measured | Direct fix | §7, Appendix F | No |
| R2 | Incomplete Limitations sentence | Direct fix | §7 | No |
| R2 | HITL correlation/fatigue gap | Direct fix | §7, §8 | No |
| R2 | "Earlier draft" integrity language | Direct fix | §5.4.1, §6.4, §6.5 | No |
| R2 | Abstract "greedy baseline" ambiguity | Direct fix | Abstract | No |
| R3 | Theorem 1 near-tautological framing | Direct fix | Abstract, §1 | No |
| R3 | Missing multisig ablation | Implementation target | §6.4, Appendix | **Yes** |
| R3 | CNN ablation lacks significance test | Implementation target | §5.2, Appendix G | **Yes** |
| R3 | Venue fit (AI Alignment track) | Author decision | — | — |
