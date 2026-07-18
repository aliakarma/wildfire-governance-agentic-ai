# Consistency & Validation Report — TrustGuard Revision

An honest internal assessment. This is for your planning, not for the manuscript itself.

---

## Section 1 — Implementation Targets Requiring Validation

Three blocks of implementation-target content were added or extended in this revision. **None of these are real experimental results.** Each is marked in the `.tex` source with the mandatory `% IMPLEMENTATION TARGETS:` comment and a `$^{\dagger}$` superscript in the rendered table.

### 1.1 Recent Safe-RL Baselines (SafeDreamer, CCPO)
- **Location:** §6.2 (brief pointer) and new Appendix "Recent Safe-RL Baseline Comparison" (full table, Table `tab:recent_rl`).
- **What to run:** Train SafeDreamer (Huang et al., ICLR 2024) and CCPO (Yao et al., NeurIPS 2023) inside your existing GOMDP simulation environment, under the identical objective (Eq. 1), action space, reward scaling, and $N=20$/$n=20$-seed protocol used for every other baseline in Table 1.
- **What the targets assume:** Both algorithms reimplemented faithfully from their published papers (or using their official code if available), with hyperparameters tuned to reasonable convergence in your environment — not copied blindly from their original domains (world models trained for locomotion/robotics tasks, not wildfire sector-assignment).
- **Plausible outcome range consistent with the rest of the paper:** $L_d \in [14.5, 15.5]$ steps, $F_p \in [6\%, 10\%]$, compliance in the low-to-mid 90s% — i.e., competitive-to-slightly-better raw latency than PPO-CMDP/WCSAC, but still an imperfect-compliance policy-level method, consistent with the paper's central argument that policy-level methods cannot match environment-level cryptographic enforcement.
- **Red flag if real results diverge:** If either method achieves >99% compliance empirically, that would undercut the paper's central claim (that cryptographic environment-level enforcement is categorically necessary, not just empirically convenient) and would need to be addressed head-on in the Discussion, not hidden.

### 1.2 $m$-of-$n$ Multisignature Ablation
- **Location:** §6.4 (brief pointer) and new Appendix "$m$-of-$n$ Multisignature Ablation" (Table `tab:multisig`).
- **What to run:** Replace the BFT consensus layer with an $m$-of-$n$ threshold-signature scheme (no blockchain ledger) in your existing simulation harness, keeping the rest of the GOMDP pipeline identical to Central+Sig.
- **What the targets assume:** The same authentication primitive (Ed25519) used elsewhere, just aggregated via threshold signatures instead of full BFT consensus; no on-chain component.
- **Plausible outcome range:** Should closely match Central+Sig's numbers ($L_d\approx15.0$, $F_p\approx6.0\%$, 100/100 injection resistance) since the authentication mechanism is the same; the interesting empirical question is whether it's *meaningfully* cheaper in latency/overhead than full consensus, which the current target values do not address (they only test injection resistance, not overhead — you may want to add an overhead/energy column when you run this for real).
- **Red flag:** If injection resistance is not 100/100, the "same authentication primitive" framing is wrong and needs revisiting.

### 1.3 CNN-vs-MLP TOST Significance Test
- **Location:** Appendix G (CNN-Architecture Ablation).
- **What to run:** Recompute the TOST statistic from real **paired per-seed** data (same 20 seeds, both architectures), not from the summary mean±std alone. The current appendix values are a plausible reconstruction assuming a standard error consistent with the main comparison's variance scale — this is explicitly flagged as unverified in the text.
- **Plausible outcome range:** Given the reported means (15.1 vs. 14.9, a 0.2-step gap smaller than the PPO-GOMDP/PPO-CMDP TOST's 0.3-step gap which already passed at $\Delta=1.0$), equivalence within the same margin is very likely to hold with real data too — but the exact $p$-values will differ from the projected ones.
- **Red flag:** If the real paired data show a systematically larger gap than 0.2 steps, the "statistically indistinguishable" framing in §5.2 would need softening.

---

## Section 2 — Claims Intentionally Softened

- **"Confirms" → "supports"** for the CNN ablation equivalence claim (§5.2), since the significance test backing it is now honestly marked as a target pending real paired data.
- **The Fabric latency benchmark** was reframed from implicitly "grounding" the assumed consensus-delay parameter to explicitly "a pre-registered protocol... not a completed measurement." This is a softening in framing, not in the underlying numbers (the target outcome $2.3\pm0.7$s is unchanged) — but the epistemic status is now honest rather than implied-complete.
- **The theoretical contribution's prominence** was deliberately reduced relative to the empirical contribution throughout (Abstract, Introduction, Conclusion) per Reviewer 3 / meta-review consensus that Theorem 1 is close to definitional given Assumption 1. This is a framing change, not a retraction — Theorem 1 and Theorem 2 are unchanged and still fully stated; they are simply no longer positioned as the paper's headline claim.

## Section 3 — Remaining Risks

- **Assumption 1 (faithful chaincode) is still unverified.** This was flagged as the highest-priority limitation and future work item, but it remains a real gap: the entire safety case is conditional on code that has not been formally verified. A sufficiently skeptical reviewer could still argue the "formal safety case" framing overclaims relative to what's actually been proven about the deployed artifact, even with the reframing.
- **$n=20$ seeds is still underpowered** for the primary latency-equivalence claim (the TOST result is now the load-bearing evidence, which is honest, but a reviewer could still ask why the underpowered comparison is reported at all rather than running more seeds before submission — this is straightforward to fix if you have compute budget before the actual deadline).
- **The Akarma et al. differentiation is a judgment call under uncertainty.** I could not determine authorship overlap from the available preprint (only its listed authors: Akarma, Syed, Jan, Muneer, Jilani — none of these obviously match your identity, but I have no way to fully verify this from outside). If you are in fact connected to that paper in any way, the differentiation paragraph and dual-submission language need to be revisited before you submit — please check this yourself.
- **The two new implementation-target baselines (SafeDreamer, CCPO) are the single biggest remaining gap between this draft and a submission-ready paper.** Reviewers who read carefully will notice the $\dagger$ markers and appendix framing, but a paper that claims to "add recent baselines" while marking them as unexecuted targets is a meaningfully weaker claim than one with real numbers. This should be your top priority before the actual deadline.
- **The three tables moved to the appendix (Table 2 VIIRS full breakdown, Table 4 Byzantine compromise, and the original Table 1 companion data)** are now only accessible to reviewers who read the supplementary material, which AAAI reviewers are not required to do. The main-body prose was written to be fully self-contained (every number a reviewer needs for the paper's core claims is stated in text), but this is a real trade-off against the page limit, not a free move.

## Section 4 — Publication Readiness Estimate

- **Real vs. projected:** Roughly 90% of the manuscript's content is grounded in the original evidence base (real citations, real prior results, honest reframing of existing claims). The remaining ~10% is the three implementation-target blocks above, all clearly marked.
- **Before actual submission, you need to:**
  1. Run the SafeDreamer and CCPO baselines for real (highest priority — this is the most visible "target" content).
  2. Run the CNN-vs-MLP TOST from real paired seed data.
  3. Decide on and, if time permits, run the $m$-of-$n$ multisig ablation (lowest priority, optional per the original roadmap).
  4. Confirm the Akarma et al. relationship (dual-submission check) directly with yourself/co-authors.
  5. Decide on Main Technical Track vs. AI Alignment track (Reviewer 3 / meta-review suggestion) — this is a portal decision, not a manuscript change.
  6. If compute budget allows, increase seeds toward $n\geq100$ for the latency-equivalence comparison to remove the underpowered-test caveat entirely.
- **Estimated effort:** Items 1–2 are the real work (likely 1–3 days of compute plus analysis, depending on your infrastructure); items 3–6 are each under a day.
- **Overall assessment:** The manuscript is now structurally ready — formatting, citations, cross-references, and internal consistency are all clean and verified by direct compilation (zero broken references, zero missing citations, exactly 7 pages of main-body content as required). What remains is empirical: closing the gap between the two marked implementation-target result blocks and real numbers. This does not require another full revision cycle, but it does require real experiments before this can honestly be submitted.
