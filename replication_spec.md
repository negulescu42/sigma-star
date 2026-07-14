# Cross-model / cross-dataset replication of the task-grounded attention certificate

Frozen before inspecting any result. Answers the #1 ask named by all three review
signals (neutral referees ×2, Mihai's note, adversarial pass): is the
diagnose-and-train result a property of the CERTIFICATE, or of one adapter-tuned
BERT on one filtered corpus?

## Grid (2 readers × 2 datasets = 4 cells)
Readers:
- BERT   = bert-base-uncased + AdapterHub/bert-base-uncased-pf-hotpotqa   (original)
- RoBERTa= roberta-base       + AdapterHub/roberta-base-pf-hotpotqa       (new, probe-confirmed loads)
Datasets:
- HotpotQA = hotpotqa/hotpot_qa distractor                                (original)
  context={title:[...], sentences:[[...]]}, supporting_facts={title,sent_id}
- 2Wiki    = scholarly-shadows-syndicate/2WikiMultihopQA_with_q_gpt35     (new, parquet mirror)
  context={title:[...], content:[[...]]}, supporting_facts={title,sent_id}
  NOTE (probe-confirmed, not assumed): supporting_facts is identical to HotpotQA;
  context uses key 'content' (not 'sentences') but is structurally the same
  parallel list-of-sentence-lists, so build_example normalizes over both keys.
  Of 300 train examples: 121 yes/no (dropped, span-only), 179 span answers,
  all 179 with answer-in-context and supporting-fact titles subset of context.

BERT×HotpotQA is the already-reported cell (hotpot_unified.json); we RE-RUN it here
too under identical code so all four cells are strictly comparable.

## Protocol (identical to hotpot_unified.py, applied to each cell)
- Same partition: near = question + annotated supporting-fact tokens + [CLS]/[SEP];
  far = remaining context, fixed before reading attention.
- Same retention filter: truncate to 512 tokens, keep only if ALL supporting
  sentences + answer span survive; span answers only (yes/no dropped).
- Reader = pretrained adapter continued-trained on 1500 retained train examples
  (matched lambda=0), evaluated on retained dev/test.
- Certificate: generalized external-partition Gibbs bound, q=2, eps=0.05,
  tau0=sqrt(d_k). Analysed layers = upper 4, all heads.

## Measures per cell (same as unified audit)
1. EM/F1 of the reader (task performance sanity).
2. Certificate audit: positive-gap rate, empirical compliance Pr(mF<=eps),
   analytic coverage, viol count, mean mF, per-layer mF.
3. Deletion coupling: relpert cert vs uncert.
4. Controls: task vs size-matched random far set, analytic coverage + paired
   bootstrap gap (does task annotation beat random? = is the partition informative).
5. Training: 1 penalty level lambda=8 (the strongest, clearest signal), 1 seed
   for cross-cells (compute budget), matched lambda=0 control -> does far-mass
   regularization raise certified coverage at bounded accuracy cost?
   (Full 3-seed 4-lambda sweep already exists for BERT×HotpotQA.)

## Promotion rule (frozen)
- REPLICATES (certificate holds 0 viol + task>random + training lifts coverage in
  >=1 new cell): the diagnose-and-train claim is a property of the certificate ->
  add a cross-model/cross-dataset paragraph + SI table, state which cells hold.
  This is the Significance-capping gap all referees named; closing it is the lift.
- PARTIAL (holds in some cells, not others): report honestly which transfer and
  which do not; scope the claim to what replicates.
- FAILS (certificate violates or training doesn't lift elsewhere): report as a
  scoping boundary; the BERT×HotpotQA result stands as single-model, claim narrowed.
- Report positive/mixed/negative truthfully. Success = coherent finding.
