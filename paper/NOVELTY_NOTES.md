# Novelty check (working notes)

Search date: 2026-10-28. Sources were abstracts and snippets; full texts were not read except where stated, so treat the "does not do" column as "not stated in the abstract".

## Closest prior work

| Work | What it does | What it does not state |
|---|---|---|
| VeriFi (Gao et al., arXiv 2205.12709; IEEE TDSC 2024) | Leaving client marks its data with backdoor-style markers; verifies by watching marker accuracy drop. Evaluates seven unlearning and five verification methods. | Finite-sample error control; a null that holds by construction; cost in energy. Markers are intrusive (backdoors in the training data). |
| PrivEraserVerify (arXiv 2604.12348, Apr 2026) | Adaptive checkpointing, layer-wise DP calibration, fingerprint-based verification. Claims 2-3x faster than retraining. | A hypothesis test with stated error bounds; ledger/provenance; energy analysis (abstract). |
| Blockchain-enhanced federated unlearning for IoT (Zuo et al., arXiv 2405.20776) | Blockchain records unlearning requests and actions, argued as a verifiable certification route. | A statistical guarantee or the certification procedure (abstract). |
| BlockFUL (arXiv 2402.16294) | Unlearning inside a blockchained FL system. | Verification of residual influence. |
| FedEraser, FedRecover, FedQUIT, FedRecovery, SIFU, certified client removal (arXiv 2404.09724) | Efficient or certified-by-noise client removal; FedQUIT reports bytes and FLOPs. | An audit that a third party can run on the released model without a retrained reference. |
| Backdoor / poison unlearning (Wu and Zhu; FedUP; FAUN; FedCARE) | Recover from poisoned clients. | Certificates of removal. |
| VOUCH (Dang, Machine Learning, accepted) | Paired ghost canaries and betting e-processes give anytime-valid certificates for centralised LLM unlearning. | Federated setting, client-level revocation, partial participation, device costs, ledger anchoring. |

## Gap this paper targets

1. No federated-unlearning verifier we found offers an exact finite-sample, anytime-valid guarantee that needs neither a retrained reference model nor backdoor markers in the data.
2. Governance (consent revocation, provenance, audit) and statistical verification are treated separately. Here the canary manifest is committed in the ledger before training and the certificate is bound to the ledger head and the released model hash.
3. Cost is rarely tied to the verification decision. Here the certificate is the acceptance test of a cheap-first ladder, and energy and bytes of the whole ladder are reported.

## Overlap with the author's VOUCH paper (must be managed)

IEEE policy requires significant extension over prior publications. Reused: the e-process verifier and its theorems, cited and called as a library. New here: the federated protocol (client-held manifests, revocation waves, partial participation), the escalation policy and its error accounting, the ledger, the cost model, and the experiments. Do not copy VOUCH text or figures.

## Risks

- Reviewers may say the linear-head-on-embeddings setting is small. A neural model would reduce this risk.
- Canary audits only have power when the model can memorise; the sweep over feature dimension makes this visible.
- The certificate is per revocation wave, not per client.

## Relation to the author's own work (stated in the paper)

FORTRESS-FL (training-time robust aggregation) and VOUCH (centralised unlearning verifier) are both cited. This paper adds checkpoints and ledger, consent registry, unlearning operators, federated canary protocol with waves, verify-then-escalate with its error accounting, the cost model and the experiments. The e-process verifier and its coverage theorem are reused from VOUCH and not claimed as new.
