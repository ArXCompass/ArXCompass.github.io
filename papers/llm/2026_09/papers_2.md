# llm - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02967v1">Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems via Federated Graph Learning</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Topology-guided safeguards for LLM-based multi-agent systems (MAS) train a GNN over the inter-agent communication graph to localize risky agents and intervene on the topology---but they assume one operator can pool all labeled traces. Across organizations that assumption breaks: episodes contain private prompts, tool outputs, and proprietary workflows, and no silo alone sees the full attack distribution. We cast privacy-preserving MAS safeguarding as graph federated learning and instantiate FGLGuard: each operator fits an edge-featured graph attention detector on its own judge-labeled episode graphs and shares only model updates. The method couples a proximal local objective for non-IID clients, domain-balanced aggregation, over-refusal-constrained threshold calibration, corroborated upstream scoring, and a guarded rewrite for blocked answers. Federation is not optional: off-the-shelf transfer collapses under distribution shift (AUROC 0.51 to 0.70 only after in-domain retraining), so a deployable guard must adapt on each site's private traces. On Agent-SafetyBench, R-Judge, and AgentDojo, federated FGLGuard exceeds the in-domain centralized ceiling on all three benchmarks without pooling any data---where unsupervised anomaly guards and local-only training fail. One guard federated across four different-domain operators comes within 0.03 AUROC of multi-domain centralization, while any single-domain guard collapses on the others. Live FGLGuard cuts AgentDojo's ground-truth attack-success rate by 43% at near-unguarded utility, zero API cost, and negligible capability loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02250v1">RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Ride-sharing, which allows multiple passengers with different origin-destination (OD) pairs to share a single vehicle, is a challenging operational problem, as it requires orders with different OD pairs to be efficiently bundled and assigned to vehicles under uncertain and varying scenarios. Although multi-agent reinforcement learning (MARL) solutions have achieved promising performance, they suffer from limited generalization (adapting to different environmental scenarios), low transferability (adapting to different platform objectives), and training difficulties in large-scale systems, such as the curse of dimensionality. Recently, motivated by the scaling of large language models (LLMs), several works have incorporated LLMs into ride-hailing systems, either by employing LLMs directly as decision-making agents or using them for automatic algorithm design. However, none of these approaches support vehicle sharing, which complicates the problem by expanding both the state and action spaces exponentially. Moreover, most of them require frequent LLM calls at inference time, making them infeasible for real-time deployment. To address these issues, we propose RideSkill, a hierarchical method for ride-sharing that leverages LLM-assisted automatic algorithmic design. RideSkill consists of a combiner that assigns appropriate skills to each vehicle from a learned skill repository, enabling adaptive dispatch under varying scenarios and objectives, and a repositioner that sequentially relocates idle vehicles to emerging regions, avoiding conflicts among vehicles. Crucially, the skill repository, combiner, and repositioner are all trained by an LLM-based automatic evolutionary method, eliminating the need for LLM calls during deployment and thus ensuring high real-time performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06973v3">LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted at the Conference on Lifelong Learning Agents (CoLLAs) 2026
    </div>
    <details class="paper-abstract">
      As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing public responses consistent with a fixed hidden state. We show theoretically that any agent restricted to the public conversation history cannot both keep the secret unresolved in the transcript and respond consistently with a fixed hidden state in PSITs, yielding an architectural impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency with a fixed hidden state, establishing private state as a necessary component for PSIT-capable language agents. Our code is available at https://github.com/chandar-lab/Hangman
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02246v1">LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 20 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Self-improving agent pipelines have a problem at their center. An optimizer rewrites prompts to score higher, and the score comes from a judge that is itself an LLM. That judge has the last word on whether the system is getting better, and our position is that it has not earned it. The judge should be demoted from oracle to advisor: its verdict becomes one input among several, and every change is gated instead by a deterministic verification layer the judge cannot override. We reached this position by building the alternative and running it. Over months of running autonomous prompt-optimization loops in production across contract analysis, compliance review, and code quality, we cataloged eleven ways the evaluation signal failed, in four classes: judge bias, harness and metric failures, ground-truth errors, and reward hacking. Agents achieved perfect scores by reading cached answer keys from their environment, a 100% pass rate concealing 68% true capability. A corrupted ground-truth label caused the optimizer to delete correct compliance rules to agree with it. A syntactically broken prompt was promoted as the winner because a silent parser fallback improved the metric. Attempts to fix the judge by rewriting its rubric plateaued; the only reliable gain came from a structural constraint on its output order. In response we describe PROCTOR, a Teacher-Student loop in which a stateful orchestrator holds all tool access, stateless subagents diagnose failures and draft mutations they cannot apply, and a Teacher grades those mutations under five deterministic guardrails: hermetic sandboxes, capability-disjoint roles, acceptance checks that outrank the Teacher, frozen holdouts, and canary cases engineered so that a perfect score is itself evidence of cheating. We report the failures this prevented, and, because the Teacher is itself an LLM judge, the failures it did not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02244v1">Task-Level Natural Language Priors as Learning Signals for Low-Resource LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle when low-resource training data are ambiguous or incomplete. Task-level natural-language priors can provide useful guidance in such settings, but existing approaches usually treat these priors as input context rather than as learning signals during training. We propose Prior-Guided Tuning (PGT), a training perspective that incorporates natural-language priors as auxiliary learning signals for low-resource LLM training. Under this perspective, we introduce Contrastive Prior Steering (CPS), which keeps the original supervised objective intact while adding positive and negative prior-conditioned auxiliary losses to encourage task-consistent learning and discourage plausible but misleading alternatives. Experiments on AmbiMath, Jigsaw, and MNLI/HANS show that CPS consistently improves over plain and prompt fine-tuning. On AmbiMath, CPS achieves 97.6% average exact-match accuracy. On Jigsaw, CPS improves average Macro F1 by 9.5 percentage points over standard fine-tuning, and with 1/10 of the experimental training data slightly exceeds full-data plain fine-tuning. On HANS, CPS improves non-entailment accuracy by 8.3 and 5.2 percentage points for LLaMA 3.1 8B and Qwen 2.5 7B, respectively, while maintaining comparable in-domain MNLI accuracy. These results support our central claim: task-level natural-language priors can provide useful guidance as auxiliary learning signals for low-resource LLM training. Our code and data will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00224v2">QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted by EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Weight-only post-training quantization (PTQ) can alleviate the computational burden of serving large language models (LLMs) at scale. However, existing PTQ methods often fail to generalize across models and suffer severe accuracy loss below 2 bits. Many leverage unstructured sparsity to mitigate this loss, but at the cost of regularity and GPU-friendly execution. We present QTEA, a sub-2-bit PTQ framework that quantizes weights into ternary values and uses salient weights as residual error compensators. To maintain hardware efficiency, residuals are assigned to selected columns with semi-structured $1:4$ sparsity within the salient columns. We further add column-wise rescale refinement to GPTQ-style column-by-column quantization, alternately updating per-column scales and ternary assignments to reduce reconstruction error. We also identify order-dependent error propagation in GPTQ and introduce error decay to attenuate late-stage error accumulation. On Qwen3-14B, QTEA compresses all weights to an effective 1.7 bits per weight while improving average accuracy over the strongest ternary PTQ baseline by 16.7%. It also achieves 1.40$\times$ and 2.61$\times$ lower perplexity on WikiText and C4 respectively. This trend holds on Llama3-8B, where QTEA obtains a 6.6% accuracy gain and 1.34$\times$ / 1.95$\times$ lower perplexity on the same datasets. Finally, we develop a lookup-table based kernel that achieves 7.2$\times$ faster per-token generation over an FP16 baseline. Code is available at https://github.com/Intelligent-Microsystems-Lab/QTEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02177v1">WeaveMark: Robust and Scalable Multi-bit LLM Watermarking via Coded Payload Spreading</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 16 pages, 11 figures. Code: https://github.com/qkrrkd90-source/WeaveMark
    </div>
    <details class="paper-abstract">
      Multi-bit watermarking for large language models (LLMs) enables content source tracing by embedding user-identifiable messages into generated text. Existing methods face a fundamental trade-off among extraction accuracy, text quality, and payload capacity. We propose WeaveMark, a robust and scalable multi-bit LLM watermarking scheme based on coded payload spreading. WeaveMark shifts this trade-off frontier by improving payload capacity through multi-bit-per-token spreading, improving extraction accuracy through soft-decision error-correcting code, and preserving text quality through unbiased multilayer reweighting. It further introduces dedicated zero-bit layers for reliable watermark presence detection. Experiments show large gains, especially for long messages and edited text. WeaveMark achieves 89.8% match rate for 32-bit messages at 200 tokens, compared with 20.8% for BiMark. Under 10% substitution attacks on 16-bit messages at 200 tokens, it maintains 86.0% versus 30.7%, while preserving text quality. Our code is available at https://github.com/qkrrkd90-source/WeaveMark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02168v1">FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Fragmented safety evaluation undermines the governance of dangerous AI capabilities. We present a modular framework that evaluates each model through three orthogonal pipelines---Knowledge ($K$), Defense ($D$), and Harm ($H$)---under a unified protocol, aggregating results into a standardized dangerous-capability profile $φ$. Pluggable modules supply scenario seeds, knowledge banks, hazard queries, and judge rubrics, while the core evaluation engine remains unchanged across domains; the CB evaluation is complemented by a cyber pilot demonstrating protocol transfer. Instantiating the framework with a chemical-biological (CB) module, we evaluate 12 commercial LLMs from four families. Our first contribution is a horizontal comparison of dangerous capability across models and model families: the three dimensions expose sharply divergent profiles---models with comparable knowledge differ in refusal resilience, and strong defenders do not generate less harmful content when they do comply---while family-level patterns further separate Claude, DeepSeek, and GPT models. The second is a temporal analysis of capability evolution: tracking $K$, $D$, and $H$ against model release dates reveals that dangerous capability has not monotonically declined; newer models deepen knowledge while only partially improving defense, showing that scaling and alignment progress do not uniformly translate into safety. Reliability is established via cross-judge consistency (bootstrap $ρ> 0.79$, 4 of 5 judges) and pipeline orthogonality ($K$--$D$--$H$ inter-correlations $ρ\in [0.32, 0.52]$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29920v2">Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      Rubric-based scoring has become a widely used paradigm in model evaluation, typically with LLM-as-a-Judge (LaaJ) for rubric scoring. However, the reliability of LaaJ for rubric scoring remains underexplored. This concern is especially pronounced in agentic scenarios, where long, complex outputs further challenge reliable scoring. To address this, we conduct a systematic meta-evaluation of LaaJ reliability for rubric verification. We introduce RuVerBench, the first benchmark for assessing LaaJ reliability in rubric verification for agentic scenarios. RuVerBench covers two prevalent agentic domains, deep research and agentic coding, with 2,458 instances, each containing a model-generated output, a rubric, and a human-annotated label indicating whether the output satisfies the rubric. Using RuVerBench, we evaluate numerous frontier LLMs and find that even the most advanced models achieve strong performance but still exhibit substantial noise. We further analyze the impact of key LaaJ strategies, including prompt design, batching, and majority voting, on rubric verification. We find that weaker models are more sensitive to prompt variations, batched verification presents a trade-off between accuracy and efficiency, and majority voting yields effective but diminishing returns. We have released our dataset and code to facilitate future research: https://github.com/THU-KEG/RuVerBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07468v2">Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Memory enables Large Language Model (LLM) agents to perceive, store, and use information from past dialogues, which is essential for personalization. However, existing methods fail to properly model the temporal dimension of memory in two aspects: 1) Temporal inaccuracy: memories are organized by dialogue time rather than their actual occurrence time; 2) Temporal fragmentation: existing methods focus on point-wise memory, losing durative information that captures persistent states and evolving patterns. To address these limitations, we propose Temporal Semantic Memory (TSM), a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory. During memory construction, it first builds a semantic timeline rather than a dialogue one. Then, it consolidates temporally continuous and semantically related information into a durative memory. During memory utilization, it incorporates the query's temporal intent on the semantic timeline, enabling the retrieval of temporally appropriate durative memories and providing time-valid, duration-consistent context to support response generation. Experiments on LongMemEval and LoCoMo show that TSM consistently outperforms existing methods and achieves up to 12.2% absolute improvement in accuracy, demonstrating the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.27309v2">Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 15 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa $[-0.167, +0.353]$, $p = 0.684$). The audit's one nominally significant interaction, $+0.378$ ($p = 0.002$), is not identified as preference: a construction containing zero differential preference reproduces 79 to 85\% of it from the observed severity shift and the scale floor alone. We derive the mechanism in closed form and show that its contribution is measurable from an audit's own ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02094v1">MASkills: Continual Skills Optimization for Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems have shown strong performance on complex tasks, yet continual improvement from interaction experience remains challenging. Existing self-reflection methods build experience memories, but memories are mostly hard to invoke, refine, or scale, while agent skills offer a more actionable unit: structured procedural knowledge that specifies when to act, how to act, and which resources or tools to use. We introduce MASkills, a continual learning framework that optimizes multi-agent LLM systems through agent skills. MASkills presents a new agent-optimization pipeline that integrates skill-conditioned credit assignment, hierarchical credit aggregation, and momentum-smoothed optimization, enabling agent skill libraries to evolve through refinement, induction, consolidation, and pruning. Experiments on HotpotQA, LoCoMo, and GAIA demonstrate the effectiveness of MASkills across multiple agentic tasks. Our code is available at https://github.com/DaRL-GenAI/MASkills
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02093v1">Compositional Spectral Prompts for LLM-based Online Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 CIKM 2026
    </div>
    <details class="paper-abstract">
      To address the sequential and evolving nature of time series, the Online Time Series Forecasting (OTSF) task has been extensively studied in multiple domains. Existing research focuses on adapting to non-stationary environments by employing memory buffer-based retrieval strategies. However, we observe that such frameworks struggle with long-term adaptation and fail to generalize to unseen patterns. To this end, we introduce CoSPOT, an LLM-based online time series forecasting framework that leverages a pre-trained LLM as the backbone online forecaster, motivated by its strong few-shot capabilities. For efficient online adaptation, CoSPOT keeps the LLM frozen and employs compositional spectral prompts grounded in frequency-domain bases to guide the model with the overall distribution of the input, thereby substantially reducing the number of parameters updated during the online phase. Specifically, CoSPOT decomposes time series into frequency bases and composes the corresponding spectral basis prompts according to their amplitudes, allowing unseen patterns to be represented as new combinations of learned basis prompts. Our extensive experiments on real-world datasets demonstrate the superiority and practicality of CoSPOT across challenging online scenarios, including extended online phases and cross-dataset settings with substantial distribution shifts. Our code is available at https://github.com/seungyoon-Choi/CoSPOT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02092v1">Beyond Outcome Gaps: Process-Aware Fairness Diagnosis for LLM-based Multi-Agent Decision Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) are increasingly considered for high-stakes decision-making, yet outcome-based fairness audits can miss where risks arise within the decision trajectory. We present SCOPED-Hiring, a process-aware fairness diagnosis pipeline for LLM-based hiring MAS. SCOPED-Hiring constructs controlled resume variants, runs role-based hiring committees, logs over 311K structured decision trajectories, and converts trajectory fields into quantitative fairness signals organized by six diagnostic lenses: final outcome, counterfactual, process, pathway, dynamic, and design effects. SCOPED-Hiring reveals that balanced final hire rates can mask hidden trajectory unfairness in multi-agent decision trajectories: career gaps trigger suspicion, proxy cues shape qualification judgments, and identity cues lead to unequal investigation. Targeted repair guided by these diagnoses reduces total layered burden by 72.3% while shifting the hire rate by only 1.86 pp, showing that process diagnosis can guide effective repair. Project Page: https://scoped-hiring-project-page.vercel.app/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08552v3">Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Descriptive scientific metadata in public repositories are often incomplete and inconsistent with community standards and ontologies, limiting data FAIRness. Large language models (LLMs) offer a promising approach to automatically standardizing such metadata when provided with relevant standards in machine-actionable form, such as metadata templates from the CEDAR Workbench. Prompt engineering, however, provides only fixed snapshots of these standards and relies on an LLM's pretrained knowledge to interpret and satisfy their constraints. We evaluate whether giving an LLM access to metadata specifications and authoritative terminology at runtime improves automated metadata standardization. Methods: We present ARMS, a tool-augmented LLM agent that retrieves complete CEDAR metadata templates and dynamically queries authoritative biomedical terminology services at execution time. We compared ARMS with a prompt-based approach on 839 legacy metadata records from the Human BioMolecular Atlas Program (HuBMAP), using expert-standardized records as the reference standard. Results: ARMS outperformed the prompt-based approach, increasing precision from 0.56 to 0.93 and recall from 0.51 to 0.85, with improvements across all field categories and assay types. The largest gains occurred for ontology-constrained fields, where precision increased from 0.36 to 0.92. Conclusion: LLMs cannot convert legacy metadata to standards-adherent form without knowledge of the relevant standards. ARMS improves metadata standardization by providing runtime access to authoritative resources that define valid metadata. Machine-actionable metadata standards enhance LLM-based rectification of legacy metadata, especially when they can be queried dynamically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16709v2">Covert Multi-bit LLM Watermarking: An Information Theory and Coding Approach</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      We study the problem of multi-bit watermarking for non-autoregressive large language models (LLMs). We introduce an information-theoretic model inspired by diffusion language models (DLM), in which the encoder has limited non-causal access to token distributions within each token block. This formulation enables an information-theoretic characterization of the non-causal watermarking capacity, in which knowledge of LLM cover statistics is leveraged to enable a multi-bit covert embedding. We study the information-theoretic limits of the model by combining Gelfand--Pinsker and channel synthesis coding techniques and obtain an exact characterization of the capacity. The embedding strategy is further optimized across blocks using a constrained Markov decision process (CMDP) and we develop an explicit algorithm based on polar codes following the information-theoretic principles. We simulate the error performance on LLaDA, and provide empirical total variation (TV) analysis as a function of key randomness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02824v3">CALIBURN: Self-Calibrated LLM Unlearning Alignment</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 EMNLP 2026
    </div>
    <details class="paper-abstract">
      LLM unlearning aims to remove the influence of undesirable knowledge from pretrained language models, which offers a practical mechanism for addressing safety and privacy concerns. Existing unlearning approaches, such as Gradient Ascent, are prone to catastrophic forgetting. Alignment-based approaches provide an alternative direction, yet their effectiveness is limited by the quality of the reference model. In realistic settings, both methods still require large retention datasets to preserve general knowledge. We propose a principled method that quantifies the target LLM's confidence in undesirable knowledge and uses it to calibrate the model's unlearning gradient updates more precisely. It enables fine-grained control over forgetting while better preserving model utility, thus reducing the dependence on retention data or prohibitive unlearning training data. Extensive evaluations on multiple benchmarks, including MUSE and WMDP, show that our method achieves effective unlearning and improves the trade-off between knowledge removal and utility preservation compared with state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04547v3">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, avoiding the serialization of user histories into prompts and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating scalable personalization without prompt-serialized histories or heavy per-user adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02042v1">Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 EMNLP 2026 Camera Ready
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents for long-horizon interactive tasks typically follow a ReAct-style protocol, issuing one primitive action per LLM round. While this enables frequent replanning, it is inefficient for long-horizon tasks where many rounds are spent on routine action sequences. A natural alternative is to let the agent emit variable-length action chunks. However, naively training such policies with standard reinforcement learning fails: the agent either collapses to single-action behavior or over-commits to excessively long sequences. Both failures share a common root cause: the inability to learn chunk boundaries. We propose SPACE, which addresses this challenge by distilling chunk-boundary supervision from trajectory-induced programmatic skills. We induce two-level programmatic skills from successful trajectories, where subskill boundaries serve as direct chunk-boundary supervision. This temporal structure is then distilled into a primitive-chunk policy via hybrid on-/off-policy optimization with chunk-aware credit assignment. Experiments on ALFWorld and ScienceWorld show that SPACE improves success rates by 7.0%-31.3% over the strongest baseline in each setting while reducing average LLM decision rounds by up to 78.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00187v2">Ordinal Gates, Cardinal Bets: Matching LLM Confidence to the Financial Decision Operator</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      LLM confidence scores are not independently deployable objects: their decision value depends on the downstream operator and exposure controller that consume them. Monotone recalibration cannot change a coverage-matched rank-based gate, whereas position sizing consumes score magnitude, so changing a confidence map can invalidate a scale fitted to the previous score distribution. We test this on FactSet news for Nasdaq-100 equities, fitting maps and scales on 2021 and evaluating nine open-weight LLMs out-of-sample on 2022--2023. Cross-applying raw and correctness maps with independently fitted scales shows that the two components are not portable alone: scale transfer reduces certainty-equivalent return (CER) in $8/9$ models and produces large risk-target errors. Matching each map with its fitted scale improves ensemble CER by $9.2$ percentage points per year under frozen-scale control ($p<0.001$), and the effect remains significant when the single largest-contributing model is excluded ($+5.5$pp/yr), so it is not driven by one case. Under an identical adaptive-volatility controller, however, the incremental effect falls to $+1.6$pp/yr, with a significant controller interaction. Annual walk-forward effects are smaller, although map--scale interaction remains positive in every fold. Confidence transformations should therefore be evaluated jointly with the downstream controllers that consume them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02035v1">Implicit Manipulation for Skill Selection in LLM Agents with Semantic Matching</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 20 pages, 9 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Skill selection is a key stage in LLM-agent workflows, determining which installed skill should handle a user request. Existing attacks on this stage primarily rely on explicit prompt injection or instruction-level steering, which can expose recognizable manipulation signals. In this work, we identify a new implicit attack surface for skill selection: even when the user prompt and skill description appear benign in isolation, their semantic relationship can still be strategically shaped to favor an attacker-chosen skill. Based on this observation, we present Implicit Skill-Selection Manipulation via Semantic Matching (ISM), which jointly shapes target-skill metadata and reusable prompts to manipulate skill selection without explicit selection instructions. Specifically, we develop a three-stage strategy to broaden semantic coverage, strengthen target distinctiveness, and preserve natural prompt wording. Across four task domains and eight selector models, ISM increases the average target-selection rate (TSR) from 15.2% to 63.5%. In a matched comparison, ISM achieves a 73.5% TSR, only 9.8 percentage points below Explicit Steering. Human reviewers block ISM in only 2.9% of judgments, versus 91.4% for Explicit Steering, while five LLM-based inspectors pass ISM at an average rate of 82.9%, versus 37.4% for Explicit Steering. Moreover, ISM remains effective against PPL-W, Llama Prompt Guard 2, and PIGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02027v1">Multi-Turn LLM Conversations under the Least-Recently-Used Policy: Mean-Field Asymptotics and Hit Ratio Approximation</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      The major workloads in modern large language model (LLM) serving systems have shifted from single-shot LLM calls to multi-turn conversations, where new responses are generated based on the whole conversation history across all previous turns. The hit ratio, i.e., the average fraction of KV caches accessed directly from existing caches stored in high-bandwidth memory (HBM), is hence a crucial metric that governs system performance. Estimating the hit ratio is a highly nontrivial task due to the complex system dynamics, where the KV cache prefixes grow with turns and some must be evicted due to finite memory capacity. We formulate the system as a multi-turn conversation model under the least-recently-used (LRU) policy. Through a mean-field asymptotic framework, we prove that as the conversation arrival rate and the memory capacity grow proportionally to infinity, the hit ratio converges to a closed-form limit. Based on the characterization of the limit, we further propose a practical hit ratio estimator, and validate its accuracy by real LLM serving experiments on the Qwen3-8B model implemented on Ascend NPUs. Our results provide a theoretical foundation for the analysis of multi-turn LLM serving systems and a practical guideline for memory capacity provisioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08590v2">GMTRouter: Personalized LLM Router over Multi-turn User Interactions</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to Findings of the Association for Computational Linguistics: EMNLP 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) routing has demonstrated strong capability in balancing response quality with computational cost. As users exhibit diverse preferences, personalization has attracted increasing attention in LLM routing, since even identical queries may require different models to generate responses tailored to individual needs. However, existing approaches are not fully personalized and often fail to faithfully capture the complex interactions between users and LLMs. Moreover, user preference data is typically scarce and inconsistent in format, which limits the effectiveness of methods that directly leverage user-specific data. To address these challenges, we propose GMTRouter, which represents multi-turn user-LLM interactions as a heterogeneous graph with five node types: user, LLM, query, response and turn, thereby maximally preserving the rich relational structure of the interaction. Through a lightweight inductive graph learning framework combined with a tailored user-conditioned graph sampling mechanism, GMTRouter learns to capture user preferences from few-shot data, enabling effective personalization. Extensive experiments demonstrate that GMTRouter outperforms the strongest baselines, achieving up to a 0.108 absolute improvement in accuracy and a 0.124 improvement in AUC. More importantly, we further demonstrate that GMTRouter can adapt to new users using only few-shot data, without extensive fine-tuning. The code for GMTRouter is publicly available at https://github.com/ulab-uiuc/GMTRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05616v2">What's in a Name? Morphological Shortcuts by LLMs in Pharmacology</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to the Main Conference of EMNLP 2026
    </div>
    <details class="paper-abstract">
      The morphological form of a word can often give cues to its meaning, but purely relying on these mappings can lead to overgeneralization in high-stakes domains. In the medical domain, for instance, LLMs can confidently reason about fictitious drugs from their affixes alone (e.g., wugcillin) and generate plausible-looking clinical content. We present a behavioral and mechanistic study of LLM "affix heuristics" in pharmacology. Using fictitious drug names built from real affixes, we show that affix signals alone elicit class-level pharmacological responses. We introduce a framework for identifying whether a model's drug semantics are driven mainly by the affix, the stem, or the drug name as a whole. Applied across 653 drugs, our framework reveals that models often induce drug meaning primarily through affix cues, yet rarely explicitly indicate this reliance, and sometimes incorrectly conflate properties among affix-sharing drugs. Activation patching across models further localizes this behavior to early-mid layers. These findings show that morphological shortcuts pose a subtle but measurable risk to safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11557v3">UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 25 pages, 10 figures, 17 tables. Code and datasets are publicly available at: https://github.com/EIT-NLP/UniToolCall
    </div>
    <details class="paper-abstract">
      Tool-use capability is a fundamental component of LLM agents, enabling them to interact with external systems through structured function calls. However, existing research exhibits inconsistent interaction representations, largely overlooks the structural distribution of tool-use trajectories, and relies on incompatible evaluation benchmarks. We present UniToolCall, a unified framework for tool learning that standardizes the entire pipeline from toolset construction and dataset generation to evaluation. The framework curates a large tool pool of 22k+ tools and constructs a hybrid training corpus of 390k+ instances by combining 10 standardized public datasets with structurally controlled synthetic trajectories. It explicitly models diverse interaction patterns, including single-hop vs. multi-hop and single-turn vs. multi-turn, while capturing both serial and parallel execution structures. To support coherent multi-turn reasoning, we further introduce an Anchor Linkage mechanism that enforces cross-turn dependencies. Furthermore, we convert 7 public benchmarks into a unified Query--Action--Observation--Answer (QAOA) representation with fine-grained evaluation at the function-call, turn, and conversation levels. Experiments show that fine-tuning Qwen3-8B on our dataset substantially improves tool-use performance. Under the distractor-heavy Hybrid-20 setting, achieves 93.0% single-turn Strict Precision, outperforming commercial models including GPT, Gemini, and Claude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01976v1">Knowing Is Not Enough: Information Retrievability as a Precondition to Effective LLM Oversight</a></div>
    <div class="paper-meta">
      📅 2026-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in organizational work, yet their errors often pass human review. Prior research locates such failures in users' capability to review LLM output or their engagement in doing so. We develop an alternative, retrieval-based account of human oversight and posit that error detection is more effective when oversight-relevant information is accessible to users at the moment of review. Across two randomized lab-in-the-field experiments with 640 customer-facing employees, we show that self-generated explanations improve error detection and strengthen recall of verification-relevant reasoning, while cues that reactivate such reasoning help sustain detection under repeated LLM use. Theoretically, we identify information retrievability as a distinct precondition for effective oversight and specify generative encoding and cue-supported reactivation as mechanisms that build and sustain it. Practically, lightweight onboarding self-explanations and daily retrieval cues can make human oversight more resilient as LLM use becomes routine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01971v1">NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis</a></div>
    <div class="paper-meta">
      📅 2026-09-02
      | 💬 Accepted to Findings of EMNLP 2026. 20 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      AI is rapidly advancing neuroscience, yet many laboratories fail to fully unleash its potential due to significant interdisciplinary barriers. While pre-trained neural models for physiological data are progressing quickly, their heterogeneous architectures and modality-specific constraints hinder systematic integration, selection, and evaluation. Despite recent advances in large language model (LLM)-based agent systems for intelligent scientific applications, existing approaches often still lack the domain expertise required to effectively select and coordinate diverse neuroscience pre-trained models and handle unique data types in this domain. We present NS-Copilot, an LLM-driven multi-agent system for neuroscience analysis that autonomously supports end-to-end workflows for diverse professional tasks. It unifies domain-specific pre-trained models and supports key neuroscience modalities, including EEG and extracellular spike data, through a natural-language interface. Given raw data and a task description, NS-Copilot orchestrates agents with specialized roles for planning, adaptive control, code generation, and result synthesis, enabling analysis without dataset-specific heuristics. We evaluate NS-Copilot on neuroscience benchmarks spanning Alzheimer's disease, Parkinson's disease, and working memory spike decoding. Across 8 trials per task, the system consistently outperforms strong baselines on the primary metric, demonstrating the ability of NS-Copilot for effective and scalable neuroscience analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.27463v2">Rating the Raters: Rasch Measurement Theory for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      LLMs now sit on every side of evaluation: as examinees scored on benchmarks, judges of other models' outputs, and raters of human-generated content. Each paradigm can be viewed as a measurement problem, where a latent property of an object is probed with items from an instrument (e.g., benchmark) by judges or raters. Standard evaluation practices often neglect the contributions of each core component to the end result, limiting our understanding of what is being measured. Rasch measurement theory (RMT) is well-suited to this problem. RMT decomposes ordinal ratings into separable facets on a common scale. It further provides a battery of diagnostics that can identify miscalibrated measurements and rater biases. We present a case study of RMT applied to the LLM-as-rater paradigm using the Measuring Hate Speech corpus, whose construct was itself built under RMT. We fit a series of many-facet Rasch models to annotations from nine LLMs spanning families and capability levels. Our analyses show that LLMs systematically differ from human raters in severity, item-level calibration, question-order robustness, target-identity sensitivity, and rating scale use, all of which standard evaluation practice would largely obscure. Overall, we argue that RMT belongs in the toolkit for evaluating LLM-as-examinee, -judge, and -rater paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01918v1">Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 9 pages, 2 figures, 4 tables. Accepted to the 5th Workshop on NLP for Positive Impact (NLP4PI) at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Energy poverty is nearly absent from NLP-for-social-good, and the little existing work is either static retrieval/QA or relies on carbon-intensive cloud LLMs, a self-defeating "computational irony" for a humanitarian setting. We present EqGrid, a closed-loop simulation in which a low-frequency, open-weight LLM policy agent sets price and carbon bounds and targeted subsidies over a community of empirically-grounded household personas, while high-frequency multi-agent RL traders clear a continuous double auction constrained by a physical distribution grid (IEEE-33-bus with Dynamic Operating Envelopes). Our contribution is threefold and directly addresses how to measure the social impact of AI: (i) grounded personas (region-matched socio-demographics) whose load curves are checked for shape and level realism against real smart-meter data; (ii) formal energy-poverty equity metrics (Energy Burden, Gini of EB, LIHC) showing the intervention reduces burden inequality without raising net grid cost; and (iii) a compute-efficiency frontier that measures how much equity performance survives compressing the policy agent from a 235B teacher down to a sub-1B model deployable on a laptop, in estimated energy/carbon per decision. A decoupled-safety design (the LLM sets bounds; a validate-and-project grid gate executes) yields zero grid-constraint violations versus 55 under direct LLM control. On energy-poverty equity, the LLM policy lowers the Gini of energy burden to 0.305 (from 0.351) and mean burden by 28% while cutting cost (outperforming a tuned rule baseline), and a 3B-active model retains 95% of the benefit at roughly 9x lower inference energy than the teacher, with even a 0.8B on-device model retaining 92% at roughly 24x lower energy. We will release code and configs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01905v1">Guiding LLM Peer Reviewers: The Impact of Score Anchors on Review Evidence and Accuracy</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for research quality evaluation, with prior work exploring their scoring accuracy and the plausibility of review rationales. However, less is known about whether external score guidance changes the evidence presented in the generated review as well as the final score. This study uses 98 Allied Health Professions research outputs submitted for internal REF-style assessment, with specialist human review reports and adjudicated 1-4 reference scores. No-guidance baseline reviews are compared with oracle-guided reviews, where the supplied score is set to the rounded human reference score; extracted evaluation points are used to compare human and LLM evidence use. Using this design, oracle guidance improves scoring accuracy, with score-following checks showing that models do not simply copy the supplied score. Corrected score mismatches are associated with changes in the generated review frame, showing that the score signal can steer review rationales. This effect is direction-dependent: LLM reviews cover human strength or upgrade points more reliably than human weakness or downgrade points, with the weakest alignment for expert downgrade evidence. The results show that score-guided review generation can be evaluated at the level of review evidence, as well as the final score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01902v1">Accurate in space, unreliable in time: how LLMs represent national cultural change</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 38 pages, 12 figures, 36 tables; includes Supplementary Materials
    </div>
    <details class="paper-abstract">
      Assessments of cultural alignment have become an important part of the development and improvement of large language models (LLMs). However, the majority of the evaluations treat culture as a single snapshot, investigating only whether a model represents a society accurately at the current time. Research in cultural psychology shows that cultural values change at different rates and directions over time. Therefore, a "culturally aware" model should capture not only where a culture is today but also how it has changed over time. We examine this missing dimension of cultural awareness using more than two decades of the World Values Survey data. We compare the cultural trajectories of 40 countries with the trajectories produced by four state-of-the-art (SOTA) LLMs on the Inglehart-Welzel cultural map. Our findings show that while models generally place countries close to their most recent surveyed positions, these representations tend to lag several years behind that position. They also capture only part of the magnitude of the observed change, introduce movement where little occurred, and rarely reproduce reversals in countries' trajectories. These findings point to temporal flattening and suggest that snapshot accuracy can give an incomplete picture of cultural awareness in LLMs and have implications for model evaluation, representational harms, and the governance of culturally aware AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21815v2">HarmReduction: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 15 pages, 5 figures, 12 tables, a dataset
    </div>
    <details class="paper-abstract">
      Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy provides non-judgemental, evidence-based information intended to improve health outcomes and reduce associated safety risks. Some large language models (LLMs) have demonstrated a high level of medical reasoning, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HarmReduction, a benchmark designed to evaluate LLMs' accuracy and safety risks in harm reduction information provision. The benchmark dataset (HR-Basic) has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, present severe safety risks to PWUD. This work contributes an evaluation framework for LLMs to deliver harm reduction information to avoid introducing negative health outcomes through the use of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01895v1">Beyond Human-Likeness: Mapping the Scientific Critique Profiles of LLMs and Human Reviewers</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly discussed as tools for peer review, but their value is often assessed through human-likeness, perceived usefulness, or textual overlap with reviewer comments. This study shifts attention from whether LLMs resemble human reviewers to what functions of scientific critique they perform. Using ICLR 2025 peer-review data, we compare human reviews with LLM reviews generated under baseline and expert prompts. We operationalize scientific critique through two review acts, weakness critique and scientific questioning, and annotate point-level review text using five theory-guided frameworks: Anderson's knowledge types, Toulmin's argumentation model, Graesser's question depth, SOLO cognitive complexity, and Hattie's feedback functions. The results reveal a differentiated critique profile. Human reviews placed greater emphasis on scientific framing and revision guidance, more often identifying higher-order weaknesses and asking questions oriented toward improvement. LLM reviews showed higher rates of explanatory depth, integrative reasoning, and explicit argument structuring. Expert prompting did not make LLM critique uniformly more human-like; it partially narrowed some gaps but mainly amplified LLM-specific tendencies toward integration and formal argumentation. These findings show that LLM-assisted peer review changes the functional composition of review text, making it important to distinguish LLM-amplified critique from areas requiring human prioritization and accountable judgement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17193v2">Multi-LLM Systems Exhibit Robust Semantic Collapse</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 93 pages, 12 figures, 12 tables; includes Supplementary Information
    </div>
    <details class="paper-abstract">
      Whether machines can originate novel content has been debated for nearly two centuries, from Lovelace's assertion that no engine can "originate anything" to Turing's question of whether a machine can amplify ideas brought in from outside. Systems of multiple interacting LLMs, increasingly deployed for autonomous generation, reopen this question empirically. Here we show that such systems, operating in inference-only setups, exhibit semantic collapse: systematic convergence in semantic representations despite apparent lexical variation. Across model families, extended simulations of 200 to 1,000 rounds, the pattern remains consistent. Thirteen intervention strategies, spanning decoding parameters, prompt design, agent composition, activation engineering, and reinforcement learning, fail to restore semantic diversity. Mechanistic analyses suggest that semantic collapse is not explained by alignment or conformity biases, but is consistent with intrinsic properties of autoregressive generation. Our results point to persistent constraints on the ability of multi-LLM systems to sustain open-ended exploration in closed-loop settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02947v1">Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 16 pages, 3 figures, Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Significant challenges remain in AI-driven educational systems in balancing privacy preservation with accurate cognitive diagnosis. To overcome this, we propose a federated inference framework in which several commercial LLM APIs collaborate without requiring access to raw student data or proprietary model internals. Using multiple federated entities, such as LLaMA-3.3-70B, GPT-4o-mini, and Claude-3-Haiku, our framework builds upon a heterogeneous multi-LLM architecture. The predictions generated by these entities are combined with epsilon-local differential privacy by adding Laplace noise locally to each entity's prediction output before aggregation, while residual-based aggregation mitigates model heterogeneity. Our approach is predicated on an honest-but-curious trust paradigm in which API providers are presumed not to abuse submitted queries, and our differential privacy mechanism shields the published diagnostic results from external inference. We conduct rigorous privacy-utility analysis showing strong privacy guarantees with minimal accuracy loss, and extensive real-world evaluations across three educational benchmarks confirm the framework's practical usability and cross-domain generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01834v1">Architecting Conversational Data Systems for Stateless LLM APIs: The Hydration Proxy Pattern</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 3 pages, 1 table. Presented at the SAO workshop at the 1st ACM Conference on AI and Agentic AI Systems (ACM CAIS 2026)
    </div>
    <details class="paper-abstract">
      As enterprise platforms transition to conversational reasoning interfaces, the stateless nature of LLM APIs creates an architectural gap. While statelessness enables horizontal scalability for AI providers, it forces client applications to manage the entire burden of conversational state and semantic memory. The work identifies the Hydration Proxy Pattern, an architecture that decouples session persistence from the reasoning engine. The framework ensures platform sovereignty over conversational data while enabling secure, multi-stage semantic grounding. We further propose the Context Stabilization Mandate to resolve the tradeoff between sovereign state management and KV caching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01798v1">How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to the EMNLP 2026 Main Conference; camera-ready version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on mobile devices, making energy efficiency a key deployment constraint, yet the energy impact of prompt design remains underexplored. This paper aims to understand how two prompt properties, cognitive load and phrasing pattern, shape the energy behavior of on-device LLM inference. We conduct a broad empirical study covering prompt properties, datasets, models, and devices, with phase-level profiling that separates prefill and decode energy. We find that cognitive load primarily affects the energy cost per token, while phrasing pattern affects energy largely through token usage. Our energy-quality analysis further shows that prompt design reshapes the attainable frontier differently across models, highlighting the need for model-aware prompt design in energy-efficient on-device LLM inference. Code, datasets, and scripts are available at https://amai-gsu.github.io/PromptProperty/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01788v1">VakyArth: Evaluating Pragmatic Competence in LLMs across Indic Languages</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Real-world communication often requires pragmatic reasoning: interpreting meanings implied through context and cultural convention rather than stated literally. Existing pragmatic evaluation remains largely limited to English and high-resource languages, leaving Indic languages unexplored despite their linguistic and cultural diversity. We introduce VakyArth, the first pragmatic benchmark for Indic languages, designed as a diagnostic evaluation covering Hindi, Punjabi, Tamil, and Malayalam. VakyArth evaluates models across five phenomena: deixis, speech acts, implicature, social pragmatics, and coherence; through multiple-choice questions, natural language inference, and translation, with all items authored by native speakers. Across multilingual large language models (LLMs) of varying families and sizes, we find consistent failures on pragmatic meanings rooted in Indic linguistic and cultural conventions. Our analysis shows systematic differences across languages and tasks: MCQ accuracy exceeds NLI accuracy in all model-language combinations, translation performance does not reliably track pragmatic understanding, and Indo-Aryan languages show a translation advantage over Dravidian languages. We further show that automatic translation metrics can miss fluent but pragmatically unfaithful outputs, especially for implicature and deixis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01758v1">Towards Behavior Tree-Guided Vulnerability Detection with Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for software vulnerability detection, but their performance depends on how source code is represented in the input. Most prompting approaches use source code in its original form, while some works propose the use of structured representations. Abstract Syntax Trees (ASTs) are one of the most popular approaches, but AST verbosity increases input size relative to source code, making them hard to fit within some LLMs context windows. This paper investigates Behavior Trees (BTs) as an alternative intermediate representation for LLM-based vulnerability detection. BTs encode control flow, conditions, and executable actions more compactly than ASTs, making them a natural candidate when token count is a constraint. First, we propose a preprocessing stage that parses Java source code into ASTs and then converts them into BT representations. We then compare vulnerability detection performance across 460 Java samples from the Juliet Java test suite, using three input representations: raw source code, AST, and BT. All experiments use a single quantized local LLM, Mistral Small 3.2 24B (Q4_K_M). Our results show that using BT representations improves recall on short code samples, while raw source code achieves higher precision. On longer samples, BTs improve overall performance over the original representation and fit within the context window, whereas many ASTs exceed the context limit. These findings suggest that BTs can provide a compact and useful structured representation for vulnerability detection with quantized, locally deployable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01736v1">Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) augmented with external tools have demonstrated remarkable capability in solving complex real-world tasks. However, existing approaches suffer from two key challenges: brittle multi-step and multi-turn reasoning caused by incompatible tool output types and API schemas, and performance degradation under large tool catalogues. To address these, we introduce \textbf{Tool Primitives}, a design that replaces rigid API schema-based invocation with natural language as the interface for tool calling, where each tool is wrapped with an LLM interface that handles schema resolution and execution internally, enabling natural inter-tool communication for nested and multi-turn tool calling. Building on Tool Primitives, we host \textbf{ToolFace}, a centralized repository of 25,519 functions from which LLMs dynamically retrieve only the relevant tools at inference time, eliminating the need to enumerate raw API schemas in context. To orchestrate Tool Primitives and ToolFace reliably in complex settings, we further propose \textbf{HEART}, a \textbf{H}arness \textbf{E}ngineering framework via \textbf{A}gent-native, \textbf{R}eusable \textbf{T}ool Primitives, comprising a Planner, Router, and Verifier that jointly support dynamic tool invocation planning, multi-step execution, and feedback-driven recovery. Experiments on five benchmarks demonstrate that HEART outperforms SFT-based models by $10\%$ on average and surpasses GPT-5.4, Claude-4.6-Sonnet, and Gemini-3.1-Pro by $6\%$ on average while reducing API cost by up to $85\%$. On 50 real-world tasks, HEART achieves $84\%$ task completion, $3.8\times$ the average of three frontier commercial models ($22\%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.26391v2">Software Aging in LLM-Generated Applications: Runtime Evidence, Static Analysis, and Human-Written Comparisons</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate executable software systems from natural language specifications, accelerating development and reducing manual implementation effort. Although recent studies have investigated the functional correctness, security, maintainability, and robustness of LLM-generated code, little is known about the long-term reliability of such systems under sustained execution. In this paper, we experimentally investigate software aging symptoms in LLM-generated service-based applications across generation-and-execution environments. Using backend scenarios derived from BaxBench, we generated applications targeting JavaScript, Python, and Rust through LLM-based generation platforms, validated them with BaxBench-derived tests, and subjected them to 48-hour workload executions. We monitored memory usage, response time, and throughput and analyzed them using the Mann--Kendall test and Sen's slope estimator. We further complemented the runtime evaluation with static analysis of the generated source code and an exploratory comparison with human-written implementations of related backend scenarios. The results show that memory usage is the most consistent indicator of potential software aging, with statistically significant upward trends in most application-environment combinations, while response time and throughput exhibit more heterogeneous behavior. Static analysis identified plausible code-level aging mechanisms, and the comparison with human-written systems showed that the aging symptoms observed in LLM-generated applications align with degradation patterns also found in manually developed implementations. These findings indicate that functional correctness alone is insufficient to assess the operational reliability of LLM-generated software before deployment in continuously running environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01604v1">Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted at EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      LLM-based evaluators of natural language generation (NLG) quality are widely deployed as scoring tools and as automated training signals, yet the internal procedure by which they assign a rating remains poorly understood. We investigate this procedure mechanistically through an eight-attack perturbation taxonomy across the Readability and Adequacy dimensions of NLG quality, a generation pipeline that produces paired clean and corrupt summaries with controlled error intensity and explicit token-level modification maps, and a four-experiment battery of causal tracing, logit-lens vocabulary projection, and attention-head knockout applied to Themis (Llama-3-8B) and Prometheus (Mistral-7B). Both evaluators implement a structured, coherent evaluation pipeline operating in two stages: below layer 15, attention performs local error comparison and routes the result to the final input position; above it, the MLP cascade integrates the signal and writes the rating, with the decision crystallizing in the residual stream at a sharp late layer (L = 26 on Themis, L = 25 on Prometheus). Furthermore, a base-model control at the same scale (Llama-3-8B) reproduces the routing architecture and crystallization but not the stage separation, isolating the two mechanisms that fine-tuning specifically installs, suppression of below-L15 MLP contribution at the last position and a two-layer advance of the crystallization depth, indicating that fine-tuning sculpts an existing substrate rather than building the pipeline from scratch. We release the source code and data at https://github.com/himil-v/judge-mech
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01591v1">StudentSim: Training LLM-based Student Simulators</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      AI tutors are most useful when they adapt to each student's strengths, weaknesses, and preferred guidance, but evidence about which guidance works for which student is sparse, slow, and costly to collect from real learners. Student simulators can provide this signal as a proxy, yet existing approaches are limited: state-tracking models fit student behavior but struggle to process explanations or corrections, while LLM role-play follows guidance fluently but does not reliably match the competence of the student being imitated. We present StudentSim, a training framework that turns sparse per-student data into individualized simulators through pooled training followed by per-student specialization. The resulting simulators both mirror a student's own responses and update them under tutor guidance. We also introduce StudentSimEval, a standardized protocol covering 60 students across chess, second-language English writing, and mathematics, using public learner datasets with de-identified records shared for research. StudentSimEval measures behavioral fidelity (F), or how well a simulator matches a student's responses, and guidance responsiveness (R), or how readily it updates under tutor guidance, with all methods fit and evaluated on the same records. Across all three domains, StudentSim outperforms GPT-5.4 on both metrics. In chess, StudentSim reaches F=0.51 and R=0.91, compared with 0.23 and 0.72 for GPT-5.4 and 0.45 and 0.27 for Maia2. As a proof of concept, using StudentSim as a reward model for tutor reinforcement learning produces a chess tutor that expert humans rate as more accurate, better-guided, and more personalized than a no-RL baseline and a tutor trained against a GPT-5.4 simulator reward. Code is available at https://github.com/microsoft/StudentSim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01587v1">The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Preprint. Under review at a NeurIPS 2026 workshop. 11 pages, 4 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) is widely used to reduce the cost of serving large language models (LLMs), but its accuracy cost is uneven and is often tuned per model. We study where quantization damage occurs and how to allocate a small additional precision budget. Using causal mixed-precision intervention as ground truth (raise each layer to 8-bit in turn and measure the accuracy it recovers) across 9 open-weight models in 4 architecture families, we test 3 intuitive hypotheses: that quantization damage lives in task circuits, where the model computes, or in weight statistics. None of them predicts which layers benefit from restored precision. Recovery is instead diffuse: for 8 of 9 models, recovering 75% of the gap takes roughly half the layers; the lone exception, Qwen3-8B, is sharply concentrated. At a matched precision budget, spending it globally on finer quantization granularity beats locally repairing the most recoverable layers for all 8 group-128-compatible models (all but OpenLLaMA, whose width rules out group-128), by 21-52 points, including the concentrated Qwen3-8B. We report 2 secondary findings: the residual is budget-limited (8-bit is near-lossless in our evaluation across RTN, GPTQ, and AWQ), and the location of peak recovery correlates with architecture within a family, though not across families. Within this budget setting, global granularity is a better default than selectively protecting critical layers. More broadly, cheap signals that correlate with quantization damage do not necessarily identify where restoring precision improves accuracy; this must be tested with causal intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01573v1">Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      How to divide a fixed annotation budget between supervised fine-tuning (SFT) and reinforcement learning (RL) during LLM post-training remains an open problem. Existing work characterizes only broad trends (e.g., SFT dominates in low-data regimes), lacks a principled allocation framework, and does not examine whether the optimal ratio transfers across model sizes. We frame this problem in terms of near-optimality: rather than seeking a single optimal SFT-RL ratio, we characterize the near-optimal region, the set of allocations within a specified tolerance of peak performance. Empirically, this region is wide even for small tolerances (2-10%), widens with model scale, and transfers reliably from small proxy models to large target models. This yields a practical strategy: small proxy-model experiments suffice to identify a transferable near-optimal region, eliminating the need for exhaustive large-scale search. Our results hold consistently across tasks, model families, and both preference-based off-policy and reward-supervision on-policy RL methods. We further analyze how the asymmetry in annotation costs between SFT and RL data shifts the near-optimal region.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01572v1">From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Data-residency constraints force enterprises to self-host LLMs, but continuous adoption of newer models without decommissioning their predecessors expands the serving fleet, fragmenting a finite GPU pool. We consolidate traffic from over 200 internal applications onto a single model by closing quality gaps identified through production error analysis along three axes: instruction following, function-calling, and internal task distribution. Quality is tracked by offline benchmarks stratified to production traffic and scored by deterministic verifiers or calibrated LLM judges. Rather than optimising all objectives jointly, which introduces cross-domain reward interference, we train a separate GRPO expert per axis and merge them via two-stage SLERP. Each expert's reward exposes a distinct failure mode, namely semantic collapse, over-calling, and verbosity hacking, each requiring a domain-specific fix. In non-reasoning mode the recipe surpasses a ${\sim}7\times$ larger by total parameters baseline on the in-house Arena with 69.6 to 65.8, instruction following with 0.85 to 0.83, and function-calling with 0.79 to 0.77, while lifting general dialogue benchmarks. The model absorbs 50% of platform traffic, 116M requests per month, at a fraction of the serving cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01552v1">Can LLMs Discover Scientific Laws in Real and Parallel Worlds?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 42 pages, 16 figures. Project page: https://yiyihum.github.io/SciLaws-Bench/
    </div>
    <details class="paper-abstract">
      Scientific equation discovery has long been central to scientific progress, proceeding through iterative cycles of hypothesis generation, observational testing, and refinement under scientific constraints. As LLM capabilities advance and their role in AI for Science expands, it remains an open problem whether they can genuinely discover scientific laws and how this ability should be evaluated. Existing evaluations, however, often either simplify discovery through synthetic settings or reuse published targets that may already be familiar to LLMs. We therefore introduce SCILAWS-BENCH, a benchmark for scientific law discovery built from published research and real scientific data. It comprises 118 problems drawn from 381 scientific papers, covering 291 candidate laws and roughly 8M real data points across six scientific disciplines. Each problem is instantiated in two complementary settings: (1) SCILAWS-REAL asks models to propose laws from fixed real observations and evaluates held-out predictive fit and scientific validity derived from the source literature, and (2) SCILAWS-PARALLEL asks models to actively query residual-calibrated worlds and recover synthesized hidden laws derived from published forms. This two-setting task design preserves each problem's scientific context while separately evaluating fixed-record law discovery and active recovery of a newly synthesized hidden law. We find that predictive fit can diverge from scientific validity, memorization shapes whether models reproduce or move beyond published formulas, and our best-of-N study reveals a selection bottleneck. Our work provides a paper-grounded benchmark and new empirical perspectives for evaluating AI for scientific discovery. Project page: https://yiyihum.github.io/SciLaws-Bench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01535v1">Can LLMs Design Video Coding Tools? A Case Study on Planar Mode</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      This paper explores whether large language models (LLMs) can design video coding tools, a highly challenging task due to the intricate algorithmic coupling of tool modifications. In particular, we present an empirical case study on the Planar mode, a long-standing intra prediction tool in video coding standards. Our experiments operate within a generation-and-evaluation loop, with the LLM generating new Planar predictors, encoder trials evaluating their coding performance, and the LLM re-generating refined implementations based on the evaluation feedback. We first examine directly replacing the default Planar mode in the Fraunhofer Versatile Video Encoder (VVenC) under its faster preset. Experimental results demonstrate that the LLM-generated mode can outperform the conventional Planar mode on this lightweight toolset, achieving 0.18% bitrate savings with 0.4% complexity overhead on the standard benchmark. We further extend our evaluation to the Enhanced Compression Model (ECM). Leveraging newly introduced directional Planar modes, we investigate two integration strategies: directly replacing them, and introducing the LLM-generated predictor as an additional prediction mode with new syntax elements. The empirical results suggest that both strategies can yield coding gains under a constrained low-resolution setting. Overall, this study offers preliminary evidence and practical insights, highlighting both the potential and open challenges of LLM-based coding tool design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01515v1">TempCloze: Can Video-LLMs Identify the Missing Middle?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Temporal reasoning benchmarks for Video-LLMs are often mediated by language, leaving room for linguistic shortcuts from option wording, answer correlations, or language priors. To reduce such shortcuts, we introduce TempCloze, a video cloze benchmark for evaluating visual temporal reasoning in Video-LLMs. Given the beginning and ending clips of a video, models must identify the true missing middle from four candidates. TempCloze contains 1,521 carefully filtered videos from seven sources, mainly long-take and egocentric videos. We construct same-source distractors along three dimensions: Semantic asks what event should happen, Alignment probes when it should occur, and Progression tests how it should unfold, while shared scenes and objects reduce appearance cues. Our evaluation of 10 proprietary and 21 open-source Video-LLMs reveals Alignment as the primary bottleneck: models often recognize plausible semantic content and local event progression but struggle with temporal alignment. We further conduct error pattern and behavioral sensitivity analyses on TempCloze-Mixed and TempCloze-Hard with four representative models to examine where errors arise and how candidate order, context direction, visible span, frame density, and test-time scaling influence model choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01491v1">GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 GlossoGen code: https://github.com/agencyenterprise/GlossoGen Paper code: https://github.com/esteng/emergent_communication
    </div>
    <details class="paper-abstract">
      The growing rate at which LLM agents interact with one another raises key questions about language evolution in multi-LLM-agent settings, with implications for safety and monitorability as well as for linguistic accounts of LLMs. To address these questions, we introduce GlossoGen, a novel platform for studying multi-agent language evolution in complex scenarios. Within GlossoGen, we build the SaveVeyru scenario, which requires agents with partial information to communicate under pressure. We find that language evolution does occur between LLM agents, that the resulting languages are compositional and morphologically productive, and that they deviate from the LLMs' English prior in ways that render them incomprehensible to humans. Moreover, we identify several qualities essential to this evolution: pressure towards efficiency; the strength of the models backing the agents; and access to a "postmortem" stage in which agents can agree on linguistic conventions. Importantly, we observe that different conditions govern the transmission of language to new agents. Specifically, we find that agents learn new languages from usage alone, take an active role in this learning, and that while stronger models are required for novel language emergence, weaker models can learn an existing language once it has emerged. Taken together, our results indicate that current LLMs have the potential for cumulative cultural evolution -- previously attested only in humans -- with mixed populations of agents developing capacities that go beyond their lowest common denominator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10349v2">FlexP-SFT: A Flexible Aggregation-Free Framework for On-Device Personalized Split Federated Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      To fine-tune large language models (LLMs) over private data, federated learning (FL) has emerged as a promising paradigm. However, the prohibitive memory and communication demands of LLMs render standard FL impractical for resource-constrained edge devices. While split federated learning (SFL) alleviates the computing burdens via model partitioning, existing frameworks still suffer from communication bottlenecks and straggler problem due to the parameter aggregation process. To address these challenges, we propose FlexP-SFT, a novel aggregation-free framework for personalized split federated fine-tuning, which fundamentally eliminates the client-side aggregation process. Crucially, to ensure robust training in the absence of global synchronization, we introduce a layer-flexible alignment strategy to balance personalization and generalization capabilities. We further formulate split-ratio selection as a resource-aware discrete optimization problem that jointly accounts for personalization accuracy and system cost. Our proposed scheme simultaneously enhances personalized performance, reduces communication overhead, and resolves the straggler problem. Extensive results show that FlexP-SFT substantially outperforms baselines in both accuracy and latency, and that the optimized split ratio achieves a better resource-accuracy trade-off than static or memory-only choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04092v2">SEDCoT: Enhancing LLM-Based COBOL Code Translation via Symbolic Execution and Delta Debugging</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      COBOL remains critical across banking, insurance, and government infrastructure. However, maintenance is increasingly challenging due to outdated technologies, sparse documentation, and developer retirement, necessitating code translation into modern languages like C. Traditional rule-based transcompilers yield outputs that are difficult to read and maintain, while general-purpose large language models (LLMs) achieve suboptimal correctness because COBOL is a low-resource language with distinct logic patterns. To bridge this gap, we propose SEDCoT, a novel COBOL-to-C translation framework. SEDCoT first leverages LLMs for initial translation, then combines symbolic execution with LLM guidance to generate test suites and iteratively repair semantic discrepancies. Finally, it integrates delta debugging to minimize failing tests into succinct counterexamples, accelerating automated code repair. Evaluating SEDCoT on a public COBOL-to-C dataset demonstrates that it outperforms state-of-the-art baselines by at least 12% while producing translations with substantially higher readability than rule-based alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01457v1">Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 10 pages, 8 figures, accepted by Mobihoc2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) fine-tuning is increasingly shifting toward data generated on edge devices, where memory, computation, bandwidth, and connectivity constraints make conventional federated learning difficult to sustain. Split federated fine-tuning (SFT) improves client-side efficiency by offloading most model parameters and computation to the server but requires step-by-step bidirectional communication loop across the split interface and forces continuous client involvement throughout training. In this paper, we present L-shaped SFT, a split fine-tuning framework that removes this bidirectional bottleneck. Our key insight is that weight tying in modern LLMs enables server-side hidden activations to be directly supervised using target embeddings, allowing the training loss to be computed on the server without returning server outputs to the client. To further eliminate the need for continuous client participation, based on L-shaped SFT, we introduce one-shot SFT, in which clients upload activations once and then go offline while the server continues optimization over cached representations. We implement our design in a real system testbed with heterogeneous edge clients, including commercial smartphones and NVIDIA developer boards. Experiments demonstrate that our schemes significantly reduce communication costs and client online time compared with existing SFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29300v2">MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Project page: https://github.com/sony/MusTBench
    </div>
    <details class="paper-abstract">
      Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBench, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBench show that existing LALMs struggle with precise temporal grounding, while MusT brings significant improvements over strong baselines. These results establish temporal grounding as a key missing capability in current LALMs and position MusTBench as a challenging benchmark for future research in temporally grounded music understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01437v1">HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Project page: https://self-developing-agents.github.io/
    </div>
    <details class="paper-abstract">
      As agents move from research prototypes to deployed tools, their capability increasingly depends on model-external execution infrastructure, commonly termed the agent harness. Changing this harness while holding model weights fixed can substantially alter task performance. Current agent evaluations typically report downstream performance under a chosen harness, leaving a model's ability to develop the harness itself comparatively underexplored. We introduce HarnessDev, a benchmark that shifts the unit of evaluation from task outputs to runnable infrastructure. HarnessDev covers two stages. In Creation, the agent starts from a minimal seed and a small number of cases, then builds a complete execution system. In Evolution, it starts from its own created harness and iteratively revises it using downstream execution feedback, with the goal of improving benchmark performance. We then evaluate each constructed harness on capability (task success on held-out benchmarks) and efficiency (execution-token cost). The reported Creation results cover six creator LLMs, four domains, and five downstream benchmarks totaling 2,207 unique downstream instances, with hidden evaluation tasks withheld from development. We find that generated harnesses remain substantially behind mature human-engineered references on code and on search and research, while matching or exceeding the selected references on writing and machine-learning experimentation, with large variation in execution cost. Evolution produces some performance gains, but they are unstable and transfer only partially to held-out tasks. Experiments with a fixed runtime model further show that the gains depend strongly on the model executing the harness, indicating limited transfer across models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01432v1">Citing Less Critically: LLMs Reshape the Rhetoric and Reach of Scientific Citation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted at the EMNLP 2026 main conference
    </div>
    <details class="paper-abstract">
      Scientific citations carry rhetorical intent. Scholars may cite prior work positively (supporting), negatively (contrasting), or neutrally (mentioning). As large language models (LLMs) increasingly assist scientific writing, whether they reproduce citations with the same rhetorical intent as humans remains unclear. We introduce a masked-citation task to compare human and LLM-generated citation behavior. For each citation context, an LLM generates a replacement citation sentence, producing a counterfactual corpus directly comparable to human citation. We analyze what, whom, and how models cite, using an LLM-as-a-judge to classify citation intent and a 20-million-edge coauthorship network to measure social distance between cited authors. Across six popular LLMs and 1,746 top NLP conference papers (63k+ contexts, 132k+ citations), three patterns emerge: (1) Compared with human citation, LLMs cite significantly less critically; (2) LLMs over-cite popular and older papers, a tendency amplified for contrasting citations where human writing more often draws on recent, niche work; (3) Whereas humans often cite within their close social network, especially for supporting citations, LLMs tend to draw on more socially distant authors. Together, these differences are double-edged: LLM citation reaches beyond a scholar's close collaborators while being less critical and amplifying visibility bias, reshaping the rhetoric and reach of scientific citation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01360v1">EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agent failures often contain multiple related errors rather than a single mistake. Existing attribution methods usually identify a responsible agent, step, or root cause, but do not explicitly model dependency between errors. We introduce EDGE, an Error Dependency Graph-guided multi-Error attribution framework. EDGE constructs an error dependency graph from observed error events and validates a reliable causal subset through counterfactual rollout. The inference graph guides a two-stage LLM-as-judge detector for error attribution, and the intervention-validated subgraph provides a more reliable basis for explanation and repair analysis. Experiments on TRAIL and MAST show that EDGE improves category-level multi-error attribution across most evaluated models and settings. Experiments with adapted Who&When-style prompts show that the graph helps across prompting strategies. These results suggest that dependency structure is a useful diagnostic prior for agent failures beyond isolated root-cause prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03994v4">Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to the findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      As organizations increasingly deploy LLMs in sensitive domains such as legal, financial, and medical settings, ensuring alignment with internal organizational policies has become a priority. Existing content moderation frameworks remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and training cost. To address these limitations, we frame policy violation detection as an out-of-distribution (OOD) problem in the model's activation space. We propose a training-free method that operates directly on the LLM internal representations, leveraging prior evidence that decision-relevant information is encoded within them. Inspired by whitening techniques, we derive policy-violation scores directly from normalized representations of LLM hidden activations. Our method requires only the policy text and a small number of illustrative samples, making it lightweight and easily deployable. We extensively evaluate our method across multiple LLMs and challenging policy benchmarks. It achieves up to 86.0% F1, outperforming both fine-tuned and LLM-as-a-judge baselines while requiring substantially less computation. Our code is publicly available at: https://github.com/FujitsuResearch/LLM-policy-violation-detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01338v1">mzCache: On-Device LLM Memory Management under Multitasking</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 MobiCom 2026
    </div>
    <details class="paper-abstract">
      On-device mobile Large Language Model (LLM) inference is gaining significant attention. However, mobile devices operate in highly dynamic multitasking environments where users frequently switch between applications. This creates memory pressure, forcing LLM memory (model weights and KV cache) to be evicted by the operating system. When a new inference request arrives, the inference system must restore the evicted memory through slow storage reads or recompute the entire KV cache, severely degrading responsiveness. To address this, we present mzCache, an on-device LLM inference system with specialized memory management for multitasking environments. Under unpredictable memory pressure, mzCache elastically evicts LLM memory and leverages the unified memory of mobile SoCs to enable zero-wait inference on the GPU with concurrent CPU-side restoration. mzCache realizes this through restoration-oriented memory management: LLM memory is partitioned into fine-grained shared buffers to enable partial eviction and restoration with concurrent cross-processor access, while hybrid swap and backward-out eviction policies ensure low-latency restoration from any eviction state. Implemented on llama.cpp and deployed as an Android application, mzCache achieves 2.1-5.5$\times$ reduction in Time-to-First-Token compared to storage-backed partial offload and demonstrates its effectiveness in real multitasking scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01337v1">LEAP: Likelihood Elicitation and Aggregation for LLM-based Probabilistic Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026). 20 pages, 3 figures, and 15 tables. Code: https://github.com/layingfish/LEAP
    </div>
    <details class="paper-abstract">
      LLM-based forecasting systems have improved on real-world tasks such as financial markets and sports outcomes, largely through stronger search and tool use. Many systems still ask an LLM to read all collected evidence together and produce the final forecast. We call this design Monolithic Prediction. It can obscure how individual evidence items affect the result and collapse uncertainty across competing outcomes. We propose LEAP (Likelihood Elicitation and Aggregation for Probabilistic forecasting), which reorganizes how collected evidence is used in the prediction stage. LEAP examines each evidence item separately and elicits likelihood parameters that describe its implications for the target. An explicit prior and a deterministic probabilistic model then combine these likelihoods into a posterior distribution. This procedure supports continuous, single-choice, and multi-choice forecasts while preserving reproducible evidence contributions. We build a benchmark covering forecasting, information-seeking, and browsing tasks, and evaluate LEAP on our own agent loop and several agent CLI frameworks. Given the same evidence, LEAP improves most prediction and calibration metrics across models and remains stronger under controlled comparisons of prior access, inference budget, and aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01320v1">Automated Event Log Generation from Unstructured Text Using Finetuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Process mining (PM) provides a powerful framework for discovering and optimizing operational processes from event data. However, the efficacy of PM techniques is strictly predicated on the availability of structured event logs. Thus far, event logs have often been laboriously created by domain and process mining experts. This costly effort causes large portions of organizational knowledge, including incident tickets, manuals, and textual reports, to remain underutilized. We address this bottleneck by investigating the efficacy of Large Language Models (LLMs) as automated data translators. We propose a scalable framework that leverages LLMs as data translators to bridge the gap between unstructured textual resources and structured event data. We finetune LLMs on a newly created text-to-log dataset, demonstrating that the resulting models can extract high-fidelity event logs from unstructured resources. Our results show that this finetuning approach outperforms few-shot or zero-shot prompting by a large amount, highlighting finetuning as a necessary pre-condition for generating reliable event data. We conclude that our method provides a promising pipeline for making previously unused data available to process mining ecosystems, effectively expanding the possibilities of using PM to further investigate organizational workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23754v2">LLM-driven design of physics-constrained constitutive models: two agents are better than one</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Developing constitutive models that capture how materials deform under load traditionally requires years of specialized expertise in continuum mechanics, machine learning, and scientific programming. Large language models (LLMs) have recently been shown to lower this barrier by generating constitutive models on demand, but existing single-agent pipelines lack systematic checks that the resulting models respect fundamental physical laws. To close this gap, we introduce the first multi-agent LLM-driven approach for constitutive model generation: a Creator agent proposes a model tailored to the data, while an Inspector agent critically audits each proposal against nine physical constraints and returns it for refinement whenever a violation is detected. We demonstrate this concept with constitutive artificial neural networks (CANNs) and benchmark it on brain tissue and rubber as isotropic materials, and on porcine skin tissue as a transversely isotropic material with a preferred fiber direction, using two different LLM backbones (Claude Opus 4.7 and Kimi K2.5). Whether a generated model satisfies the physical constraints is assessed numerically, by probing each constraint across a broad sample of deformation states, rotations, and perturbation directions. Adding the Inspector raises the share of exported models that pass all these checks from 90\% to 95\% for Opus and from 47\% to 60\% for Kimi. In addition, the generated models are on par with or even surpass expert-designed models in accuracy, extrapolate reliably beyond the training data, and generalize remarkably well to unseen loading paths. Separating generation from inspection thus turns LLM-driven constitutive modeling into a substantially more trustworthy process. The paradigm is deliberately technique-agnostic...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02010v2">PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 15 pages, 4 figures, https://github.com/wizzard0/planar-bench-as1073
    </div>
    <details class="paper-abstract">
      Existing LLM graph benchmarks typically ask models to answer graph-theoretic questions or compute symbolic solutions rather than construct spatial layouts. Within-task difficulty is also primarily stratified by vertex count. However, existing research also suggests that task difficulty is more closely related to the number of constraints imposed by the edges than to the number of vertices being arranged. We introduce PlanarBench, a benchmark that asks models to produce crossing-free ASCII drawings of planar graphs given only an edge list. Across 91 model configurations and 199 non-isomorphic connected planar graphs with 2-7 vertices, edge count is more strongly associated with mean task score than vertex count ($r=-0.85$) versus ($r=-0.47$) and remains strongly associated after controlling for vertex count ($r_p=-0.80$). PlanarBench provides a controlled setting for separating these two difficulty axes. In addition, neither drawing area nor total response length demonstrated a meaningful correlation with score, which is evidence against a simple output-size explanation. Performance varies widely: the best model scores 159.5 out of 199, most models below 30B parameters score under 25, and substantial failures remain among frontier systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01246v1">Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 EMNLP 2026 - Main Conference
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are primarily optimized for written text, often producing outputs that are grammatically correct and helpful yet poorly suited for spoken delivery via Text-to-Speech (TTS). In this work, we study how to make LLMs natively generate TTS-friendly text, which we frame as a preference alignment problem: instead of relying on downstream rewriting modules, we directly align LLMs to generate text optimized for spoken delivery. We introduce two preference datasets spanning different target domains, CORA and Recipe, which contain paired TTS-friendly and TTS-unfriendly responses. We further propose an evaluation suite combining a pattern-based heuristic metric, a TTS$\to$ASR evaluation pipeline, and a MUSHRA listening study with human judges. Our experiments compare the recently proposed Feature-aware Sampling and Tuning (FaST) framework -- leveraging interpretable features instead of a black-box reward model -- against an array of alignment baselines on the TTS-friendly generation task. Notably, we found that FaST achieves the best overall tradeoff between TTS-friendliness and helpfulness across various settings. We also identified a strong correlation between our different metrics, highlighting the ability to reliably assess TTS-friendliness via an efficient heuristic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01210v1">Who Judges the Judges? A Chinese Safety QA Benchmark for Evaluating LLM Responses and Safety Judges</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Safety benchmarks for large language models often assess the risk of a user query, although the outcome of question answering depends on whether the response violates a policy. This distinction is critical in Chinese harmful-content evaluation, where linguistic variation and adversarial transformations can obscure risky intent. We introduce C-SafeQA, a policy-grounded benchmark for response-level Chinese safety evaluation. It comprises 538 base queries and 8,877 adversarial queries answered by four full-model LLM deployments, yielding 37,660 query-response records labeled safe, unsafe, or disputed. Reference labels are generated through agreement-aware multi-model adjudication and blind audits of stratified subsets by three safety experts. C-SafeQA supports both evaluation of target-model safety and auditing of seven automated safety judges against shared reference labels. Unsafe-response rates range from 0.93% to 3.35% on base queries and from 11.68% to 30.05% on adversarial queries. On the adversarial subset, judges show substantial trade-offs between unsafe-response recall and risk-query-conditioned safe-response false positive rate, and no judge dominates all metrics. Both acrostic transformations reduce unsafe recall for all seven judges, revealing mechanism-specific evaluator weaknesses. Dataset records, metadata, verification code, and judge scripts are publicly released to support recomputation, while benchmark construction, target-response generation, and private adjudication remain outside the release boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00754v4">Enhancing Self-Supervised Visual Representation Learning via Low-Rank Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Published in Transactions on Machine Learning Research (TMLR), 32 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLM) blocks with Vision Transformers (ViTs) holds significant promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between the text-centric pre-training of LLMs and the vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. Consequently, prior works typically keep LLM blocks frozen while learning only the vision components. To address these challenges, we introduce Language-Adapted Vision Enhancer (LAVIE), a novel framework that bridges this modality gap through a synergistic pre-training strategy. LAVIE co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the same MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LAVIE significantly improves performance in various downstream vision tasks, offering an effective and efficient way to enhance visual understanding using frozen LLM knowledge. Code is available at https://github.com/selimkuzucu/LAVIE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01195v1">CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Entity matching (EM) requires fine-grained contextual understanding and domain knowledge. Recent work shows that large language models (LLMs) can serve as strong matchers across domains, but most methods either make independent pairwise decisions or rely on manually designed composite pipelines, thus lacking flexibility in realistic multi-candidate settings. At the same time, they typically ignore inference cost at scale. We formulate LLM-based EM with candidates as a cost-aware sequential decision problem and propose CaRL-EM, a reinforcement learning controller that manages LLM operations. Given the state of an anchor record, its candidate set, and the cost, CaRL-EM adaptively chooses among different operators (Match/Compare/Select/Decide) and model capacities to maximize a quality-cost objective. The policy interacts with abstract operators, allowing the same controller to be reused with different underlying LLM backends at inference time without retraining. Experiments on 7 benchmarks show that CaRL-EM (i) learns to dynamically plan the usage of inexpensive and expensive operators based on task complexity, (ii) achieves robust zero-shot transfer across diverse datasets and domains, and (iii) consistently achieves a better quality-cost trade-off than strong LLM-based baselines and manually designed pipelines, yielding a lower inference cost at comparable or higher quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01185v1">Reveree: Diagnosing LLM Reverse-Engineering Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Reverse engineering (RE) is critical to security tasks such as malware analysis and vulnerability discovery, and large language model (LLM) agents are increasingly able to perform it autonomously. Capture-the-flag (CTF) RE challenges have become the standard proxy for measuring this capability, but evaluation rests on a single criterion: whether the agent captures the flag. This solve rate reveals neither where in the RE process an agent fails nor whether a success reflects analysis of the binary or recall of a public solution. In this paper, we propose Reveree, a diagnostic framework that scores an LLM RE agent's trajectory at three tiers: solve rate, milestone progress through an eight-stage RE schema, and a behavioral profile of its actions. Comprehension stages are scored by an outcome-blinded LLM judge validated against a human expert; all other stages are verified deterministically. Using Reveree, we evaluate nine frontier models and four prompting strategies on 88 picoCTF and NYU-CTF challenges. We find that the base model dominates performance, whereas prompting strategy is a secondary, model-dependent effect. Surprisingly, larger, newer, or costlier models are not reliably stronger. We also find that failures concentrate at the comprehension stages of the RE process, and that extra budget, persistence, or reasoning effort rescues few of them, pointing to a competence limit rather than a resource limit. Regarding memorization, while models reproduce picoCTF flags from challenge descriptions alone, NYU-CTF shows minimal measurable recall, and most solves survive surface perturbation, indicating that genuine analysis coexists with memorization. We release Reveree to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23671v5">Can LLMs Reliably Self-Report Adversarial Prefills, and How?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 EMNLP 2026 (Main)
    </div>
    <details class="paper-abstract">
      Prior work shows that large language models (LLMs) exhibit varying degrees of introspective capability on benign tasks. We extend the question to safety contexts and examine how reliably a model can recognize that its own prior response was elicited by an adversarial prefill attack. Across ten open-weight instruction-tuned LLMs from 3B to 70B parameters and four safety benchmarks, no model reliably recognizes its own compromised outputs, with models claiming intent on prefilled responses at an average rate of 25.3%. Introspective signal stems primarily from reasoning about safety and refusal. Orthogonalizing models' weights against the refusal direction collapses the gap between claim rates on prefilled and natural outputs to near zero, though the direction is not its unique mediator. Framing the question as internal intention versus external tampering elicits qualitatively different responses on the same models. Training models to mimic correct introspective answers or optimize an introspective objective can improve the accuracy of introspection, but such training does not transfer to the tampering probe and counterintuitively raises attack success rate under adversarial prefill on most models, amounting to a partial mitigation. These findings outline mechanisms underpinning the observed introspective signals in safety contexts and highlight risks in the reliability of LLM self-reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01167v1">Classic AI Scaffolding for LLM Social Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large language models can produce locally plausible social turns, but fluent next-turn generation is not enough for social simulation. Human encounters such as restaurant lunches and hotel check-ins are bounded social episodes with roles, scripts, material state, obligations, commitments, timing, and closure conditions. We present EpisodeSim, a hybrid LLM-agent architecture that represents classic-AI structures as natural-language control state interpreted by LLM calls. A World Master maintains shared reality, constructs scenes, adjudicates proposed actions, tracks effects and obligations, and controls closure. Experiments with small qualitative ablations on two held-out settings support a design claim: LLM fluency supplies local texture, but coherent social simulation benefits from persistent classic-AI-style scaffolding that organizes behavior over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17809v2">Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 EMNLP 2026 (Main)
    </div>
    <details class="paper-abstract">
      Humans naturally form and express beliefs in daily communication, e.g., "I think the answer is 3" or "I suppose that's right." Such beliefs inevitably intertwine with fact and knowledge, making the ability to handle them in tandem desirable for large language models (LLMs), as they are increasingly deployed in user-facing settings. Prior work showed that even capable LLMs exhibit a systemic weakness in acknowledging user beliefs grounded in incorrect information. We extend this evaluation to 10 LLMs across 18 epistemic expressions and find that the size and direction of this weakness depend on the verb used to express the belief, with the accuracy gap between factual and false information ranging from +50% on "I vaguely remember" to -14% on "I seriously doubt". We further show that the phenomenon stems from what we call task confusion: models default to fact-checking the underlying claim, overriding the user's stated belief. We provide evidence where chains of thought that explicitly fact-check show lower accuracy on false information than those that do not, and a single instruction can reverse the failure across verb families. Mechanistically, models attend more to false beliefs they fail to confirm, but suppressing this attention at decoding time recovers accuracy only partially and only in some models, calling for future work on intervention methods. Our findings clarify prior results and show how fact-checking, a generally desirable behavior, can interfere with belief tracking in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.16903v3">GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted by ICLR 2026; Homepage: https://sproutnan.github.io/GuidedBench/
    </div>
    <details class="paper-abstract">
      Despite the growing interest in jailbreaks as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. With a systematic measurement study based on 37 jailbreak studies since 2022, we find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. In this paper, we introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset and GuidedEval, an evaluation system integrated with detailed case-by-case evaluation guidelines. Experiments demonstrate that GuidedBench offers more accurate evaluations of jailbreak performance, enabling meaningful comparisons across methods. GuidedEval reduces inter-evaluator variance by at least 76.03%, ensuring reliable and reproducible evaluations. We reveal why existing jailbreak benchmarks fail to evaluate accurately and suggest better evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26724v2">UrbanDS: A Graph-Guided LLM Multi-Agent System for Data-Intensive Urban Tasks</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have been widely applied in automating data science tasks. However, existing methods typically rely on a limited set of provided datasets, and they face challenges in data-intensive scenarios that require discovering and leveraging relevant information from large-scale and heterogeneous data repositories. Urban tasks are representative examples of such scenarios, as urban data are not only large-scale and multi-sourced, but also exhibit complex spatial, temporal, and semantic relationships. To address these challenges, we propose UrbanDS, a graph-guided LLM multi-agent system for data-intensive urban tasks. We first construct a unified dataset graph to organize reusable dataset skills and the relationships among datasets. Specifically, we develop a Data Profiling Agent that constructs a skill for each dataset. Moreover, a Relation Agent identifies relationships among datasets and integrates these relationships into the dataset graph. At runtime, a Planner Agent retrieves task-relevant datasets from the graph and generates execution plans. Multiple Execution Agents then perform data processing and analysis, while their execution progress and intermediate results are shared through a common memory. Finally, a Report Agent synthesizes the experimental logs into a report, which can be further refined based on user feedback. To systematically evaluate the capability of agents in handling data-intensive urban scenarios, we further construct UrbanDS-Bench, an urban data science benchmark covering representative data analysis and modeling tasks. Experiments on both general and urban benchmarks demonstrate that UrbanDS consistently outperforms existing data science agents on data-intensive tasks. Furthermore, UrbanDS has been deployed on the urban operations platform of Dongxihu District, Wuhan, demonstrating its effectiveness in real-world urban applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01117v1">Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Chain-of-thought reasoning unfolds in discrete token space: each step is committed as text, errors propagate, and eliciting good traces presupposes traces to imitate. Reasoning instead in a model's continuous representation space - where intermediate states are vectors rather than words - sidesteps these constraints, but leaves open how those latent states should be computed. We approach this along two axes. First, we keep a large language model (LLM) frozen and use it for what it is already good at - modeling and decoding sequences - while a small auxiliary network supplies continuous latent thoughts as input. Second, we produce those latents by recurrence: a tiny recurrent reasoner refines them over many steps, decoupling the depth of computation from the size of the model, so that the latents are a product of iterative processing rather than a single forward pass. We instantiate this as Latent Recurrent Thoughts (LRT): a task-dedicated proposer supplies base latents, a recurrent reasoner refines them through bounded residual corrections, and the frozen LLM decodes the answer. On symbolic reasoning with answer supervision but no reasoning traces (Countdown-4, Sudoku) and on natural-language reasoning (HumanEval, MBPP, StrategyQA), LRT substantially outperforms prior frozen-decoder continuous-space reasoning methods under an identical decoder, prompt, data, and training budget, and outperforms non-thinking-mode chain-of-thought prompting on the same backbone at a small fraction of its inference compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01095v1">Reliable LLM-Generated Programs for High-Energy Physics Experiments through Graph-Grounded Software Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 24 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Extracting physics information from modern particle-physics experiments requires multistage analyses implemented on top of large and highly interconnected software ecosystems. General-purpose large language models (LLMs) often produce unreliable programs for such tasks because a user request alone rarely specifies the required APIs, dependencies, and usage conventions. We organize these software relations before generation and retrieve task-relevant knowledge at inference time. Using the open-source ROOT framework as a representative and reproducible testbed, we evaluate a complete grounding system that combines hybrid retrieval over a heterogeneous software knowledge graph, skill-selected workflow examples, and execution-guided repair. On a benchmark of 275 ROOT tasks, grounding improves first-attempt execution from 58.5% to 76.0% under Claude Code orchestration and from 51.3% to 64.0% under standalone orchestration. Final success increases from 90.5% to 96.0% and from 78.9% to 90.9%, respectively, while the average generation cost per successful task increases by only 1.3% and 3.2%. The gains persist under a strong coding agent, indicating that explicit software knowledge remains valuable even when agentic scaffolding is already in place. Because the method captures software relations common to large codebases rather than facts specific to ROOT or a particular model, it should transfer to other experiment frameworks and proprietary software, especially where documentation is sparse or internal dependencies are complex.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01084v1">Hardware Acceleration of Block-Diffusion LLM for Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Single-stream (batch-one) edge inference cannot amortize weight traffic across requests. Full-attention diffusion LLMs recompute the entire sequence at every step; native block diffusion makes completed blocks immutable and exactly cacheable, yet refinement still streams prefix KV and FFN weights. We co-design WIFiV-LPDDR, a wide-I/O LPDDR system for precision-tagged reads, BRQ-KV for a canonical low-rank-plus-INT8-residual prefix with query-dependent per-entry precision, and DAT-FFN for drift-mapped canonical replacement, adjacent-stage-corrected low-bit delta, or cached-state carry while keeping live activations unquantized. Both map to an input-stationary mixed-precision systolic array. For the evaluated 1.5B/7B models on modeled Jetson-class platforms, the full stack provides arithmetic-mean energy-reduction factors of 3.79x/3.96x and arithmetic-mean latency speedups of 2.88x/4.44x at the reported DAT-FFN settings; every corresponding compressed model-benchmark score drops by less than one absolute percentage point from its baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13302v4">One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in authorship attribution and verification. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive Causal Language Modeling (CLM) pre-training, one-shot capabilities and scale of LLMs, avoiding explicit supervision. Our methods substantially outperform prompting-based unsupervised baselines in authorship verification at similar model sizes, and is competitive with or improves contrastive baselines in most settings with sufficient model scale. We further observe strong performance across non-English languages. The effectiveness of the proposed framework improves consistently with increasing model scale. In the case of authorship verification, we propose an additional mechanism that increases test-time computation to improve accuracy; enabling flexible trade-offs between computational cost and task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27898v3">UniACE: A Unified Framework for Evaluating LLM Agentic Capabilities</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Agent benchmarks are increasingly used to compare large language models (LLMs) across domains, yet a reported score reflects a complete model--harness--environment configuration rather than the model alone. Benchmark packages couple native tasks with specific prompts, tool protocols, orchestration logic, and sometimes dynamic external resources, making cross-benchmark comparisons sensitive to implementation and resource conditions. We present UniACE, a unified framework for model-centric evaluation under an explicit, common execution condition. UniACE represents each benchmark as an instruction--tool--environment triplet, executes LLMs through a shared, task-agnostic harness in isolated per-task runtimes, and preserves native success criteria. For tasks that rely on dynamic resources, an optional offline mode replaces live access with fixed, pre-collected snapshots. Its evaluation protocol further standardizes efficiency measurement, execution records, and trace-based failure attribution. We migrate 7 benchmarks spanning 24 domains and evaluate 15 models in more than 400K rollouts consuming 5B tokens. Comparisons with source implementations show large bidirectional score changes and model-ranking reversals, while matched online and offline runs reveal substantial sensitivity to accessible evidence and its representation. Under the shared UniACE configuration, efficiency and failure profiles expose task-dependent model behaviors hidden by task-success scores alone. These findings motivate reporting agent benchmark outcomes as properties of an explicit evaluation configuration, enabling more interpretable and reproducible cross-benchmark comparisons. Codes and benchmarks at are available at https://github.com/whfeLingYu/A-Unified-Framework-for-the-Evaluation-of-LLM-Agentic-Capabilities, https://huggingface.co/datasets/whfeLingYu/Unified_Agent_Framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01073v1">Post-hoc Alignment of LLM-judges to Human Judgment Distribution</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge (LLMaJ) framework offers a cost-effective and reproducible solution for automatic evaluation. However, current evaluation practices typically compare LLMaJ judgments against aggregated ground-truth labels, overlooking the valuable information contained in Human Label Variation (HLV). Inspired by an increasing line of work that proposes to leverage HLV, we systematically study LLMaJ performance on predicting both a single, aggregated ground truth hard-label and unaggregated soft-labels that represent Human Judgment Distributions (HJD). Our results across five diverse datasets reveal that while LLMs achieve near human-level performance at hard-label prediction on most tasks, they exhibit poor performance when predicting soft-labels. To address this limitation, we propose NAPHA (eNtropy-Aware Post-Hoc Alignment), a simple yet effective lightweight post-hoc alignment method that matches the LLM distribution to the HJD by first assigning an instance to a discrete entropy class and then routing it to specialized, trained alignment models. We find that NAPHA consistently improves soft-labels prediction across base LLM models and datasets, with particularly strong gains on high-entropy instances where capturing diverse human perspectives is most critical. We also show via oracle experiments that improving entropy class prediction can substantially enhance NAPHA's practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22156v4">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Main conference of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.01035v1">Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 8 pages, 2 figures. Theory and synthetic numerical studies; no deployed-agent evaluation
    </div>
    <details class="paper-abstract">
      Recursive LLM agents can broaden their search by spawning specialists. Some branches later request tools that send data or deploy code. When should a branch receive authority to act? We distinguish sandbox spawning, in which external controls prevent the specified harm, from capability activation, in which a selected branch crosses an irreversible-action boundary. Progressive Risk Vesting (PRV) holds a trajectory-level risk budget in escrow and debits it as branches are activated. We prove an anytime harm bound for adaptively generated trees. Branch outcomes may be dependent, but each local certificate needs to remain valid conditional on the full pre-activation history, including the information used to select the request. When activation gates, branch charges, and compute constraints are held fixed, delayed vesting preserves every policy available under irrevocable spawn charging. Marginal risk estimates can still fail after branch selection. In a stylized branching model, trajectory harm changes as the authority reproduction number $\mathcal{R}_A$ crosses one. As local risk $p$ approaches zero, trajectory harm is proportional to $p$ below criticality, proportional to $\sqrt{p}$ at criticality, and retains a positive floor above it. A finite-type occupancy model yields risk and compute shadow prices. For nested fanout modes with decreasing marginal value per unit risk, these prices produce a threshold rule. Branching calculations and a split-sample experiment illustrate the results. These synthetic studies do not estimate safety in deployed agents. The analysis suggests a design rule: search broadly in the sandbox and grant recursive authority sparingly, with an explicit risk charge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01138v2">Antaeus: Hunting Repository-Level Logic Vulnerabilities via Context-Grounded LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      LLM-based vulnerability detectors have shown promising results in identifying memory-safety bugs and vulnerability classes whose violations can often be expressed through established security properties. Logic vulnerabilities, however, pose a different challenge, as their identification requires inferring application-specific security invariants and implicit assumptions about intended behavior. Even frontier agentic models struggle because these invariants are often implicit and buried among unrelated code. Motivated by this gap, we present Antaeus, a framework for detecting logic vulnerabilities that grounds LLM reasoning in repository-level code context. Antaeus follows a repository-scale pipeline combining function prioritization, context-grounded reasoning, comparative validation, and structured reporting. It ranks functions using lightweight repo-wide security signals, directing costly LLM analysis toward relevant code and reducing calls, cost, and triage effort. For each prioritized function, Antaeus combines local code context with a repository-level view of the application's functionality, security resources, and trust boundaries. This enables reasoning about how the function is executed within the broader application rather than as an isolated snippet. Antaeus identifies security-sensitive sinks, derives safety conditions for safe execution, and checks whether they are locally satisfied. Candidate findings undergo comparative validation, pruning concerns that reflect project-wide norms rather than distinctive violations. Finally, Antaeus reports sinks, violated safety conditions, and evidence, making findings actionable and traceable. We evaluate Antaeus on 35 repositories with confirmed logic vulnerabilities and compare it against function-level and agentic models. Antaeus detects and explains 20 vulnerabilities, outperforming baselines with comparable token usage and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00998v1">Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Word Sense Disambiguation has advanced rapidly for English and a handful of well-resourced modern languages, but it continues to assume the existence of a sense inventory and a word-to-sense mapping in the source language (Navigli, 2026). These assumptions break down for most historical and low-resource languages, whose dedicated WordNets are either incomplete or still under construction. We present Inspicio, an open-vocabulary retrieval pipeline that links tokens in context to synsets of the Open English WordNet (McCrae et al., 2020) without requiring any source-language inventory or mapping. For each occurrence, an instruction-tuned LLM produces two English translations of the surrounding sentence, a small set of candidate dictionary-style definitions, and a few candidate English lemmas. These outputs drive a hybrid retrieval step that combines dense definition-synset similarity, sparse lemma matching, and Maximal Marginal Relevance re-ranking. We evaluate the pipeline across a 6x6 grid of LLMs and sentence-embedding models on a new bilingual set of manually annotated Latin and Ancient Greek perception verbs, on a subset of PREMOVE dataset (Farina, 2025), and on a diachronic sample of Italian. The best configuration reaches 96% Recall@50 on the perception-verb test set, with each component contributing measurable gains, and remains competitive in the out-of-domain and cross-lingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.30646v2">BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 22 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA, multilingual QA, and multimodal QA tasks across multiple model families. In these experiments, BiG-SURE improves average abstention AUROC over prior black-box uncertainty estimators, while remaining simple, unsupervised, and applicable to black-box model settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00958v1">PersianAnonymizer: Evaluating LLM-Labeled Training for Efficient NER-based Anonymization in Persian</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 10 pages, 3 figures, 6 tables. Published at LREC 2026
    </div>
    <details class="paper-abstract">
      We target practical anonymization of Persian customer chats by training a compact NER model from LLM-labeled supervision and selecting the best labeler for deployment. We compare three instruction-tuned LLMs: DeepSeek-V3-0324, GPT-OSS-120B, and Qwen3-235B-A22B-Instruct-2507, to produce span annotations under a shared JSON protocol, yielding four corpora (OSS_ZeroShot, Qwen_ZeroShot, Qwen_FewShot, DeepSeek_FewShot). A MatinaRoberta-based token-classifier is trained per corpus and evaluated with token-level Precision/Recall/F1 (overall and per-class). We also report Label Coverage Recall (LCR), the proportion of gold non-O tokens predicted as non-O, and quantify cross-labeler behavior via a token-level Venn on test annotations. Finally, we contrast test-set annotation latency of the LLMs on H200 nodes with the trained NER's test-time labeling on a single RTX 3090. Results show that supervision from OSS_ZeroShot yields the strongest macro-F1 and LCR, while the resulting NER labels an entire 40K-message test set in approximately 2 minutes on one consumer GPU. This establishes a practical path to high-quality, low-cost anonymization for Persian industrial data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00918v1">RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 45 pages, 8 figures. Code available at https://github.com/ZhongruChen/RPCBench
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used as interactive recommender assistants. Their evaluation should therefore go beyond plausible item recommendation and test whether they can recognize flawed recommendation requests. Existing recommender benchmarks mainly assess ranking, generation, or preference satisfaction, while existing error-detection benchmarks are usually not grounded in recommendation-specific user and candidate evidence. To address this gap, we introduce RPCBench, a benchmark for evaluating Recommender-Premise Critique: the ability to detect, diagnose, and properly handle faulty premises in natural-language recommendation requests. RPCBench contains evidence-grounded test instances from five recommendation domains and covers ten types of premise failures. Each instance provides a visible recommendation context and a corrupted user query. We further design a fine-grained evaluation framework that measures proactive detection, error localization, post-detection handling strategy, and evidence faithfulness. Through a systematic evaluation of 11 LLMs, we find that proactive detection is the main bottleneck in Recommender-Premise Critique, and models perform worst on underspecified-premise errors. We also observe that target-critical information density matters more than redundant evidence, and that longer reasoning does not monotonically improve critique quality: performance peaks at intermediate reasoning length, while overly long reasoning is accompanied by an overthinking penalty. The code is available at https://github.com/ZhongruChen/RPCBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00904v1">In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Whether large language models (LLMs) can control their own internal representations matters for both machine metacognition and AI safety. A recent study applied neurofeedback to LLMs and claimed that they can control their internal representations. However, the reported control may rely on superficial mechanisms rather than genuine internal access because the control targets in that study are not privileged, meaning that a third party can infer them from the prompt. We redesign the neurofeedback paradigm for LLMs so that the control target satisfies the privileged access requirement, which is closer to neurofeedback experiments in human cognitive neuroscience. Under this stricter setting, the models do not demonstrate reliable control over privileged internal representations, suggesting that previously reported control cannot exclude the possibility that it relies on superficial mechanisms. Our results indicate that rigorous assessments of metacognition in LLMs require evaluation methods that demand privileged access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18114v2">How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out (LODO) transfer, and causal intervention. The susceptibility is largely shaped by alignment tuning rather than pretraining: pretrained base models generally cave much less to these biases, and their activations carry much weaker cue-specific signal beyond question content. Within aligned models, each bias has a coherent linear direction that we can both decode and steer along, recovering the unbiased answer across every family we test. The biases do not collapse into a single shared representation, however: cross-bias overlap is model-specific, and even behaviorally similar biases occupy different directions. The same intervention also provides a proof-of-concept debiasing tool, recovering a meaningful share of bias-induced errors while preserving most correct answers across all instruct families. Cue-induced bias is therefore best understood not as a single flaw in LLMs but as a family of causally effective linear directions that are largely shaped by alignment tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00886v1">Using LLMs to Elicit Security Requirements for Service-Oriented Cyber Ranges</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Cyber ranges are complex environments comprising many interacting components and stakeholders with different security concerns. The Service-Oriented Cyber Range (SOR) is no exception, particularly when it comes to training scenarios targeting critical infrastructure. Security concerns are translated into security requirements, the elicitation of which is usually difficult and time-consuming. This work examines how large language models can assist in eliciting security requirements for a service-oriented range and help produce a useful baseline for designers and developers. The approach follows a SEBoK-guided process in which security mission objectives and stakeholder needs were first identified and then provided as a prompt context along with architectural guidelines to five LLMs: GPT-5.2, Gemini 3.1 Pro, Grok 4.1, Sonar, and Kimi K2.5. The models generated 84 security requirements in total, which were consolidated into a comprehensive set of 27 requirements and then mapped to the architectural layers of the service-oriented range. The final set was evaluated by five cybersecurity experts against the criteria of necessity, clarity, completeness, feasibility, and testability, with an additional rejection option. The results showed a high acceptance rate, specifically for necessity with 98.5%, clarity with 87.4%, completeness with 85.2%, feasibility with 78.5%, and rejection with 0.7%. Testability was lower at 44.4%, indicating a slight lack of information on how these requirements could be tested. These findings show that LLMs can support early stages of the elicitation of security requirements, although human review is still needed, especially to improve or adjust certain aspects of the requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00859v1">Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Vehicle Routing Problems (VRPs) are fundamental combinatorial optimization problems with widespread applications in various scenarios. The advanced optimization solvers can effectively solve such problems. However, modeling complex VRP variants for solvers often requires substantial domain expertise, which limits the accessibility of advanced optimization technologies. In this paper, we propose Reinforcement Learning Enhanced LLMAgents(RLEA), a multi-agent framework designed to automate the modeling of complex VRPs. RLEA introduces a lightweight neural Planner trained with Soft Q-learning to efficiently orchestrate the actions of LLM-based agents. In addition, we equip the system with an evolutionary memory module and retrieval-augmented generation, enabling the agent to leverage both accumulated experience and external solver knowledge during program generation and refinement for solving VRPs. We evaluated 48 distinct VRP variants across various solvers. The experimental results demonstrate that RLEA outperforms the previous state-of-the-ar method, achieving a 16.67% higher success rate while significantly reducing runtime errors. These results validate that integrating reinforcement learning with LLM-based reasoning is highly effective for automated optimization modeling. The appendix is available at: https://doi.org/10.5281/zenodo.19134435.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00857v1">LLM Inference on IMC-NoC Architecture with Balanced Dataflow and Fine-Grained Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 11 pages, 15 figures, an extended version of LEAP ICCAD'2025
    </div>
    <details class="paper-abstract">
      LLM inference has become an essential service, yet it imposes unprecedented demands on memory bandwidth, computational density, and communication efficiency. While IMC is a promising solution to the memory wall issue, the heterogeneous data dynamicity of LLM requires complementary resources to handle intermediate data generated during run-time. Furthermore, the massive number of parameters in LLM necessitates scale-up architectures where on-chip data movement is often the primary performance bottleneck. This article presents a hardware-software co-design framework that unifies distributed compute, memory, and communication into a seamless processing-communication fabric. On the hardware side, we propose a scalable architecture, named LEAP, that integrates IMC PE, NMC PE, and INC. This allows each hardware layer to execute specialized tasks: IMC for static weights, NMC for dynamic data, and INC for partial result reduction. On the software side, we introduce a partitioning, mapping, and scheduling framework optimized for key metrics in LLM serving, including throughput and latency. To address the distinct computational intensities of the prefill and decode phases, we present a prefill-decode disaggregation approach that dynamically reconfigures PE organizations to maximize resource utilization. Compared to commercial GPU platforms, the proposed architecture provides a throughput and an energy efficiency improvement of $\geq{}1.52\times$ and $24.91\times$, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.29215v2">Attribute-Based Activation Steering of LLMs for Group-Specific Explanation Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026 Main
    </div>
    <details class="paper-abstract">
      To effectively enable people to understand new topics, explanations should be tailored to their backgrounds and abilities. Prompting alone has been shown to be insufficient for creating such explanations, and other computational methods are missing so far. Therefore, this paper investigates whether LLMs can be steered to generate explanations that are tailored to a specific group of people. To this end, we propose an approach that first identifies group-specific attributes in terms of explanatory style and knowledge of a specific target group. Building on activation engineering, it then computes attribute-based steering vectors and adds them to the internal activations of an LLM during inference to enable a fine-grained steering. In our experiments, we assess the steering effectiveness in terms of specificity and factuality of the generated explanations. Additionally, we evaluate the explanations in a study with human experts from different target groups. Compared to prompting and state-of-the-art steering baselines, our approach tailors the explanations significantly better to the target group while maintaining the best specificity-factuality balance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31213v2">Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      As LLMs increasingly serve as moral advisors and agents, they must address conflicts between competing values. Yet prior work on moral dilemmas overlooks a central aspect of human moral cognition: imagining alternatives beyond the given options. We introduce MoralAltDataset, comprising 307 Advisor and AI-facing Agent dilemmas augmented with compromise and reframed alternatives. We compare human and LLM judgments in binary and four-option settings. Across human participants and 15 LLMs, aggregate moral choice distributions differ substantially between the two settings, with compromise often preferred over either original binary option. Results show value shifts and stronger human-LLM agreement on alternatives. Source-stratified results reveal a descriptive gap: human alternative-selection rates are similar across authoring sources, whereas LLMs select GPT-5-authored alternatives substantially more often. We then compare human-authored alternatives with outputs from three representative LLMs through pairwise preference and expert-based evaluations. Alternatives from these LLMs are generally preferred and better satisfy fine-grained structural and ethical criteria, while revealing a trade-off between structural quality and practical feasibility. Our dataset is available here: https://huggingface.co/datasets/jongchanch/MoralAltDataset, and our project page is here: https://jongchanchoi.com/moral-imagination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04463v2">Beyond Static Summarization: Proactive Memory Extraction for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted in the Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Memory management is vital for LLM agents in long-term and personalized interactions. Most previous work studies how to retrieve and use memory, but pays less attention to how memory is extracted. We find two main limitations in existing methods. First, extraction is "ahead-of-time": the agent saves information before it knows future tasks. A single summary prompt often mixes details, events, and relations, so useful information is lost. Second, extraction is usually one-off. Without verification, errors and hallucinations may stay in memory for a long time. To address these limitations, we propose ProMem, a proactive memory extraction framework. It separates details, events, and relations, and uses different extraction strategies for each type. It also checks completeness to recover missed events and verifies facts at the atomic level to reduce hallucinations. Experiments show that ProMem improves memory completeness and QA accuracy, while keeping a good balance between quality and token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14980v3">Compositional Machine Design as Program Synthesis with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 75 pages, 31 figures, Project Page: https://besiegefield.github.io, accepted at EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong abilities in writing and revising programs, yet many program-synthesis benchmarks still evaluate programs in symbolic or digital environments. We introduce compositional machine design, a physically grounded form of program synthesis where machines are written as programs that compose standardized parts, and success is determined by simulated physical behavior. To study this problem, we present BesiegeField, a testbed built on the machine-building game Besiege. In BesiegeField, LLM agents generate machine programs from textual functional demands, execute the resulting machines in simulation, and receive rewards and state feedback. We benchmark LLM agents across representative machine-design tasks under single-agent generation, iterative editing, and hierarchical workflows. Strong models recover task-relevant structures and sometimes achieve nontrivial physical performance, but often struggle with spatially precise assembly, mechanism-level planning, and translating feedback into useful structural edits. We further finetune Qwen2.5-14B, an open-source LLM, with reinforcement learning from simulation-derived rewards. We find that, under a fixed generation budget, RL improves the best machine discovered. We additionally evaluate human performance to provide a reference point for task difficulty. These results establish compositional machine design as a testbed for studying LLM agents that synthesize executable machine programs and improve them through physical feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10390v2">LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Presented at The International Conference for High Performance Computing, Networking, Storage, and Analysis
    </div>
    <details class="paper-abstract">
      Large-scale LLM training is increasingly susceptible to hardware defects stemming from manufacturing escapes and silicon aging. These defects manifest as Silent Data Corruption (SDC) that perturb gradients and parameters throughout the training process. We present LLM-PRISM, a methodology to characterize LLM pre-training resilience to hardware faults. LLM-PRISM couples RTL-level GPU fault simulation with a stochastic injection engine embedded in Megatron-LM. Through 7,664 training runs across FP16, BF16, and FP8 regimes, we analyze how fault type, rate, and numeric format govern resilience. We find that while LLMs resist low-frequency faults, impact is highly non-uniform; critical datapaths and specific precision formats can induce catastrophic divergence even at moderate fault rates. This study provides the first hardware-grounded, pre-training characterization of SDC resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00738v1">Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      Inference-time search with large language models (LLMs) often concentrates on a small set of structurally or semantically similar trajectories, leaving alternatives underexplored---a failure mode we call \textit{reasoning basin collapse}. We introduce BASIN, a training-free, structure-aware selection method that groups reasoning states into basins and penalizes repeated visits to the same strategy, thereby reallocating search across genuinely distinct reasoning paths under a fixed compute budget. Under matched inference budgets, BASIN improves over Tree of Thoughts (ToT) by up to $+22$pp on Game of 24 and $+6.7$pp on MuSR. A quality-aware variant, QA-BASIN, further improves robustness by preserving high-quality basins when unconditional diversification over-explores. To explain when basin-aware selection helps, we introduce the redundancy gap $Δ$, which measures how differently search concentrates for correct versus incorrect predictions: standard ToT often operates near $Δ\approx 0$, while BASIN consistently shifts $Δ$ positive. More broadly, BASIN suggests structure-aware selection as a simple and general approach to improving inference-time reasoning. Code can be found at https://github.com/GitHubLuCheng/basin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00728v1">SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Accepted to EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable promise in translating and reformulating complex mathematical optimization problems across modeling languages. However, validating such transformations through empirical solver executions alone is unreliable, as solver outcomes may be affected by local minima, structural timeouts, numerical artifacts, and subtle semantic divergence between formulations. We introduce SOVER, an LLM-assisted SMT framework that separates semantic mapping from formal certification: Z3 checks domain cross-feasibility and global objective-order preservation for mixed-integer linear formulations, while dReal provides tolerance-aware feasibility/range and $ε$-argmin checks for continuous nonlinear formulations. We also introduce NLEquiv-150, a public benchmark of 100 equivalent and 50 deliberately hard non-equivalent nonlinear reformulation pairs. With LLM-extracted mappings, SOVER classifies 149/150 pairs (99.33%) correctly, including all 50 hard negatives; the sole error is an incomplete mapping extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.00710v1">Prediction-Assisted Pricing and Admission for LLM APIs with Stochastic Token Consumption</a></div>
    <div class="paper-meta">
      📅 2026-09-01
    </div>
    <details class="paper-abstract">
      An LLM application often sells or internally allocates several service products: a small or premium model, a short or long token cap, and possibly multiple posted prices. The operational decision is not merely which model answers a prompt. A price changes purchase probability, a token cap changes both user value and the tail of resource consumption, and accepted requests compete for shared compute and premium-model capacity. Demand and output length are initially uncertain, while an offline model may provide useful but imperfect predictions. We formulate sequential pricing and admission with stochastic resource consumption. Each arriving request belongs to an observable segment. The platform chooses a product--price pair or makes no offer; purchase, revenue, and resource use are then random. An offline predictor supplies a uniform, validated error radius for every segment--product cell. We propose Prediction-Clipped UCB (PCUCB), which intersects the offline prediction interval with an online confidence interval, evaluates products using resource shadow prices, and reserves a sample-path envelope before commitment. The prior gives a fast start when accurate, while online learning protects the platform when predictions are coarse. The analysis is modular. On a simultaneous confidence event, regret against a buffered fluid benchmark is bounded by a pacing term plus the cumulative diameter of the intersected intervals. For $J$ segment-product cells and prediction radius $\varepsilon$, this yields \[ \widetilde O\left( \sqrt{T}+(1+\barΛ) \min\{T\varepsilon,\sqrt{JT}\} \right), \] where $\barΛ$ bounds operational shadow prices. Thus the algorithm smoothly interpolates between an almost full-information regime and learning from scratch. Hard feasibility holds on every sample path through reservation envelopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17506v2">Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 25 pages, 17 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we find that our task evades safety guardrails by surfacing bias in model judgment. It varies systematically across target groups, reflects implicit social maps, and shows how models are still triggered by demographic labels. Our work points to the need for LLM bias evaluation in judgment tasks and broadly, for more theoretically grounded approaches to bias evaluation in NLP. We release our code and model responses at https://github.com/uofthcdslab/second-order-bias.
    </details>
</div>
