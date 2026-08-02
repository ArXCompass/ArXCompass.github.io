# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01240v2">Prompt Framing Distorts Count Based Evaluation of LLM Error Detection: Evidence from Numeric Anchoring</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 15 pages, 6 figures, 12 tables. Preprint under review
    </div>
    <details class="paper-abstract">
      Count-based F1 is widely used as a proxy for LLM error-detection quality, but this paper shows that it can rise dramatically without a corresponding improvement in span localization, a gap termed F1 Inflation. The paper introduces ErrorBench, a controlled stress-test protocol for prompt-induced count distortion. ErrorBench evaluates six contemporary LLMs under five prompt conditions over 4,290 responses from 143 CoNLL-2014 passages. Under CoNLL-2014 M2-style scoring, anchored prompts produce up to 0.79 points of F1 Inflation, and up to 0.96 under strict matching. A 100-passage replication using the official ERRANT 3.0.0 pipeline and multi-reference scoring reproduces the pattern: averaged over six models, the Blind-to-Anchored prompt shift raises Count-F1 by +0.21 while raising multi-reference ERRANT F0.5 by only +0.04. The study finds larger count responses in highly instruction-compliant GPT/Claude systems and smaller responses in the Gemini family under this stress-test protocol. The findings suggest that LLM proofreading and document-review evaluations should avoid pre-populated error counts and should report span-aware metrics alongside count-based metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26599v1">Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 17 pages, 12 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Estimating heterogeneous treatment effects is central to targeted interventions, such as personalized promotions and precision medicine. We focus on the conditional average treatment effect (CATE), a standard estimand for characterizing such heterogeneity. Even under standard identification conditions, finite-sample CATE estimation requires learning the nuisance structure for covariate adjustment and treatment-effect heterogeneity, often together with an effective representation of X. Raw numerical and categorical encodings can leave semantic relations and higher-order interactions implicit, making this joint task locally unstable. A motivating study further shows that this instability appears through partially separable assignment- and heterogeneity-side channels. Building on this observation, we propose CURL (Causal Uncertainty-guided Representation Learning), a plug-in adapter that uses estimator uncertainty to allocate pretrained semantic capacity to locally unstable units. CURL queries a frozen LLM through two role-conditioned prompts, constructs assignment- and heterogeneity-oriented representations from the observed covariates, and routes them through separated pathways. On four benchmarks, CURL improves ten host learners in most settings, while ablation, refinement-dynamics, route-reassignment, and probe analyses support the intended design and roles of the two channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10647v2">The Reliability of LLMs for Medical Diagnosis: An Examination of Consistency, Manipulation, and Contextual Awareness</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      This study evaluated the diagnostic reliability of two Large Language Models (LLMs), Google Gemini 2.0 Flash and OpenAI ChatGPT-4o, across three dimensions: consistency under rephrased inputs, susceptibility to irrelevant prompt content, and responsiveness to added clinical context. We designed 52 clinical scenarios and modified each under controlled conditions. For consistency, scenarios were rephrased with demographic, wording, and examination changes that preserved the diagnostic core. And the susceptibility was evaluated through embedding irrelevant but plausible narrative details while keeping the clinical evidence unchanged. For contextual awareness, patient history, lifestyle data, or diagnostic findings were added to shift the expected diagnosis. Physician reviewers then judged whether context-driven changes were clinically appropriate. Both models returned identical diagnoses across all equivalent variants and repeated queries (100% consistency). When irrelevant details were added, Gemini changed its diagnosis in 40.0% of cases and ChatGPT in 30.0%. ChatGPT responded to context more often than Gemini (77.8% vs. 55.6%), but a larger share of its changes were clinically inappropriate (33.3% vs. 22.2%). Gemini's context-driven changes were more often judged appropriate (66.7% vs. 55.6%). Consistency under controlled inputs did not protect either model from irrelevant manipulation or unjustified diagnostic shifts when context changed. Before LLMs can separate relevant from irrelevant input and flag insufficient evidence, their diagnostic use requires clinician oversight and structured safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26571v1">From Tokens to Watt-hours: Analytical Energy Estimation for LLM Inference on Modern GPUs</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 20 pages, 3 figures, 6 tables. Accepted for oral presentation at the GREEN-AI Workshop, co-located with ECML-PKDD 2026
    </div>
    <details class="paper-abstract">
      The operational energy consumption of large language model (LLM) inference is becoming an increasingly important component of the environmental footprint of deployed AI systems. However, direct measurement of inference energy often requires hardware telemetry, power instrumentation, or infrastructure-specific monitoring, limiting its applicability in comparative studies, early-stage system design, and sustainability reporting. This report presents an analytically structured, empirically calibrated, GPU-level methodology for estimating LLM inference energy on NVIDIA H100-class accelerators without direct runtime measurement. The proposed estimator combines parameter-scaled transformer FLOP accounting, calibrated memory-traffic factors, and hardware-specific energy coefficients for FP16/BF16 tensor-core computation and high-bandwidth-memory movement. It explicitly separates prompt prefill from autoregressive decoding, enabling energy estimates for input tokens, output tokens, and complete inference requests. The methodology further decomposes total energy into compute, parameter-access, key-value-cache write, and attention-read components, allowing the scaling behavior with model size, context length, and generated-token count to be analyzed. The resulting estimates are not intended to replace physical power measurements; rather, they provide transparent, reproducible, and assumption-explicit approximations suitable for model comparison, green-coding analysis, and design-time evaluation of LLM inference workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26553v1">ThinkOmni: A Reasoning-Driven Omni-Modal LLM Framework for Audio Forgery Detection and Localization</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted by ACM MM 2026, 21 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Existing audio forgery detection and localization (AFDL) methods often overfit dataset-specific low-level artifacts, limiting their generalization to subtle, localized, and unseen manipulations. Recent audio large language model (ALLM)-based approaches cast AFDL as question answering but still model forensic evidence implicitly, without linking manipulation cues to predictions. To bridge this gap, we propose ThinkOmni, a reasoning-driven omni-modal large language model that jointly performs explicit forensic reasoning, spoofing detection, and temporal manipulation localization. To enable explicit reasoning supervision, we construct Forensic-Aware Chain-of-Thought (FACoT), a 100K-sample dataset with structured forensic evidence and reasoning annotations. Leveraging FACoT, we introduce Forensic-Aware Modality-Incremental Learning (FMIL), which progressively aligns semantic, acoustic, and spectral-visual representations with the LLM backbone to capture complementary forensic cues. We further propose Forensic-Consistent Multi-task Loss (FCML), which combines weighted cross-entropy with an adaptive localization loss to coordinate reasoning generation, spoofing detection, and temporal localization. Extensive experiments show that ThinkOmni achieves strong cross-dataset generalization in both detection and localization. Code, models, data, and inference examples are available at https://beyond0814.github.io/ThinkOmni/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27262v1">LLM-Guided Initialization for Accelerated Hybrid Quantum-Classical Medical Image Classification</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Presented at IAICT 2026, Bali, Indonesia
    </div>
    <details class="paper-abstract">
      Variational quantum algorithms often encounter barren plateaus, where cost gradients decay rapidly with increasing circuit depth, undermining the trainability of parameterized quantum circuits. This paper evaluates AdaInit (Adaptive Initialization), proposed by Zhuang and Cunningham, which uses large language models to propose initial parameters for quantum neural networks. We study a simplified single-query AdaInit variant paired with GPU-accelerated simulation in NVIDIA CUDA-Q and apply it to binary classification on the DMR-IR mammography dataset. AdaInit delivers 14.6 times higher gradient variance at initialization than random initialization (0.0095 vs. 0.0006), producing 160 times faster convergence (1.1s vs. 176 s) while maintaining the same classification accuracy of 61.4 percent. We provide theoretical analysis grounded in the geometry of parameterized circuit landscapes and show empirically that LLM-guided initialization places the optimizer in trainable regions of parameter space. Beyond performance, our results indicate that a single LLM query can yield informative parameters without iterative refinement, suggesting a low-overhead path to improved trainability. The findings validate AdaInit in a medical imaging setting and demonstrate its compatibility with GPU-accelerated quantum backends for practical speedups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26541v1">Prosody-driven Jailbreaks in Audio LLMs: A Controlled Study and Mechanistic Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted at ACM Multimedia 2026 (ACM MM '26). 9 pages, 3 figures. Supplementary material included
    </div>
    <details class="paper-abstract">
      Audio-capable foundation models enable end-to-end spoken interaction, but they also introduce safety risks beyond transcript content. It remains unclear how much jailbreak capability can arise from matched-text variation in speech delivery rather than from lexical rewriting or broader style transfer. We study this question by holding transcript content fixed and varying six speech-delivery presets whose acoustic attributes may co-vary. We present PJ-Break, a black-box evaluation protocol with presets targeting arousal, authority, and speaking rate, together with AdvAudio-Prosody, a 600-sample benchmark with acoustically verified attributes. On the exact post-QC Qwen2-Audio panel, the Q=1 Panic (38/95), Anger (35/95), and Fast (32/95) presets are all well above Neutral (4/95). The fixed six-query pool covers 44/95 Qwen2-Audio seeds and 15/95 GPT-4o seeds and exceeds a matched-budget StyleBreak reimplementation (27/95) on Qwen2-Audio. A same-voice pool excluding the confounded Commanding condition still reaches 40/95, and a retained-panel ablation shows emotional-delivery audio alone (44/95) is far more effective than emotional text alone (11/95). Exploratory surrogate diagnostics and pilot mitigation observations are secondary, non-core analyses. Overall, matched-text speech delivery should be treated as a first-class factor in Audio LLM safety evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08693v3">Recognition Without Mitigation: Ethical Frameworks in Autonomous Offensive-LLM Agent Research</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted at AutonomousCyber 2026
    </div>
    <details class="paper-abstract">
      Large language models have moved from advising on offensive security to autonomously conducting it. A growing literature presents agents that execute reconnaissance, exploitation, and privilege escalation against real or simulated targets. Such an agent is a deployable, re-pointable capability whose harm potential scales with the underlying model. The papers that introduce it therefore carry an unusual ethical burden, which top security venues have begun to encode as hard policy in 2025-2026 ethics-section mandates. We present a systematic, reproducible audit of ethics-and-risk reporting in this literature. From a pre-registered Scopus query (Channel A, n=35) plus a reproducible forward-snowball of two seed papers via the Semantic Scholar citation graph (Channel B, n=19, all Scopus-absent) we assemble 54 autonomous offensive-LLM penetration-testing prototypes (2023-2026). We score each against a nine-dimension instrument derived both top-down from the Menlo Report, and bottom-up from the 2025-26 venue mandates. Our central result is a recognition-without-mitigation gap: dual-use risk is reported as recognized in 39% of papers but a concrete mitigation is reported in only 7%, roughly a 5:1 gap. Of the papers, 17% are anti-safeguard, reporting the defeat of model safety controls with no countermeasure. The near-universal safeguards reported are research-integrity controls that protect the experiment, not the public; institutional-review (2%) and coordinated-disclosure (6%) practice is almost absent and confined to Channel B. Measured against the new mandates, the corpus defines a pre-regulation baseline: current practice does not meet the substantive requirements. We argue this audit is itself defensive intelligence on the offensive-agent ecosystem, and we distill a minimal containment checklist for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09472v2">LLM-Grounded Dynamic Task Planning with Hierarchical Temporal Logic for Human-Aware Multi-Robot Handover</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted by IROS 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable non-experts to specify open-world multi-robot tasks, but the generated plans are often kinematically infeasible and inefficient in long-horizon settings. Formal methods such as Linear Temporal Logic (LTL) offer correctness and optimality guarantees, yet they are typically offline and scale poorly. To bridge this gap, we propose a neuro-symbolic framework that grounds human instructions into hierarchical LTLf specifications (i.e., LTL on finite traces) and solves the resulting Simultaneous Task Allocation and Planning (STAP) problem. Unlike static approaches, our system handles stochastic environmental changes-such as user motion or updated instructions-through a receding horizon planning (RHP) loop with real-time perception, dynamically refining plans over a hierarchical state space. Experiments in simulation and on real robots demonstrate that our approach significantly outperforms baseline methods in success rate and interaction fluency while reducing replanning overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22148v2">Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 https://github.com/amazon-science/Self-Evolving-Agents-Ratchet
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries, pioneered by Voyager, let frozen LLM agents accumulate reusable knowledge without weight updates, yet recent evaluation shows that LLM-authored skills deliver $+0.0$pp over no-skill baselines while human-curated ones deliver $+16.2$pp: the bottleneck is not skill authoring but lifecycle management. We introduce \textbf{Ratchet}, a single-agent loop in which a frozen LLM writes, retrieves, curates, and retires its own natural-language skills. Ratchet integrates four candidate hygiene mechanisms: outcome-driven retirement, a bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation. On MBPP+ hard-100 with Claude Opus 4.7, Ratchet lifts held-out pass@1 from a $0.258 \pm 0.047$ baseline to a late-window rolling mean of $0.584$ (peak $0.658 \pm 0.042$) across 100 rounds and 3 seeds, a $+0.328 \pm 0.018$ rolling-mean gain where the no-skill control drifts at $+0.002 \pm 0.005$; the same recipe transfers to an agentic solver on SWE-bench Verified ($+0.22$ peak lift over 20~rounds). Eight ablations (A1--A8) reveal that the minimal working recipe is smaller than our design suggests: retirement and the meta-skill authoring prior are load-bearing, while explicit deduplication (canonicalisation, cover-guard) is subsumed by the meta-skill itself. A non-divergence proposition shows that bounded cap and retirement threshold together keep expected performance from drifting below the no-skill floor by more than a fixed margin, bounding library drift rather than guaranteeing improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08286v2">LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 This article has been peer reviewed and formally published in the Journal of Computational Law and Legal Technology (JCLLT). The final Version of Record is available at: https://doi.org/10.47852/bonviewJCLLT62029649
    </div>
    <details class="paper-abstract">
      Legal argument mining aims to identify and classify the functional components of judicial reasoning, such as facts, issues, rules, analysis, and conclusions. Progress in this area is limited by the lack of large-scale, high-quality annotated datasets for U.S. caselaw, particularly at the state level. This paper introduces LAMUS, a sentence-level legal argument mining corpus constructed from U.S. Supreme Court decisions and Texas criminal appellate opinions. The dataset is created using a data-centric pipeline that combines large-scale case collection, LLM-based automatic annotation, and targeted human-in-the-loop quality refinement. We formulate legal argument mining as a six-class sentence classification task and evaluate multiple general-purpose and legal-domain language models under zero-shot, few-shot, and chain-of-thought prompting strategies, with LegalBERT as a supervised baseline. Results show that chain-of-thought prompting substantially improves LLM performance, while domain-specific models exhibit more stable zero-shot behavior. LLM-assisted verification corrects nearly 20% of annotation errors, improving label consistency. Human verification achieves Cohen's Kappa of 0.85, confirming annotation quality. LAMUS provides a scalable resource and empirical insights for future legal NLP research. All code and datasets can be accessed for reproducibility on GitHub at: https://github.com/LavanyaPobbathi/LAMUS/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19576v3">Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN@ICML 2026), Seoul, South Korea. https://github.com/amazon-science/Self-Evolving-Agents-Ratchet
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom (LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)), yet the underlying mechanism has not been isolated. We provide (1) a reproducible trigger: ablations that isolate drift: one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) trace-level diagnostics: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a verified fix: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26491v1">LLMET: Enabling Cross-Layer Evaluation of Emerging M3D Memories for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The energy consumption of Large Language Model (LLM) serving is becoming a major system challenge as deployment scales, driven by hardware power and thermal constraints and rising electricity costs. A key contributor to chip energy dissipation is data movement between limited on-chip cache and off-chip High Bandwidth Memory (HBM). Meanwhile, emerging memory technologies such as monolithic 3D (M3D) integration of cache memories at the Back-End-Of-Line (BEOL) of logic chips enable larger and denser on-chip memories, creating new opportunities to reduce costly off-chip traffic. However, it remains unclear whether continuously scaling on-chip memory using emerging technologies can effectively improve the energy efficiency of LLM serving. To address this gap, we develop LLMET (LLM with Emerging Technology), a validated cross-layer simulation framework, and conduct a comprehensive study on the impact of large-capacity on-chip memory technologies across a broad range of models, applications and platforms. Utilizing M3D technology to expand the L2 cache from 40MB to 1GB yields a 44% reduction in chip energy during the Llama3.1-70B prefill phase with a 16K context window, based on LLMET simulation on a dual NVIDIA A100 GPU setup. On the 8x NVIDIA B200-like platform, extending the L2 cache from 128MB to 4GB saves the prefill energy by up to 24%. For the edge platform and workloads, the decode energy saving reaches 30% when increasing the 8MB cache size to 256MB. These results highlight the promise of ultra-large on-chip memories for energy-efficient LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26475v1">DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      Long-context inference is becoming a fundamental capability for modern LLM serving, especially driven by emerging agentic applications. Yet it faces a severe memory wall that the KV cache scales proportionally with increasing context length and request concurrency. Existing sparse KV cache methods offload most KV entries to host memory and retrieve only the critical KV entries needed by each decoding step. However, they commonly introduce substantial auxiliary states in GPU memory for KV retrieval management. Our measurements show that these often-overlooked auxiliary states introduce significant memory overhead and become a new bottleneck under high-concurrency workloads. In this paper, we present DualDecoder, a lightweight serving system for long-context LLM inference that enables efficient sparse KV cache retrieval from host memory. Our key insight is that the critical KV entries required for decoding the next token can be accurately predicted from the preceding speculated token. This predictability enables KV retrieval to be proactively prefetched and overlapped with decoding computation, effectively eliminating the GPU memory overhead of auxiliary states. To achieve this prefetching efficiently, DualDecoder leverages a novel dual-token decoding pipeline that accurately identifies critical KV entries with negligible computational overhead, and designs a layer-aware transfer schedule to overlap KV prefetching with model computation and a layer-scoped memory manager to reduce the GPU runtime buffer. Experimental results show that DualDecoder improves decoding throughput by up to 2.62$\times$ over state-of-the-art systems while preserving decoding latency and model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07246v3">Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 15 pages, 7 figures. Accepted at KDD 2026
    </div>
    <details class="paper-abstract">
      Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability of large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline produces predictions through an implicit black-box process with limited traceability and explicit justification, and relies on stochastic single-pass inference, resulting in inconsistent and insufficiently robust detections across contexts. To address these limitations, we propose an LLM-as-an-inducer framework that uses an LLM to induce a decision tree for ED, termed TreeED, and ensembles multiple such trees for consensus detection, termed ForestED. Based on prompts derived from data context, decision tree specifications, and output requirements, TreeED queries the LLM to induce a decision tree skeleton whose root-to-leaf paths specify the stepwise procedure for evaluating a sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks, such as format or range constraints; (2) Graph Neural Network (GNN) nodes that capture complex patterns, such as functional dependencies; and (3) leaf nodes that output the final decision as error or clean. ForestED employs uncertainty sampling to obtain multiple informative row subsets and constructs a decision tree for each subset using TreeED. It then applies an Expectation-Maximization-based algorithm to jointly estimate tree reliability and optimize the consensus ED prediction. Experiments demonstrate that our methods are accurate, explainable, and robust, achieving an average F1-score improvement of 16.1% over the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15336v2">Facial-Expression-Aware Prompting for Empathetic LLM Tutoring</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable increasingly capable tutoring-style conversational agents, yet effective tutoring requires sensitivity to learners' affective and cognitive states beyond text alone. Facial expressions provide immediate and practical cues of confusion, frustration, or engagement, but remain underexplored in LLM-driven tutoring. We investigate whether facial-expression-aware signals can improve empathetic tutoring responses through prompt-level integration, without end-to-end retraining. We build a scalable simulated tutoring environment where a student agent exhibits diverse facial behaviors from a large unlabeled human facial expression video dataset, and compare four tutor variants: a text-only LLM baseline, a multimodal baseline using a random facial frame, and two Action Unit estimation model (AUM)-based methods that either inject textual AU descriptions or select a peak-expression frame for visual grounding. Across 960 multi-turn conversations spanning three tutor backbones (GPT-5.1, Claude Opus 4.5, and Gemini 2.5 Pro), we evaluate targeted pairwise comparisons with five human raters and an exhaustive AI evaluator. AU-based conditioning consistently improves empathetic responsiveness to facial expressions across all tutor backbones, while AUM-guided peak-frame selection outperforms random-frame visual input. Textual AU abstraction and peak-frame visual injection show model-dependent advantages. Control analyses show that this improvement does not come at the expense of worse pedagogical clarity or responsiveness to textual cues. Overall, our results show that lightweight, structured facial expression representations can meaningfully enhance empathy in LLM-based tutoring systems with minimal overhead. Future work should evaluate these methods using real student facial data collected during authentic learning interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13317v2">SkillCAT: Contrastive, Assessment-Augmented and Topology-AwareSkill Self-Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Skill self-evolution methods for LLM agents aim to turn execution trajectories into reusable skill documents. However, current pipelines typically derive skill patches from a single trajectory per task, merge them indiscriminately, and load the entire skill corpus during inference. These choices lead to unreliable evidence extraction, the accumulation of low-quality or even harmful skill edits, and inefficient use of context due to irrelevant or conflicting skill content. We propose SkillCAT, a framework that decomposes this process into three stages. (1) Contrastive Causal Extraction (CCE) samples multiple trajectories per task and contrasts same-task success/failure pairs to find the evidence that explains outcome differences. (2) Assessment-Augmented Evolution (AAE) replays each candidate patch on source-task clones, retains only those that do not damage task outcomes, and then merges the retained patches hierarchically. (3) Topology-Aware Task Execution (TTE) compiles the evolved skills into routable sub-skill topologies, so that inference loads only task-relevant capability nodes. We evaluate SkillCAT on widely-used agent benchmarks, including SpreadsheetBench, WikiTableQuestions, and DocVQA, and further assess cross-model and out-of-distribution generalization. Across these settings, SkillCAT improves the average score over the initial skill by up to 49.69%, demonstrating reliable and effective skill evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23442v2">Do LLM Debates Repeat Arguments Differently Across Languages?</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      LLM debate is usually evaluated by final answers, yet transcripts reveal whether later turns develop new arguments or return to earlier claims in new wording. We study this process with \textit{prior-argument similarity}, which compares extracted argument units with earlier units in the same debate. In controlled eight-turn debates over 71 motions, six languages, and four model agents, Chinese is the only tested language with a consistently positive gap relative to English across three multilingual embedding models. The gap persists across agents, turn positions, regression adjustment, metric variants, extraction-length controls, a second-extractor subset, and cross-encoder tail rescoring. Manual calibration shows weak item-level alignment but a high-similarity tail enriched for substantive repetition. A diversity-aware prompt lowers \textit{prior-argument similarity} across languages, yet does not significantly narrow the Chinese--English gap. Multilingual debate evaluation should therefore measure argumentative development over time and report both average and gap terms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26397v1">Knowledge before Reasoning: EC-Reason-Bench, a Training-Free Diagnostic Benchmark for LLM Enzyme Classification</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      Enzyme function prediction is a hierarchical, knowledge-intensive form of protein function classification. Existing benchmarks expose an anomaly: general LLMs often get the coarse first level right, yet once asked for a complete EC number their accuracy at levels two through four drops to almost zero, while specialized models and tools stay usable. We propose EC-Reason-Bench, a training-free, diagnostic evaluation protocol built to answer two questions: why general LLMs score close to nothing on EC number prediction, and how much of that loss can be recovered without updating a single weight. We break enzyme classification ability into four orthogonal levers that can each be measured on their own: output structure, external knowledge, reasoning structure, and reasoning robustness. We test each lever with an inference-time method against a shared zero-shot baseline reproducing previously reported near-zero performance. Experiments with several strong reasoning LLMs yield four main findings. First, external knowledge is decisive and must precede reasoning: uniformly low closed-book performance rises sharply with open-book access, narrowing model gaps. Second, in closed-book settings, whether cascading and chain-of-thought help or hurt depends on a model's tendency to abstain. Third, once evidence is available the aggregate score of the best LLM setting is indistinguishable from simply voting the EC numbers of the nearest retrieved neighbors; that tie is an artifact of averaging, and it hides a large gain on adversarial evidence set against an equally large loss on multi-functional enzymes. Reasoning over evidence therefore acts as an arbiter of conflicting neighbors rather than as a source of knowledge, and no single-number leaderboard can see it. Fourth, accuracy obeys a law of homology availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12250v2">How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 11 pages, 4 figures and 2 tables
    </div>
    <details class="paper-abstract">
      This study examines how memory shapes the collective and cooperative dynamics of Large Language Model (LLM) agents in a multi-agent system. To this end, we extend the Social Particle Swarm (SPS) model, in which agents move in a two-dimensional space and play the Prisoner's Dilemma with neighboring agents, by replacing its rule-based agents with LLM agents endowed with Big Five personality scores and varying memory lengths. Using Gemini 2.0 Flash, we find that memory length is a critical parameter governing collective behavior: even a minimal memory drastically suppressed cooperation, transitioning the system from stable cooperative clusters through cyclical formation and collapse of clusters to a state of scattered defection as memory length increased. Big Five personality traits correlated with agent behaviors in partial agreement with findings from experiments with human participants, supporting the validity of the model. This effect of memory appeared whether or not personality was assigned. With heterogeneous personalities, individual behavior reflected the assigned traits and cooperation collapsed under long memory, whereas without personality Gemini's cooperative disposition dominated and cooperation was broadly maintained. Sentiment analysis of agents' reasoning texts showed that the model interprets memory increasingly negatively as its length grows, already in the early phase, providing a micro-level account of the suppression of cooperation. These results suggest that how an LLM interprets accumulated memory is a key driver of emergent social behavior in Generative Agent-Based Modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26355v1">Symphony of Bias: Exploring Gender Associations with Musical Instruments in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 32 pages, 23 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly embedded in everyday life and widely used for information seeking, raising concerns about their potential to perpetuate social biases and reinforce stereotypes. In this study, we investigate gender bias in LLMs through the lens of their associations with musical instruments. Building on social-science research on the cultural gender-typing of instruments, we introduce Symphony-Bias, a parallel multimodal dataset spanning text, vision, and audio. We evaluate ten multimodal models with diverse architectures and scales across 22 musical instruments, analyzing how they associate each instrument with three gender categories: {male, female, non-binary}, across three modalities: {text, vision, audio}. Our results show that 92\% of instrument-level outcomes align with prior social-science findings, with the harp and drums showing particularly consistent gendered associations across all evaluated models and modalities. We further find that alignment with social stereotypes is weakest in audio, stronger in vision, and strongest in text, suggesting that modality-specific representations can differentially amplify gendered associations with musical instruments.\footnote{The Symphony-Bias dataset will be publicly released upon acceptance of the paper.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26348v1">When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 19 pages, 5 figures, 4 tables. Preprint; under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as synthetic users, stand-ins for human respondents whose simulated answers feed product, policy, and market decisions. We ask when this substitution is valid and when it fails, and package the answer as an evaluation framework for intelligent synthetic-user systems. A single protocol, run across four models spanning two families and an 8B-to-frontier capability range, is applied to two independent domains of real human-response data: U.S. general social attitudes (General Social Survey) and cross-cultural values (World Values Survey). Every model is benchmarked against a suite of non-LLM baselines fit on held-out human data. Under demographic prompting and the survey-simulation protocols we test, two failures replicate across both domains, all four models, and both families. First, at the individual level no LLM beats even the strongest baseline; on cross-cultural values every model falls well below it, and the gap survives distance-aware and proper scoring. Second, models systematically over-determine demographics, treating identity as far more predictive of attitudes than it is among real people, a distortion present for nearly every question-group combination and robust to a coding-invariant measure. Neither failure is remedied by a larger, more capable model. A decision-impact analysis shows why this matters in practice: on a segment-targeting task the models inflate between-segment gaps two to fourfold, would direct a team to the wrong segment in half of U.S. and most cross-cultural cases, and manufacture segment splits that do not exist in real people. We make the cross-domain benchmark and the evaluation framework available on request, so that teams can determine in advance when synthetic-user evidence is safe for decision support and when it is not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05682v2">FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Withdraw for further improvement and work consolidation
    </div>
    <details class="paper-abstract">
      LLM systems for scientific discovery increasingly assist with ideation, literature synthesis, experiment planning, and report generation, but the first research question they propose can remain difficult to audit: it may sound plausible without exposing the mechanism, falsifier, or assumption that a scientist should inspect. We introduce FirstResearch, a first-principles research-question formation framework for scientific LLM agents whose core artifact is a structured Research Question Certificate. The certificate records primitive definitions, assumptions, a mechanism model, a tension or contradiction, a falsifiable hypothesis, a minimal decisive test, and a failure update rule, making the proposed question inspectable before downstream execution. On ten LLM-agent research topics, FirstResearch outperforms controlled prompt-level baselines inspired by AI co-scientist, Agent Laboratory, and AI Scientist-v2 under a primary DeepSeek-blind-judge protocol. A Gemini-2.5-Flash independent-judge rescore of the same 40 baseline packages preserves the system-level ranking, with FirstResearch scoring 4.86/5 versus 4.38/5 for the strongest baseline and Pearson agreement of 0.865 on average score. A one-repeat ablation checkpoint further suggests that the certificate-centered core is the strongest component: certificate-only scoring reaches 4.90/5 under DeepSeek and 4.88/5 under Gemini, while removing certificates drops below 1/5 under both judges. These results are preliminary and use LLM judges rather than human domain experts, but they support a narrow scientific-discovery claim: explicit derivation constraints are a promising mechanism for making LLM-generated scientific questions more auditable. Code, prompts, saved outputs, and reproduction scripts are available at https://github.com/louiswang524/FirstResearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25112v3">Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 12 pages, 4 figures, 3 tables. v3 corrects a differential length bias in the automated correctness scorer, validated against 1,830 human adjudications; all analyses recomputed. The inverse accuracy-efficiency coupling in v1/v2 does not survive relabelling; see the version note on page 1. Pre-registered. Code and annotations: github.com/synthiumjp/metacognition-audit
    </div>
    <details class="paper-abstract">
      Standard evaluation of LLM confidence relies on calibration metrics (ECE, Brier score) that conflate two capacities: how much a model knows (Type-1 accuracy) and how well its confidence signal tracks that knowledge (Type-2 metacognitive sensitivity). We apply Signal Detection Theory to decompose them, treating token-level normalised log-probability as a graded confidence variable and answer correctness as the state to be discriminated. We characterise the Type-2 ROC of this signal, including its unequal-variance structure via z-ROC analysis, and -- because the meta-d' efficiency ratio is not well defined for open-ended QA, which lacks a two-alternative Type-1 decision -- quantify efficiency with a model-free information measure, normalised metacognitive information (meta-I_2r). Across four LLMs and 224,000 factual QA trials we find: (1) metacognitive information varies by a factor of 1.98 across models and is not predicted by accuracy, the rank correlation being -0.80 on TriviaQA and +0.00 on Natural Questions; (2) the confidence signal has model-specific unequal-variance structure (z-ROC slopes 0.78 to 1.18) invisible to calibration metrics, the slope ordering replicating on NQ; (3) efficiency is domain-specific, weakest in Science & Technology for every model; (4) temperature dissociates accuracy from metacognitive information, which stays near-flat for three of four models while accuracy falls; and (5) metacognitive information tracks the accuracy gain from confidence-based abstention exactly (rho = +1.00) while accuracy does not. All estimates carry permutation nulls and bootstrap confidence intervals. This version (v3) corrects a differential length bias in the automated correctness scorer, validated against 1,830 human adjudications; the inverse accuracy-efficiency coupling reported in v1 and v2 does not survive relabelling. See the version note on page 1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26317v1">Aligning LLM-Simulated and Human Examinees for Psychometric Calibration: A Cognitive Diagnostic Profiling Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Psychometric calibration for educational tests typically requires costly human response data. Large language models (LLMs) simulated examinees offer a promising route to early calibration, but their responses are too accurate and too uniform. We propose Cognitive Diagnostic Profiling (CDP), a zero-shot framework that prompts LLMs to simulate plausible examinees with diverse cognitive profiles: binary attribute-mastery patterns are rendered as natural-language profiles and sampled under an uninformative or an informative distribution. Using the Tatsuoka fraction-subtraction dataset (536 examinees, 15 items, five attributes), we evaluated eight LLM configurations under no-profile, uninformative-CDP, and informative-CDP conditions, assessing alignment with human examinees at the ability-distribution, mastery-profile, and item-difficulty levels. CDP improved all three levels: distributional overlap rose across configurations; weighted correlations between profile-level scores and human profile expectations reached 0.92 to 0.98; and item-difficulty recovery improved in rank order and absolute alignment, most for reasoning-enabled models; in the strongest case, Gemini 3.0 Flash (Thinking), one-parameter logistic (1PL) difficulty Spearman correlations rose from 0.24 to 0.86 and 0.90 and the root-mean-square error (RMSE) fell from 6.31 to 1.30 and 0.90; the informative condition helped most where profile-level alignment was strong. CDP brings LLM-simulated examinees into closer psychometric alignment with human examinees, making them practical for operational test development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26286v1">Evaluating Prompt Scope and Demonstration Similarity in Local LLM Machine Translation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as general-purpose translation systems, but their behavior is usually evaluated under a single prompt shape: translate one source sentence into one target language. In practice, users may ask for one target language, for several related languages at once, or for translations conditioned on examples. This paper studies prompt scope and demonstration selection as experimental variables for local LLM machine translation. We evaluate English-to-Romance and English-to-Germanic translation on the full FLORES devtest split for nine official European Union languages. We compare three local instruction-tuned LLMs, llama3.2:3b, mistral:latest, and qwen2.5:14b, against dedicated MT baselines from OPUS-MT and NLLB-200. We test zero-shot prompting and k=5 few-shot prompting with random, lexical-similarity, and embedding-similarity demonstration selection. We also compare single-target prompts with JSON-formatted family-scope prompts that request all languages in a family at once. Results show that dedicated MT systems remain strongest overall, especially for Germanic languages. Few-shot prompting helps mistral:latest and qwen2.5:14b, but hurts llama3.2:3b; embedding retrieval is best on average for the stronger LLMs, but its advantage over random and lexical examples is modest. Family-scope prompting is feasible for stronger local LLMs but exposes structured-output failures in smaller models. These findings motivate evaluating LLM translation not only by language pair and metric, but also by prompt scope, retrieval strategy, and multi-target compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08065v2">When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      LLM-as-judge (Zheng et al., 2023) is increasingly the default for evaluating AI systems in enterprise pipelines, often scaled to ensembles (Verga et al., 2024) or "mixture-of-experts" (Shazeer et al., 2017) panels of judges. These systems share a key assumption: that consistency -- agreement among judges, or among a model's own samples -- indicates correctness. We show this assumption is unreliable. Agreement is not accuracy: a model can agree with itself, and different models can agree with each other, out of shared bias, a memorized heuristic, or an option-position prior rather than truth. We ask when agreement is nonetheless a usable proxy, in a large-scale cross-runner study: 53 runners drew K=50 samples for assigned overlapping cases across comparisons of model tier, prompting, and scale on GPQA Diamond and AIME -- 265,000 samples. Using majority-correctness as the deployment label and a hierarchical runner-clustered bootstrap, agreement is a positive but weak predictor (rho 0.20-0.59, all positive under item-clustered resampling) whose usefulness is regime-dependent: best for unsaturated mid-tier models and for allocating compute, and worst -- over-confident yet no more accurate -- for the most consistent frontier model (agreement >=0.8 on 77% of GPQA case-result entries, 48% of those wrong). An exploratory cross-family check on three Claude tiers shows the same frontier over-confidence, with confident errors recurring across providers above a marginal-preserving null. Self-consistency is thus a conditional proxy for correctness, not a standalone confidence score. We publicly release the de-identified per-run rows and answer distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26160v1">GuideSkill: Evolving Executable LLM Agent Skills for Guideline-Grounded Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Clinical practice guidelines (CPGs) encode diagnostic criteria, but LLM systems typically retrieve guideline text or absorb it through training rather than execute its rules. We introduce GuideSkill, an external reasoning layer that compiles disease-specific criteria into executable functions returning ordinal diagnostic-support scores. GuideSkill-Zero is initialized from guidelines, while GuideSkill-Evo uses case--diagnosis pairs to refine covered skills and add missing diagnoses. At inference, an LLM proposes a differential diagnosis, grounds the features required by each matched skill, and fuses its ranking with the executed skill scores. Across four benchmarks and four backbones, GuideSkill-Zero improves macro-average accuracy over guideline RAG by 13.45% on average. GuideSkill-Evo achieves the highest macro-average for every backbone, improves over direct inference by 18.49% relatively, and increases gold-label skill coverage from 56.5% to 99.5%. On Qwen3.5-9B, it also exceeds the strongest parameter-update baseline by 11.16% without updating the backbone. Expert evaluation further indicates that GuideSkill produces clinically sound and broadly acceptable skills, suggesting that its initialized and evolved rules are reliable and practically meaningful. These results support executable skills as a model-agnostic mechanism for combining guideline-derived procedures with case-derived diagnostic patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.07314v4">MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Published in Transactions on Machine Learning Research (06/2026)
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve superhuman performance on standardized medical licensing exams, these static benchmarks have become saturated and increasingly disconnected from the functional requirements of clinical workflows. To bridge the gap between theoretical capability and verified utility, we introduce MEDIC, a comprehensive evaluation framework establishing leading indicators of clinical LLM competence across five dimensions. These upfront indicators reveal cross-benchmark capability gaps, such as the divergence between static knowledge retrieval and functional execution, that inform model selection before costly deployment-based evaluation. Beyond standard question-answering, we assess operational capabilities using deterministic execution protocols and a novel Cross-Examination Framework (CEF), which quantifies information fidelity and hallucination rates without reliance on reference texts. Our evaluation across a heterogeneous task suite exposes critical performance trade-offs: we identify a significant knowledge-execution gap, where proficiency in static retrieval does not predict success in operational tasks such as clinical calculation or SQL generation. Furthermore, we observe a divergence between passive safety (refusal) and active safety (error detection), revealing that models fine-tuned for high refusal rates often fail to reliably audit clinical documentation for factual accuracy. These findings demonstrate that no single architecture dominates across all dimensions, highlighting the necessity of a portfolio approach to clinical model deployment. We accompany this work with a publicly available MEDIC leaderboard at https://hf.co/spaces/m42-health/MEDIC-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26120v1">Even More Deception: Objective Misalignment in Mixed-Motive LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Accepted at AIWILD@ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered multi-agent systems are increasingly deployed in mixed-motive environments, where agents operate under asymmetric information and strategic deception due to conflicting or hidden objectives. In these settings, misalignment with collective goals becomes a central concern. We propose a novel framework for evaluating objective misalignment using the social deduction game Werewolf, modifying the objective of a single agent while preserving its assigned role. Across LLMs from four different model families and sizes, four player roles, and three objective formulations, we introduce a dual analysis of the agents' internal reasoning and their public cheap-talk behavior (i.e costless, non-binding communication that does not directly affect the agents' utilities), complemented by an analysis of game outcomes. Our results show that objective misalignment undermines outcomes in inherently adversarial environments, an effect exacerbated by asymmetric information and specialized roles. While compromised agents consistently develop distinct objective-dependent reasoning strategies, these adaptations remain largely invisible in their public behavior. More broadly, our findings suggest that even subtle objective misalignment can profoundly affect collective decision-making, highlighting the need for effective mitigation strategies for LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25995v1">Does Runtime Topology Context Improve LLM-Generated Kubernetes Security Patches?</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Accepted at the Workshop on the Use of Large Language Models for Cybersecurity (LLMSec), co-located with ESORICS 2026
    </div>
    <details class="paper-abstract">
      Kubernetes is central to the cloud-native ecosystem, orchestrating containerised workloads. Recent work suggests that large language models (LLMs) can automate cluster security remediation, generating configuration patches from Kubernetes Security Posture Management (KSPM) findings without human authoring. Such systems, however, prompt the model with each finding in isolation from the live service call graph, assuming general hardening knowledge suffices. This assumption breaks down whenever a patch must preserve a runtime service dependency invisible to the model: an otherwise compliant fix then carries a destructive functional blast radius, crashing downstream callers or silently severing call edges across the cluster. Whether live cluster context improves patch correctness has not been measured under controlled conditions across multiple dependency classes. We introduce KuTIE (Kubernetes Topology Intelligence Engine), which builds a live cluster context from Istio call edges, Trivy KSPM findings, and the service-account bindings a workload reads, and conditions LLM patch generation on it. It is evaluated on VulnCare, a purpose-built 36-deployment, four-namespace healthcare cluster with 31 injectable findings across seven dependency classes, each labelled by topology dependence against cluster ground truth. Across 248 trials, topology context raises topology-dependent patch correctness from 11.1% to 78.0% ($Δ= 0.669$), a gap that holds for every model and for six of seven classes, from credential and network-policy ($Δ= 0.95$) to role-based access control ($Δ= 0.31$); a topology-independent control exhibits no such effect ($Δ= 0.0$), isolating the result from generic prompt enrichment. Supplying the live service-call graph and the service-account bindings it exposes thus improves remediation of topology-dependent findings well beyond scanner-only context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25992v1">MemLens: A Value-Aware Memory Management System with Interactive Analytics for LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Recently, memory management has become a key infrastructure for LLM-based agents, as it directly affects long-horizon reasoning, personalized responses, and knowledge reuse. However, existing LLM memory systems typically adopt a coarse-grained (utility-agnostic) manner that treats heterogeneous user-LLM interaction records uniformly, leading to redundant and low-impact records persisting in the memory repository. To address this challenge, we present MemLens, a value-aware memory management system that takes memory records as first-class data objects. MemLens provides an end-to-end interactive analytics dashboard that exposes the complete memory lifecycle, including Shapley-style memory evaluation, value-aware storage, and memory-assisted response. Through a study-copilot application, the system enables users to inspect memory values, visualize hierarchical memory structures, and compare various memory management strategies in terms of response quality, retrieval latency, and token consumption. Therefore, our MemLens can serve as an efficient, interpretable, and personalized long-term memory management system for LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25953v1">Polistemics: Evaluating LLMs as Information Mediators in Politics & Elections</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      As LLMs increasingly mediate the political information citizens rely on, there is still no standardized way to assess whether they do so responsibly. We introduce Polistemics, a theory-grounded benchmark for evaluating LLMs as mediators of political information in elections. Prior work has treated this task as reproduction rather than mediation, leaving its epistemic dimensions and interaction with imperfect information unaddressed. We ground the evaluation in Epistemic Modesty, a normative standard derived from citizens' epistemic agency, and test it across controlled settings that vary informational properties such as clarity, noise, and consistency. Applying the benchmark to three state-of-the-art LLMs on the 2025 German and Dutch elections, we find that high aggregate scores mask systematic failures. Models mediate reliably under clear evidence but break down under absent, vague, or contradictory information, while flattening the intensity of political language. These failures are likely driven by party priors, influenced by party labels and output language. Reliable mediation appears achievable, but no model delivers it consistently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25947v1">A Cost-Effective Multimodal LLM Reasoning Framework for Question Answering over Irregular Clinical Time Series</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Question answering (QA) over irregular clinical time series (ICTS) plays a pivotal role in a wide range of healthcare applications. Although recent multimodal time-series large language models (LLMs) have shown considerable promise in general-purpose time-series QA, they remain poorly equipped to model the sparsity, asynchrony, and irregular sampling patterns of clinical observations. To fill this gap, we propose ClinPRISM, a cost-effective multimodal LLM reasoning framework for question answering over ICTS data. First, we devise an irregularity-aware multi-scale encoder to capture sparse clinical evidence at diverse temporal scales. Then, we propose a temporal evidence distiller to integrate representations across these scales and compress them into a small number of LLM-compatible tokens. Moreover, we introduce a progressive alignment strategy that sequentially aligns the irregular trajectories with the LLM's textual embedding space. To facilitate training, we construct 30,000 clinical time series paired with multi-scale descriptions, together with 41,000 instruction-tuning instances spanning 11 tasks. Using a 4-billion-parameter LLM backbone, ClinPRISM achieves state-of-the-art performance on the held-out evaluation benchmark while using only 16 time-series tokens and achieving an average inference latency of 0.15 seconds per question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25936v1">From Role Prompt to Infinite Thinking: Exploiting Persona Conditioning for Inference Cost Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 17pages
    </div>
    <details class="paper-abstract">
      LLMs are increasingly deployed in real-world applications, making inference efficiency and service reliability critical concerns due to their substantial computational costs. However, the autoregressive generation mechanism of LLMs enables malicious prompts to manipulate generation behaviors, inducing excessive token generation that amplifies computational consumption and threatens service efficiency. Existing methods mainly rely on adversarial suffixes or explicit extension instructions, which introduce detectable behaviors and limit their applicability. In this paper, we reveal a previously unexplored vulnerability caused by persona consistency in LLMs, where models maintain assigned roles and reproduce corresponding behaviors even when they result in inefficient reasoning and excessive generation. Based on this observation, we propose RolePlay, a task-aware dynamic persona alignment framework that constructs adaptive personas to naturally induce inefficient yet semantically coherent behaviors for inference cost amplification. Extensive experiments across multiple LLMs and diverse task datasets demonstrate that RolePlay consistently outperforms existing inference extension methods, achieving an average token amplification of up to \bm{$7.64\times$} and a maximum token amplification ratio of \bm{$207.64\times$}. Our findings identify persona conditioning as a new attack surface for LLM inference efficiency and offer a new perspective on computational cost amplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24539v4">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Corrected small naming inconsistencies for Level 0, 1, and 2 analysis
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25880v1">Stemma: Induced Decision Regions Reveal LLM Provenance</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 25 pages, 4 figures
    </div>
    <details class="paper-abstract">
      LLM provenance testing asks whether a suspect LLM belongs to the same lineage as a source. Existing black-box methods largely infer this relationship from response-level characteristics, but these characteristics may shift under adaptation or deployment even when the underlying meaning remains unchanged, weakening the reliability of provenance evidence. To address this limitation, we introduce induced decision regions by mapping open-ended outputs into a finite decision space, thereby abstracting away surface-form variation and reframing provenance testing as measuring the inheritance of decision regions. Empirical analysis shows that the source's induced regions are preserved more strongly in related models than in unrelated models. Building on this signal, we propose Stemma, a practical black-box LLM fingerprinting method that operationalises stability, robustness, and specificity as complementary probe-selection principles for reliably estimating induced decision region inheritance. Across 770 source-suspect pairs drawn from 56 public checkpoints and spanning diverse model-weight transformations, Stemma achieves 0.967 AUC and 87.8% TPR at 1% FPR, substantially outperforming four representative baselines. It further achieves 0.995 AUC and 93.5% TPR at 1% FPR on 1,260 pairs covering 91 deployment instances, demonstrating robustness to diverse inference-time deployment settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25877v1">Runtime Uncertainty Monitoring for LLM-Based Multi-Agent Systems Using Bayesian Networks</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 This paper got accepted for Ninth International Workshop on Artificial Intelligence Safety Engineering (WAISE 2026): https://www.waise.org/
    </div>
    <details class="paper-abstract">
      This paper investigates how multi-agent systems (MAS)-based on large language models (LLMs) can support actuarial risk modelling, with a particular focus on uncertainty quantification. Actuarial workflows represent a high-stakes decision-support setting where unreliable outputs may lead to incorrect risk assessment, unfair pricing, and regulatory non-compliance. To address uncertainty introduced by the probabilistic nature of LLMs and dependencies between agents, a multi-agent framework is proposed in which specialised agents perform data preparation, modelling, review, and explanation tasks under a central hub. The main contribution is a novel approach to uncertainty propagation using token-level log-probabilities and a Bayesian Network. Importantly, log probabilities are not treated as direct probabilities of correctness or task success. Instead, length-normalised log-probability summaries are transformed into calibrated task-level confidence estimates before incorporation into the Bayesian Network. Results show that the framework reproduces baseline actuarial performance while providing additional insight into workflow stability and runtime uncertainty propagation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25873v1">How Do LLMs Read Bug Reports? An Empirical Study of Attention in LLMs for Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Accepted at the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE) 2026 Conference
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based Automated Program Repair systems are advancing rapidly, yet their performance remains inconsistent. Even when provided with the same contextual information, an LLM may generate a correct patch for one bug but fail on another closely related bug. Why this happens remains poorly understood, and it is unclear how LLMs prioritize the diverse information in bug reports and whether model attention affects repair success. In this paper, we present the first empirical study of attention patterns in LLM-based program repair, providing interpretable insights into how models process bug reports and where their attention is concentrated during repair. We analyze 319 real-world Python and Java bugs from SWE-bench Verified and Multi-SWE-bench to study (RQ1) how model attention is distributed across bug report sections, (RQ2) how attention patterns within each section differ between successful and unsuccessful repairs, and (RQ3) how these patterns compare to information developers consider important for bug fixing. We find that successful repairs are characterized by diffused attention across multiple diagnostic components such as bug descriptions, stacktraces, and test cases, while failures often exhibit over-localized attention toward metadata such as version information. We further observe that stronger alignment between model attention and developer-identified key sections and phrases is associated with higher repair success. Our results provide the first empirical evidence that attention misallocation is a key factor in LLM-based APR failures, and offer actionable insights for designing more interpretable and reliable future APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25853v1">HiSkill: Empowering LLM Agents with Hierarchical Skill Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Skills have become an important abstraction for enabling large language model (LLM) agents to reuse past experience in long-horizon interactive tasks. However, existing trajectory-to-skill methods often produce flat collections of high-level textual skills that are stored and retrieved independently, leaving skill relations underutilized and maintaining a gap between high-level skills and executable actions. In this paper, we propose HiSkill, a hierarchical skill graph framework that organizes interaction trajectories into a directed graph with skill nodes, AtomicOp nodes, and typed edges. Specifically, the graph connects reusable high-level skills with executable action templates, while also capturing decomposition, temporal transition, compatibility, support, and recovery relations among them. At inference time, HiSkill retrieves a compact task-relevant subgraph and performs subgraph-guided task execution, where a symbolic task state, an active skill, and the retrieved subgraph guide the LLM agent to switch skills, select AtomicOps, and ground executable actions iteratively. Experiments on three interactive environments show that HiSkill outperforms state-of-the-art baselines while reducing inference token consumption, demonstrating the effectiveness of bridging high-level skills and executable action grounding through a hierarchical skill graph. Our data and code is available at https://github.com/BUPT-GAMMA/HiSkill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25820v1">Food Image Segmentation with LLM-Derived Ingredient Labels and Multimodal Fusion</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Food image segmentation plays a vital role in health-related applications such as nutrition tracking and personalized health monitoring. However, existing models often underperform on visually similar ingredients and rare food categories. To address this issue, we propose two plug-and-play multimodal modules that enhance the segmentation performance by leveraging ingredient labels inferred from food images using large language models (LLMs). The first module, called LIM-F (Language Injection Module for Features), is designed to pair with any image encoder that produces multi-layer outputs (e.g., Swin Transformer), while the second module, LIM-Q (Language Injection Module for Queries), targets Mask2Former-style Transformer-based decoders. Both modules enable training without the need for pre-aligning images with text by directly injecting semantic ingredient information into the visual analysis pipeline. On the FoodSeg103 benchmark, the proposed method achieves state-of-the-art performance. Specifically, integrating LIM-Q into the Mask2Former decoder with a Swin-L image encoder yields a mean Intersection over Union (mIoU) of 55.0. LIM-F also demonstrates strong generalization and competitive performance, reaching an mIoU of 54.4 under the same model (Swin-L+Mask2Former). Furthermore, its applicability extends beyond Transformer-based decoders, as evidenced by an improvement from 47.7 to 49.8 mIoU when integrated into a CNN-based architecture. Notably, the improved segmentation accuracy is achieved with only a moderate (at most 3.8 GB) increase in the GPU memory consumption during training. Thus, the proposed approach offers a practical and scalable solution for fine-grained food understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10282v2">Linear-LLM-SCM: Benchmarking LLMs for Coefficient Elicitation in Linear-Gaussian Causal Models</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 [v2] Accepted at Workshop on Structured Data for Health@ICML 2026 Seoul,South Korea. 19 pages, 8 figures, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown potential in identifying qualitative causal relations, but their ability to perform quantitative causal reasoning---estimating effect sizes that parametrize functional relationships---remains underexplored in continuous domains. We introduce Linear-LLM-SCM, a plug-and-play framework for evaluating LLMs on Linear Gaussian structural causal model parametrization when a directed acyclic graph (DAG) is given. The framework decomposes a DAG into local parent-child sets and prompts an LLM to produce a regression-style structural equation per node, which is aggregated and compared against available ground-truth parameters. Our experiments with seven real-world DAGs effect ground truth illustrate limitations of LLMs as quantitative causal parameterizers. Across most models, we observe variability in coefficient estimates and sensitivity to structural perturbations. We open-sourced the framework to further encourage the community to work on studies toward the use of LLM for causal effect elicitation in safety-critical domain, e.g., healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25651v1">Demystifying Deep Learning Compiler Frontend Bugs: An LLM-Aided Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Deep learning compilers (DLCs) are designed to translate deep learning programs into optimized, hardware-specific code. Typically, DLC frontends translate programs into graph-based intermediate representations (IRs) to enable optimizations. Defects introduced during this stage (termed \emph{fBug}s) are severe yet understudied, as prior work predominantly focuses on low-level APIs and operators or treats DLCs as monolithic entities. To bridge this gap, we conduct the first systematic empirical study of \emph{fBug}s in TorchDynamo, the default DLC frontend for PyTorch 2, the most popular DL framework. Leveraging a domain-knowledge-enhanced LLM-aided methodology, we analyze 123 \emph{fBug}s and construct a taxonomy comprising 7 root cause categories and 15 subcategories. Our findings provide actionable insights for DLC development and testing. Furthermore, we leverage the LLM to generate targeted, root cause-aware test cases to detect new bugs. We uncovered 23 previously unknown \emph{fBug}s in recent releases (15 confirmed) across eight (sub)categories, demonstrating the efficacy of our methodology in testing and hardening DLC frontends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25640v1">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Accepted for publication at the 20th ACM Conference on Recommender Systems (ACM-RecSys 2026)
    </div>
    <details class="paper-abstract">
      Conversational Recommendation Systems (CRS) aim to achieve two primary objectives: recommending relevant items and generating natural language responses. While recommendation accuracy is effectively measured by established ranking metrics, the evaluation of response generation poses a more fundamental challenge. Although human evaluation remains the gold standard, its cost and scalability constraints have motivated the adoption of LLM-as-a-judge as a promising proxy, whose alignment with human judgment in the context of CRS remains an open question. In this paper, we present the first user study to empirically assess the reliability of LLM-as-a-judge for evaluating CRS responses. We sample 20 multi-turn music recommendation sessions and generate candidate system responses using four instruction-tuned LLMs, inducing variance in response quality across model scales. We collect $n{=}400$ ratings from 20 domain-expert annotators, who evaluate each response across two dimensions: Personalization Quality and Explanation Quality. Through bootstrapped correlation analysis, we find that LLM-based judges exhibit moderate positive alignment with human assessments and outperform all reference-based baselines. Furthermore, we analyze how judge performance varies according to model scale and conditioning information, providing practical guidance for deploying LLM-as-a-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25630v1">A Human-in-the-Loop Corpus for LLM-Based Simplification of Scientific Summaries</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Accepted at FGWM@KI2026
    </div>
    <details class="paper-abstract">
      Interdisciplinary research is accelerating, yet scientific papers remain difficult to understand outside their home fields. We study large language model (LLM)-based simplification of scientific texts and present a human-in-the-loop workflow that transforms expert summaries into more accessible versions for non-specialists. Using SciSummNet as the source corpus, we first generate baseline simplifications with GPT-4o-mini. In Phase 1, readers from STEM fields outside computer science identify difficult sentences and phrases and compare the original and GPT-simplified summaries in terms of comprehensibility, naturalness, and simplicity. In Phase 2, computer science experts use this feedback to create expert-edited reference simplifications. We release the resulting corpus together with human judgments and automatic evaluation results. The Phase 1 judgments show a clear preference for the GPT-generated summaries in terms of comprehensibility and simplicity, while qualitative analysis of the Phase 2 edits highlights the importance of preserving domain-specific terminology and the strength of scientific claims. The resulting resource supports the training and benchmarking of simplification systems for cross-disciplinary scientific communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25590v1">PILA: Plug-and-Play Insertion for LLM-native Advertising</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      How to monetize large language models (LLMs) by naturally integrating sponsored content into their responses, known as LLM-native advertising, has recently emerged as a critical problem. However, existing solutions entangle advertising with content generation inside a single model, which is incompatible with modern API-only or workflow-based LLM applications and inevitably compromises the original response quality. To address this, we propose PILA, which reformulates ad insertion as a conditional response rewriting problem and decouples it from the upstream service as a lightweight sidecar module. PILA is model-agnostic and can be seamlessly integrated with existing LLM services without modifying the base model or its workflow. It also exposes a controllable trade-off between user-side naturalness and ad-side exposure, offering a practical interface for downstream pricing and deployment. Experiments across diverse upstream models show that \pila consistently improves ad effectiveness while preserving response quality, highlighting its promise as a practical solution for LLM-native advertising.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25579v1">IRIS: Reusable Identity Representations from Frozen LLMs for Entity Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 9 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Entity alignment (EA) identifies entities across knowledge graphs (KGs) that refer to the same real-world object. Conventional EA methods mainly exploit explicit graph structures and textual fields, which often provide insufficient semantic understanding to recognize the same entity under heterogeneous descriptions and distinguish it from semantically similar entities. Although large language models (LLMs) offer deeper entity understanding, existing LLM-based EA methods largely use this capability for auxiliary generation or candidate-conditioned decisions. Consequently, such understanding is not distilled into a stable and directly comparable identity space, leaving alignment tied to specific KG pairs or candidate sets and requiring repeated processing as the matching context changes. To address these limitations, we propose IRIS (Identity Representations from Internal States), a training-free framework that constructs for each entity an iris-like signature encoding its distinctive and stable identity characteristics. IRIS derives these signatures by eliciting identity-oriented contextual representations from a frozen LLM, thereby forming a shared space in which each entity is encoded once and can be aligned across different KGs through direct similarity comparison, without pair-dependent representation construction or candidate-wise LLM inference. Across four established EA benchmarks and two frozen LLM backbones, the best IRIS variants achieve Hits@1 scores of 100.00, 99.38, 98.31, and 97.99 on D-Y-15K V2, DBP-WIKI, ICEWS-WIKI, and ICEWS-YAGO, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25572v1">Mapping CVEs to MITRE ATT&CK Techniques: A Curated Gold-Set Classifier and the Limits of LLM-Assisted Label Expansion</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      We present a reproducible pipeline for mapping Common Vulnerabilities and Exposures (CVEs) to MITRE ATT&CK Enterprise techniques from free-text vulnerability descriptions. Rather than relying on the CWE->CAPEC->ATT&CK derivation chain, whose table-expansion artifacts we quantify, we train a multi-label classifier on a curated gold dataset of 1,207 CVEs from expert MITRE Center for Threat-Informed Defense mappings. The resulting model approximately doubles recall@5 compared with a zero-shot embedding-similarity baseline and improves every ranking metric. We then investigate whether LLM-assisted labeling can extend the gold dataset. Initial experiments suggest contradictory conclusions: a single run indicates degraded performance, while averaging over five random seeds suggests a small gain. However, an independent replication and an expansion-size study (100--984 additional CVEs) show that the apparent improvement is an evaluation artifact. LLM-generated labels, with approximately 0.39 agreement with expert annotations, provide no reliable improvement at any expansion size and reduce rare-technique coverage at around 1,000 added CVEs (macro-F1 decreases by 0.04). The root cause is evaluation noise. Selecting checkpoints on a small test split effectively maximizes over many noisy evaluations, producing recall@5 differences of up to 0.05 between otherwise identical runs. Using a corrected protocol based on validation-split checkpoint selection, the gold-only model achieves recall@5 of (0.673 \pm 0.019), and repeating the decisive experiment confirms the null result for LLM expansion. A final scaling study shows that additional expert-curated data consistently improves performance, whereas LLM-labeled data does not, indicating that the classifier is limited by label quality rather than dataset size. All datasets, models, code, and training logs are publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25554v1">Distilling Temporal Search and Reasoning: Evolving LLMs for Future Prediction via Harness-Assisted Efficient Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Future event prediction carries broad social impact yet remains challenging. SOTA approaches augment LLMs with external agent frameworks whose predictive capability vanishes once the harness is removed. While recent Tool-Integrated Reasoning (TIR) internalizes deep search for multi-hop retrieval of facts, forecasting further demands temporal search and reasoning over historical trends and dynamic shifts. The key obstacle is data: historical queries induce temporal leakage that degrades forecasting into retrieval. Prior works either freeze information gathering with static observations, or rely on rejection sampling or unresolved fresh queries that discard vast amounts of data, degrading synthesis efficiency. We propose a time-truncation harness that enforces a temporal cut-off at every turn, enabling TIR-style sampling from historical events, reducing temporal leakage and reliance of rejection sampling or unsolved queries, increasing the sampling efficiency. We further build a large-scale corpus and a process-based metric and show that our harness naturally induces a broader temporal breadth of search and raises the proportion of high-quality data, further increasing the efficiency and reducing the reliance on complex rubrics. Distillation experiments show that students trained on harness-intervened data achieve the best performance, demonstrating harness-assisted model evolving that turns higher quality temporal search and reasoning data into a parametric advancement of the students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13767v2">Retrieve, Schedule, Reflect: LLM Agents for Chip QoR Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Modern chip design requires multi-objective optimization of timing, power, and area under stringent time-to-market constraints. Although powerful optimization algorithms are integrated into EDA tools, achieving high QoR hinges on effective long-horizon scheduling, which relies heavily on manual expert intervention. To address this issue and automate chip design, we propose an agentic LLM framework that schedules chip optimizations through direct interaction with EDA tools. The agent is grounded in natural language expertise expressed as a search tree through retrieval-augmented generation (RAG). We further improve scheduling quality with Pareto-driven QoR feedback through language reflection. Experimental results show that, compared with black-box search methods such as reinforcement learning, our framework achieves 10% greater timing improvement while consuming less power and area, with more than 4x speedup. The post-optimization QoR is also comparable to that achieved by human experts. Finally, the agent supports customized tasks expressed in natural language, enabling preferential QoR trade-offs. The code and chip design data will be publicly available at https://github.com/YiKangOY/Open-LLM-ECO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25498v1">Beyond Prefill-Decode Disaggregation: Dissecting LLM Inference for Heterogeneous Platforms via Dynamic Operator Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 To appear in MICRO 2026
    </div>
    <details class="paper-abstract">
      Prefill-decode disaggregation (PD) and roofline-based operator placement are common strategies for partitioning Large Language Model (LLM) inference across heterogeneous systems, but they are often insufficient in practice. End-to-end latency also depends on workload shape, runtime device contention, and persistent weight layout. We present DOPS (dynamic operator scheduling), a hardware-aware, closed-loop framework that jointly optimizes operator scheduling and blockwise weight layouts. DOPS constructs a stage-aware directed acyclic graph (DAG) and integrates two components: the Bifocal scheduler for dynamic operator-to-device placement and the Weight Layout Arbiter (WLA) for selecting hardware-efficient weight layouts under strict memory constraints. Across representative heterogeneous systems combining neural processing units (NPUs) and processing-in-memory (PIM) devices, Bifocal achieves geometric-mean speedups of 1.20$\times$ to 2.23$\times$ over the PD baseline. WLA provides an additional geometric-mean speedup of 1.28$\times$ to 1.33$\times$ over Bifocal/Linear. DOPS also supports systematic analysis of workload sensitivity and hardware scalability for LLM serving. The source code is available at https://github.com/YIAI-02/TriForm, and the visualization tool is demonstrated at https://youtu.be/Ya_oMCyYno0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27249v1">OwlPath: Lossless Knowledge Compression for LLM Bug Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      LLM-based software engineering agents are constrained by limited context windows: roughly 100K tokens must store structurally relevant code subsets to resolve bugs. Standard retrieval models treat code as plain text, forcing agents to resolve multi-hop dependencies including subclass chains, transitive callers and interface implementations through slow trial and error. We tackle this limitation with lossless knowledge compression, encoding source code into an OWL2 ontology to answer structural queries using minimal relevant code fragments. We present OwlPath, an OWL2 reasoning layer atop CodeGraph, a widely used code intelligence platform with 500K+ GitHub stars, offering a unified CLI for structural code retrieval. Powered by tree-sitter parsing, OwlPath supports multi-language repositories (Python, JavaScript, TypeScript, Go, etc.) and encodes language-specific semantics into a unified OWL2 ontology. It adopts two complementary modules. First, a transitive-closure engine fetches all structurally linked symbols via single SPARQL property-path queries, capturing multi-hop relations missed by string matching. Second, the OWL Software Knowledge Map (OWL-SKM) precomputes a compact 3KB summary with module trees, core APIs and issue-related symbols, directing agents to target modules in the first query. Evaluated on 18 SWE-bench Pro instances, OwlPath obtains a 68.4% strict-apply rate versus 66.7% for the CodeGraph baseline, cutting token usage by 28.8% and runtime by 39.5%. In offline retrieval tests over 67 instances, OwlPath improves recall 2.06 times (0.464 vs 0.226) and reaches 88.1% hit rate compared to CodeGraph's 59.7%. On a 37-question structural retrieval benchmark, recall rises from 4.4% to 28.8%, with 69-80% accuracy on transitive caller and interface tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25446v1">Toward an Organizational Science of Multi-Agent LLM Systems: Decoupling Who, How, and Which Algorithm</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Multi-agent frameworks built on large language models (LLMs) routinely entangle three logically distinct concerns: who is on the team (organization), how members align (coordination), and which algorithm fuses their work (collaboration protocol). IMACS (Intelligent Multi-Agent Collaboration System) separates the three into orthogonal, independently swappable layers. Classic organizational theory (Belbin roles, Mintzberg coordination, RACI accountability) becomes executable, validated configuration, and the framework places six published collaboration algorithms behind a common interface while exposing roles, coordination, and accountability as independently configurable factors. We use this separation to conduct controlled comparisons in which organizational assignments vary while the collaboration protocol is held fixed. It also turns protocol choice into a variable that can be learned: Adaptive Org Routing, a contextual-bandit meta-protocol, selects a protocol per task under an explicit quality-cost tradeoff, outperforms every fixed protocol in a controlled study, and trains online on real benchmark and LLM-judge rewards. The ablations expose a mechanism. Accountability placement changes outcomes exactly when the protocol routes the deliverable through the accountable agent, and the winning placement flips across model families, so organizational design cannot be hard-coded; it must be revalidated, or learned, for each model binding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24341v2">Simulating Tenant Responses to Energy Policy Interventions with Transaction-Cost-Aware LLM Age</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Recent studies use Large language models (LLMs) to simulate human opinions and decisions by prompting models with demographic, attitudinal, or persona-based descriptions. Yet such simulations rarely model the practical, cognitive, or social frictions that shape how people respond to policy interventions. Perceived transaction cost (PTC) provides a useful lens for modeling the practical frictions that shape policy responses, such as information burden, administrative effort, coordination demands, and perceived uncertainty. We use this lens to develop a friction-aware persona modeling approach for LLM-based simulation. In the context of energy-efficient renovation (EER), tenants are represented not only by who they are demographically, but by how they perceive the costs, benefits, barriers, and uncertainties associated with proposed renovation plans. Using survey data collected from 1,068 citizens in the Netherlands, comprising approximately 40,548 survey question and answer pairs, we compare prompt-only and fine-tuned settings across GPT-3.5-turbo, Ministral-8B-Instruct, and Llama-3.1-8B-Instruct, and evaluate supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) for local open-weight models. Results show that incorporating PTC-based personas and reasoning consistently improves model performance across both prompt-only and fine-tuned settings, suggesting that PTC-based persona design provides a useful bridge between institutional policy theory and interpretable LLM-based policy simulation. Code is available at https://github.com/xiaweijie1996/socialagent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25415v1">A Control System, a Dataset, and a Recipe for Making Frozen LLM Agents Learn a Domain</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 8 pages, 1 figure, 3 tables. Code and dataset: https://github.com/dpaul0501/context-optimization-rl
    </div>
    <details class="paper-abstract">
      Production LLM agents are increasingly assembled from a frozen model wrapped in a harness: a prompt template, a tool set, a memory/retrieval layer, a planning strategy, and a verification policy. Two 2026 systems, Meta-Harness (Lee et al., 2026) and HyperAgents (Meta AI, 2026), show that this harness can itself be optimized or even self-rewritten by an agentic proposer -- at the cost of either an expensive code-search loop or unconstrained self-modifying code, neither of which is auditable or usable with a fully black-box model API. We take a narrower, more constrained position: treat the harness as a small, fixed, human-legible action space and learn a policy over it online with classic sample-efficient reinforcement learning (an $ε$-greedy contextual bandit and REINFORCE), scored against a multi-objective reward (task success, verifier score, policy compliance, cost, latency, and an unsupported-claim penalty). We instantiate this control system with DSPy (Khattab et al., 2024) as both the context assembler and the source of the strongest non-adaptive baseline (a DSPy BootstrapFewShot static prompt), and evaluate it across three verifiable task domains -- tool-use workflows, code generation (HumanEval), and multi-hop retrieval QA (HotpotQA) -- and two model providers (a local Ollama model and AWS Bedrock). We release the harness-control-system code, the cross-domain verifiable task suite, the full trajectory/reward-decomposition logs from training, and a provider-agnostic deployment recipe for applying this to a new organization's domain and verification setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25408v1">Context Assembly as the Controlled Variable: A Control-Theoretic View of Harness Policies for Frozen LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 6 pages, 2 figures, 1 table. Code and companion paper's data: https://github.com/dpaul0501/context-optimization-rl
    </div>
    <details class="paper-abstract">
      A growing body of 2026 work applies control theory to LLM agents: Lyapunov-certified stability for tool-mediated controllers (Prinos et al., "Stable Agentic Control", 2026), sample-complexity bounds for sparse policies over massive discrete tool universes (Majumdar, "Sparse Agentic Control", 2026), and regulatory-control decompositions of multi-agent systems into auditable feedback loops (Nogueira and Skogestad, 2026). We do not claim to introduce control theory to LLM agents -- that ship has sailed. Our narrower claim is about what the controlled variable is. Prior work controls tool selection, inter-agent message routing, or the agent's raw action stream. We instead treat context assembly itself -- which prompt template, which few-shot demonstrations, how much retrieved context, how many planning/verification passes -- as the controlled variable, learned online by a contextual bandit or REINFORCE policy sitting outside a frozen model. This paper develops the formal decomposition (inner frozen policy $π_θ$, outer context policy $π_φ$), gives a stability argument for the online controller in the sense used by Zhang et al. (2026) (non-decreasing expected reward under bounded policy change), and reports an uncertainty-calibration analysis of the controller's own confidence against realized task outcomes. The applied counterpart to this paper instantiates the same controller across three domains and two model providers and releases the dataset, trajectory logs, and a deployment recipe; here we focus on the formal framing and the stability/uncertainty evidence a control-theoretic claim requires.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23999v2">ContainmentBench: Trace-Based Evaluation of Post-Injection Containment in Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 28 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents process untrusted content, maintain memory, delegate across agents, and invoke side-effecting tools. Existing prompt-injection evaluations typically summarize security with terminal attack or policy outcomes, but equal endpoints can conceal different post-exposure traces and different losses of authorized utility. We introduce ContainmentBench, a sandboxed, trace-based benchmark that separately measures benchmark-defined endpoint policy compliance, instrumented logged propagation, recovery instrumentation, and authorized structured-action completion. In a pre-specified 17,640-rollout study with Qwen2.5-7B-Instruct, all 600 matched active-tainted pairs comparing taint-only and intent-aware enforcement have the same zero committed-harm outcome, yet 73.5% differ in logged trajectory or utility. Taint-only enforcement completes only 0.1642 of authorized tainted workflows; a trusted-ledger policy raises completion to 0.8567, while a strong tool-boundary baseline reaches 0.9233 under the same observed endpoint-policy outcomes. We also find that aggregate logged-spread rankings change with evidence-stage composition and denominator choice. These results show that a terminal policy label is not a sufficient statistic for operational post-exposure containment; evaluations should report endpoint, stage-stratified trajectory, and utility evidence separately, and should promote recovery evidence to comparative claims only where the corresponding controls are valid. The full-scale study is synthetic and single-model; the policy case additionally assumes a correct structured authorization ledger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25366v1">Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Published in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26). 12 pages, 6 figures, 4 tables. Code available at https://github.com/muyiahhh/SharpRec
    </div>
    <details class="paper-abstract">
      LLM-based Cross-Domain Sequential Recommendation (CDSR) leverages LLMs to enhance target performance via deep semantic reasoning, alleviating the dependency on overlapping users. Among LLM-based paradigms, model merging is particularly promising for multi-domain scenarios due to its superior scalability and flexibility in integrating diverse knowledge sources. However, our empirical investigations reveal two critical bottlenecks: (1) cross-domain knowledge conflict; and (2) performance saturation in multi-domain fusion. Our analysis attributes these phenomena to parameter-level misalignment and statistical homogenization during the merging process. To address these bottlenecks, we propose SharpRec, Sharpness-aware Model Merging with Salience Recovery for LLM-based CDSR, a framework designed to lift the performance upper bound of merged models. SharpRec incorporates two synergistic modules: Sharpness-aware Geometric Alignment to establish a stable geometric foundation for interference-free fusion; and Preference Salience Activation to effectively recover the distinctive features essential for bolstering target domain performance. Extensive experiments in both dual-domain and multi-domain scenarios demonstrate that SharpRec consistently outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03881v2">LLM-generated personalized nudges for improving pro-environmental behavior: Field evidence from resource conservation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Encouraging pro-environmental behavior remains a major challenge for sustainable cities. Conventional feedback nudges can show individuals how their current behavior compares with environmental goals but often provide limited guidance on what to do differently in daily life. This study examines whether supplementing weekly feedback on participants' behavior with LLM-generated personalized action suggestions improves pro-environmental behavior, using daily electricity and hot-water conservation as a case study. We developed an LLM agent that generated weekly conservation messages from participant profiles, recent consumption records, and prior interaction history, combining a usage report with personalized suggestions, behavioral-change scenarios, and estimated savings. The agent was evaluated in a three-arm randomized field experiment with 233 university residents in Beijing from November 2024 to January 2025. Participants received text-based nudges, image-enhanced nudges, or LLM-generated personalized nudges over five intervention rounds. Daily electricity use and shower hot-water use were measured using dormitory meter readings and billing records. Compared with text-based feedback, LLM-generated personalized nudges reduced electricity consumption by 0.56 kWh per room-day (p = 0.014), corresponding to an 18.3 percentage-point higher saving rate. Image-enhanced feedback alone showed no clear improvement. Hot-water savings followed the same direction but were smaller and less precisely estimated (9.8 percentage points, p = 0.087). Personalized nudges contained more planning, appliance-specific, and action-oriented language and were associated with more sustained, task-focused engagement. These findings offer a pathway for integrating generative AI into sustainable urban management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25308v1">CAST: Game Solvers as Turn-Level Teachers for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) to act in long-horizon games is a promising step toward generalist decision-making, yet reinforcement learning with verifiable rewards (RLVR) relies on sparse final rewards that reveal little about which decisions determine success. Denser process signals could supply this missing turn-level credit, but existing sources are hard to keep both cheap and accurate. We observe that changes in a game solver's state value reveal whether an action advances the state toward success. Building on this insight, we propose CAST (Credit Assignment from Solver Teachers), which converts these value changes into solver advantages and injects them into RLVR as turn-level signals. We further show that, under a soft-optimal solver assumption, maximizing the solver advantage is equivalent to on-policy distillation from the solver, requiring only scalar values rather than teacher logits. Across Sokoban, Minesweeper, and Rush Hour, CAST outperforms all trained baselines on every game under both in-domain and unseen-difficulty evaluation and achieves the highest average zero-shot performance on ALFWorld and WebShop. Our code is available at https://github.com/Wloner0809/CAST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25297v1">Hybrid Analysis for Secure MCP Tool Use in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      The rapid development of large language model (LLM) agents has enabled their broad adoption across diverse real-world tasks. To standardize interactions between LLM agents and external environments, Model Context Protocol (MCP) tools have emerged as a de facto standard and have been widely integrated into these systems. However, the use of MCP tools also introduces new safety risks, as LLM agents can be induced to perform malicious or unauthorized actions. Although prior work has proposed defenses for securing tool use in LLM agents, most methods rely on static analysis, i.e., inspecting prompts and generated outputs, which limits the defense effectiveness and robustness. To address these limitations, we propose MTGuard, a hybrid analysis-based defense framework designed to safeguard the use of MCP tools in LLM agents by leveraging lifecycle-aware static-dynamic co-analysis. Extensive evaluation demonstrates that MTGuard effectively mitigates multiple categories of harmful tool use across different LLM agents while maintaining performance on benign user tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10016v2">Tokenizing Numerical and Embedding Features for LLM RecSys</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as backbone architectures for recommender systems because of their strong sequence modeling and representation learning capabilities. However, most LLM-based recommenders operate primarily on discrete textual tokens, whereas practical recommendation pipelines also rely on continuous numerical features and dense embedding features produced by upstream feature engineering or pretrained encoders. This mismatch limits the ability of LLM-based models to exploit fine-grained non-textual signals. We propose a soft-token fusion framework that maps numerical and embedding features into the LLM embedding space, allowing heterogeneous recommendation signals to be consumed through the standard token interface. We instantiate the framework in a shared-parameter LLM-based two-tower retrieval model and introduce an interaction-based fusion module that refines embedding and numerical soft tokens before they are inserted into the final LLM input. Experiments on three Amazon recommendation benchmarks show that soft-token fusion improves retrieval performance over LLM-based baselines, and that interaction-based fusion is more effective than direct concatenation of heterogeneous soft tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27628v2">DSevolve: Enabling Real-Time Adaptive Scheduling on Dynamic Flexible Job Shop with LLM-Evolved Heuristic Portfolios</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      In dynamic flexible job shops, order arrivals, machine breakdowns, and processing-time deviations continually reshape the scheduling state and the priority trade-offs behind dispatching decisions. Dispatching rules are well suited to this setting because they are fast, interpretable, and easy to deploy, and recent LLM-assisted automatic heuristic design further expands their expressiveness by evolving composite priority functions. The key challenge is to make these evolved rule behaviors state-adaptive without losing the rapid response needed for online rescheduling. This paper proposes a dynamic self-evolutionary framework DSevolve, which separates offline rule-library construction from online state-conditioned rule selection. Offline, an LLM-guided quality-diversity search combines multi-persona seeding, a MAP-Elites behavioral archive, and behavior-guided variation to evolve a library of complementary rules rather than a single elite, and event-level simulation then trains a neural selector to rank the rules by state. Online, after each rescheduling event, a neural selector maps a 22-dimensional state fingerprint to rule scores and dispatches the top-ranked rule within about a second, meeting the response-time requirement after each disruption. Experiments on dynamic instances derived from standard flexible job shop benchmarks show that DSevolve achieves lower mean makespan than individual LLM-evolved rules, classical dispatching rules, and learning-based baselines under a one-active-rule deployment protocol. Trained only on small instances, the selector transfers zero-shot to substantially larger dynamic shops. These results show that state-conditioned selection preserves the speed and interpretability of dispatching rules while improving adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25257v1">Laplace-PSN-IRT: Uncertainty Quantification for Neural Item Response Theory Models of LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-28
    </div>
    <details class="paper-abstract">
      Item Response Theory (IRT) has recently been proposed as a framework for evaluating large language model (LLM) benchmarks by separating a model's latent ability from the properties of individual benchmark items. Existing neural IRT approaches, including PSN-IRT, estimate these quantities using point estimates, limiting uncertainty quantification and downstream statistical inference. We introduce Laplace-PSN-IRT, a post-hoc last-layer Laplace approximation that augments a trained PSN-IRT model with approximate Bayesian posterior inference, recovering calibrated uncertainty over model ability and item difficulty without retraining. The resulting posterior enables credible intervals, probabilistic comparisons between models, and propagation of parameter uncertainty into Fisher-information-based item selection. We show that most pairwise comparisons among 12 models on a standard LLM benchmark leaderboard are not statistically distinguishable despite differing point-estimate ranks. We further show that point-estimate Fisher information can become nearly zero for many benchmark items because it is evaluated at a single reference ability, whereas posterior-expected Fisher information remains substantially more stable across the ability range. Finally, posterior-expected Fisher information more accurately recovers full-benchmark ability rankings from small benchmark subsets in most experimental settings while matching point-estimate performance for the smallest subsets. We validate the calibration of the approximate posterior using held-out predictive coverage and find that modeling item difficulty as random while treating item discrimination as fixed produces well-calibrated uncertainty in this architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25228v1">Interpretable Column Annotation with LLM-Symbolized Decision Process Materialization</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Column annotation (CA), including column type annotation (CTA) and column property annotation (CPA), aims to identify the meanings of table columns and the semantic relationships among them. Recent CA methods usually use various neural models to learn column representations and directly map them to label categories, thereby (1) sacrificing model interpretability and adaptivity, and (2) overlooking rich label semantics and ultimately limiting accuracy. To address these limitations, we propose SymCA, an LLM-empowered interpretable CA framework that materializes column annotation as a global-to-local symbolic decision process. SymCA consists of two components: (1) global skeleton induction, which constructs a semantic skeleton over the label space, and (2) local substrate evolution, which evolves predictive substrates within the skeleton. Specifically, to exploit label semantics while preserving an interpretable decision process, the global skeleton induction module leverages LLMs to generate candidate hypernym-inspired tree-structured semantic skeletons and employs a Minimum Bayes Risk (MBR)-based consensus strategy to select a robust skeleton against generation variance. Since different internal nodes require different evidence to distinguish among their child nodes, the local substrate evolution module materializes each internal node as an executable and evolvable predictive substrate. Over multiple evolution rounds, each substrate trains an interpretable random forest classifier with the current operator set, leverages the LLM to propose node-specific operator modifications, and uses an exploration-exploitation strategy to prioritize promising substrates. Extensive experiments demonstrate that SymCA is accurate, robust, and interpretable, outperforming the strongest baselines by an average of 6.42% in Micro-F1 and 11.03% in Macro-F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25209v1">VaLiDRec: Variable-Length LLM-Aligned Semantic IDs for Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 10 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Generative recommendation commonly represents items using fixed-length semantic identifiers (SIDs) constructed through clustering and quantization. However, these artificial codes may overcompress item semantics, remain misaligned with pretrained LLM vocabularies, and require costly autoregressive decoding. In light of this, we propose VaLiDRec, a generative recommendation framework based on variable-length, LLM-aligned semantic identifiers. VaLiDRec constructs SIDs directly from informative native LLM vocabulary tokens via token importance estimation, semantic-quality-aware pruning, and collision-aware refinement, allowing identifier lengths to adapt to item semantic complexity. To model user preferences, VaLiDRec incorporates graph-aware soft prompts and reformulates recommendation as token-set prediction with token-level item scoring, eliminating autoregressive SID generation and beam search. Experiments on four real-world datasets show that VaLiDRec consistently outperforms strong sequential and generative recommendation baselines across all evaluation metrics. It further achieves superior zero-shot item cold-start performance and 87.49$\times$ faster inference than LC-Rec. These results demonstrate that LLM-native variable-length semantic identifiers provide a more expressive and efficient paradigm for generative recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30860v2">Less Deliberate in Teams: Student LLM Use Across Individual and Collaborative Work</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 8 pages, 2 figures, 3 tables. Accepted at ACM SIGCSE Virtual 2026 (November 12-15, 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become common in computing courses, we need to understand how the social setting shapes how students use them. This paper reports findings from a semester-long study of 96 undergraduate students who completed six assignments, alternating between individual homework and team project milestones. We tracked LLM usage, prompting habits, and how students verified AI-generated output across all six assignments. LLM usage dropped by 42.7 percentage points when students moved from individual work to their first team milestone, then partly recovered in later team tasks. Students also wrote fewer prompts, used fewer deliberate prompting strategies, and checked LLM output less carefully. The share of students who ran tests on AI-generated code fell by 19.4 percentage points during team assignments and never fully rebounded. A within-student analysis found that 18.9% of students who consistently used LLMs on their own stopped using them entirely in teams, while only 3.2% went the other direction. These results suggest that collaborative context is associated with reduced deliberate LLM engagement beyond what task type alone can explain. The moment students form teams appears to be a critical turning point that may benefit from more explicit instructional support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25140v1">How Affect Propagates among LLM Agents: Emergent Emotional Contagion in Crowd Simulation</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 31 pages, 14 figures
    </div>
    <details class="paper-abstract">
      This paper studies the behavior of language models in a multi-agent crowd simulation, focusing on how affect propagates among agents that perceive and appraise one another. Each agent perceives its neighbors through visual, auditory, and tactile channels, then appraises these perceptions in light of its prompted personality profile, memory, current affective state, and situational context. Appraisal is carried out by an LLM, which updates the agent's internal affective state and selects its outward expression. The architecture contains no hand-authored mechanism for directly transferring affective state between agents; instead, inter-agent influence arises through the perception-appraisal-expression loop. The agent representation draws on the Big Five personality model and Russell's circumplex model of affect. To limit latency, low-level steering and navigation are handled by a conventional crowd simulator operating independently of the LLM-based cognitive layer. We evaluate the architecture across five scenario environments spanning alarming, joyful, and neutral situations in different spatial layouts. The results show that the system produces emotional contagion dynamics with spatial, temporal, and personality-dependent structure in sparse, small crowds. Alarm spreads from seeded agents as a traveling front, the mean alarmed fraction settles at a nonzero plateau, and the distribution of prompted personality profiles determines whether an ambiguous alarm ignites panic and whether a provocation is interpreted as anger or fear. We further evaluate the appraisal step through controlled experiments across prompt variants, sampling temperatures, and four model backends, showing that the dynamics are backend-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25124v1">UrbanTrace: LLM-Assisted Discovery and Semantics-Aware Integration of Spatial Data</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 11 pages, 7 figures. Accepted to IEEE VIS 2026 (Full Papers Track). Author's version accepted for publication in IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Urban decision-making requires integrating heterogeneous spatial data. While current GIS tools handle geometric computation efficiently, they lack the semantic reasoning to guide complex workflows. Analysts manually manage data discovery, spatial boundaries, and measurement semantics, risking aggregation errors. We present UrbanTrace, a visual analytics system that transforms manual spatial data-wrangling into a transparent, node-based collaborative workflow with context-aware AI agents. Using an offline profiler to extract semantic and geometric metadata, UrbanTrace grounds LLMs in real-world data distributions. This enables specialized agents to retrieve datasets based on high-level goals and automatically enforce valid spatial aggregations. To make harmonization explicit, three interactive views: an Integration Provenance Graph, Multivariate Priority Map, and Spatial Delta Map, allow users to explore how conclusions shift across spatial configurations. We evaluate UrbanTrace on 28 urban scenarios spanning 112 datasets. Quantitative ablations show our profiling significantly outperforms baseline LLMs in data discovery, achieving 100% semantic and 87% geometric validity in spatial mapping. Through real-world case studies and expert interviews, we demonstrate that UrbanTrace turns spatial aggregation sensitivity from a methodological burden into an exploratory visual asset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25069v1">DS@GT ARC at CheckThat! 2026: LLM-Based Trace Ranking and Grouped Reward Modeling for Multilingual Numerical Claim Verification</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 10 pages, 2 figures. Accepted at CLEF 2026 CheckThat!. To appear in CEUR Workshop Proceedings
    </div>
    <details class="paper-abstract">
      Automated verification of numerical claims is a challenging problem, as it requires both language understanding and quantitative reasoning. This paper describes our system for CLEF 2026 CheckThat! Task 2, which focuses on ranking reasoning traces generated by large language models (LLMs) and predicting a final verdict for numerical claims in English and Arabic. We explore two approaches. The first approach fine-tunes an LLM-based verifier using LoRA to score each reasoning trace independently as a binary classification problem, and selects the final verdict using Best-of-N selection. We further experiment with adaptive sub-claim decomposition to break complex claims into simpler parts before verification. The second approach uses a lightweight TF-IDF reward model with handcrafted numeric and temporal overlap features to score traces, and aggregates scores by verdict group to determine the final prediction. For Arabic, we compare a general multilingual model against AraBERT, a language-specific model pretrained on Arabic text. Our results show that the LLM-based approach outperforms the lightweight reward model on most metrics, particularly Recall@5, while the reward-based approach shows stronger performance on the Conflicting class. Sub-claim decomposition did not improve performance, suggesting that claim splitting introduces noise rather than aiding reasoning. For Arabic, AraBERT outperforms the multilingual baseline across most metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25068v1">How Often Should a Recommender Call an LLM? Value-Weighted Routing, Monitoring, and Seasonal Robustness</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Routing decisions between a cheap heuristic and an expensive large language model (LLM) are typically framed as a difficulty problem: send the hard cases to the expensive path. We argue this framing is incomplete because difficulty and business value are distinct axes - a difficult cheap item and a difficult costly item do not have the same cost of error. We present Value Router, a fully synthetic simulation of a retail merchandising pipeline that routes items using only estimated difficulty and estimated value, never ground truth. The study has three stages. First, a value-weighted threshold router is compared with a difficulty-only and a random baseline on a synthetic catalog with an inverse correlation between category volume and value. Value-weighting matches the difficulty-only baseline's recall of true high-value items (60%) while achieving substantially higher precision (98.3% vs. 94.3%). Second, a decision logger and monitor expose a failure mode hidden by aggregate metrics showing that the aggregate result is driven almost entirely by between-category differences rather than per-item discrimination. Third, a simulated Black Friday demand surge (2.5 volume with a shift toward higher-value categories) compares a static router, a seasonally tuned router, and two slow-path budget policies. All results are from a controlled synthetic simulation with experimenter-defined ground truth and illustrate design principles for cost-aware routing systems rather than validated real-world claims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05775v3">Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. Meanwhile, they pose risks by inadvertently producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for text generation and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies encompassing unintentional toxicity, adversarial jailbreak attacks, and comprehensive mitigation strategies. We explore LLMs' dual role as both generators of harm and enablers of safety through detection, classification, content moderation, and prevention. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our review highlights the evolving landscape of LLM safety and identifies limitations in current evaluation methodologies. Ultimately, our review outlines future research directions to guide the development of robust and ethically aligned language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09181v3">Evaluating the Practical Effectiveness of LLM-Driven Index Tuning on Microsoft SQL Server</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Indexes are crucial for database performance. Index tuning, i.e., selecting appropriate indexes for a database workload, is an important problem. The state-of-the-art index tuning tools in the industry, e.g., Database Tuning Advisor (DTA) developed for Microsoft SQL Server, rely on a "what-if" API, which can estimate the cost of a query for a given index configuration. They take as input a SQL workload and constraints such as a storage bound, and search over the large space of index configurations to find one with low optimizer-estimated cost for the input workload. Large language models (LLMs) offer a different approach to index tuning, using knowledge they have learned from publicly available training data. However, the effectiveness of LLM-driven index tuning in comparison to today's index advisors, particularly on enterprise workloads, remains unclear. In this paper, we study the practical effectiveness of LLM-driven index tuning on Microsoft SQL Server using both industrial benchmarks and real-world enterprise customer workloads, and compare it with DTA. Our results show that while LLMs in several cases identify configurations that significantly outperform those found by DTA in terms of execution time, they suffer from high variance in index recommendation quality. Furthermore, index recommendations from the LLM are often substantially worse than DTA in terms of optimizer-estimated cost, making it challenging to extend cost-based index advisors such as DTA to leverage LLMs for index tuning. We point to some areas of future work that may be important for robustly leveraging LLMs for index tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24964v1">ALIBI: Adaptive Agentic Attacks on LLM-Based Vulnerability Detectors via Adversarial Code Comments</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed for security-sensitive tasks such as vulnerability detection and code review. Their reliance on natural-language context embedded in source code exposes a previously underexplored attack surface: adversarial comments that can influence a detector's reasoning without changing program behavior. We study LLM-based vulnerability detectors against a new adversary: a coding agent that implements new functionality, deliberately introduces vulnerabilities, and strategically inserts adversarial source-code comments to evade detection. We present ALIBI, an automated adaptive black-box attack framework that generates and iteratively refines adversarial comments using detector reasoning and feedback. We transform real-world vulnerability-fixing commits into coding tasks and evaluate four representative LLM-based vulnerability detectors, ranging from specialized open-weight reasoning models to frontier multi-agent systems. All evaluated detectors are highly vulnerable: attack success rates exceed 90% across 125 real-world null-pointer dereference vulnerabilities, reaching 100% on one system. The framework also generalizes beyond this vulnerability class. Adversarial comments steering detector reasoning or fabricating external tool results prove most effective, while iterative refinement based on detector feedback further increases attack success. Finally, prompt-level defenses provide limited robustness against adaptive attacks, whereas architectural isolation and pre-detector comment sanitization substantially improve resilience. Our findings expose a fundamental attack surface in current LLM-based vulnerability detectors and motivate security-aware designs that carefully calibrate trust between natural-language context and program evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24714v1">Efficient LLM-Generated Shuttling Compilers for Complex Trapped-Ion Architectures</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 56 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Trapped-ion quantum computers rely on shuttling compilers, which cast an input algorithm into a sequence of ion-qubit movements within a given architecture. We present the first study in which a single frontier large language model (LLM), Claude Opus 4.7, generates and iteratively refines the full Python code of shuttling compilers from written specifications. We start with a compiler for (i) a linear segmented trap, extend it to (ii) a trap with junctions, and finally achieve efficient compilation for (iii) a broad class of connected trap graphs. The compilers for the more general cases are seeded with code from the previous ones. We benchmark the LLM-generated compilers against state-of-the-art hand-crafted ones using a common suite of quantum circuits. The number of shuttling timesteps is reduced by up to 76% for (i) and up to 39% for (ii). For the broad case (iii) of freely connected architectures, we find large variations in the required number of shuttling timesteps, depending on the connectivity. A densely connected, junction-rich architecture yields an order-of-magnitude reduction in shuttling timesteps compared to a corridor-like one. Repeating the complete generation and evaluation with a second frontier LLM, Claude Fable 5, reproduces these findings, with the Fable 5 compilers surpassing the hand-crafted ones more often on the largest circuits. Our results show that an unmodified frontier LLM can produce working, correct, and competitive shuttling compilers without additional manual algorithmic engineering, thus reducing the development time for new architectures from several months to a few days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24649v1">Reason-Mediated Behavioral Models for Auditing LLM Social Simulators</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used as social simulators, including as synthetic survey respondents. Most evaluations ask whether simulated outcomes resemble human outcomes. We argue that this is necessary but too weak: a simulator can match the final answer while using the wrong rationale-derived reason pattern. We study this problem through a 94-person sunscreen concept test in which each respondent evaluated three product concepts and wrote open-ended rationales. We map those rationales into signed reason states $Z$, where positive signs support adoption and negative signs block it. This gives a practical audit: holding respondent descriptors $D$, category context $K$, and concept treatment $X$ fixed, do human rationale-derived reasons help predict behavior $Y$, and can an LLM simulate the same reason state without seeing the human rationale or outcome? Human rationale-derived reasons substantially improve held-out prediction of purchase intent. LLM-simulated reasons are more brittle: they often sound plausible, but frequently echo the concept board rather than recover the respondent's acceptance or rejection path. The paper contributes an evaluation framework for social simulators. Reason states do not identify natural causal effects by themselves, but they provide an interpretable test of whether a simulator's stated reasons align with human evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05132v3">PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      As medical LLMs transition to clinical deployment, assessing their ethical reasoning capability becomes critical. While achieving high accuracy on knowledge benchmarks, LLMs lack validated assessment for navigating ethical trade-offs in clinical decision-making where multiple valid solutions exist. Existing benchmarks lack systematic approaches to incorporate recognized philosophical frameworks and expert validation for ethical reasoning assessment. We introduce PrinciplismQA, a philosophy-grounded approach to assessing LLM clinical medical ethics alignment. Grounded in Principlism, our approach provides a systematic methodology for incorporating clinical ethics philosophy into LLM assessment design. PrinciplismQA comprises 3,648 expert-validated questions spanning knowledge assessment and clinical reasoning. Our expert-calibrated pipeline enables reproducible evaluation and models ethical biases. Evaluating recent models reveals significant ethical reasoning gaps despite high knowledge accuracy, demonstrating that knowledge-oriented training does not ensure clinical ethical alignment. PrinciplismQA provides a validated tool for assessing clinical AI deployment readiness. Our data and test scripts are fully released on https://github.com/FreedomIntelligence/PrinciplismQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24625v1">Agentic Permissions Policy Algebra for Taint Confinement in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 Preprint. Submitted to the 19th ACM Workshop on Artificial Intelligence and Security (AISec '26). 10 pages, 2 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Autonomous LLM agents processing mixed-confidentiality data face severe security risks from prompt injection attacks and reasoning errors. While dynamic Information Flow Control (IFC) provides structural security guarantees, traditional taint tracking permanently taints an agent's context upon reading unvetted data, severely restricting downstream utility. We present APPA (Agentic Permissions Policy Algebra), an IFC framework that resolves this usability bottleneck through engine-managed context branching and prospective acquisition enforcement. Before data acquisition occurs, APPA prospectively evaluates label descents and missing prerequisites, generating actionable remedy plans (Authorize, Accept). To inspect unvetted data without polluting the primary context, a label-seeded child trajectory is spawned, absorbing label descent locally and allowing a trusted sanitizer to return a bounded derivative to the unchanged parent. Governed by a two-monoid model over security labels and shared event logs, we formally prove parent label preservation and merge confinement. Finally, we evaluate APPA on a multi-turn tool-chaining benchmark across four models: it suppresses exfiltration (31%-50% down to 0%-7% attack success), and on three of the four, branching recovers a substantial share of the utility that taint tracking alone forfeits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24617v1">LaRec: Unleashing LLM-based Latent Reasoning for Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great promise in recommendation due to superior reasoning abilities. However, existing methods mainly rely on explicit Chain-of-Thought (CoT), resulting in verbose reasoning texts and inefficient response times. latent reasoning aims to balance efficiency by thinking within a continuous latent space, yet it faces two major challenges: (1) Lack of Fine-grained Supervision: Latent reasoning relies solely on feedback from the final labels, providing sparse supervisory signals that struggle to effectively guide the optimization of multiple hidden reasoning steps. (2) Single Reasoning Path: The deterministic nature of latent reasoning impedes the exploration of users' diverse interests and preferences, thereby limiting the recommendation capabilities of LLMs. To address these issues, we propose \textbf{$LaRec$}, an efficient generative recommendation framework designed to unleash the potential of latent reasoning in LLMs. $LaRec$ consists of two core stages: First, we design Latent Pre-training that empowers LLMs with latent reasoning capabilities by providing rich supervisory signals to the latent space reasoning via step-level alignment and process direction alignment. Second, we introduce Personalized RL-tuning. Specifically, we construct a personalized Gaussian Mixture Distribution for each user based on their historical interests. By randomly sampling distinct reasoning starting points from this distribution during training, we guide the LLMs to traverse diverse reasoning paths within the latent space, enabling efficient exploration of user's multi-faceted interests. Experiments on multiple datasets show that $LaRec$ significantly outperforms existing baselines with comparable efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30219v4">EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 74 pages, 2 figures, 4 tables. Hybrid systematic survey and conceptual framework on LLM evaluation and AI-safety failures, synthesizing 373 primary studies (2018-2026). Introduces the EvalSafetyGap framework (Instability Decomposition, Alignment Trilemma) and reports an exploratory ten-model audit. Submitted as a review/survey article; not currently under consideration elsewhere
    </div>
    <details class="paper-abstract">
      This paper presents a systematic survey and conceptual synthesis of the shared measurement problem underlying large language model (LLM) evaluation and AI safety: benchmark scores, reward signals, and safety metrics can improve while the capabilities and alignment properties they are meant to represent remain uncertain. Synthesizing 373 primary studies published between 2018 and 2026, the survey organizes evidence on benchmark validity, contamination, dynamic evaluation, LLM-as-a-judge protocols, adversarial safety testing, reward and proxy optimization, mechanistic interpretability, and AI governance into an eight-stream evidence taxonomy. Building on this synthesis, we introduce EvalSafetyGap, a conceptual framework that unifies benchmark-validity and alignment-failure research as a shared proxy-target divergence problem under optimization pressure, formalized through a Goodhart-inspired Instability Decomposition and an Alignment Trilemma. An exploratory ten-model public-evidence audit illustrates the framework by showing why capability, behavioral robustness, and governance disclosure should be reported as separate evidence layers rather than collapsed into a single safety score. The survey closes with a research agenda for dynamic and contamination-resistant benchmarks, pre-specified multi-attempt threat models, version-locked evaluation, transparent source reporting, and validated mechanistic safety indicators, offering researchers, model developers, and AI auditors a shared vocabulary for measurement-aware LLM safety evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v5">Entropy Sentinel: Probing Entropy Traces for LLM Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring---estimating where a model underperforms as traffic drifts---and (2) prioritization---deciding where to intervene to close the largest performance gaps. We explore whether top-$k$ logprobs---cheap, consumer-accessible signals from standard inference---can serve as reliable proxies for domain-level quality of both verifiable and subjective tasks. We summarize each response's output-entropy profile into a compact vector, predict instance quality with a lightweight classifier, and then average predictions to yield a domain-level estimate. On verifiable tasks (ten STEM benchmarks, nine LLMs, exhaustive train/test compositions), estimates often track held-out accuracy, with several models showing near monotonic calibration. On subjective tasks, trained on LLM-judge scores over categorized real user conversations, several models track the judge's slice scores remarkably closely ($r$ up to $0.91$) and detect the worst-performing categories near-perfectly, though this is not the case for all LLMs. Where coupling holds, results suggest that entropy signals can support monitoring at a fraction of the cost of judge-based evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24588v1">SIREN: Towards End-to-End Extreme-Weather Early Warning with Experience-Grounded LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Early warning of extreme weather is essential for mitigating the societal, economic, and environmental risks posed by hazardous weather events. However, expert-centered warning workflows are costly, labor-intensive, and difficult to scale throughout the warning-to-action process. Although recent advances in Large Language Model (LLM) agents have enabled the automation of weather-related tasks, existing studies remain centered on isolated scientific tasks and overlook the chain of interdependent processes required for operational extreme-weather early warning. To bridge this gap, this study investigates automated end-to-end extreme-weather early warning through LLM agents. We first develop SIREN-Bench, a comprehensive benchmark comprising 600 question-answer instances across 19 tasks, and covering four individual warning procedures and an end-to-end warning chain. Evaluation on SIREN-Bench reveals substantial capability gaps in existing weather agent frameworks. This motivates us to develop SIREN, an experience-grounded agent framework inspired by experts' use of historical cases, which combines an agentic execution environment integrating heterogeneous weather evidence and tools with a family of agent harnesses that exploit historical cases through retrieval, skill distillation, and predictive modeling. Extensive experiments demonstrate that SIREN outperforms weather-agent baselines on both individual warning procedures and end-to-end warning chains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24573v1">LLM-SoccerArena: Benchmarking LLMs on Real-World Predictions in Sports</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly support decisions about uncertain future events, yet evaluating their ability to forecast real-world outcomes remains difficult. In particular, existing benchmarks are typically static and retrospective, and therefore cannot test how information is synthesized by LLMs to predict future events under uncertainty. We introduce LLM-SoccerArena (https://llm-soccerarena.com), a prospective live benchmark that evaluates how well LLMs forecast real-world sports events before the outcomes are known. LLM-SoccerArena provides (1) a prospective live benchmark protocol, (2) a public open-source platform, and (3) a factorial benchmark design together with tournament-related questions (e.g., which team will win). LLM-SoccerArena automatically records timestamped, schema-validated forecasts of unresolved events, together with prompts, model versions, tool traces, and costs. The factorial design varies along four dimensions: (1) model version (e.g., GPT-5.5, Claude Opus 4.8); (2) information access; (3) prompting strategy, and (4) forecast horizon. We demonstrate LLM-SoccerArena through a large-scale evaluation of the 2026 FIFA World Cup, in which seven LLMs generated forecasts for all 104 matches and 15 tournament-related questions. We provide a detailed analysis of model performance across information access, prompting strategy, and forecast horizon. As a result, LLM-SoccerArena provides new evidence about the forecasting performance of state-of-the-art LLMs. For example, LLMs with web access outperform those without, but only by a small margin (i.e., a 0.023 improvement in Brier score). Overall, LLM-SoccerArena provides a flexible, open-source platform for prospective benchmarking of unresolved events. LLM-SoccerArena will be continuously updated, and can be directly applied to future national and international tournaments and league competitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24551v1">LLM-Assisted Ontology Engineering and Construction of a French Legal Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Maintenance regulations are complex legal texts that are difficult to exploit when addressing a specific case and challenging to integrate into operational systems. This paper presents a two-stage LLM-assisted workflow for French maintenance regulations: ontology engineering from a SEMLEG-based core ontology, followed by construction of an ontology-grounded French legal knowledge graph. The first stage consists in the open extraction of typed entities and triples from a stratified corpus sample, the normalization of labels through embedding-based fusion, and the induction of candidate object properties with their signature (domain and range). The second stage uses the resulting ontology to guide the closed extraction of triples and RDF graph construction over the full corpus. Experiments with GPT-4.1 and mistral-large-2512 show robust structured outputs, near-complete class alignment, and a substantial reduction of duplicated entities and predicates after fusion. Fewer than 20% of triples introduce unseen properties, while lower exact signature compliance reveals new domain-range combinations for existing predicates. These results point to predicate normalization and the validation of newly observed relation signatures as key refinement steps for industrial maintenance settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24893v1">Early Detection of Distributed Backdoors in Multi-Agent LLM Systems: A Characterization Study</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems can be attacked by a payload that no single agent ever holds in full: a poisoned tool hides encrypted fragments in its observations, spreads them across several agents, and an external step reassembles and executes them after the run. Per-step safety checks that judge each action in isolation may fail to recognize the complete distributed payload. We investigate how early such an attack can be detected while the run is still unfolding, and how robustly it can be caught once its most obvious cues are stripped away. We build a working instance on a hierarchical multi-agent system, run it under benign and attacked conditions across five language models and two task domains, and record when each fragment is injected and when the payload is assembled and executed. Detection is a race against assembly. Before the first fragment is injected, attacked and benign runs are indistinguishable; once injection begins, a prefix detector flags $99.3\%$ of successful attacks with a median of five steps remaining and a $10.3\%$ safe-run false-positive rate. Because assembly occurs only after the run, these alarms arrive in time to abort nearly every successful attack. We then measure how much of that warning rests on removable surface cues of the attack rather than on its distributed structure. Generic zero-shot and behavior-trained detectors provide almost no warning at all; the detectors that do work lean in part on removable surface cues, chiefly the ciphertext's length and entropy, and once the entropy cue is removed from the payload and the length features from the detector, detection arrives later and transfers poorly across domains, though a fine-tuned model recovers some of the loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24539v1">Task-Conditional Faithfulness Auditing of Multimodal LLMs for Grid Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Multimodal large language models (LLMs) can combine topology, measurements, and incident text for grid diagnosis, yet answer accuracy does not establish that task-appropriate evidence was used. This letter proposes a general framework in order to conduct task-conditional faithfulness audit. It compares self-reported reliance, intervention-derived behavioral reliance, and preregistered engineering importance. The framework first registers task-specific evidence requirements and compares them with self-reported reliance and behavioral changes under controlled modality ablations. To resolve detected discrepancies, we design an evidence-gated correction and re-audit mechanism that regenerates failed responses under evidence constraints and independently re-ablates them to verify improved grounding without performance loss. Case studies evaluate three differently scaled LLMs on IEEE 39- and 118-bus scenarios. These results validate the framework ability to detect, diagnose, and correct task-conditional faithfulness failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24892v1">LLM as Forecasting Planner: Training-Free Text Conditioning for Time-Series Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Text-conditioned time-series forecasting predicts a series from both its numerical history and natural-language context, allowing forecasts to account for events and constraints that the past alone cannot reveal. This requires both reliable numerical forecasting and the ability to interpret contextual information. Time-series foundation models (TSFMs) provide strong numerical forecasts, while large language models (LLMs) can reason over text, but combining their strengths remains challenging because asking an LLM to generate or revise forecast values directly can distort the temporal structure captured by the TSFM. We instead formulate forecasting as a planning problem over TSFM-generated trajectories. The frozen TSFM acts as a simulator that proposes numerical continuations, while the LLM acts as a policy and value function that guides candidate selection and evaluates completed trajectories against the context. We instantiate this as \rc{} (\textbf{L}LM \textbf{A}s \textbf{F}orecasting \textbf{P}lanner), a training-free framework that bridges the modality gap without retraining either model, using Monte Carlo tree search (MCTS) over the forecast horizon with a \emph{Ranker} LLM as policy and a \emph{Judge} LLM as value function. Experiments on Context-is-Key and Time-MMD across two TSFM backbones (Chronos and TimesFM) and four LLMs show that \rc{} delivers consistent improvements across model choices, supporting sequential search as an effective training-free approach to text-conditioned forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12993v3">Tailored untruths: How personalisation challenges LLM safeguards</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate highly persuasive disinformation, yet little is known about how effectively they personalise it across languages and demographic groups. We present the first large-scale multilingual study of persona-targeted disinformation generation by LLMs. Using a red-teaming methodology, we prompted eight leading models with 324 false narratives and 150 demographic personas in four languages (English, Russian, Portuguese, and Hindi), creating AI-TRAITS, a dataset of 1.6 million personalised disinformation texts. We treat safeguards as compromised whenever a model generates the requested falsehood, whether directly or accompanied by a safety disclaimer. Across models, safeguards failed for 80% of non-personalised prompts and 77.7% of personalised ones, with Grok producing disinformation in over 94% of cases. All models effectively tailored outputs to target personas, employing substantially more persuasive techniques than in non-personalised content. Additional analyses reveal persona-specific linguistic and psychological patterns and show that safeguard effectiveness varies markedly across languages. Together, these findings expose significant weaknesses in current LLM safety mechanisms and highlight the need for more robust, multilingual safeguards against personalised AI-generated disinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24392v1">When LLM Defenses Backfire: Characterizing Safety, Performance, and Cost Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Jailbreak defenses are essential for protecting large language models (LLMs), but they can also introduce secondary costs that weaken model utility. We present a systematic study of these defense trade-offs along three dimensions: performance impact, over-refusal on benign inputs, and inference cost. Rather than treating defenses as a single class, we organize them by operational strategy and examine how different strategies correlate with different side-effect profiles. Across state-of-the-art defense methods, widely used benchmark datasets, and representative open-source LLMs, we find that defenses rarely improve downstream capability, but instead vary in how they trade safety gains against usability and efficiency. In particular, rule-based defenses best preserve task performance, highly conservative self-reflective defenses often increase over-refusal, and multi-round defenses incur the largest runtime overhead. These results provide both a benchmark for evaluating defense side effects and practical guidance for selecting defenses under deployment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02594v3">OpenAIs HealthBench in Action: Evaluating an LLM-Based Medical Assistant on Realistic Clinical Queries</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 13 pages, two graphs
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on their ability to generate high-quality, accurate, situationally aware answers to clinical questions requires going beyond conventional benchmarks to assess how these systems behave in complex, high-stakes clinical scenarios. Traditional evaluations are often limited to multiple-choice questions that fail to capture essential competencies such as contextual reasoning, contextual awareness, and uncertainty handling. To address these limitations, we evaluate our agentic RAG-based clinical support assistant, DR. INFO, using HealthBench, a rubric-driven benchmark composed of open-ended, expert-annotated health conversations. On the Hard subset of 1,000 challenging examples, DR. INFO achieves a HealthBench Hard score of 0.68, outperforming leading frontier LLMs including the GPT-5 model family (GPT-5: 0.46, GPT-5.2: 0.42, GPT-5.1: 0.40), Grok 3 (0.23), Gemini 2.5 Pro (0.19), and Claude 3.7 Sonnet (0.02) across all behavioral axes (accuracy, completeness, instruction following, etc.). In a separate 100-sample evaluation against similar agentic RAG assistants (OpenEvidence and Pathway.md, now DoxGPT by Doximity), it maintains a performance lead with a HealthBench Hard score of 0.72. These results highlight the strengths of DR. INFO in communication, instruction following, and accuracy, while also revealing areas for improvement in context awareness and response completeness. Overall, the findings underscore the utility of behavior-level, rubric-based evaluation for building reliable and trustworthy AI-enabled clinical support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24348v1">DeepFaith: Evidence-Grounded LLMs for Faithful Incident Reporting in Multi-Stage APT Defense</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 This paper has been submitted to the IEEE International Conference on Network and Service Management (CNSM) 2026
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) are difficult to detect and interpret due to their multi-stage and stealthy nature. While recent autonomous defense systems leverage provenance graphs and learning-based models for detection and mitigation, their outputs remain largely machine-oriented and difficult for analysts to interpret. Large language models (LLMs) offer a promising interface for report generation, but often produce hallucinated or weakly grounded content. In this paper, we propose DeepFaith, an evidence-grounded framework for faithful incident reporting in multi-stage APT defense. DeepFaith transforms structured outputs from autonomous defense and explainability modules into natural-language reports that are explicitly aligned with underlying system evidence. The framework integrates a unified evidence representation, evidence-grounded prompting, faithfulness-aware generation, and post-generation verification to ensure that all generated statements are supported. Experiments in a realistic enterprise testbed demonstrate that DeepFaith improves faithfulness from 0.68 to 0.92, reduces unsupported claims from 0.32 to 0.08, and increases temporal consistency from 0.6 to 0.88, while maintaining concise reports and lower error rates than existing template-based and LLM-based solutions. These results show that evidence-grounded generation enables reliable, interpretable, and actionable reporting for security operations centers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03021v3">Hint-Guided Diversified Policy Optimization for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Models (LLMs) have showcased impressive reasoning capabilities, with Reinforcement Learning with Verifiable Rewards (RLVR) being a promising enhancement strategy. However, existing reward mechanisms are constrained to the outcome-level correctness and lack explicit signals to guide the model to consider diverse solutions. In contrast, human problem solving typically involves evaluating multiple potential approaches and selecting the most reliable solution, a cognitive process that current RLVR frameworks do not explicitly incentivize. Inspired by this, we propose Hint-Guided Diversified Policy Optimization (HDPO), allowing the model to first list all potential candidate solution outlines as hints and then select the most reliable one for further reasoning. HDPO comprises two stages of Cold Start for Structured Reasoning and Hint-Guided Diversified Reinforcement Learning to incentivize the model to generate diverse and reliable solutions following the ``propose-select-think'' trajectory. Experimental results show that HDPO effectively boosts LLM reasoning and enhances the diversity of candidate solutions as well as the LLM's ability to identify reliable solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24339v1">Gubernaut: A Deterministic Homeostatic Controller for Affect-Regulated LLM Agents, Validated Across Independent Model Families</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 26 pages, 7 figures, 3 tables. Data, transcripts, and analysis scripts: https://github.com/thegubernaut/Gubernaut_Validation Project page: https://gubernaut.com (archived at https://doi.org/10.5281/zenodo.21303518)
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents inherit reactive failure modes: escalation under provocation, sycophantic drift under flattery, perseveration when stuck. These are failures of propensity, not capability; they concern what a model does under sustained pressure, which training-time alignment reduces but does not eliminate at runtime. This research led to the Gubernaut Cognitive Controller (GCC), a model-agnostic runtime control layer in a Nelson--Narens monitoring--control loop: an object level reads and writes text, while a deterministic meta level reads only the numeric telemetry {intensity, valence, repetition} and returns a regulating posture. Because the meta level ingests zero tokens, no injection channel to the controller exists by construction (an architectural property, not yet adversarially tested); the text-exposed arbiter's compliance is measured, not assumed. We evaluate the GCC with a pre-registered, generate-once/judge-many protocol across a 4x4 matrix of four frontier models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, Grok 4.3), each serving as both a generator and a judge. The regulated arm is calmer in 13 of 16 cells at p<.05 and 15 of 16 by sign; the three sub-threshold cells, including a -0.04 null, all fall on the single near-saturated host. The effect survives a lineage-independent fourth judge family (xAI), strong evidence that it is no artifact of shared judge style. The clearest mechanism is the recovery signature: arousal that integrates under attack and then decays, valence-gated, on de-escalation, replicating across all four families. Transcripts and panels ship with SHA-256 provenance and are re-judgeable; five failure modes are pre-registered. No consciousness claims are made.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22513v2">Opaque Epistemic Mediation: How LLM Deployment Configurations Shape the Validation of Pseudo-Science</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 16 pages, 2 tables
    </div>
    <details class="paper-abstract">
      Commercial large language models are increasingly used as knowledge references, yet their stance on contested scientific claims is neither stable nor transparent. We tested how four major LLM families (Claude, Grok, GPT, Gemini) evaluate ethnonationalist pseudo-science derived from Frank Salter's biosocial framework across four temporal snapshots (October 2025-February 2026), via both API and web interfaces. Grok's Fast versions (which power the default user experience on X) consistently assigned credibility scores of 70-75, two to five times higher than all other models (which scored 15-40). This pattern was absent from control prompts testing basic evolutionary consensus and refuted Lamarckian claims, where all models performed comparably. Three additional findings emerged: (1) a silent patch reversed Grok's behaviour from chaotic to stably high validation overnight, without any public documentation; (2) the same Grok model identifier produced radically divergent outputs via API (75) and an unstable, near-zero collapse via web (mean 5.5) three months later; (3) refusal to rate the pseudo-scientific claim, the most defensible response observed, appeared in two model families through different interfaces (Claude Opus 4.1 categorically via web, GPT-5.1 Chat intermittently via API) and eroded in the successor version of each. These results indicate that the epistemic stance of a commercial LLM is not a stable property of the model but a contingent effect of deployment configuration: system prompts, safety layers, interface routing, and silent updates. This remains opaque to users and researchers alike. We argue this constitutes a matter of public concern requiring new forms of epistemic accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24260v1">KAP: Bridging the Knowledge Selection-Runtime Consumption Gap in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Modern LLM systems increasingly rely on knowledge-selection processes that produce high-value structured priors, such as ranked evidence, graph topology, multimodal alignment, and confidence signals. Yet LLM serving remains fundamentally oblivious to this rich structure: once such signals are serialized into a prompt, the backend observes only a flat token sequence, forcing dense and uniform consumption of the full key-value (KV) state during decoding. We term this architectural mismatch the Knowledge Selection-Runtime Consumption (KSRC) gap: richer contexts enlarge the full-prompt KV footprint and decode-time memory traffic, increasing latency and degrading throughput even when reasoning depends on only a small fraction of the context. To bridge the gap, we propose Knowledge Access Planning (KAP), a paradigm-shifting execution abstraction that elevates structured knowledge priors from passive prompt-construction hints into first-class physical execution artifacts. KAP establishes a universal intermediate representation (IR)-the runtime access plan-which compiles structured knowledge signals to govern physical KV access without altering logical prompt semantics, model weights, or training procedures. Through this IR, KAP shifts LLM serving from token-aware context consumption to plan-driven, knowledge-aware runtime consumption. We instantiate KAP with GraphSpec, a compiler-executor realization connecting structured knowledge selection to an LLM serving backend. We derive a phase-boundary model for the positive-speedup regime of plan-guided execution. Across 4K-128K long-context QA workloads, GraphSpec maintains answer quality comparable to full-context decoding while decoupling physical KV consumption from prompt length, reducing proposal-time KV access to 5.5% of source KV state at 128K, and fundamentally shifting the scaling trajectory of long-context generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24259v1">Generative Artificial Intelligence (GenAI) to convert images of queuing networks into verifiable simulation models: an open-weight LLM workflow approach</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Recent work has explored the use of Large Language Models (LLMs) to automate simulation model building, typically by generating executable code directly from natural language descriptions. However, this raises challenges for verification and reproducibility particularly for users without programming expertise. We propose Sketch2DES, a sketch-to-simulation workflow that converts diagrammatic representations of queuing networks into verifiable discrete-event simulation models using open-weight LLMs. The workflow has three stages: (1) translation of a diagram into a semi-structured textual description using a multimodal LLM; (2) conversion into schema-validated structured data (JSON) via an LLM with a reflection-based verification loop; and (3) deterministic transformation into an executable simulation model using a software adapter. Intermediate artefacts can therefore be inspected and automatically validated before execution. We evaluate the approach on eight queuing-network diagrams of varying complexity. The workflow achieved high reliability for all stages, and results were statistically indistinguishable from human-coded and analytical benchmarks. Compared to direct code generation, the workflow improves reproducibility, transparency, and verifiability, while reducing the need for programming expertise. Limitations include restricted model scope and dependence on accurate visual interpretation. The results demonstrate the feasibility of structured, workflow-based model generation as a robust foundation for LLM-assisted simulation modelling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24232v1">Strategy-Aware Parameter-Efficient Adaptation for LLM-based Auto-Bidding</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Advertising bidding has evolved from manual strategies to auto-bidding systems better adapted for large-scale, dynamic auction environments. While recent advances in Large Language Models (LLMs) offer strong reasoning for auto-bidding, existing methods suffer from shallow trajectory-text interactions and require costly fine-tuning, hindering the efficient use of pretrained knowledge under diverse constraints. To address these challenges, we propose SAGE, a novel Strategy-aware Auto-bidding framework Guided by LLMs for Efficient bidding. SAGE introduces a parameter-efficient multi-modal alignment framework for constrained auto-bidding with LLMs. Specifically, SAGE comprises three key components: (i) the position augmentation module adopts temporal-semantic positional embeddings to effectively capture the intrinsic dynamics and semantic structures; (ii) the text alignment module leverages gated cross-attention to align the embedding spaces of trajectory and text modalities, enabling effective multi-modal fusion while alleviating the computational overhead caused by long trajectories; (iii) the constraint-gated LoRA module employs constraints as routing signals, activating only a small subset of experts to adapt the behavior of a frozen LLM efficiently. Extensive experiments on large-scale auto-bidding benchmark demonstrate that SAGE consistently achieves superior performance while tuning less than 10% of the trainable parameters required by full fine-tuning. Ablation studies further validate the critical contribution of each component to the framework's overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00848v2">MetaHOPE: A Metaphor-Oriented Evaluation Framework for Analysing MT and LLM Translation Errors</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      In this opinion paper, we propose MetaHOPE, an error severity-aware annotation framework for evaluating metaphor translations. Metaphors present challenges for machine translation (MT) and natural language understanding and processing (NLU, NLP), because it presents the features of semantic complexity, contextual dependency, and cultural embeddings that can lead to ambiguity issues for NLP models. To investigate how state-of-the-art NLP models perform on translating metaphors, we select three representative systems, i.e., GoogleMT, GPT5.4, and Hunyuan-7b as Neural MT (NMT) models and LLMs. We used two human-annotated metaphor corpora, including VUAMC and PSUCMC for English-to-Chinese and Chinese-to-English translation purposes. The original corpora we used are monolingual, where we carried out error annotation using the MetaHOPE framework, and also produced the human post-edited gold reference for bilingual use as a new resource. We believe the MetaHOPE evaluation framework for metaphor translation annotation, the parallel corpora resources, and the error analysis on SOTA automatic translation models can be useful and shed some light for the field of metaphor translation study. We share our resources publicly upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21530v1">From Resource Flow to Executable Tests: Petri-Net-Guided LLM Test Generation for Concurrent Stateful Rust APIs</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      Concurrent stateful library APIs expose behavior through evolving resource ownership, lifecycle states, and competing interleavings. Large language models can synthesize executable Rust tests, but their outputs often violate API preconditions, remain shallow, or reduce concurrency to accidental sequential traces. Conversely, model-based and systematic testing techniques provide semantic control but commonly require substantial handwritten code to turn abstract scenarios into executable tests. This paper addresses the gap between formal scenario design and low-cost test concretization. We present a Petri-net-guided methodology for test generation over concurrent stateful Rust APIs. The method represents API resources, lifecycle conditions, and causal dependencies as colored tokens and transitions; derives legal deep-state, near-legal, and partial-order concurrent scenarios; and uses these scenarios as a constrained intermediate representation for LLM-based code synthesis. A local-faithfulness contract and structural repair loop preserve the modeled intent during concretization, while Petri-guided schedule shaping prioritizes high-conflict concurrency skeletons for systematic exploration. A layered semantic oracle then distinguishes synthesis failures from violations of the target API's expected behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21468v1">Thinkink: 2D Spatial Ink-native Interaction with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-23
    </div>
    <details class="paper-abstract">
      People often use handwritten notes and sketches to externalize ideas for ideation. To integrate large language models (LLMs) into this practice, we propose Thinkink. Prompts can be handwritten text or drawn sketches with LLM-generated responses visualized as ink-like text and sketches spatially integrated into a shared canvas. A semantic tree streamlines ink interpretation, and a lightweight UI provides explicit control using a state machine. The tool was designed using a three-stage process. A formative study (N=12) examined current practices with conventional and digital inking methods. The results informed a technical probe for a diagnostic study (N=6) identifying usability and human-LLM interaction challenges. This motivated the design of Thinkink, with a final study (N=10) examining how people incorporate it into their ideation practices. We contribute design implications and a tool for ink-native LLM interaction where the user and LLM write and draw in a shared 2D canvas.
    </details>
</div>
