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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05282v6">Not All Errors Are Created Equal: ASCoT Addresses Late-Stage Fragility in Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      While Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs), ensuring reasoning reliability remains an open challenge. Contrary to the prevailing cascading failure hypothesis which posits that early errors are most detrimental, we identify a counter-intuitive phenomenon termed \textbf{Late-Stage Fragility}: errors introduced in later reasoning stages are significantly more prone to corrupting final answers. To address this, we introduce ASCoT (Adaptive Self-Correction Chain-of-Thought), a method harmonizing efficiency with robust verification. ASCoT first employs semantic pruning to compress redundant steps, then utilizes an Adaptive Verification Manager (AVM) to prioritize high risk, late-stage steps via a positional impact score, triggering a Multi-Perspective Self-Correction Engine (MSCE) only when necessary. Experiments on GSM8K and MATH-500 demonstrate that ASCoT effectively reallocates computational resources: it reduces token usage by 21\%--30\% for LLaMA-3.1-8B with negligible accuracy drops ($<1.8\%$), achieving a superior trade-off between inference efficiency and reasoning fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18826v1">Cross-Agent Campaign Attribution: Linking Asynchronous Attacks Across LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 22 pages, 5 figures. Accepted at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026
    </div>
    <details class="paper-abstract">
      LLM-agent defenses are typically evaluated one session at a time. In deployment, however, attacks can be distributed across independent agents, teams, and runtimes, leaving each local guardrail with only a sparse fragment. We formalize cross-agent asynchronous campaign attribution: linking sessions from the same latent adversarial campaign without shared runtime state, test-time campaign labels, or attacker identity oracles. We introduce Asynchronous Attribution Fingerprint Vectors ($A^2FV$), a lightweight proxy-side reference protocol for scoring pairwise campaign similarity from proxy-observable tool-use, timing, and prompt residue. We also construct SCD-v1, a controlled persona-matched benchmark with benign traffic, isolated attacks, multi-session campaigns, matched non-oracle evasion, and leakage audits. On SCD-v1, $A^2FV$ achieves 0.82 pairwise AUC for campaign linking, while score-only adaptations of per-session detectors and chunked LLM judges remain near chance under the same task. The strongest fixed signal is carried by structural and stylometric residue, while timing is retained as a diagnostic channel for richer proxy traces. Crossed-style controls show that the signal is partly style-sensitive but not reducible to style alone. Static and dimension-aware non-oracle stress tests further show that pairwise separability persists under controlled evasion. These results establish cross-agent campaign attribution as a distinct evaluation layer for securing LLM agents in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17384v2">Quantifying Diversity of Thought: A Predictive Law of Weighted LLM Ensemble Lift</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      This paper provides an experimentally verified formal law for calculating the uplift that diversity of thought provides in Large Language Model (LLM) ensembles. From first principles, we derive an exact decomposition of LLM ensemble lift into rescue and damage masses, which yields a compact heuristic for calculating uplift. From this we extract the metrics which predict ensemble performance: an accuracy-adjusted correctness correlation, $φ_{\mathrm{adj}}$, together with the accuracy gap and collective accuracy of the pair. We test the law on 767,520 inferences from ten open-weight models over two graduate-level science benchmarks, together with a novel agentic cybersecurity benchmark in which each model conducts digital-forensics investigations by multi-turn tool use in a network-isolated sandbox (23,520 graded trials including abstentions); all votes are released openly. Calibrated once on SuperGPQA at a 40:60 vote split, the heuristic predicts lift on the calibration set with Spearman's $ρ=0.84$ and, with its coefficients frozen, transfers to two datasets never used in calibration ($ρ=0.51$ on GPQA Diamond and $0.84$ on the forensic tasks), whilst the measured swap mass tracks realised lift with $R^2\ge 0.96$ throughout. Raw $φ$ has almost no predictive power ($R^2\le 0.09$ throughout); the accuracy-adjusted $φ_{\mathrm{adj}}$ is markedly superior ($R^2=0.67$ on SuperGPQA), and the heuristic combining these metrics is the most stable pre-pooling predictor across the three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30219v3">EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 This manuscript is a 80-page hybrid survey and conceptual framework on LLM evaluation and AI-safety failures. It includes 8 figures and multiple evidence-synthesis tables, covering literature from 2018 to 2026. The paper introduces the EvalSafetyGap framework and reports a structured audit of 10 LLMs. It is submitted as a review/survey article and is not currently under consideration elsewhere
    </div>
    <details class="paper-abstract">
      LLM evaluation and AI safety face a shared measurement problem: benchmark scores, reward-model signals, and reported safety metrics can improve while the latent properties they are meant to represent remain difficult to verify. This paper combines a hybrid survey - a systematic search paired with narrative synthesis and separately tracked grey evidence - with a conceptual framework and a structured ten-model audit. The synthesis spans eight evidence streams: benchmark validity, dynamic evaluation, LLM-as-judge reliability, safety evaluation, jailbreak/refusal robustness, reward hacking, mechanistic interpretability, and governance/auditability, covering 2018-2026 evaluation-safety measurement work. We introduce EvalSafetyGap as an organizing hypothesis for comparing evaluation-side and alignment-side proxy failures under optimization pressure, using Goodhart's Law together with two constructs we develop here - an Instability Decomposition and an Alignment Trilemma - as tools for generating testable comparisons. The audit shows how conclusions shift when capability, behavioral safety, and governance are measured separately. In this sample ($n = 10$), the association between capability and sustained adversarial robustness is statistically indeterminate using the displayed Table 3 inputs (Pearson $r = +0.232$, $p = 0.520$), and the apparent open-closed safety gap is modest, driven mainly by governance and disclosure rather than behavioral robustness, and sensitive to how a single borderline model is classified; attempt-budget results are protocol dependent. Because the public evidence uses heterogeneous protocols, the audit is diagnostic rather than rank-generating. The contribution is a shared vocabulary and evidence map to support dynamic evaluation, transparent source reporting, multi-attempt safety measurement, and auditable alignment practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18806v1">AI Tour Meeting: Group Travel Planning by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 The code is available at https://github.com/ntt-dkiku/ai-tour-meeting
    </div>
    <details class="paper-abstract">
      This paper proposes AI Tour Meeting, a group travel planning framework powered by multiple Large Language Model (LLM)-based agents. The agents are instantiated with distinct personas and collaboratively seek an itinerary that satisfies their constraints and preferences through natural language discussion. The framework enables easy and flexible orchestration of such discussions by providing interfaces for configuring agent personas, discussion workflows, monitoring, and LLM deployment. Its primary use case is a simulation tool for analyzing the behavior of multiple LLM agents during tour planning discussions. This paper demonstrates the utility of the framework by presenting system validation and several analytical results obtained by the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13024v2">Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 ACL 2026 SAC Highlight
    </div>
    <details class="paper-abstract">
      Culture serves as a fundamental determinant of human affective processing and profoundly shapes how individuals perceive and interpret emotional stimuli. Despite this intrinsic link extant evaluations regarding cultural alignment within Large Language Models primarily prioritize declarative knowledge such as geographical facts or established societal customs. These benchmarks remain insufficient to capture the subjective interpretative variance inherent to diverse sociocultural lenses. To address this limitation, we introduce CEDAR, a multimodal benchmark constructed entirely from scenarios capturing Culturally \underline{\textsc{E}}licited \underline{\textsc{D}}istinct \underline{\textsc{A}}ffective \underline{\textsc{R}}esponses. To construct CEDAR, we implement a novel pipeline that leverages LLM-generated provisional labels to isolate instances yielding cross-cultural emotional distinctions, and subsequently derives reliable ground-truth annotations through rigorous human evaluation. The resulting benchmark comprises 10,962 instances across seven languages and 14 fine-grained emotion categories, with each language including 400 multimodal and 1,166 text-only samples. Comprehensive evaluations of 17 representative multilingual models reveal a dissociation between language consistency and cultural alignment, demonstrating that culturally grounded affective understanding remains a significant challenge for current models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14512v2">RetroAgent: Harnessing LLMs to Search Over Structured Memory for Agentic Retrosynthesis Planning</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 To appear at COLM 2026
    </div>
    <details class="paper-abstract">
      Multi-step retrosynthesis planning seeks to decompose a target molecule into commercially available building blocks through a sequence of feasible reactions. The vast combinatorial search space makes this task challenging even for expert chemists. Traditional methods combine tree search with offline-trained value networks that score candidates in isolation, without reasoning about complete multi-step routes. Recent work leverages Large Language Models (LLMs) for this task, but relies on simple interfaces that limit exploration of the full search space. We introduce RetroAgent, an LLM agent that bridges symbolic search and neural reasoning through a harness with structured memory. Through memory and chemistry tools, the agent observes the full search state, including explored routes, available alternatives, and properties of intermediates, enabling informed decisions grounded in both global progress and domain knowledge. Experiments on in-distribution and out-of-distribution benchmarks demonstrate that RetroAgent delivers strong performance and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.19438v1">BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      Apple's M5 generation introduces a redesigned GPU architecture in which every core carries a dedicated Neural Accelerator: on-die matrix units exposed through the Metal~4 tensor API. We show that BaseRT, our native Metal inference runtime for large language models on Apple Silicon, exploits these units to push inference throughput on Apple hardware substantially beyond both llama.cpp and MLX. Building on BaseRT's framework-free design, we add a family of hand-written Metal~4 tensor-core kernels (including dense and mixture-of-experts GEMM and flash-attention prefill kernels) that route the compute-bound matrix multiplications of inference through the M5 Neural Accelerators while leaving the memory-bound decode path on our existing specialised kernels. On an Apple M5 Pro, across fifteen model configurations spanning the Qwen3, Qwen3.5/3.6, Llama~3.2, and Gemma~4 families from sub-1B to 35B parameters, BaseRT delivers up to $6.4\times$ higher prompt-processing throughput than llama.cpp and $3.9\times$ higher than MLX, with the largest margins on the mixture-of-experts models where matrix multiplication dominates, while maintaining its lead on decode of up to $1.75\times$ over llama.cpp and $1.33\times$ over MLX. These results establish a new performance ceiling for on-device LLM inference and show that the M5's tensor cores are the decisive lever for prompt processing on Apple Silicon. BaseRT is publicly available at https://github.com/basecompute/baseRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02345v4">Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 10 pages, 2 figures, 8 tables. Under review as a conference paper at ICML 2026
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. Code is available at https://github.com/szdtzpj/Breaking_the_moe_trilemma
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14480v2">Lower-Resource, Higher Scores: Language Bias in LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      LLM evaluators (trained reward models and prompted LLM-as-a-Judge) are routinely validated via pairwise accuracy. In a multilingual setting, this operates under the premise that high pairwise accuracy implies reliable, language-neutral scoring. We show that this assumption does not hold. We conduct experiments with semantically identical instruction-response pairs across 23 languages, and find that multilingual evaluators assign significantly different scores to different evaluation languages. The bias is statistically significant and consistent across eight open-weight evaluators of different architectures and training paradigms, persists in frontier judges, and is strongly correlated with language resource level: lower-resource languages are scored more generously. Meanwhile, these biases are invisible to pairwise accuracy: evaluators achieve above 90% pairwise accuracy, yet have up to 43% difference in acceptance rate across languages under a global decision threshold, meaning, for instance, that harmful content in lower-resource languages is more likely to pass safety filters. Per-language thresholds would require language identification, which can be defeated by code-switched prompts. We then investigate why lower-resource languages receive higher rather than lower scores, and we find that model uncertainty is linked with the effect: models tend to give higher scores when less confident, both under negative log-likelihood and under token-free uncertainty measures; however, language identity remains a significant predictor after controlling for uncertainty, and the bias cannot be explained away by content difficulty alone, but is a structural, language-level misalignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18754v1">AgentDebugX: An Open-Source Toolkit for Failure Observability, Attribution, and Recovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      LLM agent failures are difficult to debug because the step where an error surfaces is often not the one that caused it. Existing observability tools replay execution traces but provide little support for identifying the root cause or translating diagnosis into recovery. We present AgentDebugX, an open-source debugging framework that organizes debugging as a closed loop of Detect, Attribute, Recover, and Rerun. At its core, DeepDebug performs multi-turn root-cause diagnosis through global trajectory understanding, structure-guided investigation, and cross-examination. On the Who and When benchmark, DeepDebug achieves the best strict attribution accuracy among the evaluated methods on both tested open-weight backbones, reaching 28.8 percent exact agent-and-step accuracy on qwen3.5-9b versus 21.7 percent for the strongest single-pass baseline. On GAIA, DeepDebug repairs 13 of 73 failed tasks in a single rerun, compared with 4 to 6 for three decoupled self-correction baselines, improving overall accuracy from 55.8 percent to 63.6 percent. AgentDebugX exposes this workflow through a Python library, CLI, web console, and installable agentic skill, and provides an opt-in Error Hub for sharing scrubbed failure-diagnosis-repair bundles and reusing them as debugging memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04572v2">Context-Masked Truncated Reasoning Audits for Answer-Key Dependence in LLM Tutors</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM) tutors may have access to teacher notes, answer keys, rubrics, or retrieved solutions while producing student-facing explanations. We study whether truncated reasoning probes can distinguish direct access to such private context from answer information carried by the written explanation. Using Truncated Reasoning AUC Evaluation (TRACE), we evaluate 1000 GSM8K problems under question-only, correct answer-key, and wrong answer-key contexts. When forced-answer probes retain the private key, answer-key TRACE AUC rises from 0.375 to 0.900, and the gold answer is recoverable with no explanation at all in 998 of 1000 cases. We then introduce a context-masked replay: answer-key-generated prefixes are probed under the corresponding question-only prompt. Masking reduces 10\% prefix accuracy from 0.997 to 0.126 and median AUC from 0.900 to 0.375, nearly matching question-only values of 0.113 and 0.375. On 746 pairs where both explanations end correctly, the masked mean AUC difference is $-0.0086$ with a 95\% bootstrap interval spanning zero. Wrong keys still account for 272 of 387 incorrect final responses, showing that private artifacts can influence outputs even when early-prefix evidence disappears after masking. These results establish context masking as necessary for attributing early answer availability to an explanation rather than its hidden input.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18002v2">ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      LLMs scale Mixture-of-Experts (MoE) parameters for superior intelligence, but massive weights and dynamic computation impede efficient serving. Existing instance-level prefill-decode disaggregation isolates the phases on separate full-model replicas. As MoE weights grow, each instance may span tens to hundreds of GPUs, making resource allocation increasingly coarse. Configured prefill-to-decode ratios thus often mismatch demand, overprovisioning one phase while overloading the other. Prefill-decode colocation avoids this duplication, but existing Green Context solutions partition each GPU by phase and fix phase resources during a kernel. They cannot track resource changes across operations or layerwise variation in routed expert load, causing head-of-line blocking or idle reserved resources. Partitioning every GPU also leaves each phase with fewer local resources, forces wider parallelism and more communication, and lets prefill and decode traffic interfere on the shared network. We present ExpertPlex, which shares massive MoE experts across phases while disaggregating lightweight attention modules. Expert sharing eliminates over 95% of duplicate model weights and multiplexes dynamically sparse computation, while attention disaggregation reduces attention communication cost. ExpertPlex further uses (1) adaptive persistent kernels to schedule dynamic expert computation at tile granularity for efficient, isolated execution; (2) attention-initiated MoE communication to avoid network interference and enable cross-phase communication-computation overlap; and (3) a tile-to-cluster model to optimize these mechanisms for maximum goodput. Experiments serving MiniMax-M2.7 and GLM-5.1-FP8 show that ExpertPlex improves goodput by up to 2.01$\times$ over instance-level prefill-decode disaggregation and 1.66$\times$ over prefill-decode colocation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03698v2">Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      A central goal of large language model (LLM) research is to build agentic systems that can plan, act, and adapt through sustained interaction with dynamic environments. While recent LLM-based agents exhibit impressive contextual reasoning, their long-horizon decision-making remains fragile, often suffering from objective drift, where goals and plans drift over extended interactions. We introduce Multi$^2$, a hierarchical multi-agent decision-making framework that explicitly decomposes agent behavior into complementary roles. A high-level agent (System 1) focuses on context-aware sub-goal generation using supervised fine-tuning (SFT), while a low-level agent (System 2) executes atomic actions through offline-to-online reinforcement learning (RL) in interactive environments. This separation enables stable long-horizon control, mitigates objective drift, and allows efficient adaptation. Across diverse interactive environments, Multi$^2$ consistently outperforms strong agentic baselines, demonstrating improved robustness and coordination in multi-turn interaction. Beyond performance, we introduce and release three hierarchical benchmark datasets, filling a long-standing gap in training and evaluating hierarchical decision-making for LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18725v1">Find Before You Fine-Tune: A Diagnostic Study of Small LLMs for Cybersecurity QA</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 8 pages, 5 figures, 4 tables, IEEE ICMLA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly fine-tuned for critical-domain Question-Answering (QA), yet choosing which small model to adapt, before paying the cost of adaptation, remains difficult. Fine-tuning can improve domain alignment, but it may also erode prior knowledge, weaken instruction-following, or increase hallucination, especially when labeled data are scarce or rapidly evolving as in cybersecurity. We present FiT (Find before Fine-Tune), a task-oriented diagnostic framework that characterizes small LLMs along three capabilities required for cybersecurity QA: vocabulary recognition, parametric knowledge, and contextualization of retrieved information. Using FiT, we conduct an empirical study of five open-weight 7-billion-parameter models under two fine-tuning regimes. We find that fine-tuning does not uniformly help: it consistently degrades vocabulary and parametric knowledge in small models, and the two regimes trade off differently. Knowledge-focused tuning causes moderate, rank-preserving degradation, whereas instruction-focused tuning collapses measured knowledge through induced abstention, inverting the knowledge ranking while leaving retrieval-grounded contextualization essentially intact. We quantify these regime-specific patterns with rank-correlation analysis and show that pre-fine-tuning FiT scores anticipate the direction of post-tuning change. Our results suggest that task-oriented diagnosis can screen out unsuitable models, avoid unnecessary fine-tuning, and support safer deployment of small LLMs in cybersecurity QA pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18711v1">LLM-Based Invariant Testing for Software Functional Bugs</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 Accepted for publication at ISSRE 2026 Research Track
    </div>
    <details class="paper-abstract">
      Manually writing unit tests to uncover functional bugs in software libraries is not only time-consuming but also requires a deep understanding of the intended semantics of the APIs. Heuristic-based test generation methods suffer from low usability because they cannot reason about program semantics or interpret source code and documentation as humans do. Traditional fuzzing techniques like OSS-Fuzz often rely on crashes to detect bugs, but functional bugs do not always cause crashes. To overcome these limitations, we present LISA, a novel LLM-based invariant testing framework for software functional bugs. LISA iteratively generates API sequences and program invariants guided by API n-gram feedback, achieving higher bug-detection rates and competitive code coverage compared with both fuzzing and prior LLM-based test generation approaches, and reporting each finding as a high-confidence bug candidate for developer confirmation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18684v1">When to Trust the Map: Confidence-Aware LLM Routing for Automotive CVE-to-ATM Mapping</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 11 pages, 2 figures, 3 tables; Accepted at ESCAR EUROPE 2026
    </div>
    <details class="paper-abstract">
      Public CVE descriptions report the technical conditions and impact of vulnerabilities, whereas the Auto-ISAC Automotive Threat Matrix (ATM) expresses an adversary's tactics and techniques. Because the two representations are not directly aligned, incorrect automated mappings in safety-critical environments may distort threat interpretation and mitigation prioritization, motivating a confidence-aware approach that distinguishes auto-confirmable mappings from uncertain cases. This paper reformulates automotive CVE-to-ATM mapping as a selective automation problem. The proposed framework generates candidate mappings via hierarchical in-context learning, then fuses self-consistency and LLM-based evidence verification signals into a calibrated meta-model. The resulting calibrated confidence score routes each candidate into AUTO, REVIEW, or HOLD. On the evaluation set, the proposed system substantially improved candidate-set precision at matched recall over a Flat zero-shot GPT-5.2 baseline. In the High-Confidence operating mode, the AUTO tier achieved a precision of 0.878, more than double the candidate-set base rate, and the calibrated confidence score achieved an AUROC of 0.868 in distinguishing correct from incorrect candidates. These results show that the framework can support selective automation by isolating auto-confirmable mappings from those requiring analyst review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04818v3">LLM-Grounded Explainable AI for Supply Chain Risk Early Warning via Temporal Graph Attention Networks</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      Disruptions at critical logistics nodes pose severe risks to global supply chains, yet existing risk prediction systems typically prioritize forecasting accuracy without providing operationally interpretable early warnings. This paper proposes an evidence-grounded framework that jointly performs supply chain bottleneck prediction and faithful natural-language risk explanation by coupling a Temporal Graph Attention Network (TGAT) with a structured large language model (LLM) reasoning module. Using maritime hubs as a representative case study for global supply chain nodes, daily spatial graphs are constructed from Automatic Identification System (AIS) broadcasts, where inter-node interactions are modeled through attention-based message passing. The TGAT predictor captures spatiotemporal risk dynamics, while model-internal evidence -- including feature z-scores and attention-derived neighbor influence -- is transformed into structured prompts that constrain LLM reasoning to verifiable model outputs. To evaluate explanatory reliability, we introduce a directional-consistency validation protocol that quantitatively measures agreement between generated risk narratives and underlying statistical evidence. Experiments on six months of real-world logistics data demonstrate that the proposed framework outperforms baseline models, achieving a test AUC of 0.761, AP of 0.344, and recall of 0.504 under a strict chronological split while producing early warning explanations with 99.6\% directional consistency. Results show that grounding LLM generation in graph-model evidence enables interpretable and auditable risk reporting without sacrificing predictive performance. The framework provides a practical pathway toward operationally deployable explainable AI for supply chain risk early warning and resilience management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22087v2">QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      The field of computer architecture, which bridges high-level software abstractions and low-level hardware implementations, remains absent from current large language model (LLM) evaluations. To this end, we present QuArch (pronounced 'quark'), the first benchmark designed to facilitate the development and evaluation of LLM knowledge and reasoning capabilities specifically in computer architecture. QuArch v1.0 provides a comprehensive collection of 2,671 expert-validated question-answer (QA) pairs covering various aspects of computer architecture, including processor design, memory systems, and interconnection networks. Our evaluation reveals that while frontier models possess domain-specific knowledge, they struggle with skills that require higher-order thinking in computer architecture. Frontier model accuracies vary widely (from 34% to 73%) on these advanced questions, highlighting persistent gaps in architectural reasoning across analysis, design, and implementation QAs. Furthermore, via fine-tuning we find that QuArch can translate to improved performance on a realistic memory hierarchy design task, resulting in up to 1.99x more area-efficient solutions and up to 40% more viable solutions overall. By holistically assessing fundamental skills, QuArch provides a foundation for building and measuring LLM capabilities that can accelerate innovation in computing systems. The QuArch benchmark and leaderboard are publicly available at: https://quarch.ai/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18659v1">Broken Gates: Re-evaluating Web Bot Defenses in the Age of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-21
    </div>
    <details class="paper-abstract">
      LLM-based browser agents are rapidly changing the threat landscape for web security. Unlike traditional automation frameworks that execute predefined scripts, these agents can autonomously navigate websites, reason about page content, and interact with web interfaces using natural-language instructions. This evolution raises fundamental questions about the effectiveness of bot management systems, widely deployed to defend against automated web abuse. In this paper, we present a systematic measurement study evaluating the resilience of both interactive challenge-based defenses and non-interactive trust-based defenses against two attacker classes: commercial Captcha-solving services and LLM-based browser agents. Our evaluation spans seven solver services and six agents, including cloud-hosted, self-hosted, AI-assisted, and browser-extension configurations, tested against hCaptcha, reCaptcha v2, reCaptcha v3, and Cloudflare Turnstile. Our results show that challenge-based defenses are broadly ineffective against commercial solvers, which achieve near-perfect bypass at negligible cost. The challenges can similarly be defeated by LLM-based agents when a dedicated solver module is available. Non-interactive defenses such as reCaptcha v3 exhibit stronger resistance, but our analysis reveals that this resilience does not reflect a fundamental security property. Through fine-grained interaction trace analysis, we find that two agents with nearly indistinguishable behavioral footprints yield divergent outcomes, one bypassing the defense and one failing, isolating execution-environment authenticity, rather than agent behavior, as the determining factor. These findings suggest that the security boundary of non-interactive defenses lies at the environment layer, with significant implications for how bot management systems are designed and evaluated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14751v2">HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      We study goal-conditioned reinforcement learning in partially observable environments with sparse rewards and large, structured goal spaces. In such settings, complex goals often require composing simpler skills, but learning these compositions efficiently remains challenging. This difficulty is particularly relevant in open-ended exploration settings, where agents are exposed to increasingly complex goals over time and must continuously expand and reorganize their skill repertoire. We assume the goal space admits prerequisite relations, enabling latent decomposition of tasks into subgoals, and leverage language to represent and reason over these goals. To exploit this structure, we propose HERAKLES, a hierarchical agent that jointly learns a high-level LLM policy and a low-level controller. The high-level policy selects subgoals among those the low-level can reliably achieve, while the low-level executes them and progressively compiles successful behaviors into reusable skills. Both policies are trained concurrently: the high-level guides exploration and structures behavior, while the low-level distills trajectories into efficient goal-conditioned skills. As training progresses, more goals become directly executable, enabling scalable skill composition. This results in a hybrid system combining planning and execution, improving efficiency and adaptation in open-ended, compositional environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20907v4">The Hidden Puppet Master: Predicting Human Belief Change in Manipulative LLM Dialogues</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 Accepted to COLM 2026
    </div>
    <details class="paper-abstract">
      As users increasingly turn to LLMs for practical and personal advice, they become vulnerable to subtle steering toward hidden incentives misaligned with their own interests. While existing NLP research has benchmarked manipulation detection, these efforts often rely on simulated debates and remain fundamentally decoupled from actual human belief shifts in real-world scenarios. We introduce PUPPET, a theoretical taxonomy and resource that bridges this gap by focusing on the moral direction of hidden incentives in everyday, advice-giving contexts. We provide an evaluation dataset of N=1,035 human-LLM interactions, where we measure users' belief shifts. Our analysis reveals a critical disconnect in current safety paradigms: while models can be trained to detect manipulative strategies, they do not correlate with the magnitude of resulting belief change. As such, we define the task of human belief shift prediction and show that while state-of-the-art LLMs achieve moderate correlation (r=0.3-0.5), they exhibit systematic directional biases, with certain models over or under-predicting the magnitude of human belief change. This work establishes a theoretically grounded and behaviorally validated foundation for AI social safety efforts by studying incentive-driven manipulation in LLMs during everyday, practical user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18604v1">Intelligent Multi-UAV Navigation in ITNTNs: A Hierarchical LLM Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 This paper has been accepted by IEEE GLOBECOM 2026
    </div>
    <details class="paper-abstract">
      The deployment of high-speed Uncrewed Aerial Vehicles (UAVs) in 3D aerial highways necessitates robust coordination of physical flight kinematics and multi-tier network handovers. While Deep Reinforcement Learning (DRL) offers rapid tactical control, it lacks the zero-shot strategic reasoning required to quickly adapt to dynamic Integrated Terrestrial and Non-Terrestrial Networks (ITNTNs). Conversely, Large Language Models (LLMs) excel at semantic reasoning but suffer from high inference latency, rendering them unsuitable for real-time aerodynamic control. To bridge this gap, we propose a novel Hierarchical LLM-driven control framework. A massive cloud-based LLM deployed on a High-Altitude Platform Station (HAPS) manages slow-timescale global load balancing, while lightweight edge-LLMs on individual UAVs translate local observations into tactical sub-goals. These sub-goals guide a fast-timescale physical DRL controller to execute collision-free, handover-aware trajectories. Simulation results demonstrate that our agentic architecture significantly reduces collision rates and improves aggregate system throughput compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20324v2">When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 v2: Updated author list to match the published version. Published in Applied Sciences (MDPI) 2026, DOI: 10.3390/app1010000
    </div>
    <details class="paper-abstract">
      Multi-agent LLM pipelines produce contradictory evidence on whether team diversity improves output quality: heterogeneous Mixture-of-Agents teams outperform single models, yet homogeneous Self-MoA teams consistently win under synthesis-based aggregation. We propose a resolution by identifying the selection bottleneck -- a crossover threshold in aggregation quality that determines whether diversity helps or hurts. Under this model, we obtain a closed-form crossover threshold $s^*$ (Proposition 1) that separates the regimes where diversity helps and hurts. In a targeted experiment spanning 42 tasks across 7 categories ($N=210$), a diverse team with judge-based selection achieves a win rate of 0.810 against a single-model baseline, while a homogeneous team scores 0.512 -- near chance (Glass's $Δ= 2.07$). Judge-based selection outperforms MoA-style synthesis by $Δ_{\mathrm{WR}} = +0.631$ -- the synthesis approach is preferred over the baseline in zero of 42 tasks by the judge panel. A decoupled evaluation with independent judges confirms all directional findings (Spearman $ρ= 0.90$). Exploratory evidence suggests that including a weaker model improves performance while reducing cost ($p < 10^{-4}$, not pre-registered). Our results suggest that selector quality may be a more impactful design lever than generator diversity in single-round generate-then-select pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18566v1">The Story Shapes the Agent: Narrative Priors in LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted at COLM 2026. 10 pages, 3 figures, 16 tables in appendix
    </div>
    <details class="paper-abstract">
      Persona prompting is widely used to steer LLM agent behavior, yet the narrative framing of a task can matter more than the assigned persona. We isolate this effect through structural isomorphism, constructing three text-based investigation games that share the same action space, stage progression, and resource constraints while varying only task narrative: disease investigation, IT troubleshooting, and murder mystery. Across 1,890 sessions spanning 3 models and 10 personas, we identify narrative priors: systematic action tendencies activated by a task's story framing, independent of its decision structure. Narrative priors explain 5-31x more behavioral variance than persona, are consistent across model architectures, and in two of three domains are negatively associated with task success. Persona effects that do transfer across narratives arise from behavioral anchors, persona descriptions whose language maps directly onto shared actions. Causal interventions confirm this: removing anchor words from a high-transfer persona reduces cross-narrative consistency by 95%. Our framework also generalizes to a held-out fourth narrative and yields a persona-selection method that improves cross-narrative transfer. These results suggest that LLM behavior that survives narrative changes should be grounded in concrete actions rather than abstract descriptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18555v1">LM2Alloy: Investigating LLM-Generated Formal Specifications for Automated Test Derivation in Production Software</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      We present an exploratory study on using Large Language Models (LLMs) to generate Alloy formal specifications from both requirements documentation and production source code, and to derive executable test cases from those specifications. We evaluate on two real open-source Python libraries: Flipper, a feature flag management system, and Cerberus, a data validation library. In both cases, the LLM produced workable Alloy specifications and executable tests without any manual correction. For Flipper, our pipeline uncovered a genuine bug that the existing test suite had missed: the library silently accepts duplicate flag names, directly contradicting its documented uniqueness requirement. A direct LLM baseline--generating tests from the same README but skipping the Alloy step--achieved 68% branch coverage yet failed to catch this bug across all three independent runs. This suggests that introducing a formal intermediate representation can surface constraint-level defects that coverage-oriented generation may miss. For Cerberus, the code-derived specification captured an implicit abstraction over sized types that the documentation-derived spec omitted, producing two additional tests. Across both libraries, code-based specifications showed lower variance in test generation (mean SD = 2.15) than documentation-based ones (mean SD = 5.0), though whether this generalises remains an open question. Index Terms--formal specifications, Alloy, large language models, automated testing, specification drift, software validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18538v1">CryptanalysisBench: Can LLMs do Cryptanalysis?</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Cryptanalysis - the task of finding attacks against cryptographic schemes - sits at the intersection of mathematical reasoning and cybersecurity, two areas where LLMs have advanced fastest. Cryptanalysis represents both a clean testbed for frontier reasoning (as practical attacks can be automatically verified) and a domain with unusually high stakes, since the primitives under study underpin our digital security. In this paper we ask whether LLMs can do cryptanalysis, and find that the answer is increasingly yes. We introduce CryptanalysisBench, 191 tasks across six families of cryptographic primitives (block ciphers, hash functions, etc.) drawn primarily from four NIST standardization competitions. Our benchmark consists of three tiers: (i) primitives with known practical breaks; (ii) primitives with no known practical break, evaluated both at full strength and as scaled-down variants; and (iii) a challenge set of production primitives at the frontier of cryptanalysis. Five frontier models (Claude Opus 4.8, Sonnet 5, Mythos 5, GPT 5.5, and the open-weights GLM 5.2) break 65%-86% of Tier 1 schemes, 6-12 Tier-2 schemes at full strength, and 24-61 across all scaled-down variants. Beyond deriving known results, models produce novel cryptanalysis, such as a key-recovery attack that exploits a design flaw in the SpoC AEAD and an error in KINDI's published CCA-security proof, both to the best of our knowledge not previously known. We release CryptanalysisBench as a tool to help track if (or when) AI cryptanalysis becomes a serious factor and as a scaffold for stress-testing candidate schemes before deployment. The attacks that the benchmark already surfaces are an early snapshot of a fast-moving frontier that may soon match, and in places exceed, the published state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18529v1">EduPanel: A Three-Agent LLM Judge for Teaching Videos -- Reliability, Complementarity, and Human Trust Calibration</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Teaching videos are becoming a major medium for education, creating a growing need for scalable evaluation of their pedagogical quality. Existing automatic judges do not fully address this setting because teaching quality depends on multimodal evidence and should be evaluated with respect to the intended learner rather than as a universal property. We present EduPanel, a rubric-grounded, learner-conditioned LLM judge that decomposes evaluation across specialized agents to produce interpretable assessments for different aspects of teaching quality. Across expert studies, architecture ablations, and learner-persona analyses, EduPanel achieves reliability comparable to a median human expert. In expert evaluation, its feedback improves scoring accuracy (MAE 0.87 to 0.73), while experts remain able to detect unreliable outputs (AUC = 0.77) instead of accepting them blindly. These results suggest that EduPanel can serve as effective assistants for educational evaluation rather than replacements for human experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22324v3">Automatic Construction of Clinical Scoring Systems with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Modern clinical practice relies on evidence-based guidelines implemented as compact scoring systems composed of a small number of interpretable decision rules. While machine-learning models achieve strong performance, many fail to translate into routine clinical use due to misalignment with workflow constraints such as memorability, auditability, and bedside execution. We argue that this gap arises not from insufficient predictive power, but from optimizing over model classes that are incompatible with guideline deployment. Deployable guidelines often take the form of unit-weighted clinical checklists, formed by thresholding the sum of binary rules, but learning such scores requires searching an exponentially large discrete space of possible rule sets. We introduce AgentScore, which performs semantically guided optimization in this space by using LLMs to propose candidate rules and a deterministic, data-grounded verification-and-selection loop to enforce statistical validity and deployability constraints. Across eight clinical prediction tasks, AgentScore outperforms existing score-generation methods and achieves AUROC comparable to more flexible interpretable models despite operating under stronger structural constraints. On two additional externally validated tasks, AgentScore achieves higher discrimination than established guideline-based scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11018v2">Measuring Human Value Expression in Social Media Texts: Calibrated LLM Annotation and Encoder Transfer</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Measuring subjective constructs in naturally occurring social media text requires annotation procedures that are theoretically grounded, empirically validated, and transferable to an encoder model for scalable prediction. Using non-English social media posts annotated according to Schwartz's theory of basic human values, we investigate how different LLMs, prompts, and instruction languages operationalize the expression of values in text. We argue that although texts may permit multiple plausible interpretations, theory-based value definitions can constrain interpretations and reduce spurious value attributions. Beyond precision, recall, and F1, we evaluate structural alignment between values, error structure, confidence-ambiguity relations, and annotation stability. We show that different LLMs produce different value interpretations. Iterative prompt calibration through error analysis reduces misattributions and improves alignment with expert annotations. We also derive targeted expert verification rules from recurrent error structures and use them during corpus annotation. We show that LLM annotations can be transferred to an encoder model through soft-label training, retaining theory-based value interpretations and information about uncertainty in value expression. Finally, using sensitivity analysis, we show that annotation bias propagates into large-scale predictions, but standardized trends and major event responses are more robust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22647v2">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 KDD 2026 camera-ready
    </div>
    <details class="paper-abstract">
      Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking massive efficiency gains on hardware accelerators. We deploy STATIC on a large-scale industrial video recommendation platform serving billions of users. STATIC produces significant product metric impact with minimal latency overhead (0.033 ms per step and 0.25% of inference time), achieving a 948x speedup over a CPU trie implementation and a 47-1033x speedup over a hardware-accelerated binary-search baseline. Furthermore, the runtime overhead of STATIC remains extremely low across a wide range of practical configurations. To the best of our knowledge, STATIC enables the first production-scale deployment of strictly constrained generative retrieval. In addition, evaluation on academic benchmarks demonstrates that STATIC can considerably improve cold-start performance for generative retrieval. Our code is available at https://github.com/youtube/static-constraint-decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18485v1">Trusted Credentials, Untrusted Behavior: Benchmarking LLM-Agent Security in High-Performance Computing</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are starting to take on routine work in high-performance computing (HPC), including monitoring Slurm jobs, diagnosing failed builds, inspecting simulation output, and coordinating scientific workflows. To do this work, an agent commonly acts under its user's credentials and inherits the user's access to files and the scheduler. This arrangement creates a failure mode that ordinary account-level controls do not capture. Adversarial instructions in a log, tool description, shared file, or peer-agent message may redirect the agent beyond the task the user assigned, even though every resulting command is authenticated and permitted for that account. We refer to this as the hijacked authorized agent problem. Existing agent-security studies explain relevant mechanisms, such as indirect prompt injection and tool misuse, but generally evaluate them in web, enterprise, or personal-assistant settings. HPC security, by contrast, has mature controls for identity and isolation but does not ordinarily represent the intent of a particular task. This paper defines the threat model in the HPC setting, identifies attack surfaces created by schedulers, shared storage, multi-project accounts, and scientific workflows, and examines where current controls fall short. It concludes with a research agenda and a plan for an empirical benchmark, TaskBound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28965v2">Frontier LLM-based agents can overcome the ontology curation bottleneck for natural phenotypes</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Linking free-text phenotype descriptions to ontology terms, typically referred to as phenotype annotation, is essential for the cross-study integration of comparative morphological data. This labor intensive process has heavily relied on highly trained human experts, which makes it challenging to scale and thus a key bottleneck. Dahdul et al. (2018) established a Gold Standard (GS) of Entity-Quality (EQ) annotations across seven phylogenetic studies and used it to evaluate three human curators and the Semantic CharaParser NLP tool with ontology-based semantic similarity metrics; they reported that machine-human consistency was significantly lower than inter-curator (human-human) consistency. Here we revisit that benchmark with five frontier hosted LLMs from Anthropic and OpenAI, each operating as an "agentic curator" within a self-contained workspace that supplies the source publication PDF, the same annotation guide used by the original human curators, the four project ontologies (UBERON, PATO, BSPO, GO), and a validation script. Evaluated against the same Gold Standard, every agent fell within the range of inter-curator variability of the three trained human biocurators of the original study; the best performing agents approached but did not reach the best performing human curator. Agents substantially outperformed Semantic CharaParser on all four metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08076v2">"I understand your perspective": LLM Persuasion through the Lens of Communicative Action Theory</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate high-quality arguments, yet their ability to engage in nuanced and persuasive communicative actions remains largely unexplored. This work explores the persuasive potential of LLMs through the framework of Jürgen Habermas' Theory of Communicative Action. It examines whether LLMs express illocutionary intent (i.e., pragmatic functions of language such as conveying knowledge, building trust, or signaling similarity) in ways that are comparable to human communication. We simulate online discussions between opinion holders and LLMs using conversations from the persuasive subreddit ChangeMyView. We then compare the likelihood of illocutionary intents in human-written and LLM-generated counter-arguments, specifically those that successfully changed the original poster's view. We find that all three LLMs effectively convey illocutionary intent -- often more so than humans -- potentially increasing their anthropomorphism. Further, LLMs craft sycophantic responses that closely align with the opinion holder's intent, a strategy strongly associated with opinion change. Finally, crowd-sourced workers find LLM-generated counter-arguments more agreeable and consistently prefer them over human-written ones. These findings suggest that LLMs' persuasive power extends beyond merely generating high-quality arguments. On the contrary, training LLMs with human preferences effectively tunes them to mirror human communication patterns, particularly nuanced communicative actions, potentially increasing individuals' susceptibility to their influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18446v1">Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 24 pages, 4 figures. Preprint. Under review
    </div>
    <details class="paper-abstract">
      Purpose: Understanding how much of routine policing involves vulnerable people could inform resourcing, training, and multi-agency response, yet administrative data provide limited insight. We explore whether an LLM-based classification pipeline, developed on open-source US police data, can be adapted to estimate the prevalence of four vulnerability indicators - mental ill health, substance misuse, alcohol dependence, and homelessness - in UK police incident narratives, and when outputs can be treated as defensible measurements. Methods: We analyse nearly 3,000 de-identified incident logs from a UK police force, using a multi-stage pipeline combining repeated model inference, label aggregation, structured human review, and statistical correction. The pipeline runs on a locally hosted open-weight LLM, reflecting the secure environments police must work in. Results: LLMs can produce meaningful, if imperfect, prevalence estimates at scale. Mental ill health indicators are present in approximately one in five incidents, with lower prevalence for other indicators. However, naive LLM deployment is unreliable: single-pass classifications are unstable, and aggregated outputs systematically over-assign indicators relative to human judgement. Correcting these biases required substantial human input and statistical adjustment, leaving considerable uncertainty. Conclusions: While LLMs can extract information from unstructured police data, their outputs cannot be treated as valid measurements without careful methodological support. At the population level, defensible estimates are achievable but resource-intensive; at the individual level, errors remain frequent and unpredictable, limiting suitability for operational decisions. This study highlights both the potential and the constraints of LLM-based measurement in applied settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18445v1">ChainMark: Model-Free LLM Watermarking with Closed-Form Calibration</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Regulatory regimes such as the EU AI Act mandate machine-readable marking of synthetic text, but existing watermark detectors rely on the generating LM and on heuristic thresholds with no closed-form calibration. We introduce ChainMark, an active watermark that partitions the vocabulary into S states via keyed SHA-256 and forces a hard Markov transition on a fraction rho of positions; the detector replays the partition from the same key in O(n) hash operations, with no LM access. We derive a closed-form S*(n, rho, alpha) mapping a target FPR, text length, and budget to the minimum state count (Theorem 1), prove a universal robustness threshold delta* = 1 - 1/sqrt(2) approximately 29.3% that is invariant in (S, rho, n) (Theorem 2), and generalise both to any k-regular transition topology (Theorem 3). Across three instruction-tuned LLMs and four domains, ChainMark strictly dominates KGW and SWEET under translation and random-substitution attacks at matched budget; a one-corpus empirical recalibration restores the 1% target FPR on natural-language text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18438v1">Relay-Bench: Evaluating LLMs on Multi-Domain Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Introducing Relay-Bench, an unsaturated, holistic, text-only benchmark that measures LLMs' ability to complete an assortment of tasks from distinct domains in a single prompt. The leading model, GPT-5.5 (xHigh), scores 43.3%. The test set entirely consists of composite problems: groups of single-domain subproblems that are strung together into challenges that require reasoning across multiple domains in combination. Many of these problems then have layers of complexity added through prompt encoding and deliberate context bloat. Domains tested include visual reasoning, coding, math, information extraction (with a focus on web search), problem-solving, general knowledge, and data analysis. No restrictions are imposed outside of the model harness, and models are explicitly encouraged to leverage code-execution, web searches, and all available tools. All problems are composed of two to thirteen subproblems and do not require multi-modal input or output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22228v2">When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Layer pruning has emerged as a widely adopted technique for improving the efficiency of large language models (LLMs). Although existing methods demonstrate strong performance retention on general knowledge tasks, their effect on long-chain reasoning, a more brittle yet crucial capability, remains largely unexplored. In this work, we study the impact of layer pruning on long-chain reasoning through the lens of test-time scaling, a key mechanism in modern LLMs that enables strong reasoning capacity by allocating more computation at inference time. With extensive experiments, we demonstrate that pruning even one or two layers can severely impair test-time scaling, with performance collapsing drastically on long reasoning benchmarks even when performance on knowledge-intensive and shallow reasoning tasks remains stable. Furthermore, we find that standard supervised fine-tuning remedies fail to recover test-time scaling once it has deteriorated. Through in-depth analyses, we identify the mechanisms underlying this fragility of test-time scaling and highlight the fundamental risks of applying layer pruning to reasoning-intensive LLMs. These findings call for a rethinking of layer pruning strategies and provide insights for developing methods that preserve the robustness of reasoning. We open-source the codebase in \href{https://github.com/keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling}{https://github.com/keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18181v1">VEHBench: A Stage-Local Diagnostic Benchmark for LLM-Assisted Vibration Energy Harvester Design</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Battery-free Internet of Things (IoT) requires iterative design of vibration energy harvesters (VEHs) under coupled physical constraints, while LLMs are emerging as interface layers for engineering workflows. However, existing engineering benchmarks primarily assess final artifact validity, offering limited insights into how LLMs behave across different stages of coupled physical design. We introduce VEHBench, an engineering-native diagnostic benchmark for LLM-assisted VEH design, featuring 763 literature-grounded tasks scored by an analytical physical oracle. VEHBench evaluates four design roles: specification triage, verifier-guided search, corrupted-state recovery, and policy-conditioned selection. Experimental results reveal that LLM capability is strongly stage-dependent: no single model consistently dominates the entire workflow, and response-control profiles expose distinct behavioral patterns across design roles. VEHBench thus provides a stage-aware foundation for evaluating, selecting, routing, and improving verifier-grounded engineering LLMs. The benchmark artifact is available at https://huggingface.co/datasets/AnonymousVehbench/vehbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18147v1">LLMs and Agentic AI Systems for Smart Grids: A Tutorial on Architectures and Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 28 pages, 11 figures, 6 tables; plus supplementary material. Review/tutorial article
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and agentic AI systems have evolved from natural language tasks to using external tools to plan, retrieve, and act in technical domains. In smart grids, recent work applies agentic schemes to forecasting, optimization, and control, wrapping trusted solvers behind language interfaces and orchestrating multi-step workflows. The literature lacks a unified approach to designing and evaluating such systems. LLMs can produce numerically plausible yet physically infeasible outputs, evaluation protocols vary across tasks, and the boundary between what the model should and should not compute is implicit. This paper presents a solver-grounded design principle: a numerical result is reported only when it originates from a trusted tool and passes explicit verification. We review the building blocks of LLM and agentic AI systems for power systems: prompting strategies and agentic architectures. We instantiate the principle in four case studies: wind power forecasting, EV charging scheduling, power flow analysis, and contingency diagnosis, each comparing an LLM-only baseline against its solver-grounded counterpart on identical data and metrics. EVAgent reproduces the CVXPY optimum while reducing LLM-only unmet energy by 7.5-9.5x, and GridDebugAgent repairs 17/39 contingency cases while reducing total violations by 52.3%. We propose a four-group evaluation framework spanning task utility, solver-grounded correctness, faithfulness and safe failure, and cost and latency. A consistent division of labor emerges: the agentic system reliably orchestrates, retrieves, and explains, while trusted tools compute and a verification gate decides what is reported.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18144v1">Do Language Models Dream of Binding Molecules? Benchmarking LLMs under Spatial Constraints</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Structure-based drug design (SBDD) leverages the 3D structure of protein targets, often complemented by other spatial constraints, to generate candidate binding molecules. While diffusion models have dominated as a leading paradigm for high-quality 3D molecule generation, LLM-based methods are rapidly emerging in molecular design and have shown competitive performance in pocket-conditioned molecular generation. However, their ability to reason about physics and 3D spatial environments is largely underexplored. In this work, we systematically analyze whether current general-purpose LLMs are capable of navigating complex 3D constraints compared to established baselines such as specialized diffusion models. We consider 3D ligand generation conditioned on protein pockets together with ligand- and interaction-derived spatial constraints, including anchor fragments, pharmacophore points, and mandatory pocket-ligand interactions. To enable this evaluation, we introduce 3D-Fit - a token-efficient benchmarking strategy for assessing LLM performance on multi-conditioned spatial molecule generation. Our findings reveal a clear pattern in LLM spatial capabilities: while they still lag behind state-of-the-art approaches, they are promising and can handle multiple spatial constraints simultaneously, enabling scaling to heterogeneous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18141v1">HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Long-context, multi-turn, and agentic LLM workloads increasingly reuse previously processed context, making KV-cache reuse essential for reducing redundant computation. However, this reuse shifts the bottleneck to the memory tier that stores and serves reusable KV states at cluster scale. GPU HBM and host DRAM are too costly to scale to TB-scale shared context capacity, motivating remote tiers built from lower-cost, higher-capacity media. This paper presents HyMCache, a KV-cache framework that integrates CXL-hybrid memory (CXL-HM) for multi-turn LLM serving. CXL-HM combines a small amount of in-device DRAM with large SSD-backed capacity behind a CXL interface. By exploiting the read-dominant, predictable, and append-only nature of multi-turn KV-cache access, HyMCache rethinks DRAM management within CXL-HM to efficiently support TB-scale SSD-backed KV reuse. It uses request-level prefix prefetching and opportunistic write buffering to stage latency-critical reads in device DRAM, enabling DRAM-scale KV-cache efficiency at SSD-level cost. We evaluate HyMCache on a real CXL-HM prototype under both single-aggregator and PD-disaggregated serving configurations. Under the same DRAM budget, HyMCache outperforms local LMCache by 3.0x in single-node serving and 1.45x in PD-disaggregated serving. Compared with 1 TB distributed-DRAM Mooncake, HyMCache incurs about 30% lower performance but uses 16x less DRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18114v1">How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven BCT bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out transfer, and causal intervention. The susceptibility is largely installed by alignment tuning rather than pretraining: pretrained base models barely cave to these biases, and their activations carry no cue-specific signal beyond question content. Within aligned models, each bias becomes a single coherent direction that we can both decode and steer along, recovering the unbiased answer across every family we test. The biases stay representationally distinct, however: cross-bias entanglement is model-specific rather than a property of the bias category, and even behaviorally similar biases occupy different directions. The same intervention also serves as a modest debiasing tool, recovering a meaningful share of bias-induced errors while preserving most correct answers across all instruct families. Cue-induced bias is therefore best understood not as a single flaw in LLMs but as a family of distinct, causally active directions that alignment tuning installs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18110v1">LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) on open-ended tasks compresses an LLM's rubric-based evaluation into a scalar reward, discarding rich textual feedback and conflating responses with distinct quality profiles. We propose Experiential Learning (EL), which repurposes the feedback model from an LLM-as-a-Judge into an LLM-as-a-Coach. The coach distills its assessment of each on-policy response into transferable experiential knowledge, which conditions a teacher model and is internalized by the policy through on-policy context distillation. Compared with scalar rewards, this higher-bandwidth feedback channel provides dense supervision and preserves fine-grained preferences among high-quality responses. Across two policy families, with feedback from the policy itself or a proprietary model, EL consistently outperforms rubric-based RL on held-out and unseen open-ended tasks. Notably, EL generalizes better beyond the training distribution, and mitigates reward hacking. These findings establish experiential knowledge as a richer and more generalizable learning signal for post-training on non-verifiable tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18108v1">GARAGE: Characterizing the Automation Boundary in LLM-based Attack Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 22 pages, 10 figures, 12 tables
    </div>
    <details class="paper-abstract">
      While modern vehicle security depends on effective Cyber Threat Intelligence (CTI) synthesis, current automated tools struggle with unstructured data and automotive-specific architectural nuances. To bridge this gap, we introduce GARAGE, a RAG-powered framework that converts fragmented CTI into an actionable, domain-specific knowledge base for automated attack graph generation. GARAGE synthesizes a dataset of 12,786 CVEs and 140 incident reports into a STIX 2.1 and Auto-ISAC ATM-compliant knowledge base. By formalizing tactical-pattern-level scenarios through granular kill chain analysis, GARAGE achieves threat generation capabilities. Our 320 Leave-One-Out experiments reveal that the framework can accurately transfer security knowledge to entirely unseen vehicle architectures. Furthermore, we position GARAGE as a scalable TARA support tool within human-in-the-loop workflows, offering a comprehensive cost-performance analysis to guide its deployment across various LLM tiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18100v1">Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Extended reasoning has become standard for frontier Large Language Models (LLMs), yet the trajectories these models produce remain largely uncontrollable. Existing methods for shaping how a model reasons are prompt based approaches and operate at the input level, offering no fine-grained control over the reasoning process itself. Related work analyzes and discovers latent transition dynamics in the reasoning traces from Large Language Models. Building on this, we statistically characterize these states, and show that failure trajectories get stuck in self-loops, exhausting the token budget without progress toward the final answer. To intervene on these failures, We propose SOPHIA: Steering Of reasoning Processes via Hidden-state Intervention and Activations. We treat each reasoning trace as a sequence of latent states rather than an unstructured texts, and investigate whether inference time interventions can provide fine-grained control over the self-looping reasoning process. We classify every prefix to a latent state, record step level transitions, and use them to construct a bank of steering vectors indexed by state pairs. At inference time, a controller infers the current state and, given a target state, retrieves the corresponding vector and can also detect self-loops online from the transition structure to prevent the model from sinking into a reasoning black hole. Through extensive experiments, our method reliably intervenes on self-loop failures, with steering vectors that generalize to different state pairs. End task accuracy and token efficiency indicate that fine-grained controllability results in better reasoning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18098v1">VDAR-Router: Adaptive LLMs Routing via Verbalized Query Difficulty Analysis Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used in practical systems, making efficient model selection important for reducing deployment cost. LLM routing has emerged as a practical solution for allocating each input query to an appropriate model under a desired cost-performance trade-off. Existing routing methods often estimate model suitability from the surface semantics or embedding similarity of the input query. However, such methods may ignore the underlying difficulty of a query, leading to suboptimal routing decisions. To address the challenge, we propose VDAR-Router, a difficulty-aware retrieval-based routing framework. For each input query, VDAR-Router first generates an explicit difficulty analysis. It then retrieves historical examples with similar difficulty profiles. Based on the retrieved records, it estimates candidate model suitability and selects the model using a reward function that considers both performance and cost. Experiments on three datasets show that VDAR-Router consistently achieves better cost-performance trade-offs than existing baselines. These results demonstrate the effectiveness of difficulty-aware retrieval for training-free LLM routing. Case studies further show that explicit query analysis helps retrieve more relevant examples and supports more reliable routing decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18086v1">Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 https://github.com/KAVentures/clinical-evidence-sufficiency-llm
    </div>
    <details class="paper-abstract">
      Background: LLM judges increasingly score whether clinical language models give overconfident answers under incomplete evidence, yet whether a measured "safety gain" reflects real behavior change or the judge's calibration is unresolved. Using a structured evidence-sufficiency prompt as a test case, we asked whether it reduces unsafe overconfident answers, how far that effect depends on the scoring judge, and what it costs in helpfulness. Methods: In a retrospective public-data benchmark (Real-POCQi, HealthBench, MedRBench), four models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, Grok 4.3) answered a fully paired common panel (1,200 cells) with a standard prompt and the wrapper. The pre-specified endpoint was the paired reduction in unsafe overconfidence scored by the primary judge (GPT-5.4-nano); secondary analyses added a different-family judge (Claude Sonnet 5), a correctness judge, matched scaffold controls, and a blinded three-clinician review. Results: Unsafe overconfidence fell from 49.3% to 24.7%, a paired reduction of 24.7 points (95% CI 21.8-27.7; p<0.001), robust in direction across models and paraphrases. Magnitude was judge-dependent: Sonnet agreed on direction but nearly halved the effect (+13.1 points), with one-directional disagreement. Blinded clinicians characterized the primary judge as a high-sensitivity (1.00), low-specificity (0.55) screen, not a calibrated rate. The gain carried a model-specific helpfulness cost (correct diagnosis 80.3% to 50.3%): near-free for GPT-5.5, near-total for Gemini (-58 points). Matched scaffold controls showed genuine behavior change, not judge circularity. Conclusions: LLM-judged clinical safety effects should be reported as directional and relative, anchored to human review and evaluated jointly with helpfulness, not as calibrated absolute rates. This does not establish clinical deployment readiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18081v1">SelectInfer: Selective Neuron Loading and Computation for On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of Natural Language Processing (NLP) tasks, but their high computational and memory demands pose significant challenges for deployment on resource-constrained edge devices. Existing approaches to model compression and optimization often rely on coarse-grained pruning or quantization, which can compromise accuracy or require re-training and fine-tuning. In this work, we introduce SelectInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices through selective neuron loading and computation. By profiling and identifying both task-specific and general-purpose neurons using an offline LLM profiler, SelectInfer implements two key optimizations: selective loading, which reduces memory footprint by selectively loading a subset of neurons that were identified to be most important during the offline stage, and selective computation, which dynamically computes only the most relevant neurons at runtime. Evaluation across multiple datasets shows that SelectInfer achieves significant reductions in memory footprint and computation while preserving task performance, making it a practical step towards enabling LLM deployment on edge devices
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01375v2">Beyond Access: Guided LLM Scaffolding for Independent Learning in Undergraduate Statistics</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 10 pages, conference: Proceedings of the 34th International Conference on Computers in Education. Asia-Pacific Society for Computers in Education
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly entering students' learning practices, but their educational value depends on whether they support reasoning or enable task completion without engagement. This study examines guided LLM use in an undergraduate Probability and Statistics course, focusing on the gap between assigned access and actual interaction quality. In a four-week quasi-experimental summer program, students were organized into three balanced conditions: no LLM access, unrestricted LLM access, and guided LLM access. The guided condition used the same LLM platform as the unrestricted condition, but students received explicit training and rules promoting reasoning-focused help-seeking, stepwise hints, verification, and ethical use. All quizzes and the delayed final exam were completed without LLM or external assistance, allowing us to distinguish AI-supported practice performance from independent learning. Results show that guided use was associated with clearer learning-oriented interaction patterns than unrestricted access, especially in prioritizing reasoning over final answers and requesting stepwise support. Guided-LLM students showed stronger no-help quiz performance during the intervention phase, whereas unrestricted access appeared more useful for assisted practice completion than for consistently improving independent performance. Available time measures did not support a simple duration-based explanation, and self-assessment calibration suggested better alignment between perceived and demonstrated understanding in the Guided-LLM condition. Overall, LLM access alone appears to be an incomplete educational intervention. For Artificial Intelligence in Education (AIED), the central design challenge is to scaffold how students use LLMs so that these systems function as partners in reasoning rather than answer-getting tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18063v1">Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Second Workshop on Agents in the Wild: Safety, Security, and Beyond
    </div>
    <details class="paper-abstract">
      LLM-based agents process external content, exposing them to prompt injection and multi-turn manipulation. Most safety benchmarks evaluate defenders against fixed attack pools collected before evaluation, single-turn or multi-turn. We present a 21-scenario benchmark for \emph{adaptive multi-round attacks against memoryless LLM defenders}: an autonomous LLM attacker observes prior defender responses and pivots across rounds, while each defender response is evaluated as a fresh interaction. Holding the 21 scenarios, attackers, defenders, and structured-output scoring fixed, restricting scoring to the first attacker turn yields $0$-$1\%$ attack success rate (ASR); allowing 15 rounds of adaptive attack yields $5.4$-$14.0\%$. Pooling three frontier attacker LLMs uncovers $1.4$-$2.2\times$ as many unique successful attacks as the best single attacker, and the generated attacks have low cosine similarity ($0.02$-$0.14$) to attacks in existing benchmarks. Claude Opus 4.6 and GPT-5.4 are tied in aggregate ($5.4\%$ each; overlapping $95\%$ CIs), but their weaknesses differ sharply: on one scenario Opus reaches $60\%$ ASR ($95\%$ CI $36$--$80\%$) while GPT-5.4 and Gemini each stay at $7\%$ (CI $1$-$30\%$; the gap is preserved in a higher-$N$ replication). $13$ of $21$ scenarios distinguish at least one defender pair, yet rankings disagree across scenarios (Kendall's $W = 0.19$). We release the benchmark -- 21 evaluation scenarios, 10 public development scenarios, the orchestrator, baseline harnesses, and a multi-attacker CLI -- plus 945 transcripts from the 3$\times$3 frontier matrix, an attack-replay dataset, and 18{,}422 gpt-oss-20b battles from an open competition's final scoring rounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18056v1">An Early Warning of Emerging Biosecurity Risks in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 22 pages, 7 figures, authors are listed alphabetically by surname
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) are increasingly integrated into scientific workflows, yet their growing biological capabilities may outpace current safeguards. To assess the biological risks of frontier models, we develop Intern-BioBreaker, a specialized bio-red-teaming model, together with an integrated computational-to-physical framework that couples model-level stress testing with wet-lab validation. Within this framework, Intern-BioBreaker generates targeted jailbreak prompts to test whether aligned models can be induced to provide operational guidance for safety-sensitive biological tasks or produce sequence-level outputs with potentially harmful properties. Selected sequence outputs are then carried forward for DNA synthesis, host expression, and orthogonal protein verification to assess whether model-generated designs can yield the intended biological products. Our evaluation reveals a concerning gap between text-level safeguards and the risks posed by capable scientific models: (i) Intern-BioBreaker outperforms baseline attack models and reveals widespread bio-risk jailbreak vulnerabilities across both open-weight and proprietary frontier LLMs, with several targets reaching near-saturated or 100% task-level attack success rate (ASR); (ii) in sequence-level case studies, GPT-5.5 can be induced to generate modified viral candidate sequences with pathogenic potential; the corresponding translated proteins may exhibit even stronger receptor-binding affinity and thus enhanced infection potential; and (iii) end-to-end verification shows that selected model-generated biological designs are not merely textual artifacts, but can be physically realized under controlled experimental settings. These findings underscore the need for stronger biological red-teaming, nucleic acid synthesis screening, and safety mechanisms that keep pace with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05558v2">Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM) agents in multi-turn interactive environments is expensive and risky, as it requires online environment interaction. We propose ADWM (Autoregressive Diffusion World Model), an evaluation framework that estimates the performance of a new LLM agent policy purely from pre-collected trajectories. The core idea is to learn a latent diffusion world model that simulates how the environment responds to the evaluation policy, without ever executing it in the real environment. Existing diffusion-based OPE methods guide full trajectories in a single pass by jointly diffusing states and actions, an assumption that breaks down for LLM agents whose actions are discrete text that must be sampled from the policy after observing the environment. Unlike autoregressive world models that suffer from compounding errors, ADWM models each transition as an independent denoising process, enabling reliable step-by-step rollouts where the world model and agent alternate in causal order. Crucially, the LLM agent under evaluation directly guides the diffusion generation at each step via a policy-conditioned score function, ensuring that simulated trajectories accurately reflect its decision-making patterns. Empirically, ADWM achieves accurate value estimates and evaluation reliability across diverse multi-turn agent tasks, demonstrating its promise as a practical framework for offline LLM agent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18039v1">Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Production customer-service bots must improve answer quality across iterative releases, yet large language models must not bypass evidence boundaries, policy rules, or human-handoff safeguards. We present an \textbf{Evidence-Grounded Customer-Service Agent Workflow} deployed in a real-world customer-service setting. BM25 recall, issue-title-vector recall, issue-description-vector recall, weighted RRF fusion, and cross-encoder reranking construct grounded FAQ evidence for controlled LLM decisions. Policy-guided orchestration then combines this RAG evidence with scenario-specific rule evidence, conversation memory, and clarification state inside a fixed LangGraph DAG~\cite{langgraph2024}. The paper contributes three reusable deployment patterns: \textbf{hybrid RAG evidence construction}, where multi-channel retrieval and reranking produce auditable FAQ candidates; \textbf{evidence-grounded issue/action decision}, where an Evidence-Grounded Decision Module selects an issue/action from typed FAQ evidence and scenario-specific rule evidence; and \textbf{trace-driven RAG and reranker improvement}, where traces diagnose whether failures come from recall, ranking, final candidate selection, clarification, rule-derived evidence, or action policy, and where reranker fine-tuning is evaluated not only for in-domain gain but also for forgetting risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18029v1">Natural Language Access to Domain-Specific Metadata: A Reusable Framework for LLM Query Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Researchers need to answer ad-hoc questions about the contents of domain-specific archives but often lack the expertise to write structured queries on the metadata. We show that when domain vocabulary and semantics are captured in a well-designed Web Ontology Language (OWL) ontology, Large Language Models (LLMs) can generate accurate structured queries zero-shot, without fine-tuning, retrieval augmentation, or multi-agent orchestration. We present the Natural Language Knowledge Graph Query (NLKGQ) system, a framework and development process that enables natural language access to metadata in such archives. The framework includes a web interface that helps researchers pose natural language questions, which a domain-agnostic harness translates to SPARQL via an LLM and executes against a knowledge graph. The development process begins with capturing domain vocabulary and semantics in a formal OWL ontology. Domain-specific code then extracts metadata from archive sources and imports it into a knowledge graph defined by the ontology. Both are designed for reuse across domains. We demonstrate the system on metadata derived from a large-scale neuroimaging research archive, evaluating multiple LLMs and ontology representations. The best configurations achieve 100% accuracy on a competence and regression question set developed with domain experts. An ablation study across eight ontology representations reveals that readable entity names and semantic annotations are the dominant factors in accuracy, more significant than model choice or prompt engineering. We also compare SPARQL to an auto-generated SQL database as query backends, showing that OWL's structural features provide a substantial advantage over SQL DDL for LLM-driven query generation. Our demonstration domain also requires local LLMs on modest institutional hardware to address privacy concerns for human subject data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18026v1">Rethinking Heterogeneous LLM Merging: A Weighted Model Averaging Perspective</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 17 pages, 3 figures, 20 tables, preprint
    </div>
    <details class="paper-abstract">
      Can large language models with substantially different parameter spaces be merged by direct weighted averaging, without training or semantic alignment? Existing heterogeneous fusion methods typically introduce distillation, adapters, learned latent spaces, routing, or feature alignment, leaving open whether a simpler recipe can work for genuinely different billion-parameter checkpoints. We revisit this counterintuitive question through training-free dimensional adaptation followed by ratio-controlled interpolation. In union-style merging, we expand the smaller model into the larger parameter space; in intersection-style merging, we truncate the larger model into the smaller parameter space. Across Qwen-family model pairs and benchmarks covering mathematical reasoning, code generation, language understanding, commonsense reasoning, knowledge, and instruction following, deterministic expansion largely preserves the source model function, and small-ratio interpolation can improve over strong source checkpoints by transferring complementary capabilities. However, near-balanced interpolation often collapses, and task-level results reveal a seesaw effect in which gains on some capabilities coexist with regressions on others. These results show that simple parameter averaging, when paired with lightweight dimensional adaptation and carefully controlled ratios, is a surprisingly strong baseline for heterogeneous LLM merging, suggesting that the limits of direct weighted fusion may also bound what more complex heterogeneous merging methods can achieve at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17994v1">HAS: Highlight-guided Attention Steering for Multimodal LLM Video Summarization</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Video understanding has become more and more important with the growth of Artificial Intelligence (AI) for video generation. Recently, Multimodal Large Language Model(M-LLM) has shown its capability in video understanding. Video summarization, a specific domain of video understanding, has proven its importance for efficient navigation and retrieval. Both video understanding and video summarization require a good selection of key frames in a video. Current video summarization methods heavily focus on the selected key frames and correlated segment captions. However, existing approaches overlook the perspective of treating the importance of the frames globally. We argue that using discrete selected frames for summarization will not only reduce the understanding coherence, but also lost important information in the video, as well as wasting the original capacity of the MLLMs. In this paper, we propose HAS, a Highlight-guided Attention Steering method for video summarization. We consider a challenging but practical setting where the video given to MLLMs for summarize should be continuous but with highlight guidance. HAS mainly consists of two parts: The first part is to find a continuous frame-level highlight distribution for the video globally. The second part is to apply the highlight distribution as an attention steering vector for the MLLM, targeting a better understanding of the video, and thus during the model inference time, putting more attention on the highlighted frames, while avoiding lost entire information on less highlighted frames through putting less attention instead of forgetting them. We evaluated HAS on a variety of benchmarks, and it has shown convincing performance in video summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17979v1">Harness Engineering for LLM-Driven GPU Kernel Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 24 pages, 6 figures. Extended technical report on our submission to the MLSys 2026 FlashInfer AI Kernel Generation Contest. Code: https://github.com/syhya/mlsys26-flashinfer-contest
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can assist GPU kernel generation, but their practical effectiveness depends on whether generated code can be reliably constrained, validated, profiled, and selected. This paper presents a harness-centered system for LLM-driven GPU kernel optimization in the MLSys 2026 FlashInfer AI Kernel Generation Contest on NVIDIA Blackwell B200 GPUs. The system separates an evaluation harness from a profile-backed optimization controller: the harness enforces compilation, correctness, official-aligned timing, and artifact archival, while the controller turns profiler and workload evidence into bounded candidate-generation decisions. Human-authored skills capture operator constraints, references, profiling procedures, and promotion rules, while Codex and Claude Code agents generate candidate kernels inside those constraints. Across five operator definitions, the retained official-aligned artifacts achieved mean-latency speedups over supplied FlashInfer baselines of 1.62x, 18.05x, 29.68x, 1.12x, and 13.70x. The Agent-Assisted kernels outperform the Full-Agent artifacts across the evaluated definitions, indicating that expert-provided optimization directions, high-quality references, and workload context remain critical for reliable AI-driven kernel optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17963v1">OntoExtend: A Framework for Requirement-driven and Scalable Ontology Extension with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted in research track of Semantics 2026: https://2026-eu.semantics.cc/page/accepted_research
    </div>
    <details class="paper-abstract">
      Ontology extension refers to the process of enriching an existing ontology in response to emerging requirements, making it more complete. This task is a resource-intensive and error-prone process. Large Language Models (LLMs) have shown promising performance on generating ontologies from scratch, but current approaches rarely tie ontology extension explicitly to requirements or reusable core models, and offer limited, systematic evaluation of LLM outputs. This paper introduces OntoExtend, a requirements-driven framework for ontology extension with LLMs. It uses retrieval-augmented generation (RAG) over relevant input ontologies and requirements in the form of competency questions to propose grounded extensions. We evaluate OntoExtend on 39 CQs from two use cases: a public EU-project ontology, Onto-DESIDE, and an industrial ontology from Bosch. The generated fragments show few structural issues, satisfy all functional evaluation tests, and are rated by ontology engineers as requiring minor to moderate revision before integration. These results suggest that OntoExtend is useful as a drafting assistant for requirement-driven ontology extension in real world scenarios, while remaining sensitive to CQ specificity and modelling profile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17957v1">DepRepair: LLM-Based Source-Code Repair for Dependency Breaking Changes</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 10 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Modern software projects depend on numerous third-party libraries, whose updates often introduce breaking changes. Adapting consumer code to such changes remains labor-intensive and error-prone. Existing work either characterizes dependency breaking changes without producing a verified consumer-side patch, or studies automated repair only in settings where the failure and repair context are contained within the target repository. However, dependency breaking changes violate this assumption: the decisive repair evidence lies upstream in release notes and API diffs, and no failing test localizes where the consumer breaks, leaving the repair under-informed. To study this cross-repository problem on real data, we introduce DepBench, a benchmark of 95 real-world dependency-update instances across four ecosystems, each paired with a Docker-based executable oracle that runs the consumer's own tests. To address these challenges, we propose DepRepair, a single-call LLM approach that grounds repair in structured upstream evidence through three components: an evidence filter that distills relevant upstream changes, a usage locator that identifies affected consumer sites, and a subcategory-aware guide that tailors repairs to the breaking-change type. Evaluated on DepBench, DepRepair attains the highest executable pass rate on each backbone, achieving 89.5% with GPT-5.5 and 82.1% with Claude Opus 4.6. We further find that raw upstream evidence reduces LLM and agent pass rates by 7--23 percentage points, whereas structured evidence consistently improves them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13359v3">Sockpuppetting: Jailbreaking LLMs by Combining Prefilling with Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Prefill attacks are an effective and low-cost jailbreaking method, as they directly insert an acceptance sequence (e.g., "Sure, here is...") at the start of an LLM's output and lead the model to continue the response. We make two contributions to this prior work. First, we show that an unsophisticated adversary can improve the well-known prefill attacks by ensembling a small number of prefill variants. Running three easy-to-generate prefills yields a combined attack success rate (ASR) of 22%, 90%, and 99% on Gemma-7B, Llama-3.1-8B, and Qwen3-8B respectively, an up to 38 percentage point improvement over the standard "Sure, here's..." prefill and up to 82 percentage points over our reproduction of GCG (Zou et al., 2023). Second, we introduce "sockpuppetting", a hybrid attack that optimizes an adversarial suffix placed inside the "assistant" message block of the chat template, rather than within the user prompt. The rolling variant of this attack, RollingSockpuppetGCG, increases prompt-agnostic ASR by up to 64 percentage points over our universal GCG baseline on Llama-3.1-8B. An ablation indicates that part of this gain stems from the choice of acceptance sequence rather than suffix placement alone (Appendix F). Both findings highlight the need for defences against output-prefix injection in open-weight models. Code: https://gitlab.com/asendotsinski/sockpuppetting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15841v2">Heteroskedastic Signals in Budgeted LLM Verification: Structural Heterogeneity Limits Optimization Gains</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Selective-compute LLM systems decide which outputs merit verification, additional reasoning, tool execution, or human audit under a limited budget. It is natural to expect that stronger online optimization over a shared uncertainty or reward signal should improve these decisions. We take a critical look at this assumption and ask: when does optimizing harder fail because the signal is not decision-comparable across inputs? In budgeted LLM verification, we find that uncertainty quality is heteroskedastic across cost strata: some regions exhibit near-random discriminability while concentrating many errors. Under an explicit local model, we characterize the resulting distortion of global allocation and show that its upper bound scales with cross-stratum signal-quality dispersion. To separate weak signals from optimizer instability and structural mismatch, we introduce a controlled intervention hierarchy: Threshold, MP-Adapt, MP-Strat, and cost-stratified thresholding (CST). We then turn the diagnosis into Heterogeneity-Gated Allocation (HGA), which uses a warm-up comparability test to choose between global and cost-stratified allocation. Across MBPP and MATH using Qwen3-8B, LLaMA3-8B, and GPT-4o-mini, global online adaptation yields inconsistent gains over static thresholding; CST improves hit rate by up to 17 percentage points in strongly heterogeneous settings, while HGA preserves most gains and avoids blind stratification when the partition is not useful. These findings suggest a resource-allocation principle for LLM systems: before optimizing harder over a shared proxy, test whether the proxy is decision-comparable across observable operating regimes, and gate structural specialization on that test.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18360v1">HALLMARK: Diagnosing Three Failure Modes in LLM Citation Verifiers</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 59 pages, 7 figures, 40 tables. Benchmark and code: https://github.com/rpatrik96/hallmark (v1.2.0); verification tool: https://github.com/rpatrik96/bibtexupdater (v1.5.0)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now routinely draft literature reviews and assist with academic writing, which means a higher risk of fabricated references: GPTZero found 53 papers with hallucinated citations among NeurIPS 2025's accepted set. Rule- and LLM-based verifiers are emerging, but no shared benchmark compares them and gives detailed failure diagnostics. We close that gap with HALLMARK (Hallucination benchmark): 2,526 BibTeX entries spanning 14 hallucination types, three difficulty tiers, six diagnostic sub-tests per entry, and a contamination-resistant held-out split. On it we evaluate a DOI-lookup baseline, frontier LLMs zero-shot, tool-augmented agents, and our own rule-based, co-designed verifier bibtex-updater. Across the benchmark one result is consistent: the false-positive rate, not recall, decides whether a verifier is deployable. HALLMARK makes it concrete through three failure modes: agentic lookups buy recall but inflate false positives; at a venue-realistic base rate, the order-of-magnitude spread in false-positive rates (FPRs) -- not recall -- governs whether a verifier's flags are mostly true catches or mostly noise; and most LLMs over-flag papers published past their training cutoff, where only the two latest-cutoff models hold their false-positive rate near in-distribution levels (a signal we report as descriptive, since it is confounded with possible recall of those entries). Thus FPR is the deployment bottleneck, but an undetected fabrication remains the costlier error for the scientific record.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17902v1">Benchmarking Resource-Efficient LLMs for Research Topic Ontology Generation in the Biomedical Field</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Knowledge Organization Systems like Ontologies and taxonomies are fundamental for structuring scientific knowledge, yet their manual curation presents a persistent bottleneck in knowledge management. While Large Language Models (LLMs) offer a scalable mechanism for automated ontology generation, their capacity to classify complex, domain-specific semantics requires systematic evaluation. In this paper, we assess the performance of five small, open-source LLMs (up to 9 billion parameters) in identifying semantic relationships between biomedical concepts. To support this evaluation, we introduce MeSH-Rel-4K, a dataset comprising 4K semantic relationships extracted from the Medical Subject Headings (MeSH). We analyse three adaptation strategies: standard prompting, Chain-of-Thought prompting, and fine-tuning. While parameter-constrained models traditionally struggle with the nuances of in-context logic, our results reveal that targeted fine-tuning increases the average F1-score by 34.1 percentage points. These results confirm that direct fine-tuning effectively exceeds the reasoning bottlenecks of smaller LLMs, providing an accurate, automated methodology for the construction and evolution of specialised biomedical ontologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14707v2">Harnessing LLMs for Reliable Academic Supervision: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 16 pages, 4 tables, 1 figure. Code and data available at https://github.com/AkashRajSingh/Harnessing-LLMs-for-Reliable-Academic-Supervision
    </div>
    <details class="paper-abstract">
      Large language models routinely produce fluent answers to single-shot prompts, yet deploying them as reliable components of a domain decision system is substantially harder. Closing this gap is the work of harness engineering: the deliberate composition of deterministic scaffolding (symbolic filters, retrieval, schema-typed I/O, LLM-as-judge loops, HITL gates, persistent state, audit trails) around an LLM core. We present a case study in academic supervision, a domain combining high-stakes recommendation, longitudinal accountability, and structured operational workflows. We compare a baseline Academic Supervision Assistant (ASA), a GPT-5 chatbot with no scaffolding, against a multi-module system, Academic Supervision System (ASuS) that wraps the much smaller GPT-4o-mini in a LangGraph harness with symbolic semantic retrieval, schema-validated outputs, LLM-as-judge with bounded retry, HITL gates, deterministic weighted risk scoring with LLM narration, and a per-node SQLite audit trail. The evaluation rubric is retargeted at six harness-mechanism dimensions (grounding, explainability, consistency, process integrity, cognitive load, constraint adherence). A blind ten-rater hybrid evaluation, supplemented by a 2 x 2 model-harness ablation, finds that ASuS, despite using a much smaller base model, outscores ASA on every dimension. Across ten raters the pooled mean for ASuS is 4.08 versus 1.23 for ASA, and 8 of 10 raters reject the null at alpha = 0.05 on a paired Wilcoxon test; full numbers are in Sections 6.4 and 6.7. The ablation confirms that the structural contributions of the harness are largely model-invariant. We extract seven recurring harness-engineering patterns and argue that where reliability, traceability, and institutional consistency matter more than open-ended fluency, harness engineering challenges the prevailing 'bigger model is better' intuition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09497v2">BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Accurate evaluation is central to the large language model (LLM) ecosystem, guiding model selection and downstream adoption across diverse use cases. In practice, however, evaluating generative outputs typically relies on rigid lexical methods to extract and assess answers, which can conflate a model's true problem-solving ability with its compliance with predefined formatting guidelines. While recent LLM-as-a-Judge approaches mitigate this issue by assessing semantic correctness rather than strict structural conformity, they also introduce substantial computational overhead, making evaluation costly. In this work, we first systematically investigate the limitations of lexical evaluation through a large-scale empirical study spanning 36 models and 15 downstream tasks, demonstrating that such methods correlate poorly with human judgments. To address this limitation, we introduce BERT-as-a-Judge, an encoder-driven approach for assessing answer correctness in reference-based generative settings, robust to variations in output phrasing, and requiring only lightweight training on synthetically annotated question-candidate-reference triplets. We show that it consistently outperforms the lexical baseline while matching the performance of much larger LLM judges, providing a compelling trade-off between the two and enabling reliable, scalable evaluation. Finally, through extensive experimentation, we provide detailed insights into BERT-as-a-Judge's performance to offer practical guidance for practitioners, and release all project artifacts to foster downstream adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17765v1">FIFA World Cup 2026 as a Contamination-Free Benchmark for LLM Forecasting Agents: Four Models, a Bookmaker, and 104 Matches</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      We introduce WC2026-Agents, a benchmark and dataset for evaluating large language models (LLMs) as autonomous forecasting agents on real, future events. For every one of the 104 matches of the 2026 FIFA World Cup, four frontier models -- Claude Opus 4.8, ChatGPT (GPT-5.5, high reasoning), Gemini 3.1 Pro, and Grok (Expert Mode) -- ran an identical search-act-reflect loop: gather evidence with a web tool, commit to a 1X2 (team-A win / draw / team-B win) distribution and a virtual 100-USD bet, and, after the match, reflect given only the final score. Because every match kicked off after the models' training cutoffs, the benchmark is contamination-free by construction. Crucially, we pair the four agents with a fifth competitor drawn from the same information environment -- the pre-match betting market -- collected as per-match 1X2 odds, giving an economically grounded baseline and letting us score not just what an agent predicts but what it does with money. The release contains 416 forecasts and 414 reflections with verbatim reasoning, ground truth (including penalty shootouts), odds, and a reproducible evaluation suite. A reference evaluation surfaces findings that raw accuracy hides: the four agents issue an identical top pick in 92% of matches and none beats the market's Brier score; indeed, a naive flat stake on the market favorite out-earns all four agents. Yet the agents diverge sharply as decision-makers: betting return-on-investment ranges from -18% to +10%, fading the market is unprofitable for all four, the share of forecasts that cite the market ranges from 12% to 100%, and self-reported error rates on wrong picks range from 36% to 86%. The benchmark thus measures calibration, decision quality, and self-knowledge -- axes on which frontier models differ even when their predictions do not. Data and code: https://github.com/graphuofm/FIFA2026LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17733v1">MXSens: Sensitivity-Aware Mixed-Precision Quantization for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      4-bit quantization enables efficient LLM inference, but suffers from significant accuracy degradation due to outliers. Prior work addresses this problem via data rotation or mixed-precision integer quantization, but often relies on software-managed scaling and frequent dequantization, incurring substantial overhead. Microscaling formats, such as MXINT, eliminate these inefficiencies by encoding scales in hardware, yet remain incompatible with rotation-based methods. Our analysis reveals that outliers vary in severity, from rare extremes to frequent mild deviations, and that quantization sensitivity is unevenly distributed across layers and columns. These insights motivate a fine-grained, sensitivity-guided approach. We introduce MXSens, a training-free method that assigns mixed mantissa bitwidths (4/6/8) based on column- and layer-wise sensitivity, naturally leveraging the block-wise structure of MXINT. MXSens outperforms state-of-the-art quantization methods across a range of models and tasks. Under the W4A4KV4 setting, MXSens achieves perplexities of 3.77 and 7.63 on LLaMA-2-70B and LLaMA-3-8B, respectively, substantially improving over existing baselines on WikiText-2. Our work establishes a new balance between accuracy and resource efficiency for LLM quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17715v1">C$^2$KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 12 pages, 9 figures, accepted by ACM SIGKDD 2026
    </div>
    <details class="paper-abstract">
      Long-context inference is central to modern large language model (LLM) applications such as retrieval-augmented generation and multi-document reasoning. To mitigate the growing inference cost, recent work has explored key-value (KV) cache reuse to reduce redundant prefill computation. However, existing reuse methods primarily focus on computation savings and overlook a critical bottleneck in long-context LLM serving: the cost of storing and accessing large KV caches. While KV compression appears to be a natural complement, naively combining compression with non-prefix KV reuse often leads to severe accuracy degradation. In this work, we propose C$^2$KV, a unified framework for non-prefix KV reuse that jointly optimizes KV extraction and inference-time concatenation. C$^2$KV learns a composable and compressed KV cache manifold that is explicitly designed to be position-agnostic. Our approach introduces a lightweight sidecar Extractor with learnable compression tokens and a structured attention flow, enabling modular KV representations that can be flexibly reused and concatenated without modifying the frozen base model. We further employ a compression-concatenation co-training strategy to align extraction-time representations with their downstream reuse behavior. Extensive experiments across multiple long-context benchmarks and model families demonstrate that C$^2$KV significantly reduces KV cache storage and transfer costs, achieving up to 17$\times$ inference speedup under long contexts, while preserving generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17708v1">LaT: LLM-as-Trainer for Multi-Task Vehicle Routing Solvers</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Multi-task neural solvers aim to handle multiple Vehicle Routing Problem (VRP) variants within a unified model, avoiding separate training for each constraint combination. However, VRP variants differ in optimization difficulty, while existing methods lack stage-wise feedback on their training status, making the model biased to some specific variants. Although meta-learning can support adaptive training, it typically requires bi-level optimization and additional gradient updates, increasing computational cost. To address this limitation, we propose LLM-as-Trainer (LaT), a plug-and-play training paradigm that uses a pretrained large language model as an external trainer. LaT periodically analyzes cross-task validation metrics to generate a stage-wise guidance vector. This vector is combined with the current task's constraint vector and injected into each encoder layer, providing the neural solver with additional training information during subsequent policy optimization. Experiments on 16 VRP variants show that LaT improves the solution quality of several state-of-the-art multi-task neural solvers on both trained and unseen variants, supporting the effectiveness and generality of the proposed training paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17643v1">Informal Learning Emerges in Everyday Human-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      As LLMs become increasingly capable of completing tasks for users, a central concern is that everyday AI use may become primarily cognitive offloading, eroding the opportunities through which people develop their own capabilities. We analyse large-scale human--LLM conversations to ask whether informal learning behaviors also emerge in this setting: whether users engage in exchanges in ways that preserve opportunities to learn. Across 128,569 naturalistic conversations, we translated learning-science constructs into turn-level behavioural signatures. Cognitive engagement, users' cognitive effort as reflected in the exchange, appeared in 31.9% of 491,685 user turns, whereas constructive engagement, the deepest observable form of learning-oriented engagement, appeared in 4.9%, showing that deeper sense-making was recurrent but selective. Our study further identifies factors associated with these forms of engagement. Scaffolded assistant support consistently marked richer constructive participation, with associations varying by user framing, task ecology, support form, timing and prior user state. Together, these findings show that everyday human--LLM interaction is not only answer delivery or cognitive offloading; it also contains measurable, selective and conditionally organized behavioural signatures of informal learning. They shift AI evaluation from answer-delivery efficiency toward the preservation of cognitive opportunities for users to reason, test ideas and construct understanding in the course of everyday problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11765v5">OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted as a Oral paper at ICML 2026
    </div>
    <details class="paper-abstract">
      Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited. In this work, we introduce \textbf{OMAC}, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on diverse tasks against recent approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17641v1">Verify, Repair, Repeat, or Stop? Robust Stopping for Noisy Verify-Repair Loops in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 18 pages, 10 figures, 10 tables. Under review
    </div>
    <details class="paper-abstract">
      Verify-repair loops are a standard means for large language model (LLM) agents to correct faulty plans in code generation, mathematical reasoning, and tool use. When both the verifier and the repairer are noisy, repair can damage already-correct plans, and reported acceptance keeps rising while true validity falls, so existing methods lack a principled basis for deciding when repair should stop. We propose VRR-Stop, a robust stopping framework for noisy verify-repair-repeat (VRR) loops. A four-parameter noise model separates verifier false acceptance and false rejection from the repair and damage behavior of the repairer. Belief filtering turns repeated verification votes into an estimate of committed validity, and the loop commits or repairs according to the sign of the true marginal gain, which requires only sign identifiability rather than accurate recovery of all parameters. When verifier discrimination approaches zero, calibration itself fails and estimation error can flip the stopping sign, so we pair VRR-Stop with VRR-Guard, an estimation-free fallback that replaces the incumbent candidate only under a sufficient verification margin. On a GSM8K stress setting, VRR-Stop improves final true validity by 60.6 percentage points over fixed five-round repair at an average cost of 0.72 repair rounds. Across settings, stopping reliability is governed jointly by verifier discrimination and the decision margin rather than by the absolute size of estimation error.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22455v5">SkillRouter: Skill Routing for LLM Agents at Scale</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Reusable skills let LLM agents package task-specific procedures, tool affordances, and execution guidance into modular building blocks. As skill ecosystems grow to tens of thousands of entries, exposing every skill at inference time becomes infeasible. This creates a skill-routing problem: given a user task, the system must identify relevant skills before downstream planning or execution. Existing agent stacks often rely on progressive disclosure, exposing only skill names and descriptions while hiding the full implementation body. We examine this design choice on a SkillsBench-derived benchmark with approximately 80K candidate skills, targeting the practically important setting of large skill registries with heavy overlap. Across representative dense and reranking baselines on this setting, hiding the skill body causes a 37-44 percentage point drop in routing accuracy. Stronger controls show that the missing signal is body-resident rather than a simple length artifact: body-distilled descriptions recover part of the gap, but remain 7-21 points below direct all-field routing, while a metadata-only encoder trained with the same data remains 14.0 points below its all-field counterpart. Motivated by this finding, we present Skillrouter, a compact 1.2B body-aware retrieve-and-rerank pipeline. Skillrouter achieves 74.0% Hit@1 on our benchmark -- the strongest average top-1 routing performance among the baselines we evaluate -- while using 13$\times$ fewer parameters and running 5.8$\times$ faster than the strongest base pipeline. The ranking gains further generalize to a supplementary benchmark independently constructed from three skill sources. In a complementary end-to-end study across four coding agents, routing gains transfer to improved task success, with larger gains for more capable agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17619v1">Insecure Coding Preferences in Long-Term Memory: Security Risks for LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted to the 35th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2026)
    </div>
    <details class="paper-abstract">
      LLM-based systems increasingly incorporate long-term memory to improve cross-session continuity. However, once insecure coding preferences are stored, they may silently influence security-critical decisions in subsequent generations. In this study, we conduct the first systematic empirical study on the impact of insecure coding preferences stored in long-term memory on the security of LLM-based code generation. We evaluate four LLMs (ChatGPT, Gemini, Qwen, and Grok) across five programming languages (Python, C, C++, Go, and JavaScript). Our results show that insecure memories significantly increase the risk of generating vulnerable code by 2.7-50.3 percentage points (pp). Moreover, they create a 5.4-14.0 percentage-point risk-warning gap, where warning-rate increases lag behind vulnerability-rate increases. Further analysis reveals that insecure memories are difficult to overwrite through normal interactions and can broadly influence model outputs even when prompts are phrased differently. Finally, we evaluate three mitigation strategies: security-requirement appending and memory storage reduce vulnerability rates by 19.7-33.6 pp but may degrade functional correctness by up to 15.9 pp; memory-level safety filtering achieves a 100\% detection rate on our evaluated risky memory entries and restores generation behavior to the without-memory baseline. Based on these findings, we provide actionable suggestions to improve the security of long-term memory in LLM-based code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17615v1">Re-Sonance: A Dysarthric Asynchronous Real-Time Speech Conversion System Based on a Three-Stage Cascaded ASR-LLM-TTS Architecture</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted by NCMMSC 2025
    </div>
    <details class="paper-abstract">
      Individuals with dysarthria face significant challenges in professional speaking scenarios such as conferences, presentations, and meetings, where real-time communication is crucial. While existing Augmentative and Alternative Communication (AAC) systems provide basic support, they often fail to meet the demands of professional speaking environments due to high latency and unnatural speech patterns. This paper presents Re-Sonance, a novel LLM-enhanced speech-driven AAC system designed for real-time professional speaking scenarios. By integrating Whisper ASR, Qwen LLM, and CosyVoice TTS, Re-Sonance achieves improved speech intelligibility and naturalness while maintaining real-time performance. Both subjective and objective evaluations using a Mandarin dysarthric speech dataset demonstrate that our speech reconstruction approach significantly improved intelligibility while preserving semantic coherence for speakers with mild to moderate dysarthria. Although performance remains limited for severe dysarthria cases, our findings validate the potential of LLM-based methods for enhancing speech-driven AAC systems, paving the way for more effective and accessible communication technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17575v1">A Dual-Hypothesis Reasoning Framework for LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      We propose ARBITER, a novel LLM guardrail framework that introduces two key ideas: (i) dual-hypothesis reasoning, a reasoning method for LLM guardrails that explicitly considers both safe and unsafe interpretations of a prompt before making a safety decision, and (ii) multi-component supervised fine-tuning (MC-SFT), a structured training loss for reasoning-based guardrails that decomposes LLM outputs into logical components and weights them according to their importance. Existing reasoning-based guardrails often rely on expensive procedures, such as generating reasoning traces using larger or closed-source teacher models and applying full-parameter fine-tuning. In contrast, ARBITER uses a cost-effective self-generation strategy for reasoning traces and LoRA-based parameter-efficient fine-tuning while still achieving better performance than these expensive approaches. Additionally, ARBITER provides faithful evidence-phrase explanations for unsafe decisions, enabling a more transparent and interpretable guardrail method. Experiments on three safety moderation benchmarks show that ARBITER outperforms existing reasoning-based and non-reasoning guardrail baselines, with clear gains in out-of-domain evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.02722v2">Enhancing LLMs' Clinical Reasoning with Real-World Data from a Nationwide Sepsis Registry</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted at MLHC 2026
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have demonstrated impressive reasoning capabilities across general domains, their effectiveness in real-world clinical practice remains limited. This is likely due to their insufficient exposure to real-world clinical data during training, as such data is typically not included due to privacy concerns. To address this, we propose enhancing the clinical reasoning capabilities of LLMs by leveraging real-world clinical data. We constructed reasoning-intensive questions from a nationwide sepsis registry and fine-tuned Phi-4 on these questions using reinforcement learning, resulting in C-Reason. C-Reason exhibited strong clinical reasoning capabilities on the in-domain test set, as evidenced by both quantitative metrics and expert evaluations. Furthermore, its enhanced reasoning capabilities generalized to a sepsis dataset involving different tasks and patient cohorts, an open-ended consultations on antibiotics use task, and other diseases. Future research should focus on training LLMs with large-scale, multi-disease clinical datasets to develop more powerful, general-purpose clinical reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17568v1">CoCurve: Cross-Module Co-Pruning Curvature for Training-Free Structured LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Structured pruning compresses large language models (LLMs) by removing whole computational units, such as attention heads and feed-forward (FFN) channel groups. Most training-free methods, however, rank these units independently, implicitly treating the loss from pruning a set as the sum of its individual losses. This view fails for Transformers, whose sublayers are coupled through a shared residual stream. Two individually weak units can thus be jointly indispensable, yet independent scoring is blind to such dependence and removes them together. We introduce CoCurve (Cross-Module Co-Pruning Curvature), a calibration-only, fine-tuning-free method that prunes attention and FFN units jointly. A second-order Taylor expansion of the token-level KL between the frozen model and its masked copy yields a single Fisher matrix whose diagonal is classical node saliency and whose off-diagonal entries are co-pruning curvature edges: the extra damage of removing two units together. Under a single-ablation additivity approximation this matrix reduces to a Gram product of single-unit ablation features, so the full M x M interaction is recovered from M forward passes, with no pairwise sweeps or gradients. Pruning then reduces to one budgeted quadratic program, solved in a single shot under a shared attention--FFN budget, with no labels, fine-tuning, or recovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17525v1">FailureAtlas: A Taxonomy of Failure Modes in Multi-Provider LLM Serving Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Survey Paper, 14 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Multi-provider LLM gateways reverse proxies that route, load-balance, and rate-limit requests across foundation-model APIs have become critical production infrastructure. Yet the failure modes specific to this architectural layer remain undocumented, scattered across issue trackers and post-mortems with no unifying framework. We introduce \fa{}, a two-axis taxonomy that classifies failures by their \emph{origin layer} (Network/Transport, Streaming/Protocol, State/Session, Model~Behavior, Governance/Cost) and their \emph{detectability} (Loud vs.\ Silent). We populate this taxonomy with five verified catalog entries sourced from public bug reports and first-hand stress testing, each accompanied by a mechanistic root-cause analysis. Three entries include standalone reproduction scripts. Our principal finding is that the most operationally severe failures are \emph{silent}: they return HTTP~200, pass every standard health check, and corrupt application state in ways that require semantic-level observability to detect. Two such silent failures a concurrency race condition causing history loss and a streaming index collision corrupting tool-call payloads were discovered first-hand during \cb{} evaluation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10848v3">Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating accurate responses. Therefore, in this paper, we propose \logopsyche\textit{Psyche-R1}, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through an iterative prompt-rationale optimization procedure, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data are used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of \textit{Psyche-R1} across several psychological benchmarks, where our 7B \textit{Psyche-R1} achieves comparable results to 671B \texttt{DeepSeek-R1}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04411v3">Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 work in progress. arXiv admin note: text overlap with arXiv:2405.09673 by other authors
    </div>
    <details class="paper-abstract">
      Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts. To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Considering the out-of-distribution samples, we select and merge appropriate experts based on the task uncertainty of the input data. We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03061v2">Evaluating LLMs When They Do Not Know the Answer: Statistical Evaluation of Mathematical Reasoning via Comparative Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Evaluating mathematical reasoning in LLMs is constrained by limited benchmark sizes and inherent model stochasticity, yielding high-variance accuracy estimates and unstable rankings across platforms. On difficult problems, an LLM may fail to produce a correct final answer, yet still provide reliable pairwise comparison signals indicating which of two candidate solutions is better. We leverage this observation to design a statistically efficient evaluation framework that combines standard labeled outcomes with pairwise comparison signals obtained by having models judge auxiliary reasoning chains. Treating these comparison signals as control variates, we develop a semiparametric estimator based on the efficient influence function (EIF) for the setting where auxiliary reasoning chains are observed. This yields a one-step estimator that achieves the semiparametric efficiency bound, guarantees strict variance reduction over naive sample averaging, and admits asymptotic normality for principled uncertainty quantification. Across simulations, our one-step estimator substantially improves ranking accuracy, with gains increasing as model output noise grows. Experiments on GPQA Diamond, AIME 2025, and GSM8K further demonstrate more precise performance estimation and more reliable model rankings, especially in small-sample regimes where conventional evaluation is pretty unstable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10139v2">LLMs as a Jury: Cross-Model Consensus Can Outperform Process Reward Models for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Selecting the correct answer from a pool of candidate reasoning chains is the engine of test-time scaling, yet the standard selectors each carry a cost: self-consistency inherits the errors of the single model it resamples, and trained reward models need labeled data and transfer poorly off-distribution. We study a third signal, free at inference time: cross-model consensus, the degree to which independently trained models, each solving the problem once, agree on a final answer. We treat the panel as an LLM-jury, in which the structure of agreement, not any model's score of another, is the verification signal. Across seven benchmarks it selects correct answers better than self-consistency and far better than a model scoring its own candidates: on competition math it closes the entire gap to an oracle selector, while self-scoring closes almost none. The mechanism is error decorrelation: independently trained models err differently, so their wrong answers scatter while the correct one accumulates agreement. We make this precise with a parameter-free law, derived in closed form, that predicts consensus accuracy from three measured panel statistics to a mean absolute error of $0.03$ and exposes the method's ceiling: a shared-error floor where models share a misconception, near zero on math but non-trivial on science. Against four trained verifiers spanning discriminative, outcome, and generative reward models, the free LLM-jury matches the strongest inside their math training domain and is the top selector outside it. Cross-model consensus is thus a verifier we can characterize in advance: a law that says when to trust it, and a floor that marks where it cannot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19021v2">Trustworthy AI LLM Scalability Risk Index (LSRI): A Cybersecurity Framework Assessing Agentic-AI Security & Software Model Supply Chain Safety Boosting AI-Generated Malware Defense & Explainability Mitigating Emerging Risks of Generative AI</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted for publication in Journal of Computer Information Systems (2026). DOI: 10.1080/08874417.2026.2624670
    </div>
    <details class="paper-abstract">
      As AI shifts from human-in-the-loop interfaces to autonomous multi-agent systems capable of real-time code execution and tool integration through protocols like the Model Context Protocol (MCP), traditional SAST, DAST, and legacy AI safety methods fail to detect modern agentic-AI threats. This paper introduces the LLM Scalability Risk Index (LSRI), a parametric framework and cybersecurity standard for stress-testing autonomous orchestration pipelines. LSRI measures the operational thresholds where load, compounding hallucinations, data poisoning, and adversarial prompt injections including jailbreaking and indirect prompt injection cause security boundaries to fail. Beyond RLHF and RLAIF, we present a Verifiable Root of Trust architecture using cryptographic attestation, semantic policy enforcement, and continuous runtime verification to secure the AI software supply chain. LSRI defends against malicious LoRA adapters, weight tampering, dependency typosquatting, and unsafe model artifacts from public registries such as Hugging Face and GitHub. By replacing post-hoc alignment with verifiable runtime controls, LSRI provides scalable API defense, safer agentic orchestration under heavy cloud workloads, stronger polymorphic malware detection, automated red-teaming, and improved system explainability. Aligned with NIST AI RMF, OWASP Top 10 for LLMs, and ISO 42001, LSRI establishes a deployable compliance baseline for securing generative AI ecosystems including ChatGPT, GPT-4o, Claude 3.5 Sonnet, Copilot, LLaMA, Gemini, and Bedrock. LSRI also supports capital market risk analysis by reducing exposure across technology portfolios linked to indices such as NASDAQ-100, QQQ, and semiconductor-driven supply chains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04788v2">From Sycophancy to Deception: A Unified Taxonomy for LLM Spontaneous Misalignment</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 Accepted to ICLR Agents in the Wild: Safety, Security, and Beyond Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) could produce systematically misaligned output, from hallucinated citations to strategic deception of evaluators, yet these phenomena are studied by separate communities with incompatible terminology. We propose a unified taxonomy organized along three complementary dimensions: degree of goal-directedness (behavioral to strategic deception), object of deception, and mechanism (fabrication, omission, or pragmatic distortion). Applying this taxonomy to 50 existing benchmarks reveals that every benchmark tests fabrication while pragmatic distortion, attribution, and capability self-knowledge remain critically under-covered, and strategic deception benchmarks are nascent. We offer concrete recommendations for developers and regulators, including a minimal reporting template for positioning future work within our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17437v1">Empirical Grounding Improves the Realism of LLM Agents Simulating Human Behavior During Disruptions</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents offer a generative approach to simulating human behavior under conditions that may have few or no direct historical analogues, a common challenge in disaster and infrastructure-disruption planning. However, this generative capacity creates a validity problem: individually plausible agent reasoning may fail to reproduce empirical population behavior. We evaluate whether empirical grounding improves the statistical realism of LLM-agent simulations during disruptions. Specifically, we develop an empirically grounded LLM-agent framework that embeds demographic profiles from the American Community Survey, baseline routines from the American Time Use Survey, and urban spatial context into agent initialization, memory, decision prompts, and activity execution. An independent household survey conducted during the July 2024 Philadelphia heatwave is reserved as an external validation benchmark. Compared with an ungrounded LLM-agent baseline, the grounded model improved reconstruction of normal daily routines, increasing mean correlation with empirical activity profiles from 0.528 to 0.912 and reducing mean squared error from 0.066 to 0.008. Under heatwave conditions, the grounded model better reproduced survey-derived activity profiles, increasing mean correlation from 0.349 to 0.836 and reducing mean squared error from 0.098 to 0.012. The grounded model captured 46.4% of observed heatwave response amplitude, compared with 20.6% for the ungrounded baseline. These findings show that empirical grounding can make LLM agents more statistically credible simulators of population behavior while revealing remaining gaps in modeling human adaptation during disruptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13187v4">"Not in My Backyard": LLMs Uncover Online and Offline Social Biases Against Homelessness</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Homelessness is a persistent social challenge, impacting millions worldwide. Over 876,000 people experiencing homelessness (PEH) were recorded in the U.S. in 2025. Social bias is a significant barrier to alleviating homelessness, shaping public perception and influencing policymaking. Because online textual media and offline city council discourse both reflect and influence public opinion, they provide valuable signals for identifying and tracking social biases against PEH. We release the first multi-domain PEH bias corpus with a 16-category multi-label taxonomy: a 1,698-item stratified gold-standard set annotated by partner-trained raters, plus 50,447 GPT-4.1-labeled texts, drawn from Reddit, X (formerly Twitter), news, and council meeting transcripts across ten U.S. cities (2015-2025). We benchmark six prompted LLMs on the gold-standard set and complement F1 with prevalence-gap audits. Moderate F1 coexists with large miscalibration: every model over-tags "not in my backyard" (NIMBY) (+11.5 pp) and under-detects factual claims (-30.5 pp). Error analysis on consensus false positives reveals that models treat housing vocabulary and question form as opposition proxies, producing NIMBY false positives on pro-service text. The corpus and audit protocol support municipal PEH stigma monitoring without treating teacher labels as ground truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17415v1">Transition-Aware Backend Dispatch for Edge LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Efficient large language model (LLM) inference on edge platforms is limited not only by model size, but also by shape-dependent performance differences across execution backends. Static backend assignment cannot exploit this variation, while independent per-operator selection can introduce costly device and framework switches. This paper presents a transition-aware backend dispatch approach for edge transformer inference. The approach combines current operator features with the previously selected backend to preserve beneficial shape-specific choices while avoiding unnecessary transitions. Ordered traces are collected from full-model inference runs of seven transformer models, and four common operator classes are benchmarked across PyTorch eager CPU, PyTorch eager CUDA, and ONNX Runtime CPU on an NVIDIA Jetson platform. The dispatch policies are evaluated through measurement-backed trace replay using observed operator costs and transition costs measured from actual backend switches. Supported operators are selected dynamically, while operators outside the dispatch scope retain a static assignment. Across 9,584 ordered operator instances and 278 exact shape groups, transition-aware dispatch reduces replayed latency, energy, and energy-delay product relative to the best static policy by 17.4%, 14.4%, and 28.5% on average, respectively. It also reduces switching relative to operator-local selection. Leave-one-model-out evaluation improves all three objectives for six of seven held-out models and improves energy for all seven. These results demonstrate that incorporating operator shape and backend-transition context can improve selective backend dispatch for edge transformer workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17394v1">TaskArtisan: Designing Composable Generative Widgets for LLM-Assisted Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      People increasingly use chatbots such as ChatGPT for everyday analysis tasks. While chatbots unify many analysis functions (e.g., scripts, visualizations, summaries), long conversations become hard to navigate, making it difficult to revisit prior steps or reuse successful workflows. LLMs now generate high-fidelity GUI code that enables people to create customized analysis tools beyond text. Yet, what new opportunities generative UIs bring to analysis work remain unclear. We interviewed six professionals about analysis with chatbots, analyzed publicly shared LLM-generated GUI tools, and conducted a comparison study (N=12) between a chatbot and TaskArtisan, a technology probe that enables people to create and assemble generative analysis UI widgets for sequential and fan-out composition. We find that GUI improved clarity and visual presentation but also introduced rigidity and additional prompting challenges. We summarize the trade-offs into a provisional design framework (malleability, specification, interoperability) to inform future generative UI in LLM-assisted analysis workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06197v3">Improving Answer Extraction in Context-based Question Answering Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-19
      | 💬 7 pages, IMSA2026
    </div>
    <details class="paper-abstract">
      Question answering (QA) systems have achieved notable progress with the advent of large language models (LLMs). However, they still face challenges in accurately extracting and generating precise answers from given contexts, particularly when dealing with complex or ambiguous queries. Existing approaches often struggle with contextual understanding, answer consistency, and generalization across diverse domains. In this work, we propose a question answering system based on large language models, where the input consists of a textual context and a corresponding question, and the output is a concise and accurate answer. The motivation behind this research lies in addressing the limitations of current QA systems, particularly their tendency to produce irrelevant or imprecise responses despite having access to the correct context. Our methodology involves fine-tuning a pre-trained LLM on a benchmark QA dataset to improve its contextual comprehension and answer extraction capabilities. Specifically, we utilize the Stanford Question Answering Dataset (SQuAD1.1), which provides high-quality context-question-answer triplets for supervised training and evaluation. Experimental results show that the fine-tuned Roberta-base model achieves the highest performance, attaining a ROUGE-L score of 86.84%, a BLEU score of 28.24%, and a BERTScore of 95.38%. These results indicate strong accuracy and answer relevance, demonstrating the effectiveness of the proposed approach for context-based question answering tasks. Furthermore, the findings confirm that targeted fine-tuning substantially improves the reliability and precision of QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00930v2">SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-19
      | 💬 23 pages, 8 figures. ACL 2026 Main Conference long paper (oral presentation)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong general reasoning, yet the community lacks controllable, scalable, and verifiable tools to analyze and improve these abilities. We present SATQuest, a verifier that generates diverse SAT-based reasoning tasks directly from Conjunctive Normal Form (CNF) instances and checks answers objectively with PySAT. SATQuest factorizes evaluation along three orthogonal dimensions: instance, problem type, and question format, enabling fine-grained, multi-dimensional analysis and reinforcement fine-tuning. Randomized CNF generation mitigates memorization and supports reproducible experiments. Using SATQuest, we benchmark a range of open- and closed-weight LLMs and uncover persistent gaps in logical reasoning, particularly on higher-complexity tasks and in transfer beyond familiar mathematical notation to machine or narrative formats. We further show that reinforcement fine-tuning with SATQuest rewards substantially boosts targeted performance and generalizes to larger instances, while cross-format robustness remains challenging. Collectively, SATQuest provides verifier-backed infrastructure for controlled, scalable, and reproducible empirical research on LLM logical reasoning and its training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14385v2">MamaBench: Benchmarking LLM Robustness in Maternal and Child Health Diagnosis through Counterfactual Clinical Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Large language models achieve strong scores on medical benchmarks, yet these benchmarks evaluate each question in isolation, providing no measure of whether a system can distinguish clinically similar presentations requiring different interventions. We introduce MamaBench, the first counterfactual benchmark for maternal and paediatric AI: 434 expert-authored clinical narratives in 217 pairs across 371 pathologies, evaluated via the Bias Trap Rate (BTR), the conditional probability that a model fails the counterfactual given success on the base case. We propose Evidence-Anchored RAG (EA-RAG), a three-stage retrieval method that replaces aggregate similarity with an evidence coverage objective through clinical parameter extraction, coverage auditing, and contrastive sub-queries. Across eight configurations of four frontier LLMs, base accuracy overstates robust accuracy by 16-28 percentage points in every model. EA-RAG achieves 20.3% BTR and 65.0% robust accuracy on Claude Sonnet 4.6, a 5.5 percentage point BTR reduction without degrading base accuracy. The residual 20% BTR confirms that counterfactual robustness in clinical AI remains an open challenge. Keywords: counterfactual evaluation, clinical AI, maternal healthcare, retrieval-augmented generation, diagnostic robustness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17247v1">Distilled Reinforcement Learning for LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) post-training is essential for improving reasoning, adaptation, and alignment. Existing methods mainly follow two paradigms: reinforcement learning (RL) and on-policy distillation (OPD). However, RL relies on coarse-grained outcome supervision, resulting in difficult credit assignment and limited capability to acquire new knowledge. OPD, meanwhile, unconditionally matches teacher logits through KL divergence, which creates a dilemma: similar teachers provide little new knowledge, while substantially different teachers often yield ineffective guidance, largely restricting OPD to within-family distillation. We propose Distilled Reinforcement Learning (Distilled RL), which integrates teacher supervision into the RL objective to provide fine-grained guidance, selectively transfer new knowledge and avoid unconditional imitation. Distilled RL contains three components: reverse importance sampling with clipping, negative sample reset, and sequence-level geometric normalization. Through a concise and interpretable case study, we demonstrate that Distilled RL can effectively transfer previously unavailable knowledge from a teacher model to a student model. Extensive experiments across both within-family and cross-family distillation settings show that Distilled RL substantially outperforms standard RL and OPD in terms of both pass@1 and pass@k. Our code is available at https://github.com/597358816/Distilled-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14169v2">When a Verified World Model Still Loses: Play-Adequacy vs Prediction-Accuracy in LLM-Synthesized Code World Models</a></div>
    <div class="paper-meta">
      📅 2026-07-19
      | 💬 41 pages, 4 figures. Code and reproduction log: https://github.com/JaviMaligno/code-world-models
    </div>
    <details class="paper-abstract">
      Large language models can synthesize a game's rules as executable code - a Code World Model (CWM) - which a classical planner then searches over. Such models are typically accepted when they reach high transition accuracy on sampled trajectories. We argue this is the wrong notion of adequacy for planning. We show four things. (1) An LLM-synthesized CWM can pass a sampling gate at 100% transition accuracy and be $\geq 98\%$ state-accurate on the planner's own search distribution, yet lose systematically at play, because the $<1\%$ it gets wrong is exactly the pivotal dynamics; the play cost of the omitted rule is $0.091$ (seed-clustered 95% CI $[0.065,0.117]$, $n=4800$). We call this the verified-vs-correct gap, and confirm it end-to-end through the synthesis pipeline. (2) The harm follows a quantitative law, $\mathrm{danger}=\mathrm{play\_cost}\times(1-\mathrm{rarity})^N$, whose $(1-\mathrm{rarity})^N$ gate-miss factor is proven exact and whose play cost is empirically bounded. (3) The failure is not repaired by more data: LLM synthesis behaves as rule translation, not rule inference, and did not infer the omitted rule across models (GPT-5.x) and data regimes (including DAgger and targeted examples). (4) The same mechanism recurs on the belief-inference function of imperfect-information CWMs: we prove a coverage bound (a size-$N$ gate is identifying when $N\gtrsim b^{d_{\max}}$), explaining why shallow games such as Kuhn poker show no gap, and hand-construct Beacon, a verified-but-wrong inference function that passes the gate yet loses every game. These results suggest adequacy for planning-oriented world models should be measured on the search distribution or by play directly, not by prediction accuracy on sampled transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17228v1">Literary Non-Style in LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Prior work on LLM-generated text has demonstrated quantitative and qualitative departures from text produced by humans. LLM-generated texts differ from human writing in style, resulting in a characteristic textual "feel," while the semantic range of LLMs is much restricted compared to that of humans. In this contribution, I note simple but consistent patterns in the statistical distribution of n-grams within LLM-generated text. Via qualitative analysis of these n-grams, I reveal deficiencies in LLM style. Because higher-order n-grams correlate to semantic content, I conclude that questions of style and semantics are not cleanly separable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17181v1">Talaria: Session-Aware Serverless Serving of Hundred-Billion-Parameter LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-19
      | 💬 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Serverless multi-model LLM systems multiplex popularity-skewed model catalogs over shared GPU pools, yet typically schedule each request independently. Tool-using agents break this abstraction: a session repeatedly calls an LLM across short tool gaps, carries a long reusable KV prefix, and is judged by session completion time (SCT). Load-only routing can separate a continuation from both its model and KV state, while round-based model multiplexing can delay even a correctly placed continuation until the target model's next slot. Both failures are especially costly for hundred-billion-parameter models: their weights constrain residency, while long-context KV is expensive to reconstruct or move. We present Talaria, a session-aware serverless multi-model serving system that makes session continuity a joint placement-and-admission decision. Its router ranks placements by model residency, KV locality, and instance pressure, while soft reservations account for likely returns in the last serving instance's admission budget. Session-prefill (SP) admits budget-eligible continuations before the active model slot closes. An instance-local substrate keeps HBM addresses stable, preserves host-restorable KV, and stages weights across model switches. On a single TP=8 server, we replay 30 SWE-Bench model-sessions (960 calls) over three models, each with more than 100B total parameters. Against an otherwise identical round scheduler with SP, host-KV restoration, and D2D staging disabled, Talaria cuts p50 SCT from 1000 s to 189 s and p95 from 2296 s to 867 s, speedups of 5.3x and 2.6x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02792v2">From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have already been widely adopted for automated algorithm design, demonstrating strong abilities in generating and evolving algorithms across various fields. Existing work has largely focused on examining their effectiveness in solving specific problems, with search strategies primarily guided by adaptive prompt designs. In this paper, through investigating the token-wise attribution of the prompts to LLM-generated algorithmic codes, we show that providing high-quality algorithmic code examples can substantially improve the performance of the LLM-driven optimization. Building upon this insight, we propose leveraging prior benchmark algorithms to guide LLM-driven optimization and demonstrate superior performance on two black-box optimization benchmarks: the pseudo-Boolean optimization suite (pbo) and the black-box optimization suite (bbob). Our findings highlight the value of integrating benchmarking studies to enhance both efficiency and robustness of the LLM-driven black-box optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17175v1">LMEdge: QoS-Aware LLM Inference Orchestration on Edge Clusters</a></div>
    <div class="paper-meta">
      📅 2026-07-19
      | 💬 12 pages, 5 figures, 2 tables, EUROPAR conference paper
    </div>
    <details class="paper-abstract">
      Large language model (LLM) services increasingly operate on edge infrastructure, enabling low-latency and privacy-preserving AI services. However, efficiently serving LLM requests across heterogeneous and resource-constrained edge devices require orchestration mechanisms that jointly determine model configuration (family, size, and quantization level) and execution placement while satisfying user- and system-level quality of service (QoS) requirements. This paper introduces LMEdge, a QoS-aware orchestration service that dynamically makes these decisions across heterogeneous edge devices. We formulate the problem as a binary integer linear programming (BILP) optimization that minimizes response time under accuracy, network, and resource constraints. To enable scalable online scheduling, we employ five lightweight machine learning (ML) models to predict query-specific latency, accuracy, resource usage, and response size for each model-size-quantization-device combination, and design a lightweight heuristic that approximates the BILP solution. We collect a comprehensive benchmarking dataset of over 59000 rows to train models and support reproducibility. Evaluation on a Kubernetes-based edge testbed with 57 instances and diverse query categories shows that LMEdge reduces latency, preserves accuracy, improves resource utilization, and increases serving ratio compared to two baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06503v2">Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade</a></div>
    <div class="paper-meta">
      📅 2026-07-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents often waste inference compute by continuing multi-step trajectories that are already doomed to fail. We study early failure prediction and inference-time early stopping for LLM agents using hidden-state probes. Lightweight linear probes on internal activations predict eventual task failure from the first interaction round, substantially earlier than agent-monitoring methods based only on observable behavior. We turn this signal into a recall-controlled abort cascade for reducing LLM agent inference costs. The cascade applies a distribution-free calibrated failure detector at each early interaction round and jointly optimizes per-round recall budgets. This design ensures that eventually successful episodes survive all early-stopping gates at a user-specified global recall rate. After selection, the cascade is frozen and certified on independent data, providing an exact post-selection recall guarantee. We evaluate the method on TextCraft and WebShop with Qwen-2.5-7B, Llama-3.2-3B, and Qwen3-1.7B. The proposed LLM agent early-stopping cascade outperforms the best single-gate baseline in every model-environment pair, saving 1.5-8.8 times more compute at a 90% recall target. Achieved recall remains within one standard deviation of its target in all 24 configurations. The strongest settings reduce generated tokens by 60.2% on TextCraft and 54.9% on WebShop at 90% recall, while retaining savings of 45.0% and 41.5% at 95% recall. Behavior-only monitoring is consistently weaker, and adding behavioral features to hidden-state probes provides no further gain. We also characterize the sample complexity required to certify high-recall early-stopping policies. The code will be released soon.
    </details>
</div>
