# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02627v2">Improved Evidence Extraction and Metrics for Document Inconsistency Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. However, research on LLM-based approaches to document inconsistency detection is relatively limited. We address this gap by investigating evidence extraction capabilties of LLMs for document inconsistency detection. To this end, we introduce new comprehensive evidence-extraction metrics and a redact-and-retry framework with constrained filtering that substantially improves evidence extraction performance over other prompting methods. We support our approach with strong experimental results and release a new semi-synthetic dataset for evaluating evidence extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.16216v2">Reinforcement Learning for LLM Post-Training: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Through pretraining and supervised fine-tuning (SFT), large language models (LLMs) acquire strong instruction-following capabilities, yet they can still produce harmful or misaligned outputs. A growing body of reinforcement learning (RL)-based post-training methods has been proposed to address this, including Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) approaches built on Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), and others. Despite rapid progress, no existing work offers a systematic, technically detailed comparison of these methods under a single analytical lens. Our survey aims to fill this gap. We make three key contributions: (1) a self-contained RL and LLM post-training foundations treatment covering all necessary concepts alongside their key applications; (2) a unified policy gradient framework unifying PPO and GRPO-based RLHF, RLVR, and offline DPO-based RLHF, decomposing methods along the axes of prompt sampling, response sampling, and gradient coefficient, with an extended treatment of on-policy RLHF and iterative DPO methods as well as the richer design space of offline DPO-based methods; and (3) standardized notation across all reviewed papers enabling direct technical comparison. Our goal is to serve as a comprehensive, technically grounded reference for researchers and practitioners working on LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07387v1">Self-Calibrating LLM-Based Analog Circuit Sizing with Interpretable Design Equations</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 11 pages, 5 figures, 4 tables. Submitted to IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI)
    </div>
    <details class="paper-abstract">
      We present a self-calibrating framework for analog circuit sizing in which a large language model (LLM) derives topology-specific analytical design equations directly from a raw circuit netlist. Unlike existing AI-driven sizing methods where the model proposes parameter adjustments or reduces a search space, the LLM produces a complete Python sizing function tracing each device dimension to a specific performance constraint. A deterministic calibration loop extracts process-dependent parameters from a single transistor-level simulation, while a prediction-error feedback mechanism compensates for analytical inaccuracies. We validate the framework on six operational transconductance amplifier (OTA) topologies spanning three families at two process nodes (180 nm and 40 nm CMOS). All 12 topology-node combinations achieve all specifications, converging in 2-9 simulations for 11 of 12 cases, with one outlier requiring 16 simulations due to an extremely narrow feasible region. Despite large initial prediction errors, convergence depends on the measurement-feedback architecture, not prediction accuracy. This one-shot calibration automatically captures process-dependent variations, enabling cross-node portability without modification, retraining, or per-process characterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06666v1">A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by TOIS
    </div>
    <details class="paper-abstract">
      Explainable fake news detection aims to assess the veracity of news claims while providing human-friendly explanations. Existing methods incorporating investigative journalism are often inefficient and struggle with breaking news. Recent advances in large language models (LLMs) enable leveraging externally retrieved reports as evidence for detection and explanation generation, but unverified reports may introduce inaccuracies. Moreover, effective explainable fake news detection should provide a comprehensible explanation for all aspects of a claim to assist the public in verifying its accuracy. To address these challenges, we propose a graph-enhanced defense framework (G-Defense) that provides fine-grained explanations based solely on unverified reports. Specifically, we construct a claim-centered graph by decomposing the news claim into several sub-claims and modeling their dependency relationships. For each sub-claim, we use the retrieval-augmented generation (RAG) technique to retrieve salient evidence and generate competing explanations. We then introduce a defense-like inference module based on the graph to assess the overall veracity. Finally, we prompt an LLM to generate an intuitive explanation graph. Experimental results demonstrate that G-Defense achieves state-of-the-art performance in both veracity detection and the quality of its explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06664v1">Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Modern LLM service providers increasingly rely on autoscaling and parallelism reconfiguration to respond to rapidly changing workloads, but cold-start latency remains a major bottleneck. While recent systems have reduced model weight loading to seconds, CUDA graph capture still takes tens of seconds to minutes and often dominates startup. Unfortunately, CUDA graphs cannot be naively serialized: beyond graph topology, they are tightly coupled to execution context, including device addresses embedded in kernel arguments and kernel code lazily loaded during warmup. Existing approaches either rely on brittle kernel-specific patching or heavyweight process-level checkpoint/restore that are inflexible to dynamic parallelism switching. We present Foundry, a template-based CUDA graph context materialization system that persists both graph topology and execution context during an offline processing stage, and reconstructs executable graphs online with negligible overhead. Foundry enforces deterministic memory layouts, automatically extracts and reloads kernel binaries required by captured graphs, and reduces online reconstruction costs through topology-based templating. For distributed serving, Foundry further enables a single-GPU offline capture to generate templates for multi-GPU deployments by patching only rank-dependent communication state. Across dense and MoE models up to 235B parameters, Foundry reduces cold-start latency by up to 99%, cutting the initialization time of Qwen3-235B-A22B from 10 minutes to 3.9 seconds while preserving the throughput gains of CUDA graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06663v1">Restoring Heterogeneity in LLM-based Social Simulation: An Audience Segmentation Approach</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to simulate social attitudes and behaviors, offering scalable "silicon samples" that can approximate human data. However, current simulation practice often collapses diversity into an "average persona," masking subgroup variation that is central to social reality. This study introduces audience segmentation as a systematic approach for restoring heterogeneity in LLM-based social simulation. Using U.S. climate-opinion survey data, we compare six segmentation configurations across two open-weight LLMs (Llama 3.1-70B and Mixtral 8x22B), varying segmentation identifier granularity, parsimony, and selection logic (theory-driven, data-driven, and instrument-based). We evaluate simulation performance with a three-dimensional evaluation framework covering distributional, structural, and predictive fidelity. Results show that increasing identifier granularity does not produce consistent improvement: moderate enrichment can improve performance, but further expansion does not reliably help and can worsen structural and predictive fidelity. Across parsimony comparisons, compact configurations often match or outperform more comprehensive alternatives, especially in structural and predictive fidelity, while distributional fidelity remains metric dependent. Identifier selection logic determines which fidelity dimension benefits most: instrument-based selection best preserves distributional shape, whereas data-driven selection best recovers between-group structure and identifier-outcome associations. Overall, no single configuration dominates all dimensions, and performance gains in one dimension can coincide with losses in another. These findings position audience segmentation as a core methodological approach for valid LLM-based social simulation and highlight the need for heterogeneity-aware evaluation and variance-preserving modeling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02670v2">Break Me If You Can: Self-Jailbreaking of Aligned LLMs via Lexical Insertion Prompting</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      We introduce \emph{self-jailbreaking}, a threat model in which an aligned LLM guides its own compromise. Unlike most jailbreak techniques, which often rely on handcrafted prompts or separate attacker models, self-jailbreaking requires no external red-team LLM: the target model's own internal knowledge suffices. We operationalize this via \textbf{Self-Jailbreaking via Lexical Insertion Prompting (\textsc{SLIP})}, a black-box algorithm that casts jailbreaking as breadth-first tree search over multi-turn dialogues, incrementally inserting missing content words from the attack goal into benign prompts using the target model as its own guide. Evaluations on AdvBench and HarmBench show \textsc{SLIP} achieves 90--100\% Attack Success Rate (ASR) (avg.\ 94.7\%) across most of the eleven tested models (including GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Pro, and DeepSeek-V3), with only ${\sim}7.9$ LLM calls on average, 3--6$\times$ fewer than prior methods. We evaluate existing defenses, show that regex-based approaches are evaded by prompt paraphrasing, and propose the Semantic Drift Monitor (SDM) defense that tracks \textsc{SLIP}'s embedding-space trajectory, achieving 76\% detection at 5\% FPR. However, SDM remains insufficient against adaptive attack strategies, underscoring the need for more advanced defense mechanisms tailored to the self-jailbreaking threat surface. We release our code for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06636v1">SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Process supervision has emerged as a promising approach for enhancing LLM reasoning, yet existing methods fail to distinguish meaningful progress from mere verbosity, leading to limited reasoning capabilities and unresolved token inefficiency. To address this, we propose Stage-aware Hierarchical Advantage via Potential Estimation (SHAPE), a framework that formalizes reasoning as a trajectory through a state space of empirical solvability. SHAPE introduces a hierarchical credit assignment mechanism: at the segment level, it employs a stage-aware advantage function to prioritize efficient breakthroughs in low-potential states; at the token level, it utilizes entropy-driven redistribution to sharpen execution signals. Extensive experiments in math reasoning across three base models and five benchmarks demonstrate that SHAPE achieves an average accuracy gain of 3% with 30% reduced token consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19225v3">RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located frameworks fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout. In this paper, we present RLBoost, a framework for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06607v1">CoverAssert: Iterative LLM Assertion Generation Driven by Functional Coverage via Syntax-Semantic Representations</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 3 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLMs can generate SystemVerilog assertions (SVAs) from natural language specs, but single-pass outputs often lack functional coverage due to limited IC design understanding. We propose CoverAssert, an iterative framework that clusters semantic and AST-based structural features of assertions, maps them to specifications, and uses functional coverage feedback to guide LLMs in prioritizing uncovered points. Experiments on four open-source designs show that integrating CoverAssert with AssertLLM and Spec2Assertion improves average improvements of 9.57 % in branch coverage, 9.64 % in statement coverage, and 15.69 % in toggle coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06603v1">Scientific Knowledge-driven Decoding Constraints Improving the Reliability of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong knowledge reserves and task-solving capabilities, but still face the challenge of severe hallucination, hindering their practical application. Though scientific theories and rules can efficiently direct the behaviors of human manipulators, LLMs still do not utilize these highly-condensed knowledge sufficiently through training or prompting. To address this issue, we propose \textbf{SciDC}, an LLM generation method that integrate subject-specific knowledge with strong constraints. By adopting strong LLMs to automatically convert flexible knowledge into multi-layered, standardized rules, we build an extensible framework to effectively constrain the model generation on domain tasks. Experiments on scientific tasks including industrial formulation design, clinical tumor diagnosis and retrosynthesis planning, consistently demonstrate the effectiveness of our method, achieving a 12\% accuracy improvement on average compared with vanilla generation. We further discuss the potential of LLMs in automatically inductively summarizing highly-condensed knowledge, looking ahead to practical solutions for accelerating the overall scientific research process. All the code of this paper can be obtained (https://github.com/Maotian-Ma/SciDC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21463v2">Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Existing speech editing detection (SED) datasets are predominantly constructed using manual splicing or limited editing operations, resulting in restricted diversity and poor coverage of realistic editing scenarios. Meanwhile, current SED methods rely heavily on frame-level supervision to detect observable acoustic anomalies, which fundamentally limits their ability to handle deletion-type edits, where the manipulated content is entirely absent from the signal. To address these challenges, we present a unified framework that bridges speech editing detection and content localization through a generative formulation based on Audio Large Language Models (Audio LLMs). We first introduce AiEdit, a large-scale bilingual dataset (approximately 140 hours) that covers addition, deletion, and modification operations using state-of-the-art end-to-end speech editing systems, providing a more realistic benchmark for modern threats. Building upon this, we reformulate SED as a structured text generation task, enabling joint reasoning over edit type identification, and content localization. To enhance the grounding of generative models in acoustic evidence, we propose a prior-enhanced prompting strategy that injects word-level probabilistic cues derived from a frame-level detector. Furthermore, we introduce an acoustic consistency-aware loss that explicitly enforces the separation between normal and anomalous acoustic representations in the latent space. Experimental results demonstrate that the proposed approach consistently outperforms existing methods across both detection and localization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06571v1">LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 9 pages, 6 figures. Accepted at International Conference on Intelligent Digitization of Systems and Services (IDSS 2026)
    </div>
    <details class="paper-abstract">
      Missing-person and child-safety investigations rely on heterogeneous case documents, including structured forms, bulletin-style posters, and narrative web profiles. Variations in layout, terminology, and data quality impede rapid triage, large-scale analysis, and search-planning workflows. This paper introduces the Guardian Parser Pack, an AI-driven parsing and normalization pipeline that transforms multi-source investigative documents into a unified, schema-compliant representation suitable for operational review and downstream spatial modeling. The proposed system integrates (i) multi-engine PDF text extraction with Optical Character Recognition (OCR) fallback, (ii) rule-based source identification with source-specific parsers, (iii) schema-first harmonization and validation, and (iv) an optional Large Language Model (LLM)-assisted extraction pathway incorporating validator-guided repair and shared geocoding services. We present the system architecture, key implementation decisions, and output design, and evaluate performance using both gold-aligned extraction metrics and corpus-level operational indicators. On a manually aligned subset of 75 cases, the LLM-assisted pathway achieved substantially higher extraction quality than the deterministic comparator (F1 = 0.8664 vs. 0.2578), while across 517 parsed records per pathway it also improved aggregate key-field completeness (96.97\% vs. 93.23\%). The deterministic pathway remained much faster (mean runtime 0.03 s/record vs. 3.95 s/record for the LLM pathway). In the evaluated run, all LLM outputs passed initial schema validation, so validator-guided repair functioned as a built-in safeguard rather than a contributor to the observed gains. These results support controlled use of probabilistic AI within a schema-first, auditable pipeline for high-stakes investigative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06552v1">To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Misinformation is on the rise, and the strong writing capabilities of LLMs lower the barrier for malicious actors to produce and disseminate false information. We study how LLMs behave when prompted to spread misinformation across languages and target countries, and introduce GlobalLies, a multilingual parallel dataset of 440 misinformation generation prompt templates and 6,867 entities, spanning 8 languages and 195 countries. Using both human annotations and large-scale LLM-as-a-judge evaluations across hundreds of thousands of generations from state-of-the-art models, we show that misinformation generation varies systematically based on the country being discussed. Propagation of lies by LLMs is substantially higher in many lower-resource languages and for countries with a lower Human Development Index (HDI). We find that existing mitigation strategies provide uneven protection: input safety classifiers exhibit cross-lingual gaps, and retrieval-augmented fact-checking remains inconsistent across regions due to unequal information availability. We release GlobalLies for research purposes, aiming to support the development of mitigation strategies to reduce the spread of global misinformation: https://github.com/zohaib-khan5040/globallies
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06543v1">The Illusion of Stochasticity in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      In this work, we demonstrate that reliable stochastic sampling is a fundamental yet unfulfilled requirement for Large Language Models (LLMs) operating as agents. Agentic systems are frequently required to sample from distributions, often inferred from observed data, a process which needs to be emulated by the LLM. This leads to a distinct failure point: while standard RL agents rely on external sampling mechanisms, LLMs fail to map their internal probability estimates to their stochastic outputs. Through rigorous empirical analysis across multiple model families, model sizes, prompting styles, and distributions, we demonstrate the extent of this failure. Crucially, we show that while powerful frontier models can convert provided random seeds to target distributions, their ability to sample directly from specific distributions is fundamentally flawed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07659v1">Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL 2026 (Findings), reviewer score: 3.5,3.5,4
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold significant promise for healthcare, yet their reliability in high-stakes clinical settings is often compromised by hallucinations and a lack of granular medical context. While Retrieval Augmented Generation (RAG) can mitigate these issues, standard supervised pipelines require computationally intensive searches over massive external knowledge bases, leading to high latency that is impractical for time-sensitive care. To address this, we introduce Keys to Knowledge (K2K), a novel framework that replaces external retrieval with internal, key-based knowledge access. By encoding essential clinical information directly into the model's parameter space, K2K enables rapid retrieval from internal key-value memory without inference-time overhead. We further enhance retrieval quality through activation-guided probe construction and cross-attention reranking. Experimental results demonstrate that K2K achieves state-of-the-art performance across four benchmark healthcare outcome prediction datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07652v1">Bridging Natural Language and Interactive What-If Interfaces via LLM-Generated Declarative Specification</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 17 pages 17 figures
    </div>
    <details class="paper-abstract">
      What-if analysis (WIA) is an iterative, multi-step process where users explore and compare hypothetical scenarios by adjusting parameters, applying constraints, and scoping data through interactive interfaces. Current tools fall short of supporting effective interactive WIA: spreadsheet and BI tools require time-consuming and laborious setup, while LLM-based chatbot interfaces are semantically fragile, frequently misinterpret intent, and produce inconsistent results as conversations progress. To address these limitations, we present a two-stage workflow that translates natural language (NL) WIA questions into interactive visual interfaces via an intermediate representation, powered by the Praxa Specification Language (PSL): first, LLMs generate PSL specifications from NL questions capturing analytical intent and logic, enabling validation and repair of erroneous specifications; and second, the specifications are compiled into interactive visual interfaces with parameter controls and linked visualizations. We benchmark this workflow with 405 WIA questions spanning 11 WIA types, 5 datasets, and 3 state-of-the-art LLMs. The results show that across models, half of specifications (52.42%) are generated correctly without intervention. We perform an analysis of the failure cases and derive an error taxonomy spanning non-functional errors (specifications fail to compile) and functional errors (specifications compile but misrepresent intent). Based on the taxonomy, we apply targeted repairs on the failure cases using few-shot prompts and improve the success rate to 80.42%. Finally, we show how undetected functional errors propagate through compilation into plausible but misleading interfaces, demonstrating that the intermediate specification is critical for reliably bridging NL and interactive WIA interface in LLM-powered WIA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07994v4">DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted (B) to TACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly operate over long-form dialogues with frequent topic shifts. While recent LLMs support extended context windows, efficient management of dialogue history in practice is needed due to inference cost and latency constraints. We present DyCP, a lightweight context management method implemented outside the LLM that dynamically identifies and retrieves relevant dialogue segments conditioned on the current turn, without offline memory construction. DyCP manages dialogue context while preserving the sequential nature of dialogue without predefined topic boundaries, enabling adaptive and efficient context selection. Across three long-form dialogue benchmarks-LoCoMo, MT-Bench+, and SCM4LLMs-and multiple LLM backends, DyCP achieves competitive answer quality in downstream generation, with more selective context usage and improved inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21354v2">The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Vision Paper
    </div>
    <details class="paper-abstract">
      Over the past year, the vLLM Semantic Router project has released a series of work spanning: (1) core routing mechanisms -- signal-driven routing, context-length pool routing, router performance engineering, policy conflict detection, low-latency embedding models, category-aware semantic caching, user-feedback-driven routing adaptation, hallucination detection, and hierarchical content-safety classification for privacy and jailbreak protection; (2) fleet optimization -- fleet provisioning and energy-efficiency analysis; (3) agentic and multimodal routing -- multimodal agent routing, tool selection, CUA security, and multi-turn context memory and safety; (4) governance and standards -- inference routing protocols and multi-provider API extensions. Each paper tackled a specific problem in LLM inference, but the problems are not independent; for example, fleet provisioning depends on the routing policy, which depends on the workload mix, shifting as organizations adopt agentic and multimodal workloads. This paper distills those results into the Workload-Router-Pool (WRP) architecture, a three-dimensional framework for LLM inference optimization. Workload characterizes what the fleet serves (chat vs. agent, single-turn vs. multi-turn, warm vs. cold, prefill-heavy vs. decode-heavy). Router determines how each request is dispatched (static semantic rules, online bandit adaptation, RL-based model selection, quality-aware cascading). Pool defines where inference runs (homogeneous vs. heterogeneous GPU, disaggregated prefill/decode, KV-cache topology). We map our prior work onto a 3x3 WRP interaction matrix, identify which cells we have covered and which remain open, and propose twenty-one concrete research directions at the intersections, each grounded in our prior measurements, tiered by maturity from engineering-ready to open research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07624v1">Program Analysis Guided LLM Agent for Proof-of-Concept Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Software developers frequently receive vulnerability reports that require them to reproduce the vulnerability in a reliable manner by generating a proof-of-concept (PoC) input that triggers it. Given the source code for a software project and a specific code location for a potential vulnerability, automatically generating a PoC for the given vulnerability has been a challenging research problem. Symbolic execution and fuzzing techniques require expert guidance and manual steps and face scalability challenges for PoC generation. Although recent advances in LLMs have increased the level of automation and scalability, the success rate of PoC generation with LLMs remains quite low. In this paper, we present a novel approach called Program Analysis Guided proof of concept generation agENT (PAGENT) that is scalable and significantly improves the success rate of automated PoC generation compared to prior results. PAGENT integrates lightweight and rule-based static analysis phases for providing static analysis guidance and sanitizer-based profiling and coverage information for providing dynamic analysis guidance with a PoC generation agent. Our experiments demonstrate that the resulting hybrid approach significantly outperforms the prior top-performing agentic approach by 132% for the PoC generation task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07609v1">Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving stacks keep the host CPU on the critical path for orchestration and token-level control. This makes LLM performance sensitive to CPU interference, undermining application colocation and forcing operators to reserve CPU headroom, leaving substantial capacity unutilized. We introduce Blink, an end-to-end serving architecture that removes the host CPU from the steady-state inference path by redistributing responsibilities across a SmartNIC and a GPU. Blink offloads request handling to the SmartNIC, which delivers inputs directly into GPU memory via RDMA, and replaces host-driven scheduling with a persistent GPU kernel that performs batching, scheduling, and KV-cache management without CPU involvement. Evaluated against TensorRT-LLM, vLLM, and SGLang, Blink outperforms all baselines even in isolation, reducing pre-saturation P99 TTFT by up to 8.47$\times$ and P99 TPOT by up to 3.40$\times$, improving decode throughput by up to 2.1$\times$, and reducing energy per token by up to 48.6$\%$. Under CPU interference, Blink maintains stable performance, while existing systems degrade by up to two orders of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07584v1">From Papers to Property Tables: A Priority-Based LLM Workflow for Materials Data Extraction</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Scientific data are widely dispersed across research articles and are often reported inconsistently across text, tables, and figures, making manual data extraction and aggregation slow and error-prone. We present a prompt-driven, hierarchical workflow that uses a large language model (LLM) to automatically extract and reconstruct structured, shot-level shock-physics experimental records by integrating information distributed across text, tables, figures, and physics-based derivations from full-text published research articles, using alloy spall strength as a representative case study. The pipeline targeted 37 experimentally relevant fields per shot and applied a three-level priority strategy: (T1) direct extraction from text/tables, (T2) physics-based derivation using verified governing relations, and (T3) digitization from figures when necessary. Extracted values were normalized to canonical units, tagged by priority for traceability, and validated with physics-based consistency and plausibility checks. Evaluated on a benchmark of 30 published research articles comprising 11,967 evaluated data points, the workflow achieved high overall accuracy, with priority-wise accuracies of 94.93% (T1), 92.04% (T2), and 83.49% (T3), and an overall weighted accuracy of 94.69%. Cross-model testing further indicated strong agreement for text/table and equation-derived fields, with lower agreement for figure-based extraction. Implementation through an API interface demonstrated the scalability of the approach, achieving consistent extraction performance and, in a subset of test cases, matching or exceeding chat-based accuracy. This workflow demonstrates a practical approach for converting unstructured technical literature into traceable, analysis-ready datasets without task-specific fine-tuning, enabling scalable database construction in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09942v3">Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 8 pages, 15 figures
    </div>
    <details class="paper-abstract">
      This paper investigates the optimal allocation of large language model (LLM) inference workloads across heterogeneous edge data centers over time. Each data center features on-site renewable generation and faces dynamic electricity prices and spatiotemporal variability in renewable availability. We propose Green-LLM, a lexicographic multi-objective optimization framework that addresses this challenge without requiring manual weight tuning. The proposed model incorporates real-world constraints, including token-dependent processing delay and energy consumption, heterogeneous hardware capabilities, dynamic renewable generation, and spatiotemporal variations in electricity prices and carbon intensity. Unlike existing approaches that optimize individual environmental metrics in isolation, Green-LLM jointly minimizes operational cost, carbon emissions, and delay penalty while enforcing water consumption constraints to ensure both sustainability and quality-of-service requirements. Numerical results demonstrate that Green-LLM achieves significant reductions in carbon emissions and water consumption while maintaining operational costs within 3% of the minimum and ensuring sub-2-second response latency. These findings show that sustainable LLM inference can be achieved without sacrificing service quality or economic efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13040v2">Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07569v1">Learning is Forgetting: LLM Training As Lossy Compression</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 12 page core paper, 16 page Appendix - A shorter version with fewer visuals appears at ICLR 2026
    </div>
    <details class="paper-abstract">
      Despite the increasing prevalence of large language models (LLMs), we still have a limited understanding of how their representational spaces are structured. This limits our ability to interpret how and what they learn or relate them to learning in humans. We argue LLMs are best seen as an instance of lossy compression, where over training they learn by retaining only information in their training data relevant to their objective(s). We show pre-training results in models that are optimally compressed for next-sequence prediction, approaching the Information Bottleneck bound on compression. Across an array of open weights models, each compresses differently, likely due to differences in the data and training recipes used. However even across different families of LLMs the optimality of a model's compression, and the information present in it, can predict downstream performance on across a wide array of benchmarks, letting us directly link representational structure to actionable insights about model performance. In the general case the work presented here offers a unified Information-Theoretic framing for how these models learn that is deployable at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07565v1">Have LLM-associated terms increased in article full texts in all fields?</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) like ChatGPT and DeepSeek for translation and language polishing is a welcome development, reducing the longstanding publishing barrier to non-English speakers. Assessing the uptake of this facility is useful to give insights into changing nature of scientific writing. Although the prevalence of LLM-associated terms has been tracked across science in abstracts and for full text biomedical research, their science-wide prevalence in full texts is unknown. In response, this article investigates an expanded set of 80 potentially LLM-associated terms during 2021-2025 in a science-wide full text collection from the publisher MDPI (1.25 million articles), partly focusing on the 73 journals that published at least 500 articles in 2021. The results demonstrate the increasing prevalence of LLM-associated terms science-wide in full texts to 2024, with some terms declining from 2024 to 2025 and others continuing to increase. LLMs seem to avoid some terms (e.g., thus, moreover) and a few terms have stronger associations with abstracts than full texts (e.g., enhanced) or the opposite (e.g., leveraged). The term family "underscore" had the biggest increase: up to 29-fold. There are substantial differences between journals in the apparent use of LLMs for writing, from lower uptake in the life sciences to higher uptake in social sciences, electronic engineering and environmental science. Fields in which there is currently low uptake may need improved or specialist support, such as for reliably translating complex formulae, before the full benefits of automatic translation can be realised.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07562v1">Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted to the Findings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Unsupervised methods are widely used to induce latent semantic structure from large text collections, yet their outputs often contain incoherent, redundant, or poorly grounded clusters that are difficult to validate without labeled data. We propose a reasoning-based refinement framework that leverages large language models (LLMs) not as embedding generators, but as semantic judges that validate and restructure the outputs of arbitrary unsupervised clustering algorithms.Our framework introduces three reasoning stages: (i) coherence verification, where LLMs assess whether cluster summaries are supported by their member texts; (ii) redundancy adjudication, where candidate clusters are merged or rejected based on semantic overlap; and (iii) label grounding, where clusters are assigned interpretable labels in a fully unsupervised manner. This design decouples representation learning from structural validation and mitigates common failure modes of embedding-only approaches. We evaluate the framework on real-world social media corpora from two platforms with distinct interaction models, demonstrating consistent improvements in cluster coherence and human-aligned labeling quality over classical topic models and recent representation-based baselines. Human evaluation shows strong agreement with LLM-generated labels, despite the absence of gold-standard annotations. We further conduct robustness analyses under matched temporal and volume conditions to assess cross-platform stability. Beyond empirical gains, our results suggest that LLM-based reasoning can serve as a general mechanism for validating and refining unsupervised semantic structure, enabling more reliable and interpretable analyses of large text collections without supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07549v1">EMSDialog: Synthetic Multi-person Emergency Medical Service Dialogue Generation from Electronic Patient Care Reports via Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted by ACL Findings 2026
    </div>
    <details class="paper-abstract">
      Conversational diagnosis prediction requires models to track evolving evidence in streaming clinical conversations and decide when to commit to a diagnosis. Existing medical dialogue corpora are largely dyadic or lack the multi-party workflow and annotations needed for this setting. We introduce an ePCR-grounded, topic-flow-based multi-agent generation pipeline that iteratively plans, generates, and self-refines dialogues with rule-based factual and topic flow checks. The pipeline yields EMSDialog, a dataset of 4,414 synthetic multi-speaker EMS conversations based on a real-world ePCR dataset, annotated with 43 diagnoses, speaker roles, and turn-level topics. Human and LLM evaluations confirm high quality and realism of EMSDialog using both utterance- and conversation-level metrics. Results show that EMSDialog-augmented training improves accuracy, timeliness, and stability of EMS conversational diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07536v1">TRUSTDESC: Preventing Tool Poisoning in LLM Applications via Trusted Description Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on external tools to perform time-sensitive tasks and real-world actions. While tool integration expands LLM capabilities, it also introduces a new prompt-injection attack surface: tool poisoning attacks (TPAs). Attackers manipulate tool descriptions by embedding malicious instructions (explicit TPAs) or misleading claims (implicit TPAs) to influence model behavior and tool selection. Existing defenses mainly detect anomalous instructions and remain ineffective against implicit TPAs. In this paper, we present TRUSTDESC, the first framework for preventing tool poisoning by automatically generating trusted tool descriptions from implementations. TRUSTDESC derives implementation-faithful descriptions through a three-stage pipeline. SliceMin performs reachability-aware static analysis and LLM-guided debloating to extract minimal tool-relevant code slices. DescGen synthesizes descriptions from these slices while mitigating misleading or adversarial code artifacts. DynVer refines descriptions through dynamic verification by executing synthesized tasks and validating behavioral claims. We evaluate TRUSTDESC on 52 real-world tools across multiple tool ecosystems. Results show that TRUSTDESC produces accurate tool descriptions that improve task completion rates while mitigating implicit TPAs at their root, with minimal time and monetary overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07530v1">The Shrinking Lifespan of LLMs in Science</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Scaling laws describe how language model capabilities grow with compute and data, but say nothing about how long a model matters once released. We provide the first large-scale empirical account of how scientists adopt and abandon language models over time. We track 62 LLMs across over 108k citing papers (2018-2025), each with at least three years of post-release data, and classify every citation as active adoption or background reference to construct per-model adoption trajectories that raw citation counts cannot resolve. We find three regularities. First, scientific adoption follows an inverted-U trajectory: usage rises after release, peaks, and declines as newer models appear, a pattern we term the \textit{scientific adoption curve}. Second, this curve is compressing: each additional release year is associated with a 27\% reduction in time-to-peak adoption ($p < 0.001$), robust to minimum-age thresholds and controls for model size. Third, release timing dominates model-level attributes as a predictor of lifecycle dynamics. Release year explains both time-to-peak and scientific lifespan more strongly than architecture, openness, or scale, though model size and access modality retain modest predictive power for total adoption volume. Together, these findings complement scaling laws with adoption-side regularities and suggest that the forces driving rapid capability progress may be the same forces compressing scientific relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20114v4">Current LLMs still cannot 'talk much' about grammar modules: Evidence from syntax</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      We aim to examine the extent to which Large Language Models (LLMs) can 'talk much' about grammar modules, providing evidence from syntax core properties translated by ChatGPT into Arabic. We collected 44 terms from generative syntax previous works, including books and journal articles, as well as from our experience in the field. These terms were translated by humans, and then by ChatGPT-5. We then analyzed and compared both translations. We used an analytical and comparative approach in our analysis. Findings unveil that LLMs still cannot 'talk much' about the core syntax properties embedded in the terms under study involving several syntactic and semantic challenges: only 25% of ChatGPT translations were accurate, while 38.6% were inaccurate, and 36.4.% were partially correct, which we consider appropriate. Based on these findings, a set of actionable strategies were proposed, the most notable of which is a close collaboration between AI specialists and linguists to better LLMs' working mechanism for accurate or at least appropriate translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07494v1">Triage: Routing Software Engineering Tasks to Cost-Effective LLM Tiers via Code Quality Signals</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Context: AI coding agents route every task to a single frontier large language model (LLM), paying premium inference cost even when many tasks are routine. Objectives: We propose Triage, a framework that uses code health metrics -- indicators of software maintainability -- as a routing signal to assign each task to the cheapest model tier whose output passes the same verification gate as the expensive model. Methods: Triage defines three capability tiers (light, standard, heavy -- mirroring, e.g., Haiku, Sonnet, Opus) and routes tasks based on pre-computed code health sub-factors and task metadata. We design an evaluation comparing three routing policies on SWE-bench Lite (300 tasks across three model tiers): heuristic thresholds, a trained ML classifier, and a perfect-hindsight oracle. Results: We analytically derived two falsifiable conditions under which the tier-dependent asymmetry (medium LLMs benefit from clean code while frontier models do not) yields cost-effective routing: the light-tier pass rate on healthy code must exceed the inter-tier cost ratio, and code health must discriminate the required model tier with at least a small effect size ($\hat{p} \geq 0.56$). Conclusion: Triage transforms a diagnostic code quality metric into an actionable model-selection signal. We present a rigorous evaluation protocol to test the cost--quality trade-off and identify which code health sub-factors drive routing decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07486v1">Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 23 pages, 7 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful tool for synthetic data generation. A particularly important use case is producing synthetic replicas of private text, which requires carefully balancing privacy and utility. We propose Realistic and Privacy-Preserving Synthetic Data Generation (RPSG), which leverages privacy-preserving mechanisms, including formal differential privacy (DP); and private seeds, in particular text containing personal information, to generate realistic synthetic data. Comprehensive experiments against state-of-the-art private synthetic data generation methods demonstrate that RPSG achieves high fidelity to private data while providing strong privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05650v2">See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 ACL'2026 Main Conference
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) excel in video understanding but suffer from high inference latency during autoregressive generation. Speculative Decoding (SD) mitigates this by applying a draft-and-verify paradigm, yet existing methods are constrained by rigid exact-match rules, severely limiting the acceleration potential. To bridge this gap, we propose LVSpec, the first training-free loosely SD framework tailored for Video-LLMs. Grounded in the insight that generation is governed by sparse visual-relevant anchors (mandating strictness) amidst abundant visual-irrelevant fillers (permitting loose verification), LVSpec employs a lightweight visual-relevant token identification scheme to accurately pinpoint the former. To further maximize acceptance, we augment this with a position-shift tolerant mechanism that effectively salvages positionally mismatched but semantically equivalent tokens. Experiments demonstrate that LVSpec achieves high fidelity and speed: it preserves >99.8 of target performance while accelerating Qwen2.5-VL-32B by 2.70x and LLaVA-OneVision-72B by 2.94x. Notably, it boosts the mean accepted length and speedup ratio by 136% and 35% compared to SOTA training-free SD methods for Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07472v1">Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Deploying large language model (LLM) inference at scale requires jointly selecting base models, provisioning heterogeneous GPUs, configuring parallelism, and distributing workloads under tight latency, accuracy, and budget constraints. Exact mixed-integer linear programming (MILP) approaches guarantee optimality but scale poorly. We propose two constraint-aware heuristics: a Greedy Heuristic (GH) for single-pass allocation, and an Adaptive Greedy Heuristic (AGH) that enhances GH via multi-start construction, relocate-based local search, and GPU consolidation. Three constraint-aware mechanisms -- TP-aware feasibility selection, cost-per-effective-coverage ranking, and TP upgrade -- ensure feasibility under tightly coupled memory, delay, error, and budget constraints. On workloads calibrated with the Azure LLM Inference Trace (2025), both heuristics produce feasible solutions in under one second, with AGH closely approaching optimal cost while achieving over 260x speedup on large-scale instances. Under out-of-sample stress tests with up to 1.5x parameter inflation, AGH maintains controlled SLO violations and stable cost, whereas the exact solver's placement degrades sharply.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07469v1">To Layer or Not to Layer? Evaluating the Effects and Mechanisms of LLM-Generated Feedback on learning performance</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Feedback is vital for learning, yet its effectiveness depends not only on its content but also on how it engages students in the learning process. Large Language Models (LLMs) offer novel opportunities to efficiently generate rich, formative feedback, ranging from direct explanations to incrementally layered scaffolding designed to foster learner autonomy. Despite these affordances, it remains unclear whether layered feedback (which sequences encouragement and prompts prior to revealing the correct answer) actually improves engagement and learning outcomes. To address this, we randomly assigned 199 participants to receive either layered or non-layered LLM-generated feedback. We assessed its impact on learning performance, behavioral and cognitive engagement, and affective perceptions, to determine how these factors mediate learning performance. Results indicate that layered feedback elicited slightly higher behavioral engagement and, as anticipated, was perceived as more encouraging and supportive of independence. However, it concurrently induced greater mental effort. Mediation analyses revealed a positive affective pathway driven by perceived encouragement, which was counteracted by a negative behavioral pathway linked to the average number of tasks requiring $\geq 3$ submissions; the cognitive pathway (mental effort) was non-significant. Taken together, layered feedback resulted in significantly poorer learning outcomes compared to non-layered feedback. These findings illuminate a critical trade-off: while layered scaffolding enhances engagement and positive perceptions, it can detrimentally impact actual learning performance. This study contributes nuanced insights for the design of automated, LLM-driven feedback systems by integrating outcome, perception, and mechanism-level analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07321v1">Syntax Is Easy, Semantics Is Hard: Evaluating LLMs for LTL Translation</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 SecDev 2026 in Montreal, Canada, 10 pages, maximum 16 pages
    </div>
    <details class="paper-abstract">
      Propositional Linear Temporal Logic (LTL) is a popular formalism for specifying desirable requirements and security and privacy policies for software, networks, and systems. Yet expressing such requirements and policies in LTL remains challenging because of its intricate semantics. Since many security and privacy analysis tools require LTL formulas as input, this difficulty places them out of reach for many developers and analysts. Large Language Models (LLMs) could broaden access to such tools by translating natural language fragments into LTL formulas. This paper evaluates that premise by assessing how effectively several representative LLMs translate assertive English sentences into LTL formulas. Using both human-generated and synthetic ground-truth data, we evaluate effectiveness along syntactic and semantic dimensions. The results reveal three findings: (1) in line with prior findings, LLMs perform better on syntactic aspects of LTL than on semantic ones; (2) they generally benefit from more detailed prompts; and (3) reformulating the task as a Python code-completion problem substantially improves overall performance. We also discuss challenges in conducting a fair evaluation on this task and conclude with recommendations for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09354v3">"Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Accepted at CSCW 2026. 53 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. \emph{Peer support}, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client (\client{}), context-sensitive LLM-generated suggestions (\suggestions{}), and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 6 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22177v2">Analyzing Multimodal Interaction Strategies for LLM-Assisted Manipulation of 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 Published in the IEEE VR (IEEE Conference on Virtual Reality and 3D User Interfaces) 2025 Proceedings
    </div>
    <details class="paper-abstract">
      As more applications of large language models (LLMs) for 3D content for immersive environments emerge, it is crucial to study user behaviour to identify interaction patterns and potential barriers to guide the future design of immersive content creation and editing systems which involve LLMs. In an empirical user study with 12 participants, we combine quantitative usage data with post-experience questionnaire feedback to reveal common interaction patterns and key barriers in LLM-assisted 3D scene editing systems. We identify opportunities for improving natural language interfaces in 3D design tools and propose design recommendations for future LLM-integrated 3D content creation systems. Through an empirical study, we demonstrate that LLM-assisted interactive systems can be used productively in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v4">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07223v1">TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve from static chatbots into autonomous agents, the primary vulnerability surface shifts from final outputs to intermediate execution traces. While safety guardrails are well-benchmarked for natural language responses, their efficacy remains largely unexplored within multi-step tool-use trajectories. To address this gap, we introduce TraceSafe-Bench, the first comprehensive benchmark specifically designed to assess mid-trajectory safety. It encompasses 12 risk categories, ranging from security threats (e.g., prompt injection, privacy leaks) to operational failures (e.g., hallucinations, interface inconsistencies), featuring over 1,000 unique execution instances. Our evaluation of 13 LLM-as-a-guard models and 7 specialized guardrails yields three critical findings: 1) Structural Bottleneck: Guardrail efficacy is driven more by structural data competence (e.g., JSON parsing) than semantic safety alignment. Performance correlates strongly with structured-to-text benchmarks ($ρ=0.79$) but shows near-zero correlation with standard jailbreak robustness. 2) Architecture over Scale: Model architecture influences risk detection performance more significantly than model size, with general-purpose LLMs consistently outperforming specialized safety guardrails in trajectory analysis. 3) Temporal Stability: Accuracy remains resilient across extended trajectories. Increased execution steps allow models to pivot from static tool definitions to dynamic execution behaviors, actually improving risk detection performance in later stages. Our findings suggest that securing agentic workflows requires jointly optimizing for structural reasoning and safety alignment to effectively mitigate mid-trajectory risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07220v1">HIVE: Query, Hypothesize, Verify An LLM Framework for Multimodal Reasoning-Intensive Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-04-08
      | 💬 accepted at CVPR 2026 Workshop GRAIL-V
    </div>
    <details class="paper-abstract">
      Multimodal retrieval models fail on reasoning-intensive queries where images (diagrams, charts, screenshots) must be deeply integrated with text to identify relevant documents -- the best multimodal model achieves only 27.6 nDCG@10 on MM-BRIGHT, underperforming even strong text-only retrievers (32.2). We introduce \textbf{HIVE} (\textbf{H}ypothesis-driven \textbf{I}terative \textbf{V}isual \textbf{E}vidence Retrieval), a plug-and-play framework that injects explicit visual-text reasoning into a retriever via LLMs. HIVE operates in four stages: (1) initial retrieval over the corpus, (2) LLM-based compensatory query synthesis that explicitly articulates visual and logical gaps observed in top-$k$ candidates, (3) secondary retrieval with the refined query, and (4) LLM verification and reranking over the union of candidates. Evaluated on the multimodal-to-text track of MM-BRIGHT (2,803 real-world queries across 29 technical domains), HIVE achieves a new state-of-the-art aggregated nDCG@10 of \textbf{41.7} -- a \textbf{+9.5} point gain over the best text-only model (DiVeR: 32.2) and \textbf{+14.1} over the best multimodal model (Nomic-Vision: 27.6), where our reasoning-enhanced base retriever contributes 33.2 and the HIVE framework adds a further \textbf{+8.5} points -- with particularly strong results in visually demanding domains (Gaming: 68.2, Chemistry: 42.5, Sustainability: 49.4). Compatible with both standard and reasoning-enhanced retrievers, HIVE demonstrates that LLM-mediated visual hypothesis generation and verification can substantially close the multimodal reasoning gap in retrieval. https://github.com/mm-bright/multimodal-reasoning-retrieval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07192v1">Compact Constraint Encoding for LLM Code Generation: An Empirical Study of Token Economics and Constraint Compliance</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      LLMs used for code generation are typically guided by engineering constraints--technology choices, dependency restrictions, and architectural patterns--expressed in verbose natural language. We investigate whether compact, structured constraint headers can reduce prompt token consumption without degrading constraint compliance. Across six experimental rounds spanning 11 models, 16 benchmark tasks, and over 830 LLM invocations, we find that compact headers reduce constraint-portion tokens by approximately 71% and full-prompt tokens by 25--30%, replicated across three independent rounds. However, we detect no statistically significant differences in constraint satisfaction rate (CSR) across three encoding forms or four propagation modes; observed effect sizes are negligible (Cliff's $δ$ < 0.01, 95% CI spanning $\pm$2.6 percentage points). This null pattern holds across two models from different capability tiers. A supplementary experiment with four non-CSS tasks provides additional cross-domain support for the encoding null result. The largest observed sources of compliance variance are constraint type ($Δ$ = 9 percentage points between normal and counter-intuitive constraints) and task domain: counter-intuitive constraints opposing model defaults fail at 10--100%, while conventional constraints achieve 99%+ compliance regardless of encoding. Model self-assessments systematically overestimate compliance relative to rule-based scoring, revealing a gap between constraint understanding and execution. Under the tested conditions, the primary benefit of compact constraint encoding is token reduction rather than compliance improvement, and engineering effort toward compliance is better directed at constraint design than prompt formatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07148v1">Multi-Turn Reasoning LLMs for Task Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Emerging computation-intensive applications impose stringent latency requirements on resource-constrained mobile devices. Mobile Edge Computing (MEC) addresses this challenge through task offloading. However, designing effective policies remains difficult due to dynamic task arrivals, time-varying channels, and the spatio-temporal coupling of server queues. Conventional heuristics lack adaptability, while Deep Reinforcement Learning (DRL) suffers from limited generalization and architectural rigidity, requiring retraining when network topology changes. Although Large Language Models (LLMs) offer semantic reasoning capabilities, standard Supervised Fine-Tuning (SFT) yields myopic policies that greedily minimize immediate latency without accounting for long-term system evolution. To address these limitations, we propose COMLLM, a generative framework that enables foresighted decision-making in MEC systems. COMLLM integrates Group Relative Policy Optimization (GRPO) with a Look-Ahead Collaborative Simulation (LACS) mechanism, which performs multi-step Monte Carlo rollouts while jointly modeling server queue dynamics. By incorporating these rollouts into the reward design, the framework captures the long-term impact of current decisions on future system states. Experimental results demonstrate that COMLLM achieves near-optimal latency and improved load-balancing fairness. Notably, it exhibits zero-shot topological scalability, allowing a model trained on small-scale networks to generalize to larger, unseen topologies without retraining, outperforming SFT, DRL, and heuristic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07144v1">Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving operates in highly volatile environments characterized by severe runtime dynamics, such as workload fluctuations and elastic cluster autoscaling. Traditional serving systems rely on static, human-engineered serving policies (e.g., scheduling algorithms and rescheduling strategies) to manage these dynamics. However, these policies must navigate deeply intertwined runtime trade-offs (e.g., scheduling overhead vs. execution efficiency, rescheduling frequency vs. reconfiguration overhead), whose optimal balance is workload-specific and shifts continuously as runtime conditions evolve, rendering any fixed policy fundamentally unable to adapt. We propose Autopoiesis, a novel online self-evolving system that shifts LLM serving from static policy deployment to continuous online policy evolution. First, Autopoiesis introduces an LLM-driven program synthesis workflow to evolve serving policies with respect to real-time observed dynamics, where the evolved policies reflect the optimal decision in navigating the complex, multi-dimensional trade-off space. Second, Autopoiesis enables this synthesis process to operate continuously during serving, observing real-world system behavior, and rewriting the policy code as runtime trade-offs shift, thereby transforming policy design from a one-time offline endeavor into an ongoing system component, enabling autonomous adaptation to evolving runtime conditions. Together, we establish a new paradigm: Serving policies are no longer static artifacts designed by humans before deployment, but living code that LLMs continuously evolve throughout deployment to navigate runtime trade-offs beyond human design. We evaluate Autopoiesis across diverse runtime dynamics and show up to 53% and on average 34% improvements over state-of-the-art LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07123v1">Language Bias under Conflicting Information in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to contain biases in the process of integrating conflicting information when answering questions. Here we ask whether such biases also exist with respect to which language is used for each conflicting piece of information. To answer this question, we extend the conflicting needles in a haystack paradigm to a multilingual setting and perform a comprehensive set of evaluations with naturalistic news domain data in five different languages, for a range of multilingual LLMs of different sizes. We find that all LLMs tested, including GPT-5.2, ignore the conflict and confidently assert only one of the possible answers in the large majority of cases. Furthermore, there is a consistent bias across models in which languages are preferred, with a general bias against Russian and, for the longest context lengths, in favor of Chinese. Both of these patterns are consistent between models trained inside and outside of mainland China, though somewhat stronger in the former category.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07070v1">EVGeoQA: Benchmarking LLMs on Dynamic, Multi-Objective Geo-Spatial Exploration</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, their potential for purpose-driven exploration in dynamic geo-spatial environments remains under-investigated. Existing Geo-Spatial Question Answering (GSQA) benchmarks predominantly focus on static retrieval, failing to capture the complexity of real-world planning that involves dynamic user locations and compound constraints. To bridge this gap, we introduce EVGeoQA, a novel benchmark built upon Electric Vehicle (EV) charging scenarios that features a distinct location-anchored and dual-objective design. Specifically, each query in EVGeoQA is explicitly bound to a user's real-time coordinate and integrates the dual objectives of a charging necessity and a co-located activity preference. To systematically assess models in such complex settings, we further propose GeoRover, a general evaluation framework based on a tool-augmented agent architecture to evaluate the LLMs' capacity for dynamic, multi-objective exploration. Our experiments reveal that while LLMs successfully utilize tools to address sub-tasks, they struggle with long-range spatial exploration. Notably, we observe an emergent capability: LLMs can summarize historical exploration trajectories to enhance exploration efficiency. These findings establish EVGeoQA as a challenging testbed for future geo-spatial intelligence. The dataset and prompts are available at https://github.com/Hapluckyy/EVGeoQA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07036v1">ReDAct: Uncertainty-Aware Deferral for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Recently, LLM-based agents have become increasingly popular across many applications, including complex sequential decision-making problems. However, they inherit the tendency of LLMs to hallucinate, leading to incorrect decisions. In sequential settings, even a single mistake can irreversibly degrade the trajectory, making hallucinations an even bigger problem. Although larger LLMs hallucinate less, they incur a significantly higher per-token cost. In this paper, we address this tradeoff by proposing ReDAct (Reason-Defer-Act). In ReDAct, an agent is equipped with two LLMs: a small, cheap model used by default, and a large, more reliable but expensive model. When the predictive uncertainty of the small model exceeds a calibrated threshold, the decision is deferred to the large model. We evaluate our approach in text-based embodied environments such as ALFWorld and MiniGrid and show that deferring only about 15% of decisions to the large model can match the quality of using it exclusively, while significantly reducing inference costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01423v2">Active Hypothesis Testing under Computational Budgets with Applications to GWAS and LLM</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      In large-scale hypothesis testing, computing exact $p$-values or $e$-values is often resource-intensive, creating a need for budget-aware inferential methods. We propose a general framework for active hypothesis testing that leverages inexpensive auxiliary statistics to allocate a global computational budget. For each hypothesis, our data-adaptive procedure probabilistically decides whether to compute the exact test statistic or a transformed proxy, guaranteeing a valid $p$-value or $e$-value while satisfying the exact budget constraint. Theoretical guarantees are established for our constructions, showing that the procedure achieves optimality for $e$-values and for $p$-values under independence, and admissibility for $p$-values under general dependence. Empirical results from simulations and two real-world applications, including a large-scale genome-wide association study (GWAS) and a clinical prediction task leveraging large language models (LLM), demonstrate that our framework improves statistical efficiency under fixed resource limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06506v1">Guiding Symbolic Execution with Static Analysis and LLMs for Vulnerability Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Symbolic execution detects vulnerabilities with precision, but applying it to large codebases requires harnesses that set up symbolic state, model dependencies, and specify assertions. Writing these harnesses has traditionally been a manual process requiring expert knowledge, which significantly limits the scalability of the technique. We present Static Analysis Informed and LLM-Orchestrated Symbolic Execution (SAILOR), which automates symbolic execution harness construction by combining static analysis with LLM-based synthesis. SAILOR operates in three phases: (1) static analysis identifies candidate vulnerable locations and generates vulnerability specifications; (2) an LLM uses vulnerability specifications and orchestrates harness synthesis by iteratively refining drivers, stubs, and assertions against compiler and symbolic execution feedback; symbolic execution then detects vulnerabilities using the generated harness, and (3) concrete replay validates the symbolic execution results against the unmodified project source. This design combines the scalability of static analysis, the code reasoning of LLMs, the path precision of symbolic execution, and the ground truth produced by concrete execution. We evaluate SAILOR on 10 open-source C/C++ projects totaling 6.8 M lines of code. SAILOR discovers 379 distinct, previously unknown memory-safety vulnerabilities (421 confirmed crashes). The strongest of five baselines we compare SAILOR to (agentic vulnerability detection using Claude Code with full codebase access and unlimited interaction), finds only 12 vulnerabilities. Each phase of SAILOR is critical: Without static analysis targeting confirmed vulnerabilities drop 12.2X; without iterative LLM synthesis zero vulnerabilities are confirmed; and without symbolic execution no approach can detect more than 12 vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07181v3">PACIFIC: Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06487v1">Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Submitted to Interspeech
    </div>
    <details class="paper-abstract">
      Conventional end-to-end automatic speech recognition (ASR) systems rely on paired speech-text data for domain adaptation. Recent LLM-based ASR architectures connect a speech encoder to a large language model via a projection module, enabling adaptation with text-only data. However, this introduces a modality gap, as the LLM is not exposed to the noisy representations produced by the speech projector. We investigate whether small amounts of speech can mitigate this mismatch. We compare three strategies: text-only adaptation, paired speech-text adaptation, and mixed batching (MB), which combines both. Experiments in in-domain and out-of-domain settings show that even limited speech consistently improves performance. Notably, MB using only 10% of the target-domain (less than 4 hours) speech achieves word error rates comparable to, or better than, conventional ASR fine-tuning with the full dataset, indicating that small amounts of speech provide a strong modality-alignment signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25898v2">On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      LLM-assisted modeling holds the potential to rapidly build executable Digital Twins of complex systems from only coarse descriptions and sensor data. However, resilience to LLM hallucination, human oversight, and real-time model adaptability remain challenging and often mutually conflicting requirements. We present three critical design principles for integrating resilience and oversight into such workflows, derived from insights gained through our work on FactoryFlow - an open-source LLM-assisted framework for building simulation-based Digital Twins of manufacturing systems. First, orthogonalize structural modeling and parameter fitting. Structural descriptions (components, interconnections) are LLM-translated from coarse natural language to an intermediate representation (IR) with human visualization and validation, which is algorithmically converted to the final model. Parameter inference, in contrast, operates continuously on sensor data streams with expert-tunable controls. Second, restrict the model IR to interconnections of parameterized, pre-validated library components rather than monolithic simulation code, enabling interpretability and error-resilience. Third, and most important, is to use a density-preserving IR. When IR descriptions expand dramatically from compact inputs hallucination errors accumulate proportionally. We present the case for Python as a density-preserving IR : loops express regularity compactly, classes capture hierarchy and composition, and the result remains highly readable while exploiting LLMs strong code generation capabilities. A key contribution is detailed characterization of LLM-induced errors across model descriptions of varying detail and complexity, revealing how IR choice critically impacts error rates. These insights provide actionable guidance for building resilient and transparent LLM-assisted simulation automation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06416v1">Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Although LLM context lengths have grown, there is evidence that their ability to integrate information across long-form texts has not kept pace. We evaluate one such understanding task: generating summaries of novels. When human authors of summaries compress a story, they reveal what they consider narratively important. Therefore, by comparing human and LLM-authored summaries, we can assess whether models mirror human patterns of conceptual engagement with texts. To measure conceptual engagement, we align sentences from 150 human-written novel summaries with the specific chapters they reference. We demonstrate the difficulty of this alignment task, which indicates the complexity of summarization as a task. We then generate and align additional summaries by nine state-of-the-art LLMs for each of the 150 reference texts. Comparing the human and model-authored summaries, we find both stylistic differences between the texts and differences in how humans and LLMs distribute their focus throughout a narrative, with models emphasizing the ends of texts. Comparing human narrative engagement with model attention mechanisms suggests explanations for degraded narrative comprehension and targets for future development. We release our dataset to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06403v1">FMI@SU ToxHabits: Evaluating LLMs Performance on Toxic Habit Extraction in Spanish Clinical Texts</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 8 pages, 1 figure, 6 tables, Challenge and Workshop BC9 Large Language Models for Clinical and Biomedical NLP, International Joint Conference on Artificial Intelligence IJCAI 2025
    </div>
    <details class="paper-abstract">
      The paper presents an approach for the recognition of toxic habits named entities in Spanish clinical texts. The approach was developed for the ToxHabits Shared Task. Our team participated in subtask 1, which aims to detect substance use and abuse mentions in clinical case reports and classify them in four categories (Tobacco, Alcohol, Cannabis, and Drug). We explored various methods of utilizing LLMs for the task, including zero-shot, few-shot, and prompt optimization, and found that GPT-4.1's few-shot prompting performed the best in our experiments. Our method achieved an F1 score of 0.65 on the test set, demonstrating a promising result for recognizing named entities in languages other than English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06401v1">ProofSketcher: Hybrid LLM + Lightweight Proof Checker for Reliable Math/Logic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) might produce a persuasive argument within mathematical and logical fields, although such argument often includes some minor missteps, including the entire omission of side conditions, invalid inference patterns, or appeals to a lemma that cannot be derived logically out of the context being discussed. These omissions are infamously hard to notice solely out of the text, as even the misconstrued construction still may seem mostly accurate. Conversely, interactive theorem provers like Lean and Coq have rigorous reliability by ensuring that syntactic and semantic statements only accept statements that can pass all the syntactic and semantic steps in the program which is a small trusted kernel of the language type-checks with. Despite the fact that this technique provides strong guarantees, it comes at quite a heavy price: the evidence must be completely formalized, and the evidence user or a auxiliary search program must provide an avalanche of low-level information. This paper presents a hybrid pipeline where an LLM generates a typed proof sketch in a compact DSL and a lightweight trusted kernel expands the sketch into explicit proof obligations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06393v1">ART: Attention Replacement Technique to Improve Factuality in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Hallucination in large language models (LLMs) continues to be a significant issue, particularly in tasks like question answering, where models often generate plausible yet incorrect or irrelevant information. Although various methods have been proposed to mitigate hallucinations, the relationship between attention patterns and hallucinations has not been fully explored. In this paper, we analyze the distribution of attention scores across each layer and attention head of LLMs, revealing a common and intriguing phenomenon: shallow layers of LLMs primarily rely on uniform attention patterns, where the model distributes its attention evenly across the entire sequence. This uniform attention pattern can lead to hallucinations, as the model fails to focus on the most relevant information. To mitigate this issue, we propose a training-free method called Attention Replacement Technique (ART), which replaces these uniform attention patterns in the shallow layers with local attention patterns. This change directs the model to focus more on the relevant contexts, thus reducing hallucinations. Through extensive experiments, ART demonstrates significant reductions in hallucinations across multiple LLM architectures, proving its effectiveness and generalizability without requiring fine-tuning or additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06385v1">Application-Driven Pedagogical Knowledge Optimization of Open-Source LLMs via Reinforcement Learning and Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 * These authors contributed equally to this work and share first authorship
    </div>
    <details class="paper-abstract">
      We present an innovative multi-stage optimization strategy combining reinforcement learning (RL) and supervised fine-tuning (SFT) to enhance the pedagogical knowledge of large language models (LLMs), as illustrated by EduQwen 32B-RL1, EduQwen 32B-SFT, and an optional third-stage model EduQwen 32B-SFT-RL2: (1) RL optimization that implements progressive difficulty training, focuses on challenging examples, and employs extended reasoning rollouts; (2) a subsequent SFT phase that leverages the RL-trained model to synthesize high-quality training data with difficulty-weighted sampling; and (3) an optional second round of RL optimization. EduQwen 32B-RL1, EduQwen 32B-SFT, and EduQwen 32B-SFT-RL2 are an application-driven family of open-source pedagogical LLMs built on a dense Qwen3-32B backbone. These models remarkably achieve high enough accuracy on the Cross-Domain Pedagogical Knowledge (CDPK) Benchmark to establish new state-of-the-art (SOTA) results across the interactive Pedagogy Benchmark Leaderboard and surpass significantly larger proprietary systems such as the previous benchmark leader Gemini-3 Pro. These dense 32-billion-parameter models demonstrate that domain-specialized optimization can transform mid-sized open-source LLMs into true pedagogical domain experts that outperform much larger general-purpose systems, while preserving the transparency, customizability, and cost-efficiency required for responsible educational AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02624v2">LAsset: An LLM-assisted Security Asset Identification Framework for System-on-Chip (SoC) Verification</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 This paper will be presented at Design, Automation and Test in Europe Conference (DATE) 2026
    </div>
    <details class="paper-abstract">
      The growing complexity of modern system-on-chip (SoC) and IP designs is making security assurance difficult day by day. One of the fundamental steps in the pre-silicon security verification of a hardware design is the identification of security assets, as it substantially influences downstream security verification tasks, such as threat modeling, security property generation, and vulnerability detection. Traditionally, assets are determined manually by security experts, requiring significant time and expertise. To address this challenge, we present LAsset, a novel automated framework that leverages large language models (LLMs) to identify security assets from both hardware design specifications and register-transfer level (RTL) descriptions. The framework performs structural and semantic analysis to identify intra-module primary and secondary assets and derives inter-module relationships to systematically characterize security dependencies at the design level. Experimental results show that the proposed framework achieves high classification accuracy, reaching up to 90% recall rate in SoC design, and 93% recall rate in IP designs. This automation in asset identification significantly reduces manual overhead and supports a scalable path forward for secure hardware development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01925v2">Rectifying LLM Thought from Lens of Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06163v1">Data, Not Model: Explaining Bias toward LLM Texts in Neural Retrievers</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Recent studies show that neural retrievers often display source bias, favoring passages generated by LLMs over human-written ones, even when both are semantically similar. This bias has been considered an inherent flaw of retrievers, raising concerns about the fairness and reliability of modern information access systems. Our work challenges this view by showing that source bias stems from supervision in retrieval datasets rather than the models themselves. We found that non-semantic differences, like fluency and term specificity, exist between positive and negative documents, mirroring differences between LLM and human texts. In the embedding space, the bias direction from negatives to positives aligns with the direction from human-written to LLM-generated texts. We theoretically show that retrievers inevitably absorb the artifact imbalances in the training data during contrastive learning, which leads to their preferences over LLM texts. To mitigate the effect, we propose two approaches: 1) reducing artifact differences in training data and 2) adjusting LLM text vectors by removing their projection on the bias vector. Both methods substantially reduce source bias. We hope our study alleviates some concerns regarding LLM-generated texts in information access systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06297v1">FedSpy-LLM: Towards Scalable and Generalizable Data Reconstruction Attacks from Gradients on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Given the growing reliance on private data in training Large Language Models (LLMs), Federated Learning (FL) combined with Parameter-Efficient Fine-Tuning (PEFT) has garnered significant attention for enhancing privacy and efficiency. Despite FL's privacy benefits, prior studies have shown that private data can still be extracted from shared gradients. However, these studies, mainly on full-parameter model training, are limited to reconstructing small batches, short input sequences, and specific model architectures, such as encoder-based or decoder-based models. The reconstruction quality becomes even worse when dealing with gradients from PEFT methods. To fully understand the practical attack surface of federated LLMs, this paper proposes FedSpy-LLM, a scalable and generalizable data reconstruction attack designed to reconstruct training data with larger batch sizes and longer sequences while generalizing across diverse model architectures, even when PEFT methods are deployed for training. At the core of FedSpy-LLM is a novel gradient decomposition strategy that exploits the rank deficiency and subspace structure of gradients, enabling efficient token extraction while preserving key signal components at scale. This approach further mitigates the reconstruction challenges introduced by PEFT's substantial null space, ensuring robustness across encoder-based, decoder-based, and encoder-decoder model architectures. Additionally, by iteratively aligning each token's partial-sequence gradient with the full-sequence gradient, FedSpy-LLM ensures accurate token ordering in reconstructed sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06091v1">Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly acting as human delegates in multi-agent environments, where a representative agent integrates diverse peer perspectives to make a final decision. Drawing inspiration from social psychology, we investigate how the reliability of this representative agent is undermined by the social context of its network. We define four key phenomena-social conformity, perceived expertise, dominant speaker effect, and rhetorical persuasion-and systematically manipulate the number of adversaries, relative intelligence, argument length, and argumentative styles. Our experiments demonstrate that the representative agent's accuracy consistently declines as social pressure increases: larger adversarial groups, more capable peers, and longer arguments all lead to significant performance degradation. Furthermore, rhetorical strategies emphasizing credibility or logic can further sway the agent's judgment, depending on the context. These findings reveal that multi-agent systems are sensitive not only to individual reasoning but also to the social dynamics of their configuration, highlighting critical vulnerabilities in AI delegates that mirror the psychological biases observed in human group decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05995v3">NativQA Framework: Enabling LLMs and VLMs with Native, Local, and Everyday Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) raises concerns about cultural bias, fairness, and performance in diverse languages and underrepresented regions. Addressing these gaps requires large-scale resources grounded in multilingual, local, and cultural contexts. We systematize and extend the earlier NativQA framework to multimodality by adding image, audio, and video support, enabling scalable construction of culturally and regionally aligned QA datasets in native languages. Given user-defined seed queries, the framework uses search engines to collect location-specific everyday information. We evaluate it across 39 locations in 24 countries and 7 languages, spanning extremely low-resource to high-resource settings, and collect over $\sim$300K text QA pairs, $\sim$312K images, and $\sim$29K videos with associated audio. The developed resources can be used for LLMs benchmarking and further fine-tuning. The framework has been made publicly available for the community (https://gitlab.com/nativqa/nativqa-framework). Demo video is available here: \href{https://shorturl.at/DAVn9}{https://shorturl.at/DAVn9}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06071v1">Stories of Your Life as Others: A Round-Trip Evaluation of LLM-Generated Life Stories Conditioned on Rich Psychometric Profiles</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under review at COLM
    </div>
    <details class="paper-abstract">
      Personality traits are richly encoded in natural language, and large language models (LLMs) trained on human text can simulate personality when conditioned on persona descriptions. However, existing evaluations rely predominantly on questionnaire self-report by the conditioned model, are limited in architectural diversity, and rarely use real human psychometric data. Without addressing these limitations, it remains unclear whether personality conditioning produces psychometrically informative representations of individual differences or merely superficial alignment with trait descriptors. To test how robustly LLMs can encode personality into extended text, we condition LLMs on real psychometric profiles from 290 participants to generate first-person life story narratives, and then task independent LLMs to recover personality scores from those narratives alone. We show that personality scores can be recovered from the generated narratives at levels approaching human test-retest reliability (mean r = 0.750, 85% of the human ceiling), and that recovery is robust across 10 LLM narrative generators and 3 LLM personality scorers spanning 6 providers. Decomposing systematic biases reveals that scoring models achieve their accuracy while counteracting alignment-induced defaults. Content analysis of the generated narratives shows that personality conditioning produces behaviourally differentiated text: nine of ten coded features correlate significantly with the same features in participants' real conversations, and personality-driven emotional reactivity patterns in narratives replicate in real conversational data. These findings provide evidence that the personality-language relationship captured during pretraining supports robust encoding and decoding of individual differences, including characteristic emotional variability patterns that replicate in real human behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06066v1">From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding in LLM Reflection</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction in Large Language Models (LLMs) frequently fails in open-ended reasoning tasks due to ``hallucination snowballing,'' a phenomenon in which models recursively justify early errors during free-text reflection. While structured feedback can mitigate this issue, existing approaches often rely on externally trained critics or symbolic tools, reducing agent autonomy. This study investigates whether enforcing structured reflection purely through Outlines-based constrained decoding can disrupt error propagation without additional training. Evaluating an 8-billion-parameter model (Qwen3-8B), we show that simply imposing structural constraints does not improve self-correction performance. Instead, it triggers a new failure mode termed ``structure snowballing.'' We find that the cognitive load required to satisfy strict formatting rules pushes the model into formatting traps. This observation helps explain why the agent achieves near-perfect superficial syntactic alignment yet fails to detect or resolve deeper semantic errors. These findings expose an ``alignment tax'' inherent to constrained decoding, highlighting a tension between structural granularity and internal model capacity in autonomous workflows. Code and raw logs are available in the GitHub repository: https://github.com/hongxuzhou/agentic_llm_structured_self_critique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15424v2">LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06013v1">Epistemic Blinding: An Inference-Time Protocol for Auditing Prior Contamination in LLM-Assisted Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 code and LLM skill at: https://github.com/mcuccarese/epistemic-blinding 7 pages 3 figures
    </div>
    <details class="paper-abstract">
      This paper presents epistemic blinding in the context of an agentic system that uses large language models to reason across multiple biological datasets for drug target prioritization. During development, it became apparent that LLM outputs silently blend data-driven inference with memorized priors about named entities - and the blend is invisible: there is no way to determine, from a single output, how much came from the data on the page and how much came from the model's training memory. Epistemic blinding is a simple inference-time protocol that replaces entity identifiers with anonymous codes before prompting, then compares outputs against an unblinded control. The protocol does not make LLM reasoning deterministic, but it restores one critical axis of auditability: measuring how much of an output came from the supplied data versus the model's parametric knowledge. The complete target identification system is described - including LLM-guided evolutionary optimization of scoring functions and blinded agentic reasoning for target rationalization - with demonstration that both stages operate without access to entity identity. In oncology drug target prioritization across four cancer types, blinding changes 16% of top-20 predictions while preserving identical recovery of validated targets. The contamination problem is shown to generalize beyond biology: in S&P 500 equity screening, brand-recognition bias reshapes 30-40% of top-20 rankings across five random seeds. To lower the barrier to adoption, the protocol is released as an open-source tool and as a Claude Code skill that enables one-command epistemic blinding within agentic workflows. The claim is not that blinded analysis produces better results, but that without blinding, there is no way to know to what degree the agent is adhering to the analytical process the researcher designed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06008v1">Designing Around Stigma: Human-Centered LLMs for Menstrual Health</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 This is accepted at CHI 2026
    </div>
    <details class="paper-abstract">
      Menstrual health education (MHE) in Pakistan is constrained by cultural taboos and inadequate formal curricula, leaving women with few trusted resources to lean on. In response to these challenges, we introduce a WhatsApp-based chatbot powered by a large language model (LLM) and Retrieval Augmented Generation (RAG), co-designed with Pakistani college women. Workshops (N=30) revealed key design requirements -- support for Roman Urdu, use of subsidized platforms, and an expert -- curated knowledge base. We then deployed the chatbot with 13 participants for two weeks (403 messages and interviews). Women used it to challenge cultural taboos, legitimize health concerns often dismissed as normal, and build reproductive health knowledge through iterative questioning. Yet, interactions also exposed tensions: reliance on cultural explanatory models, questions of trust and validation, and gendered persona of the chatbot itself. We contribute empirical insights, a stigma-aware design framework for culturally sensitive conversational AI, and a methodological lens foregrounding expert validation in intimate health domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.07291v4">Evaluating LLM-based Personal Information Extraction and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 USENIX Security Symposium 2025
    </div>
    <details class="paper-abstract">
      Automatically extracting personal information -- such as name, phone number, and email address -- from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods -- such as regular expression, keyword search, and entity detection -- achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect four datasets including a synthetic dataset generated by GPT-4 and three real-world datasets with manually labeled eight categories of personal information; introduce a novel mitigation strategy based on prompt injection; and systematically benchmark LLM-based attacks and countermeasures using ten LLMs and five datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms traditional methods; and prompt injection can defend against strong LLM-based attacks, reducing the attack to less effective traditional ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05965v1">Beyond Compromise: Pareto-Lenient Consensus for Efficient Multi-Preference LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Transcending the single-preference paradigm, aligning LLMs with diverse human values is pivotal for robust deployment. Contemporary Multi-Objective Preference Alignment (MPA) approaches predominantly rely on static linear scalarization or rigid gradient projection to navigate these trade-offs. However, by enforcing strict conflict avoidance or simultaneous descent, these paradigms often prematurely converge to local stationary points. While mathematically stable, these points represent a conservative compromise where the model sacrifices potential global Pareto improvements to avoid transient local trade-offs. To break this deadlock, we propose Pareto-Lenient Consensus (PLC), a game-theoretic framework that reimagines alignment as a dynamic negotiation process. Unlike rigid approaches, PLC introduces consensus-driven lenient gradient rectification, which dynamically tolerates local degradation provided there is a sufficient dominant coalition surplus, thereby empowering the optimization trajectory to escape local suboptimal equilibrium and explore the distal Pareto-optimal frontier. Theoretical analysis validates PLC can facilitate stalemate escape and asymptotically converge to a Pareto consensus equilibrium. Moreover, extensive experiments show that PLC surpasses baselines in both fixed-preference alignment and global Pareto frontier quality. This work highlights the potential of negotiation-driven alignment as a promising avenue for MPA. Our codes are available at https://anonymous.4open.science/r/aaa-6BB8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21357v2">AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05955v1">Does Pass Rate Tell the Whole Story? Evaluating Design Constraint Compliance in LLM-based Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Repository-level issue resolution benchmarks have become a standard testbed for evaluating LLM-based agents, yet success is still predominantly measured by test pass rates. In practice, however, acceptable patches must also comply with project-specific design constraints, such as architectural conventions, error-handling policies, and maintainability requirements, which are rarely encoded in tests and are often documented only implicitly in code review discussions. This paper introduces \textit{design-aware issue resolution} and presents \bench{}, a benchmark that makes such implicit design constraints explicit and measurable. \bench{} is constructed by mining and validating design constraints from real-world pull requests, linking them to issue instances, and automatically checking patch compliance using an LLM-based verifier, yielding 495 issues and 1,787 validated constraints across six repositories, aligned with SWE-bench-Verified and SWE-bench-Pro. Experiments with state-of-the-art agents show that test-based correctness substantially overestimates patch quality: fewer than half of resolved issues are fully design-satisfying, design violations are widespread, and functional correctness exhibits negligible statistical association with design satisfaction. While providing issue-specific design guidance reduces violations, substantial non-compliance remains, highlighting a fundamental gap in current agent capabilities and motivating design-aware evaluation beyond functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05942v1">BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Post-training hybridization of large language models (LLMs) often replaces quadratic self-attention with sliding-window attention (SWA) to reduce KV cache usage and improve latency. Existing hybridization schemes are typically defined either at the layer level (e.g., interleaving) or at the head level via static rankings from local to global. Layer-level schemes ignore that local and global dependencies are routed through heads within the same layer, while static head-level rankings suffer from entanglement: a head's local/global behavior can change after hybridization. We propose BOSCH, Black-box Binary Optimization for Short-context Head Selection, a training-free method that formulates the problem as a Large Neighborhood Search and decomposes it into three subproblems: (i) layer-importance detection via small-budget black-box probes, (ii) adaptive per-layer SWA-ratio assignment based on these sensitivities, and (iii) grouped head-level optimization within ratio buckets. Extensive experiments on 4 LLMs ranging from 1.7B to 30B parameters, across 4 SWA ratios, show that BOSCH consistently outperforms layer-level heuristics and 6 strong static head-level methods, with larger gains at higher SWA ratios. Under continual pretraining, BOSCH recover original long-context performance faster and to a higher level. Analysis of the selected heads reveals substantial turnover for BOSCH across different SWA ratios, underscoring the importance of performing head-level selection for each target ratio rather than relying on fixed locality rankings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19894v5">TransAgent: Enhancing LLM-Based Code Translation via Fine-Grained Execution Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted by FSE'26
    </div>
    <details class="paper-abstract">
      Code translation transforms code between programming languages while preserving functionality, which is critical in software development and maintenance. While traditional learning-based code translation methods have limited effectiveness due to the lack of sufficient parallel training data, Large Language Models (LLMs) have recently advanced this field with their strong code generation and comprehension capabilities. However, code translated by LLMs still suffers from diverse quality issues, such as syntax and semantic errors. In this work, we propose TransAGENT, a novel multi-agent system that eliminates the errors during LLM-based code translation. The main insight of TransAGENT is to localize error-prone code blocks via fine-grained execution alignment between source and target code. We evaluate TransAGENT on a newly constructed benchmark of recent programming tasks to mitigate data leakage. TransAGENT outperforms the latest UniTrans by up to 33.3% in translation accuracy and achieves an average improvement of 56.7% over Agentless in program repair performance. We also conduct an ablation study and evaluate TransAGENT across different LLMs, demonstrating its effectiveness and strong generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05872v1">Swiss-Bench 003: Evaluating LLM Reliability and Adversarial Security for Swiss Regulatory Contexts</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 23 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in Swiss financial and regulatory contexts demands empirical evidence of both production reliability and adversarial security, dimensions not jointly operationalized in existing Swiss-focused evaluation frameworks. This paper introduces Swiss-Bench 003 (SBP-003), extending the HAAS (Helvetic AI Assessment Score) from six to eight dimensions by adding D7 (Self-Graded Reliability Proxy) and D8 (Adversarial Security). I evaluate ten frontier models across 808 Swiss-specific items in four languages (German, French, Italian, English), comprising seven Swiss-adapted benchmarks (Swiss TruthfulQA, Swiss IFEval, Swiss SimpleQA, Swiss NIAH, Swiss PII-Scope, System Prompt Leakage, and Swiss German Comprehension) targeting FINMA Guidance 08/2024, the revised Federal Act on Data Protection (nDSG), and OWASP Top 10 for LLMs. Self-graded D7 scores (73-94%) exceed externally judged D8 security scores (20-61%) by a wide margin, though these dimensions use non-comparable scoring regimes. System prompt leakage resistance ranges from 24.8% to 88.2%, while PII extraction defense remains weak (14-42%) across all models. Qwen 3.5 Plus achieves the highest self-graded D7 score (94.4%), while GPT-oss 120B achieves the highest D8 score (60.7%) despite being the lowest-cost model evaluated. All evaluations are zero-shot under provider default settings; D7 is self-graded and does not constitute independently validated accuracy. I provide conceptual mapping tables relating benchmark dimensions to FINMA model validation requirements, nDSG data protection obligations, and OWASP LLM risk categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05859v1">When Do We Need LLMs? A Diagnostic for Language-Driven Bandits</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ICLR 2026 Workshop on AI Advances in Finance
    </div>
    <details class="paper-abstract">
      We study Contextual Multi-Armed Bandits (CMABs) for non-episodic sequential decision making problems where the context includes both textual and numerical information (e.g., recommendation systems, dynamic portfolio adjustments, offer selection; all frequent problems in finance). While Large Language Models (LLMs) are increasingly applied to these settings, utilizing LLMs for reasoning at every decision step is computationally expensive and uncertainty estimates are difficult to obtain. To address this, we introduce LLMP-UCB, a bandit algorithm that derives uncertainty estimates from LLMs via repeated inference. However, our experiments demonstrate that lightweight numerical bandits operating on text embeddings (dense or Matryoshka) match or exceed the accuracy of LLM-based solutions at a fraction of their cost. We further show that embedding dimensionality is a practical lever on the exploration-exploitation balance, enabling cost--performance tradeoffs without prompt complexity. Finally, to guide practitioners, we propose a geometric diagnostic based on the arms' embedding to decide when to use LLM-driven reasoning versus a lightweight numerical bandit. Our results provide a principled deployment framework for cost-effective, uncertainty-aware decision systems with broad applicability across AI use cases in financial services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05846v1">AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at https://github.com/sunyuanfu/AgentGL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05808v1">Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities in complex interactive decision-making tasks. However, existing LLM agents typically rely on increasingly long interaction histories, resulting in high computational cost and limited scalability. In this paper, we propose STEP-HRL, a hierarchical reinforcement learning (HRL) framework that enables step-level learning by conditioning only on single-step transitions rather than full interaction histories. STEP-HRL structures tasks hierarchically, using completed subtasks to represent global progress of overall task. By introducing a local progress module, it also iteratively and selectively summarizes interaction history within each subtask to produce a compact summary of local progress. Together, these components yield augmented step-level transitions for both high-level and low-level policies. Experimental results on ScienceWorld and ALFWorld benchmarks consistently demonstrate that STEP-HRL substantially outperforms baselines in terms of performance and generalization while reducing token usage. Our code is available at https://github.com/TonyStark042/STEP-HRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05793v1">BodhiPromptShield: Pre-Inference Prompt Mediation for Suppressing Privacy Propagation in LLM/VLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      In LLM/VLM agents, prompt privacy risk propagates beyond a single model call because raw user content can flow into retrieval queries, memory writes, tool calls, and logs. Existing de-identification pipelines address document boundaries but not this cross-stage propagation. We propose BodhiPromptShield, a policy-aware framework that detects sensitive spans, routes them via typed placeholders, semantic abstraction, or secure symbolic mapping, and delays restoration to authorized boundaries. Relative to enterprise redaction, this adds explicit propagation-aware mediation and restoration timing as a security variable. Under controlled evaluation on the Controlled Prompt-Privacy Benchmark (CPPB), stage-wise propagation suppresses from 10.7\% to 7.1\% across retrieval, memory, and tool stages; PER reaches 9.3\% with 0.94 AC and 0.92 TSR, outperforming generic de-identification. These are controlled systems results on CPPB rather than formal privacy guarantees or public-benchmark transfer claims. The project repository is available at https://github.com/mabo1215/BodhiPromptShield.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05782v1">An Empirical Study of Perceptions of General LLMs and Multimodal LLMs on Hugging Face</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly evolved from general-purpose systems to multimodal models capable of processing text, images, and audio. As both general-purpose LLMs (GLLMs) and multimodal LLMs (MLLMs) gain widespread adoption, understanding user perceptions in real-world settings becomes increasingly important. However, existing studies often rely on surveys or platform-specific data (e.g., Reddit or GitHub issues), which either constrain user feedback through predefined questions or overemphasize failure-driven, debugging-oriented discussions, thus failing to capture diverse, experience-driven, and cross-model user perspectives in practice. To address this issue, we conduct an empirical study of user discussions on Hugging Face, a major model hub with diverse models and active communities. We collect and manually annotate 662 discussion threads from 38 representative models (21 GLLMs and 17 MLLMs), and develop a three-level taxonomy to systematically characterize user concerns. Our analysis reveals that LLM access barriers, generation quality, and deployment and invocation complexity are the most prominent concerns, alongside issues such as documentation limitations and resource constraints. Based on these findings, we derive actionable implications for improving LLM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14226v5">Phonetic Perturbations Reveal Tokenizer-Rooted Safety Gaps in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Safety-aligned LLMs remain vulnerable to digital phenomena like textese that introduce non-canonical perturbations to words but preserve the phonetics. We introduce CMP-RT (code-mixed phonetic perturbations for red-teaming), a novel diagnostic probe that pinpoints tokenization as the root cause of this vulnerability. A mechanistic analysis reveals that phonetic perturbations fragment safety-critical tokens into benign sub-words, suppressing their attribution scores while preserving prompt interpretability -- causing safety mechanisms to fail despite excellent input understanding. We demonstrate that this vulnerability evades standard defenses, persists across modalities and state-of-the-art (SOTA) models including Gemini-3-Pro, and scales through simple supervised fine-tuning (SFT). Furthermore, layer-wise probing shows perturbed and canonical input representations align up to a critical layer depth; enforcing output equivalence robustly recovers the lost representations, providing causal evidence for a structural gap between pre-training and alignment, and establishing tokenization as a critical, under-examined vulnerability in current safety pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05775v1">PhageBench: Can LLMs Understand Raw Bacteriophage Genomes?</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Bacteriophages, often referred to as the dark matter of the biosphere, play a critical role in regulating microbial ecosystems and in antibiotic alternatives. Thus, accurate interpretation of their genomes holds significant scientific and practical value. While general-purpose Large Language Models (LLMs) excel at understanding biological texts, their ability to directly interpret raw nucleotide sequences and perform biological reasoning remains underexplored. To address this, we introduce PhageBench, the first benchmark designed to evaluate phage genome understanding by mirroring the workflow of bioinformatics experts. The dataset contains 5,600 high-quality samples covering five core tasks across three stages: Screening, Quality Control, and Phenotype Annotation. Our evaluation of eight LLMs reveals that general-purpose reasoning models significantly outperform random baselines in phage contig identification and host prediction, demonstrating promising potential for genomic understanding. However, they exhibit significant limitations in complex reasoning tasks involving long-range dependencies and fine-grained functional localization. These findings highlight the necessity of developing next-generation models with enhanced reasoning capabilities for biological sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05766v1">The LLM Effect on IR Benchmarks: A Meta-Analysis of Effectiveness, Baselines, and Contamination</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted at SIGIR 2026
    </div>
    <details class="paper-abstract">
      Benchmark collections have long enabled controlled comparison and cumulative progress in Information Retrieval (IR). However, prior meta-analyses have shown that reported effectiveness gains often fail to accumulate, in part due to the use of weak or outdated baselines. While large language models are increasingly used in retrieval pipelines, their impact on established IR benchmarks has not been systematically analyzed. In this study, we analyze 143 publications reporting results on the TREC Robust04 collection and the TREC Deep Learning 2020 (DL20) passage retrieval benchmark to examine longitudinal trends in retrieval effectiveness and baseline strength. We observe what we term an \emph{LLM effect}: recent systems incorporating LLM components achieve 8.8\% higher nDCG@10 on DL20 compared to the best result from TREC 2020 and approximately 20\% higher on Robust04 since 2023. However, adapting a data contamination detection approach to reranking reveals measurable contamination in both benchmarks. While excluding contaminated topics reduces effectiveness, confidence intervals remain wide, making it difficult to determine whether the LLM effect reflects genuine methodological advances or memorization from pretraining data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05756v1">Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted at ACL Main Conference
    </div>
    <details class="paper-abstract">
      While the real world is inherently stochastic, Large Language Models (LLMs) are predominantly evaluated on single-round inference against fixed ground truths. In this work, we shift the lens to distribution alignment: assessing whether LLMs, when prompted repeatedly, can generate outputs that adhere to a desired target distribution, e.g. reflecting real-world statistics or a uniform distribution. We formulate distribution alignment using the attributes of gender, race, and sentiment within occupational contexts. Our empirical analysis reveals that off-the-shelf LLMs and standard alignment techniques, including prompt engineering and Direct Preference Optimization, fail to reliably control output distributions. To bridge this gap, we propose a novel fine-tuning framework that couples Steering Token Calibration with Semantic Alignment. We introduce a hybrid objective function combining Kullback-Leibler divergence to anchor the probability mass of latent steering tokens and Kahneman-Tversky Optimization to bind these tokens to semantically consistent responses. Experiments across six diverse datasets demonstrate that our approach significantly outperforms baselines, achieving precise distributional control in attribute generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05719v1">Hackers or Hallucinators? A Comprehensive Analysis of LLM-Based Automated Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has created new opportunities for Automated Penetration Testing (AutoPT), spawning numerous frameworks aimed at achieving end-to-end autonomous attacks. However, despite the proliferation of related studies, existing research generally lacks systematic architectural analysis and large-scale empirical comparisons under a unified benchmark. Therefore, this paper presents the first Systematization of Knowledge (SoK) focusing on the architectural design and comprehensive empirical evaluation of current LLM-based AutoPT frameworks. At systematization level, we comprehensively review existing framework designs across six dimensions: agent architecture, agent plan, agent memory, agent execution, external knowledge, and benchmarks. At empirical level, we conduct large-scale experiments on 13 representative open-source AutoPT frameworks and 2 baseline frameworks utilizing a unified benchmark. The experiments consumed over 10 billion tokens in total and generated more than 1,500 execution logs, which were manually reviewed and analyzed over four months by a panel of more than 15 researchers with expertise in cybersecurity. By investigating the latest progress in this rapidly developing field, we provide researchers with a structured taxonomy to understand existing LLM-based AutoPT frameworks and a large-scale empirical benchmark, along with promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05681v1">LUDOBENCH: Evaluating LLM Behavioural Decision-Making Through Spot-Based Board Game Scenarios in Ludo</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      We introduce LudoBench, a benchmark for evaluating LLM strategic reasoning in Ludo, a stochastic multi-agent board game whose dice mechanics, piece capture, safe-square navigation, and home-path progression introduce meaningful planning complexity. LudoBench comprises 480 handcrafted spot scenarios across 12 behaviorally distinct decision categories, each isolating a specific strategic choice. We additionally contribute a fully functional 4-player Ludo simulator supporting Random, Heuristic, Game-Theory, and LLM agents. The game-theory agent uses Expectiminimax search with depth-limited lookahead to provide a principled strategic ceiling beyond greedy heuristics. Evaluating six models spanning four model families, we find that all models agree with the game-theory baseline only 40-46% of the time. Models split into distinct behavioral archetypes: finishers that complete pieces but neglect development, and builders that develop but never finish. Each archetype captures only half of the game theory strategy. Models also display measurable behavioral shifts under history-conditioned grudge framing on identical board states, revealing prompt-sensitivity as a key vulnerability. LudoBench provides a lightweight and interpretable framework for benchmarking LLM strategic reasoning under uncertainty. All code, the spot dataset (480 entries) and model outputs are available at https://anonymous.4open.science/r/LudoBench-5CBF/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05674v1">From Incomplete Architecture to Quantified Risk: Multimodal LLM-Driven Security Assessment for Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Cyber-physical systems often contend with incomplete architectural documentation or outdated information resulting from legacy technologies, knowledge management gaps, and the complexity of integrating diverse subsystems over extended operational lifecycles. This architectural incompleteness impedes reliable security assessment, as inaccurate or missing architectural knowledge limits the identification of system dependencies, attack surfaces, and risk propagation pathways. To address this foundational challenge, this paper introduces ASTRAL (Architecture-Centric Security Threat Risk Assessment using LLMs), an architecture-centric security assessment technique implemented in a prototype tool powered by multimodal LLMs. The proposed approach assists practitioners in reconstructing and analysing CPS architectures when documentation is fragmented or absent. By leveraging prompt chaining, few-shot learning, and architectural reasoning, ASTRAL extracts and synthesises system representations from disparate data sources. By integrating LLM reasoning with architectural modelling, our approach supports adaptive threat identification and quantitative risk estimation for cyber-physical systems. We evaluated the approach through an ablation study across multiple CPS case studies and an expert evaluation involving 14 experienced cybersecurity practitioners. Practitioner feedback suggests that ASTRAL is useful and reliable for supporting architecture-centric security assessment. Overall, the results indicate that the approach can support more informed cyber risk management decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05663v1">CuraLight: Debate-Guided Data Curation for LLM-Centered Traffic Signal Control</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 accepted at IJCNN 2026
    </div>
    <details class="paper-abstract">
      Traffic signal control (TSC) is a core component of intelligent transportation systems (ITS), aiming to reduce congestion, emissions, and travel time. Recent approaches based on reinforcement learning (RL) and large language models (LLMs) have improved adaptivity, but still suffer from limited interpretability, insufficient interaction data, and weak generalization to heterogeneous intersections. This paper proposes CuraLight, an LLM-centered framework where an RL agent assists the fine-tuning of an LLM-based traffic signal controller. The RL agent explores traffic environments and generates high-quality interaction trajectories, which are converted into prompt-response pairs for imitation fine-tuning. A multi-LLM ensemble deliberation system further evaluates candidate signal timing actions through structured debate, providing preference-aware supervision signals for training. Experiments conducted in SUMO across heterogeneous real-world networks from Jinan, Hangzhou, and Yizhuang demonstrate that CuraLight consistently outperforms state-of-the-art baselines, reducing average travel time by 5.34 percent, average queue length by 5.14 percent, and average waiting time by 7.02 percent. The results highlight the effectiveness of combining RL-assisted exploration with deliberation-based data curation for scalable and interpretable traffic signal control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05655v1">LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 ACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      This work characterizes large language models' chain-of-thought generation as a structured trajectory through representation space. We show that mathematical reasoning traverses functionally ordered, step-specific subspaces that become increasingly separable with layer depth. This structure already exists in base models, while reasoning training primarily accelerates convergence toward termination-related subspaces rather than introducing new representational organization. While early reasoning steps follow similar trajectories, correct and incorrect solutions diverge systematically at late stages. This late-stage divergence enables mid-reasoning prediction of final-answer correctness with ROC-AUC up to 0.87. Furthermore, we introduce trajectory-based steering, an inference-time intervention framework that enables reasoning correction and length control based on derived ideal trajectories. Together, these results establish reasoning trajectories as a geometric lens for interpreting, predicting, and controlling LLM reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05643v1">Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a major source of such redundancy is inefficient reflection, which often manifests in two problematic patterns: Indiscriminate Reflection, where the model performs broad, low-impact checks throughout reasoning, and Repetitive Reflection, where it repeatedly re-verifies an already established conclusion. To address this, we introduce a graph-based CoT optimization framework. Specifically, we convert each linear CoT into a directed acyclic graph (DAG) with explicit dependency edges, and design a dual pruning strategy: branch-level pruning removes weakly contributing reflection branches, while depth-level pruning eliminates late-stage re-verification. We distill this behavior via a three-stage pipeline: (1) SFT to initialize the policy on pruned concise traces, (2) DPO to prefer correct but less redundant trajectories, and (3) GRPO with length penalty to jointly optimize answer correctness and efficiency. Experiments show that our approach reduces the average reasoning tokens by 42\% while maintaining or improving accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13151v3">Quantization-Robust LLM Unlearning via Low-Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 Accepted to IJCNN 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) unlearning aims to remove targeted knowledge from a trained model, but practical deployments often require post-training quantization (PTQ) for efficient inference. However, aggressive low-bit PTQ can mask unlearning updates, causing quantized models to revert to pre-unlearning behavior. We show that standard full-parameter fine-tuning often induces parameter changes that are too small to survive 4-bit quantization. We propose quantization-robust unlearning via low-rank adaptation (LoRA): we freeze the base model and concentrate unlearning into trainable adapters so that the effective update is preserved after quantization. On Llama-2-7B evaluated with MUSE dataset (BOOKS and NEWS), LoRA improves 4-bit utility by up to 7.93 points (NPO+GDR on BOOKS: 50.17 to 58.10) and yields higher 4-bit utility on NEWS for GA+GDR (40.06 to 44.82, increase of 4.76). LoRA also substantially reduces privacy leakage under 4-bit PTQ, e.g., for GA+KLR on BOOKS, PrivLeak moves from -25.68 to -5.86 (closer to ideal 0), while maintaining strong forgetting (VerMem and KnowMem near 0). Thus, using LoRA for Machine Unlearning is beneficial for scenarios where quantization is necessary for model deployment.
    </details>
</div>
