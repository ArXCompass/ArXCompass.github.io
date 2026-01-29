# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06780v3">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted and published in Findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03746v2">Whose Facts Win? LLM Source Preferences under Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Data and code: https://github.com/JaSchuste/llm-source-preference
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15793v3">Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted by AAAI 2026, camera ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including texts, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Yet it remains unclear whether such biases are systematic, which data-level factors drive them, and what internal mechanisms underlie their emergence. In this paper, we present the first comprehensive study of format bias in LLMs through a three-stage empirical analysis. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage examines how key data-level factors influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its effectiveness. Our results show that format bias is consistent across model families, driven by information richness, structure quality, and representation type, and is closely associated with attention imbalance within the LLMs. Based on these investigations, we identify three future research directions to reduce format bias: enhancing data pre-processing through format repair and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08343v1">When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems routinely generate multiple candidate responses that are aggregated by an LLM judge. To reduce the dominant prefill cost in such pipelines, recent work advocates KV cache reuse across partially shared contexts and reports substantial speedups for generation agents. In this work, we show that these efficiency gains do not transfer uniformly to judge-centric inference. Across GSM8K, MMLU, and HumanEval, we find that reuse strategies that are effective for execution agents can severely perturb judge behavior: end-task accuracy may appear stable, yet the judge's selection becomes highly inconsistent with dense prefill. We quantify this risk using Judge Consistency Rate (JCR) and provide diagnostics showing that reuse systematically weakens cross-candidate attention, especially for later candidate blocks. Our ablation further demonstrates that explicit cross-candidate interaction is crucial for preserving dense-prefill decisions. Overall, our results identify a previously overlooked failure mode of KV cache reuse and highlight judge-centric inference as a distinct regime that demands dedicated, risk-aware system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07198v3">Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 20 pages, 5 figures, 21 tables. Accepted at NeurIPS 2025. Corresponding author: Xuan Zhang (xuanzhang2199@gmail.com)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive generalization abilities, yet adapting them effectively across multiple heterogeneous domains remains challenging due to inter-domain interference. To overcome this challenge, we propose a partition-based multi-stage fine-tuning framework designed to exploit inter-domain synergies while minimizing negative transfer. Our approach strategically partitions domains into subsets (stages) by balancing domain discrepancy, synergy, and model capacity constraints. We theoretically analyze the proposed framework and derive novel generalization bounds that justify our partitioning strategy. Extensive empirical evaluations on various language understanding tasks show that our method consistently outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08271v1">Sparsity Is Necessary: Polynomial-Time Stability for Agentic LLMs in Large Action Spaces</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM systems expose a control regime that learning theory has largely ignored: sequential decision-making with a massive discrete action universe (tools, APIs, documents) in which only a small, unknown subset is relevant for any fixed task distribution. We formalize this setting as Sparse Agentic Control (SAC), where policies admit block-sparse representations over M >> 1 actions and rewards depend on sparse main effects and (optionally) sparse synergies. We study ell_{1,2}-regularized policy learning through a convex surrogate and establish sharp, compressed-sensing-style results: (i) estimation and value suboptimality scale as k (log M / T)^{1/2} under a Policy-RSC condition; (ii) exact tool-support recovery holds via primal-dual witness arguments when T > k log M under incoherence and beta-min; and (iii) any dense policy class requires Omega(M) samples, explaining the instability of prompt-only controllers. We further show that under partial observability, LLMs matter only through a belief/representation error epsilon_b, yielding an additive O(epsilon_b) degradation while preserving logarithmic dependence on M. Extensions cover tuning-free, online, robust, group-sparse, and interaction-aware SAC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02369v3">AutoContext: Instance-Level Context Learning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Current LLM agents typically lack instance-level context, which comprises concrete facts such as environment structure, system configurations, and local mechanics. Consequently, existing methods are forced to intertwine exploration with task execution. This coupling leads to redundant interactions and fragile decision-making, as agents must repeatedly rediscover the same information for every new task. To address this, we introduce AutoContext, a method that decouples exploration from task solving. AutoContext performs a systematic, one-off exploration to construct a reusable knowledge graph for each environment instance. This structured context allows off-the-shelf agents to access necessary facts directly, eliminating redundant exploration. Experiments across TextWorld, ALFWorld, Crafter, and InterCode-Bash demonstrate substantial gains: for example, the success rate of a ReAct agent on TextWorld improves from 37% to 95%, highlighting the critical role of structured instance context in efficient agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02962v6">RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities, are prone to generating hallucinated or outdated content due to their static internal knowledge. While Retrieval-Augmented Generation (RAG) integrated with Reinforcement Learning (RL) offers a solution, these methods are fundamentally constrained by a single-query mode, leading to prohibitive latency and inherent brittleness. To overcome these limitations, we introduce RAG-R1, a novel two-stage training framework centered around multi-query parallelism. Our framework enables LLMs to adaptively leverage internal and external knowledge during the reasoning process while transitioning from the single-query mode to multi-query parallelism. This architectural shift bolsters reasoning robustness while significantly reducing inference latency. Extensive experiments on seven question-answering benchmarks confirm the superiority of our method, which outperforms the strongest baseline by up to 13.7% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08237v1">The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Reward engineering, the manual specification of reward functions to induce desired agent behavior, remains a fundamental challenge in multi-agent reinforcement learning. This difficulty is amplified by credit assignment ambiguity, environmental non-stationarity, and the combinatorial growth of interaction complexity. We argue that recent advances in large language models (LLMs) point toward a shift from hand-crafted numerical rewards to language-based objective specifications. Prior work has shown that LLMs can synthesize reward functions directly from natural language descriptions (e.g., EUREKA) and adapt reward formulations online with minimal human intervention (e.g., CARD). In parallel, the emerging paradigm of Reinforcement Learning from Verifiable Rewards (RLVR) provides empirical evidence that language-mediated supervision can serve as a viable alternative to traditional reward engineering. We conceptualize this transition along three dimensions: semantic reward specification, dynamic reward adaptation, and improved alignment with human intent, while noting open challenges related to computational overhead, robustness to hallucination, and scalability to large multi-agent systems. We conclude by outlining a research direction in which coordination arises from shared semantic representations rather than explicitly engineered numerical signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08196v1">Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into autonomous agents has enabled complex tool use, yet in high-stakes domains, these systems must strictly adhere to regulatory standards beyond simple functional correctness. However, existing benchmarks often overlook implicit regulatory compliance, thus failing to evaluate whether LLMs can autonomously enforce mandatory safety constraints. To fill this gap, we introduce LogiSafetyGen, a framework that converts unstructured regulations into Linear Temporal Logic oracles and employs logic-guided fuzzing to synthesize valid, safety-critical traces. Building on this framework, we construct LogiSafetyBench, a benchmark comprising 240 human-verified tasks that require LLMs to generate Python programs that satisfy both functional objectives and latent compliance rules. Evaluations of 13 state-of-the-art (SOTA) LLMs reveal that larger models, despite achieving better functional correctness, frequently prioritize task completion over safety, which results in non-compliant behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17627v2">The Evolution of Thought: Tracking LLM Overthinking via Reasoning Dynamics Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Test-time scaling via explicit reasoning trajectories significantly boosts large language model (LLM) performance but often triggers overthinking. To explore this, we analyze reasoning through two lenses: Reasoning Length Dynamics, which reveals a compensatory trade-off between thinking and answer content length that eventually leads to thinking redundancy, and Reasoning Semantic Dynamics, which identifies semantic convergence and repetitive oscillations. These dynamics uncover an instance-specific Reasoning Completion Point (RCP), beyond which computation continues without further performance gain. Since the RCP varies across instances, we propose a Reasoning Completion Point Detector (RCPD), an inference-time early-exit method that identifies the RCP by monitoring the rank dynamics of termination tokens (e.g., </think>). Across AIME and GPQA benchmarks using Qwen3 and DeepSeek-R1, RCPD reduces token usage by up to 44% while preserving accuracy, offering a principled approach to efficient test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08187v1">Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising capabilities in Text-Attributed Graph (TAG) understanding. Recent studies typically focus on verbalizing the graph structures via handcrafted prompts, feeding the target node and its neighborhood context into LLMs. However, constrained by the context window, existing methods mainly resort to random sampling, often implemented via dropping node/edge randomly, which inevitably introduces noise and cause reasoning instability. We argue that graphs inherently contain rich structural and semantic information, and that their effective exploitation can unlock potential gains in LLMs reasoning performance. To this end, we propose Homophily-aware Structural and Semantic Compression for LLMs (HS2C), a framework centered on exploiting graph homophily. Structurally, guided by the principle of Structural Entropy minimization, we perform a global hierarchical partition that decodes the graph's essential topology. This partition identifies naturally cohesive, homophilic communities, while discarding stochastic connectivity noise. Semantically, we deliver the detected structural homophily to the LLM, empowering it to perform differentiated semantic aggregation based on predefined community type. This process compresses redundant background contexts into concise community-level consensus, selectively preserving semantically homophilic information aligned with the target nodes. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families demonstrate that, by feeding LLMs with structurally and semantically compressed inputs, HS2C simultaneously enhances the compression rate and downstream inference accuracy, validating its superiority and scalability. Extensions to 7 diverse graph-level benchmarks further consolidate HS2C's task generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18882v5">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08166v1">ZeroDVFS: Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 39 pages, 12 figures, 8 tables (including appendix)
    </div>
    <details class="paper-abstract">
      Dynamic voltage and frequency scaling (DVFS) and task-to-core allocation are critical for thermal management and balancing energy and performance in embedded systems. Existing approaches either rely on utilization-based heuristics that overlook stall times, or require extensive offline profiling for table generation, preventing runtime adaptation. We propose a model-based hierarchical multi-agent reinforcement learning (MARL) framework for thermal- and energy-aware scheduling on multi-core platforms. Two collaborative agents decompose the exponential action space, achieving 358ms latency for subsequent decisions. First decisions require 3.5 to 8.0s including one-time LLM feature extraction. An accurate environment model leverages regression techniques to predict thermal dynamics and performance states. When combined with LLM-extracted semantic features, the environment model enables zero-shot deployment for new workloads on trained platforms by generating synthetic training data without requiring workload-specific profiling samples. We introduce LLM-based semantic feature extraction that characterizes OpenMP programs through 13 code-level features without execution. The Dyna-Q-inspired framework integrates direct reinforcement learning with model-based planning, achieving 20x faster convergence than model-free methods. Experiments on BOTS and PolybenchC benchmarks across NVIDIA Jetson TX2, Jetson Orin NX, RubikPi, and Intel Core i7 demonstrate 7.09x better energy efficiency and 4.0x better makespan than Linux ondemand governor. First-decision latency is 8,300x faster than table-based profiling, enabling practical deployment in dynamic embedded systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06266v2">Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to SANER 2026 (IEEE International Conference on Software Analysis, Evolution and Reengineering)
    </div>
    <details class="paper-abstract">
      Self-admitted technical debt (SATD), referring to comments flagged by developers that explicitly acknowledge suboptimal code or incomplete functionality, has received extensive attention in machine learning (ML) and traditional (Non-ML) software. However, little is known about how SATD manifests and evolves in contemporary Large Language Model (LLM)-based systems, whose architectures, workflows, and dependencies differ fundamentally from both traditional and pre-LLM ML software. In this paper, we conduct the first empirical study of SATD in the LLM era, replicating and extending prior work on ML technical debt to modern LLM-based systems. We compare SATD prevalence across LLM, ML, and non-ML repositories across a total of 477 repositories (159 per category). We perform survival analysis of SATD introduction and removal to understand the dynamics of technical debt across different development paradigms. Surprisingly, despite their architectural complexity, our results reveal that LLM repositories accumulate SATD at similar rates to ML systems (3.95% vs. 4.10%). However, we observe that LLM repositories remain debt-free 2.4x longer than ML repositories (a median of 492 days vs. 204 days), and then start to accumulate technical debt rapidly. Moreover, our qualitative analysis of 377 SATD instances reveals three new forms of technical debt unique to LLM-based development that have not been reported in prior research: Model-Stack Workaround Debt, Model Dependency Debt, and Performance Optimization Debt. Finally, by mapping SATD to stages of the LLM development pipeline, we observe that debt concentrates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20002v3">LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 This submission is withdrawn due to internal review and compliance considerations
    </div>
    <details class="paper-abstract">
      Time-series forecasting in real-world applications such as finance and energy often faces challenges due to limited training data and complex, noisy temporal dynamics. Existing deep forecasting models typically supervise predictions using full-length temporal windows, which include substantial high-frequency noise and obscure long-term trends. Moreover, auxiliary variables containing rich domain-specific information are often underutilized, especially in few-shot settings. To address these challenges, we propose LoFT-LLM, a frequency-aware forecasting pipeline that integrates low-frequency learning with semantic calibration via a large language model (LLM). Firstly, a Patch Low-Frequency forecasting Module (PLFM) extracts stable low-frequency trends from localized spectral patches. Secondly, a residual learner then models high-frequency variations. Finally, a fine-tuned LLM refines the predictions by incorporating auxiliary context and domain knowledge through structured natural language prompts. Extensive experiments on financial and energy datasets demonstrate that LoFT-LLM significantly outperforms strong baselines under both full-data and few-shot regimes, delivering superior accuracy, robustness, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08107v1">STO-RL: Offline RL under Sparse Rewards via LLM-Guided Subgoal Temporal Order</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted at International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      Offline reinforcement learning (RL) enables policy learning from pre-collected datasets, avoiding costly and risky online interactions, but it often struggles with long-horizon tasks involving sparse rewards. Existing goal-conditioned and hierarchical offline RL methods decompose such tasks and generate intermediate rewards to mitigate limitations of traditional offline RL, but usually overlook temporal dependencies among subgoals and rely on imprecise reward shaping, leading to suboptimal policies. To address these issues, we propose STO-RL (Offline RL using LLM-Guided Subgoal Temporal Order), an offline RL framework that leverages large language models (LLMs) to generate temporally ordered subgoal sequences and corresponding state-to-subgoal-stage mappings. Using this temporal structure, STO-RL applies potential-based reward shaping to transform sparse terminal rewards into dense, temporally consistent signals, promoting subgoal progress while avoiding suboptimal solutions. The resulting augmented dataset with shaped rewards enables efficient offline training of high-performing policies. Evaluations on four discrete and continuous sparse-reward benchmarks demonstrate that STO-RL consistently outperforms state-of-the-art offline goal-conditioned and hierarchical RL baselines, achieving faster convergence, higher success rates, and shorter trajectories. Ablation studies further confirm STO-RL's robustness to imperfect or noisy LLM-generated subgoal sequences, demonstrating that LLM-guided subgoal temporal structures combined with theoretically grounded reward shaping provide a practical and scalable solution for long-horizon offline RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08089v1">Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Public large language models (LLMs) are typically safety-aligned during pretraining, yet task-specific fine-tuning required for deployment often erodes this alignment and introduces safety risks. Existing defenses either embed safety recovery into fine-tuning or rely on fine-tuning-derived priors for post-hoc correction, leaving safety recovery tightly coupled with training and incurring high computational overhead and a complex workflow. To address these challenges, we propose \texttt{Q-realign}, a post-hoc defense method based on post-training quantization, guided by an analysis of representational structure. By reframing quantization as a dual-objective procedure for compression and safety, \texttt{Q-realign} decouples safety alignment from fine-tuning and naturally piggybacks into modern deployment pipelines. Experiments across multiple models and datasets demonstrate that our method substantially reduces unsafe behaviors while preserving task performance, with significant reductions in memory usage and GPU hours. Notably, our approach can recover the safety alignment of a fine-tuned 7B LLM on a single RTX 4090 within 40 minutes. Overall, our work provides a practical, turnkey solution for safety-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09029v1">Proactively Detecting Threats: A Novel Approach Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 2025 International Conference on Cyber-Enabled Distributed Computing and Knowledge Discovery (CyberC)
    </div>
    <details class="paper-abstract">
      Enterprise security faces escalating threats from sophisticated malware, compounded by expanding digital operations. This paper presents the first systematic evaluation of large language models (LLMs) to proactively identify indicators of compromise (IOCs) from unstructured web-based threat intelligence sources, distinguishing it from reactive malware detection approaches. We developed an automated system that pulls IOCs from 15 web-based threat report sources to evaluate six LLM models (Gemini, Qwen, and Llama variants). Our evaluation of 479 webpages containing 2,658 IOCs (711 IPv4 addresses, 502 IPv6 addresses, 1,445 domains) reveals significant performance variations. Gemini 1.5 Pro achieved 0.958 precision and 0.788 specificity for malicious IOC identification, while demonstrating perfect recall (1.0) for actual threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.04083v3">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to appear in proceedings of the 3rd Workshop on Artificial Intelligence for Scientific Publications (WASP@AACL 2025). Code and data are available here: https://github.com/bionlproc/ZeroShotRE
    </div>
    <details class="paper-abstract">
      Extracting relations from scientific literature is a fundamental task in biomedical NLP because entities and relations among them drive hypothesis generation and knowledge discovery. As literature grows rapidly, relation extraction (RE) is indispensable to curate knowledge graphs to be used as computable structured and symbolic representations. With the rise of LLMs, it is pertinent to examine if it is better to skip tailoring supervised RE methods, save annotation burden, and just use zero shot RE (ZSRE) via LLM API calls. In this paper, we propose a benchmark with seven biomedical RE datasets with interesting characteristics and evaluate three Open AI models (GPT-4, o1, and GPT-OSS-120B) for end-to-end ZSRE. We show that LLM-based ZSRE is inching closer to supervised methods in performances on some datasets but still struggles on complex inputs expressing multiple relations with different predicates. Our error analysis reveals scope for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15801v2">LingVarBench: Benchmarking LLMs on Entity Recognitions and Linguistic Verbalization Patterns in Phone-Call Transcripts</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to EACL 2026 (Industry Track); to appear in the proceedings
    </div>
    <details class="paper-abstract">
      We study structured entity extraction from phone-call transcripts in customer-support and healthcare settings, where annotation is costly, and data access is limited by privacy and consent. Existing methods degrade under disfluencies, interruptions, and speaker overlap, yet large real-call corpora are rarely shareable. We introduce LingVarBench, a benchmark and semantic synthetic data generation pipeline that generates linguistically varied training data via (1) LLM-sampled entity values, (2) curated linguistic verbalization patterns covering diverse disfluencies and entity-specific readout styles, and (3) a value-transcript consistency filter. Using this dataset, DSPy's SIMBA automatically synthesizes and optimizes extraction prompts, reducing manual prompt engineering and targeting robustness to verbal variation. On real customer transcripts, prompts optimized solely on LingVarBench outperform zero-shot baselines and match or closely approach human-tuned prompts for structured entities such as ZIP code, date of birth, and name (F1 approximately 94-95 percent). For subjective questionnaire items, optimized prompts substantially improve over zero-shot performance and approach human-tuned prompts. LingVarBench offers a practical and cost-efficient path to deployment in a direct-answer setting, with real annotations later enabling additional refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09017v1">Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has necessitated more robust evaluation methods that go beyond static benchmarks, which are increasingly prone to data saturation and leakage. In this paper, we propose a dynamic benchmarking framework for evaluating multilingual and multicultural capabilities through the social deduction game Spyfall. In our setup, models must engage in strategic dialogue to either identify a secret agent or avoid detection, utilizing culturally relevant locations or local foods. Our results show that our game-based rankings align closely with the Chatbot Arena. However, we find a significant performance gap in non-English contexts: models are generally less proficient when handling locally specific entities and often struggle with rule-following or strategic integrity in non-English languages. We demonstrate that this game-based approach provides a scalable, leakage-resistant, and culturally nuanced alternative to traditional NLP benchmarks. The game history can be accessed here https://huggingface.co/datasets/haryoaw/cultural-spyfall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20634v2">Recidivism and Peer Influence with LLM Text Embeddings in Low Security Correctional Facilities</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Studying peer effects in language is critical because they often reflect behavioral and personality traits that are important determinants of economic outcomes. However, language is unstructured, non-numeric, and high-dimensional. We combine Large Language Model (LLM) embeddings with structural econometric identification to provide a unified framework for identifying peer effects in language. This unified framework is applied to 80,000-120,000 written exchanges among residents of low security correctional facilities. The LLM language profiles predict three-year recidivism 30\% more accurately than pre-entry covariates alone, showing that text representations capture meaningful signals. We analyze peer effects on multidimensional language embeddings while addressing network endogeneity. We develop novel instrumental variable estimators for peer effects that accommodate multivariate outcomes, sparse networks, and multidimensional latent variables. Our methods achieve root-N consistency and asymptotic normality under realistic sparsity conditions, relaxing the dense-network assumption. Results reveal significant peer effects in residents' language profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v1">Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08998v1">On the Flakiness of LLM-Generated Tests for Industrial and Open-Source Database Management Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 12 pages, 5 tables, 3 figures, accepted at the 48th International Conference on Software Engineering: Software Engineering in Practice (ICSE SEIP 2026)
    </div>
    <details class="paper-abstract">
      Flaky tests are a common problem in software testing. They produce inconsistent results when executed multiple times on the same code, invalidating the assumption that a test failure indicates a software defect. Recent work on LLM-based test generation has identified flakiness as a potential problem with generated tests. However, its prevalence and underlying causes are unclear. We examined the flakiness of LLM-generated tests in the context of four relational database management systems: SAP HANA, DuckDB, MySQL, and SQLite. We amplified test suites with two LLMs, GPT-4o and Mistral-Large-Instruct-2407, to assess the flakiness of the generated test cases. Our results suggest that generated tests have a slightly higher proportion of flaky tests compared to existing tests. Based on a manual inspection, we found that the most common root cause of flakiness was the reliance of a test on a certain order that is not guaranteed ("unordered collection"), which was present in 72 of 115 flaky tests (63%). Furthermore, both LLMs transferred the flakiness from the existing tests to the newly generated tests via the provided prompt context. Our experiments suggest that flakiness transfer is more prevalent in closed-source systems such as SAP HANA than in open-source systems. Our study informs developers on what types of flakiness to expect from LLM-generated tests. It also highlights the importance of providing LLMs with tailored context when employing LLMs for test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08919v1">Fine Grained Evaluation of LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      A good deal of recent research has focused on how Large Language Models (LLMs) may be used as `judges' in place of humans to evaluate the quality of the output produced by various text / image processing systems. Within this broader context, a number of studies have investigated the specific question of how effectively LLMs can be used as relevance assessors for the standard ad hoc task in Information Retrieval (IR). We extend these studies by looking at additional questions. Most importantly, we use a Wikipedia based test collection created by the INEX initiative, and prompt LLMs to not only judge whether documents are relevant / non-relevant, but to highlight relevant passages in documents that it regards as useful. The human relevance assessors involved in creating this collection were given analogous instructions, i.e., they were asked to highlight all passages within a document that respond to the information need expressed in a query. This enables us to evaluate the quality of LLMs as judges not only at the document level, but to also quantify how often these `judges' are right for the right reasons. Our findings suggest that LLMs-as-judges work best under human supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08892v1">Evaluating Role-Consistency in LLMs for Counselor Training</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08829v1">Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 In submission. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      In this work, we explore the Large Language Model (LLM) agent reviewer dynamics in an Elo-ranked review system using real-world conference paper submissions. Multiple LLM agent reviewers with different personas are engage in multi round review interactions moderated by an Area Chair. We compare a baseline setting with conditions that incorporate Elo ratings and reviewer memory. Our simulation results showcase several interesting findings, including how incorporating Elo improves Area Chair decision accuracy, as well as reviewers' adaptive review strategy that exploits our Elo system without improving review effort. Our code is available at https://github.com/hsiangwei0903/EloReview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08773v1">Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 46 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation for software engineering often relies on vector similarity search, which captures topical similarity but can fail on multi-hop architectural reasoning such as controller to service to repository chains, interface-driven wiring, and inheritance. This paper benchmarks three retrieval pipelines on Java codebases (Shopizer, with additional runs on ThingsBoard and OpenMRS Core): (A) vector-only No-Graph RAG, (B) an LLM-generated knowledge graph RAG (LLM-KB), and (C) a deterministic AST-derived knowledge graph RAG (DKB) built with Tree-sitter and bidirectional traversal. Using 15 architecture and code-tracing queries per repository, we measure indexing time, query latency, corpus coverage, cost, and answer correctness. DKB builds its graph in seconds, while LLM-KB requires much longer graph generation. LLM-KB also shows indexing incompleteness: on Shopizer, 377 files are skipped or missed, reducing embedded chunk coverage and graph size compared to DKB. End-to-end cost is modest for DKB relative to the vector-only baseline but much higher for LLM-KB, especially as repository scale increases. Query latency is similar for No-Graph and DKB, while LLM-KB is slower and more variable. On the Shopizer question suite, DKB achieves the highest correctness, LLM-KB is close behind, and the vector-only baseline performs worst on upstream architectural queries and has the highest hallucination risk. Overall, deterministic AST-derived graphs provide more reliable coverage and multi-hop grounding than LLM-extracted graphs at substantially lower indexing cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05467v2">STELP: Secure Transpilation and Execution of LLM-Generated Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08763v1">Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07767v1">Are LLM Decisions Faithful to Verbal Confidence?</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce surprisingly sophisticated estimates of their own uncertainty. However, it remains unclear to what extent this expressed confidence is tied to the reasoning, knowledge, or decision making of the model. To test this, we introduce $\textbf{RiskEval}$: a framework designed to evaluate whether models adjust their abstention policies in response to varying error penalties. Our evaluation of several frontier models reveals a critical dissociation: models are neither cost-aware when articulating their verbal confidence, nor strategically responsive when deciding whether to engage or abstain under high-penalty conditions. Even when extreme penalties render frequent abstention the mathematically optimal strategy, models almost never abstain, resulting in utility collapse. This indicates that calibrated verbal confidence scores may not be sufficient to create trustworthy and interpretable AI systems, as current models lack the strategic agency to convert uncertainty signals into optimal and risk-sensitive decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14656v2">Cost-Awareness in Tree-Search LLM Planning: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Planning under resource constraints is central to real-world decision making, yet most large language model (LLM) planners assume uniform action costs. We systematically analyze whether tree-search LLM planners are cost-aware and whether they efficiently generate budget-feasible plans. In contrast to black-box prompting, explicit search trees expose intermediate decisions, node evaluations, and failure modes, which allows for controlled ablations of planner behavior. We study depth-first search, breadth-first search, Monte Carlo Tree Search, and bidirectional search within a unified framework. Our experiments show that existing tree-based LLM planners often struggle to find cost-optimal plans, and that additional search computation does not reliably improve optimality. Among the methods evaluated, bidirectional search achieves the best overall efficiency and success rate. MCTS achieves the highest optimality on short-horizon tasks. Tree-search planners are especially valuable for studying LLM planning because their reasoning steps are explicit, in contrast to plain LLMs that internalize planning dynamics through post-training trajectories. Our findings suggest that improving LLM planning under resource constraints will likely require new search algorithms, rather than solely scaling inference-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07667v1">Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Source code is available at https://github.com/TANIGUCHIREI/ASL
    </div>
    <details class="paper-abstract">
      Due to the prevalence of large language models (LLMs), key-value (KV) cache reduction for LLM inference has received remarkable attention. Among numerous works that have been proposed in recent years, layer-wise token pruning approaches, which select a subset of tokens at particular layers to retain in KV cache and prune others, are one of the most popular schemes. They primarily adopt a set of pre-defined layers, at which tokens are selected. Such design is inflexible in the sense that the accuracy significantly varies across tasks and deteriorates in harder tasks such as KV retrieval. In this paper, we propose ASL, a training-free method that adaptively chooses the selection layer for KV cache reduction, exploiting the variance of token ranks ordered by attention score. The proposed method balances the performance across different tasks while meeting the user-specified KV budget requirement. ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage. By evaluations on the InfiniteBench, RULER, and NIAH benchmarks, we show that equipped with one-shot token selection, where tokens are selected at a layer and propagated to deeper layers, ASL outperforms state-of-the-art layer-wise token selection methods in accuracy while maintaining decoding speed and KV cache reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11218v3">The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Code: https://github.com/WorldHellow/SLAQ/tree/main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07593v1">GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      RTL design often relies heavily on ad-hoc testbench creation early in the design cycle. While large language models (LLMs) show promise for RTL code generation, their ability to reason about hardware specifications and generate targeted test plans remains largely unexplored. We present the first systematic study of LLM reasoning capabilities for RTL verification stimuli generation, establishing a two-stage framework that decomposes test plan generation from testbench execution. Our benchmark reveals that state-of-the-art models, including DeepSeek-R1 and Claude-4.0-Sonnet, achieve only 15.7-21.7% success rates on generating stimuli that pass golden RTL designs. To improve LLM generated stimuli, we develop a comprehensive training methodology combining supervised fine-tuning with a novel reinforcement learning approach, GRPO with State Mutation (GRPO-SMu), which enhances exploration by varying input mutations. Our approach leverages a tree-based branching mutation strategy to construct training data comprising equivalent and mutated trees, moving beyond linear mutation approaches to provide rich learning signals. Training on this curated dataset, our 7B parameter model achieves a 33.3% golden test pass rate and a 13.9% mutation detection rate, representing a 17.6% absolute improvement over baseline and outperforming much larger general-purpose models. These results demonstrate that specialized training methodologies can significantly enhance LLM reasoning capabilities for hardware verification tasks, establishing a foundation for automated sub-unit testing in semiconductor design workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06747v2">LLMs Enable Bag-of-Texts Representations for Short-Text Clustering</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      In this paper, we propose a training-free method for unsupervised short text clustering that relies less on careful selection of embedders than other methods. In customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these settings, no labeled data is typically available, and the number of clusters is not known. Recent approaches to short-text clustering in label-free settings incorporate LLM output to refine existing embeddings. While LLMs can identify similar texts effectively, the resulting similarities may not be directly represented by distances in the dense vector space, as they depend on the original embedding. We therefore propose a method for transforming LLM judgments directly into a bag-of-texts representation in which texts are initialized to be equidistant, without assuming any prior distance relationships. Our method achieves comparable or superior results to state-of-the-art methods, but without embeddings optimization or assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show how our method scales to large datasets, reducing the computational cost of the LLM use. The flexibility and scalability of our method make it more aligned with real-world training-free scenarios than existing clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07568v1">d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs, such as parallel decoding and random-order generation. However, realizing these benefits in practice is non-trivial, as dLLMs inherently face an accuracy-parallelism trade-off. Despite increasing interest, existing methods typically focus on only one-side of the coin, targeting either efficiency or performance. To address this limitation, we propose d3LLM (Pseudo-Distilled Diffusion Large Language Model), striking a balance between accuracy and parallelism: (i) during training, we introduce pseudo-trajectory distillation to teach the model which tokens can be decoded confidently at early steps, thereby improving parallelism; (ii) during inference, we employ entropy-based multi-block decoding with a KV-cache refresh mechanism to achieve high parallelism while maintaining accuracy. To better evaluate dLLMs, we also introduce AUP (Accuracy Under Parallelism), a new metric that jointly measures accuracy and parallelism. Experiments demonstrate that our d3LLM achieves up to 10$\times$ speedup over vanilla LLaDA/Dream and 5$\times$ speedup over AR models without much accuracy drop. Our code is available at https://github.com/hao-ai-lab/d3LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03027v2">Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Preference alignment methods such as RLHF and Direct Preference Optimization (DPO) improve instruction following, but they can also reinforce hallucinations when preference judgments reward fluency and confidence over factual correctness. We introduce F-DPO (Factuality-aware Direct Preference Optimization), a simple extension of DPO that uses only binary factuality labels. F-DPO (i) applies a label-flipping transformation that corrects misordered preference pairs so the chosen response is never less factual than the rejected one, and (ii) adds a factuality-aware margin that emphasizes pairs with clear correctness differences, while reducing to standard DPO when both responses share the same factuality. We construct factuality-aware preference data by augmenting DPO pairs with binary factuality indicators and synthetic hallucinated variants. Across seven open-weight LLMs (1B-14B), F-DPO consistently improves factuality and reduces hallucination rates relative to both base models and standard DPO. On Qwen3-8B, F-DPO reduces hallucination rates by five times (from 0.424 to 0.084) while improving factuality scores by 50 percent (from 5.26 to 7.90). F-DPO also generalizes to out-of-distribution benchmarks: on TruthfulQA, Qwen2.5-14B achieves plus 17 percent MC1 accuracy (0.500 to 0.585) and plus 49 percent MC2 accuracy (0.357 to 0.531). F-DPO requires no auxiliary reward model, token-level annotations, or multi-stage training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00333v2">IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      While large language models excel on high-resource multilingual tasks, low- and extremely low-resource Indic languages remain severely under-evaluated. We present IndicParam, a human-curated benchmark of over 13,000 multiple-choice questions covering 11 such languages (Nepali, Gujarati, Marathi, Odia as low-resource; Dogri, Maithili, Rajasthani, Sanskrit, Bodo, Santali, Konkani as extremely low-resource) plus Sanskrit-English code-mixed set. We evaluated 20 LLMs, both proprietary and open-weights, which reveals that even the top-performing \texttt{Gemini-2.5} reaches 58\% average accuracy, followed by \texttt{GPT-5} (45) and \texttt{DeepSeek-3.2} (43.1). We additionally label each question as knowledge-oriented or purely linguistic to discriminate factual recall from grammatical proficiency. Further, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. \benchmark\ provides insights into limitations of cross-lingual transfer and establishes a challenging benchmark for Indic languages. The dataset is available at https://huggingface.co/datasets/bharatgenai/IndicParam. Scripts to run benchmark are present at https://github.com/ayushbits/IndicParam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07506v1">Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Under review, 21 pgs, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07504v1">FROAV: A Framework for RAG Observation and Agent Verification - Lowering the Barrier to LLM Agent Research</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 8 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) and their integration into autonomous agent systems has created unprecedented opportunities for document analysis, decision support, and knowledge retrieval. However, the complexity of developing, evaluating, and iterating on LLM-based agent workflows presents significant barriers to researchers, particularly those without extensive software engineering expertise. We present FROAV (Framework for RAG Observation and Agent Verification), an open-source research platform that democratizes LLM agent research by providing a plug-and-play architecture combining visual workflow orchestration, a comprehensive evaluation framework, and extensible Python integration. FROAV implements a multi-stage Retrieval-Augmented Generation (RAG) pipeline coupled with a rigorous "LLM-as-a-Judge" evaluation system, all accessible through intuitive graphical interfaces. Our framework integrates n8n for no-code workflow design, PostgreSQL for granular data management, FastAPI for flexible backend logic, and Streamlit for human-in-the-loop interaction. Through this integrated ecosystem, researchers can rapidly prototype RAG strategies, conduct prompt engineering experiments, validate agent performance against human judgments, and collect structured feedback-all without writing infrastructure code. We demonstrate the framework's utility through its application to financial document analysis, while emphasizing its material-agnostic architecture that adapts to any domain requiring semantic analysis. FROAV represents a significant step toward making LLM agent research accessible to a broader scientific community, enabling researchers to focus on hypothesis testing and algorithmic innovation rather than system integration challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16674v4">Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for text generation, making it crucial to address potential bias. This study investigates ideological framing bias in LLM-generated articles, focusing on the subtle and subjective nature of such bias in journalistic contexts. We evaluate eight widely used LLMs on two datasets-POLIGEN and ECONOLEX-covering political and economic discourse where framing bias is most pronounced. Beyond text generation, LLMs are increasingly used as evaluators (LLM-as-a-judge), providing feedback that can shape human judgment or inform newer model versions. Inspired by the Socratic method, we further analyze LLMs' feedback on their own outputs to identify inconsistencies in their reasoning. Our results show that most LLMs can accurately annotate ideologically framed text, with GPT-4o achieving human-level accuracy and high agreement with human annotators. However, Socratic probing reveals that when confronted with binary comparisons, LLMs often exhibit preference toward one perspective or perceive certain viewpoints as less biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07475v1">ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      The emergence of fine-grained numerical formats like NVFP4 presents new opportunities for efficient Large Language Model (LLM) inference. However, it is difficult to adapt existing Post-Training Quantization (PTQ) strategies to these formats: rotation-based methods compromise fine-grained block isolation; smoothing techniques struggle with significant 4-bit quantization errors; and mixed-precision approaches often conflict with hardware constraints on unified-precision computation. To address these challenges, we propose ARCQuant, a framework that boosts NVFP4 performance via Augmented Residual Channels. Distinct from methods that compromise block isolation or hardware uniformity, ARCQuant maintains a strictly unified NVFP4 format by augmenting the activation matrix with quantized residual channels. This design integrates the error compensation process directly into the matrix reduction dimension, enabling the use of standard, highly optimized GEMM kernels with minimal overhead. Theoretical analysis confirms that the worst-case error bound of our dual-stage NVFP4 quantization is comparable to that of standard 8-bit formats such as MXFP8. Extensive experiments on LLaMA and Qwen models demonstrate that ARCQuant achieves state-of-the-art accuracy, comparable to full-precision baselines in perplexity and downstream tasks. Furthermore, deployment on RTX 5090 and RTX PRO 6000 GPUs confirms practical benefits, achieving up to 3x speedup over FP16. Our code is available at https://github.com/actypedef/ARCQuant .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07469v1">Knowledge Distillation for LLM-Based Human Activity Recognition in Homes</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Human Activity Recognition (HAR) is a central problem for context-aware applications, especially for smart homes and assisted living. A few very recent studies have shown that Large Language Models (LLMs) can be used for HAR at home, reaching high performance and addressing key challenges. In this paper, we provide new experimental results regarding the use of LLMs for HAR, on two state-of-the-art datasets. More specifically, we show how recognition performance evolves depending on the size of the LLM used. Moreover, we experiment on the use of knowledge distillation techniques to fine-tune smaller LLMs with HAR reasoning examples generated by larger LLMs. We show that such fine-tuned models can perform almost as well as the largest LLMs, while having 50 times less parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07468v1">Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Memory enables Large Language Model (LLM) agents to perceive, store, and use information from past dialogues, which is essential for personalization. However, existing methods fail to properly model the temporal dimension of memory in two aspects: 1) Temporal inaccuracy: memories are organized by dialogue time rather than their actual occurrence time; 2) Temporal fragmentation: existing methods focus on point-wise memory, losing durative information that captures persistent states and evolving patterns. To address these limitations, we propose Temporal Semantic Memory (TSM), a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory. During memory construction, it first builds a semantic timeline rather than a dialogue one. Then, it consolidates temporally continuous and semantically related information into a durative memory. During memory utilization, it incorporates the query's temporal intent on the semantic timeline, enabling the retrieval of temporally appropriate durative memories and providing time-valid, duration-consistent context to support response generation. Experiments on LongMemEval and LoCoMo show that TSM consistently outperforms existing methods and achieves up to 12.2% absolute improvement in accuracy, demonstrating the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08997v2">Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on Large Language Models (LLMs) show exceptional promise for complex collaborative problem-solving, yet they face fundamental challenges stemming from context window limitations that impair memory consistency, role adherence, and procedural integrity. This paper introduces Intrinsic Memory Agents, a novel framework that addresses these limitations through agent-specific memories that evolve intrinsically with agent outputs. Specifically, our method maintains role-aligned memory that preserves specialized perspectives while focusing on task-relevant information. Our approach utilises a generic memory template applicable to new problems without the need to hand-craft specific memory prompts. We benchmark our approach on the PDDL, FEVER, and ALFWorld datasets, comparing its performance to existing state-of-the-art multi-agentic memory approaches and showing state-of-the-art or comparable performance across all three, with the highest consistency. An additional evaluation is performed on a complex data pipeline design task, and we demonstrate that our approach produces higher quality designs across 5 metrics: scalability, reliability, usability, cost-effectiveness, and documentation, plus additional qualitative evidence of the improvements. Our findings suggest that addressing memory limitations through intrinsic approaches can improve the capabilities of multi-agent LLM systems on structured planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01452v2">Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 19 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with zeroth-order (ZO) optimization reduces memory by approximating gradients through function evaluations. However, existing methods essentially perform updates in a one-dimensional space, and suffer from collapse or substantial performance degradation under low-precision training. We introduce BSZO, an adaptive \textbf{B}ayesian \textbf{S}ubspace \textbf{Z}eroth-Order \textbf{O}ptimizer, which applies Kalman filtering to combine finite-difference information across multiple perturbation directions within a subspace. By treating each finite-difference measurement as a noisy observation, BSZO builds a posterior distribution over the subspace-projected gradient and updates it through Bayesian inference, with a residual-based adaptive mechanism to adapt to noise variations. Theoretical analysis shows that BSZO improves the convergence rate by a factor of $k/γ$ compared to standard ZO methods. Experiments on RoBERTa, Mistral, and OPT models show that BSZO outperforms the baselines across various tasks, achieving up to 6.67\% absolute average improvement on OPT-13B while remaining robust under fp16/bf16 precision and keeping memory usage close to inference-only baselines (1.00$\times$--1.08$\times$ of MeZO).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07422v1">Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Despite their impressive capabilities, large language models (LLMs) frequently generate hallucinations. Previous work shows that their internal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct information pathways: (1) a Question-Anchored pathway that depends on question-answer information flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we validate and disentangle these pathways through attention knockout and token patching. Afterwards, we uncover notable and intriguing properties of these two mechanisms. Further experiments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truthfulness, offering directions for more reliable and self-aware generative systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07368v1">Interpretable Text Classification Applied to the Detection of LLM-generated Creative Writing</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted for publication at ICAART 2026 (https://icaart.scitevents.org/?y=2026)
    </div>
    <details class="paper-abstract">
      We consider the problem of distinguishing human-written creative fiction (excerpts from novels) from similar text generated by an LLM. Our results show that, while human observers perform poorly (near chance levels) on this binary classification task, a variety of machine-learning models achieve accuracy in the range 0.93 - 0.98 over a previously unseen test set, even using only short samples and single-token (unigram) features. We therefore employ an inherently interpretable (linear) classifier (with a test accuracy of 0.98), in order to elucidate the underlying reasons for this high accuracy. In our analysis, we identify specific unigram features indicative of LLM-generated text, one of the most important being that the LLM tends to use a larger variety of synonyms, thereby skewing the probability distributions in a manner that is easy to detect for a machine learning classifier, yet very difficult for a human observer. Four additional explanation categories were also identified, namely, temporal drift, Americanisms, foreign language usage, and colloquialisms. As identification of the AI-generated text depends on a constellation of such features, the classification appears robust, and therefore not easy to circumvent by malicious actors intent on misrepresenting AI-generated text as human work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04668v3">Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over 10 rounds, we measure leakage as exact-match recovery of ground-truth PII from attacker outputs. We evaluate six canonical topologies (complete, ring, chain, tree, star, star-ring) across $n\in\{4,5,6\}$, attacker-target placements, and base models. Results are consistent: denser connectivity, shorter attacker-target distance, and higher target centrality increase leakage; most leakage occurs in early rounds and then plateaus; model choice shifts absolute rates but preserves topology ordering; spatiotemporal/location attributes leak more readily than identity credentials or regulated identifiers. We distill practical guidance for system design: favor sparse or hierarchical connectivity, maximize attacker-target separation, and restrict hub/shortcut pathways via topology-aware access control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04136v2">Implementation of transformer-based LLMs with large-scale optoelectronic neurons on a CMOS compatible platform</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      The recent rapid deployment of datacenter infrastructures for performing large language models (LLMs) and related artificial intelligence (AI) applications in the clouds is predicted to incur an exponentially growing energy consumption in the near-term future. In this paper, we propose and analyze the implementation of the transformer model, which is the cornerstone of the modern LLMs, with novel large-scale optoelectronic neurons (OENs) constructed over a complementary metal-oxide-semiconductor (CMOS) compatible platform. With all of the required optoelectronic devices and electronic circuits integrated in a chiplet only about 2 cm by 3 cm in size, 175 billon parameters in the case of GPT-3 are shown to perform inference at an unprecedented speed of 12.6 POPS using only 40 nm CMOS process node, orchestrated by an optoelectronic version of systolic array with no data skew and negligible propagation delay, along with a high power efficiency of 74 TOPS/W and a high area efficiency of 19 TOPS/mm^2. The influence of the quantization formats and the hardware induced errors are numerically investigated, and are shown to have a minimal impact. Our study presents a new yet practical path toward analog neural processing units (NPUs) to complement existing digital processing units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07320v1">Segmental Advantage Estimation: Enhancing PPO for Long-Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) for reasoning tasks is increasingly driven by Reinforcement Learning with Verifiable Rewards (RLVR), where Proximal Policy Optimization (PPO) provides a principled framework for stable policy updates. However, the practical application of PPO is hindered by unreliable advantage estimation in the sparse-reward RLVR regime. This issue arises because the sparse rewards in RLVR lead to inaccurate intermediate value predictions, which in turn introduce significant bias when aggregated at every token by Generalized Advantage Estimation (GAE). To address this, we introduce Segmental Advantage Estimation (SAE), which mitigates the bias that GAE can incur in RLVR. Our key insight is that aggregating $n$-step advantages at every token(as in GAE) is unnecessary and often introduces excessive bias, since individual tokens carry minimal information. Instead, SAE first partitions the generated sequence into coherent sub-segments using low-probability tokens as heuristic boundaries. It then selectively computes variance-reduced advantage estimates only from these information-rich segment transitions, effectively filtering out noise from intermediate tokens. Our experiments demonstrate that SAE achieves superior performance, with marked improvements in final scores, training stability, and sample efficiency. These gains are shown to be consistent across multiple model sizes, and a correlation analysis confirms that our proposed advantage estimator achieves a higher correlation with an approximate ground-truth advantage, justifying its superior performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07309v1">ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 17 pages, 12 figures. Project page: https://arkazhuo.github.io/ARM-homepage/
    </div>
    <details class="paper-abstract">
      Interactive large language model agents have advanced rapidly, but most remain specialized to a single environment and fail to adapt robustly to other environments. Model merging offers a training-free alternative by integrating multiple experts into a single model. In this paper, we propose Agent-Role Merging (ARM), an activation-guided, role-conditioned neuron transplantation method for model merging in LLM agents. ARM improves existing merging methods from static natural language tasks to multi-turn agent scenarios, and over the generalization ability across various interactive environments. This is achieved with a well designed 3-step framework: 1) constructing merged backbones, 2) selection based on its role-conditioned activation analysis, and 3) neuron transplantation for fine-grained refinements. Without gradient-based optimization, ARM improves cross-benchmark generalization while enjoying efficiency. Across diverse domains, the model obtained via ARM merging outperforms prior model merging methods and domain-specific expert models, while demonstrating strong out-of-domain generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24565v2">MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11343v3">SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 AAAI'26 Oral
    </div>
    <details class="paper-abstract">
      The proliferation of high-quality text from Large Language Models (LLMs) demands reliable and efficient detection methods. While existing training-free approaches show promise, they often rely on surface-level statistics and overlook fundamental signal properties of the text generation process. In this work, we reframe detection as a signal processing problem, introducing a novel paradigm that analyzes the sequence of token log-probabilities in the frequency domain. By systematically analyzing the signal's spectral properties using the global Discrete Fourier Transform (DFT) and the local Short-Time Fourier Transform (STFT), we find that human-written text consistently exhibits significantly higher spectral energy. This higher energy reflects the larger-amplitude fluctuations inherent in human writing compared to the suppressed dynamics of LLM-generated text. Based on this key insight, we construct SpecDetect, a detector built on a single, robust feature from the global DFT: DFT total energy. We also propose an enhanced version, SpecDetect++, which incorporates a sampling discrepancy mechanism to further boost robustness. Extensive experiments show that our approach outperforms the state-of-the-art model while running in nearly half the time. Our work introduces a new, efficient, and interpretable pathway for LLM-generated text detection, showing that classical signal processing techniques offer a surprisingly powerful solution to this modern challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07248v1">DarwinTOD: LLM Driven Lifelong Self Evolution for Task Oriented Dialog Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Traditional task-oriented dialog systems are unable to evolve from ongoing interactions or adapt to new domains after deployment, that is a critical limitation in real-world dynamic environments. Continual learning approaches depend on episodic retraining with human curated data, failing to achieve autonomy lifelong improvement. While evolutionary computation and LLM driven self improvement offer promising mechanisms for dialog optimization, they lack a unified framework for holistic, iterative strategy refinement. To bridge this gap, we propose DarwinTOD, a lifelong self evolving dialog framework that systematically integrates these two paradigms, enabling continuous strategy optimization from a zero-shot base without task specific fine-tuning. DarwinTOD maintains an Evolvable Strategy Bank and operates through a dual-loop process: online multi-agent dialog execution with peer critique, and offline structured evolutionary operations that refine the strategy bank using accumulated feedback. This closed-loop design enables autonomous continuous improvement without human intervention. Extensive experiments show that DarwinTOD surpasses previous state-of-the-art methods and exhibits continuous performance gains throughout evolution. Our work provides a novel framework for building dialog systems with lifelong self evolution capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07206v1">LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language model (LLM) routing assigns each query to the most suitable model from an ensemble. We introduce LLMRouterBench, a large-scale benchmark and unified framework for LLM routing. It comprises over 400K instances from 21 datasets and 33 models. Moreover, it provides comprehensive metrics for both performance-oriented routing and performance-cost trade-off routing, and integrates 10 representative routing baselines. Using LLMRouterBench, we systematically re-evaluate the field. While confirming strong model complementarity-the central premise of LLM routing-we find that many routing methods exhibit similar performance under unified evaluation, and several recent approaches, including commercial routers, fail to reliably outperform a simple baseline. Meanwhile, a substantial gap remains to the Oracle, driven primarily by persistent model-recall failures. We further show that backbone embedding models have limited impact, that larger ensembles exhibit diminishing returns compared to careful model curation, and that the benchmark also enables latency-aware analysis. All code and data are available at https://github.com/ynulihao/LLMRouterBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07200v1">Safeguarding LLM Fine-tuning via Push-Pull Distributional Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      The inherent safety alignment of Large Language Models (LLMs) is prone to erosion during fine-tuning, even when using seemingly innocuous datasets. While existing defenses attempt to mitigate this via data selection, they typically rely on heuristic, instance-level assessments that neglect the global geometry of the data distribution and fail to explicitly repel harmful patterns. To address this, we introduce Safety Optimal Transport (SOT), a novel framework that reframes safe fine-tuning from an instance-level filtering challenge to a distribution-level alignment task grounded in Optimal Transport (OT). At its core is a dual-reference ``push-pull'' weight-learning mechanism: SOT optimizes sample importance by actively pulling the downstream distribution towards a trusted safe anchor while simultaneously pushing it away from a general harmful reference. This establishes a robust geometric safety boundary that effectively purifies the training data. Extensive experiments across diverse model families and domains demonstrate that SOT significantly enhances model safety while maintaining competitive downstream performance, achieving a superior safety-utility trade-off compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07197v1">Beyond Variance: Knowledge-Aware LLM Compression via Fisher-Aligned Subspace Diagnostics</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Post-training activation compression is essential for deploying Large Language Models (LLMs) on resource-constrained hardware. However, standard methods like Singular Value Decomposition (SVD) are gradient-blind: they preserve high-variance dimensions regardless of their impact on factual knowledge preservation. We introduce Fisher-Aligned Subspace Compression (FASC), a knowledge-aware compression framework that selects subspaces by directly modeling activation-gradient coupling, minimizing a second-order surrogate of the loss function. FASC leverages the Fisher Information Matrix to identify dimensions critical for factual knowledge, which often reside in low-variance but high-gradient-sensitivity subspaces. We propose the Dependence Violation Score (\r{ho}) as a general-purpose diagnostic metric that quantifies activation-gradient coupling, revealing where factual knowledge is stored within transformer architectures. Extensive experiments on Mistral-7B and Llama-3-8B demonstrate that FASC preserves 6-8% more accuracy on knowledge-intensive benchmarks (MMLU, LAMA) compared to variance-based methods at 50% rank reduction, effectively enabling a 7B model to match the factual recall of a 13B uncompressed model. Our analysis reveals that \r{ho} serves as a fundamental signal of stored knowledge, with high-\r{ho} layers emerging only when models internalize factual associations during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01202v2">Forget BIT, It is All about TOKEN: Towards Semantic Information Theory for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in numerous real-world applications. While the vast majority of research conducted from an experimental perspective is progressing rapidly, it demands substantial computational power, data, and other resources. Therefore, how to open the black-box of LLMs from a theoretical standpoint has become a critical challenge. This paper takes the theory of rate-distortion function, directed information, and Granger causality as its starting point to investigate the information-theoretic principles behind LLMs, leading to the development of semantic information theory for LLMs, where the fundamental unit is token, rather than bits that lacks any semantic meaning. By defining the probabilistic model of LLMs, we discuss structure-agnostic information-theoretic measures, such as the directed rate-distortion function in pre-training, the directed rate-reward function in post-training, and the semantic information flow in inference phase. This paper also delves deeply into the theory of token-level semantic embedding and the information-theoretically optimal vectorization method. Thereafter, we propose a general definition of autoregression LLM, where the Transformer architecture and its performance such as ELBO, generalization error bound, memory capacity, and semantic information measures can be derived theoretically. Other architectures, such as Mamba/Mamba2 and LLaDA, are also discussed in our framework. Consequently, this paper provides a theoretical framework for understanding LLMs from the perspective of semantic information theory, which also offers the necessary theoretical tools for further in-depth research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05635v2">Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Preserving privacy in sensitive data while pretraining large language models on small, domain-specific corpora presents a significant challenge. In this work, we take an exploratory step toward privacy-preserving continual pretraining by proposing an entity-based framework that synthesizes encrypted training data to protect personally identifiable information (PII). Our approach constructs a weighted entity graph to guide data synthesis and applies deterministic encryption to PII entities, enabling LLMs to encode new knowledge through continual pretraining while granting authorized access to sensitive data through decryption keys. Our results on limited-scale datasets demonstrate that our pretrained models outperform base models and ensure PII security, while exhibiting a modest performance gap compared to models trained on unencrypted synthetic data. We further show that increasing the number of entities and leveraging graph-based synthesis improves model performance, and that encrypted models retain instruction-following capabilities with long retrieved contexts. We discuss the security implications and limitations of deterministic encryption, positioning this work as an initial investigation into the design space of encrypted data pretraining for privacy-preserving LLMs. Our code is available at https://github.com/DataArcTech/SoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07190v1">Active Context Compression: Autonomous Memory Management in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 8 pages, 2 figures, 2 tables. IEEE conference format
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents struggle with long-horizon software engineering tasks due to "Context Bloat." As interaction history grows, computational costs explode, latency increases, and reasoning capabilities degrade due to distraction by irrelevant past errors. Existing solutions often rely on passive, external summarization mechanisms that the agent cannot control. This paper proposes Focus, an agent-centric architecture inspired by the biological exploration strategies of Physarum polycephalum (slime mold). The Focus Agent autonomously decides when to consolidate key learnings into a persistent "Knowledge" block and actively withdraws (prunes) the raw interaction history. Using an optimized scaffold matching industry best practices (persistent bash + string-replacement editor), we evaluated Focus on N=5 context-intensive instances from SWE-bench Lite using Claude Haiku 4.5. With aggressive prompting that encourages frequent compression, Focus achieves 22.7% token reduction (14.9M -> 11.5M tokens) while maintaining identical accuracy (3/5 = 60% for both agents). Focus performed 6.0 autonomous compressions per task on average, with token savings up to 57% on individual instances. We demonstrate that capable models can autonomously self-regulate their context when given appropriate tools and prompting, opening pathways for cost-aware agentic systems without sacrificing task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05504v2">Memory Poisoning Attack and Defense on Memory Based LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07160v1">AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 33 pages,7 figures,16 tables
    </div>
    <details class="paper-abstract">
      To meet the ever-increasing demand for computational efficiency, Neural Processing Units (NPUs) have become critical in modern AI infrastructure. However, unlocking their full potential requires developing high-performance compute kernels using vendor-specific Domain-Specific Languages (DSLs), a task that demands deep hardware expertise and is labor-intensive. While Large Language Models (LLMs) have shown promise in general code generation, they struggle with the strict constraints and scarcity of training data in the NPU domain. Our preliminary study reveals that state-of-the-art general-purpose LLMs fail to generate functional complex kernels for Ascend NPUs, yielding a near-zero success rate. To address these challenges, we propose AscendKernelGen, a generation-evaluation integrated framework for NPU kernel development. We introduce Ascend-CoT, a high-quality dataset incorporating chain-of-thought reasoning derived from real-world kernel implementations, and KernelGen-LM, a domain-adaptive model trained via supervised fine-tuning and reinforcement learning with execution feedback. Furthermore, we design NPUKernelBench, a comprehensive benchmark for assessing compilation, correctness, and performance across varying complexity levels. Experimental results demonstrate that our approach significantly bridges the gap between general LLMs and hardware-specific coding. Specifically, the compilation success rate on complex Level-2 kernels improves from 0% to 95.5% (Pass@10), while functional correctness achieves 64.3% compared to the baseline's complete failure. These results highlight the critical role of domain-specific reasoning and rigorous evaluation in automating accelerator-aware code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07122v1">Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.27313v2">LLM generation novelty through the lens of semantic similarity</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Generation novelty is a key indicator of an LLM's ability to generalize, yet measuring it against full pretraining corpora is computationally challenging. Existing evaluations often rely on lexical overlap, failing to detect paraphrased text, or do not consider the full pretraining corpus. We frame novelty as a semantic retrieval problem. This framing enables us to address novelty with modern embedding and indexing pipelines, allowing for efficient analysis at pre-training scale. Specifically, we propose a three-stage framework that retrieves semantically similar samples, reranks them at varying subsequence lengths, and calibrates scores using a human novelty reference for interpretability. We apply this framework to the SmolLM model family and report three key findings: (1) models draw on pre-training data across much longer sequences than previously reported; (2) some task domains systematically promote or suppress generation novelty; and (3) instruction tuning not only alters style but also increases novelty. These results highlight the value of semantic novelty analysis for studying generalization. To support reproducibility and further research, we release ~20 TB of corpus chunks and index artifacts at https://huggingface.co/datasets/stai-tuebingen/faiss-smollm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08045v1">Cognitive Biases in LLM-Assisted Software Development</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 13 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) in software development is transforming programming from a solution-generative to a solution-evaluative activity. This shift opens a pathway for new cognitive challenges that amplify existing decision-making biases or create entirely novel ones. One such type of challenge stems from cognitive biases, which are thinking patterns that lead people away from logical reasoning and result in sub-optimal decisions. How do cognitive biases manifest and impact decision-making in emerging AI-collaborative development? This paper presents the first comprehensive study of cognitive biases in LLM-assisted development. We employ a mixed-methods approach, combining observational studies with 14 student and professional developers, followed by surveys with 22 additional developers. We qualitatively compare categories of biases affecting developers against the traditional non-LLM workflows. Our findings suggest that LLM-related actions are more likely to be associated with novel biases. Through a systematic analysis of 90 cognitive biases specific to developer-LLM interactions, we develop a taxonomy of 15 bias categories validated by cognitive psychologists. We found that 48.8% of total programmer actions are biased, and developer-LLM interactions account for 56.4% of these biased actions. We discuss how these bias categories manifest, present tools and practices for developers, and recommendations for LLM tool builders to help mitigate cognitive biases in human-AI programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08999v2">MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 29 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels-standard execution, confirmation workflow, and Docker isolation - while maintaining backward compatibility with standard MCP clients. However, reliable execution within this framework requires models that can strictly adhere to protocol schemas. To this end, we also fine-tuned the Qwen3 4B and 8B model family on the Agent-Ark/Toucan-1.5M dataset using four Reinforcement Learning techniques: Group Relative Policy Optimization (GRPO), Dr. GRPO, Beta Normalization Policy Optimization (BNPO), and Decoupled Alignment Policy Optimization (DAPO). Evaluated on the MCPToolBench++ benchmark, our optimized model achieves an F1 score of 73.0% that outperforms GPT-OSS-120B (62.17%) and remains competitive with the 70B+ parameter baselines. Evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08012v1">Towards Verifiably Safe Tool Use for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 4 pages, 1 figure; accepted to ICSE NIER 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based AI agents extend LLM capabilities by enabling access to tools such as data sources, APIs, search engines, code sandboxes, and even other agents. While this empowers agents to perform complex tasks, LLMs may invoke unintended tool interactions and introduce risks, such as leaking sensitive data or overwriting critical records, which are unacceptable in enterprise contexts. Current approaches to mitigate these risks, such as model-based safeguards, enhance agents' reliability but cannot guarantee system safety. Methods like information flow control (IFC) and temporal constraints aim to provide guarantees but often require extensive human annotation. We propose a process that starts with applying System-Theoretic Process Analysis (STPA) to identify hazards in agent workflows, derive safety requirements, and formalize them as enforceable specifications on data flows and tool sequences. To enable this, we introduce a capability-enhanced Model Context Protocol (MCP) framework that requires structured labels on capabilities, confidentiality, and trust level. Together, these contributions aim to shift LLM-based agent safety from ad hoc reliability fixes to proactive guardrails with formal guarantees, while reducing dependence on user confirmation and making autonomy a deliberate design choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08003v1">LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with creative generation, and multi-agent frameworks that improve reasoning through interaction can paradoxically hinder creativity by inducing content homogenization. We introduce LLM Review, a peer-review-inspired framework implementing Blind Peer Review: agents exchange targeted feedback while revising independently, preserving divergent creative trajectories. To enable rigorous evaluation, we propose SciFi-100, a science fiction writing dataset with a unified framework combining LLM-as-a-judge scoring, human annotation, and rule-based novelty metrics. Experiments demonstrate that LLM Review consistently outperforms multi-agent baselines, and smaller models with our framework can surpass larger single-agent models, suggesting interaction structure may substitute for model scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08000v1">Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Ensuring that Large Language Models (LLMs) adhere to safety principles without refusing benign requests remains a significant challenge. While OpenAI introduces deliberative alignment (DA) to enhance the safety of its o-series models through reasoning over detailed ``code-like'' safety rules, the effectiveness of this approach in open-source LLMs, which typically lack advanced reasoning capabilities, is understudied. In this work, we systematically evaluate the impact of explicitly specifying extensive safety codes versus demonstrating them through illustrative cases. We find that referencing explicit codes inconsistently improves harmlessness and systematically degrades helpfulness, whereas training on case-augmented simple codes yields more robust and generalized safety behaviors. By guiding LLMs with case-augmented reasoning instead of extensive code-like safety rules, we avoid rigid adherence to narrowly enumerated rules and enable broader adaptability. Building on these insights, we propose CADA, a case-augmented deliberative alignment method for LLMs utilizing reinforcement learning on self-generated safety reasoning chains. CADA effectively enhances harmlessness, improves robustness against attacks, and reduces over-refusal while preserving utility across diverse benchmarks, offering a practical alternative to rule-only DA for improving safety while maintaining helpfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07994v1">DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted (B) to TACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit increased response latency and degraded answer quality as dialogue length grows, making effective context management essential. However, existing methods rely on extra LLM calls to build memory or perform offline memory construction without considering the current user utterance, which can introduce inefficiencies or disrupt conversational continuity. We introduce DyCP, a lightweight context management method that dynamically segment and retrieve relevant memory at query time. It preserves the sequential structure of dialogue without predefined topic boundaries and supports efficient, adaptive context retrieval. Across three long-form dialogue benchmarks, LoCoMo, MT-Bench+, and SCM4LLMs, and multiple LLMs, DyCP consistently improves answer quality while reducing response latency. We also examine the gap between modern LLMs' expanded context windows and their actual long-context processing capacity, highlighting the continued importance of effective context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07972v1">Knowing But Not Doing: Convergent Morality and Divergent Action in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Value alignment is central to the development of safe and socially compatible artificial intelligence. However, how Large Language Models (LLMs) represent and enact human values in real-world decision contexts remains under-explored. We present ValAct-15k, a dataset of 3,000 advice-seeking scenarios derived from Reddit, designed to elicit ten values defined by Schwartz Theory of Basic Human Values. Using both the scenario-based questions and the traditional value questionnaire, we evaluate ten frontier LLMs (five from U.S. companies, five from Chinese ones) and human participants ($n = 55$). We find near-perfect cross-model consistency in scenario-based decisions (Pearson $r \approx 1.0$), contrasting sharply with the broad variability observed among humans ($r \in [-0.79, 0.98]$). Yet, both humans and LLMs show weak correspondence between self-reported and enacted values ($r = 0.4, 0.3$), revealing a systematic knowledge-action gap. When instructed to "hold" a specific value, LLMs' performance declines up to $6.6%$ compared to merely selecting the value, indicating a role-play aversion. These findings suggest that while alignment training yields normative value convergence, it does not eliminate the human-like incoherence between knowing and acting upon values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.20993v2">SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted into 18th Edition of International Conference on Agents and Artificial Intelligence (ICAART)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07935v1">Towards Specialized Generalists: A Multi-Task MoE-LoRA Framework for Domain-Specific LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Models (LLMs) has shifted focus from general-purpose capabilities to domain-specific expertise. However, adapting LLMs to specialized fields such as medicine presents two challenge: (1) the "Stability-Plasticity Dilemma", where the model must acquire complex clinical knowledge without suffering from catastrophic forgetting of general world knowledge; and (2) "Task Interference", where disparate sub-tasks, such as medical diagnosis, report summarization, and drug-drug interaction prediction, compete for limited low-rank parameter space. In this paper, we propose Med-MoE-LoRA, a novel framework that integrates Mixture-of-Experts (MoE) with Low-Rank Adaptation (LoRA) to enable efficient multi-task domain adaptation, especially for medical scenarios. Drawing inspiration from recent advances, our framework employs an asymmetric expert distribution where deeper layers are equipped with a higher density of LoRA experts to capture complex semantic abstractions. We further introduce a "Knowledge-Preservation Plugin", inspired by LoRA MoE, to isolate and protect general-purpose reasoning. By utilizing soft merging with adaptive routing and rank-wise decoupling, Med-MoE-LoRA achieves superior performance in medical benchmarks while reducing interference. Experimental results demonstrate that our approach consistently outperforms standard LoRA and conventional MoE architectures across multiple clinical NLP tasks while retaining the model's general cognitive capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09750v1">SAGE: Tool-Augmented LLM Task Solving Strategies in Scalable Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have proven to work well in question-answering scenarios, but real-world applications often require access to tools for live information or actuation. For this, LLMs can be extended with tools, which are often defined in advance, also allowing for some fine-tuning for specific use cases. However, rapidly evolving software landscapes and individual services require the constant development and integration of new tools. Domain- or company-specific tools can greatly elevate the usefulness of an LLM, but such custom tools can be problematic to integrate, or the LLM may fail to reliably understand and use them. For this, we need strategies to define new tools and integrate them into the LLM dynamically, as well as robust and scalable zero-shot prompting methods that can make use of those tools in an efficient manner. In this paper, we present SAGE, a specialized conversational AI interface, based on the OPACA framework for tool discovery and execution. The integration with OPACA makes it easy to add new tools or services for the LLM to use, while SAGE itself presents rich extensibility and modularity. This not only provides the ability to seamlessly switch between different models (e.g. GPT, LLAMA), but also to add and select prompting methods, involving various setups of differently prompted agents for selecting and executing tools and evaluating the results. We implemented a number of task-solving strategies, making use of agentic concepts and prompting methods in various degrees of complexity, and evaluated those against a comprehensive set of benchmark services. The results are promising and highlight the distinct strengths and weaknesses of different task-solving strategies. Both SAGE and the OPACA framework, as well as the different benchmark services and results, are available as Open Source/Open Data on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07806v1">The Confidence Trap: Gender Bias and Predictive Certainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 AAAI 2026 (AISI Track), Oral. Project page: https://bit.ly/4p8OKQD
    </div>
    <details class="paper-abstract">
      The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs' confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07796v1">Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20443v2">Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Guidance with Proxy Constraint</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods still suffer from over-unlearning due to the lack of a principled mechanism to regulate the forgetting boundary, leading to unnecessary utility degradation and heightened privacy and robustness risks. In this work, we propose EGUP (Entanglement-Guided Unlearning with Proxy Constraint), a novel framework that leverages entanglement and proxy constraint to guide the unlearning process while mitigating over-unlearning. Within each iteration, EGUP employs inter-sample entanglement to adaptively reweight the unlearning strength, assigning greater unlearning efforts to forget samples that are semantically closer to retained knowledge. Across iterations, EGUP leverages intra-sample entanglement to track the representation shift of each forget sample and dynamically adjust its unlearning effort. In addition, we incorporate a proxy constraint that approximates the model's expected outputs after unlearning, forming a reference boundary that softly regularizes the unlearning process. EGUP is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EGUP on the TOFU and MUSE benchmarks, demonstrating consistent improvements in the unlearning-utility trade-off across multiple LLMs. Moreover, EGUP achieves performance close to the retrained model while remaining scalable and robust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.06472v2">KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 24 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1\% LLM-verified correctness and reducing conflict edges by 18.6\% through multi-layer assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25602v2">TRUE: A Reproducible Framework for LLM-Driven Relevance Judgment in Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      LLM-based relevance judgment generation has become a crucial approach in advancing evaluation methodologies in Information Retrieval (IR). It has progressed significantly, often showing high correlation with human judgments as reflected in LLMJudge leaderboards \cite{rahmani2025judging}. However, existing methods for relevance judgments, rely heavily on sensitive prompting strategies, lacking standardized workflows for generating reliable labels. To fill this gap, we reintroduce our method, \textit{Task-aware Rubric-based Evaluation} (TRUE), for relevance judgment generation. Originally developed for usefulness evaluation in search sessions, we extend TRUE to mitigate the gap in relevance judgment due to its demonstrated effectiveness and reproducible workflow. This framework leverages iterative data sampling and reasoning to evaluate relevance judgments across multiple factors including intent, coverage, specificity, accuracy and usefulness. In this paper, we evaluate TRUE on the TREC DL 2019, 2020 and LLMJudge datasets and our results show that TRUE achieves strong performance on the system-ranking LLM leaderboards. The primary focus of this work is to provide a reproducible framework for LLM-based relevance judgments, and we further analyze the effectiveness of TRUE across multiple dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07877v1">E^2-LLM: Bridging Neural Signals and Interpretable Affective Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Emotion recognition from electroencephalography (EEG) signals remains challenging due to high inter-subject variability, limited labeled data, and the lack of interpretable reasoning in existing approaches. While recent multimodal large language models (MLLMs) have advanced emotion analysis, they have not been adapted to handle the unique spatiotemporal characteristics of neural signals. We present E^2-LLM (EEG-to-Emotion Large Language Model), the first MLLM framework for interpretable emotion analysis from EEG. E^2-LLM integrates a pretrained EEG encoder with Qwen-based LLMs through learnable projection layers, employing a multi-stage training pipeline that encompasses emotion-discriminative pretraining, cross-modal alignment, and instruction tuning with chain-of-thought reasoning. We design a comprehensive evaluation protocol covering basic emotion prediction, multi-task reasoning, and zero-shot scenario understanding. Experiments on the dataset across seven emotion categories demonstrate that E^2-LLM achieves excellent performance on emotion classification, with larger variants showing enhanced reliability and superior zero-shot generalization to complex reasoning scenarios. Our work establishes a new paradigm combining physiological signals with LLM reasoning capabilities, showing that model scaling improves both recognition accuracy and interpretable emotional understanding in affective computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05623v3">Deployability-Centric Infrastructure-as-Code Generation: Fail, Learn, Refine, and Succeed through LLM-Empowered DevOps Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted by FSE 2026
    </div>
    <details class="paper-abstract">
      Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions. However, current evaluation focuses on syntactic correctness while ignoring deployability, the critical measure of the utility of IaC configuration files. Six state-of-the-art LLMs performed poorly on deployability, achieving only 20.8$\sim$30.2% deployment success rate on the first attempt. In this paper, we construct DPIaC-Eval, the first deployability-centric IaC template benchmark consisting of 153 real-world scenarios cross 58 unique services. Also, we propose an LLM-based deployability-centric framework, dubbed IaCGen, that uses iterative feedback mechanism encompassing format verification, syntax checking, and live deployment stages, thereby closely mirroring the real DevOps workflows. Results show that IaCGen can make 54.6$\sim$91.6% generated IaC templates from all evaluated models deployable in the first 10 iterations. Additionally, human-in-the-loop feedback that provide direct guidance for the deployability errors, can further boost the performance to over 90% passItr@25 on all evaluated LLMs. Furthermore, we explore the trustworthiness of the generated IaC templates on user intent alignment and security compliance. The poor performance (25.2% user requirement coverage and 8.4% security compliance rate) indicates a critical need for continued research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07084v1">How Secure is Secure Code Generation? Adversarial Prompts Put LLM Defenses to the Test</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Recent secure code generation methods, using vulnerability-aware fine-tuning, prefix-tuning, and prompt optimization, claim to prevent LLMs from producing insecure code. However, their robustness under adversarial conditions remains untested, and current evaluations decouple security from functionality, potentially inflating reported gains. We present the first systematic adversarial audit of state-of-the-art secure code generation methods (SVEN, SafeCoder, PromSec). We subject them to realistic prompt perturbations such as paraphrasing, cue inversion, and context manipulation that developers might inadvertently introduce or adversaries deliberately exploit. To enable fair comparison, we evaluate all methods under consistent conditions, jointly assessing security and functionality using multiple analyzers and executable tests. Our findings reveal critical robustness gaps: static analyzers overestimate security by 7 to 21 times, with 37 to 60% of ``secure'' outputs being non-functional. Under adversarial conditions, true secure-and-functional rates collapse to 3 to 17%. Based on these findings, we propose best practices for building and evaluating robust secure code generation methods. Our code is available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02598v2">LongDA: Benchmarking LLM Agents for Long-Document Data Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21144v2">NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07072v1">Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on retrieving information from external corpora. This creates a new attack surface: indirect prompt injection (IPI), where hidden instructions are planted in the corpora and hijack model behavior once retrieved. Previous studies have highlighted this risk but often avoid the hardest step: ensuring that malicious content is actually retrieved. In practice, unoptimized IPI is rarely retrieved under natural queries, which leaves its real-world impact unclear. We address this challenge by decomposing the malicious content into a trigger fragment that guarantees retrieval and an attack fragment that encodes arbitrary attack objectives. Based on this idea, we design an efficient and effective black-box attack algorithm that constructs a compact trigger fragment to guarantee retrieval for any attack fragment. Our attack requires only API access to embedding models, is cost-efficient (as little as $0.21 per target user query on OpenAI's embedding models), and achieves near-100% retrieval across 11 benchmarks and 8 embedding models (including both open-source models and proprietary services). Based on this attack, we present the first end-to-end IPI exploits under natural queries and realistic external corpora, spanning both RAG and agentic systems with diverse attack objectives. These results establish IPI as a practical and severe threat: when a user issued a natural query to summarize emails on frequently asked topics, a single poisoned email was sufficient to coerce GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent workflow. We further evaluate several defenses and find that they are insufficient to prevent the retrieval of malicious text, highlighting retrieval as a critical open vulnerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20573v2">Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 2.0$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17331v2">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation combines eye-tracking data analysis, including exploratory explainable machine learning analysis with SHAP, and standardized questionnaires (SUS, IPQ, CSQ-VR, NASA-TLX) to examine user experience through both objective gaze-based measures and subjective self-reports of usability, presence, cybersickness, and cognitive load. Our findings show no statistically significant differences in usability, presence, or cybersickness between LLM-driven locomotion and established methods such as teleportation, suggesting its potential as a viable, natural language-based, hands-free alternative. In addition, eye-tracking analysis revealed patterns suggesting tendency toward increased user attention and engagement in the LLM-driven condition. Complementary to these findings, exploratory SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18882v4">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07006v1">LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted as a full paper at the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into human-in-the-loop content moderation systems, a central challenge is deciding when their outputs can be trusted versus when escalation for human review is preferable. We propose a novel framework for supervised LLM uncertainty quantification, learning a dedicated meta-model based on LLM Performance Predictors (LPPs) derived from LLM outputs: log-probabilities, entropy, and novel uncertainty attribution indicators. We demonstrate that our method enables cost-aware selective classification in real-world human-AI workflows: escalating high-risk cases while automating the rest. Experiments across state-of-the-art LLMs, including both off-the-shelf (Gemini, GPT) and open-source (Llama, Qwen), on multimodal and multilingual moderation tasks, show significant improvements over existing uncertainty estimators in accuracy-cost trade-offs. Beyond uncertainty estimation, the LPPs enhance explainability by providing new insights into failure conditions (e.g., ambiguous content vs. under-specified policy). This work establishes a principled framework for uncertainty-aware, scalable, and responsible human-AI moderation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07005v1">MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Log parsing converts semi-structured logs into structured templates, forming a critical foundation for downstream analysis. Traditional syntax and semantic-based parsers often struggle with semantic variations in evolving logs and data scarcity stemming from their limited domain coverage. Recent large language model (LLM)-based parsers leverage in-context learning (ICL) to extract semantics from examples, demonstrating superior accuracy. However, LLM-based parsers face two main challenges: 1) underutilization of ICL capabilities, particularly in dynamic example selection and cross-domain generalization, leading to inconsistent performance; 2) time-consuming and costly LLM querying. To address these challenges, we present MicLog, the first progressive meta in-context learning (ProgMeta-ICL) log parsing framework that combines meta-learning with ICL on small open-source LLMs (i.e., Qwen-2.5-3B). Specifically, MicLog: i) enhances LLMs' ICL capability through a zero-shot to k-shot ProgMeta-ICL paradigm, employing weighted DBSCAN candidate sampling and enhanced BM25 demonstration selection; ii) accelerates parsing via a multi-level pre-query cache that dynamically matches and refines recently parsed templates. Evaluated on Loghub-2.0, MicLog achieves 10.3% higher parsing accuracy than the state-of-the-art parser while reducing parsing time by 42.4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06979v1">MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted to EMNLP 2025 (System Demonstrations)
    </div>
    <details class="paper-abstract">
      The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06973v1">LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v3">Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 The reference list has been updated to reflect recent work
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06966v1">RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) evolve from static dialogue interfaces to autonomous general agents, effective memory is paramount to ensuring long-term consistency. However, existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture **"long-term project-oriented"** interactions where agents must track evolving goals. To bridge this gap, we introduce **RealMem**, the first benchmark grounded in realistic project scenarios. RealMem comprises over 2,000 cross-session dialogues across eleven scenarios, utilizing natural user queries for evaluation. We propose a synthesis pipeline that integrates Project Foundation Construction, Multi-Agent Dialogue Generation, and Memory and Schedule Management to simulate the dynamic evolution of memory. Experiments reveal that current memory systems face significant challenges in managing the long-term project states and dynamic context dependencies inherent in real-world projects. Our code and datasets are available at [https://github.com/AvatarMemory/RealMemBench](https://github.com/AvatarMemory/RealMemBench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06959v1">HAS-VQ: Hessian-Adaptive Sparse Vector Quantization for High-Fidelity LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Post-training quantization is essential for deploying Large Language Models (LLMs) on resource- constrained devices. However, standard integer quantization (e.g., INT4) fundamentally degrades per- formance by imposing a uniform grid on the heavy-tailed distribution of weight parameters, particularly in smaller-scale models (e.g., <2B parameters). We introduce HAS-VQ (Hessian-Adaptive Sparse Vec- tor Quantization), a compression framework that strictly decouples high-sensitivity outliers from the bulk weight distribution using second-order sensitivity analysis. HAS-VQ employs a Hessian-Masked Decoupling strategy to isolate sensitive parameters, followed by robust Vector Quantization (VQ) of the remaining dense body. Crucially, we introduce a residual sparse feedback mechanism that corrects quan- tization errors in the most sensitive dimensions, ensuring exact reconstruction of outliers. We evaluate HAS-VQ on SmolLM2-1.7B, demonstrating two distinct regimes of superiority: (1) Pareto Dominance over Integer Baselines: At 4.23 effective bits-per-parameter (BPP), we achieve a perplexity of 14.23, significantly outperforming the standard INT4 baseline (20.03 PPL at 4.71 BPP). (2) High-Fidelity Compression: Relative to the FP16 baseline, HAS-VQ achieves a 2.3x reduction in model size (7.03 BPP) while maintaining statistically indistinguishable perplexity (10.12 vs. 10.04), effectively offering a lossless compression alternative for bandwidth-constrained environments. The code is available at https://github.com/VladimerKhasia/HASVQ
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06914v1">Towards Compositional Generalization in LLMs for Smart Contract Security: A Case Study on Reentrancy Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities in natural language understanding and generation. Despite being trained on large-scale, high-quality data, LLMs still fail to outperform traditional static analysis tools in specialized domains like smart contract vulnerability detection. To address this issue, this paper proposes a post-training algorithm based on atomic task decomposition and fusion. This algorithm aims to achieve combinatorial generalization under limited data by decomposing complex reasoning tasks. Specifically, we decompose the reentrancy vulnerability detection task into four linearly independent atomic tasks: identifying external calls, identifying state updates, identifying data dependencies between external calls and state updates, and determining their data flow order. These tasks form the core components of our approach. By training on synthetic datasets, we generate three compiler-verified datasets. We then employ the Slither tool to extract structural information from the control flow graph and data flow graph, which is used to fine-tune the LLM's adapter. Experimental results demonstrate that low-rank normalization fusion with the LoRA adapter improves the LLM's reentrancy vulnerability detection accuracy to 98.2%, surpassing state-of-the-art methods. On 31 real-world contracts, the algorithm achieves a 20% higher recall than traditional analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08870v2">Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      LLM agents are widely deployed in complex interactive tasks, yet privacy constraints often preclude centralized optimization and co-evolution across dynamic environments. Despite the demonstrated success of Federated Learning (FL) on static datasets, its effectiveness in open-ended, self-evolving agent systems remains largely unexplored. In such settings, the direct application of standard FL is particularly challenging, as heterogeneous tasks and sparse, trajectory-level reward signals give rise to severe gradient instability, which undermines the global optimization process. To bridge this gap, we propose Fed-SE, a Federated Self-Evolution framework for LLM agents that establishes a local evolution-global aggregation paradigm. Locally, agents employ parameter-efficient fine-tuning on filtered, high-return trajectories to achieve stable gradient updates. Globally, Fed-SE aggregates updates within a low-rank subspace, reducing communication cost across clients. Experiments across five heterogeneous environments demonstrate that Fed-SE improves average task success rates by 10\% over the state-of-the-art FedIT, validating its effectiveness in cross-environment knowledge transfer under privacy constraints.
    </details>
</div>
