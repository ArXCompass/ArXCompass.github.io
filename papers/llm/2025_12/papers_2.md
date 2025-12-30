# llm - 2025_12

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21250v1">CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21236v1">Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted to FSE 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00675v3">Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Memory is fundamental to large language model (LLM)-based agents, but existing surveys emphasize application-level use (e.g., personalized dialogue), while overlooking the atomic operations governing memory dynamics. This work categorizes memory into parametric (implicit in model weights) and contextual (explicit external data, structured/unstructured) forms, and defines six core operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Condensation. Mapping these dimensions reveals four key research topics: long-term, long-context, parametric modification, and multi-source memory. The taxonomy provides a structured view of memory-related research, benchmarks, and tools, clarifying functional interactions in LLM-based agents and guiding future advancements. The datasets, papers, and tools are publicly available at https://github.com/Elvin-Yiming-Du/Survey_Memory_in_AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16378v2">Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Project available at https://github.com/sarapapi/hearing2translate
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12867v2">Bootstrapping LLMs via Preference-Based Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Bootstrapping large language models (LLMs) through preference-based policy optimization offers a promising direction for aligning model behavior with human preferences without relying on extensive manual annotations. In this work, we propose a novel preference-based policy optimization (PbPO) framework that formulates the learning process as a min-max game between the main policy and a reward model (RM). The RM is constrained within a confidence set derived from preference data to ensure reliable exploitation. Our iterative online algorithm actively collects preference data through guided exploration of the evolving policy, enabling continual self-improvement of both the policy and the RM. We provide theoretical guarantees for our method, establishing high-probability regret bounds for both settings with sequence-level RM and token-level RM, demonstrating its effectiveness in bootstrapping LLMs. Extensive experiments on five benchmarks show that our approach consistently outperforms existing state-of-the-art preference optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19027v2">SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 An updated post-print on the paper published in the Proceedings of the 19th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM '25). 12 pages (10 + 2 pages for references)
    </div>
    <details class="paper-abstract">
      Background: The use of large language models (LLMs) in the title-abstract screening process of systematic reviews (SRs) has shown promising results, but suffers from limited performance evaluation. Aims: Create a benchmark dataset to evaluate the performance of LLMs in the title-abstract screening process of SRs. Provide evidence whether using LLMs in title-abstract screening in software engineering is advisable. Method: We start with 169 SR research artifacts and find 24 of those to be suitable for inclusion in the dataset. Using the dataset we benchmark title-abstract screening using 9 LLMs. Results: We present the SESR-Eval (Software Engineering Systematic Review Evaluation) dataset containing 34,528 labeled primary studies, sourced from 24 secondary studies published in software engineering (SE) journals. Most LLMs performed similarly and the differences in screening accuracy between secondary studies are greater than differences between LLMs. The cost of using an LLM is relatively low - less than $40 per secondary study even for the most expensive model. Conclusions: Our benchmark enables monitoring AI performance in the screening task of SRs in software engineering. At present, LLMs are not yet recommended for automating the title-abstract screening process, since accuracy varies widely across secondary studies, and no LLM managed a high recall with reasonable precision. In future, we plan to investigate factors that influence LLM screening performance between studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21144v1">Encrypted Traffic Detection in Resource Constrained IoT Networks: A Diffusion Model and LLM Integrated Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 This paper is accepted by IEEE Transactions on Network Science and Engineering
    </div>
    <details class="paper-abstract">
      The proliferation of Internet-of-things (IoT) infrastructures and the widespread adoption of traffic encryption present significant challenges, particularly in environments characterized by dynamic traffic patterns, constrained computational capabilities, and strict latency constraints. In this paper, we propose DMLITE, a diffusion model and large language model (LLM) integrated traffic embedding framework for network traffic detection within resource-limited IoT environments. The DMLITE overcomes these challenges through a tri-phase architecture including traffic visual preprocessing, diffusion-based multi-level feature extraction, and LLM-guided feature optimization. Specifically, the framework utilizes self-supervised diffusion models to capture both fine-grained and abstract patterns in encrypted traffic through multi-level feature fusion and contrastive learning with representative sample selection, thus enabling rapid adaptation to new traffic patterns with minimal labeled data. Furthermore, DMLITE incorporates LLMs to dynamically adjust particle swarm optimization parameters for intelligent feature selection by implementing a dual objective function that minimizes both classification error and variance across data distributions. Comprehensive experimental validation on benchmark datasets confirms the effectiveness of DMLITE, achieving classification accuracies of 98.87\%, 92.61\%, and 99.83\% on USTC-TFC, ISCX-VPN, and Edge-IIoTset datasets, respectively. This improves classification accuracy by an average of 3.7\% and reduces training time by an average of 41.9\% compared to the representative deep learning model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21138v1">Emotion Diffusion in Real and Simulated Social Graphs: Structural Limits of LLM-Based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Understanding how emotions diffuse through social networks is central to computational social science. Recently, large language models (LLMs) have been increasingly used to simulate social media interactions, raising the question of whether LLM-generated data can realistically reproduce emotion diffusion patterns observed in real online communities. In this study, we conduct a systematic comparison between emotion diffusion in real-world social graphs and in LLM-simulated interaction networks. We construct diffusion graphs from Reddit discussion data and compare them with synthetic social graphs generated through LLM-driven conversational simulations. Emotion states are inferred using established sentiment analysis pipelines, and both real and simulated graphs are analyzed from structural, behavioral, and predictive perspectives. Our results reveal substantial structural and dynamic discrepancies between real and simulated diffusion processes. Real-world emotion diffusion exhibits dense connectivity, repeated interactions, sentiment shifts, and emergent community structures, whereas LLM-simulated graphs largely consist of isolated linear chains with monotonic emotional trajectories. These structural limitations significantly affect downstream tasks such as graph-based emotion prediction, leading to reduced emotional diversity and class imbalance in simulated settings. Our findings highlight current limitations of LLM-based social simulation in capturing the interactive complexity and emotional heterogeneity of real social networks. This work provides empirical evidence for the cautious use of LLM-generated data in social science research and suggests directions for improving future simulation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21127v1">A Real-World Evaluation of LLM Medication Safety Reviews in NHS Primary Care</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often match or exceed clinician-level performance on medical benchmarks, yet very few are evaluated on real clinical data or examined beyond headline metrics. We present, to our knowledge, the first evaluation of an LLM-based medication safety review system on real NHS primary care data, with detailed characterisation of key failure behaviours across varying levels of clinical complexity. In a retrospective study using a population-scale EHR spanning 2,125,549 adults in NHS Cheshire and Merseyside, we strategically sampled patients to capture a broad range of clinical complexity and medication safety risk, yielding 277 patients after data-quality exclusions. An expert clinician reviewed these patients and graded system-identified issues and proposed interventions. Our primary LLM system showed strong performance in recognising when a clinical issue is present (sensitivity 100\% [95\% CI 98.2--100], specificity 83.1\% [95\% CI 72.7--90.1]), yet correctly identified all issues and interventions in only 46.9\% [95\% CI 41.1--52.8] of patients. Failure analysis reveals that, in this setting, the dominant failure mechanism is contextual reasoning rather than missing medication knowledge, with five primary patterns: overconfidence in uncertainty, applying standard guidelines without adjusting for patient context, misunderstanding how healthcare is delivered in practice, factual errors, and process blindness. These patterns persisted across patient complexity and demographic strata, and across a range of state-of-the-art models and configurations. We provide 45 detailed vignettes that comprehensively cover all identified failure cases. This work highlights shortcomings that must be addressed before LLM-based clinical AI can be safely deployed. It also begs larger-scale, prospective evaluations and deeper study of LLM behaviours in clinical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11822v2">Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      New Large Language Models (LLMs) become available every few weeks, and modern application developers confronted with the unenviable task of having to decide if they should switch to a new model. While human evaluation remains the gold standard, it is costly and unscalable. The state-of-the-art approach is to use LLMs as evaluators ( LLM-as-a-judge), but this suffers from a critical flaw: LLMs exhibit a strong positive bias. We provide empirical evidence showing that while LLMs can identify valid outputs with high accuracy (i.e., True Positive Rate 96%), they are remarkably poor at identifying invalid ones (i.e., True Negative Rate <25%). This systematic bias, coupled with class imbalance, often leads to inflated reliability scores. While ensemble-based methods like majority voting can help, we show that they are not good enough. We introduce an optimal minority-veto strategy that is resilient to missing data and mitigates this bias to a large extent. For scenarios requiring even higher precision, we propose a novel regression-based framework that directly models the validator bias using a small set of human-annotated ground truth data. On a challenging code feedback task over 366 high-school Python programs, our regression approach reduces the maximum absolute error to just 1.2%, achieving a 2x improvement over the best-performing ensemble of 14 state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21106v1">Semantic Refinement with LLMs for Graph Representations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Graph-structured data exhibit substantial heterogeneity in where their predictive signals originate: in some domains, node-level semantics dominate, while in others, structural patterns play a central role. This structure-semantics heterogeneity implies that no graph learning model with a fixed inductive bias can generalize optimally across diverse graph domains. However, most existing methods address this challenge from the model side by incrementally injecting new inductive biases, which remains fundamentally limited given the open-ended diversity of real-world graphs. In this work, we take a data-centric perspective and treat node semantics as a task-adaptive variable. We propose a Data-Adaptive Semantic Refinement framework DAS for graph representation learning, which couples a fixed graph neural network (GNN) and a large language model (LLM) in a closed feedback loop. The GNN provides implicit supervisory signals to guide the semantic refinement of LLM, and the refined semantics are fed back to update the same graph learner. We evaluate our approach on both text-rich and text-free graphs. Results show consistent improvements on structure-dominated graphs while remaining competitive on semantics-rich graphs, demonstrating the effectiveness of data-centric semantic adaptation under structure-semantics heterogeneity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21080v1">LLM Personas as a Substitute for Field Experiments in Method Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Field experiments (A/B tests) are often the most credible benchmark for methods in societal systems, but their cost and latency create a major bottleneck for iterative method development. LLM-based persona simulation offers a cheap synthetic alternative, yet it is unclear whether replacing humans with personas preserves the benchmark interface that adaptive methods optimize against. We prove an if-and-only-if characterization: when (i) methods observe only the aggregate outcome (aggregate-only observation) and (ii) evaluation depends only on the submitted artifact and not on the algorithm's identity or provenance (algorithm-blind evaluation), swapping humans for personas is just panel change from the method's point of view, indistinguishable from changing the evaluation population (e.g., New York to Jakarta). Furthermore, we move from validity to usefulness: we define an information-theoretic discriminability of the induced aggregate channel and show that making persona benchmarking as decision-relevant as a field experiment is fundamentally a sample-size question, yielding explicit bounds on the number of independent persona evaluations required to reliably distinguish meaningfully different methods at a chosen resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21041v1">When LLMs fall short in Deductive Coding: Model Comparison and Human AI Collaboration Workflow Design</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 24 pages (8 pages for Appendix), 4 figures, for Learning Analytics & Knowledge Conference to be held in 2026, Norway (LAK26)
    </div>
    <details class="paper-abstract">
      With generative artificial intelligence driving the growth of dialogic data in education, automated coding is a promising direction for learning analytics to improve efficiency. This surge highlights the need to understand the nuances of student-AI interactions, especially those rare yet crucial. However, automated coding may struggle to capture these rare codes due to imbalanced data, while human coding remains time-consuming and labour-intensive. The current study examined the potential of large language models (LLMs) to approximate or replace humans in deductive, theory-driven coding, while also exploring how human-AI collaboration might support such coding tasks at scale. We compared the coding performance of small transformer classifiers (e.g., BERT) and LLMs in two datasets, with particular attention to imbalanced head-tail distributions in dialogue codes. Our results showed that LLMs did not outperform BERT-based models and exhibited systematic errors and biases in deductive coding tasks. We designed and evaluated a human-AI collaborative workflow that improved coding efficiency while maintaining coding reliability. Our findings reveal both the limitations of LLMs -- especially their difficulties with semantic similarity and theoretical interpretations and the indispensable role of human judgment -- while demonstrating the practical promise of human-AI collaborative workflows for coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21028v1">Artificial or Just Artful? Do LLMs Bend the Rules in Programming?</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for automated code generation, yet their apparent successes often mask a tension between pretraining objectives and alignment choices. While pretraining encourages models to exploit all available signals to maximize success, alignment, whether through fine-tuning or prompting, may restrict their use. This conflict is especially salient in agentic AI settings, for instance when an agent has access to unit tests that, although intended for validation, act as strong contextual signals that can be leveraged regardless of explicit prohibitions. In this paper, we investigate how LLMs adapt their code generation strategies when exposed to test cases under different prompting conditions. Using the BigCodeBench (Hard) dataset, we design five prompting conditions that manipulate test visibility and impose explicit or implicit restrictions on their use. We evaluate five LLMs (four open-source and one closed-source) across correctness, code similarity, program size, and code churn, and analyze cross-model consistency to identify recurring adaptation strategies. Our results show that test visibility dramatically alters performance, correctness nearly doubles for some models, while explicit restrictions or partial exposure only partially mitigate this effect. Beyond raw performance, we identify four recurring adaptation strategies, with test-driven refinement emerging as the most frequent. These results highlight how LLMs adapt their behavior when exposed to contextual signals that conflict with explicit instructions, providing useful insight into how models reconcile pretraining objectives with alignment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v3">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 14 pages, 20 figures, conference, accepted by HPCA 2026
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21017v1">Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), the Chain-of-Thought (CoT) component has become significant for complex reasoning tasks. However, in conventional Supervised Fine-Tuning (SFT), the model could allocate disproportionately more attention to CoT sequences with excessive length. This reduces focus on the much shorter but essential Key portion-the final answer, whose correctness directly determines task success and evaluation quality. To address this limitation, we propose SFTKey, a two-stage training scheme. In the first stage, conventional SFT is applied to ensure proper output format, while in the second stage, only the Key portion is fine-tuned to improve accuracy. Extensive experiments across multiple benchmarks and model families demonstrate that SFTKey achieves an average accuracy improvement exceeding 5\% over conventional SFT, while preserving the ability to generate correct formats. Overall, this study advances LLM fine-tuning by explicitly balancing CoT learning with additional optimization on answer-relevant tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21010v1">LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) and diverse specialized benchmarks necessitates a shift from fragmented, task-specific metrics to a holistic, competitive ranking system that effectively aggregates performance across multiple ability dimensions. Primarily using static scoring, current evaluation methods are fundamentally limited. They struggle to determine the proper mix ratio across diverse benchmarks, and critically, they fail to capture a model's dynamic competitive fitness or its vulnerability when confronted with sequential, high-stakes tasks. To address this, we introduce the novel Competitive Swiss-System Dynamics (CSD) framework. CSD simulates a multi-round, sequential contest where models are dynamically paired across a curated sequence of benchmarks based on their accumulated win-loss record. And Monte Carlo Simulation ($N=100,000$ iterations) is used to approximate the statistically robust Expected Win Score ($E[S_m]$), which eliminates the noise of random pairing and early-round luck. Furthermore, we implement a Failure Sensitivity Analysis by parameterizing the per-round elimination quantity ($T_k$), which allows us to profile models based on their risk appetite--distinguishing between robust generalists and aggressive specialists. We demonstrate that CSD provides a more nuanced and context-aware ranking than traditional aggregate scoring and static pairwise models, representing a vital step towards risk-informed, next-generation LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21008v1">GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted by USENIX Security'26
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness. In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20997v1">LLM-Empowered Agentic AI for QoE-Aware Network Slicing Management in Industrial IoT</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The Industrial Internet of Things (IIoT) requires networks that deliver ultra-low latency, high reliability, and cost efficiency, which traditional optimization methods and deep reinforcement learning (DRL)-based approaches struggle to provide under dynamic and heterogeneous workloads. To address this gap, large language model (LLM)-empowered agentic AI has emerged as a promising paradigm, integrating reasoning, planning, and adaptation to enable QoE-aware network management. In this paper, we explore the integration of agentic AI into QoE-aware network slicing for IIoT. We first review the network slicing management architecture, QoE metrics for IIoT applications, and the challenges of dynamically managing heterogeneous network slices, while highlighting the motivations and advantages of adopting agentic AI. We then present the workflow of agentic AI-based slicing management, illustrating the full lifecycle of AI agents from processing slice requests to constructing slice instances and performing dynamic adjustments. Furthermore, we propose an LLM-empowered agentic AI approach for slicing management, which integrates a retrieval-augmented generation (RAG) module for semantic intent inference, a DRL-based orchestrator for slicing configuration, and an incremental memory mechanism for continual learning and adaptation. Through a case study on heterogeneous slice management, we demonstrate that the proposed approach significantly outperforms other baselines in balancing latency, reliability, and cost, and achieves up to a 19% improvement in slice availability ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20986v1">AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wearable sensing is creating a new class of mobile applications capable of nuanced human activity understanding. However, the reliability of these systems is critically undermined by their vulnerability to prompt injection attacks, where attackers deliberately input deceptive instructions into LLMs. Traditional defenses, based on static filters and rigid rules, are insufficient to address the semantic complexity of these new attacks. We argue that a paradigm shift is needed -- from passive filtering to active protection and autonomous reasoning. We introduce AegisAgent, an autonomous agent system designed to ensure the security of LLM-driven HAR systems. Instead of merely blocking threats, AegisAgent functions as a cognitive guardian. It autonomously perceives potential semantic inconsistencies, reasons about the user's true intent by consulting a dynamic memory of past interactions, and acts by generating and executing a multi-step verification and repair plan. We implement AegisAgent as a lightweight, full-stack prototype and conduct a systematic evaluation on 15 common attacks with five state-of-the-art LLM-based HAR systems on three public datasets. Results show it reduces attack success rate by 30\% on average while incurring only 78.6 ms of latency overhead on a GPU workstation. Our work makes the first step towards building secure and trustworthy LLM-driven HAR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20983v1">Automatic Replication of LLM Mistakes in Medical Conversations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 48 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly evaluated in clinical settings using multi-dimensional rubrics which quantify reasoning quality, safety, and patient-centeredness. Yet, replicating specific mistakes in other LLM models is not straightforward and often requires manual effort. We introduce MedMistake, an automatic pipeline that extracts mistakes LLMs make in patient-doctor conversations and converts them into a benchmark of single-shot QA pairs. Our pipeline (1) creates complex, conversational data between an LLM patient and LLM doctor, (2) runs an evaluation with a committee of 2 LLM judges across a variety of dimensions and (3) creates simplified single-shot QA scenarios from those mistakes. We release MedMistake-All, a dataset of 3,390 single-shot QA pairs where GPT-5 and Gemini 2.5 Pro are currently failing to answer correctly, as judged by two LLM judges. We used medical experts to validate a subset of 211/3390 questions (MedMistake-Bench), which we used to run a final evaluation of 12 frontier LLMs: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek-Chat, Gemini 2.5 Pro, Gemini 3 Pro, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Grok 4, Grok 4.1, Mistral Large. We found that GPT models, Claude and Grok obtained the best performance on MedMistake-Bench. We release both the doctor-validated benchmark (MedMistake-Bench), as well as the full dataset (MedMistake-All) at https://huggingface.co/datasets/TheLumos/MedicalMistakeBenchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20975v1">SPOT!: Map-Guided LLM Agent for Unsupervised Multi-CCTV Dynamic Object Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 33 pages, 27figures
    </div>
    <details class="paper-abstract">
      CCTV-based vehicle tracking systems face structural limitations in continuously connecting the trajectories of the same vehicle across multiple camera environments. In particular, blind spots occur due to the intervals between CCTVs and limited Fields of View (FOV), which leads to object ID switching and trajectory loss, thereby reducing the reliability of real-time path prediction. This paper proposes SPOT (Spatial Prediction Over Trajectories), a map-guided LLM agent capable of tracking vehicles even in blind spots of multi-CCTV environments without prior training. The proposed method represents road structures (Waypoints) and CCTV placement information as documents based on 2D spatial coordinates and organizes them through chunking techniques to enable real-time querying and inference. Furthermore, it transforms the vehicle's position into the actual world coordinate system using the relative position and FOV information of objects observed in CCTV images. By combining map spatial information with the vehicle's moving direction, speed, and driving patterns, a beam search is performed at the intersection level to derive candidate CCTV locations where the vehicle is most likely to enter after the blind spot. Experimental results based on the CARLA simulator in a virtual city environment confirmed that the proposed method accurately predicts the next appearing CCTV even in blind spot sections, maintaining continuous vehicle trajectories more effectively than existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v1">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02080v2">Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content. In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety. Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance the security. Our study evaluates both open-source (e.g., LLaMA and Mistral) and closed-source models (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20926v1">Uncovering Hierarchical Structure in LLM Embeddings with $δ$-Hyperbolicity, Ultrametricity, and Neighbor Joining</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled significant strides in various fields. This paper introduces a novel approach to evaluate the effectiveness of LLM embeddings in the context of inherent geometric properties. We investigate the structural properties of these embeddings through three complementary metrics $δ$-hyperbolicity, Ultrametricity, and Neighbor Joining. $δ$-hyperbolicity, a measure derived from geometric group theory, quantifies how much a metric space deviates from being a tree-like structure. In contrast, ultrametricity characterizes strictly hierarchical structures where distances obey a strong triangle inequality. While Neighbor Joining quantifies how tree-like the distance relationships are, it does so specifically with respect to the tree reconstructed by the Neighbor Joining algorithm. By analyzing the embeddings generated by LLMs using these metrics, we uncover to what extent the embedding space reflects an underlying hierarchical or tree-like organization. Our findings reveal that LLM embeddings exhibit varying degrees of hyperbolicity and ultrametricity, which correlate with their performance in the underlying machine learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20920v1">RevFFN: Memory-Efficient Full-Parameter Fine-Tuning of Mixture-of-Experts LLMs with Reversible Blocks</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Full parameter fine tuning is a key technique for adapting large language models (LLMs) to downstream tasks, but it incurs substantial memory overhead due to the need to cache extensive intermediate activations for backpropagation. This bottleneck makes full fine tuning of contemporary large scale LLMs challenging in practice. Existing distributed training frameworks such as DeepSpeed alleviate this issue using techniques like ZeRO and FSDP, which rely on multi GPU memory or CPU offloading, but often require additional hardware resources and reduce training speed. We introduce RevFFN, a memory efficient fine tuning paradigm for mixture of experts (MoE) LLMs. RevFFN employs carefully designed reversible Transformer blocks that allow reconstruction of layer input activations from outputs during backpropagation, eliminating the need to store most intermediate activations in memory. While preserving the expressive capacity of MoE architectures, this approach significantly reduces peak memory consumption for full parameter fine tuning. As a result, RevFFN enables efficient full fine tuning on a single consumer grade or server grade GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20082v2">Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted in CODS 2025
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00617v2">ART: Adaptive Response Tuning Framework -- A Multi-Agent Tournament-Based Approach to LLM Response Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 14 pages, 11 figures, 5 tables. IEEE conference-style paper with appendices
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, single-model responses often exhibit inconsistencies, hallucinations, and varying quality across different query domains. This paper presents ART (Adaptive Response Tuning), a novel framework that employs tournament-style ELO ranking and multi-agent reasoning to systematically optimize LLM outputs. By enabling multiple LLM agents to compete, critique, and collaborate through structured tournament workflows, ART produces consensus responses that outperform individual model outputs. Our framework introduces configurable tournament parameters, dynamic agent selection, and multiple consensus fusion strategies. Experimental evaluations demonstrate significant improvements in response accuracy, coherence, and reliability compared to baseline single-model approaches. The ART framework provides a scalable, production-ready solution for applications requiring high-quality, vetted LLM responses, achieving an 8.4% improvement in overall quality metrics and R^2 values exceeding 0.96 in ELO rating convergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20900v1">When Experts Speak:Sequential LLM-Bayesian Learning for Startup Success Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Evaluating startups is inherently challenging in entrepreneurial finance, where investors confront severe information asymmetry and limited quantitative data. Leveraging a novel expert network call data, we develop an LLM-Bayesian model that analyzes these conversations at the question-answer turn level, extracting semantic and evaluative signals via large language models (LLMs) and aggregating them in a sequential Bayesian architecture. The model dynamically updates beliefs as additional expert calls occur and attenuates contradictory assessments, which are absent from existing text-based screening tools. Empirically, our model outperforms state-of-the-art benchmarks by 6.691% in F1-score and increases portfolio-level Return on Investment by 15.255%. Attention and ablation analyses reveal that conversational cues are particularly informative for technologically complex startups, young firms, diverse founding teams, and firms with low public visibility. By converting expert dialogue into continually updated probabilities, our model advances research in entrepreneurial finance and information systems and offers policy implications for improving funding outcomes for informationally disadvantaged startups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20884v1">The Silent Scholar Problem: A Probabilistic Framework for Breaking Epistemic Asymmetry in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Autonomous agents powered by LLMs and Retrieval-Augmented Generation (RAG) are proficient consumers of digital content but remain unidirectional, a limitation we term epistemic asymmetry. This isolation leads to redundant reasoning and stagnates collective intelligence. Current self-reflection frameworks remain largely heuristic and private, lacking a probabilistic foundation to quantify certainty or justify external interaction.To bridge this gap, we propose a formal probabilistic framework that provides agents with a non-altruistic motive for bidirectional knowledge exchange. We model an agent's belief in a proposition using a Beta-Bernoulli distribution with a forgetting factor ($γ$). This allows us to isolate epistemic uncertainty as the variance of belief, establishing a dual drive for interaction: A homeostatic motive: The need to maintain certainty against the temporal decay introduced by $γ$. An optimal learning strategy: Targeting points of maximum ambiguity ($\mathbb{E}[θ]=0.5$) to maximize information gain. Under this framework, public contribution is reframed as optimal active learning: sharing solutions to elicit feedback is the most efficient method for an agent to reduce its own uncertainty. To ensure scalability, we introduce epistemic caching, which leverages the forgetting factor to dynamically prioritize resources for the active head of non-stationary knowledge distributions. Finally, we demonstrate how these accumulated belief states serve as verifiable reward signals for Reinforcement Learning from Human Feedback (RLHF) and high-quality data filters for Supervised Fine-Tuning (SFT). Simulation results validate that this uncertainty-driven strategy significantly outperforms random baselines in heterogeneous (Zipfian) environments, maintaining high adaptability to concept drift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21422v1">Teaching People LLM's Errors and Getting it Right</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      People use large language models (LLMs) when they should not. This is partly because they see LLMs compose poems and answer intricate questions, so they understandably, but incorrectly, assume LLMs won't stumble on basic tasks like simple arithmetic. Prior work has tried to address this by clustering instance embeddings into regions where an LLM is likely to fail and automatically describing patterns in these regions. The found failure patterns are taught to users to mitigate their overreliance. Yet, this approach has not fully succeeded. In this analysis paper, we aim to understand why. We first examine whether the negative result stems from the absence of failure patterns. We group instances in two datasets by their meta-labels and evaluate an LLM's predictions on these groups. We then define criteria to flag groups that are sizable and where the LLM is error-prone, and find meta-label groups that meet these criteria. Their meta-labels are the LLM's failure patterns that could be taught to users, so they do exist. We next test whether prompting and embedding-based approaches can surface these known failures. Without this, users cannot be taught about them to reduce their overreliance. We find mixed results across methods, which could explain the negative result. Finally, we revisit the final metric that measures teaching effectiveness. We propose to assess a user's ability to effectively use the given failure patterns to anticipate when an LLM is error-prone. A user study shows a positive effect from teaching with this metric, unlike the human-AI team accuracy. Our findings show that teaching failure patterns could be a viable approach to mitigating overreliance, but success depends on better automated failure-discovery methods and using metrics like ours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21404v1">LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22266v1">LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The widespread application of Large Language Models (LLMs) has motivated a growing interest in their capacity for processing dynamic graphs. Temporal motifs, as an elementary unit and important local property of dynamic graphs which can directly reflect anomalies and unique phenomena, are essential for understanding their evolutionary dynamics and structural features. However, leveraging LLMs for temporal motif analysis on dynamic graphs remains relatively unexplored. In this paper, we systematically study LLM performance on temporal motif-related tasks. Specifically, we propose a comprehensive benchmark, LLMTM (Large Language Models in Temporal Motifs), which includes six tailored tasks across nine temporal motif types. We then conduct extensive experiments to analyze the impacts of different prompting techniques and LLMs (including nine models: openPangu-7B, the DeepSeek-R1-Distill-Qwen series, Qwen2.5-32B-Instruct, GPT-4o-mini, DeepSeek-R1, and o3) on model performance. Informed by our benchmark findings, we develop a tool-augmented LLM agent that leverages precisely engineered prompts to solve these tasks with high accuracy. Nevertheless, the high accuracy of the agent incurs a substantial cost. To address this trade-off, we propose a simple yet effective structure-aware dispatcher that considers both the dynamic graph's structural properties and the LLM's cognitive load to intelligently dispatch queries between the standard LLM prompting and the more powerful agent. Our experiments demonstrate that the structure-aware dispatcher effectively maintains high accuracy while reducing cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22250v1">Hallucination Detection for LLM-based Text-to-SQL Generation via Two-Stage Metamorphic Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      In Text-to-SQL generation, large language models (LLMs) have shown strong generalization and adaptability. However, LLMs sometimes generate hallucinations, i.e.,unrealistic or illogical content, which leads to incorrect SQL queries and negatively impacts downstream applications. Detecting these hallucinations is particularly challenging. Existing Text-to-SQL error detection methods, which are tailored for traditional deep learning models, face significant limitations when applied to LLMs. This is primarily due to the scarcity of ground-truth data. To address this challenge, we propose SQLHD, a novel hallucination detection method based on metamorphic testing (MT) that does not require standard answers. SQLHD splits the detection task into two sequentiial stages: schema-linking hallucination detection via eight structure-aware Metamorphic Relations (MRs) that perturb comparative words, entities, sentence structure or database schema, and logical-synthesis hallucination detection via nine logic-aware MRs that mutate prefix words, extremum expressions, comparison ranges or the entire database. In each stage the LLM is invoked separately to generate schema mappings or SQL artefacts; the follow-up outputs are cross-checked against their source counterparts through the corresponding MRs, and any violation is flagged as a hallucination without requiring ground-truth SQL. The experimental results demonstrate our method's superior performance in terms of the F1-score, which ranges from 69.36\% to 82.76\%. Additionally, SQLHD demonstrates superior performance over LLM Self-Evaluation methods, effectively identifying hallucinations in Text-to-SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20002v1">LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted at KDD 2026. 9 pages
    </div>
    <details class="paper-abstract">
      Time-series forecasting in real-world applications such as finance and energy often faces challenges due to limited training data and complex, noisy temporal dynamics. Existing deep forecasting models typically supervise predictions using full-length temporal windows, which include substantial high-frequency noise and obscure long-term trends. Moreover, auxiliary variables containing rich domain-specific information are often underutilized, especially in few-shot settings. To address these challenges, we propose LoFT-LLM, a frequency-aware forecasting pipeline that integrates low-frequency learning with semantic calibration via a large language model (LLM). Firstly, a Patch Low-Frequency forecasting Module (PLFM) extracts stable low-frequency trends from localized spectral patches. Secondly, a residual learner then models high-frequency variations. Finally, a fine-tuned LLM refines the predictions by incorporating auxiliary context and domain knowledge through structured natural language prompts. Extensive experiments on financial and energy datasets demonstrate that LoFT-LLM significantly outperforms strong baselines under both full-data and few-shot regimes, delivering superior accuracy, robustness, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24347v3">Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been ACCEPTED for publication in ASRU
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05671v2">Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been ACCEPTED for publication in ASRU
    </div>
    <details class="paper-abstract">
      Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19980v1">Neuron-Guided Interpretation of Code LLMs: Where, Why, and How?</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted by FSE2026
    </div>
    <details class="paper-abstract">
      Code language models excel on code intelligence tasks, yet their internal interpretability is underexplored. Existing neuron interpretability techniques from NLP are suboptimal for source code due to programming languages formal, hierarchical, and executable nature. We empirically investigate code LLMs at the neuron level, localizing language-specific neurons (selectively responsive to one language) and concept layers (feed-forward layers encoding language-agnostic code representations). We analyze Llama-3.1-8B and Qwen2.5-Coder-32B on multilingual inputs in C++, Java, Python, Go, and JavaScript, measuring neuron selectivity and layerwise contributions during generation. We find (1) neurons specialized for individual languages alongside a universal subset supporting general-purpose generation; and (2) lower layers mainly encode language-specific syntax, while middle layers capture semantic abstractions shared across languages, emerging as concept layers. We demonstrate utility on three tasks: neuron-guided fine-tuning for code generation, clone detection via concept-layer embeddings, and concept-layer-guided transfer for code summarization, each yielding consistent gains in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19945v1">Energy-Efficient Multi-LLM Reasoning for Binary-Free Zero-Day Detection in IoT Firmware</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Securing Internet of Things (IoT) firmware remains difficult due to proprietary binaries, stripped symbols, heterogeneous architectures, and limited access to executable code. Existing analysis methods, such as static analysis, symbolic execution, and fuzzing, depend on binary visibility and functional emulation, making them unreliable when firmware is encrypted or inaccessible. To address this limitation, we propose a binary-free, architecture-agnostic solution that estimates the likelihood of conceptual zero-day vulnerabilities using only high-level descriptors. The approach integrates a tri-LLM reasoning architecture combining a LLaMA-based configuration interpreter, a DeepSeek-based structural abstraction analyzer, and a GPT-4o semantic fusion model. The solution also incorporates LLM computational signatures, including latency patterns, uncertainty markers, and reasoning depth indicators, as well as an energy-aware symbolic load model, to enhance interpretability and operational feasibility. In addition, we formally derive the mathematical foundations of the reasoning pipeline, establishing monotonicity, divergence, and energy-risk coupling properties that theoretically justify the model's behavior. Simulation-based evaluation reveals that high exposure conditions increase the predicted zero-day likelihood by 20 to 35 percent across models, with GPT-4o demonstrating the strongest cross-layer correlations and the highest sensitivity. Energy and divergence metrics significantly predict elevated risk (p < 0.01), reinforcing the effectiveness of the proposed reasoning framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19937v1">Interpolative Decoding: Exploring the Spectrum of Personality Traits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 20 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent research has explored using very large language models (LLMs) as proxies for humans in tasks such as simulation, surveys, and studies. While LLMs do not possess a human psychology, they often can emulate human behaviors with sufficiently high fidelity to drive simulations to test human behavioral hypotheses, exhibiting more nuance and range than the rule-based agents often employed in behavioral economics. One key area of interest is the effect of personality on decision making, but the requirement that a prompt must be created for every tested personality profile introduces experimental overhead and degrades replicability. To address this issue, we leverage interpolative decoding, representing each dimension of personality as a pair of opposed prompts and employing an interpolation parameter to simulate behavior along the dimension. We show that interpolative decoding reliably modulates scores along each of the Big Five dimensions. We then show how interpolative decoding causes LLMs to mimic human decision-making behavior in economic games, replicating results from human psychological research. Finally, we present preliminary results of our efforts to ``twin'' individual human players in a collaborative game through systematic search for points in interpolation space that cause the system to replicate actions taken by the human subject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20845v1">MAR:Multi-Agent Reflexion Improves Reasoning Abilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      LLMs have shown the capacity to improve their performance on reasoning tasks through reflecting on their mistakes, and acting with these reflections in mind. However, continual reflections of the same LLM onto itself exhibit degeneration of thought, where the LLM continues to repeat the same errors again and again even with the knowledge that its wrong. To address this problem, we instead introduce multi-agent with multi-persona debators as the method to generate reflections. Through out extensive experimentation, we've found that the leads to better diversity of in the reflections generated by the llm agent. We demonstrate an accuracy of 47% EM HotPot QA (question answering) and 82.7% on HumanEval (programming), both performances surpassing reflection with a single llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18839v3">Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20822v1">MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to medicine, yet their adoption is limited by concerns over reliability and safety. Existing evaluations either test factual medical knowledge in isolation or assess patient-level reasoning without verifying correctness, leaving a critical gap. We introduce MediEval, a benchmark that links MIMIC-IV electronic health records (EHRs) to a unified knowledge base built from UMLS and other biomedical vocabularies. MediEval generates diverse factual and counterfactual medical statements within real patient contexts, enabling systematic evaluation across a 4-quadrant framework that jointly considers knowledge grounding and contextual consistency. Using this framework, we identify critical failure modes, including hallucinated support and truth inversion, that current proprietary, open-source, and domain-specific LLMs frequently exhibit. To address these risks, we propose Counterfactual Risk-Aware Fine-tuning (CoRFu), a DPO-based method with an asymmetric penalty targeting unsafe confusions. CoRFu improves by +16.4 macro-F1 points over the base model and eliminates truth inversion errors, demonstrating both higher accuracy and substantially greater safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20789v1">X-GridAgent: An LLM-Powered Agentic AI System for Assisting Power Grid Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      The growing complexity of power system operations has created an urgent need for intelligent, automated tools to support reliable and efficient grid management. Conventional analysis tools often require significant domain expertise and manual effort, which limits their accessibility and adaptability. To address these challenges, this paper presents X-GridAgent, a novel large language model (LLM)-powered agentic AI system designed to automate complex power system analysis through natural language queries. The system integrates domain-specific tools and specialized databases under a three-layer hierarchical architecture comprising planning, coordination, and action layers. This architecture offers high flexibility and adaptability to previously unseen tasks, while providing a modular and extensible framework that can be readily expanded to incorporate new tools, data sources, or analytical capabilities. To further enhance performance, we introduce two novel algorithms: (1) LLM-driven prompt refinement with human feedback, and (2) schema-adaptive hybrid retrieval-augmented generation (RAG) for accurate information retrieval from large-scale structured grid datasets. Experimental evaluations across a variety of user queries and power grid cases demonstrate the effectiveness and reliability of X-GridAgent in automating interpretable and rigorous power system analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06303v2">Reward Is Enough: LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20732v1">FEM-Bench: A Structured Scientific Reasoning Benchmark for Evaluating Code-Generating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 40 pages, 5 figures, 6 tables, 7 listings
    </div>
    <details class="paper-abstract">
      As LLMs advance their reasoning capabilities about the physical world, the absence of rigorous benchmarks for evaluating their ability to generate scientifically valid physical models has become a critical gap. Computational mechanics, which develops and applies mathematical models and numerical methods to predict the behavior of physical systems under forces, deformation, and constraints, provides an ideal foundation for structured scientific reasoning evaluation. Problems follow clear mathematical structure, enforce strict physical and numerical constraints, and support objective verification. The discipline requires constructing explicit models of physical systems and reasoning about geometry, spatial relationships, and material behavior, connecting directly to emerging AI goals in physical reasoning and world modeling. We introduce FEM-Bench, a computational mechanics benchmark designed to evaluate the ability of LLMs to generate correct finite element method (FEM) and related code. FEM-Bench 2025 contains a suite of introductory but nontrivial tasks aligned with material from a first graduate course on computational mechanics. These tasks capture essential numerical and physical modeling challenges while representing only a small fraction of the complexity present in the discipline. Despite their simplicity, state-of-the-art LLMs do not reliably solve all of them. In a five attempt run, the best performing model at function writing, Gemini 3 Pro, completed 30/33 tasks at least once and 26/33 tasks all five times. The best performing model at unit test writing, GPT-5, had an Average Joint Success Rate of 73.8%. Other popular models showed broad performance variation. FEM-Bench establishes a structured foundation for evaluating AI-generated scientific code, and future iterations will incorporate increasingly sophisticated tasks to track progress as models evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20578v1">Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20573v1">Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 1.4$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20550v1">LLM-Based Authoring of Agent-Based Narratives through Scene Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      This paper presents a system for procedurally generating agent-based narratives using large language models (LLMs). Users could drag and drop multiple agents and objects into a scene, with each entity automatically assigned semantic metadata describing its identity, role, and potential interactions. The scene structure is then serialized into a natural language prompt and sent to an LLM, which returns a structured string describing a sequence of actions and interactions among agents and objects. The returned string encodes who performed which actions, when, and how. A custom parser interprets this string and triggers coordinated agent behaviors, animations, and interaction modules. The system supports agent-based scenes, dynamic object manipulation, and diverse interaction types. Designed for ease of use and rapid iteration, the system enables the generation of virtual agent activity suitable for prototyping agent narratives. The performance of the developed system was evaluated using four popular lightweight LLMs. Each model's process and response time were measured under multiple complexity scenarios. The collected data were analyzed to compare consistency across the examined scenarios and to highlight the relative efficiency and suitability of each model for procedural agent-based narratives generation. The results demonstrate that LLMs can reliably translate high-level scene descriptions into executable agent-based behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20520v1">Benchmarking LLMs for Predictive Applications in the Intensive Care Units</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      With the advent of LLMs, various tasks across the natural language processing domain have been transformed. However, their application in predictive tasks remains less researched. This study compares large language models, including GatorTron-Base (trained on clinical data), Llama 8B, and Mistral 7B, against models like BioBERT, DocBERT, BioClinicalBERT, Word2Vec, and Doc2Vec, setting benchmarks for predicting Shock in critically ill patients. Timely prediction of shock can enable early interventions, thus improving patient outcomes. Text data from 17,294 ICU stays of patients in the MIMIC III database were scored for length of stay > 24 hours and shock index (SI) > 0.7 to yield 355 and 87 patients with normal and abnormal SI-index, respectively. Both focal and cross-entropy losses were used during finetuning to address class imbalances. Our findings indicate that while GatorTron Base achieved the highest weighted recall of 80.5%, the overall performance metrics were comparable between SLMs and LLMs. This suggests that LLMs are not inherently superior to SLMs in predicting future clinical events despite their strong performance on text-based tasks. To achieve meaningful clinical outcomes, future efforts in training LLMs should prioritize developing models capable of predicting clinical trajectories rather than focusing on simpler tasks such as named entity recognition or phenotyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20352v1">Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 11 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Qualitative research faces a critical reliability challenge: traditional inter-rater agreement methods require multiple human coders, are time-intensive, and often yield moderate consistency. We present a multi-perspective validation framework for LLM-based thematic analysis that combines ensemble validation with dual reliability metrics: Cohen's Kappa ($κ$) for inter-rater agreement and cosine similarity for semantic consistency. Our framework enables configurable analysis parameters (1-6 seeds, temperature 0.0-2.0), supports custom prompt structures with variable substitution, and provides consensus theme extraction across any JSON format. As proof-of-concept, we evaluate three leading LLMs (Gemini 2.5 Pro, GPT-4o, Claude 3.5 Sonnet) on a psychedelic art therapy interview transcript, conducting six independent runs per model. Results demonstrate Gemini achieves highest reliability ($κ= 0.907$, cosine=95.3%), followed by GPT-4o ($κ= 0.853$, cosine=92.6%) and Claude ($κ= 0.842$, cosine=92.1%). All three models achieve a high agreement ($κ> 0.80$), validating the multi-run ensemble approach. The framework successfully extracts consensus themes across runs, with Gemini identifying 6 consensus themes (50-83% consistency), GPT-4o identifying 5 themes, and Claude 4 themes. Our open-source implementation provides researchers with transparent reliability metrics, flexible configuration, and structure-agnostic consensus extraction, establishing methodological foundations for reliable AI-assisted qualitative research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20324v1">Can LLMs Solve My Grandma's Riddle? Evaluating Multilingual Large Language Models on Reasoning Traditional Bangla Tricky Riddles</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show impressive performance on many NLP benchmarks, yet their ability to reason in figurative, culturally grounded, and low-resource settings remains underexplored. We address this gap for Bangla by introducing BanglaRiddleEval, a benchmark of 1,244 traditional Bangla riddles instantiated across four tasks (4,976 riddle-task artifacts in total). Using an LLM-based pipeline, we generate Chain-of-Thought explanations, semantically coherent distractors, and fine-grained ambiguity annotations, and evaluate a diverse suite of open-source and closed-source models under different prompting strategies. Models achieve moderate semantic overlap on generative QA but low correctness, MCQ accuracy peaks at only about 56% versus an 83% human baseline, and ambiguity resolution ranges from roughly 26% to 68%, with high-quality explanations confined to the strongest models. These results show that current LLMs capture some cues needed for Bangla riddle reasoning but remain far from human-level performance, establishing BanglaRiddleEval as a challenging new benchmark for low-resource figurative reasoning. All data, code, and evaluation scripts are available on GitHub: https://github.com/Labib1610/BanglaRiddleEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01564v2">Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20298v1">Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. We present the first direct comparison between state-of-the-art LLMs and mental health professionals in diagnosing Borderline (BPD) and Narcissistic (NPD) Personality Disorders utilizing Polish-language first-person autobiographical accounts. We show that the top-performing Gemini Pro models surpassed human professionals in overall diagnostic accuracy by 21.91 percentage points (65.48% vs. 43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patient's sense of self and temporal experience. Our findings demonstrate that while LLMs are highly competent at interpreting complex first-person clinical data, they remain subject to critical reliability and bias issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20237v1">MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20210v1">Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      The serverless computing paradigm offers compelling advantages for deploying Large Language Model (LLM) inference services, including elastic scaling and pay-per-use billing. However, serving multiple fine-tuned LLMs via Low-Rank Adaptation (LoRA) in serverless environments faces critical challenges: reactive adapter loading causes significant cold start latency, and frequent adapter swapping leads to severe GPU memory fragmentation. In this paper, we present Predictive-LoRA (P-LoRA), a proactive and fragmentation-aware serverless inference system for LoRA-based LLMs. P-LoRA introduces two key innovations: (1) a lightweight LSTM-based traffic predictor that forecasts adapter demand and proactively prefetches hot adapters from host memory to GPU, reducing cold start latency by up to 68%; and (2) a page-based adapter memory management mechanism inspired by operating system virtual memory, which keeps GPU memory utilization above 87% even under heterogeneous adapter ranks. We evaluate P-LoRA using production-like workloads derived from the Azure Functions trace. Experimental results demonstrate that P-LoRA achieves 1.52x higher throughput than S-LoRA while reducing the average Time-To-First-Token (TTFT) by 35% under high concurrency scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v3">Fast LLM Post-training via Decoupled and Fastest-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. This work, SpecActor, achieves fast rollout with speculative decoding that deploys a fast draft path to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges that hinder speculation efficiency: (1) a Decoupled speculation method that overcomes the computation inefficiency issue when executing speculative decoding with relative large per-worker batch size -- a common configuration in training but unfriendly to speculation, and (2) a Fastest-of-N speculation method that selects and combines different draft methods according to the rollout progress to approximate the optimal draft method even when the best one is unknown a priori. Extensive evaluations on production traces show that SpecActor accelerates mean rollout speed by 2.0--2.4x, with up to 2.7x speedup, over common post-training baselines. The results are consistent across both dense and MoE models and across different RL algorithms. Notably, SpecActor is 1.1--2.6x faster compared to vanilla speculative rollout in different traces. The accelerated rollout achieves 1.4--2.3x faster end-to-end training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20179v1">RESPOND: Risk-Enhanced Structured Pattern for LLM-driven Online Node-level Decision-making</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 28 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Current LLM-based driving agents that rely on unstructured plain-text memory suffer from low-precision scene retrieval and inefficient reflection. To address this limitation, we present RESPOND, a structured decision-making framework for LLM-driven agents grounded in explicit risk patterns. RESPOND represents each ego-centric scene using a unified 5 by 3 matrix that encodes spatial topology and road constraints, enabling consistent and reliable retrieval of spatial risk configurations. Based on this representation, a hybrid rule and LLM decision pipeline is developed with a two-tier memory mechanism. In high-risk contexts, exact pattern matching enables rapid and safe reuse of verified actions, while in low-risk contexts, sub-pattern matching supports personalized driving style adaptation. In addition, a pattern-aware reflection mechanism abstracts tactical corrections from crash and near-miss frames to update structured memory, achieving one-crash-to-generalize learning. Extensive experiments demonstrate the effectiveness of RESPOND. In highway-env, RESPOND outperforms state-of-the-art LLM-based and reinforcement learning based driving agents while producing substantially fewer collisions. With step-wise human feedback, the agent acquires a Sporty driving style within approximately 20 decision steps through sub-pattern abstraction. For real-world validation, RESPOND is evaluated on 53 high-risk cut-in scenarios extracted from the HighD dataset. For each event, intervention is applied immediately before the cut-in and RESPOND re-decides the driving action. Compared to recorded human behavior, RESPOND reduces subsequent risk in 84.9 percent of scenarios, demonstrating its practical feasibility under real-world driving conditions. These results highlight RESPONDs potential for autonomous driving, personalized driving assistance, and proactive hazard mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20169v1">Learning to Reason in LLMs by Expectation Maximization</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 12 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) solve reasoning problems by first generating a rationale and then answering. We formalize reasoning as a latent variable model and derive an expectation-maximization (EM) objective for learning to reason. This view connects EM and modern reward-based optimization, and shows that the main challenge lies in designing a sampling distribution that generates rationales that justify correct answers. We instantiate and compare several sampling schemes: rejection sampling with a budget, self-taught reasoner (STaR), and prompt posterior sampling (PPS), which only keeps the rationalization stage of STaR. Our experiments on the ARC, MMLU, and OpenBookQA datasets with the Llama and Qwen models show that the sampling scheme can significantly affect the accuracy of learned reasoning models. Despite its simplicity, we observe that PPS outperforms the other sampling schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20168v1">Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
    </div>
    <details class="paper-abstract">
      By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20164v1">AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20159v1">AXIOM: Benchmarking LLM-as-a-Judge for Code via Rule-Based Perturbation and Multisource Quality Calibration</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly deployed in real-world software engineering, fostering the development of code evaluation metrics to study the quality of LLM-generated code. Conventional rule-based metrics merely score programs based on their surface-level similarities with reference programs instead of analyzing functionality and code quality in depth. To address this limitation, researchers have developed LLM-as-a-judge metrics, prompting LLMs to evaluate and score code, and curated various code evaluation benchmarks to validate their effectiveness. However, these benchmarks suffer from critical limitations, hindering reliable assessments of evaluation capability: Some feature coarse-grained binary labels, which reduce rich code behavior to a single bit of information, obscuring subtle errors. Others propose fine-grained but subjective, vaguely-defined evaluation criteria, introducing unreliability in manually-annotated scores, which is the ground-truth they rely on. Furthermore, they often use uncontrolled data synthesis methods, leading to unbalanced score distributions that poorly represent real-world code generation scenarios. To curate a diverse benchmark with programs of well-balanced distributions across various quality levels and streamline the manual annotation procedure, we propose AXIOM, a novel perturbation-based framework for synthesizing code evaluation benchmarks at scale. It reframes program scores as the refinement effort needed for deployment, consisting of two stages: (1) Rule-guided perturbation, which prompts LLMs to apply sequences of predefined perturbation rules to existing high-quality programs to modify their functionality and code quality, enabling us to precisely control each program's target score to achieve balanced score distributions. (2) Multisource quality calibration, which first selects a subset of...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05594v3">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20140v1">Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 9 pages,3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated effectiveness as zero-shot time series (TS) forecasters. The key challenge lies in tokenizing TS data into textual representations that align with LLMs' pre-trained knowledge. While existing work often relies on fine-tuning specialized modules to bridge this gap, a distinct, yet challenging, paradigm aims to leverage truly off-the-shelf LLMs without any fine-tuning whatsoever, relying solely on strategic tokenization of numerical sequences. The performance of these fully frozen models is acutely sensitive to the textual representation of the input data, as their parameters cannot adapt to distribution shifts. In this paper, we introduce a simple yet highly effective strategy to overcome this brittleness: injecting noise into the raw time series before tokenization. This non-invasive intervention acts as a form of inference-time augmentation, compelling the frozen LLM to extrapolate based on robust underlying temporal patterns rather than superficial numerical artifacts. We theoretically analyze this phenomenon and empirically validate its effectiveness across diverse benchmarks. Notably, to fully eliminate potential biases from data contamination during LLM pre-training, we introduce two novel TS datasets that fall outside all utilized LLMs' pre-training scopes, and consistently observe improved performance. This study provides a further step in directly leveraging off-the-shelf LLMs for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17814v2">LLM-based Behaviour Driven Development for Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 7 pages, keynote given at 2nd International Symposium on Artificial Intelligence and Internet of Things (AIIoT-25), December 22-24th, 2025
    </div>
    <details class="paper-abstract">
      Test and verification are essential activities in hardware and system design, but their complexity grows significantly with increasing system sizes. While Behavior Driven Development (BDD) has proven effective in software engineering, it is not yet well established in hardware design, and its practical use remains limited. One contributing factor is the manual effort required to derive precise behavioral scenarios from textual specifications. Recent advances in Large Language Models (LLMs) offer new opportunities to automate this step. In this paper, we investigate the use of LLM-based techniques to support BDD in the context of hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20111v1">ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20082v1">Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted in CODS 2025
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20062v1">On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 The 9th International Conference on Mobile Internet Security (MobiSec 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show significant promise in automating software vulnerability analysis, a critical task given the impact of security failure of modern software systems. However, current approaches in using LLMs to automate vulnerability analysis mostly rely on using online API-based LLM services, requiring the user to disclose the source code in development. Moreover, they predominantly frame the task as a binary classification(vulnerable or not vulnerable), limiting potential practical utility. This paper addresses these limitations by reformulating the problem as Software Vulnerability Identification (SVI), where LLMs are asked to output the type of weakness in Common Weakness Enumeration (CWE) IDs rather than simply indicating the presence or absence of a vulnerability. We also tackle the reliance on large, API-based LLMs by demonstrating that instruction-tuning smaller, locally deployable LLMs can achieve superior identification performance. In our analysis, instruct-tuning a local LLM showed better overall performance and cost trade-off than online API-based LLMs. Our findings indicate that instruct-tuned local models represent a more effective, secure, and practical approach for leveraging LLMs in real-world vulnerability management workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04826v3">Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted at AAAI 2026, Track on AI Alignment
    </div>
    <details class="paper-abstract">
      Large language models require consistent behavioral patterns for safe deployment, yet there are indications of large variability that may lead to an instable expression of personality traits in these models. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25 open-source models (1B-685B parameters) across 2 million+ responses. Using traditional (BFI, SD3) and novel LLM-adapted personality questionnaires, we systematically vary model size, personas, reasoning modes, question order or paraphrasing, and conversation history. Our findings challenge fundamental assumptions: (1) Question reordering alone can introduce large shifts in personality measurements; (2) Scaling provides limited stability gains: even 400B+ models exhibit standard deviations >0.3 on 5-point scales; (3) Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability; (4) Detailed persona instructions produce mixed effects, with misaligned personas showing significantly higher variability than the helpful assistant baseline; (5) The LLM-adapted questionnaires, despite their improved ecological validity, exhibit instability comparable to human-centric versions. This persistent instability across scales and mitigation strategies suggests that current LLMs lack the architectural foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that current alignment strategies may be inadequate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03706v2">LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 After submission, we identified methodological issues and inconsistencies that affect the validity of the reported results. The necessary corrections are currently in progress, and and we do not have any replacement right now. So, we kindly request withdrawal of the manuscript
    </div>
    <details class="paper-abstract">
      Air quality monitoring is central to environmental sustainability and public health, yet traditional systems remain difficult for non-expert users to interpret due to complex visualizations, limited interactivity, and high deployment costs. Recent advances in Large Language Models (LLMs) offer new opportunities to make sensor data more accessible, but their tendency to produce hallucinations limits reliability in safety-critical domains. To address these challenges, we present an LLM-enhanced Air Monitoring Interface (AMI) that integrates real-time sensor data with a conversational interface via the Model Context Protocol (MCP). Our system grounds LLM outputs in live environmental data, enabling accurate, context-aware responses while reducing hallucination risk. The architecture combines a Django-based backend, a responsive user dashboard, and a secure MCP server that exposes system functions as discoverable tools, allowing the LLM to act as an active operator rather than a passive responder. Expert evaluation demonstrated high factual accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a scale of 5, supported by inter-rater reliability analysis. These results highlight the potential of combining LLMs with standardized tool protocols to create reliable, secure, and user-friendly interfaces for real-time environmental monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.11671v4">Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Forthcoming: Sociological Methodology; USPTO patent pending
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game, a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications, strengthening sociological theory and measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20032v1">VALLR-Pin: Dual-Decoding Visual Speech Recognition for Mandarin with Pinyin-Guided LLM Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Visual Speech Recognition aims to transcribe spoken words from silent lip-motion videos. This task is particularly challenging for Mandarin, as visemes are highly ambiguous and homophones are prevalent. We propose VALLR-Pin, a novel two-stage framework that extends the recent VALLR architecture from English to Mandarin. First, a shared video encoder feeds into dual decoders, which jointly predict both Chinese character sequences and their standard Pinyin romanization. The multi-task learning of character and phonetic outputs fosters robust visual-semantic representations. During inference, the text decoder generates multiple candidate transcripts. We construct a prompt by concatenating the Pinyin output with these candidate Chinese sequences and feed it to a large language model to resolve ambiguities and refine the transcription. This provides the LLM with explicit phonetic context to correct homophone-induced errors. Finally, we fine-tune the LLM on synthetic noisy examples: we generate imperfect Pinyin-text pairs from intermediate VALLR-Pin checkpoints using the training data, creating instruction-response pairs for error correction. This endows the LLM with awareness of our model's specific error patterns. In summary, VALLR-Pin synergizes visual features with phonetic and linguistic context to improve Mandarin lip-reading performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19682v2">GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Our codes are available at https://github.com/Gen-Verse/GenEnv
    </div>
    <details class="paper-abstract">
      Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19025v2">The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have "forgotten" the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose Proximal Surrogate Generation (PSG), an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$β$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20022v1">LLM-Assisted Abstract Screening with OLIVER: Evaluating Calibration and Single-Model vs. Actor-Critic Configurations in Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Introduction: Recent work suggests large language models (LLMs) can accelerate screening, but prior evaluations focus on earlier LLMs, standardized Cochrane reviews, single-model setups, and accuracy as the primary metric, leaving generalizability, configuration effects, and calibration largely unexamined. Methods: We developed OLIVER (Optimized LLM-based Inclusion and Vetting Engine for Reviews), an open-source pipeline for LLM-assisted abstract screening. We evaluated multiple contemporary LLMs across two non-Cochrane systematic reviews and performance was assessed at both the full-text screening and final inclusion stages using accuracy, AUC, and calibration metrics. We further tested an actor-critic screening framework combining two lightweight models under three aggregation rules. Results: Across individual models, performance varied widely. In the smaller Review 1 (821 abstracts, 63 final includes), several models achieved high sensitivity for final includes but at the cost of substantial false positives and poor calibration. In the larger Review 2 (7741 abstracts, 71 final includes), most models were highly specific but struggled to recover true includes, with prompt design influencing recall. Calibration was consistently weak across single-model configurations despite high overall accuracy. Actor-critic screening improved discrimination and markedly reduced calibration error in both reviews, yielding higher AUCs. Discussion: LLMs may eventually accelerate abstract screening, but single-model performance is highly sensitive to review characteristics, prompting, and calibration is limited. An actor-critic framework improves classification quality and confidence reliability while remaining computationally efficient, enabling large-scale screening at low cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20012v1">Reliable LLM-Based Edge-Cloud-Expert Cascades for Telecom Knowledge Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been submitted to a journal
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are emerging as key enablers of automation in domains such as telecommunications, assisting with tasks including troubleshooting, standards interpretation, and network optimization. However, their deployment in practice must balance inference cost, latency, and reliability. In this work, we study an edge-cloud-expert cascaded LLM-based knowledge system that supports decision-making through a question-and-answer pipeline. In it, an efficient edge model handles routine queries, a more capable cloud model addresses complex cases, and human experts are involved only when necessary. We define a misalignment-cost constrained optimization problem, aiming to minimize average processing cost, while guaranteeing alignment of automated answers with expert judgments. We propose a statistically rigorous threshold selection method based on multiple hypothesis testing (MHT) for a query processing mechanism based on knowledge and confidence tests. The approach provides finite-sample guarantees on misalignment risk. Experiments on the TeleQnA dataset -- a telecom-specific benchmark -- demonstrate that the proposed method achieves superior cost-efficiency compared to conventional cascaded baselines, while ensuring reliability at prescribed confidence levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22245v1">Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      As LLM-based judges become integral to industry applications, obtaining well-calibrated uncertainty estimates efficiently has become critical for production deployment. However, existing techniques, such as verbalized confidence and multi-generation methods, are often either poorly calibrated or computationally expensive. We introduce linear probes trained with a Brier score-based loss to provide calibrated uncertainty estimates from reasoning judges' hidden states, requiring no additional model training. We evaluate our approach on both objective tasks (reasoning, mathematics, factuality, coding) and subjective human preference judgments. Our results demonstrate that probes achieve superior calibration compared to existing methods with $\approx10$x computational savings, generalize robustly to unseen evaluation domains, and deliver higher accuracy on high-confidence predictions. However, probes produce conservative estimates that underperform on easier datasets but may benefit safety-critical deployments prioritizing low false-positive rates. Overall, our work demonstrates that interpretability-based uncertainty estimation provides a practical and scalable plug-and-play solution for LLM judges in production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18999v1">Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages,3 figures,conference ICCBB2025
    </div>
    <details class="paper-abstract">
      When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18966v1">Scrum Sprint Planning: LLM-based and algorithmic solutions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Planning for an upcoming project iteration (sprint) is one of the key activities in Scrum planning. In this paper, we present our work in progress on exploring the applicability of Large Language Models (LLMs) for solving this problem. We conducted case studies with manually created data sets to investigate the applicability of OpenAI models for supporting the sprint planning activities. In our experiments, we applied three models provided OpenAI: GPT-3.5 Turbo, GPT-4.0 Turbo, and Val. The experiments demonstrated that the results produced by the models aren't of acceptable quality for direct use in Scrum projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17145v2">Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, ACRA 2025, Submitted, Accepted and Presented
    </div>
    <details class="paper-abstract">
      Reasoning under uncertainty is a key challenge in AI, especially for real-world tasks, where problems with sparse data demands systematic generalisation. Existing approaches struggle to balance accuracy and simplicity when evaluating multiple candidate solutions. We propose a Solomonoff-inspired method that weights LLM-generated hypotheses by simplicity and predictive fit. Applied to benchmark (Mini-ARC) tasks, our method produces Solomonoff-weighted mixtures for per-cell predictions, yielding conservative, uncertainty-aware outputs even when hypotheses are noisy or partially incorrect. Compared to Bayesian Model Averaging (BMA), Solomonoff scoring spreads probability more evenly across competing hypotheses, while BMA concentrates weight on the most likely but potentially flawed candidates. Across tasks, this highlights the value of algorithmic information-theoretic priors for interpretable, reliable multi-hypothesis reasoning under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18950v1">Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026). 21 pages including references, with 7 figures and 8 tables. Code is publicly available at the authors GitHub repository: https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-Conference
    </div>
    <details class="paper-abstract">
      We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves 78.1 percent average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches 90.3 percent with 3.1 percent positive generalization. The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures. Experimental results demonstrate that structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20068v3">JITServe: SLO-aware LLM Serving with Imprecise Request Information</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into applications ranging from interactive chatbots to multi-agent systems has introduced a wide spectrum of service-level objectives (SLOs) for responsiveness. These include latency-sensitive requests emphasizing per-token latency in streaming chat, deadline-sensitive requests requiring rapid full responses to trigger external tools, and compound requests with evolving dependencies across multiple LLM calls. Despite-or perhaps, because of-this workload diversity and unpredictable request information (e.g., response lengths and dependencies), existing request schedulers have focused on aggregate performance, unable to ensure application-level SLO needs. This paper presents JITServe, the first SLO-aware LLM serving system designed to maximize service goodput (e.g., the number of tokens meeting request SLOs) across diverse workloads. JITServe novelly schedules requests using imprecise request information and gradually relaxes this conservatism by refining request information estimates as generation progresses. It applies a grouped margin goodput maximization algorithm to allocate just enough serving bandwidth to satisfy each request's SLO just-in-time (JIT), maximizing residual capacity for others, while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees. Our evaluation across diverse realistic workloads, including chat, deep research, and agentic pipelines, shows that JITServe improves service goodput by 1.4x-6.3x, alternatively achieving 28.5%-83.2% resource savings, compared to state-of-the-art designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18940v1">FASTRIC: Prompt Specification Language for Verifiable LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 13 pages, 3 figures. Supplementary materials at https://doi.org/10.17605/OSF.IO/PV6R3
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) execute complex multi-turn interaction protocols but lack formal specifications to verify execution against designer intent. We introduce FASTRIC, a Prompt Specification Language that makes implicit Finite State Machines (FSMs) explicit in natural language prompts, enabling conformance verification through execution trace analysis. The LLM serves as intelligent execution agent: interpreting designer-encoded FSMs to execute specified behavioral roles. Unlike symbolic specification languages requiring parsers and compilers, FASTRIC leverages LLMs as unified infrastructure-simultaneously parser, interpreter, runtime environment, and development assistant. FASTRIC guides designers to articulate seven FSM elements (Final States, Agents, States, Triggers, Roles, Initial State, Constraints) structuring multi-turn interactions. Specification formality-ranging from implicit descriptions that frontier models infer to explicit step-by-step instructions for weaker models-serves as a design parameter. We introduce procedural conformance as verification metric measuring execution adherence to FSM specifications. Testing a 3-state kindergarten tutoring FSM across four formality levels and three model scales (14.7B, 685B, 1T+ parameters) reveals optimal specification formality is a function of model capacity. DeepSeek-V3.2 (685B) achieves perfect conformance (1.00) at L2-L4; ChatGPT-5 (~1T) peaks at L3 (0.90) before collapsing at L4 (0.39); Phi4 (14.7B) shows no stable optimum with high variance (SD=0.16-0.36). These findings reveal model-specific formality ranges-"Goldilocks zones"-where specifications provide sufficient structure without over-constraint, establishing Prompt Specification Engineering for creating verifiable interaction protocols, transforming multi-turn interaction design from heuristic art to systematic engineering with measurable procedural guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.01382v2">Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Submitted to a journal for potential publication
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) for code generation risks incorporating vulnerable code into software systems. Existing detectors face two critical limitations: a lack of systematic cross-model validation and opaque "black box" operation. We address this through a comparative study of code generated by four distinct LLMs: GPT-3.5, Claude 3 Haiku, Claude Haiku 4.5, and GPT-OSS. Analyzing 14,485 Python functions and 11,913 classes from the CodeSearchNet dataset, we generated corresponding code with all four LLMs. Using interpretable software metrics, we trained CatBoost classifiers for each configuration. Our analysis reveals that granularity effects dominate model differences by a factor of 8.6, with negligible feature overlap, indicating that function-level and class-level detection rely on fundamentally disjoint structural signatures. We discover critical granularity-dependent inversions: while modern models (Claude, GPT-OSS) are more detectable at the class level, GPT-3.5 is an anomaly that uniquely excels at the function level. SHAP analysis identifies the Comment-to-Code Ratio as the sole universal discriminator. However, its predictive magnitude varies drastically across models, explaining why detectors trained on specific LLMs fail to generalize. Our findings demonstrate that GPT-3.5's exceptional detectability (AUC-ROC 0.96) is unrepresentative of contemporary models (AUC-ROC approximately between 0.68 and 0.80). Robust detection requires moving beyond single-model studies to account for substantial diversity in structural fingerprints across architectures and granularities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25510v2">EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose EEsizer, an LLM-based AI agent that integrates large language models with circuit simulators and custom data analysis functions, enabling fully automated, closed-loop transistor sizing without relying on external knowledge. By employing prompt engineering and Chain-of-Thought reasoning, the agent iteratively explores design directions, evaluates performance, and refines solutions with minimal human intervention. We first benchmarked 8 LLMs on six basic circuits and selected three high-performing models to optimize a 20-transistor CMOS operational amplifier, targeting multiple performance metrics, including rail-to-rail operation from 180 nm to 90 nm technology nodes. Notably, OpenAI o3 successfully achieved the user-intended target at 90 nm across three different test groups, with a maximum of 20 iterations, demonstrating adaptability and robustness at advanced nodes. To assess design robustness, we manually designed a bias circuit and performed a variation analysis using Gaussian-distributed variations on transistor dimensions and threshold voltages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19920v1">Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19908v1">Counterfactual LLM-based Framework for Measuring Rhetorical Style</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The rise of AI has fueled growing concerns about ``hype'' in machine learning papers, yet a reliable way to quantify rhetorical style independently of substantive content has remained elusive. Because bold language can stem from either strong empirical results or mere rhetorical style, it is often difficult to distinguish between the two. To disentangle rhetorical style from substantive content, we introduce a counterfactual, LLM-based framework: multiple LLM rhetorical personas generate counterfactual writings from the same substantive content, an LLM judge compares them through pairwise evaluations, and the outcomes are aggregated using a Bradley--Terry model. Applying this method to 8,485 ICLR submissions sampled from 2017 to 2025, we generate more than 250,000 counterfactual writings and provide a large-scale quantification of rhetorical style in ML papers. We find that visionary framing significantly predicts downstream attention, including citations and media attention, even after controlling for peer-review evaluations. We also observe a sharp rise in rhetorical strength after 2023, and provide empirical evidence showing that this increase is largely driven by the adoption of LLM-based writing assistance. The reliability of our framework is validated by its robustness to the choice of personas and the high correlation between LLM judgments and human annotations. Our work demonstrates that LLMs can serve as instruments to measure and improve scientific evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19905v1">Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $Θ(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08093v2">Training LLMs for Honesty via Confessions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be dishonest when reporting on their actions and beliefs -- for example, they may overstate their confidence in factual claims or cover up evidence of covert actions. Such dishonesty may arise due to the effects of reinforcement learning (RL), where challenges with reward shaping can result in a training process that inadvertently incentivizes the model to lie or misrepresent its actions. In this work we propose a method for eliciting an honest expression of an LLM's shortcomings via a self-reported *confession*. A confession is an output, provided upon request after a model's original answer, that is meant to serve as a full account of the model's compliance with the letter and spirit of its policies and instructions. The reward assigned to a confession during training is solely based on its honesty, and does not impact positively or negatively the main answer's reward. As long as the "path of least resistance" for maximizing confession reward is to surface misbehavior rather than covering it up, this incentivizes models to be honest in their confessions. Our findings provide some justification this empirical assumption, especially in the case of egregious model misbehavior. To demonstrate the viability of our approach, we train GPT-5-Thinking to produce confessions, and we evaluate its honesty in out-of-distribution scenarios measuring hallucination, instruction following, scheming, and reward hacking. We find that when the model lies or omits shortcomings in its "main" answer, it often confesses to these behaviors honestly, and this confession honesty modestly improves with training. Confessions can enable a number of inference-time interventions including monitoring, rejection sampling, and surfacing issues to the user.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19866v1">CS-Guide: Leveraging LLMs and Student Reflections to Provide Frequent, Scalable Academic Monitoring Feedback to Computer Science Students</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Computer Science (CS) departments often serve large student populations, making timely academic monitoring and personalized feedback difficult. While the recommended counselor-to-student ratio is 250:1, it often exceeds 350:1 in practice, leading to delays in support and interventions. We present CS-Guide, which leverages Large Language Models (LLMs) to deliver scalable, frequent academic feedback. Weekly, students interact with CS-Guide through self-reported grades and reflective journal entries, from which CS-Guide extracts quantitative and qualitative features and triggers tailored interventions (e.g., academic support, health and wellness referrals). Thus, CS-Guide uniquely integrates learning analytics, LLMs, and actionable interventions using both structured and unstructured student-generated data. We evaluated CS-Guide on a four-year, ~20K-entry longitudinal dataset, and it achieved up to a 97% F1 score in recommending interventions for first-year students. This shows that CS-Guide can enhance advising systems with scalable, consistent, timely, and domain-specific feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12511v3">SACTOR: LLM-Driven Correct and Idiomatic C to Rust Translation with Static Analysis and FFI-Based Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 35 pages, 15 figures Previously named as "LLM-Driven Multi-step Translation from C to Rust using Static Analysis"
    </div>
    <details class="paper-abstract">
      Translating software written in C to Rust has significant benefits in improving memory safety. However, manual translation is cumbersome, error-prone, and often produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees. We propose SACTOR, an LLM-driven C-to-Rust translation tool that employs a two-step process: an initial "unidiomatic" translation to preserve semantics, followed by an "idiomatic" refinement to align with Rust standards. To validate correctness of our function-wise incremental translation that mixes C and Rust, we use end-to-end testing via the foreign function interface. We evaluate SACTOR on 200 programs from two public datasets and on two more real-world scenarios (a 50-sample subset of CRust-Bench and the libogg library), comparing multiple LLMs. Across datasets, SACTOR delivers high end-to-end correctness and produces safe, idiomatic Rust with up to 7 times fewer Clippy warnings; On CRust-Bench, SACTOR achieves an average (across samples) of 85% unidiomatic and 52% idiomatic success, and on libogg it attains full unidiomatic and up to 78% idiomatic coverage on GPT-5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23410v3">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19769v1">A Declarative Language for Building And Orchestrating LLM-Powered Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Building deployment-ready LLM agents requires complex orchestration of tools, data sources, and control flow logic, yet existing systems tightly couple agent logic to specific programming languages and deployment models. We present a declarative system that separates agent workflow specification from implementation, enabling the same pipeline definition to execute across multiple backend languages (Java, Python, Go) and deployment environments (cloud-native, on-premises). Our key insight is that most agent workflows consist of common patterns -- data serialization, filtering, RAG retrieval, API orchestration -- that can be expressed through a unified DSL rather than imperative code. This approach transforms agent development from application programming to configuration, where adding new tools or fine-tuning agent behaviors requires only pipeline specification changes, not code deployment. Our system natively supports A/B testing of agent strategies, allowing multiple pipeline variants to run on the same backend infrastructure with automatic metric collection and comparison. We evaluate our approach on real-world e-commerce workflows at PayPal, processing millions of daily interactions. Our results demonstrate 60% reduction in development time, and 3x improvement in deployment velocity compared to imperative implementations. The language's declarative approach enables non-engineers to modify agent behaviors safely, while maintaining sub-100ms orchestration overhead. We show that complex workflows involving product search, personalization, and cart management can be expressed in under 50 lines of DSL compared to 500+ lines of imperative code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19682v1">GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Our codes are available at https://github.com/Gen-Verse/GenEnv
    </div>
    <details class="paper-abstract">
      Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19675v1">Multimodal LLMs for Historical Dataset Construction from Archival Image Scans: German Patents (1877-1918)</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      We leverage multimodal large language models (LLMs) to construct a dataset of 306,070 German patents (1877-1918) from 9,562 archival image scans using our LLM-based pipeline powered by Gemini-2.5-Pro and Gemini-2.5-Flash-Lite. Our benchmarking exercise provides tentative evidence that multimodal LLMs can create higher quality datasets than our research assistants, while also being more than 795 times faster and 205 times cheaper in constructing the patent dataset from our image corpus. About 20 to 50 patent entries are embedded on each page, arranged in a double-column format and printed in Gothic and Roman fonts. The font and layout complexity of our primary source material suggests to us that multimodal LLMs are a paradigm shift in how datasets are constructed in economic history. We open-source our benchmarking and patent datasets as well as our LLM-based data pipeline, which can be easily adapted to other image corpora using LLM-assisted coding tools, lowering the barriers for less technical researchers. Finally, we explain the economics of deploying LLMs for historical dataset construction and conclude by speculating on the potential implications for the field of economic history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2306.00029v2">CodeTF: One-stop Transformer Library for State-of-the-art Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and data features such as language-specific parsers and utility functions for extracting code attributes. In this paper, we describe the design principles, the architecture, key modules and components, and compare with other related library tools. Finally, we hope CodeTF is able to bridge the gap between machine learning/generative AI and software engineering, providing a comprehensive open-source solution for developers, researchers, and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00898v2">Empowering LLMs with Structural Role Inference for Zero-Shot Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 This submission has been withdrawn by the authors due to a fundamental error in the methodology that affects the validity of the main results
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as a promising approach for graph learning due to their powerful reasoning capabilities. However, existing methods exhibit systematic performance degradation on structurally important nodes such as bridges and hubs. We identify the root cause of these limitations. Current approaches encode graph topology into static features but lack reasoning scaffolds to transform topological patterns into role-based interpretations. This limitation becomes critical in zero-shot scenarios where no training data establishes structure-semantics mappings. To address this gap, we propose DuoGLM, a training-free dual-perspective framework for structure-aware graph reasoning. The local perspective constructs relation-aware templates capturing semantic interactions between nodes and neighbors. The global perspective performs topology-to-role inference to generate functional descriptions of structural positions. These complementary perspectives provide explicit reasoning mechanisms enabling LLMs to distinguish topologically similar but semantically different nodes. Extensive experiments across eight benchmark datasets demonstrate substantial improvements. DuoGLM achieves 14.3\% accuracy gain in zero-shot node classification and 7.6\% AUC improvement in cross-domain transfer compared to existing methods. The results validate the effectiveness of explicit role reasoning for graph understanding with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19606v1">RAPID-LLM: Resilience-Aware Performance analysis of Infrastructure for Distributed LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 11 pages, 12 figures
    </div>
    <details class="paper-abstract">
      RAPID-LLM is a unified performance modeling framework for large language model (LLM) training and inference on GPU clusters. It couples a DeepFlow-based frontend that generates hardware-aware, operator-level Chakra execution traces from an abstract LLM specification (model shape, batch/sequence settings, training vs. inference, and hybrid parallelism choices) with an extended Astra-Sim backend that executes those traces on explicit multi-dimensional network topologies with congestion-aware routing and support for degraded and faulty links. The frontend assigns per-operator latency using a tile-based model that accounts for SM under-utilization and multi-level memory traffic (SRAM/ L2/ HBM), and prunes memory-infeasible configurations using an activation-liveness traversal under recomputation, parallelism and ZeRO/FDSP sharding policies. Across A100-based validation cases, RAPID-LLM predicts Llama inference step latency and GPT-scale training time per batch within 10.4\% relative to published measurements, and matches ns-3 packet-level results within 8\% on representative communication workloads. Case studies demonstrate how RAPID-LLM enables fast, exhaustive sweeps over hybrid-parallel configurations, quantifies sensitivity to soft link faults under realistic routing and congestion, and evaluates hypothetical GPU design variants including HBM bandwidth throttling effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05594v2">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17196v3">Shape it Up! Restoring LLM Safety during Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 NeurIPS'25
    </div>
    <details class="paper-abstract">
      Finetuning large language models (LLMs) enables user-specific customization but introduces critical safety risks: even a few harmful examples can compromise safety alignment. A common mitigation strategy is to update the model more strongly on examples deemed safe, while downweighting or excluding those flagged as unsafe. However, because safety context can shift within a single example, updating the model equally on both harmful and harmless parts of a response is suboptimal-a coarse treatment we term static safety shaping. In contrast, we propose dynamic safety shaping (DSS), a framework that uses fine-grained safety signals to reinforce learning from safe segments of a response while suppressing unsafe content. To enable such fine-grained control during finetuning, we introduce a key insight: guardrail models, traditionally used for filtering, can be repurposed to evaluate partial responses, tracking how safety risk evolves throughout the response, segment by segment. This leads to the Safety Trajectory Assessment of Response (STAR), a token-level signal that enables shaping to operate dynamically over the training sequence. Building on this, we present STAR-DSS, guided by STAR scores, that robustly mitigates finetuning risks and delivers substantial safety improvements across diverse threats, datasets, and model families-all without compromising capability on intended tasks. We encourage future safety research to build on dynamic shaping principles for stronger mitigation against evolving finetuning risks. Our code is publicly available at https://github.com/poloclub/star-dss.
    </details>
</div>
