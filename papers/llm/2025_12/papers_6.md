# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12868v1">Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel on multiple-choice clinical diagnosis benchmarks, yet it is unclear how much of this performance reflects underlying probabilistic reasoning. We study this through questions from MedQA, where the task is to select the most likely diagnosis. We introduce the Frequency-Based Probabilistic Ranker (FBPR), a lightweight method that scores options with a smoothed Naive Bayes over concept-diagnosis co-occurrence statistics from a large corpus. When co-occurrence statistics were sourced from the pretraining corpora for OLMo and Llama, FBPR achieves comparable performance to the corresponding LLMs pretrained on that same corpus. Direct LLM inference and FBPR largely get different questions correct, with an overlap only slightly above random chance, indicating complementary strengths of each method. These findings highlight the continued value of explicit probabilistic baselines: they provide a meaningful performance reference point and a complementary signal for potential hybridization. While the performance of LLMs seems to be driven by a mechanism other than simple frequency aggregation, we show that an approach similar to the historically grounded, low-complexity expert systems still accounts for a substantial portion of benchmark performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.23260v2">From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 The paper is published in ICT Express (Elsevier)
    </div>
    <details class="paper-abstract">
      Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12812v1">Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, LLaMA</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Prompt engineering has emerged as a critical factor influencing large language model (LLM) performance, yet the impact of pragmatic elements such as linguistic tone and politeness remains underexplored, particularly across different model families. In this work, we propose a systematic evaluation framework to examine how interaction tone affects model accuracy and apply it to three recently released and widely available LLMs: GPT-4o mini (OpenAI), Gemini 2.0 Flash (Google DeepMind), and Llama 4 Scout (Meta). Using the MMMLU benchmark, we evaluate model performance under Very Friendly, Neutral, and Very Rude prompt variants across six tasks spanning STEM and Humanities domains, and analyze pairwise accuracy differences with statistical significance testing. Our results show that tone sensitivity is both model-dependent and domain-specific. Neutral or Very Friendly prompts generally yield higher accuracy than Very Rude prompts, but statistically significant effects appear only in a subset of Humanities tasks, where rude tone reduces accuracy for GPT and Llama, while Gemini remains comparatively tone-insensitive. When performance is aggregated across tasks within each domain, tone effects diminish and largely lose statistical significance. Compared with earlier researches, these findings suggest that dataset scale and coverage materially influence the detection of tone effects. Overall, our study indicates that while interaction tone can matter in specific interpretive settings, modern LLMs are broadly robust to tonal variation in typical mixed-domain use, providing practical guidance for prompt design and model selection in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12801v1">Fine-Grained Energy Prediction For Parallellized LLM Inference With PIE-P</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), energy costs of running LLMs is quickly becoming a critical concern. However, precisely measuring the energy consumption of LLMs is often infeasible because hardware-based power monitors are not always accessible and software-based energy measurement tools are not accurate. While various prediction techniques have been developed to estimate LLM energy consumption, these approaches are limited to single-GPU environments and thus are not applicable to modern LLM inference which is typically parallelized across multiple GPUs. In this work, we remedy this gap and introduce PIE-P, a fine-grained energy prediction framework for multi-GPU inference, including tensor, pipeline, and data parallelism. Predicting the energy under parallelized inference is complicated by the non-determinism in inter-GPU communication, additional communication overheads, and difficulties in isolating energy during the communication/synchronization phase. We develop a scalable prediction framework that addresses these issues via precise sampling, fine-grained modeling of inter-GPU communication, and careful accounting of parallelization overhead. Our evaluation results show that PIE-P yields accurate and fine-grained energy predictions across parallelism strategies, significantly outperforming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24850v2">Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 22 pages, 10 figures. Code available at https://github.com/Tim-Siu/reinforcement-distillation
    </div>
    <details class="paper-abstract">
      Recent advances in model distillation show that data from advanced reasoning models can effectively train smaller student models. However, standard practices discard incorrect reasoning traces -- valuable, yet underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? We employ a two-stage training recipe: first, Supervised Fine-Tuning (SFT) on positive traces, followed by a refinement stage using both positive and negative traces. We find that a simple REINFORCE-style objective, which we term the Reinforcement Distillation (REDI) objective, outperforms established preference optimization methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate the effectiveness of this approach. Notably, our Qwen-REDI-1.5B model, trained on just 131k traces from the open Open-R1 dataset, achieves an 83.1% score on MATH-500. Its performance matches that of DeepSeek-R1-Distill-Qwen-1.5B, a model trained on 800k proprietary data. This result showcases the remarkable data efficiency of utilizing previously discarded negative traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12476v1">HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale and new GPUs are released even more frequently, there is an increasing demand for LLM post-training in heterogeneous environments to fully leverage underutilized mid-range or previous-generation GPUs across regions and alleviate the shortage of homogeneous high-end GPUs within a single region. However, achieving high-performance reinforcement learning (RL) training for LLMs on such computing resources remains challenging because the workflow involves multiple models and tasks with complex computation and data dependencies. In this paper, we present HetRL, a distributed system for efficient RL training in infrastructures with heterogeneous GPUs and networks. HetRL formulates the scheduling of RL training in heterogeneous environments as a constrained joint optimization problem and introduces a novel scheduling algorithm that (1) decomposes the complex search space with a multi-level search framework; and (2) allocates the search budget via successive halving. Our extensive evaluation, consuming 20,000 GPU-hours, shows that HetRL delivers up to 9.17x the throughput of state-of-the-art systems, and 3.17x on average, under various workloads and settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14285v3">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted at the 11th IEEE WIECON-ECE 2025
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00057v2">Incoherence as Oracle-less Measure of Error in LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted at AAAI'26 (extended version). 8 pages + refs and appendix
    </div>
    <details class="paper-abstract">
      Generating code from a natural language programming task is one of the most successful applications of Large Language Models (LLMs). Yet, the generated program may be buggy. Without an oracle, such as an existing, correct implementation or a formal specification, can we somehow estimate how likely the generated program is correct? In this paper, we propose a measure of incorrectness, called *incoherence*, that can be estimated efficiently in the absence of an oracle and allows us to establish a lower bound on the error, i.e., the probability that the LLM-generated program for that specification is incorrect. In our experiments, our incoherence-based methodology can automatically identify about two-thirds of incorrect programs without reports of false positives for the average task. In fact, *an oracle-based evaluation of LLMs can be reliably replaced by an incoherence-based evaluation*. In particular, we find a very strong agreement between the ranking of LLMs by the number of programs deemed correct via an oracle (pass@1) and the ranking of LLMs by the number of programs deemed correct via incoherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17904v2">BreakFun: Jailbreaking LLMs via Schema Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06380v2">Protecting Bystander Privacy via Selective Hearing in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Dataset: https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench
    </div>
    <details class="paper-abstract">
      Audio Large language models (LLMs) are increasingly deployed in the real world, where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences did not consider. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures, including both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. In addition, we propose Selective Efficacy (SE), a novel metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LLMs reveals substantial bystander privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we also present Bystander Privacy Fine-Tuning (BPFT), a novel training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. We show that BPFT yields substantial gains, achieving an absolute 47% higher bystander accuracy under selective mode and an absolute 16% higher SE compared to Gemini 2.5 Pro, which is the best audio LLM without BPFT. Together, SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12313v1">Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 17 pages, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08697v3">LLMs as Span Annotators: A Comparative Study of LLMs and Humans</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Span annotation - annotating specific text features at the span level - can be used to evaluate texts where single-score metrics fail to provide actionable feedback. Until recently, span annotation was done by human annotators or fine-tuned models. In this paper, we study whether large language models (LLMs) can serve as an alternative to human annotators. We compare the abilities of LLMs to skilled human annotators on three span annotation tasks: evaluating data-to-text generation, identifying translation errors, and detecting propaganda techniques. We show that overall, LLMs have only moderate inter-annotator agreement (IAA) with human annotators. However, we demonstrate that LLMs make errors at a similar rate as skilled crowdworkers. LLMs also produce annotations at a fraction of the cost per output annotation. We release the dataset of over 40k model and human span annotations for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v1">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 14 pages, 20 figures, conference
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12281v1">Cognitive-YOLO: LLM-Driven Architecture Synthesis from First Principles of Data for Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 12 pages, 4 figures, 3 ttables
    </div>
    <details class="paper-abstract">
      Designing high-performance object detection architectures is a complex task, where traditional manual design is time-consuming and labor-intensive, and Neural Architecture Search (NAS) is computationally prohibitive. While recent approaches using Large Language Models (LLMs) show promise, they often function as iterative optimizers within a search loop, rather than generating architectures directly from a holistic understanding of the data. To address this gap, we propose Cognitive-YOLO, a novel framework for LLM-driven architecture synthesis that generates network configurations directly from the intrinsic characteristics of the dataset. Our method consists of three stages: first, an analysis module extracts key meta-features (e.g., object scale distribution and scene density) from the target dataset; second, the LLM reasons upon these features, augmented with state-of-the-art components retrieved via Retrieval-Augmented Generation (RAG), to synthesize the architecture into a structured Neural Architecture Description Language (NADL); finally, a compiler instantiates this description into a deployable model. Extensive experiments on five diverse object detection datasets demonstrate that our proposed Cognitive-YOLO consistently generates superior architectures, achieving highly competitive performance and demonstrating a superior performance-per-parameter trade-off compared to strong baseline models across multiple benchmarks. Crucially, our ablation studies prove that the LLM's data-driven reasoning is the primary driver of performance, demonstrating that a deep understanding of data "first principles" is more critical for achieving a superior architecture than simply retrieving SOTA components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14479v5">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12224v1">Cluster-guided LLM-Based Anonymization of Software Analytics Data: Studying Privacy-Utility Trade-offs in JIT Defect Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      The increasing use of machine learning (ML) for Just-In-Time (JIT) defect prediction raises concerns about privacy leakage from software analytics data. Existing anonymization methods, such as tabular transformations and graph perturbations, often overlook contextual dependencies among software metrics, leading to suboptimal privacy-utility tradeoffs. Leveraging the contextual reasoning of Large Language Models (LLMs), we propose a cluster-guided anonymization technique that preserves contextual and statistical relationships within JIT datasets. Our method groups commits into feature-based clusters and employs an LLM to generate context-aware parameter configurations for each commit cluster, defining alpha-beta ratios and churn mixture distributions used for anonymization. Our evaluation on six projects (Cassandra, Flink, Groovy, Ignite, OpenStack, and Qt) shows that our LLM-based approach achieves privacy level 2 (IPR >= 80 percent), improving privacy by 18 to 25 percent over four state-of-the-art graph-based anonymization baselines while maintaining comparable F1 scores. Our results demonstrate that LLMs can act as adaptive anonymization engines when provided with cluster-specific statistical information about similar data points, enabling context-sensitive and privacy-preserving software analytics without compromising predictive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12177v1">Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted for publication in the proceedings of the IEEE International Conference on Big Data (IEEE BigData 2025)
    </div>
    <details class="paper-abstract">
      Indoor navigation remains a critical challenge for people with visual impairments. The current solutions mainly rely on infrastructure-based systems, which limit their ability to navigate safely in dynamic environments. We propose a novel navigation approach that utilizes a foundation model to transform floor plans into navigable knowledge graphs and generate human-readable navigation instructions. Floorplan2Guide integrates a large language model (LLM) to extract spatial information from architectural layouts, reducing the manual preprocessing required by earlier floorplan parsing methods. Experimental results indicate that few-shot learning improves navigation accuracy in comparison to zero-shot learning on simulated and real-world evaluations. Claude 3.7 Sonnet achieves the highest accuracy among the evaluated models, with 92.31%, 76.92%, and 61.54% on the short, medium, and long routes, respectively, under 5-shot prompting of the MP-1 floor plan. The success rate of graph-based spatial structure is 15.4% higher than that of direct visual reasoning among all models, which confirms that graphical representation and in-context learning enhance navigation performance and make our solution more precise for indoor navigation of Blind and Low Vision (BLV) users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12154v1">Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12117v1">Citation-Grounded Code Comprehension: Preventing LLM Hallucination Through Hybrid Retrieval and Graph-Augmented Context</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Large language models have become essential tools for code comprehension, enabling developers to query unfamiliar codebases through natural language interfaces. However, LLM hallucination, generating plausible but factually incorrect citations to source code, remains a critical barrier to reliable developer assistance. This paper addresses the challenges of achieving verifiable, citation grounded code comprehension through hybrid retrieval and lightweight structural reasoning. Our work is grounded in systematic evaluation across 30 Python repositories with 180 developer queries, comparing retrieval modalities, graph expansion strategies, and citation verification mechanisms. We find that challenges of citation accuracy arise from the interplay between sparse lexical matching, dense semantic similarity, and cross file architectural dependencies. Among these, cross file evidence discovery is the largest contributor to citation completeness, but it is largely overlooked because existing systems rely on pure textual similarity without leveraging code structure. We advocate for citation grounded generation as an architectural principle for code comprehension systems and demonstrate this need by achieving 92 percent citation accuracy with zero hallucinations. Specifically, we develop a hybrid retrieval system combining BM25 sparse matching, BGE dense embeddings, and Neo4j graph expansion via import relationships, which outperforms single mode baselines by 14 to 18 percentage points while discovering cross file evidence missed by pure text similarity in 62 percent of architectural queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17833v2">Learning to Debug: LLM-Organized Knowledge Trees for Solving RTL Assertion Failures</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Debugging is the dominant cost in modern hardware verification, where assertion failures are among the most frequent and expensive to resolve. While Large Language Models (LLMs) show promise, they often fail to capture the precise, reusable expertise that engineers apply, leading to inaccurate responses. We propose GROVE, a hierarchical knowledge management framework that learns and organizes reusable debugging expertise into an LLM-organized knowledge tree for solving assertion failures. GROVE distills debugging knowledge from prior cases and organizes it into a vertical tree of configurable depth, with each node encoding a concise knowledge item and explicit applicability conditions. During training, GROVE uses a parallel, gradient-free loop where an LLM proposes tree modifications as structured JSON edits by learning from the cases. At test time, a budget-aware iterative zoom is performed to navigate the tree, retrieving a small set of applicable knowledge items that guide a base LLM's hypothesis generation and fix proposals. Evaluated on a suite of assertion-failure cases, GROVE delivers consistent gains in pass@1 and pass@5, demonstrating the value of structured knowledge evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.11641v4">Revolutionizing Finance with LLMs: An Overview of Applications and Insights</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) like ChatGPT have seen considerable advancements and have been applied in diverse fields. Built on the Transformer architecture, these models are trained on extensive datasets, enabling them to understand and generate human language effectively. In the financial domain, the deployment of LLMs is gaining momentum. These models are being utilized for automating financial report generation, forecasting market trends, analyzing investor sentiment, and offering personalized financial advice. Leveraging their natural language processing capabilities, LLMs can distill key insights from vast financial data, aiding institutions in making informed investment choices and enhancing both operational efficiency and customer satisfaction. In this study, we provide a comprehensive overview of the emerging integration of LLMs into various financial tasks. Additionally, we conducted holistic tests on multiple financial tasks through the combination of natural language instructions. Our findings show that GPT-4 effectively follow prompt instructions across various financial tasks. This survey and evaluation of LLMs in the financial domain aim to deepen the understanding of LLMs' current role in finance for both financial practitioners and LLM researchers, identify new research and application prospects, and highlight how these technologies can be leveraged to solve practical challenges in the finance industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05578v2">The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the need to minimize harmful memorization with model utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16368v3">SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard. To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions. We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v2">BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11270v1">A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025 Workshop: Multi-Turn Interactions in Large Language Models. 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Applying reinforcement learning (RL) to real-world tasks requires converting informal descriptions into a formal Markov decision process (MDP), implementing an executable environment, and training a policy agent. Automating this process is challenging due to modeling errors, fragile code, and misaligned objectives, which often impede policy training. We introduce an agentic large language model (LLM)-based framework for automated MDP modeling and policy generation (A-LAMP), that automatically translates free-form natural language task descriptions into an MDP formulation and trained policy. The framework decomposes modeling, coding, and training into verifiable stages, ensuring semantic alignment throughout the pipeline. Across both classic control and custom RL domains, A-LAMP consistently achieves higher policy generation capability than a single state-of-the-art LLM model. Notably, even its lightweight variant, which is built on smaller language models, approaches the performance of much larger models. Failure analysis reveals why these improvements occur. In addition, a case study also demonstrates that A-LAMP generates environments and policies that preserve the task's optimality, confirming its correctness and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11261v1">Leveraging LLMs for Title and Abstract Screening for Systematic Review: A Cost-Effective Dynamic Few-Shot Learning Approach</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 22 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Systematic reviews are a key component of evidence-based medicine, playing a critical role in synthesizing existing research evidence and guiding clinical decisions. However, with the rapid growth of research publications, conducting systematic reviews has become increasingly burdensome, with title and abstract screening being one of the most time-consuming and resource-intensive steps. To mitigate this issue, we designed a two-stage dynamic few-shot learning (DFSL) approach aimed at improving the efficiency and performance of large language models (LLMs) in the title and abstract screening task. Specifically, this approach first uses a low-cost LLM for initial screening, then re-evaluates low-confidence instances using a high-performance LLM, thereby enhancing screening performance while controlling computational costs. We evaluated this approach across 10 systematic reviews, and the results demonstrate its strong generalizability and cost-effectiveness, with potential to reduce manual screening burden and accelerate the systematic review process in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00862v3">HBLLM: Wavelet-Enhanced High-Fidelity 1-Bit Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      We introduce HBLLM, a wavelet-enhanced high-fidelity $1$-bit post-training quantization method for Large Language Models (LLMs). By leveraging Haar wavelet transforms to enhance expressive capacity through frequency decomposition, HBLLM significantly improves quantization fidelity while maintaining minimal overhead. This approach features two innovative structure-aware grouping strategies: (1) frequency-aware multi-parameter intra-row grouping and (2) $\ell_2$-norm-based saliency-driven column selection. For non-salient weights, a shared mean is employed across quantization groups within each frequency band to optimize storage efficiency. Experiments conducted on the OPT and LLaMA models demonstrate that HBLLM achieves state-of-the-art performance in $1$-bit quantization, attaining a perplexity of $6.71$ on LLaMA$2$-$13$B with an average weight storage of only $1.08$ bits. Code available at: https://github.com/Yeyke/HBLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11221v1">Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery: Sublinear Memory Growth for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 6 pages, 3 tables , 1 figure
    </div>
    <details class="paper-abstract">
      We present Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery (ASR-KF-EGR), a training-free inference-time framework for efficient large language model generation. Our method introduces a reversible soft-freeze mechanism that temporarily suspends key-value (KV) updates for low-importance tokens identified within a sliding attention window. Unlike eviction-based approaches that permanently discard context, ASR-KF-EGR preserves all tokens in off-GPU storage and restores them on demand. We extend the framework with sublinear freeze scheduling, where freeze duration grows sublinearly with repeated low-importance detections, preventing over-aggressive compression. Preliminary experiments on LLaMA-3 8B demonstrate 55-67% reduction in active KV cache size while maintaining generation quality and passing needle-in-haystack retrieval tests. The method is architecture-agnostic, requires no fine-tuning, and provides a practical solution for memory-constrained deployment of long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10257v2">Reject or Not?: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before any downstream processing. To address the limited query-rejection capability of current voice assistants, this paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together with a personalized query-rejection method based on large language models. On the data side, we construct the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds, valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides a reproducible data foundation, evaluation standard and extensible technical framework for reliability research in smart-home voice interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23453v2">Counterfactual Evaluation for Blind Attack Detection in LLM-based Evaluation Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      This paper investigates defenses for LLM-based evaluation systems against prompt injection. We formalize a class of threats called blind attacks, where a candidate answer is crafted independently of the true answer to deceive the evaluator. To counter such attacks, we propose a framework that augments Standard Evaluation (SE) with Counterfactual Evaluation (CFE), which re-evaluates the submission against a deliberately false ground-truth answer. An attack is detected if the system validates an answer under both standard and counterfactual conditions. Experiments show that while standard evaluation is highly vulnerable, our SE+CFE framework significantly improves security by boosting attack detection with minimal performance trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03846v3">Do LLM Evaluators Prefer Themselves for a Reason?</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Added LMArena experiments
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference harmful, or does it simply reflect the genuinely higher-quality outputs of stronger models? Answering this has been difficult as prior works mostly relied on subjective tasks that lack an objective ground truth, meaning that either preference can be reasonably justified. To address this ambiguity, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful (favoring objectively worse responses) from legitimate (favoring genuinely superior ones) self-preference. Our large-scale experiments across 7 model families reveal three key findings: (1) While stronger models exhibit greater self-preference, much of this preference aligns with objectively superior performance, indicating stronger models prefer themselves mostly legitimately. (2) Harmful self-preference persists when evaluator models err as generators, and stronger models display more pronounced harmful self-preference bias when they do err. This suggests stronger models struggle more to recognize when they are wrong. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. Additionally, we experiment with LMArena and show that our findings extend beyond verifiable benchmarks to real-world, subjective domains. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12072v1">VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Arxiv Submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used to generate synthetic datasets for the evaluation and training of downstream models. However, prior work has noted that such generated data lacks diversity. In this paper, we propose Voyager, a novel principled approach to generate diverse datasets. Our approach is iterative and directly optimizes a mathematical quantity that optimizes the diversity of the dataset using the machinery of determinantal point processes. Furthermore, our approach is training-free, applicable to closed-source models, and scalable. In addition to providing theoretical justification for the working of our method, we also demonstrate through comprehensive experiments that Voyager significantly outperforms popular baseline approaches by providing a 1.5-3x improvement in diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25540v2">RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Manual labeling limits the scale, accuracy, and timeliness of patient outcomes research in radiation oncology. We present RadOnc-GPT, an autonomous large language model (LLM)-based agent capable of independently retrieving patient-specific information, iteratively assessing evidence, and returning structured outcomes. Our evaluation explicitly validates RadOnc-GPT across two clearly defined tiers of increasing complexity: (1) a structured quality assurance (QA) tier, assessing the accurate retrieval of demographic and radiotherapy treatment plan details, followed by (2) a complex clinical outcomes labeling tier involving determination of mandibular osteoradionecrosis (ORN) in head-and-neck cancer patients and detection of cancer recurrence in independent prostate and head-and-neck cancer cohorts requiring combined interpretation of structured and unstructured patient data. The QA tier establishes foundational trust in structured-data retrieval, a critical prerequisite for successful complex clinical outcome labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15455v2">CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise impressive capabilities, yet their multi-billion-parameter scale makes on-device or low-resource deployment prohibitive. Mixed-precision quantization offers a compelling solution, but existing methods struggle when the average precision drops below four bits, as they rely on isolated, layer-specific metrics that overlook critical inter-layer interactions affecting overall performance. To address these limitations, we first frame the mixed-precision quantization problem as a cooperative game among layers and introduce Shapley-based Progressive Quantization Estimation (SPQE) to efficiently obtain accurate Shapley estimates of layer sensitivities and inter-layer interactions. Leveraging the SPQE estimates, we propose Cooperative Game Inspired Mixed-Precision Quantization (CoopQ) which translates these Shapley estimates into a binary quadratic optimization formulation, assigning either 2 or 4-bit precision to layers under strict memory constraints. Comprehensive experiments conducted on Llama-3, Gemma-2, and Qwen-3 models across three independent PTQ backends (Quanto, HQQ, GPTQ) demonstrate CoopQ's scalability and consistently superior performance compared to methods relying solely on isolated metrics. Across average precisions spanning 4 bit down to 2 bit, CoopQ cuts Perplexity by 20 - 80 % relative to the best baseline, with the margin growing as the bit-width tightens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02718v3">Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput. Our code is available at https://github.com/fzwark/PORT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11986v1">Learning to Extract Context for Context-Aware LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11755v1">SUMFORU: An LLM-Based Review Summarization Framework for Personalized Purchase Decision Support</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Code available at https://github.com/Harry20030331/SumForU
    </div>
    <details class="paper-abstract">
      Online product reviews contain rich but noisy signals that overwhelm users and hinder effective decision-making. Existing LLM-based summarizers remain generic and fail to account for individual preferences, limiting their practical utility. We propose SUMFORU, a steerable review summarization framework that aligns outputs with explicit user personas to support personalized purchase decisions. Our approach integrates a high-quality data pipeline built from the Amazon 2023 Review Dataset with a two-stage alignment procedure: (1) persona-aware Supervised Fine-Tuning (SFT) via asymmetric knowledge distillation, and (2) Reinforcement Learning with AI Feedback (RLAIF) using a preference estimator to capture fine-grained, persona-relevant signals. We evaluate the model across rule-based, LLM-based, and human-centered metrics, demonstrating consistent improvements in consistency, grounding, and preference alignment. Our framework achieves the highest performance across all evaluation settings and generalizes effectively to unseen product categories. Our results highlight the promise of steerable pluralistic alignment for building next-generation personalized decision-support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01374v4">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed Reasoning Compiler) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14741v1">Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Backdoor attacks embed malicious behaviors into Large Language Models (LLMs), enabling adversaries to trigger harmful outputs or bypass safety controls. However, the persistence of the implanted backdoors under user-driven post-deployment continual fine-tuning has been rarely examined. Most prior works evaluate the effectiveness and generalization of implanted backdoors only at releasing and empirical evidence shows that naively injected backdoor persistence degrades after updates. In this work, we study whether and how implanted backdoors persist through a multi-stage post-deployment fine-tuning. We propose P-Trojan, a trigger-based attack algorithm that explicitly optimizes for backdoor persistence across repeated updates. By aligning poisoned gradients with those of clean tasks on token embeddings, the implanted backdoor mapping is less likely to be suppressed or forgotten during subsequent updates. Theoretical analysis shows the feasibility of such persistent backdoor attacks after continual fine-tuning. And experiments conducted on the Qwen2.5 and LLaMA3 families of LLMs, as well as diverse task sequences, demonstrate that P-Trojan achieves over 99% persistence while preserving clean-task accuracy. Our findings highlight the need for persistence-aware evaluation and stronger defenses in realistic model adaptation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15765v1">Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Data is a critical asset for training large language models (LLMs), alongside compute resources and skilled workers. While some training data is publicly available, substantial investment is required to generate proprietary datasets, such as human preference annotations or to curate new ones from existing sources. As larger datasets generally yield better model performance, two natural questions arise. First, how can data owners make informed decisions about curation strategies and data sources investment? Second, how can multiple data owners collaboratively pool their resources to train superior models while fairly distributing the benefits? This problem, data valuation, which is not specific to large language models, has been addressed by the machine learning community through the lens of cooperative game theory, with the Shapley value being the prevalent solution concept. However, computing Shapley values is notoriously expensive for data valuation, typically requiring numerous model retrainings, which can become prohibitive for large machine learning models. In this work, we demonstrate that this computational challenge is dramatically simplified for LLMs trained with Direct Preference Optimization (DPO). We show how the specific mathematical structure of DPO enables scalable Shapley value computation. We believe this observation unlocks many applications at the intersection of data valuation and large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10061v2">Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present Text2Graph, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark Text2Graph on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11689v1">Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Supplementary material in https://github.com/mavivi95/resilience_humans_vs_LLM/blob/main/Supplementary_File.pdf
    </div>
    <details class="paper-abstract">
      This paper presents a comparative analysis of cooperative resilience in multi-agent systems, defined as the ability to anticipate, resist, recover from, and transform to disruptive events that affect collective well-being. We focus on mixed-motive social dilemmas instantiated as a \textit{Tragedy of the Commons} environment from the Melting Pot suite, where we systematically compare human groups and Large Language Model (LLM)-based agents, each evaluated with and without explicit communication. Cooperative resilience is assessed under a continuously disruptive condition induced by a persistent unsustainable consumption bot, together with intermittent environmental shocks implemented as stochastic removal of shared resources across scenarios. This experimental design establishes a benchmark for cooperative resilience across agent architectures and interaction modalities, constituting a key step toward systematically comparing humans and LLM-based agents. Using this framework, we find that human groups with communication achieve the highest cooperative resilience compared to all other groups. Communication also improves the resilience of LLM agents, but their performance remains below human levels. Motivated by the performance of humans, we further examine a long-horizon setting with harsher environmental conditions, where humans sustain the shared resource and maintain high resilience in diverse disruption scenarios. Together, these results suggest that human decision-making under adverse social conditions can inform the design of artificial agents that promote prosocial and resilient behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11661v1">From Verification Burden to Trusted Collaboration: Design Goals for LLM-Assisted Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in academic writing practices. Although numerous studies have explored how researchers employ these tools for scientific writing, their concrete implementation, limitations, and design challenges within the literature review process remain underexplored. In this paper, we report a user study with researchers across multiple disciplines to characterize current practices, benefits, and \textit{pain points} in using LLMs to investigate related work. We identified three recurring gaps: (i) lack of trust in outputs, (ii) persistent verification burden, and (iii) requiring multiple tools. This motivates our proposal of six design goals and a high-level framework that operationalizes them through improved related papers visualization, verification at every step, and human-feedback alignment with generation-guided explanations. Overall, by grounding our work in the practical, day-to-day needs of researchers, we designed a framework that addresses these limitations and models real-world LLM-assisted writing, advancing trust through verifiable actions and fostering practical collaboration between researchers and AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11550v1">PD-Swap: Prefill-Decode Logic Swapping for End-to-End LLM Inference on Edge FPGAs via Dynamic Partial Reconfiguration</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Aggressively quantized large language models (LLMs), such as BitNet-style 1.58-bit Transformers with ternary weights, make it feasible to deploy generative AI on low-power edge FPGAs. However, as prompts grow to tens of thousands of tokens, edge hardware performance drops sharply with sequence length due to quadratic prefill cost and rapidly increasing KV-cache bandwidth demands, making inference latency of longer context length a first-order system concern. Recent studies on LLMs expose a fundamental prefill-decode asymmetry: prefill is compute-bound and dominated by dense matrix-matrix operations, whereas decoding is memory-bandwidth-bound and dominated by KV-cache traffic. A static accelerator must provision resources and a single dataflow for both regimes, leading to duplicated attention logic, underutilized fabric, and tight LUT/URAM limits that cap model size and usable context. We propose a prefill--decode disaggregated LLM accelerator, PD-Swap, that uses Dynamic Partial Reconfiguration (DPR) to time-multiplex the attention module on edge FPGAs. The core table-lookup ternary matrix multiplication and weight-buffering engines remain static, while the attention subsystem is a reconfigurable partition with two phase-specialized architectures: a compute-heavy, token-parallel prefill engine and a bandwidth-optimized, KV-cache-centric decoding engine. A roofline-inspired model and design space exploration jointly optimize reconfigurable-region size, parallelism under reconfiguration and routability constraints, and reconfiguration latency is hidden by computation latency. PD-Swap achieves up to 27~tokens/s decoding throughput, outperforming prior state-of-the-art works by 1.3x--2.1x (larger gains at longer context lengths), without extra area cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11509v1">Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10493v2">Decoding Human-LLM Collaboration in Coding: An Empirical Study of Multi-Turn Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly acting as dynamic conversational interfaces, supporting multi-turn interactions that mimic human-like conversation and facilitate complex tasks like coding. While datasets such as LMSYS-Chat-1M and WildChat capture real-world user-LLM conversations, few studies systematically explore the mechanisms of human-LLM collaboration in coding scenarios. What tortuous paths do users experience during the interaction process? How well do the LLMs follow instructions? Are users satisfied? In this paper, we conduct an empirical analysis on human-LLM coding collaboration using LMSYS-Chat-1M and WildChat datasets to explore the human-LLM collaboration mechanism, LLMs' instruction following ability, and human satisfaction. This study yields interesting findings: 1) Task types shape interaction patterns(linear, star and tree), with code quality optimization favoring linear patterns, design-driven tasks leaning toward tree structures, and queries preferring star patterns; 2) Bug fixing and code refactoring pose greater challenges to LLMs' instruction following, with non-compliance rates notably higher than in information querying; 3) Code quality optimization and requirements-driven development tasks show lower user satisfaction, whereas structured knowledge queries and algorithm designs yield higher levels. These insights offer recommendations for improving LLM interfaces and user satisfaction in coding collaborations, while highlighting avenues for future research on adaptive dialogue systems. We believe this work broadens understanding of human-LLM synergies and supports more effective AI-assisted development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11421v1">Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Accepted to AAAI 2026 Workshop on Trust and Control in Agentic AI (TrustAgent)
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate strong reasoning and generation abilities, yet their behavior in multi-turn tasks often lacks reliability and verifiability. We present a task completion framework that enables LLM-based agents to act under explicit behavioral guidance in environments described by reinforcement learning formalisms with defined observation, action, and reward signals. The framework integrates three components: a lightweight task profiler that selects reasoning and generation strategies, a reasoning module that learns verifiable observation - action mappings, and a generation module that enforces constraint-compliant outputs through validation or deterministic synthesis. We show that as the agent interacts with the environment, these components co-evolve, yielding trustworthy behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06393v2">Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance on many natural language tasks, yet their generalisation under structured perturbations of logical rule systems remains insufficiently characterised. We present a controlled evaluation framework that probes reasoning reliability through four stress tests: (1) rule deletion, removing redundant versus essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites based on equivalence laws (contraposition, double negation, implication-to-disjunction, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that composes 2--5 transformations. Across three representative model families -- BERT, Qwen2, and LLaMA-like models -- all models attain Acc$=1.0000$ on the base split and show no degradation under redundant rule deletion. In contrast, essential rule deletion yields a pronounced decrease to near-chance performance, and injecting explicit contradictions reduces accuracy to 0.0000. Under logic-preserving rewrites, accuracy is largely preserved for single-law transformations with only small degradations in a few cases, whereas multi-law stacking exposes model-dependent sensitivity: BERT matches the base condition, TinyLlama shows only marginal degradation, and Qwen2 exhibits a substantial drop. Overall, the results indicate that contemporary LLMs are generally stable under semantic-preserving reformulations, yet remain brittle to missing or inconsistent evidence and may degrade under composed logical transformations depending on the model family. The proposed framework provides a concise diagnostic tool for isolating these failure modes and for evaluating logical generalisation beyond surface-form variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11402v1">REMODEL-LLM: Transforming C code to Java using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      The automated translation of C code to Java code is a notoriously difficult task, fraught with challenges stemming from fundamental paradigm shifts (procedural vs. Object Oriented), memory models (manual pointers vs. Garbage Collection), and incompatible data types. This paper investigates the efficacy of 19 small, quantized LLMs (under 20 billion parameters) for the C to Java translation task. We use a novel, hybrid pipeline that leverages Abstract Syntax Trees (ASTs) for semantic decomposition and employs a highly constrained, rule based prompting strategy. The results are stark: a clear multi tiered performance divide emerged. The vast majority of models (Tier 3, e.g., llama3.1, gemma3, starcoder2) failed 100\% of the tests, proving incapable of generating even basic, runnable Java boilerplate. A small middle tier (Tier 2, e.g., mistral-nemo and mistral) produced runnable code but was plagued by dangerous semantic failures and wrong translations. Only three models (Tier 1: phi4, deepseek-coder-v2, codeqwen) proved viable, passing over 50\% of the test suite. Even these top models failed on the most complex C concepts, such as function pointers, sizeof, and enum logic, revealing a hard ceiling for the reasoning capabilities of current quantized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.07020v2">Driving Through Uncertainty: Risk-Averse Control with LLM Commonsense for Autonomous Driving under Perception Deficits</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Existing protocols typically default to entirely risk-avoidant actions such as immediate stops, which are detrimental to navigation goals and lack flexibility for rare driving scenarios. Yet, in cases of minor risk, halting the vehicle may be unnecessary, and more adaptive responses are preferable. In this paper, we propose LLM-RCO, a risk-averse framework leveraging large language models (LLMs) to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules interacting with the dynamic driving environment: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator, enabling proactive and context-aware actions in such challenging conditions. To enhance the driving decision-making of LLMs, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, annotated for LLM fine-tuning in hazard detection and motion planning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that LLM-RCO promotes proactive maneuvers over purely risk-averse actions in perception deficit scenarios, underscoring its value for boosting autonomous driving resilience against perception loss challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11388v1">Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 To appear at IEEE Big Data 2025
    </div>
    <details class="paper-abstract">
      We investigated the impact of data selection on machine translation fine-tuning for open LLMs. Using Japanese-English corpora, we compare five selectors: TF-IDF, COMET Kiwi, QuRate, FD-Score, and random selection, under controlled training conditions. We observed that semantic selectors consistently outperform lexical and geometry-based heuristics, and that even when the selected data differ by less than 3%, the impact on model performance is substantial, underscoring the sensitivity of fine-tuning to data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10501v2">Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17281v4">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10611v1">Phythesis: Physics-Guided Evolutionary Scene Synthesis for Energy-Efficient Data Center Design via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Data center (DC) infrastructure serves as the backbone to support the escalating demand for computing capacity. Traditional design methodologies that blend human expertise with specialized simulation tools scale poorly with the increasing system complexity. Recent studies adopt generative artificial intelligence to design plausible human-centric indoor layouts. However, they do not consider the underlying physics, making them unsuitable for the DC design that sets quantifiable operational objectives and strict physical constraints. To bridge the gap, we propose Phythesis, a novel framework that synergizes large language models (LLMs) and physics-guided evolutionary optimization to automate simulation-ready (SimReady) scene synthesis for energy-efficient DC design. Phythesis employs an iterative bi-level optimization architecture, where (i) the LLM-driven optimization level generates physically plausible three-dimensional layouts and self-criticizes them to refine the scene topology, and (ii) the physics-informed optimization level identifies the optimal asset parameters and selects the best asset combination. Experiments on three generation scales show that Phythesis achieves 57.3% generation success rate increase and 11.5% power usage effectiveness (PUE) improvement, compared with the vanilla LLM-based solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10610v1">Thinking While Driving: A Concurrent Framework for Real-Time, LLM-Based Adaptive Routing</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We present Thinking While Driving, a concurrent routing framework that integrates LLMs into a graph-based traffic environment. Unlike approaches that require agents to stop and deliberate, our system enables LLM-based route planning while agents are moving, significantly reducing intersection wait times. Under high traffic, agents average just 0.75 seconds of decision latency. To coordinate many agents in real-time, we implement a non-blocking asynchronous architecture using Unity coroutines and a dedicated request manager. The environment is a weighted undirected graph with live congestion metrics, updated continuously by the agents to enable shared perception. Our results show LLM-driven agents can dynamically adapt to traffic, reroute around congestion, and exhibit behaviors beyond static pathfinding, all while maintaining real-time performance. This work provides a reproducible framework for future research in adaptive routing and multi-agent cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10551v1">LLM-Auction: Generative Auction towards LLM-Native Advertising</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) necessitates novel monetization strategies, among which LLM-native advertising has emerged as a promising paradigm by naturally integrating advertisement within LLM-generated responses. However, this paradigm fundamentally shifts the auction object from discrete ad slots to the distribution over LLM outputs, posing new challenges for designing auction mechanisms. Existing mechanisms for LLM-native advertising adopt frameworks that decouple auction and generation, which either ignore externalities or require multiple LLM inferences for ad allocation, rendering them impractical for industrial scenarios. To address these challenges, we propose LLM-Auction, which to the best of our knowledge is the first learning-based generative auction mechanism that integrates auction and LLM generation for LLM-native advertising. By formulating the allocation optimization as a preference alignment problem between LLM outputs and the mechanism's objective which reflects both advertisers' expected value and user experience, we introduce Iterative Reward-Preference Optimization (IRPO) algorithm that alternately optimizes the reward model and the LLM. This approach enables the LLM to inherently model allocation externalities without any extra inference cost. We further identify the allocation monotonicity and continuity of LLM-Auction, which allows us to prove that a simple first-price payment rule exhibits favorable incentive properties. Additionally, we design an LLM-as-a-judge simulation environment to facilitate large-scale data construction and enable comprehensive quantitative evaluation of the mechanism's performance. Extensive quantitative and qualitative experiments demonstrate that LLM-Auction significantly outperforms existing baselines in allocation efficiency, while achieving the desired mechanism properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10545v1">XDoGE: Multilingual Data Reweighting to Enhance Language Inclusivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted and presented at the LLMs4All workshop at the IEEE BigData 2025 Conference, Macau - December 8-11, 2025
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) are trained on massive amounts of text data, primarily from a few dominant languages. Studies suggest that this over-reliance on high-resource languages, such as English, hampers LLM performance in mid- and low-resource languages. To mitigate this problem, we propose to (i) optimize the language distribution by training a small proxy model within a domain-reweighing DoGE algorithm that we extend to XDoGE for a multilingual setup, and (ii) rescale the data and train a full-size model with the established language weights either from scratch or within a continual pre-training phase (CPT). We target six languages possessing a variety of geographic and intra- and inter-language-family relations, namely, English and Spanish (high-resource), Portuguese and Catalan (mid-resource), Galician and Basque (low-resource). We experiment with Salamandra-2b, which is a promising model for these languages. We investigate the effects of substantial data repetition on minor languages and under-sampling on dominant languages using the IberoBench framework for quantitative evaluation. Finally, we release a new promising IberianLLM-7B-Instruct model centering on Iberian languages and English that we pretrained from scratch and further improved using CPT with the XDoGE weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10501v1">Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10493v1">Decoding Human-LLM Collaboration in Coding: An Empirical Study of Multi-Turn Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly acting as dynamic conversational interfaces, supporting multi-turn interactions that mimic human-like conversation and facilitate complex tasks like coding. While datasets such as LMSYS-Chat-1M and WildChat capture real-world user-LLM conversations, few studies systematically explore the mechanisms of human-LLM collaboration in coding scenarios. What tortuous paths do users experience during the interaction process? How well do the LLMs follow instructions? Are users satisfied? In this paper, we conduct an empirical analysis on human-LLM coding collaboration using LMSYS-Chat-1M and WildChat datasets to explore the human-LLM collaboration mechanism, LLMs' instruction following ability, and human satisfaction. This study yields interesting findings: 1) Task types shape interaction patterns(linear, star and tree), with code quality optimization favoring linear patterns, design-driven tasks leaning toward tree structures, and queries preferring star patterns; 2) Bug fixing and code refactoring pose greater challenges to LLMs' instruction following, with non-compliance rates notably higher than in information querying; 3) Code quality optimization and requirements-driven development tasks show lower user satisfaction, whereas structured knowledge queries and algorithm designs yield higher levels. These insights offer recommendations for improving LLM interfaces and user satisfaction in coding collaborations, while highlighting avenues for future research on adaptive dialogue systems. We believe this work broadens understanding of human-LLM synergies and supports more effective AI-assisted development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10487v1">LLM-Assisted AHP for Explainable Cyber Range Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Cyber Ranges (CRs) have emerged as prominent platforms for cybersecurity training and education, especially for Critical Infrastructure (CI) sectors that face rising cyber threats. One way to address these threats is through hands-on exercises that bridge IT and OT domains to improve defensive readiness. However, consistently evaluating whether a CR platform is suitable and effective remains a challenge. This paper proposes an evaluation framework for CRs, emphasizing mission-critical settings by using a multi-criteria decision-making approach. We define a set of evaluation criteria that capture technical fidelity, training and assessment capabilities, scalability, usability, and other relevant factors. To weight and aggregate these criteria, we employ the Analytic Hierarchy Process (AHP), supported by a simulated panel of multidisciplinary experts implemented through a Large Language Model (LLM). This LLM-assisted expert reasoning enables consistent and reproducible pairwise comparisons across criteria without requiring direct expert convening. The framework's output equals quantitative scores that facilitate objective comparison of CR platforms and highlight areas for improvement. Overall, this work lays the foundation for a standardized and explainable evaluation methodology to guide both providers and end-users of CRs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10485v1">From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that both graph neural network (GNN)-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by consistent data distributions and heuristic or partially noisy labels. In this study, we systematically evaluate two representative DL models-ReVeal and LineVul-across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each respective dataset, and their code representations are analyzed using t-SNE to uncover vulnerability related patterns. To assess realistic applicability, we deploy these models along with four pretrained LLMs, Claude 3.5 Sonnet, GPT-o3-mini, GPT-4o, and GPT-5 on a curated dataset, VentiVul, comprising 20 recently (May 2025) fixed vulnerabilities from the Linux kernel. Our experiments reveal that current models struggle to distinguish vulnerable from non-vulnerable code in representation space and generalize poorly across datasets with differing distributions. When evaluated on VentiVul, our newly constructed time-wise out-of-distribution dataset, performance drops sharply, with most models failing to detect vulnerabilities reliably. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations and higher-quality datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17552v3">Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10453v1">Grammaticality Judgments in Humans and Language Models: Revisiting Generative Grammar with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 2 figures
    </div>
    <details class="paper-abstract">
      What counts as evidence for syntactic structure? In traditional generative grammar, systematic contrasts in grammaticality such as subject-auxiliary inversion and the licensing of parasitic gaps are taken as evidence for an internal, hierarchical grammar. In this paper, we test whether large language models (LLMs), trained only on surface forms, reproduce these contrasts in ways that imply an underlying structural representation. We focus on two classic constructions: subject-auxiliary inversion (testing recognition of the subject boundary) and parasitic gap licensing (testing abstract dependency structure). We evaluate models including GPT-4 and LLaMA-3 using prompts eliciting acceptability ratings. Results show that LLMs reliably distinguish between grammatical and ungrammatical variants in both constructions, and as such support that they are sensitive to structure and not just linear order. Structural generalizations, distinct from cognitive knowledge, emerge from predictive training on surface forms, suggesting functional sensitivity to syntax without explicit encoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10449v1">When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10415v1">How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10370v1">LLM-Empowered Representation Learning for Emerging Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of recommending emerging items, whose interactions gradually accumulate over time. Existing methods often overlook this dynamic process, typically assuming that emerging items have few or even no historical interactions. Such an assumption oversimplifies the problem, as a good model must preserve the uniqueness of emerging items while leveraging their shared patterns with established ones. To address this challenge, we propose EmerFlow, a novel LLM-empowered representation learning framework that generates distinctive embeddings for emerging items. It first enriches the raw features of emerging items through LLM reasoning, then aligns these representations with the embedding space of the existing recommendation model. Finally, new interactions are incorporated through meta-learning to refine the embeddings. This enables EmerFlow to learn expressive embeddings for emerging items from only limited interactions. Extensive experiments across diverse domains, including movies and pharmaceuticals, show that EmerFlow consistently outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16528v2">Aligning ASR Evaluation with Human and LLM Judgments: Intelligibility Metrics Using Phonetic, Semantic, and NLI Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 5 pages, 2 figures, Interspeech 2025
    </div>
    <details class="paper-abstract">
      Traditional ASR metrics like WER and CER fail to capture intelligibility, especially for dysarthric and dysphonic speech, where semantic alignment matters more than exact word matches. ASR systems struggle with these speech types, often producing errors like phoneme repetitions and imprecise consonants, yet the meaning remains clear to human listeners. We identify two key challenges: (1) Existing metrics do not adequately reflect intelligibility, and (2) while LLMs can refine ASR output, their effectiveness in correcting ASR transcripts of dysarthric speech remains underexplored. To address this, we propose a novel metric integrating Natural Language Inference (NLI) scores, semantic similarity, and phonetic similarity. Our ASR evaluation metric achieves a 0.890 correlation with human judgments on Speech Accessibility Project data, surpassing traditional methods and emphasizing the need to prioritize intelligibility over error-based measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20068v2">JITServe: SLO-aware LLM Serving with Imprecise Request Information</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into applications ranging from interactive chatbots to multi-agent systems has introduced a wide spectrum of service-level objectives (SLOs) for responsiveness. These include latency-sensitive requests emphasizing per-token latency in streaming chat, deadline-sensitive requests requiring rapid full responses to trigger external tools, and compound requests with evolving dependencies across multiple LLM calls. Despite-or perhaps, because of-this workload diversity and unpredictable request information (e.g., response lengths and dependencies), existing request schedulers have focused on aggregate performance, unable to ensure application-level SLO needs. This paper presents JITServe, the first SLO-aware LLM serving system designed to maximize service goodput (e.g., the number of tokens meeting request SLOs) across diverse workloads. JITServe novelly schedules requests using imprecise request information and gradually relaxes this conservatism by refining request information estimates as generation progresses. It applies a grouped margin goodput maximization algorithm to allocate just enough serving bandwidth to satisfy each request's SLO just-in-time (JIT), maximizing residual capacity for others, while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees. Our evaluation across diverse realistic workloads, including chat, deep research, and agentic pipelines, shows that JITServe improves service goodput by 1.4x-6.3x, alternatively achieving 28.5%-83.2% resource savings, compared to state-of-the-art designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10324v1">EchoingPixels: Cross-Modal Adaptive Token Reduction for Efficient Audio-Visual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Audio-Visual Large Language Models (AV-LLMs) face prohibitive computational overhead from massive audio and video tokens. Token reduction, while extensively explored for video-only LLMs, is insufficient for the audio-visual domain, as these unimodal methods cannot leverage audio-visual cross-modal synergies. Furthermore, the distinct and dynamic information densities of audio and video render static budgets per modality suboptimal. How to perform token reduction on a joint audio-visual stream thus remains an unaddressed bottleneck. To fill this gap, we introduce EchoingPixels, a framework inspired by the coexistence and interaction of visuals and sound in real-world scenes. The core of our framework is the Cross-Modal Semantic Sieve (CS2), a module enabling early audio-visual interaction. Instead of compressing modalities independently, CS2 co-attends to the joint multimodal stream and reduces tokens from an entire combined pool of audio-visual tokens rather than using fixed budgets per modality. This single-pool approach allows it to adaptively allocate the token budget across both modalities and dynamically identify salient tokens in concert. To ensure this aggressive reduction preserves the vital temporal modeling capability, we co-design a Synchronization-Augmented RoPE (Sync-RoPE) to maintain critical temporal relationships for the sparsely selected tokens. Extensive experiments demonstrate that EchoingPixels achieves performance comparable to strong baselines using only 5-20% of the original tokens, with a 2-3x speedup and memory reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15257v2">When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-weight LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively disentangled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training methods such as supervised fine-tuning or reinforcement learning, our training-free language-reasoning disentanglement achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17208v2">A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03420v3">HarnessAgent: Scaling Automatic Fuzzing Harness Construction with Tool-Augmented LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based techniques have achieved notable progress in generating harnesses for program fuzzing. However, applying them to arbitrary functions (especially internal functions) \textit{at scale} remains challenging due to the requirement of sophisticated contextual information, such as specification, dependencies, and usage examples. State-of-the-art methods heavily rely on static or incomplete context provisioning, causing failure of generating functional harnesses. Furthermore, LLMs tend to exploit harness validation metrics, producing plausible yet logically useless code. % Therefore, harness generation across large and diverse projects continues to face challenges in reliable compilation, robust code retrieval, and comprehensive validation. To address these challenges, we present HarnessAgent, a tool-augmented agentic framework that achieves fully automated, scalable harness construction over hundreds of OSS-Fuzz targets. HarnessAgent introduces three key innovations: 1) a rule-based strategy to identify and minimize various compilation errors; 2) a hybrid tool pool for precise and robust symbol source code retrieval; and 3) an enhanced harness validation pipeline that detects fake definitions. We evaluate HarnessAgent on 243 target functions from OSS-Fuzz projects (65 C projects and 178 C++ projects). It improves the three-shot success rate by approximately 20\% compared to state-of-the-art techniques, reaching 87\% for C and 81\% for C++. Our one-hour fuzzing results show that more than 75\% of the harnesses generated by HarnessAgent increase the target function coverage, surpassing the baselines by over 10\%. In addition, the hybrid tool-pool system of HarnessAgent achieves a response rate of over 90\% for source code retrieval, outperforming Fuzz Introspector by more than 30\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17508v3">On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Project Page: https://github.com/complex-reasoning/RPG
    </div>
    <details class="paper-abstract">
      Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a clipped-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. We extend our experiments to 8K context length, and RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%). Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) clipped importance sampling, and (c) an iterative reference-policy update scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18428v2">AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Optimization modeling enables critical decisions across industries but remains difficult to automate: informal language must be mapped to precise mathematical formulations and executable solver code. Prior LLM approaches either rely on brittle prompting or costly retraining with limited generalization. We present AlphaOPT, a self-improving experience library that enables an LLM to learn from limited demonstrations (even answers alone, without gold-standard programs) and solver feedback - without annotated reasoning traces or parameter updates. AlphaOPT operates in a continual two-phase cycle: (i) a Library Learning phase that reflects on failed attempts, extracting solver-verified, structured insights as {taxonomy, condition, explanation, example}; and (ii) a Library Evolution phase that diagnoses retrieval misalignments and refines the applicability conditions of stored insights, improving transfer across tasks. This design (1) learns efficiently from limited demonstrations without curated rationales, (2) expands continually without costly retraining by updating the library rather than model weights, and (3) makes knowledge explicit and interpretable for human inspection and intervention. Experiments show that AlphaOPT steadily improves with more data (65% to 72% from 100 to 300 training items) and surpasses the strongest baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only on answers. Code and data are available at: https://github.com/Minw913/AlphaOPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08417v2">Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19877v2">It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10257v1">Reject or Not?: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before any downstream processing. To address the limited query-rejection capability of current voice assistants, this paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together with a personalized query-rejection method based on large language models. On the data side, we construct the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds, valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides a reproducible data foundation, evaluation standard and extensible technical framework for reliability research in smart-home voice interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12820v2">Emotional Support with LLM-based Empathetic Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10187v1">MiniF2F-Dafny: LLM-Guided Mathematical Theorem Proving via Auto-Active Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We present miniF2F-Dafny, the first translation of the mathematical reasoning benchmark miniF2F to an automated theorem prover: Dafny. Previously, the benchmark existed only in interactive theorem provers (Lean, Isabelle, HOL Light, Metamath). We find that Dafny's automation verifies 99/244 (40.6%) of the test set and 109/244 (44.7%) of the validation set with empty proofs--requiring no manual proof steps. For problems where empty proofs fail, we evaluate 12 off-the-shelf LLMs on providing proof hints. The best model we test achieves 55.7% pass@4 success rate employing iterative error correction. These preliminary results highlight an effective division of labor: LLMs provide high-level guidance while automation handles low-level details. Our benchmark can be found on GitHub at http://github.com/dafny-lang/miniF2F .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10172v1">Offscript: Automated Auditing of Instruction Adherence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and generative search systems are increasingly used for information seeking by diverse populations with varying preferences for knowledge sourcing and presentation. While users can customize LLM behavior through custom instructions and behavioral prompts, no mechanism exists to evaluate whether these instructions are being followed effectively. We present Offscript, an automated auditing tool that efficiently identifies potential instruction following failures in LLMs. In a pilot study analyzing custom instructions sourced from Reddit, Offscript detected potential deviations from instructed behavior in 86.4% of conversations, 22.2% of which were confirmed as material violations through human review. Our findings suggest that automated auditing serves as a viable approach for evaluating compliance to behavioral instructions related to information seeking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13161v2">Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v1">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation has become the de facto standard for scaling model assessment, but the practice is statistically unsound: uncalibrated scores can invert preferences, naive confidence intervals on uncalibrated scores achieve near-0% coverage, and importance-weighted estimators collapse under limited overlap despite high effective sample size (ESS). We introduce Causal Judge Evaluation (CJE), a framework that fixes all three failures. On n=4,961 Chatbot Arena prompts (after filtering from 5k), CJE achieves 99% pairwise ranking accuracy at full sample size (94% averaged across configurations), matching oracle quality, at 14x lower cost (for ranking 5 policies) by calibrating a 16x cheaper judge on just 5% oracle labels (~250 labels). CJE combines three components: (i) AutoCal-R, reward calibration via mean-preserving isotonic regression; (ii) SIMCal-W, weight stabilization via stacking of S-monotone candidates; and (iii) Oracle-Uncertainty Aware (OUA) inference that propagates calibration uncertainty into confidence intervals. We formalize the Coverage-Limited Efficiency (CLE) diagnostic, which explains why IPS-style estimators fail even when ESS exceeds 90%: the logger rarely visits regions where target policies concentrate. Key findings: SNIPS inverts rankings even with reward calibration (38% pairwise, negative Kendall's tau) due to weight instability; calibrated IPS remains near-random (47%) despite weight stabilization, consistent with CLE; OUA improves coverage from near-0% to ~86% (Direct) and ~96% (stacked-DR), where naive intervals severely under-cover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11143v1">Automated Penetration Testing with LLM Agents and Classical Planning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      While penetration testing plays a vital role in cybersecurity, achieving fully automated, hands-off-the-keyboard execution remains a significant research challenge. In this paper, we introduce the "Planner-Executor-Perceptor (PEP)" design paradigm and use it to systematically review existing work and identify the key challenges in this area. We also evaluate existing penetration testing systems, with a particular focus on the use of Large Language Model (LLM) agents for this task. The results show that the out-of-the-box Claude Code and Sonnet 4.5 exhibit superior penetration capabilities observed to date, substantially outperforming all prior systems. However, a detailed analysis of their testing processes reveals specific strengths and limitations; notably, LLM agents struggle with maintaining coherent long-horizon plans, performing complex reasoning, and effectively utilizing specialized tools. These limitations significantly constrain its overall capability, efficiency, and stability. To address these limitations, we propose CHECKMATE, a framework that integrates enhanced classical planning with LLM agents, providing an external, structured "brain" that mitigates the inherent weaknesses of LLM agents. Our evaluation shows that CHECKMATE outperforms the state-of-the-art system (Claude Code) in penetration capability, improving benchmark success rates by over 20%. In addition, it delivers substantially greater stability, cutting both time and monetary costs by more than 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04573v4">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05790v3">Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Low-rank optimization has emerged as a promising approach to enabling memory-efficient training of large language models (LLMs). Existing low-rank optimization methods typically project gradients onto a low-rank subspace, reducing the memory cost of storing optimizer states. A key challenge in these methods is selecting suitable subspaces to ensure an effective optimization trajectory. Most existing approaches select the dominant subspace to preserve gradient information, as this intuitively provides the best approximation. However, we find that in practice, the dominant subspace stops changing during pretraining, thereby constraining weight updates to similar subspaces. In this paper, we propose importance sampling for low-rank optimization in LLM pretraining with a provable convergence guarantee, which the dominant subspace approach does not have. Empirically, we demonstrate that our method significantly outperforms previous methods in LLM pretraining tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07462v2">Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16134v3">Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit position bias systematically underweighting information based on its location in the context but how this bias varies across languages and models remains unclear. We conduct a multilingual study across five typologically diverse languages (English, Russian, German, Hindi, Vietnamese) and five model architectures, analyzing how position bias interacts with prompting strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven but shows language-specific nuances. Notably, Qwen2.5-7B-Instruct, DeepSeek 7B Chat and Mistral 7B consistently favor late positions challenging the common assumption of universal early-token preference. (2) Explicitly instructing the model, in the presence of irrelevant distractors, that "the most relevant context to the query is marked as 1" unexpectedly reduces accuracy across all languages, questioning standard prompt-engineering practices. (3) Accuracy consistently drops most when relevant information appears in the middle of the context, yet this is not reflected in a corresponding increase in output entropy, suggesting the model remains confident even when it fails to use mid-context cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09321v2">ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 To appear in NDSS 2026. For slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf
    </div>
    <details class="paper-abstract">
      Prompt injection attacks aim to contaminate the input data of an LLM to mislead it into completing an attacker-chosen task instead of the intended task. In many applications and agents, the input data originates from multiple sources, with each source contributing a segment of the overall input. In these multi-source scenarios, an attacker may control only a subset of the sources and contaminate the corresponding segments, but typically does not know the order in which the segments are arranged within the input. Existing prompt injection attacks either assume that the entire input data comes from a single source under the attacker's control or ignore the uncertainty in the ordering of segments from different sources. As a result, their success is limited in domains involving multi-source data. In this work, we propose ObliInjection, the first prompt injection attack targeting LLM applications and agents with multi-source input data. ObliInjection introduces two key technical innovations: the order-oblivious loss, which quantifies the likelihood that the LLM will complete the attacker-chosen task regardless of how the clean and contaminated segments are ordered; and the orderGCG algorithm, which is tailored to minimize the order-oblivious loss and optimize the contaminated segments. Comprehensive experiments across three datasets spanning diverse application domains and twelve LLMs demonstrate that ObliInjection is highly effective, even when only one out of 6-100 segments in the input data is contaminated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15447v2">From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by ECAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11925v1">FloraForge: LLM-Assisted Procedural Generation of Editable and Analysis-Ready 3D Plant Geometric Models For Agricultural Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Accurate 3D plant models are crucial for computational phenotyping and physics-based simulation; however, current approaches face significant limitations. Learning-based reconstruction methods require extensive species-specific training data and lack editability. Procedural modeling offers parametric control but demands specialized expertise in geometric modeling and an in-depth understanding of complex procedural rules, making it inaccessible to domain scientists. We present FloraForge, an LLM-assisted framework that enables domain experts to generate biologically accurate, fully parametric 3D plant models through iterative natural language Plant Refinements (PR), minimizing programming expertise. Our framework leverages LLM-enabled co-design to refine Python scripts that generate parameterized plant geometries as hierarchical B-spline surface representations with botanical constraints with explicit control points and parametric deformation functions. This representation can be easily tessellated into polygonal meshes with arbitrary precision, ensuring compatibility with functional structural plant analysis workflows such as light simulation, computational fluid dynamics, and finite element analysis. We demonstrate the framework on maize, soybean, and mung bean, fitting procedural models to empirical point cloud data through manual refinement of the Plant Descriptor (PD), human-readable files. The pipeline generates dual outputs: triangular meshes for visualization and triangular meshes with additional parametric metadata for quantitative analysis. This approach uniquely combines LLM-assisted template creation, mathematically continuous representations enabling both phenotyping and rendering, and direct parametric control through PD. The framework democratizes sophisticated geometric modeling for plant science while maintaining mathematical rigor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11920v1">CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted to FPGA'26 Oral
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing tasks, but their deployment in datacenter environments faces significant challenges due to the massive memory requirements of key-value (KV) caches. During the autoregressive decoding process, KV caches consume substantial GPU memory, limiting batch sizes and overall system throughput. To address these challenges, we propose \textbf{CXL-SpecKV}, a novel disaggregated KV-cache architecture that leverages Compute Express Link (CXL) interconnects and FPGA accelerators to enable efficient speculative execution and memory disaggregation. Our approach introduces three key innovations: (i) a CXL-based memory disaggregation framework that offloads KV-caches to remote FPGA memory with low latency, (ii) a speculative KV-cache prefetching mechanism that predicts and preloads future tokens' cache entries, and (iii) an FPGA-accelerated KV-cache compression and decompression engine that reduces memory bandwidth requirements by up to 4$\times$. When evaluated on state-of-the-art LLM models, CXL-SpecKV achieves up to 3.2$\times$ higher throughput compared to GPU-only baselines, while reducing memory costs by 2.8$\times$ and maintaining accuracy. Our system demonstrates that intelligent memory disaggregation combined with speculative execution can effectively address the memory wall challenge in large-scale LLM serving. Our code implementation has been open-sourced at https://github.com/FastLM/CXL-SpecKV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10931v1">Asynchronous Reasoning: Training-Free Interactive Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities and safety, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embedded assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of rotary embeddings to enable LLMs built for sequential interactions to simultaneously think, listen, and generate outputs. We evaluate our approach on math, commonsense, and safety reasoning and find that it can generate accurate thinking-augmented answers in real time, reducing time to first non-thinking token from minutes to <= 5s. and the overall real-time delays by 6-11x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06982v2">LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning
    </div>
    <details class="paper-abstract">
      Designing state encoders for reinforcement learning (RL) with multiple information sources -- such as sensor measurements, time-series signals, image observations, and textual instructions -- remains underexplored and often requires manual design. We formalize this challenge as a problem of composite neural architecture search (NAS), where multiple source-specific modules and a fusion module are jointly optimized. Existing NAS methods overlook useful side information from the intermediate outputs of these modules -- such as their representation quality -- limiting sample efficiency in multi-source RL settings. To address this, we propose an LLM-driven NAS pipeline in which the LLM serves as a neural architecture design agent, leveraging language-model priors and intermediate-output signals to guide sample-efficient search for high-performing composite state encoders. On a mixed-autonomy traffic control task, our approach discovers higher-performing architectures with fewer candidate evaluations than traditional NAS baselines and the LLM-based GENIUS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10922v1">SparseSwaps: Tractable LLM Pruning Mask Refinement at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 15 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The resource requirements of Neural Networks can be significantly reduced through pruning -- the removal of seemingly less important parameters. However, with the rise of Large Language Models (LLMs), full retraining to recover pruning-induced performance degradation is often prohibitive and classical approaches such as global magnitude pruning are suboptimal on Transformer architectures. State-of-the-art methods hence solve a layer-wise mask selection problem, the problem of finding a pruning mask which minimizes the per-layer pruning error on a small set of calibration data. Exactly solving this problem to optimality using Integer Programming (IP) solvers is computationally infeasible due to its combinatorial nature and the size of the search space, and existing approaches therefore rely on approximations or heuristics. In this work, we demonstrate that the mask selection problem can be made drastically more tractable at LLM scale. To that end, we decouple the rows by enforcing equal sparsity levels per row. This allows us to derive optimal 1-swaps (exchanging one kept and one pruned weight) that can be computed efficiently using the Gram matrix of the calibration data. Using these observations, we propose a tractable and simple 1-swap algorithm that warm starts from any pruning mask, runs efficiently on GPUs at LLM scale, and is essentially hyperparameter-free. We demonstrate that our approach reduces per-layer pruning error by up to 60% over Wanda (Sun et al., 2023) and consistently improves perplexity and zero-shot accuracy across state-of-the-art GPT architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10895v1">LLMs Can Assist with Proposal Selection at Large User Facilities</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 9 pages, 8figures
    </div>
    <details class="paper-abstract">
      We explore how large language models (LLMs) can enhance the proposal selection process at large user facilities, offering a scalable, consistent, and cost-effective alternative to traditional human review. Proposal selection depends on assessing the relative strength among submitted proposals; however, traditional human scoring often suffers from weak inter-proposal correlations and is subject to reviewer bias and inconsistency. A pairwise preference-based approach is logically superior, providing a more rigorous and internally consistent basis for ranking, but its quadratic workload makes it impractical for human reviewers. We address this limitation using LLMs. Leveraging the uniquely well-curated proposals and publication records from three beamlines at the Spallation Neutron Source (SNS), Oak Ridge National Laboratory (ORNL), we show that the LLM rankings correlate strongly with the human rankings (Spearman $ρ\simeq 0.2-0.8$, improving to $\geq 0.5$ after 10\% outlier removal). Moreover, LLM performance is no worse than that of human reviewers in identifying proposals with high publication potential, while costing over two orders of magnitude less. Beyond ranking, LLMs enable advanced analyses that are challenging for humans, such as quantitative assessment of proposal similarity via embedding models, which provides information crucial for review committees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10793v1">LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs such as OpenAI GPT, Google Gemini, or DeepSeek) to deliver accurate and cost-aware predictions across multi-class and multi-label tasks. The package provides a simple high-level interface (AutoFusionClassifier) that trains the full pipeline end-to-end with minimal configuration, and a flexible API for advanced users. Under the hood, LabelFusion integrates vector signals from both sources by concatenating the ML backbone's embeddings with the LLM-derived per-class scores -- obtained through structured prompt-engineering strategies -- and feeds this joint representation into a compact multi-layer perceptron (FusionMLP) that produces the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and traditional transformer-based classifiers, yielding robust performance across domains -- achieving 92.4% accuracy on AG News and 92.3% on 10-class Reuters 21578 topic classification -- while enabling practical trade-offs between accuracy, latency, and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10780v1">Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01951v2">The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10687v1">Challenges of Evaluating LLM Safety for User Welfare</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Paper accepted at IASEAI'26; please cite that peer-reviewed version instead
    </div>
    <details class="paper-abstract">
      Safety evaluations of large language models (LLMs) typically focus on universal risks like dangerous capabilities or undesirable propensities. However, millions use LLMs for personal advice on high-stakes topics like finance and health, where harms are context-dependent rather than universal. While frameworks like the OECD's AI classification recognize the need to assess individual risks, user-welfare safety evaluations remain underdeveloped. We argue that developing such evaluations is non-trivial due to fundamental questions about accounting for user context in evaluation design. In this exploratory study, we evaluated advice on finance and health from GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across user profiles of varying vulnerability. First, we demonstrate that evaluators must have access to rich user context: identical LLM responses were rated significantly safer by context-blind evaluators than by those aware of user circumstances, with safety scores for high-vulnerability users dropping from safe (5/7) to somewhat unsafe (3/7). One might assume this gap could be addressed by creating realistic user prompts containing key contextual information. However, our second study challenges this: we rerun the evaluation on prompts containing context users report they would disclose, finding no significant improvement. Our work establishes that effective user-welfare safety evaluation requires evaluators to assess responses against diverse user profiles, as realistic user context disclosure alone proves insufficient, particularly for vulnerable populations. By demonstrating a methodology for context-aware evaluation, this study provides both a starting point for such assessments and foundational evidence that evaluating individual welfare demands approaches distinct from existing universal-risk frameworks. We publish our code and dataset to aid future developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10665v1">On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Working Paper
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLM) based multi-agent systems become increasingly prevalent, the collective behaviors, e.g., collective intelligence, of such artificial communities have drawn growing attention. This work aims to answer a fundamental question: How does diversity of values shape the collective behavior of AI communities? Using naturalistic value elicitation grounded in the prevalent Schwartz's Theory of Basic Human Values, we constructed multi-agent simulations where communities with varying numbers of agents engaged in open-ended interactions and constitution formation. The results show that value diversity enhances value stability, fosters emergent behaviors, and brings more creative principles developed by the agents themselves without external guidance. However, these effects also show diminishing returns: extreme heterogeneity induces instability. This work positions value diversity as a new axis of future AI capability, bridging AI ability and sociological studies of institutional emergence.
    </details>
</div>
