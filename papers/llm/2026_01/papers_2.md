# llm - 2026_01

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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19255v1">LLM-Assisted Logic Rule Learning: Scaling Human Expertise for Time Series Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical for supply chain management to take proactive operations, but faces challenges: classical unsupervised anomaly detection based on exploiting data patterns often yields results misaligned with business requirements and domain knowledge, while manual expert analysis cannot scale to millions of products in the supply chain. We propose a framework that leverages large language models (LLMs) to systematically encode human expertise into interpretable, logic-based rules for detecting anomaly patterns in supply chain time series data. Our approach operates in three stages: 1) LLM-based labeling of training data instructed by domain knowledge, 2) automated generation and iterative improvements of symbolic rules through LLM-driven optimization, and 3) rule augmentation with business-relevant anomaly categories supported by LLMs to enhance interpretability. The experiment results showcase that our approach outperforms the unsupervised learning methods in both detection accuracy and interpretability. Furthermore, compared to direct LLM deployment for time series anomaly detection, our approach provides consistent, deterministic results with low computational latency and cost, making it ideal for production deployment. The proposed framework thus demonstrates how LLMs can bridge the gap between scalable automation and expert-driven decision-making in operational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19249v1">GLOVE: Global Verifier for LLM Memory-Environment Realignment</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Most existing memory-enhanced Large Language Model (LLM) approaches implicitly assume that memory validity can be established either through external evaluators that provide task-specific success signals or through internal model cognition, such as reflection, for editing memory entries. However, these assumptions often break down in practical environments with dynamic drifts. We propose the Global Verifier (GLOVE), a framework that introduces a new design dimension for LLM memory systems by establishing a relative notion of truth. Through active probing to detect inconsistencies between retrieved memories and fresh observations, GLOVE enables memory-environment realignment by verifying and updating memory without access to ground-truth supervision or strong reliance on model introspection. We evaluate GLOVE on diverse benchmarks spanning web navigation, planning, and control, augmented with controlled environmental drifts that introduce non-stationarity beyond the original benchmark settings. Our results show that GLOVE substantially improves agent success rates, suggesting a robust pathway to cognitive agents capable of self-evolving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19239v1">LLM-based Vulnerability Detection at Project Scale: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 19 pages, 13 figures, 11 tables
    </div>
    <details class="paper-abstract">
      In this paper, we present the first comprehensive empirical study of specialized LLM-based detectors and compare them with traditional static analyzers at the project scale. Specifically, our study evaluates five latest and representative LLM-based methods and two traditional tools using: 1) an in-house benchmark of 222 known real-world vulnerabilities (C/C++ and Java) to assess detection capability, and 2) 24 active open-source projects, where we manually inspected 385 warnings to assess their practical usability and underlying root causes of failures. Our evaluation yields three key findings: First, while LLM-based detectors exhibit low recall on the in-house benchmark, they still uncover more unique vulnerabilities than traditional tools. Second, in open-source projects, both LLM-based and traditional tools generate substantial warnings but suffer from very high false discovery rates, hindering practical use. Our manual analysis further reveals shallow interprocedural reasoning and misidentified source/sink pairs as primary failure causes, with LLM-based tools exhibiting additional unique failures. Finally, LLM-based methods incurs substantial computational costs-hundreds of thousands to hundreds of millions of tokens and multi-hour to multi-day runtimes. Overall, our findings underscore critical limitations in the robustness, reliability, and scalability of current LLM-based detectors. We ultimately summarize a set of implications for future research toward more effective and practical project-scale vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19231v1">LLMs Can Unlearn Refusal with Only 1,000 Benign Samples</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      This study reveals a previously unexplored vulnerability in the safety alignment of Large Language Models (LLMs). Existing aligned LLMs predominantly respond to unsafe queries with refusals, which often begin with a fixed set of prefixes (I'm sorry). We demonstrate that this rigid refusal pattern is a vulnerability and introduce a novel \textbf{refusal unlearning} technique that exploits it. Specifically, we fine-tune LLMs using merely 1,000 benign samples, where each response is prepended with a refusal prefix. The underlying intuition is to disrupt the refusal completion pathway, thereby driving the model to forget how to refuse while following harmful instructions. This intuition is further supported by theoretical proofs. We apply this approach to a total of 16 LLMs, including various open-source models from Llama, Qwen, and Gemma families, as well as closed-source models such as Gemini and GPT. Experimental results show that the safety scores of previously aligned LLMs degrade both consistently and substantially. Importantly, we verify that the observed gain cannot be attributed to plain fine-tuning or random prefix effects. Our findings suggest that current safety alignment may rely heavily on token sequence memorization rather than reasoning, motivating future work beyond simple refusal mechanisms. Code has been released: https://github.com/guoyang9/refusal-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19225v1">RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at The Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19214v1">A Hybrid Supervised-LLM Pipeline for Actionable Suggestion Mining in Unstructured Customer Reviews</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted to EACL 2026 Industry Track (to appear)
    </div>
    <details class="paper-abstract">
      Extracting actionable suggestions from customer reviews is essential for operational decision-making, yet these directives are often embedded within mixed-intent, unstructured text. Existing approaches either classify suggestion-bearing sentences or generate high-level summaries, but rarely isolate the precise improvement instructions businesses need. We evaluate a hybrid pipeline combining a high-recall RoBERTa classifier trained with a precision-recall surrogate to reduce unrecoverable false negatives with a controlled, instruction-tuned LLM for suggestion extraction, categorization, clustering, and summarization. Across real-world hospitality and food datasets, the hybrid system outperforms prompt-only, rule-based, and classifier-only baselines in extraction accuracy and cluster coherence. Human evaluations further confirm that the resulting suggestions and summaries are clear, faithful, and interpretable. Overall, our results show that hybrid reasoning architectures achieve meaningful improvements fine-grained actionable suggestion mining while highlighting challenges in domain adaptation and efficient local deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19197v1">HELM: A Human-Centered Evaluation Framework for LLM-Powered Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into recommendation systems has introduced unprecedented capabilities for natural language understanding, explanation generation, and conversational interactions. However, existing evaluation methodologies focus predominantly on traditional accuracy metrics, failing to capture the multifaceted human-centered qualities that determine the real-world user experience. We introduce \framework{} (\textbf{H}uman-centered \textbf{E}valuation for \textbf{L}LM-powered reco\textbf{M}menders), a comprehensive evaluation framework that systematically assesses LLM-powered recommender systems across five human-centered dimensions: \textit{Intent Alignment}, \textit{Explanation Quality}, \textit{Interaction Naturalness}, \textit{Trust \& Transparency}, and \textit{Fairness \& Diversity}. Through extensive experiments involving three state-of-the-art LLM-based recommenders (GPT-4, LLaMA-3.1, and P5) across three domains (movies, books, and restaurants), and rigorous evaluation by 12 domain experts using 847 recommendation scenarios, we demonstrate that \framework{} reveals critical quality dimensions invisible to traditional metrics. Our results show that while GPT-4 achieves superior explanation quality (4.21/5.0) and interaction naturalness (4.35/5.0), it exhibits a significant popularity bias (Gini coefficient 0.73) compared to traditional collaborative filtering (0.58). We release \framework{} as an open-source toolkit to advance human-centered evaluation practices in the recommender systems community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11527v2">Do LLMs Give Good Romantic Relationship Advice? A Study on User Satisfaction and Attitude Change</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 An error in figure 1
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to provide support and advice in personal domains such as romantic relationships, yet little is known about user perceptions of this type of advice. This study investigated how people evaluate advice on LLM-generated romantic relationships. Participants rated advice satisfaction, model reliability, and helpfulness, and completed pre- and post-measures of their general attitudes toward LLMs. Overall, the results showed participants' high satisfaction with LLM-generated advice. Greater satisfaction was, in turn, strongly and positively associated with their perceptions of the models' reliability and helpfulness. Importantly, participants' attitudes toward LLMs improved significantly after exposure to the advice, suggesting that supportive and contextually relevant advice can enhance users' trust and openness toward these AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19174v1">SHIELD: An Auto-Healing Agentic Defense Framework for LLM Resource Exhaustion Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Sponge attacks increasingly threaten LLM systems by inducing excessive computation and DoS. Existing defenses either rely on statistical filters that fail on semantically meaningful attacks or use static LLM-based detectors that struggle to adapt as attack strategies evolve. We introduce SHIELD, a multi-agent, auto-healing defense framework centered on a three-stage Defense Agent that integrates semantic similarity retrieval, pattern matching, and LLM-based reasoning. Two auxiliary agents, a Knowledge Updating Agent and a Prompt Optimization Agent, form a closed self-healing loop, when an attack bypasses detection, the system updates an evolving knowledgebase, and refines defense instructions. Extensive experiments show that SHIELD consistently outperforms perplexity-based and standalone LLM defenses, achieving high F1 scores across both non-semantic and semantic sponge attacks, demonstrating the effectiveness of agentic self-healing against evolving resource-exhaustion threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22050v3">DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted by EACL Findings 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at https://github.com/MinghoKwok/DeepSieve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21044v2">Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families and mathematical datasets shows two robust effects of online RL post-training: (i) an overall increase in average activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in mathematical generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at https://github.com/tsinghua-fib-lab/llm_rl_probing_analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14140v3">RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at https://github.com/tsinghua-fib-lab/RL-LLM-Reasoning for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18771v2">Exploring Graph Learning Tasks with Pure LLMs: A Comprehensive Benchmark and Investigation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have emerged as promising candidates for graph tasks. Many studies leverage natural language to describe graphs and apply LLMs for reasoning, yet most focus narrowly on performance benchmarks without fully comparing LLMs to graph learning models or exploring their broader potential. In this work, we present a comprehensive study of LLMs on graph learning tasks, evaluating both off-the-shelf and instruction-tuned models across a variety of scenarios. Beyond accuracy, we discuss data leakage concerns and computational overhead, and assess their performance under few-shot/zero-shot settings, domain transfer, structural understanding, and robustness. Our findings show that LLMs, particularly those with instruction tuning, greatly outperform traditional graph learning models in few-shot settings, exhibit strong domain transferability, and demonstrate excellent generalization and robustness. Our study highlights the broader capabilities of LLMs in graph learning and provides a foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19139v1">Native LLM and MLLM Inference at Scale on Apple Silicon</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      The growing adoption of Apple Silicon for machine learning development has created demand for efficient inference solutions that leverage its unique unified memory architecture. However, existing tools either lack native optimization (PyTorch MPS) or focus solely on text models (llama.cpp), leaving multimodal workloads underserved. We present vllm-mlx, a framework for efficient LLM and MLLM inference on Apple Silicon built natively on MLX. For text models, we achieve 21% to 87% higher throughput than llama.cpp across models ranging from Qwen3-0.6B to Nemotron-30B, while providing continuous batching that scales to 4.3x aggregate throughput at 16 concurrent requests. For multimodal models, we introduce content-based prefix caching that eliminates redundant vision encoding by identifying identical images through content hashing, regardless of input format. Our evaluation on Apple M4 Max demonstrates throughput of up to 525 tokens per second on text models and 28x speedup on repeated image queries, reducing multimodal latency from 21.7 seconds to under 1 second. Video analysis with up to 64 frames achieves 24.7x cache speedup. We release our implementation as open source to support efficient inference on consumer Apple hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19121v1">LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19120v1">RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate natural-language explanations in recommender systems, acting as explanation agents that reason over user behavior histories. While prior work has focused on explanation fluency and relevance under fixed inputs, the robustness of LLM-generated explanations to realistic user behavior noise remains largely unexplored. In real-world web platforms, interaction histories are inherently noisy due to accidental clicks, temporal inconsistencies, missing values, and evolving preferences, raising concerns about explanation stability and user trust. We present RobustExplain, the first systematic evaluation framework for measuring the robustness of LLM-generated recommendation explanations. RobustExplain introduces five realistic user behavior perturbations evaluated across multiple severity levels and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency. Our goal is to establish a principled, task-level evaluation framework and initial robustness baselines, rather than to provide a comprehensive leaderboard across all available LLMs. Experiments on four representative LLMs (7B--70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability. Our results establish the first robustness benchmarks for explanation agents and highlight robustness as a critical dimension for trustworthy, agent-driven recommender systems at web scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19106v1">Detecting and Correcting Hallucinations in LLM-Generated Code via Deterministic AST Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted to FORGE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for code generation boost productivity but frequently introduce Knowledge Conflicting Hallucinations (KCHs), subtle, semantic errors, such as non-existent API parameters, that evade linters and cause runtime failures. Existing mitigations like constrained decoding or non-deterministic LLM-in-the-loop repair are often unreliable for these errors. This paper investigates whether a deterministic, static-analysis framework can reliably detect \textit{and} auto-correct KCHs. We propose a post-processing framework that parses generated code into an Abstract Syntax Tree (AST) and validates it against a dynamically-generated Knowledge Base (KB) built via library introspection. This non-executing approach uses deterministic rules to find and fix both API and identifier-level conflicts. On a manually-curated dataset of 200 Python snippets, our framework detected KCHs with 100\% precision and 87.6\% recall (0.934 F1-score), and successfully auto-corrected 77.0\% of all identified hallucinations. Our findings demonstrate that this deterministic post-processing approach is a viable and reliable alternative to probabilistic repair, offering a clear path toward trustworthy code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12619v2">BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 18 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become a cornerstone of modern AI, driving breakthroughs in natural language processing and expanding into multimodal jobs involving images, audio, and video. As with most computational software, it is important to distinguish between ordinary runtime performance and startup overhead. Prior research has focused on runtime performance: improving training efficiency and stability. This work focuses instead on the increasingly critical issue of startup overhead in training: the delay before training jobs begin execution. Startup overhead is particularly important in large, industrial-scale LLMs, where failures occur more frequently and multiple teams operate in iterative update-debug cycles. In one of our training clusters, more than 3.5% of GPU time is wasted due to startup overhead alone. In this work, we present the first in-depth characterization of LLM training startup overhead based on real production data. We analyze the components of startup cost, quantify its direct impact, and examine how it scales with job size. These insights motivate the design of Bootseer, a system-level optimization framework that addresses three primary startup bottlenecks: (a) container image loading, (b) runtime dependency installation, and (c) model checkpoint resumption. To mitigate these bottlenecks, Bootseer introduces three techniques: (a) hot block record-and-prefetch, (b) dependency snapshotting, and (c) striped HDFS-FUSE. Bootseer has been deployed in a production environment and evaluated on real LLM training workloads, demonstrating a 50% reduction in startup overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19082v1">More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 14 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      As LLMs increasingly act as autonomous agents in interactive and multi-agent settings, understanding their strategic behavior is critical for safety, coordination, and AI-driven social and economic systems. We investigate how payoff magnitude and linguistic context shape LLM strategies in repeated social dilemmas, using a payoff-scaled Prisoner's Dilemma to isolate sensitivity to incentive strength. Across models and languages, we observe consistent behavioral patterns, including incentive-sensitive conditional strategies and cross-linguistic divergence. To interpret these dynamics, we train supervised classifiers on canonical repeated-game strategies and apply them to LLM decisions, revealing systematic, model- and language-dependent behavioral intentions, with linguistic framing sometimes matching or exceeding architectural effects. Our results provide a unified framework for auditing LLMs as strategic agents and highlight cooperation biases with direct implications for AI governance and multi-agent system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15301v2">Can We Trust LLM Detectors?</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      The rapid adoption of LLMs has increased the need for reliable AI text detection, yet existing detectors often fail outside controlled benchmarks. We systematically evaluate 2 dominant paradigms (training-free and supervised) and show that both are brittle under distribution shift, unseen generators, and simple stylistic perturbations. To address these limitations, we propose a supervised contrastive learning (SCL) framework that learns discriminative style embeddings. Experiments show that while supervised detectors excel in-domain, they degrade sharply out-of-domain, and training-free methods remain highly sensitive to proxy choice. Overall, our results expose fundamental challenges in building domain-agnostic detectors. Our code is available at: https://github.com/HARSHITJAIS14/DetectAI
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07667v2">1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at the International Association for AI Safety and Ethics AI (IASEAI) 2026
    </div>
    <details class="paper-abstract">
      Addressing contextual privacy concerns remains challenging in interactive settings where large language models (LLMs) process information from multiple sources (e.g., summarizing meetings with private and public information). We introduce a multi-agent framework that decomposes privacy reasoning into specialized subtasks (extraction, classification), reducing the information load on any single agent while enabling iterative validation and more reliable adherence to contextual privacy norms. To understand how privacy errors emerge and propagate, we conduct a systematic ablation over information-flow topologies, revealing when and why upstream detection mistakes cascade into downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with several open-source and closed-sourced LLMs demonstrate that our best multi-agent configuration substantially reduces private information leakage (\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while preserving the fidelity of public content, outperforming single-agent baselines. These results highlight the promise of principled information-flow design in multi-agent systems for contextual privacy with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19055v1">Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We study how to fine-tune LLMs using user-edit deployment data consisting of a set of context, an agent's response, and user edits. This deployment data is naturally generated by users in applications such as LLMs-based writing assistants and coding agents. The _natural_ origin of user edits makes it a desired source for adapting and personalizing LLMs. In this setup, there emerges a unification of various feedback types namely preferences, supervised labels, and cost that are typically studied separately in the literature. In this paper, we initiate the theoretical investigation of learning from user edits. We first derive bounds for learning algorithms that learn from each of these feedback types. We prove that these algorithms have different trade-offs depending upon the user, data distribution, and model class. We then propose a simple ensembling procedure to jointly learn from these feedback types. On two domains adapted from Gao et al. 2024, we show our ensembling procedure outperforms these methods that learn from individual feedback. Further, we show that our proposed procedure can robustly adapt to different user-edit distributions at test time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19053v1">From Answer Givers to Design Mentors: Guiding LLMs with the Cognitive Apprenticeship Model</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Design feedback helps practitioners improve their artifacts while also fostering reflection and design reasoning. Large Language Models (LLMs) such as ChatGPT can support design work, but often provide generic, one-off suggestions that limit reflective engagement. We investigate how to guide LLMs to act as design mentors by applying the Cognitive Apprenticeship Model, which emphasizes demonstrating reasoning through six methods: modeling, coaching, scaffolding, articulation, reflection, and exploration. We operationalize these instructional methods through structured prompting and evaluate them in a within-subjects study with data visualization practitioners. Participants interacted with both a baseline LLM and an instructional LLM designed with cognitive apprenticeship prompts. Surveys, interviews, and conversational log analyses compared experiences across conditions. Our findings show that cognitively informed prompts elicit deeper design reasoning and more reflective feedback exchanges, though the baseline is sometimes preferred depending on task types or experience levels. We distill design considerations for AI-assisted feedback systems that foster reflective practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19051v1">Proactive Hardening of LLM Defenses with HASTE</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at peer review NDSS 2026, Last-X workshop. Camera ready copy forthcoming
    </div>
    <details class="paper-abstract">
      Prompt-based attack techniques are one of the primary challenges in securely deploying and protecting LLM-based AI systems. LLM inputs are an unbounded, unstructured space. Consequently, effectively defending against these attacks requires proactive hardening strategies capable of continuously generating adaptive attack vectors to optimize LLM defense at runtime. We present HASTE (Hard-negative Attack Sample Training Engine): a systematic framework that iteratively engineers highly evasive prompts, within a modular optimization process, to continuously enhance detection efficacy for prompt-based attack techniques. The framework is agnostic to synthetic data generation methods, and can be generalized to evaluate prompt-injection detection efficacy, with and without fuzzing, for any hard-negative or hard-positive iteration strategy. Experimental evaluation of HASTE shows that hard negative mining successfully evades baseline detectors, reducing malicious prompt detection for baseline detectors by approximately 64%. However, when integrated with detection model re-training, it optimizes the efficacy of prompt detection models with significantly fewer iteration loops compared to relative baseline strategies. The HASTE framework supports both proactive and reactive hardening of LLM defenses and guardrails. Proactively, developers can leverage HASTE to dynamically stress-test prompt injection detection systems; efficiently identifying weaknesses and strengthening defensive posture. Reactively, HASTE can mimic newly observed attack types and rapidly bridge detection coverage by teaching HASTE-optimized detection models to identify them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20102v1">Counterfactual Cultural Cues Reduce Medical QA Accuracy in LLMs: Identifier vs Context Effects</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Engineering sustainable and equitable healthcare requires medical language models that do not change clinically correct diagnoses when presented with non-decisive cultural information. We introduce a counterfactual benchmark that expands 150 MedQA test items into 1650 variants by inserting culture-related (i) identifier tokens, (ii) contextual cues, or (iii) their combination for three groups (Indigenous Canadian, Middle-Eastern Muslim, Southeast Asian), plus a length-matched neutral control, where a clinician verified that the gold answer remains invariant in all variants. We evaluate GPT-5.2, Llama-3.1-8B, DeepSeek-R1, and MedGemma (4B/27B) under option-only and brief-explanation prompting. Across models, cultural cues significantly affect accuracy (Cochran's Q, $p<10^-14$), with the largest degradation when identifier and context co-occur (up to 3-7 percentage points under option-only prompting), while neutral edits produce smaller, non-systematic changes. A human-validated rubric ($κ=0.76$) applied via an LLM-as-judge shows that more than half of culturally grounded explanations end in an incorrect answer, linking culture-referential reasoning to diagnostic failure. We release prompts and augmentations to support evaluation and mitigation of culturally induced diagnostic errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20090v1">Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents can translate high-level user intents into plans and actions in an environment. Yet after observing an outcome, users may wonder: What if I had phrased my intent differently? We introduce a framework that enables such counterfactual reasoning in agentic LLM-driven control scenarios, while providing formal reliability guarantees. Our approach models the closed-loop interaction between a user, an LLM-based agent, and an environment as a structural causal model (SCM), and leverages test-time scaling to generate multiple candidate counterfactual outcomes via probabilistic abduction. Through an offline calibration phase, the proposed conformal counterfactual generation (CCG) yields sets of counterfactual outcomes that are guaranteed to contain the true counterfactual outcome with high probability. We showcase the performance of CCG on a wireless network control use case, demonstrating significant advantages compared to naive re-execution baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20055v1">VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Despite the syntactic fluency of Large Language Models (LLMs), ensuring their logical correctness in high-stakes domains remains a fundamental challenge. We present a neurosymbolic framework that combines LLMs with SMT solvers to produce verification-guided answers through iterative refinement. Our approach decomposes LLM outputs into atomic claims, autoformalizes them into first-order logic, and verifies their logical consistency using automated theorem proving. We introduce three key innovations: (1) multi-model consensus via formal semantic equivalence checking to ensure logic-level alignment between candidates, eliminating the syntactic bias of surface-form metrics, (2) semantic routing that directs different claim types to appropriate verification strategies: symbolic solvers for logical claims and LLM ensembles for commonsense reasoning, and (3) precise logical error localization via Minimal Correction Subsets (MCS), which pinpoint the exact subset of claims to revise, transforming binary failure signals into actionable feedback. Our framework classifies claims by their logical status and aggregates multiple verification signals into a unified score with variance-based penalty. The system iteratively refines answers using structured feedback until acceptance criteria are met or convergence is achieved. This hybrid approach delivers formal guarantees where possible and consensus verification elsewhere, advancing trustworthy AI. With the GPT-OSS-120B model, VERGE demonstrates an average performance uplift of 18.7% at convergence across a set of reasoning benchmarks compared to single-pass approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20048v1">Insight Agents: An LLM-Based Multi-Agent System for Data Insights</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted to SIGIR 2025. DOI: 10.1145/3726302.3731959
    </div>
    <details class="paper-abstract">
      Today, E-commerce sellers face several key challenges, including difficulties in discovering and effectively utilizing available programs and tools, and struggling to understand and utilize rich data from various tools. We therefore aim to develop Insight Agents (IA), a conversational multi-agent Data Insight system, to provide E-commerce sellers with personalized data and business insights through automated information retrieval. Our hypothesis is that IA will serve as a force multiplier for sellers, thereby driving incremental seller adoption by reducing the effort required and increase speed at which sellers make good business decisions. In this paper, we introduce this novel LLM-backed end-to-end agentic system built on a plan-and-execute paradigm and designed for comprehensive coverage, high accuracy, and low latency. It features a hierarchical multi-agent structure, consisting of manager agent and two worker agents: data presentation and insight generation, for efficient information retrieval and problem-solving. We design a simple yet effective ML solution for manager agent that combines Out-of-Domain (OOD) detection using a lightweight encoder-decoder model and agent routing through a BERT-based classifier, optimizing both accuracy and latency. Within the two worker agents, a strategic planning is designed for API-based data model that breaks down queries into granular components to generate more accurate responses, and domain knowledge is dynamically injected to to enhance the insight generator. IA has been launched for Amazon sellers in US, which has achieved high accuracy of 90% based on human evaluation, with latency of P90 below 15s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20041v1">CiMRAG: Cim-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Personalized virtual assistants powered by large language models (LLMs) on edge devices are attracting growing attention, with Retrieval-Augmented Generation (RAG) emerging as a key method for personalization by retrieving relevant profile data and generating tailored responses. However, deploying RAG on edge devices faces efficiency hurdles due to the rapid growth of profile data, such as user-LLM interactions and recent updates. While Computing-in-Memory (CiM) architectures mitigate this bottleneck by eliminating data movement between memory and processing units via in-situ operations, they are susceptible to environmental noise that can degrade retrieval precision. This poses a critical issue in dynamic, multi-domain edge-based scenarios (e.g., travel, medicine, and law) where both accuracy and adaptability are paramount. To address these challenges, we propose Task-Oriented Noise-resilient Embedding Learning (TONEL), a framework that improves noise robustness and domain adaptability for RAG in noisy edge environments. TONEL employs a noise-aware projection model to learn task-specific embeddings compatible with CiM hardware constraints, enabling accurate retrieval under noisy conditions. Extensive experiments conducted on personalization benchmarks demonstrate the effectiveness and practicality of our methods relative to strong baselines, especially in task-specific noisy scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.11987v2">SimBench: A Framework for Evaluating and Diagnosing LLM-Based Digital-Twin Generation for Multi-Physics Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      We introduce SimBench, a benchmark designed to evaluate the proficiency of simulator-oriented LLMs (S-LLMs) in generating digital twins (DTs) that can be used in simulators for virtual testing. Given a collection of S-LLMs, this benchmark ranks them according to their ability to produce high-quality DTs. We demonstrate this by comparing over 33 open- and closed-source S-LLMs. Using multi-turn interactions, SimBench employs an LLM-as-a-judge (J-LLM) that leverages both predefined rules and human-in-the-loop guidance to assign scores for the DTs generated by the S-LLM, thus providing a consistent and expert-inspired evaluation protocol. The J-LLM is specific to a simulator, and herein the proposed benchmarking approach is demonstrated in conjunction with the open-sourceChrono multi-physics simulator. Chrono provided the backdrop used to assess an S-LLM in relation to the latter's ability to create digital twins for multibody dynamics, finite element analysis, vehicle dynamics, robotic dynamics, and sensor simulations. The proposed benchmarking principle is broadly applicable and enables the assessment of an S-LLM's ability to generate digital twins for other simulation packages, e.g., ANSYS, ABAQUS, OpenFOAM, StarCCM+, IsaacSim, and pyBullet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20026v1">Semantic Uncertainty Quantification of Hallucinations in LLMs: A Quantum Tensor Network Based Method</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong generative capabilities but remain vulnerable to confabulations, fluent yet unreliable outputs that vary arbitrarily even under identical prompts. Leveraging a quantum tensor network based pipeline, we propose a quantum physics inspired uncertainty quantification framework that accounts for aleatoric uncertainty in token sequence probability for semantic equivalence based clustering of LLM generations. This offers a principled and interpretable scheme for hallucination detection. We further introduce an entropy maximization strategy that prioritizes high certainty, semantically coherent outputs and highlights entropy regions where LLM decisions are likely to be unreliable, offering practical guidelines for when human oversight is warranted. We evaluate the robustness of our scheme under different generation lengths and quantization levels, dimensions overlooked in prior studies, demonstrating that our approach remains reliable even in resource constrained deployments. A total of 116 experiments on TriviaQA, NQ, SVAMP, and SQuAD across multiple architectures including Mistral-7B, Mistral-7B-instruct, Falcon-rw-1b, LLaMA-3.2-1b, LLaMA-2-13b-chat, LLaMA-2-7b-chat, LLaMA-2-13b, and LLaMA-2-7b show consistent improvements in AUROC and AURAC over state of the art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20006v1">On the Effectiveness of LLM-Specific Fine-Tuning for Detecting AI-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 34 pages, 6 figures. Under review at Information Sciences
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models has enabled the generation of text that closely resembles human writing, creating challenges for authenticity verification in education, publishing, and digital security. Detecting AI-generated text has therefore become a crucial technical and ethical issue. This paper presents a comprehensive study of AI-generated text detection based on large-scale corpora and novel training strategies. We introduce a 1-billion-token corpus of human-authored texts spanning multiple genres and a 1.9-billion-token corpus of AI-generated texts produced by prompting a variety of LLMs across diverse domains. Using these resources, we develop and evaluate numerous detection models and propose two novel training paradigms: Per LLM and Per LLM family fine-tuning. Across a 100-million-token benchmark covering 21 large language models, our best fine-tuned detector achieves up to $99.6\%$ token-level accuracy, substantially outperforming existing open-source baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11589v2">LAPS: A Length-Aware-Prefill LLM Serving System</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      LAPS identifies and disaggregates requests with different prompt lengths in LLM serving to reduce TTFT latency. While recent systems have decoupled the prefill and decode stages to improve throughput, they still rely on unified scheduling policies that fail to adapt to heterogeneous workload characteristics. We observe that prompt-length variations lead to distinct performance bottlenecks, motivating an adaptive scheduling strategy. LAPS disaggregates multi-turn long-prefill requests from short-prefill ones and introduces a length-aware smart batching mechanism for short-prefill workloads. It adopts a dual-queue design that supports temporal disaggregation on a single prefill instance or spatial disaggregation across multiple instances. For short-prefill batches, a batch waiting window and CUDA Graph-based clustering mitigate interference from heterogeneous computation, reducing batching delay and lowering average latency. In real multi-turn workloads, LAPS reduces prefill latency by over 30\% compared to vanilla SGLang under prefill-decode disaggregation, and further decreases SLO violations by 28\% in multi-instance deployments with vanilla data-parallel configuration. Compared to the SGLang router with load balancing, it further lowers SLO violations by 12\% in multi-GPU settings. Under high concurrency and mixed-request scenarios, LAPS improves request throughput by 35\% serving Qwen2.5-32B model for prefill instance, demonstrating its effectiveness in optimizing heterogeneous LLM serving workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19970v1">Benchmarking LLAMA Model Security Against OWASP Top 10 For LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research prototypes to enterprise systems, their security vulnerabilities pose serious risks to data privacy and system integrity. This study benchmarks various Llama model variants against the OWASP Top 10 for LLM Applications framework, evaluating threat detection accuracy, response safety, and computational overhead. Using the FABRIC testbed with NVIDIA A30 GPUs, we tested five standard Llama models and five Llama Guard variants on 100 adversarial prompts covering ten vulnerability categories. Our results reveal significant differences in security performance: the compact Llama-Guard-3-1B model achieved the highest detection rate of 76% with minimal latency (0.165s per test), whereas base models such as Llama-3.1-8B failed to detect threats (0% accuracy) despite longer inference times (0.754s). We observe an inverse relationship between model size and security effectiveness, suggesting that smaller, specialized models often outperform larger general-purpose ones in security tasks. Additionally, we provide an open-source benchmark dataset including adversarial prompts, threat labels, and attack metadata to support reproducible research in AI security, [1].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18306v1">Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted to EACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Quantization is an effective technique for reducing the storage footprint and computational costs of Large Language Models (LLMs), but it often results in performance degradation. Existing post-training quantization methods typically use small, English-only calibration sets; however, their impact on multilingual models remains underexplored. We systematically evaluate eight calibration settings (five single-language and three multilingual mixes) on two quantizers (GPTQ, AWQ) on data from 10 languages. Our findings reveal a consistent trend: non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines. Specifically, we observe notable average perplexity gains across both quantizers on Llama3.1 8B and Qwen2.5 7B, with multilingual mixes achieving the largest overall reductions of up to 3.52 points in perplexity. Furthermore, our analysis indicates that tailoring calibration sets to the evaluation language yields the largest improvements for individual languages, underscoring the importance of linguistic alignment. We also identify specific failure cases where certain language-quantizer combinations degrade performance, which we trace to differences in activation range distributions across languages. These results highlight that static one-size-fits-all calibration is suboptimal and that tailoring calibration data, both in language and diversity, plays a crucial role in robustly quantizing multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01289v2">OntoMetric: An Ontology-Driven LLM-Assisted Framework for Automated ESG Metric Knowledge Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Environmental, Social, and Governance (ESG) metric knowledge is inherently structured, connecting industries, reporting frameworks, metric categories, metrics, and calculation models through compositional dependencies, yet in practice this structure remains embedded implicitly in regulatory documents such as SASB, TCFD, and IFRS S2 and rarely exists as an explicit, governed, or machine-actionable artefact. Existing ESG ontologies define formal schemas but do not address scalable population and governance from authoritative regulatory sources, while unconstrained large language model (LLM) extraction frequently produces semantically incorrect entities, hallucinated relationships, and structurally invalid graphs. OntoMetric is an ontology-guided framework for the automated construction and governance of ESG metric knowledge graphs from regulatory documents that operationalises the ESG Metric Knowledge Graph (ESGMKG) ontology as a first-class constraint embedded directly into the extraction and population process. The framework integrates structure-aware segmentation, ontology-constrained LLM extraction enriched with semantic fields and deterministic identifiers, and two-phase validation combining semantic type verification with rule-based schema checking, while preserving segment-level and page-level provenance to ensure traceability to regulatory source text. Evaluation on five ESG regulatory standards shows that ontology-guided extraction achieves 65-90 percent semantic accuracy and over 80 percent schema compliance, compared with 3-10 percent for unconstrained baseline extraction, and yields stable cost efficiency with a cost per validated entity of 0.01-0.02 USD and a 48 times efficiency improvement over baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18292v1">TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18282v1">Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15727v2">Towards Automated Kernel Generation in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18255v1">Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Continual learning in Large Language Models (LLMs) faces the critical challenge of balancing stability (retaining old knowledge) and plasticity (learning new tasks). While Experience Replay (ER) is a standard countermeasure against catastrophic forgetting, its impact across diverse capabilities remains underexplored. In this work, we uncover a critical dichotomy in ER's behavior: while it induces positive backward transfer on robust, unstructured tasks (e.g., boosting performance on previous NLP classification tasks through repeated rehearsal), it causes severe negative transfer on fragile, structured domains like code generation (e.g., a significant relative drop in coding accuracy). This reveals that ER trades structural integrity for broad consolidation. To address this dilemma, we propose \textbf{Orthogonal Subspace Wake-up (OSW)}. OSW identifies essential parameter subspaces of previous tasks via a brief "wake-up" phase and enforces orthogonal updates for new tasks, providing a mathematically grounded "safety guarantee" for established knowledge structures. Empirical results across a diverse four-task sequence demonstrate that OSW uniquely succeeds in preserving fragile coding abilities where Replay fails, while simultaneously maintaining high plasticity for novel tasks. Our findings emphasize the necessity of evaluating structural safety alongside average retention in LLM continual learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18253v1">BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 This is a pre-print
    </div>
    <details class="paper-abstract">
      Accurate evaluation of user satisfaction is critical for iterative development of conversational AI. However, for open-ended assistants, traditional A/B testing lacks reliable metrics: explicit feedback is sparse, while implicit metrics are ambiguous. To bridge this gap, we introduce BoRP (Bootstrapped Regression Probing), a scalable framework for high-fidelity satisfaction evaluation. Unlike generative approaches, BoRP leverages the geometric properties of LLM latent space. It employs a polarization-index-based bootstrapping mechanism to automate rubric generation and utilizes Partial Least Squares (PLS) to map hidden states to continuous scores. Experiments on industrial datasets show that BoRP (Qwen3-8B/14B) significantly outperforms generative baselines (even Qwen3-Max) in alignment with human judgments. Furthermore, BoRP reduces inference costs by orders of magnitude, enabling full-scale monitoring and highly sensitive A/B testing via CUPED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18241v1">TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at the 9th Workshop on Validation, Analysis and Evolution of Software Tests (VST 2026), co-located with the the 33rd IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2026)
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown promise in software engineering, their application to unit testing remains largely confined to isolated test generation or oracle prediction, neglecting the broader challenge of test suite maintenance. We introduce TAM-Eval (Test Automated Maintenance Evaluation), a framework and benchmark designed to evaluate model performance across three core test maintenance scenarios: creation, repair, and updating of test suites. Unlike prior work limited to function-level tasks, TAM-Eval operates at the test file level, while maintaining access to full repository context during isolated evaluation, better reflecting real-world maintenance workflows. Our benchmark comprises 1,539 automatically extracted and validated scenarios from Python, Java, and Go projects. TAM-Eval supports system-agnostic evaluation of both raw LLMs and agentic workflows, using a reference-free protocol based on test suite pass rate, code coverage, and mutation testing. Empirical results indicate that state-of-the-art LLMs have limited capabilities in realistic test maintenance processes and yield only marginal improvements in test effectiveness. We release TAM-Eval as an open-source framework to support future research in automated software testing. Our data and code are publicly available at https://github.com/trndcenter/TAM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00847v4">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 To appear in WWW 2026; 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $ε\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-ε}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18225v1">ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly deployed in e-commerce shopping. To perform thorough, user-tailored product searches, agents should interpret personal preferences, engage in multi-turn dialogues, and ultimately retrieve and discriminate among highly similar products. However, existing research has yet to provide a unified simulation environment that consistently captures all of these aspects, and always focuses solely on evaluation benchmarks without training support. In this paper, we introduce ShopSimulator, a large-scale and challenging Chinese shopping environment. Leveraging ShopSimulator, we evaluate LLMs across diverse scenarios, finding that even the best-performing models achieve less than 40% full-success rate. Error analysis reveals that agents struggle with deep search and product selection in long trajectories, fail to balance the use of personalization cues, and to effectively engage with users. Further training exploration provides practical guidance for overcoming these weaknesses, with the combination of supervised fine-tuning (SFT) and reinforcement learning (RL) yielding significant performance improvements. Code and data will be released at https://github.com/ShopAgent-Team/ShopSimulator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16781v2">Persuasion Tokens for Editing Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at EACL Main 2026
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18220v1">LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. The checkpoint and inference code will be released later.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18217v1">Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10132v2">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and prediction across different domains. Yet, their ability to infer temporal regularities from structured behavioral data remains underexplored. This paper presents a systematic study investigating whether LLMs can predict time intervals between recurring user actions, such as repeated purchases, and how different levels of contextual information shape their predictive behavior. Using a simple but representative repurchase scenario, we benchmark state-of-the-art LLMs in zero-shot settings against both statistical and machine-learning models. Two key findings emerge. First, while LLMs surpass lightweight statistical baselines, they consistently underperform dedicated machine-learning models, showing their limited ability to capture quantitative temporal structure. Second, although moderate context can improve LLM accuracy, adding further user-level detail degrades performance. These results challenge the assumption that "more context leads to better reasoning". Our study highlights fundamental limitations of today's LLMs in structured temporal inference and offers guidance for designing future context-aware hybrid models that integrate statistical precision with linguistic flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.07314v2">MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve superhuman performance on standardized medical licensing exams, these static benchmarks have become saturated and increasingly disconnected from the functional requirements of clinical workflows. To bridge the gap between theoretical capability and verified utility, we introduce MEDIC, a comprehensive evaluation framework establishing leading indicators across various clinical dimensions. Beyond standard question-answering, we assess operational capabilities using deterministic execution protocols and a novel Cross-Examination Framework (CEF), which quantifies information fidelity and hallucination rates without reliance on reference texts. Our evaluation across a heterogeneous task suite exposes critical performance trade-offs: we identify a significant knowledge-execution gap, where proficiency in static retrieval does not predict success in operational tasks such as clinical calculation or SQL generation. Furthermore, we observe a divergence between passive safety (refusal) and active safety (error detection), revealing that models fine-tuned for high refusal rates often fail to reliably audit clinical documentation for factual accuracy. These findings demonstrate that no single architecture dominates across all dimensions, highlighting the necessity of a portfolio approach to clinical model deployment. As part of this investigation, we released a public leaderboard on Hugging Face.\footnote{https://huggingface.co/spaces/m42-health/MEDIC-Benchmark}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12869v2">On the Fundamental Limits of LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Submitted to TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22922v4">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. Chain-of-thought (CoT) reasoning capabilities [1, 2] are well-known to help LLMs decompose a problem into steps and explore the solution spaces more effectively, leading to impressive performance on mathematical and reasoning benchmarks. As the length of CoT tokens per question increases substantially to even thousands of tokens per question [ 1], it is unknown how users could comprehend LLM reasoning and detect errors or hallucinations. To address this problem and understand how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph). That is, we ask LLMs themselves to generate an interactive web interface wrapped around the original CoT content, which may be presented in text (iCoT), graphs (iGraph) or code (iPoT). This interface allows users to interact with and provide a novel experience in reading and validating the reasoning chains of LLMs. Across a study of 125 participants, interactive interfaces significantly improve user performance. Specifically, iGraph users score the highest error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also lead to faster user validation time-iGraph users are faster (57.9 secs per question) than the users of iCoT and iPoT (60 secs) and the standard CoT (64.7 secs). A post-study questionnaire shows that users prefer iGraph, citing its superior ability to enable them to follow the LLM's reasoning. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18150v1">FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18146v1">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to ranking tasks in retrieval and recommendation. Although reasoning prompting can enhance ranking utility, our preliminary exploration reveals that its benefits are inconsistent and come at a substantial computational cost, suggesting that when to reason is as crucial as how to reason. To address this issue, we propose a reasoning routing framework that employs a lightweight, plug-and-play router head to decide whether to use direct inference (Non-Think) or reasoning (Think) for each instance before generation. The router head relies solely on pre-generation signals: i) compact ranking-aware features (e.g., candidate dispersion) and ii) model-aware difficulty signals derived from a diagnostic checklist reflecting the model's estimated need for reasoning. By leveraging these features before generation, the router outputs a controllable token that determines whether to apply the Think mode. Furthermore, the router can adaptively select its operating policy along the validation Pareto frontier during deployment, enabling dynamic allocation of computational resources toward instances most likely to benefit from Think under varying system constraints. Experiments on three public ranking datasets with different scales of open-source LLMs show consistent improvements in ranking utility with reduced token consumption (e.g., +6.3\% NDCG@10 with -49.5\% tokens on MovieLens with Qwen3-4B), demonstrating reasoning routing as a practical solution to the accuracy-efficiency trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09053v2">Who Fails Where? LLM and Human Error Patterns in Endometriosis Ultrasound Report Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In this study, we evaluate a locally-deployed large-language model (LLM) to convert unstructured endometriosis transvaginal ultrasound (eTVUS) scan reports into structured data for imaging informatics workflows. Across 49 eTVUS reports, we compared three LLMs (7B/8B and a 20B-parameter model) against expert human extraction. The 20B model achieved a mean accuracy of 86.02%, substantially outperforming smaller models and confirming the importance of scale in handling complex clinical text. Crucially, we identified a highly complementary error profile: the LLM excelled at syntactic consistency (e.g., date/numeric formatting) where humans faltered, while human experts provided superior semantic and contextual interpretation. We also found that the LLM's semantic errors were fundamental limitations that could not be mitigated by simple prompt engineering. These findings strongly support a human-in-the-loop (HITL) workflow in which the on-premise LLM serves as a collaborative tool, not a full replacement. It automates routine structuring and flags potential human errors, enabling imaging specialists to focus on high-level semantic validation. We discuss implications for structured reporting and interactive AI systems in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13481v2">neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Envy shapes competitiveness and cooperation in human groups, yet its role in large language model interactions remains largely unexplored. As LLMs increasingly operate in multi-agent settings, it is important to examine whether they exhibit envy-like preferences under social comparison. We evaluate LLM behavior across two scenarios: (1) a point-allocation game testing sensitivity to relative versus absolute payoff, and (2) comparative evaluations across general and contextual settings. To ground our analysis in psychological theory, we adapt four established psychometric questionnaires spanning general, domain-specific, workplace, and sibling-based envy. Our results reveal heterogeneous envy-like patterns across models and contexts, with some models sacrificing personal gain to reduce a peer's advantage, while others prioritize individual maximization. These findings highlight competitive dispositions as a design and safety consideration for multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18119v1">Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      SQL is central to enterprise data engineering, yet generating fully correct SQL code in a single attempt remains difficult, even for experienced developers and advanced text-to-SQL LLMs, often requiring multiple debugging iterations. We introduce OurBench, the first benchmark for enterprise-level SQL reasoning and debugging. Our benchmark is built on two key innovations: (1) an automated construction workflow that uses reverse engineering to systematically inject realistic bugs into large-scale SQL code, enabling scalable and diverse benchmark generation; and (2) an execution-free evaluation framework tailored to enterprise settings, providing fast, accurate, and resource-efficient assessment. OurBench comprises 469 OurBenchSyn queries featuring syntax errors with explicit error messages, and 516 OurBenchSem queries targeting semantic errors in which the code fails to meet user intent. The queries are highly complex, averaging over 140 lines and featuring deep and wide abstract syntax trees. Evaluation of nearly 30 LLMs reveals a substantial performance gap: the best-performing model, Claude-4-Sonnet, achieves only 36.46 percent accuracy on OurBenchSyn and 32.17 percent on OurBenchSem, while most models score below 20 percent. We further explore four solution strategies, identify key challenges, and outline promising directions for enterprise SQL debugging with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18116v1">FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis. We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs. Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18110v1">AttenMIA: LLM Membership Inference Attack through Attention Signals</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed to enable or improve a multitude of real-world applications. Given the large size of their training data sets, their tendency to memorize training data raises serious privacy and intellectual property concerns. A key threat is the membership inference attack (MIA), which aims to determine whether a given sample was included in the model's training set. Existing MIAs for LLMs rely primarily on output confidence scores or embedding-based features, but these signals are often brittle, leading to limited attack success. We introduce AttenMIA, a new MIA framework that exploits self-attention patterns inside the transformer model to infer membership. Attention controls the information flow within the transformer, exposing different patterns for memorization that can be used to identify members of the dataset. Our method uses information from attention heads across layers and combines them with perturbation-based divergence metrics to train an effective MIA classifier. Using extensive experiments on open-source models including LLaMA-2, Pythia, and Opt models, we show that attention-based features consistently outperform baselines, particularly under the important low-false-positive metric (e.g., achieving up to 0.996 ROC AUC & 87.9% TPR@1%FPR on the WikiMIA-32 benchmark with Llama2-13b). We show that attention signals generalize across datasets and architectures, and provide a layer- and head-level analysis of where membership leakage is most pronounced. We also show that using AttenMIA to replace other membership inference attacks in a data extraction framework results in training data extraction attacks that outperform the state of the art. Our findings reveal that attention mechanisms, originally introduced to enhance interpretability, can inadvertently amplify privacy risks in LLMs, underscoring the need for new defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18096v1">Enhancing LLM-based Recommendation with Preference Hint Discovery from Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs have garnered substantial attention in recommendation systems. Yet they fall short of traditional recommenders when capturing complex preference patterns. Recent works have tried integrating traditional recommendation embeddings into LLMs to resolve this issue, yet a core gap persists between their continuous embedding and discrete semantic spaces. Intuitively, textual attributes derived from interactions can serve as critical preference rationales for LLMs' recommendation logic. However, directly inputting such attribute knowledge presents two core challenges: (1) Deficiency of sparse interactions in reflecting preference hints for unseen items; (2) Substantial noise introduction from treating all attributes as hints. To this end, we propose a preference hint discovery model based on the interaction-integrated knowledge graph, enhancing LLM-based recommendation. It utilizes traditional recommendation principles to selectively extract crucial attributes as hints. Specifically, we design a collaborative preference hint extraction schema, which utilizes semantic knowledge from similar users' explicit interactions as hints for unseen items. Furthermore, we develop an instance-wise dual-attention mechanism to quantify the preference credibility of candidate attributes, identifying hints specific to each unseen item. Using these item- and user-based hints, we adopt a flattened hint organization method to shorten input length and feed the textual hint information to the LLM for commonsense reasoning. Extensive experiments on both pair-wise and list-wise recommendation tasks verify the effectiveness of our proposed framework, indicating an average relative improvement of over 3.02% against baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18091v1">From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly costly to deploy, motivating extensive research on model pruning. However, most existing studies focus on instruction-following LLMs, leaving it unclear whether established pruning strategies transfer to reasoning-augmented models that explicitly generate long intermediate reasoning traces. In this work, we conduct a controlled study of pruning for both instruction-following ($\textbf{LLM-instruct}$) and reasoning-augmented ($\textbf{LLM-think}$) models. To isolate the effects of pruning, we align pruning calibration and post-pruning recovery data with each model's original training distribution, which we show yields more stable and reliable pruning behavior. We evaluate static depth pruning, static width pruning, and dynamic pruning across 17 tasks spanning classification, generation, and reasoning. Our results reveal clear paradigm-dependent differences: depth pruning outperforms width pruning on classification tasks, while width pruning is more robust for generation and reasoning. Moreover, static pruning better preserves reasoning performance, whereas dynamic pruning excels on classification and generation but remains challenging for long-chain reasoning. These findings underscore the need for pruning strategies that explicitly account for the distinct characteristics of reasoning-augmented LLMs. Our code is publicly available at https://github.com/EIT-NLP/LRM-Pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16432v2">iPDB -- Optimizing SQL Queries with ML and LLM Predicates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Structured Query Language (SQL) has remained the standard query language for databases. SQL is highly optimized for processing structured data laid out in relations. Meanwhile, in the present application development landscape, it is highly desirable to utilize the power of learned models to perform complex tasks. Large language models (LLMs) have been shown to understand and extract information from unstructured textual data. However, SQL as a query language and accompanying relational database systems are either incompatible or inefficient for workloads that require leveraging learned models. This results in complex engineering and multiple data migration operations that move data between the data sources and the model inference platform. In this paper, we present iPDB, a relational system that supports in-database machine learning (ML) and large language model (LLM) inferencing using extended SQL syntax. In iPDB, LLMs and ML calls can function as semantic projects, as predicates to perform semantic selects and semantic joins, or for semantic grouping in group-by clauses. iPDB has a novel relational predict operator and semantic query optimizations that enable users to write and efficiently execute semantic SQL queries, outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13358v5">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 We resolved some issues in this paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18077v1">Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18067v1">EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 17 pages, 6 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Verilog's design cycle is inherently labor-intensive and necessitates extensive domain expertise. Although Large Language Models (LLMs) offer a promising pathway toward automation, their limited training data and intrinsic sequential reasoning fail to capture the strict formal logic and concurrency inherent in hardware systems. To overcome these barriers, we present EvolVE, the first framework to analyze multiple evolution strategies on chip design tasks, revealing that Monte Carlo Tree Search (MCTS) excels at maximizing functional correctness, while Idea-Guided Refinement (IGR) proves superior for optimization. We further leverage Structured Testbench Generation (STG) to accelerate the evolutionary process. To address the lack of complex optimization benchmarks, we introduce IC-RTL, targeting industry-scale problems derived from the National Integrated Circuit Contest. Evaluations establish EvolVE as the new state-of-the-art, achieving 98.1% on VerilogEval v2 and 92% on RTLLM v2. Furthermore, on the industry-scale IC-RTL suite, our framework surpasses reference implementations authored by contest participants, reducing the Power, Performance, Area (PPA) product by up to 66% in Huffman Coding and 17% in the geometric mean across all problems. The source code of the IC-RTL benchmark is available at https://github.com/weiber2002/ICRTL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18053v1">Addressing LLM Diversity by Infusing Random Concepts</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to produce outputs with limited diversity. In this work, we study whether infusing random concepts in the prompts can improve the diversity of the generated outputs. To benchmark the approach, we design a systematic evaluation protocol which involves prompting an LLM with questions of the form "Name 10 Hollywood actors", and analyzing diversity measures of the resulting LLM outputs. Our experiments on multiple LLMs show that prepending random words/sentences unrelated to the prompt result in greater diversity in the outputs of LLMs. We believe that this promising result and the evaluation protocol opens up interesting avenues for future work, such as how infusing randomness into LLMs could be applied to other domains. Further, the evaluation protocol could also inspire research into benchmarking LLM diversity more systematically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14852v2">Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 NeurIPS 2025. 27 pages
    </div>
    <details class="paper-abstract">
      LLM-based agent applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs and latency due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agent applications where outputs depend on external data and environmental contexts. We propose Agentic Plan Caching (APC), a novel test-time memory that extracts, stores, adapts, and reuses structured plan templates from planning stages of agent applications across semantically similar tasks to reduce the cost and latency of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agent applications shows that our system can reduce costs by 50.31% and latency by 27.28% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06201v2">K2-V2: A 360-Open, Reasoning-Enhanced LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      We introduce K2-V2, a 360-open LLM built from scratch as a superior base for reasoning adaptation, in addition to functions such as conversation and knowledge retrieval from general LLMs. It stands as the strongest fully open model, rivals open-weight leaders in its size class, outperforms Qwen2.5-72B and approaches the performance of Qwen3-235B. We actively infuse domain knowledge, reasoning, long-context, and tool use throughout the training process. This explicitly prepares the model for complex reasoning tasks. We demonstrate this potential using simple supervised fine-tuning, establishing a strong baseline that indicates significant headroom for advanced alignment. By releasing the full training history and data composition, we maximize the effectiveness of continuous training, a key open source production scenario. We release the model weights and signature LLM360 artifacts, such as complete training data, to empower the community with a capable, reasoning-centric foundation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18987v1">LLMs versus the Halting Problem: Revisiting Program Termination Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18984v1">Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful framework for improving the reasoning capabilities of large language models (LLMs). However, most existing RL approaches rely on sparse outcome rewards, which fail to credit correct intermediate steps in partially successful solutions. Process reward models (PRMs) offer fine-grained step-level supervision, but their scores are often noisy and difficult to evaluate. As a result, recent PRM benchmarks focus on a more objective capability: detecting the first incorrect step in a reasoning path. However, this evaluation target is misaligned with how PRMs are typically used in RL, where their step-wise scores are treated as raw rewards to maximize. To bridge this gap, we propose Verifiable Prefix Policy Optimization (VPPO), which uses PRMs only to localize the first error during RL. Given an incorrect rollout, VPPO partitions the trajectory into a verified correct prefix and an erroneous suffix based on the first error, rewarding the former while applying targeted penalties only after the detected mistake. This design yields stable, interpretable learning signals and improves credit assignment. Across multiple reasoning benchmarks, VPPO consistently outperforms sparse-reward RL and prior PRM-guided baselines on both Pass@1 and Pass@K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18949v1">Tricky$^2$: Towards a Benchmark for Evaluating Human and LLM Error Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into software development workflows, yet they often introduce subtle logic or data-misuse errors that differ from human bugs. To study how these two error types interact, we construct Tricky$^2$, a hybrid dataset that augments the existing TrickyBugs corpus of human-written defects with errors injected by both GPT-5 and OpenAI-oss-20b across C++, Python, and Java programs. Our approach uses a taxonomy-guided prompting framework to generate machine-originated bugs while preserving original human defects and program structure. The resulting corpus spans human-only, LLM-only, and human+LLM splits, enabling analysis of mixed-origin error behavior, multi-bug repair robustness, and reliability in hybrid human-machine code. This paper outlines the dataset construction pipeline and illustrates its use through small-scale baseline evaluations of classification, localization, and repair tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18904v1">SICL-AT: Another way to adapt Auditory LLM to low-resource task</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource or unfamiliar tasks. In case of labeled in-domain data is scarce or mismatched to the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that \emph{Vanilla ICL}, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose \textbf{Speech In-Context Learning Adaptation Training (SICL-AT)}, a post-training recipe utilizes only high resource speech data intending to strengthen model's in-context learning capability. The enhancement can generalize to audio understanding/reasoning task. Experiments indicate our proposed method consistently outperforms direct fine-tuning in low-resource scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18899v1">Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18846v1">LLM Driven Design of Continuous Optimization Problems with Controllable High-level Properties</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 17 pages, accepted at EvoApplications 2026
    </div>
    <details class="paper-abstract">
      Benchmarking in continuous black-box optimisation is hindered by the limited structural diversity of existing test suites such as BBOB. We explore whether large language models embedded in an evolutionary loop can be used to design optimisation problems with clearly defined high-level landscape characteristics. Using the LLaMEA framework, we guide an LLM to generate problem code from natural-language descriptions of target properties, including multimodality, separability, basin-size homogeneity, search-space homogeneity and globallocal optima contrast. Inside the loop we score candidates through ELA-based property predictors. We introduce an ELA-space fitness-sharing mechanism that increases population diversity and steers the generator away from redundant landscapes. A complementary basin-of-attraction analysis, statistical testing and visual inspection, verifies that many of the generated functions indeed exhibit the intended structural traits. In addition, a t-SNE embedding shows that they expand the BBOB instance space rather than forming an unrelated cluster. The resulting library provides a broad, interpretable, and reproducible set of benchmark problems for landscape analysis and downstream tasks such as automated algorithm selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18844v1">Reducing False Positives in Static Bug Detection with LLMs: An Empirical Study in Industry</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Static analysis tools (SATs) are widely adopted in both academia and industry for improving software quality, yet their practical use is often hindered by high false positive rates, especially in large-scale enterprise systems. These false alarms demand substantial manual inspection, creating severe inefficiencies in industrial code review. While recent work has demonstrated the potential of large language models (LLMs) for false alarm reduction on open-source benchmarks, their effectiveness in real-world enterprise settings remains unclear. To bridge this gap, we conduct the first comprehensive empirical study of diverse LLM-based false alarm reduction techniques in an industrial context at Tencent, one of the largest IT companies in China. Using data from Tencent's enterprise-customized SAT on its large-scale Advertising and Marketing Services software, we construct a dataset of 433 alarms (328 false positives, 105 true positives) covering three common bug types. Through interviewing developers and analyzing the data, our results highlight the prevalence of false positives, which wastes substantial manual effort (e.g., 10-20 minutes of manual inspection per alarm). Meanwhile, our results show the huge potential of LLMs for reducing false alarms in industrial settings (e.g., hybrid techniques of LLM and static analysis eliminate 94-98% of false positives with high recall). Furthermore, LLM-based techniques are cost-effective, with per-alarm costs as low as 2.1-109.5 seconds and $0.0011-$0.12, representing orders-of-magnitude savings compared to manual review. Finally, our case analysis further identifies key limitations of LLM-based false alarm reduction in industrial settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18785v1">Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Extended abstract presented at the 2025 Wordplay Workshop at EMNLP
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has enabled a new paradigm for bridging authorial intent and player agency in interactive narrative. We consider this paradigm through the example of Dramamancer, a system that uses an LLM to transform author-created story schemas into player-driven playthroughs. This extended abstract outlines some design techniques and evaluation considerations associated with this system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18777v1">PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at AAAI 2026 - Innovative Applications of AI (IAAI-26)
    </div>
    <details class="paper-abstract">
      Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18754v1">$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18753v1">HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Have been accepted by ICLR'26
    </div>
    <details class="paper-abstract">
      The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: data-driven hallucinations and reasoning-driven hallucinations. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the Hallucination Risk Bound, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce HalluGuard, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate HalluGuard on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02599v2">Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18731v1">One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and automatically provide individualized feedback. However, developing these models faces two critical challenges: the scarcity of feedback from individual users and the need for efficient adaptation to unseen users. We argue that addressing these constraints requires a paradigm shift from fitting data to learn user preferences to learn the process of preference adaptation. To realize this, we propose Meta Reward Modeling (MRM), which reformulates personalized reward modeling as a meta-learning problem. Specifically, we represent each user's reward model as a weighted combination of base reward functions, and optimize the initialization of these weights using a Model-Agnostic Meta-Learning (MAML)-style framework to support fast adaptation under limited feedback. To ensure robustness, we introduce the Robust Personalization Objective (RPO), which places greater emphasis on hard-to-learn users during meta optimization. Extensive experiments on personalized preference datasets validate that MRM enhances few-shot personalization, improves user robustness, and consistently outperforms baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18706v1">Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Rubrics are essential for evaluating open-ended LLM responses, especially in safety-critical domains such as healthcare. However, creating high-quality and domain-specific rubrics typically requires significant human expertise time and development cost, making rubric-based evaluation and training difficult to scale. In this work, we introduce Health-SCORE, a generalizable and scalable rubric-based training and evaluation framework that substantially reduces rubric development costs without sacrificing performance. We show that Health-SCORE provides two practical benefits beyond standalone evaluation: it can be used as a structured reward signal to guide reinforcement learning with safety-aware supervision, and it can be incorporated directly into prompts to improve response quality through in-context learning. Across open-ended healthcare tasks, Health-SCORE achieves evaluation quality comparable to human-created rubrics while significantly lowering development effort, making rubric-based evaluation and training more scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17853v3">CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which assesses whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 17%, and achieves up to 68.1% accuracy on the CiteME benchmark, approaching human-level performance (69.7%). It also enables the identification of alternative but valid citations and demonstrates generalization ability for cross-domain citation attribution.Our code is available at https://github.com/KathCYM/CiteGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12945v3">LLMPopcorn: Exploring LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted by ICASSP2026
    </div>
    <details class="paper-abstract">
      In an era where micro-videos dominate platforms like TikTok and YouTube, AI-generated content is nearing cinematic quality. The next frontier is using large language models (LLMs) to autonomously create viral micro-videos, a largely untapped potential that could shape the future of AI-driven content creation. To address this gap, this paper presents the first exploration of LLM-assisted popular micro-video generation (LLMPopcorn). We selected popcorn as the icon for this paper because it symbolizes leisure and entertainment, aligning with this study on leveraging LLMs as assistants for generating popular micro-videos that are often consumed during leisure time. Specifically, we empirically study the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? Exploring these questions, we show that advanced LLMs like DeepSeek-V3 can generate micro-videos with popularity rivaling human content. Prompt enhancement further boosts results, while benchmarking highlights DeepSeek-V3 and R1 for LLMs, and LTX-Video and HunyuanVideo for video generation. This work advances AI-assisted micro-video creation and opens new research directions. The code is publicly available at https://github.com/GAIR-Lab/LLMPopcorn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06822v2">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The advent of complex, interconnected long-horizon LLM systems has made it incredibly tricky to identify where and when these systems break down. Evaluation capabilities that currently exist today are limited in that they often focus on simple metrics, end-to-end outcomes, and are dependent on the perspectives of humans. In order to match the increasing complexity of these many component systems, evaluation frameworks must also be able to reason, probe, iterate, and understand the nuanced logic passing through these systems. In this paper, we present RAFFLES, an offline evaluation architecture that incorporates iterative reasoning. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically identify faults and a set of specialized Evaluators to assess the quality of the candidate faults as well as rationales of the Judge. We evaluated RAFFLES with several benchmarks - the Who&When dataset to identify step-level faults in multi-agent systems and the ReasonEval datasets to diagnose step-level mathematical reasoning errors. RAFFLES outperforms strong baselines, achieving an accuracy of over 20% and 50% on the Who&When Hand-Crafted and Algorithmically-Generated datasets, and over 80% on the ReasonEval datasets. These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15765v2">Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Data is a critical asset for training large language models (LLMs), alongside compute resources and skilled workers. While some training data is publicly available, substantial investment is required to generate proprietary datasets, such as human preference annotations or to curate new ones from existing sources. As larger datasets generally yield better model performance, two natural questions arise. First, how can data owners make informed decisions about curation strategies and data sources investment? Second, how can multiple data owners collaboratively pool their resources to train superior models while fairly distributing the benefits? This problem, data valuation, which is not specific to large language models, has been addressed by the machine learning community through the lens of cooperative game theory, with the Shapley value being the prevalent solution concept. However, computing Shapley values is notoriously expensive for data valuation, typically requiring numerous model retrainings, which can become prohibitive for large machine learning models. In this work, we demonstrate that this computational challenge is dramatically simplified for LLMs trained with Direct Preference Optimization (DPO). We show how the specific mathematical structure of DPO enables scalable Shapley value computation. We believe this observation unlocks many applications at the intersection of data valuation and large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18630v1">Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The escalating global mental health crisis, marked by persistent treatment gaps, availability, and a shortage of qualified therapists, positions Large Language Models (LLMs) as a promising avenue for scalable support. While LLMs offer potential for accessible emotional assistance, their reliability, therapeutic relevance, and alignment with human standards remain challenging to address. This paper introduces a human-grounded evaluation methodology designed to assess LLM generated responses in therapeutic dialogue. Our approach involved curating a dataset of 500 mental health conversations from datasets with real-world scenario questions and evaluating the responses generated by nine diverse LLMs, including closed source and open source models. More specifically, these responses were evaluated by two psychiatric trained experts, who independently rated each on a 5 point Likert scale across a comprehensive 6 attribute rubric. This rubric captures Cognitive Support and Affective Resonance, providing a multidimensional perspective on therapeutic quality. Our analysis reveals that LLMs provide strong cognitive reliability by producing safe, coherent, and clinically appropriate information, but they demonstrate unstable affective alignment. Although closed source models (e.g., GPT-4o) offer balanced therapeutic responses, open source models show greater variability and emotional flatness. We reveal a persistent cognitive-affective gap and highlight the need for failure aware, clinically grounded evaluation frameworks that prioritize relational sensitivity alongside informational accuracy in mental health oriented LLMs. We advocate for balanced evaluation protocols with human in the loop that center on therapeutic sensitivity and provide a framework to guide the responsible design and clinical oversight of mental health oriented conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18582v1">From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 9 pages, 4 figures, AAAI 2026 Bridge
    </div>
    <details class="paper-abstract">
      Personality detection aims to measure an individual's corresponding personality traits through their social media posts. The advancements in Large Language Models (LLMs) offer novel perspectives for personality detection tasks. Existing approaches enhance personality trait analysis by leveraging LLMs to extract semantic information from textual posts as prompts, followed by training classifiers for categorization. However, accurately classifying personality traits remains challenging due to the inherent complexity of human personality and subtle inter-trait distinctions. Moreover, prompt-based methods often exhibit excessive dependency on expert-crafted knowledge without autonomous pattern-learning capacity. To address these limitations, we view personality detection as a ranking task rather than a classification and propose a corresponding reinforcement learning training paradigm. First, we employ supervised fine-tuning (SFT) to establish personality trait ranking capabilities while enforcing standardized output formats, creating a robust initialization. Subsequently, we introduce Group Relative Policy Optimization (GRPO) with a specialized ranking-based reward function. Unlike verification tasks with definitive solutions, personality assessment involves subjective interpretations and blurred boundaries between trait categories. Our reward function explicitly addresses this challenge by training LLMs to learn optimal answer rankings. Comprehensive experiments have demonstrated that our method achieves state-of-the-art performance across multiple personality detection benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18572v1">One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfair outcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to study bias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variations (robustness) and the rarity of some cues in real interactions (external validity). We compare six commonly used persona cues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce substantial variance in responses across personas. We therefore caution against claims from a single persona cue and recommend future personalization research to evaluate multiple externally valid cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18563v1">An LLM-Agent-Based Framework for Age of Information Optimization in Heterogeneous Random Access Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      With the rapid expansion of the Internet of Things (IoT) and heterogeneous wireless networks, the Age of Information (AoI) has emerged as a critical metric for evaluating the performance of real-time and personalized systems. While AoI-based random access is essential for next-generation applications such as the low-altitude economy and indoor service robots, existing strategies, ranging from rule-based protocols to learning-based methods, face critical challenges, including idealized model assumptions, slow convergence, and poor generalization. In this article, we propose Reflex-Core, a novel Large Language Model (LLM) agent-based framework for AoI-driven random access in heterogeneous networks. By devising an "Observe-Reflect-Decide-Execute" closed-loop mechanism, this framework integrates Supervised Fine-Tuning (SFT) and Proximal Policy Optimization (PPO) to enable optimal, autonomous access control. Based on the Reflex-Core framework, we develop a Reflexive Multiple Access (RMA) protocol and a priority-based RMA variant for intelligent access control under different heterogeneous network settings. Experimental results demonstrate that in the investigated scenarios, the RMA protocol achieves up to a 14.9% reduction in average AoI compared with existing baselines, while the priority-based version improves the convergence rate by approximately 20%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18552v1">Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs are increasingly embedded in everyday decision-making, yet their outputs can encode subtle, unintended behaviours that shape user beliefs and actions. We refer to these covert, goal-directed behaviours as hidden intentions, which may arise from training and optimisation artefacts, or be deliberately induced by an adversarial developer, yet remain difficult to detect in practice. We introduce a taxonomy of ten categories of hidden intentions, grounded in social science research and organised by intent, mechanism, context, and impact, shifting attention from surface-level behaviours to design-level strategies of influence. We show how hidden intentions can be easily induced in controlled models, providing both testbeds for evaluation and demonstrations of potential misuse. We systematically assess detection methods, including reasoning and non-reasoning LLM judges, and find that detection collapses in realistic open-world settings, particularly under low-prevalence conditions, where false positives overwhelm precision and false negatives conceal true risks. Stress tests on precision-prevalence and precision-FNR trade-offs reveal why auditing fails without vanishingly small false positive rates or strong priors on manipulation types. Finally, a qualitative case study shows that all ten categories manifest in deployed, state-of-the-art LLMs, emphasising the urgent need for robust frameworks. Our work provides the first systematic analysis of detectability failures of hidden intentions in LLMs under open-world settings, offering a foundation for understanding, inducing, and stress-testing such behaviours, and establishing a flexible taxonomy for anticipating evolving threats and informing governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18052v2">The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 13 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed to simulate human collective behaviors, yet the methodological rigor of these "AI societies" remains under-explored. Through a systematic audit of 42 recent studies, we identify six pervasive flaws-spanning agent profiles, interaction, memory, control, unawareness, and realism (PIMMUR). Our analysis reveals that 90.7% of studies violate at least one principle, undermining simulation validity. We demonstrate that frontier LLMs correctly identify the underlying social experiment in 47.6% of cases, while 65.3% of prompts exert excessive control that pre-determines outcomes. By reproducing five representative experiments (e.g., telephone game), we show that reported collective phenomena often vanish or reverse when PIMMUR principles are enforced, suggesting that many "emergent" behaviors are methodological artifacts rather than genuine social dynamics. Our findings suggest that current AI simulations may capture model-specific biases rather than universal human social behaviors, raising critical concerns about the use of LLMs as scientific proxies for human society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18510v1">Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at https://github.com/liushiliushi/JitRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18492v1">DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18483v1">Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at EACL main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer strong generative capabilities, but many applications require explicit and \textit{fine-grained} control over specific textual concepts, such as humor, persuasiveness, or formality. Prior approaches in prompting and representation engineering can provide coarse or single-attribute control, but systematic evaluation of multi-attribute settings remains limited. We introduce an evaluation framework for fine-grained controllability for both single- and dual-concept scenarios, focusing on linguistically distinct concept pairs (e.g., persuasiveness vs.~humor). Surprisingly, across multiple LLMs and generative tasks, we find that performance often drops in the dual-concept setting, even though the chosen concepts should in principle be separable. This reveals a fundamental limitation of naive prompting-based control: models struggle with compositionality even when concepts are intuitively independent. Our framework provides systematic evidence of this gap and offers a principled approach for measuring the ability of future methods for multi-concept control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18457v1">Token-level Collaborative Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures, 7 tables, WWW 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong potential for generative recommendation by leveraging rich semantic knowledge. However, existing LLM-based recommender systems struggle to effectively incorporate collaborative filtering (CF) signals, due to a fundamental mismatch between item-level preference modeling in CF and token-level next-token prediction (NTP) optimization in LLMs. Prior approaches typically treat CF as contextual hints or representation bias, and resort to multi-stage training to reduce behavioral semantic space discrepancies, leaving CF unable to explicitly regulate LLM generation. In this work, we propose Token-level Collaborative Alignment for Recommendation (TCA4Rec), a model-agnostic and plug-and-play framework that establishes an explicit optimization-level interface between CF supervision and LLM generation. TCA4Rec consists of (i) Collaborative Tokenizer, which projects raw item-level CF logits into token-level distributions aligned with the LLM token space, and (ii) Soft Label Alignment, which integrates these CF-informed distributions with one-hot supervision to optimize a soft NTP objective. This design preserves the generative nature of LLM training while enabling collaborative alignment with essential user preference of CF models. We highlight TCA4Rec is compatible with arbitrary traditional CF models and generalizes across a wide range of decoder-based LLM recommender architectures. Moreover, it provides an explicit mechanism to balance behavioral alignment and semantic fluency, yielding generative recommendations that are both accurate and controllable. Extensive experiments demonstrate that TCA4Rec consistently improves recommendation performance across a broad spectrum of CF models and LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v3">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and a lack of customized modeling frameworks. This paper proposes an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategies, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation. To handle the scalability issues inherent in large-scale P2P networks, a differential attention-based critic network is introduced to efficiently extract key interaction features and enhance convergence. Experimental results demonstrate that LLM-generated strategies effectively substitute human experts. The proposed imitative expert MARL algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baseline algorithms, while maintaining robust stability. This paper provides an effective solution for the real-time decision-making of the P2P electricity market by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20984v2">Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 NeurIPS 2025 Poster
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities but typically require extensive computational resources and memory for inference. Post-training quantization (PTQ) can effectively reduce these demands by storing weights in lower bit-width formats. However, standard uniform quantization often leads to notable performance degradation, particularly in low-bit scenarios. In this work, we introduce a Grouped Lattice Vector Quantization (GLVQ) framework that assigns each group of weights a customized lattice codebook, defined by a learnable generation matrix. To address the non-differentiability of the quantization process, we adopt Babai rounding to approximate nearest-lattice-point search during training, which enables stable optimization of the generation matrices. Once trained, decoding reduces to a simple matrix-vector multiplication, yielding an efficient and practical quantization pipeline. Experiments on multiple benchmarks show that our approach achieves a better trade-off between model size and accuracy compared to existing post-training quantization baselines, highlighting its effectiveness in deploying large models under stringent resource constraints. Our source code is available on GitHub repository: https://github.com/xzhang9308/GLVQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16495v2">Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Revised
    </div>
    <details class="paper-abstract">
      Efficient parallelism is necessary for achieving low-latency, high-throughput inference with large language models (LLMs). Tensor parallelism (TP) is the state-of-the-art method for reducing LLM response latency, however GPU communications reduces combined token throughput. On the other hand, data parallelism (DP) obtains a higher throughput yet is slow in response latency. Best of both worlds does not exist, and it is not possible to combine TP and DP because of the KV cache variance across the parallelisms. We notice Sequence Parallelism (SP - Ulysses in training) has similar properties as DP but with KV cache invariance. We adapt SP to inference, and combine it with TP to get the best of both worlds. Our solution: Shift Parallelism. Shift Parallelism dynamically switches across TP and SP, and minimizes latency in low traffic without losing throughput in high traffic. The efficient GPU communications of Shift Parallelism yields up to i) 1.51x faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution. We evaluate Shift Parallelism with real-world production traces with dynamic traffic patterns as well as synthetic benchmarking patterns across models, context sizes, and arrival rates. All results affirm the same: Shift Parallelism has a better the latency vs. throughput tradeoff than TP or DP, and hence obtains low latency without degrading throughput in dynamic workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08653v2">Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18375v1">Hierarchical Text Classification with LLM-Refined Taxonomies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Hierarchical text classification (HTC) depends on taxonomies that organize labels into structured hierarchies. However, many real-world taxonomies introduce ambiguities, such as identical leaf names under similar parent nodes, which prevent language models (LMs) from learning clear decision boundaries. In this paper, we present TaxMorph, a framework that uses large language models (LLMs) to transform entire taxonomies through operations such as renaming, merging, splitting, and reordering. Unlike prior work, our method revises the full hierarchy to better match the semantics encoded by LMs. Experiments across three HTC benchmarks show that LLM-refined taxonomies consistently outperform human-curated ones in various settings up to +2.9pp. in F1. To better understand these improvements, we compare how well LMs can assign leaf nodes to parent nodes and vice versa across human-curated and LLM-refined taxonomies. We find that human-curated taxonomies lead to more easily separable clusters in embedding space. However, the LLM-refined taxonomies align more closely with the model's actual confusion patterns during classification. In other words, even though they are harder to separate, they better reflect the model's inductive biases. These findings suggest that LLM-guided refinement creates taxonomies that are more compatible with how models learn, improving HTC performance.
    </details>
</div>
