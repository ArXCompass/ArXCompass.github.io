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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05786v3">FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 9 pages, 3 figures, 3 tables and 2 algorithms
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14275v3">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17567v1">Real-Time Trend Prediction via Continually-Aligned LLM Query Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Trending news detection in low-traffic search environments faces a fundamental cold-start problem, where a lack of query volume prevents systems from identifying emerging or long-tail trends. Existing methods relying on keyword frequency or query spikes are inherently slow and ineffective in these sparse settings, lagging behind real-world shifts in attention. We introduce RTTP, a novel Real-Time Trending Prediction framework that generates search queries directly from news content instead of waiting for users to issue them. RTTP leverages a continual learning LLM (CL-LLM) that converts posts into search-style queries and scores them using engagement strength + creator authority, enabling early trend surfacing before search volume forms. To ensure adaptation without degrading reasoning, we propose Mix-Policy DPO, a new preference-based continual learning approach that combines on-policy stability with off-policy novelty to mitigate catastrophic forgetting during model upgrades. Deployed at production scale on Facebook and Meta AI products, RTTP delivers +91.4% improvement in tail-trend detection precision@500 and +19% query generation accuracy over industry baselines, while sustaining stable performance after multi-week online training. This work demonstrates that LLM-generated synthetic search signals, when aligned and continually updated, unlock timely trend understanding in low-traffic search environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17551v1">GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Paper under submisison
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities, but their broad deployment is limited by significant computational resource demands, particularly energy consumption during inference. Static, one-model-fits-all inference strategies are often inefficient, as they do not exploit the diverse range of available models or adapt to varying query requirements. This paper presents GreenServ, a dynamic, context-aware routing framework that optimizes the trade-off between inference accuracy and energy efficiency. GreenServ extracts lightweight contextual features from each query, including task type, semantic cluster, and text complexity, and routes queries to the most suitable model from a heterogeneous pool, based on observed accuracy and energy usage. We employ a multi-armed bandit approach to learn adaptive routing policies online. This approach operates under partial feedback, eliminates the need for extensive offline calibration, and streamlines the integration of new models into the inference pipeline. We evaluated GreenServ across five benchmark tasks and a pool of 16 contemporary open-access LLMs. Experimental results show that GreenServ consistently outperforms static (single-model) and random baselines. In particular, compared to random routing, GreenServ achieved a 22% increase in accuracy while reducing cumulative energy consumption by 31%. Finally, we evaluated GreenServ with RouterBench, achieving an average accuracy of 71.7% with a peak accuracy of 75.7%. All artifacts are open-source and available as an anonymous repository for review purposes here: https://anonymous.4open.science/r/llm-inference-router-EBEA/README.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15338v2">HeartLLM: Discretized ECG Tokenization for LLM-Based Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Electrocardiography (ECG) plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present HeartLLM, a novel framework that integrates time-series (TS) and language modeling by enabling large language models (LLMs) to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into quantized codes using a lead-wise encoder and quantization module. These quantized codes are then mapped to an extended ECG vocabulary to form ECG tokens, enabling the model to process both ECG and natural language inputs within a unified framework. To bridge the modality gap, we pretrain the model on an autoregressive ECG token forecasting task, allowing the LLM to capture temporal dynamics through its inherent language modeling capability. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, HeartLLM achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating discretized ECG tokens into LLMs for medical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17540v1">Ethical Risk Assessment of the Data Harnessing Process of LLM supported on Consensus of Well-known Multi-Ethical Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have revolutionized natural language processing, unlocking unprecedented capabilities in communication, automation, and knowledge generation. However, the ethical implications of LLM development, particularly in data harnessing, remain a critical challenge. Despite widespread discussion about the ethical compliance of LLMs -- especially concerning their data harnessing processes, there remains a notable absence of concrete frameworks to systematically guide or measure the ethical risks involved. In this paper we discuss a potential pathway for building an Ethical Risk Scoring (ERS) system to quantitatively assess the ethical integrity of the data harnessing process for AI systems. This system is based on a set of assessment questions grounded in core ethical principles, which are, in turn, supported by commanding ethical theories. By integrating measurable scoring mechanisms, this approach aims to foster responsible LLM development, balancing technological innovation with ethical accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13742v2">Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) judges exhibit strong reasoning capabilities but are limited to textual content. This leaves current automatic Speech-to-Speech (S2S) evaluation methods reliant on opaque and expensive Audio Language Models (ALMs). In this work, we propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. To demonstrate the strength of the framework, we first introduce a Human Chain-of-Thought (HCoT) annotation protocol to improve the diagnostic capability of existing judge benchmarks by separating evaluation into explicit dimensions: content (C), voice quality (VQ), and paralinguistics (P). Using this data, TRACE constructs a textual blueprint of inexpensive audio signals and prompts an LLM to render dimension-wise judgments, fusing them into an overall rating via a deterministic policy. TRACE achieves higher agreement with human raters than ALMs and transcript-only LLM judges while being significantly more cost-effective. We will release the HCoT annotations and the TRACE framework to enable scalable and human-aligned S2S evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17527v1">Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      As LLMs increasingly function as economic agents, the specific mechanisms LLMs use to update their belief with heterogeneous signals remain opaque. We design experiments and develop a Behavioral Kalman Filter framework to quantify how LLM-based agents update expectations, acting as households or firm CEOs, update expectations when presented with individual and aggregate signals. The results from experiments and model estimation reveal four consistent patterns: (1) agents' weighting of priors and signals deviates from unity; (2) both household and firm CEO agents place substantially larger weights on individual signals compared to aggregate signals; (3) we identify a significant and negative interaction between concurrent signals, implying that the presence of multiple information sources diminishes the marginal weight assigned to each individual signal; and (4) expectation formation patterns differ significantly between household and firm CEO agents. Finally, we demonstrate that LoRA fine-tuning mitigates, but does not fully eliminate, behavioral biases in LLM expectation formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21981v2">Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17471v1">PatchIsland: Orchestration of LLM Agents for Continuous Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Continuous fuzzing platforms such as OSS-Fuzz uncover large numbers of vulnerabilities, yet the subsequent repair process remains largely manual. Unfortunately, existing Automated Vulnerability Repair (AVR) techniques -- including recent LLM-based systems -- are not directly applicable to continuous fuzzing. This is because these systems are designed and evaluated on a static, single-run benchmark setting, making them ill-suited for the diverse, noisy, and failure-prone environments in continuous fuzzing. To address these issues, we introduce PatchIsland, a system for Continuous Vulnerability Repair (CVR) that tightly integrates with continuous fuzzing pipelines. PatchIsland employs an ensemble of diverse LLM agents. By leveraging multiple LLM agents, PatchIsland can cover a wider range of settings (e.g., different projects, bug types, and programming languages) and also improve operational robustness. In addition, PatchIsland utilizes a two-phase patch-based deduplication to mitigate duplicate crashes and patches, which can be problematic in continuous fuzzing. In our internal evaluation, PatchIsland repaired 84 of 92 vulnerabilities, demonstrating strong repair capability. In the official AIxCC competition, the system operated with no human intervention in a fully autonomous environment and successfully patched 31 out of 43 vulnerabilities, achieving a repair rate of 72.1\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18951v4">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 29 pages, 10 figures, NeurIPS 2025 Main
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v3">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11056v3">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 TheWebConf 2026 Industry
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under ``standard'' and ``reasoning-augmented'' inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17421v1">Oops, Wait: Token-Level Signals as a Lens into LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The emergence of discourse-like tokens such as "wait" and "therefore" in large language models (LLMs) has offered a unique window into their reasoning processes. However, systematic analyses of how such signals vary across training strategies and model scales remain lacking. In this paper, we analyze token-level signals through token probabilities across various models. We find that specific tokens strongly correlate with reasoning correctness, varying with training strategies while remaining stable across model scales. A closer look at the "wait" token in relation to answer probability demonstrates that models fine-tuned on small-scale datasets acquire reasoning ability through such signals but exploit them only partially. This work provides a systematic lens to observe and understand the dynamics of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17418v1">GraphPilot: GUI Task Automation with One-Step LLM Reasoning Powered by Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 This paper is accepted by the Journal of Intelligent Computing and Networking (JICN) for publication
    </div>
    <details class="paper-abstract">
      Mobile graphical user interface (GUI) agents are designed to automate everyday tasks on smartphones. Recent advances in large language models (LLMs) have significantly enhanced the capabilities of mobile GUI agents. However, most LLM-powered mobile GUI agents operate in stepwise query-act loops, which incur high latency due to repeated LLM queries. We present GraphPilot, a mobile GUI agent that leverages knowledge graphs of the target apps to complete user tasks in almost one LLM query. GraphPilot operates in two complementary phases to enable efficient and reliable LLM-powered GUI task automation. In the offline phase, it explores target apps, records and analyzes interaction history, and constructs an app-specific knowledge graph that encodes functions of pages and elements as well as transition rules for each app. In the online phase, given an app and a user task, it leverages the knowledge graph of the given app to guide the reasoning process of LLM. When the reasoning process encounters uncertainty, GraphPilot dynamically requests the HTML representation of the current interface to refine subsequent reasoning. Finally, a validator checks the generated sequence of actions against the transition rules in the knowledge graph, performing iterative corrections to ensure it is valid. The structured, informative information in the knowledge graph allows the LLM to plan the complete sequence of actions required to complete the user task. On the DroidTask benchmark, GraphPilot improves task completion rate over Mind2Web and AutoDroid, while substantially reducing latency and the number of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17399v1">ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved rapid progress in Chinese language understanding, yet accurately evaluating their capabilities remains challenged by benchmark saturation and prohibitive computational costs. While static leaderboards provide snapshot rankings, they often mask the structural trade-offs between capabilities. In this work, we present ReLE (Robust Efficient Live Evaluation), a scalable system designed to diagnose Capability Anisotropy, the non-uniformity of model performance across domains. Using ReLE, we evaluate 304 models (189 commercial, 115 open-source) across a Domain $\times$ Capability orthogonal matrix comprising 207,843 samples. We introduce two methodological contributions to address current evaluation pitfalls: (1) A Symbolic-Grounded Hybrid Scoring Mechanism that eliminates embedding-based false positives in reasoning tasks; (2) A Dynamic Variance-Aware Scheduler based on Neyman allocation with noise correction, which reduces compute costs by 70\% compared to full-pass evaluations while maintaining a ranking correlation of $ρ=0.96$. Our analysis reveals that aggregate rankings are highly sensitive to weighting schemes: models exhibit a Rank Stability Amplitude (RSA) of 11.4 in ReLE versus $\sim$5.0 in traditional benchmarks, confirming that modern models are highly specialized rather than generally superior. We position ReLE not as a replacement for comprehensive static benchmarks, but as a high-frequency diagnostic monitor for the evolving model landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17397v1">CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL MME workshop paper
    </div>
    <details class="paper-abstract">
      Knowledge Editing (KE) has emerged as a promising paradigm for updating facts in Large Language Models (LLMs) without retraining. However, progress in Multilingual Knowledge Editing (MKE) is currently hindered by biased evaluation frameworks. We observe that existing MKE benchmarks are typically constructed by mechanically translating English-centric datasets into target languages (e.g., English-to-Chinese). This approach introduces translation artifacts and neglects culturally specific entities native to the target language, failing to reflect the true knowledge distribution of LLMs. To address this, we propose CLM-Bench, a culture-aware benchmark constructed using a native Chinese-first methodology. We curate 1,010 high-quality CounterFact pairs rooted in Chinese cultural contexts and align them with English counterparts. Using CLM-Bench, we conduct extensive experiments on representative LLMs (e.g., Llama-3, Qwen2) and reveal a significant Cross-lingual Misalignment: edits in one language function independently and fail to propagate to the other. We further provide a geometric explanation via layer-wise representation analysis, demonstrating that edit vectors for Chinese and English are nearly orthogonal -- residing in disjoint subspaces -- while mixed-lingual editing exhibits linear additivity of these vectors. Our findings challenge the effectiveness of current methods in cross-lingual transfer and underscore the importance of culturally native benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22767v2">TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically deployed using a fixed architecture, despite growing evidence that not all layers contribute equally to every downstream task. In this work, we introduce TALE (Task-Aware Layer Elimination), an inference-time method that improves task performance by selectively removing layers that are irrelevant or detrimental for a given task. TALE optimizes task-specific validation performance, yielding a task-adapted architecture without retraining or modifying model weights. Across 9 tasks and 5 model families, under both zero-shot and few-shot settings, we show that TALE consistently matches or surpasses baseline performance while simultaneously reducing computational cost, outperforming general and layer-wise pruning approaches such as SLEB. Beyond inference-time gains, TALE synergizes with fine-tuning and few-shot learning, where task-adapted architectures lead to additional performance improvements. Computing TALE for a new task requires modest resources (1-2 GPU hours on an A100), making it a practical and deployable solution for task-specialized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15830v2">DAIQ: Auditing Demographic Attribute Inference from Question in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent evaluations of Large language models (LLMs) audit social bias primarily through prompts that explicitly reference demographic attributes, overlooking whether models infer sensitive demographics from neutral questions. Such inference constitutes epistemic overreach and raises concerns for privacy. We introduce Demographic Attribute Inference from Questions (DAIQ), a diagnostic audit framework for evaluating demographic inference under epistemic uncertainty. We evaluate 18 open- and closed-source LLMs across six real-world domains and five demographic attributes. We find that many models infer demographics from neutral questions, defaulting to socially dominant categories and producing stereotype-aligned rationales. These behaviors persist across model families, scales and decoding settings, indicating reliance on learned population priors. We further show that inferred demographics can condition downstream responses and that abstention oriented prompting substantially reduces unintended inference without model fine-tuning. Our results suggest that current bias evaluations are incomplete and motivate evaluation standards that assess not only how models respond to demographic information, but whether they should infer it at all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23277v2">Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) often suffers from long and noisy retrieved contexts. Prior context compression methods rely on predefined importance metrics or supervised compression models, rather than on the model's own inference-time behavior. We propose Sentinel, a lightweight sentence-level compression framework that treats context compression as an understanding decoding problem. Sentinel probes native attention behaviors of a frozen LLM with a lightweight readout to decode which parts of the context are actually utilized when answering a query, rather than using attention as a direct relevance score. We empirically observe that decoded relevance signals exhibit sufficient consistency across model scales to support effective compression with compact proxy models. On LongBench, Sentinel with a 0.5B proxy model achieves up to 5x compression while matching the QA performance of 7B-scale baselines, and despite being trained only on English QA data, generalizes effectively to Chinese and out-of-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16559v4">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17346v1">Multi-Agent Learning Path Planning via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into intelligent tutoring systems offers transformative potential for personalized learning in higher education. However, most existing learning path planning approaches lack transparency, adaptability, and learner-centered explainability. To address these challenges, this study proposes a novel Multi-Agent Learning Path Planning (MALPP) framework that leverages a role- and rule-based collaboration mechanism among intelligent agents, each powered by LLMs. The framework includes three task-specific agents: a learner analytics agent, a path planning agent, and a reflection agent. These agents collaborate via structured prompts and predefined rules to analyze learning profiles, generate tailored learning paths, and iteratively refine them with interpretable feedback. Grounded in Cognitive Load Theory and Zone of Proximal Development, the system ensures that recommended paths are cognitively aligned and pedagogically meaningful. Experiments conducted on the MOOCCubeX dataset using seven LLMs show that MALPP significantly outperforms baseline models in path quality, knowledge sequence consistency, and cognitive load alignment. Ablation studies further validate the effectiveness of the collaborative mechanism and theoretical constraints. This research contributes to the development of trustworthy, explainable AI in education and demonstrates a scalable approach to learner-centered adaptive instruction powered by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17343v1">Are We Evaluating the Edit Locality of LLM Model Editing Properly?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Model editing has recently emerged as a popular paradigm for efficiently updating knowledge in LLMs. A central desideratum of updating knowledge is to balance editing efficacy, i.e., the successful injection of target knowledge, and specificity (also known as edit locality), i.e., the preservation of existing non-target knowledge. However, we find that existing specificity evaluation protocols are inadequate for this purpose. We systematically elaborated on the three fundamental issues it faces. Beyond the conceptual issues, we further empirically demonstrate that existing specificity metrics are weakly correlated with the strength of specificity regularizers. We also find that current metrics lack sufficient sensitivity, rendering them ineffective at distinguishing the specificity performance of different methods. Finally, we propose a constructive evaluation protocol. Under this protocol, the conflict between open-ended LLMs and the assumption of determined answers is eliminated, query-independent fluency biases are avoided, and the evaluation strictness can be smoothly adjusted within a near-continuous space. Experiments across various LLMs, datasets, and editing methods show that metrics derived from the proposed protocol are more sensitive to changes in the strength of specificity regularizers and exhibit strong correlation with them, enabling more fine-grained discrimination of different methods' knowledge preservation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v5">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Locating files and functions requiring modification in large software repositories is challenging due to their scale and structural complexity. Existing LLM-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which often overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool: jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a base pretrained model, without relying on closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and the 32B model exceeding closed-source models such as GPT-5 on most metrics. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05758v2">EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17570v2">GreedySnake: Accelerating SSD-Offloaded LLM Training with Efficient Scheduling and Optimizer Step Overlapping</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      SSD-offloaded training offers a practical and promising approach to making LLM training cost-effective. Building on gradient accumulation with micro-batches, this paper introduces GreedySnake, a new SSD-offloaded training system that employs vertical scheduling, which executes all microbatches of a layer before proceeding to the next. Compared to existing systems that use horizontal scheduling (i.e., executing micro-batches sequentially), GreedySnake achieves higher training throughput with smaller batch sizes, bringing the system much closer to the ideal scenario predicted by the roofline model. To further mitigate the I/O bottleneck, GreedySnake overlaps part of the optimization step with the forward pass of the next iteration. Experimental results on A100 GPUs show that GreedySnake achieves saturated training throughput improvements over ZeRO-Infinity: 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20697v2">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate more advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response entropy. By constraining entropy, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task performance and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14005v3">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious instructions, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, an effective and efficient detection method based on the observation that instruction-tuned LLMs internally encode distinguishable signals for prompts containing injected instructions. PIShield leverages residual-stream representations and a simple linear classifier to detect prompt injection, without expensive model fine-tuning or response generation. We conduct extensive evaluations on a diverse set of short- and long-context benchmarks. The results show that PIShield consistently achieves low false positive and false negative rates, significantly outperforming existing baselines. These findings demonstrate that internal representations of instruction-tuned LLMs provide a powerful and practical foundation for prompt injection detection in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17292v1">Risk-based test framework for LLM features in regulated software</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large language models are increasingly embedded in regulated and safety-critical software, including clinical research platforms and healthcare information systems. While these features enable natural language search, summarization, and configuration assistance, they introduce risks such as hallucinations, harmful or out-of-scope advice, privacy and security issues, bias, instability under change, and adversarial misuse. Prior work on machine learning testing and AI assurance offers useful concepts but limited guidance for interactive, product-embedded assistants. This paper proposes a risk-based testing framework for LLM features in regulated software: a six-category risk taxonomy, a layered test strategy mapping risks to concrete tests across guardrail, orchestration, and system layers, and a case study applying the approach to a Knowledgebase assistant in a clinical research platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17284v1">Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models in Medical Question Answering is severely hampered by ambiguous user queries, a significant safety risk that demonstrably reduces answer accuracy in high-stakes healthcare settings. In this paper, we formalize this challenge by linking input ambiguity to aleatoric uncertainty (AU), which is the irreducible uncertainty arising from underspecified input. To facilitate research in this direction, we construct CV-MedBench, the first benchmark designed for studying input ambiguity in Medical QA. Using this benchmark, we analyze AU from a representation engineering perspective, revealing that AU is linearly encoded in LLM's internal activation patterns. Leveraging this insight, we introduce a novel AU-guided "Clarify-Before-Answer" framework, which incorporates AU-Probe - a lightweight module that detects input ambiguity directly from hidden states. Unlike existing uncertainty estimation methods, AU-Probe requires neither LLM fine-tuning nor multiple forward passes, enabling an efficient mechanism to proactively request user clarification and significantly enhance safety. Extensive experiments across four open LLMs demonstrate the effectiveness of our QA framework, with an average accuracy improvement of 9.48% over baselines. Our framework provides an efficient and robust solution for safe Medical QA, strengthening the reliability of health-related applications. The code is available at https://github.com/yaokunliu/AU-Med.git, and the CV-MedBench dataset is released on Hugging Face at https://huggingface.co/datasets/yaokunl/CV-MedBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17275v1">Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 12 pages,
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate exceptional performance in surface-level text generation, their nature in handling complex multi-step reasoning tasks often remains one of ``statistical fitting'' rather than systematic logical deduction. Traditional Reinforcement Learning (RL) attempts to mitigate this by introducing a ``think-before-speak'' paradigm. However, applying RL directly in high-dimensional, discrete token spaces faces three inherent challenges: sample-inefficient rollouts, high gradient estimation variance, and the risk of catastrophic forgetting. To fundamentally address these structural bottlenecks, we propose \textbf{DeepLatent Reasoning (DLR)}, a latent-space bidirectional contrastive reinforcement learning framework. This framework shifts the trial-and-error cost from expensive token-level full sequence generation to the continuous latent manifold. Specifically, we introduce a lightweight assistant model to efficiently sample $K$ reasoning chain encodings within the latent space. These encodings are filtered via a dual reward mechanism based on correctness and formatting; only high-value latent trajectories are fed into a \textbf{frozen main model} for single-pass decoding. To maximize reasoning diversity while maintaining coherence, we design a contrastive learning objective to enable directed exploration within the latent space. Since the main model parameters remain frozen during optimization, this method mathematically eliminates catastrophic forgetting. Experiments demonstrate that under comparable GPU computational budgets, DLR achieves more stable training convergence, supports longer-horizon reasoning chains, and facilitates the sustainable accumulation of reasoning capabilities, providing a viable path toward reliable and scalable reinforcement learning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17261v1">AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 21 pages in total, including 9 pages of main text, with 4 figures and 3 tables. This manuscript is submitted to arXiv
    </div>
    <details class="paper-abstract">
      Zeroth-Order (ZO) optimization has emerged as a promising solution for fine-tuning LLMs under strict memory constraints, as it avoids the prohibitive memory cost of storing activations for backpropagation. However, existing ZO methods typically employ isotropic perturbations, neglecting the rich structural information available during the forward pass. In this paper, we identify a crucial link between gradient formation and activation structure: the gradient of a linear layer is confined to the subspace spanned by its input activations. Leveraging this insight, we propose Activation-Guided Zeroth-Order optimization (AGZO). Unlike prior methods, AGZO extracts a compact, activation-informed subspace on the fly during the forward pass and restricts perturbations to this low-rank subspace. We provide a theoretical framework showing that AGZO optimizes a subspace-smoothed objective and provably yields update directions with higher cosine similarity to the true gradient than isotropic baselines. Empirically, we evaluate AGZO on Qwen3 and Pangu models across various benchmarks. AGZO consistently outperforms state-of-the-art ZO baselines and significantly narrows the performance gap with first-order fine-tuning, while maintaining almost the same peak memory footprint as other ZO methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16979v1">A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the curvature evolution of the loss landscape is fundamental to analyzing the training dynamics of neural networks. The most commonly studied measure, Hessian sharpness ($λ_{\max}^H$) -- the largest eigenvalue of the loss Hessian -- determines local training stability and interacts with the learning rate throughout training. Despite its significance in analyzing training dynamics, direct measurement of Hessian sharpness remains prohibitive for Large Language Models (LLMs) due to high computational cost. We analyze $\textit{critical sharpness}$ ($λ_c$), a computationally efficient measure requiring fewer than $10$ forward passes given the update direction $Δ\mathbfθ$. Critically, this measure captures well-documented Hessian sharpness phenomena, including progressive sharpening and Edge of Stability. Using this measure, we provide the first demonstration of these sharpness phenomena at scale, up to $7$B parameters, spanning both pre-training and mid-training of OLMo-2 models. We further introduce $\textit{relative critical sharpness}$ ($λ_c^{1\to 2}$), which quantifies the curvature of one loss landscape while optimizing another, to analyze the transition from pre-training to fine-tuning and guide data mixing strategies. Critical sharpness provides practitioners with a practical tool for diagnosing curvature dynamics and informing data composition choices at scale. More broadly, our work shows that scalable curvature measures can provide actionable insights for large-scale training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18261v3">LLM Reasoning for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Published on Proceedings of the ACM on Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16964v1">AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has sparked growing interest in their integration into autonomous systems for reasoning-driven perception, planning, and decision-making. However, evaluating and training such agentic AI models remains challenging due to the lack of large-scale, structured, and safety-critical benchmarks. This paper introduces AgentDrive, an open benchmark dataset containing 300,000 LLM-generated driving scenarios designed for training, fine-tuning, and evaluating autonomous agents under diverse conditions. AgentDrive formalizes a factorized scenario space across seven orthogonal axes: scenario type, driver behavior, environment, road layout, objective, difficulty, and traffic density. An LLM-driven prompt-to-JSON pipeline generates semantically rich, simulation-ready specifications that are validated against physical and schema constraints. Each scenario undergoes simulation rollouts, surrogate safety metric computation, and rule-based outcome labeling. To complement simulation-based evaluation, we introduce AgentDrive-MCQ, a 100,000-question multiple-choice benchmark spanning five reasoning dimensions: physics, policy, hybrid, scenario, and comparative reasoning. We conduct a large-scale evaluation of fifty leading LLMs on AgentDrive-MCQ. Results show that while proprietary frontier models perform best in contextual and policy reasoning, advanced open models are rapidly closing the gap in structured and physics-grounded reasoning. We release the AgentDrive dataset, AgentDrive-MCQ benchmark, evaluation code, and related materials at https://github.com/maferrag/AgentDrive
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16956v1">DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The rapid growth of Large Transformer-based models, specifically Large Language Models (LLMs), now scaling to trillions of parameters, has necessitated training across thousands of GPUs using complex hybrid parallelism strategies (e.g., data, tensor, and pipeline parallelism). Checkpointing this massive, distributed state is critical for a wide range of use cases, such as resilience, suspend-resume, investigating undesirable training trajectories, and explaining model evolution. However, existing checkpointing solutions typically treat model state as opaque binary blobs, ignoring the ``3D heterogeneity'' of the underlying data structures--varying by memory location (GPU vs. Host), number of ``logical'' objects sharded and split across multiple files, data types (tensors vs. Python objects), and their serialization requirements. This results in significant runtime overheads due to blocking device-to-host transfers, data-oblivious serialization, and storage I/O contention. In this paper, we introduce DataStates-LLM, a novel checkpointing architecture that leverages State Providers to decouple state abstraction from data movement. DataStates-LLM exploits the immutability of model parameters during the forward and backward passes to perform ``lazy'', non-blocking asynchronous snapshots. By introducing State Providers, we efficiently coalesce fragmented, heterogeneous shards and overlap the serialization of metadata with bulk tensor I/O. We evaluate DataStates-LLM on models up to 70B parameters on 256 A100-40GB GPUs. Our results demonstrate that DataStates-LLM achieves up to 4$\times$ higher checkpointing throughput and reduces end-to-end training time by up to 2.2$\times$ compared to state-of-the-art solutions, effectively mitigating the serialization and heterogeneity bottlenecks in extreme-scale LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.17406v3">ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      In cybersecurity, security analysts constantly face the challenge of mitigating newly discovered vulnerabilities in real-time, with over 300,000 vulnerabilities identified since 1999. The sheer volume of known vulnerabilities complicates the detection of patterns for unknown threats. While LLMs can assist, they often hallucinate and lack alignment with recent threats. Over 40,000 vulnerabilities have been identified in 2024 alone, which are introduced after most popular LLMs' (e.g., GPT-5) training data cutoff. This raises a major challenge of leveraging LLMs in cybersecurity, where accuracy and up-to-date information are paramount. Therefore, we aim to improve the adaptation of LLMs in vulnerability analysis by mimicking how an analyst performs such tasks. We propose ProveRAG, an LLM-powered system designed to assist in rapidly analyzing vulnerabilities with automated retrieval augmentation of web data while self-evaluating its responses with verifiable evidence. ProveRAG incorporates a self-critique mechanism to help alleviate the omission and hallucination common in the output of LLMs applied in cybersecurity applications. The system cross-references data from verifiable sources (NVD and CWE), giving analysts confidence in the actionable insights provided. Our results indicate that ProveRAG excels in delivering verifiable evidence to the user with over 99% and 97% accuracy in exploitation and mitigation strategies, respectively. ProveRAG guides analysts to secure their systems more effectively by overcoming temporal and context-window limitations while also documenting the process for future audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16890v1">LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Automated fact-checking (AFC) systems are susceptible to adversarial attacks, enabling false claims to evade detection. Existing adversarial frameworks typically rely on injecting noise or altering semantics, yet no existing framework exploits the adversarial potential of persuasion techniques, which are widely used in disinformation campaigns to manipulate audiences. In this paper, we introduce a novel class of persuasive adversarial attacks on AFCs by employing a generative LLM to rephrase claims using persuasion techniques. Considering 15 techniques grouped into 6 categories, we study the effects of persuasion on both claim verification and evidence retrieval using a decoupled evaluation strategy. Experiments on the FEVER and FEVEROUS benchmarks show that persuasion attacks can substantially degrade both verification performance and evidence retrieval. Our analysis identifies persuasion techniques as a potent class of adversarial attacks, highlighting the need for more robust AFC systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00570v2">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enhanced conventional recommendation models via user profiling, which generates representative textual profiles from users' historical interactions. However, their direct application to session-based recommendation (SBR) remains challenging due to severe session context scarcity and poor scalability. In this paper, we propose SPRINT, a scalable SBR framework that incorporates reliable and informative intents while ensuring high efficiency in both training and inference. SPRINT constrains LLM-based profiling with a global intent pool and validates inferred intents based on recommendation performance to mitigate noise and hallucinations under limited context. To ensure scalability, LLMs are selectively invoked only for uncertain sessions during training, while a lightweight intent predictor generalizes intent prediction to all sessions without LLM dependency at inference time. Experiments on real-world datasets show that SPRINT consistently outperforms state-of-the-art methods while providing more explainable recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v4">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16781v1">Persuasion Tokens for Editing Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted at EACL Main 2026
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16775v1">LLM-powered Real-time Patent Citation Recommendation for Financial Technologies</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Rapid financial innovation has been accompanied by a sharp increase in patenting activity, making timely and comprehensive prior-art discovery more difficult. This problem is especially evident in financial technologies, where innovations develop quickly, patent collections grow continuously, and citation recommendation systems must be updated as new applications arrive. Existing patent retrieval and citation recommendation methods typically rely on static indexes or periodic retraining, which limits their ability to operate effectively in such dynamic settings. In this study, we propose a real-time patent citation recommendation framework designed for large and fast-changing financial patent corpora. Using a dataset of 428,843 financial patents granted by the China National Intellectual Property Administration (CNIPA) between 2000 and 2024, we build a three-stage recommendation pipeline. The pipeline uses large language model (LLM) embeddings to represent the semantic content of patent abstracts, applies efficient approximate nearest-neighbor search to construct a manageable candidate set, and ranks candidates by semantic similarity to produce top-k citation recommendations. In addition to improving recommendation accuracy, the proposed framework directly addresses the dynamic nature of patent systems. By using an incremental indexing strategy based on hierarchical navigable small-world (HNSW) graphs, newly issued patents can be added without rebuilding the entire index. A rolling day-by-day update experiment shows that incremental updating improves recall while substantially reducing computational cost compared with rebuild-based indexing. The proposed method also consistently outperforms traditional text-based baselines and alternative nearest-neighbor retrieval approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16766v1">Do LLM hallucination detectors suffer from low-resource effect?</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted at EACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      LLMs, while outperforming humans in a wide range of tasks, can still fail in unanticipated ways. We focus on two pervasive failure modes: (i) hallucinations, where models produce incorrect information about the world, and (ii) the low-resource effect, where the models show impressive performance in high-resource languages like English but the performance degrades significantly in low-resource languages like Bengali. We study the intersection of these issues and ask: do hallucination detectors suffer from the low-resource effect? We conduct experiments on five tasks across three domains (factual recall, STEM, and Humanities). Experiments with four LLMs and three hallucination detectors reveal a curious finding: As expected, the task accuracies in low-resource languages experience large drops (compared to English). However, the drop in detectors' accuracy is often several times smaller than the drop in task accuracy. Our findings suggest that even in low-resource languages, the internal mechanisms of LLMs might encode signals about their uncertainty. Further, the detectors are robust within language (even for non-English) and in multilingual setups, but not in cross-lingual settings without in-language supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04633v2">Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 10 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The unjudged document problem, where systems that did not contribute to the original judgement pool may retrieve documents without a relevance judgement, is a key obstacle to the reuseability of test collections in information retrieval. While the de facto standard to deal with the problem is to treat unjudged documents as non-relevant, many alternatives have been proposed, such as the use of large language models (LLMs) as a relevance judge (LLM-as-a-judge). However, this has been criticized, among other things, as circular, since the same LLM can be used as the ranker and the judge. We propose to train topic-specific relevance classifiers instead: By finetuning monoT5 with independent LoRA weight adaptation on the judgments of a single assessor for a single topic's pool, we align it to that assessor's notion of relevance for the topic. The system rankings obtained through our classifier's relevance judgments achieve a Spearmans' $ρ$ correlation of $>0.94$ with ground truth system rankings. As little as 128 initial human judgments per topic suffice to improve the comparability of models, compared to treating unjudged documents as non-relevant, while achieving more reliability than existing LLM-as-a-judge approaches. Topic-specific relevance classifiers are thus a lightweight and straightforward way to tackle the unjudged document problem, while maintaining human judgments as the gold standard for retrieval evaluation. Code, models, and data are made openly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06161v2">LATTLE: LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Transfer learning on tabular data is challenging due to disparate feature spaces across domains, in contrast to the homogeneous structures of image and text. Large language models (LLMs) offer a knowledge base to improve the limited effectiveness of cross-domain transfer learning for tabular data. However, LLM performance often stagnates due to subjective text prompts and the computational limitations of in-context learning. We present a novel language-to-tabular context-learning method that uses attention-specific transformer weights, enabling seamless transfer learning across disparate tabular data sets. The LLM attention transplant mechanism facilitates a domain-agnostic transfer learning, eliminating the need for shared features between tables, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of disjoint source-target data sets and 12 baseline methods demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and models trained on thousands to billions of tabular samples. The proposed cross-domain attention transfer demonstrates an effective solution for adapting LLMs to learning non-text tabular data in a low-resource environment. The source code of the LATTLE implementation is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16711v1">Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to EACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      Generalization to unseen concepts is a central challenge due to the scarcity of human annotations in Mention-agnostic Biomedical Concept Recognition (MA-BCR). This work makes two key contributions to systematically address this issue. First, we propose an evaluation framework built on hierarchical concept indices and novel metrics to measure generalization. Second, we explore LLM-based Auto-Labeled Data (ALD) as a scalable resource, creating a task-specific pipeline for its generation. Our research unequivocally shows that while LLM-generated ALD cannot fully substitute for manual annotations, it is a valuable resource for improving generalization, successfully providing models with the broader coverage and structural knowledge needed to approach recognizing unseen concepts. Code and datasets are available at https://github.com/bio-ie-tool/hi-ald.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16699v1">Supporting Stakeholder Requirements Expression with LLM Revisions: An Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 This paper has been accepted at the research track of the 32nd International Working Conference on Requirements Engineering: Foundation for Software Quality (REFSQ 2026)
    </div>
    <details class="paper-abstract">
      Stakeholders often struggle to accurately express their requirements due to articulation barriers arising from limited domain knowledge or from cognitive constraints. This can cause misalignment between expressed and intended requirements, complicating elicitation and validation. Traditional elicitation techniques, such as interviews and follow-up sessions, are time-consuming and risk distorting stakeholders' original intent across iterations. Large Language Models (LLMs) can infer user intentions from context, suggesting potential for assisting stakeholders in expressing their needs. This raises the questions of (i) how effectively LLMs can support requirement expression and (ii) whether such support benefits stakeholders with limited domain expertise. We conducted a study with 26 participants who produced 130 requirement statements. Each participant first expressed requirements unaided, then evaluated LLM-generated revisions tailored to their context. Participants rated LLM revisions significantly higher than their original statements across all dimensions-alignment with intent, readability, reasoning, and unambiguity. Qualitative feedback further showed that LLM revisions often surfaced tacit details stakeholders considered important and helped them better understand their own requirements. We present and evaluate a stakeholder-centered approach that leverages LLMs as articulation aids in requirements elicitation and validation. Our results show that LLM-assisted reformulation improves perceived completeness, clarity, and alignment of requirements. By keeping stakeholders in the validation loop, this approach promotes responsible and trustworthy use of AI in Requirements Engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14558v2">LLM Jailbreak Detection for (Almost) Free!</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EMNLP 2025 (Findings) https://aclanthology.org/2025.findings-emnlp.309/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16669v1">PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied to legal domain-specific tasks, evaluating their ability to perform legal work in real-world settings has become essential. However, existing legal benchmarks rely on simplified and highly standardized tasks, failing to capture the ambiguity, complexity, and reasoning demands of real legal practice. Moreover, prior evaluations often adopt coarse, single-dimensional metrics and do not explicitly assess fine-grained legal reasoning. To address these limitations, we introduce PLawBench, a Practical Law Benchmark designed to evaluate LLMs in realistic legal practice scenarios. Grounded in real-world legal workflows, PLawBench models the core processes of legal practitioners through three task categories: public legal consultation, practical case analysis, and legal document generation. These tasks assess a model's ability to identify legal issues and key facts, perform structured legal reasoning, and generate legally coherent documents. PLawBench comprises 850 questions across 13 practical legal scenarios, with each question accompanied by expert-designed evaluation rubrics, resulting in approximately 12,500 rubric items for fine-grained assessment. Using an LLM-based evaluator aligned with human expert judgments, we evaluate 10 state-of-the-art LLMs. Experimental results show that none achieves strong performance on PLawBench, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs. Data is available at: https://github.com/skylenage/PLawbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16651v1">Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03093v2">Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 To appear in Proc. ICASSP 2026, May 04-08, 2026, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16621v1">How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses. However, irrelevant personalized memories are often introduced into the context, interfering with the LLM's intent understanding. To comprehensively investigate the dual effects of personalization, we develop RPEval, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol. RPEval reveals the widespread phenomenon of irrational personalization in existing LLMs and, through error pattern analysis, illustrates its negative impact on user experience. Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on RPEval, and resolves 80% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at https://github.com/XueyangFeng/RPEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16618v1">PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) excel in many tasks, their application to Speech-to-Speech Translation (S2ST) is underexplored and hindered by data scarcity. To bridge this gap, we propose PROST-LLM (PROgressive Speech-to-speech Translation) to enhance the S2ST capabilities in LLMs progressively. First, we fine-tune the LLMs with the CVSS corpus, employing designed tri-task learning and chain of modality methods to boost the initial performance. Then, leveraging the fine-tuned model, we generate preference pairs through self-sampling and back-translation without human evaluation. Finally, these preference pairs are used for preference optimization to enhance the model's S2ST capability further. Extensive experiments confirm the effectiveness of our proposed PROST-LLM in improving the S2ST capability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14122v2">Benchmarking LLMs for Political Science: A United Nations Perspective</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 This paper has been accepted at AAAI 2026 as an oral paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: https://github.com/yueqingliang1/UNBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12146v2">From LLMs to Agents in Programming: The Impact of Providing an LLM with a Compiler</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated a remarkable capability in natural language and program generation and software development. However, the source code generated by the LLMs does not always meet quality requirements and may fail to compile. Therefore, many studies evolve into agents that can reason about the problem before generating the source code for the solution. The goal of this paper is to study the degree to which such agents benefit from access to software development tools, in our case, a gcc compiler. We conduct a computational experiment on the RosettaCode dataset, on 699 programming tasks in C. We evaluate how the integration with a compiler shifts the role of the language model from a passive generator to an active agent capable of iteratively developing runnable programs based on feedback from the compiler. We evaluated 16 language models with sizes ranging from small (135 million) to medium (3 billion) and large (70 billion). Our results show that access to a compiler improved the compilation success by 5.3 to 79.4 percentage units in compilation without affecting the semantics of the generated program. Syntax errors dropped by 75%, and errors related to undefined references dropped by 87% for the tasks where the agents outperformed the baselines. We also observed that in some cases, smaller models with a compiler outperform larger models with a compiler. We conclude that it is essential for LLMs to have access to software engineering tools to enhance their performance and reduce the need for large models in software engineering, such as reducing our energy footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16549v1">LLM is Not All You Need: A Systematic Evaluation of ML vs. Foundation Models for text and image based Medical Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 9 pages, 5 figures, 3 tables, paper accepted in AAIML'26 conference
    </div>
    <details class="paper-abstract">
      The combination of multimodal Vision-Language Models (VLMs) and Large Language Models (LLMs) opens up new possibilities for medical classification. This work offers a rigorous, unified benchmark by using four publicly available datasets covering text and image modalities (binary and multiclass complexity) that contrasts traditional Machine Learning (ML) with contemporary transformer-based techniques. We evaluated three model classes for each task: Classical ML (LR, LightGBM, ResNet-50), Prompt-Based LLMs/VLMs (Gemini 2.5), and Fine-Tuned PEFT Models (LoRA-adapted Gemma3 variants). All experiments used consistent data splits and aligned metrics. According to our results, traditional machine learning (ML) models set a high standard by consistently achieving the best overall performance across most medical categorization tasks. This was especially true for structured text-based datasets, where the classical models performed exceptionally well. In stark contrast, the LoRA-tuned Gemma variants consistently showed the worst performance across all text and image experiments, failing to generalize from the minimal fine-tuning provided. However, the zero-shot LLM/VLM pipelines (Gemini 2.5) had mixed results; they performed poorly on text-based tasks, but demonstrated competitive performance on the multiclass image task, matching the classical ResNet-50 baseline. These results demonstrate that in many medical categorization scenarios, established machine learning models continue to be the most reliable option. The experiment suggests that foundation models are not universally superior and that the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) is highly dependent on the adaptation strategy, as minimal fine-tuning proved detrimental in this study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16540v1">Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16527v1">Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02979v2">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16512v1">SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EACL 2026 camera ready (Main Track)
    </div>
    <details class="paper-abstract">
      With the advent of large language models (LLMs), it has become common practice for users to draft text and utilize LLMs to enhance its quality through paraphrasing. However, this process can sometimes result in the loss or distortion of the original intended meaning. Due to the human-like quality of LLM-generated text, traditional detection methods often fail, particularly when text is paraphrased to closely mimic original content. In response to these challenges, we propose a novel approach named SearchLLM, designed to identify LLM-paraphrased text by leveraging search engine capabilities to locate potential original text sources. By analyzing similarities between the input and regenerated versions of candidate sources, SearchLLM effectively distinguishes LLM-paraphrased content. SearchLLM is designed as a proxy layer, allowing seamless integration with existing detectors to enhance their performance. Experimental results across various LLMs demonstrate that SearchLLM consistently enhances the accuracy of recent detectors in detecting LLM-paraphrased text that closely mimics original content. Furthermore, SearchLLM also helps the detectors prevent paraphrasing attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16508v1">Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 4 pages plus 6 pages of bibliography and appendix
    </div>
    <details class="paper-abstract">
      Single-prompt evaluations dominate current LLM benchmarking, yet they fail to capture the conversational dynamics where real-world harm occurs. In this study, we examined whether conversation length affects response veracity by evaluating LLM performance on the BoolQ dataset under varying length and scaffolding conditions. Our results across three distinct LLMs revealed model-specific vulnerabilities that are invisible under single-turn testing. The length-dependent and scaffold-specific effects we observed demonstrate a fundamental limitation of static evaluations, as deployment-relevant vulnerabilities could only be spotted in a multi-turn conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04740v2">StealthGraph: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16492v1">LLM-based Semantic Search for Conversational Queries in E-commerce</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Conversational user queries are increasingly challenging traditional e-commerce platforms, whose search systems are typically optimized for keyword-based queries. We present an LLM-based semantic search framework that effectively captures user intent from conversational queries by combining domain-specific embeddings with structured filters. To address the challenge of limited labeled data, we generate synthetic data using LLMs to guide the fine-tuning of two models: an embedding model that positions semantically similar products close together in the representation space, and a generative model for converting natural language queries into structured constraints. By combining similarity-based retrieval with constraint-based filtering, our framework achieves strong precision and recall across various settings compared to baseline approaches on a real-world dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16479v1">Doc2AHP: Inferring Structured Multi-Criteria Decision Models via Semantic Trees with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable proficiency in semantic understanding, they often struggle to ensure structural consistency and reasoning reliability in complex decision-making tasks that demand rigorous logic. Although classical decision theories, such as the Analytic Hierarchy Process (AHP), offer systematic rational frameworks, their construction relies heavily on labor-intensive domain expertise, creating an "expert bottleneck" that hinders scalability in general scenarios. To bridge the gap between the generalization capabilities of LLMs and the rigor of decision theory, we propose Doc2AHP, a novel structured inference framework guided by AHP principles. Eliminating the need for extensive annotated data or manual intervention, our approach leverages the structural principles of AHP as constraints to direct the LLM in a constrained search within the unstructured document space, thereby enforcing the logical entailment between parent and child nodes. Furthermore, we introduce a multi-agent weighting mechanism coupled with an adaptive consistency optimization strategy to ensure the numerical consistency of weight allocation. Empirical results demonstrate that Doc2AHP not only empowers non-expert users to construct high-quality decision models from scratch but also significantly outperforms direct generative baselines in both logical completeness and downstream task accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v3">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16458v1">Bridging Expert Reasoning and LLM Detection: A Knowledge-Driven Framework for Malicious Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Open-source ecosystems such as NPM and PyPI are increasingly targeted by supply chain attacks, yet existing detection methods either depend on fragile handcrafted rules or data-driven features that fail to capture evolving attack semantics. We present IntelGuard, a retrieval-augmented generation (RAG) based framework that integrates expert analytical reasoning into automated malicious package detection. IntelGuard constructs a structured knowledge base from over 8,000 threat intelligence reports, linking malicious code snippets with behavioral descriptions and expert reasoning. When analyzing new packages, it retrieves semantically similar malicious examples and applies LLM-guided reasoning to assess whether code behaviors align with intended functionality. Experiments on 4,027 real-world packages show that IntelGuard achieves 99% accuracy and a 0.50% false positive rate, while maintaining 96.5% accuracy on obfuscated code. Deployed on PyPI.org, it discovered 54 previously unreported malicious packages, demonstrating interpretable and robust detection guided by expert knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11960v2">R$^2$PO: Decoupling Training Trajectories from Inference Responses for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning has become a central paradigm for improving LLM reasoning. However, existing methods use a single policy to produce both inference responses and training optimization trajectories. The objective conflict between generating stable inference responses and diverse training trajectories leads to insufficient exploration, which harms reasoning capability. In this paper, to address the problem, we propose R$^2$PO (Residual Rollout Policy Optimization), which introduces a lightweight Residual Rollout-Head atop the policy to decouple training trajectories from inference responses, enabling controlled trajectory diversification during training while keeping inference generation stable. Experiments across multiple benchmarks show that our method consistently outperforms baselines, achieving average accuracy gains of 3.4% on MATH-500 and 1.3% on APPS, while also reducing formatting errors and mitigating length bias for stable optimization. Our code is publicly available at https://github.com/RRPO-ARR/Code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16432v1">iPDB -- Optimizing SQL Queries with ML and LLM Predicates</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Structured Query Language (SQL) has remained the standard query language for databases. SQL is highly optimized for processing structured data laid out in relations. Meanwhile, in the present application development landscape, it is highly desirable to utilize the power of learned models to perform complex tasks. Large language models (LLMs) have been shown to understand and extract information from unstructured textual data. However, SQL as a query language and accompanying relational database systems are either incompatible or inefficient for workloads that require leveraging learned models. This results in complex engineering and multiple data migration operations that move data between the data sources and the model inference platform. In this paper, we present iPDB, a relational system that supports in-database machine learning (ML) and large language model (LLM) inferencing using extended SQL syntax. In iPDB, LLMs and ML calls can function as semantic projects, as predicates to perform semantic selects and semantic joins, or for semantic grouping in group-by clauses. iPDB has a novel relational predict operator and semantic query optimizations that enable users to write and efficiently execute semantic SQL queries, outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16407v1">Jacobian Scopes: token-level causal attributions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 12 pages, 15 figures, under review at ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. By analyzing the linearized relations of final hidden state with respect to inputs, Jacobian Scopes quantify how input tokens influence a model's prediction. We introduce three variants - Semantic, Fisher, and Temperature Scopes - which respectively target sensitivity of specific logits, the full predictive distribution, and model confidence (inverse temperature). Through case studies spanning instruction understanding, translation and in-context learning (ICL), we uncover interesting findings, such as when Jacobian Scopes point to implicit political biases. We believe that our proposed methods also shed light on recently debated mechanisms underlying in-context time-series forecasting. Our code and interactive demonstrations are publicly available at https://github.com/AntonioLiu97/JacobianScopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10004v2">Exploring LLMs for Scientific Information Extraction Using The SciEx Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to the KGML Bridge at AAAI 2026 (non-archival)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17008v2">Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exposes notable limitations, particularly in scenarios requiring long-horizon reasoning. To address these challenges, we investigate more stable and effective advantage estimation strategies, especially for multi-turn settings. We first explore Proximal Policy Optimization (PPO) as an alternative and find it to be more robust than GRPO. To further enhance PPO in multi-turn scenarios, we introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO, both with and without long reasoning components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06753v2">Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The advent of ultra-low-bit LLM models (1/1.58/2-bit), which match the perplexity and end-task performance of their full-precision counterparts using the same model size, is ushering in a new era of LLM inference for resource-constrained environments such as edge devices and AI PCs. While these quantization advances promise models that are more cost-effective in terms of latency, memory, throughput, and energy consumption, the computational efficiency of state-of-the-art (SOTA) inference runtimes (e.g., bitnet.cpp) used to deploy them remains underexplored. In this work, we take a bottom-up approach: we first design and implement 1-bit and 2-bit microkernels optimized for modern CPUs, achieving peak computational efficiency across a variety of CPU platforms. We integrate these microkernels into a state-of-the-art LLM inference framework, namely PyTorch-TPP, and present end-to-end inference results with 2-bit models that outperform the current SOTA runtime bitnet.cpp by up to 2.2x, and deliver up to 7x speedup compared to the 16-bit model inference. We then extend this work to Intel GPUs where we design and implement mixed precision, 2-bit GEMM kernels, and show their performance to be close to optimal. We integrated our optimized Xe2 kernels in the vLLM framework as a quantization plugin and evaluated end-to-end LLM inference results for a range of LLM models and Xe2 GPUs. Depending on the model and platform, we see a 4x - 8x reduction in GEMM time compared to the BF16 case, and we get up to 6.3x speedup in end-to-end latency compared to the BF16 execution. Our optimized runtime advances the state of LLM inference on AI PCs and Intel Xe GPUs, paving the way for efficient deployment of ultra-low-bit LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11242v5">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in child-facing contexts such as education, companionship, creative tools, but their deployment raises safety, privacy, developmental, and security risks. We conduct a systematic literature review of child-LLM interaction risks and organize findings into a structured map that separates (i) parent-reported concerns, (ii) empirically documented harms, and (iii) gaps between perceived and observed risk. Moving beyond descriptive listing, we compare how different evidence streams in surveys, incident reports, youth interaction logs, and governance guidance operationalize "harm," where they conflict, and what mitigations they imply. Based on this synthesis, we propose a protection framework that couples child-specific content safety and developmental sensitivity with security-grade controls for adversarial misuse, including prompt injection and multimodal jailbreak pathways. The framework specifies measurable evaluation targets (e.g., harmful-content avoidance, age-calibrated readability, bias parity checks, prompt-injection robustness, and monitoring transparency) to support developers, educators, and policymakers in assessing and improving child-safe LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17178v1">TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Hardware Trojans (HTs) remain a critical threat because learning-based detectors often overfit to narrow trigger/payload patterns and small, stylized benchmarks. We introduce TrojanGYM, an agentic, LLM-driven framework that automatically curates HT insertions to expose detector blind spots while preserving design correctness. Given high-level HT specifications, a suite of cooperating LLM agents (instantiated with GPT-4, LLaMA-3.3-70B, and Gemini-2.5Pro) proposes and refines RTL modifications that realize diverse triggers and payloads without impacting normal functionality. TrojanGYM implements a feedback-driven benchmark generation loop co-designed with HT detectors, in which constraint-aware syntactic checking and GNN-based HT detectors provide feedback that iteratively refines HT specifications and insertion strategies to better surface detector blind spots. We further propose Robust-GNN4TJ, a new implementation of the GNN4TJ with improved graph extraction, training robustness, and prediction reliability, especially on LLM-generated HT designs. On the most challenging TrojanGYM-generated benchmarks, Robust-GNN4TJ raises HT detection rates from 0% to 60% relative to a prior GNN-based detector. We instantiate TrojanGYM on SRAM, AES-128, and UART designs at RTL level, and show that it systematically produces diverse, functionally correct HTs that reach up to 83.33% evasion rates against modern GNN-based detectors, revealing robustness gaps that are not apparent when these detectors are evaluated solely on existing TrustHub-style benchmarks. Post peer-review, we will release all codes and artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17172v1">Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly capable of generating personalized, persuasive text at scale, raising new questions about bias and fairness in automated communication. This paper presents the first systematic analysis of how LLMs behave when tasked with demographic-conditioned targeted messaging. We introduce a controlled evaluation framework using three leading models -- GPT-4o, Llama-3.3, and Mistral-Large 2.1 -- across two generation settings: Standalone Generation, which isolates intrinsic demographic effects, and Context-Rich Generation, which incorporates thematic and regional context to emulate realistic targeting. We evaluate generated messages along three dimensions: lexical content, language style, and persuasive framing. We instantiate this framework on climate communication and find consistent age- and gender-based asymmetries across models: male- and youth-targeted messages emphasize agency, innovation, and assertiveness, while female- and senior-targeted messages stress warmth, care, and tradition. Contextual prompts systematically amplify these disparities, with persuasion scores significantly higher for messages tailored to younger or male audiences. Our findings demonstrate how demographic stereotypes can surface and intensify in LLM-generated targeted communication, underscoring the need for bias-aware generation pipelines and transparent auditing frameworks that explicitly account for demographic conditioning in socially sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04663v4">Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) remains challenging due to inconsistency, bias, and the absence of transparent decision criteria in automated judging. We present Debate, Deliberate, Decide (D3), a cost-aware, adversarial multi-agent framework that orchestrates structured debate among role-specialized agents (advocates, a judge, and an optional jury) to produce reliable and interpretable evaluations. D3 instantiates two complementary protocols: (1) Multi-Advocate One-Round Evaluation (MORE), which elicits k parallel defenses per answer to amplify signal via diverse advocacy, and (2) Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping, which iteratively refines arguments under an explicit token budget and convergence checks. We develop a probabilistic model of score gaps that (i) characterizes reliability and convergence under iterative debate and (ii) explains the separation gains from parallel advocacy. Under mild assumptions, the posterior distribution of the round-r gap concentrates around the true difference and the probability of mis-ranking vanishes; moreover, aggregating across k advocates provably increases expected score separation. We complement theory with a rigorous experimental suite across MT-Bench, AlignBench, and AUTO-J, showing state-of-the-art agreement with human judgments (accuracy and Cohen's kappa), reduced positional and verbosity biases via anonymization and role diversification, and a favorable cost-accuracy frontier enabled by budgeted stopping. Ablations and qualitative analyses isolate the contributions of debate, aggregation, and anonymity. Together, these results establish D3 as a principled, practical recipe for reliable, interpretable, and cost-aware LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21080v2">LLM Personas as a Substitute for Field Experiments in Method Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Field experiments (A/B tests) are often the most credible benchmark for methods (algorithms) in societal systems, but their cost and latency bottleneck rapid methodological progress. LLM-based persona simulation offers a cheap synthetic alternative, yet it is unclear whether replacing humans with personas preserves the benchmark interface that adaptive methods optimize against. We prove an if-and-only-if characterization: when (i) methods observe only the aggregate outcome (aggregate-only observation) and (ii) evaluation depends only on the submitted artifact and not on the method's identity or provenance (method-blind evaluation), swapping humans for personas is just panel change from the method's point of view, indistinguishable from changing the evaluation population (e.g., New York to Jakarta). Furthermore, we move from validity to usefulness: we define an information-theoretic discriminability of the induced aggregate channel and show that making persona benchmarking as decision-relevant as a field experiment is fundamentally a sample-size question, yielding explicit bounds on the number of independent persona evaluations required to reliably distinguish meaningfully different methods at a chosen resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17133v1">Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to AAAI 2026. 13 pages, 3 figures, 10 tables. Code available at: https://github.com/FujitsuResearch/knexa-fl
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for specialized domains is constrained by a fundamental challenge: the need for diverse, cross-organizational data conflicts with the principles of data privacy and sovereignty. While Federated Learning (FL) provides a framework for collaboration without raw data exchange, its classic centralized form introduces a single point of failure and remains vulnerable to model inversion attacks. Decentralized FL (DFL) mitigates this risk by removing the central aggregator but typically relies on inefficient, random peer-to-peer (P2P) pairings, forming a collaboration graph that is blind to agent heterogeneity and risks negative transfer. This paper introduces KNEXA-FL, a novel framework for orchestrated decentralization that resolves this trade-off. KNEXA-FL employs a non-aggregating Central Profiler/Matchmaker (CPM) that formulates P2P collaboration as a contextual bandit problem, using a LinUCB algorithm on abstract agent profiles to learn an optimal matchmaking policy. It orchestrates direct knowledge exchange between heterogeneous, PEFT-based LLM agents via secure distillation, without ever accessing the models themselves. Our comprehensive experiments on a challenging code generation task show that KNEXA-FL yields substantial gains, improving Pass@1 by approx. 50% relative to random P2P collaboration. Critically, our orchestrated approach demonstrates stable convergence, in stark contrast to a powerful centralized distillation baseline which suffers from catastrophic performance collapse. Our work establishes adaptive, learning-based orchestration as a foundational principle for building robust and effective decentralized AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08741v2">Coordinates from Context: Using LLMs to Ground Complex Location References</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EACL 2026
    </div>
    <details class="paper-abstract">
      Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17087v1">Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on τ-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16206v1">LLM-in-Sandbox Elicits General Agentic Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Project Page: https://llm-in-sandbox.github.io
    </div>
    <details class="paper-abstract">
      We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10637v2">LLMs Homogenize Values in Constructive Arguments on Value-Laden Topics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to promote prosocial and constructive discourse online. Yet little is known about how these models negotiate and shape underlying values when reframing people's arguments on value-laden topics. We conducted experiments with 465 participants from India and the United States, who wrote comments on homophobic and Islamophobic threads, and reviewed human-written and LLM-rewritten constructive versions of these comments. Our analysis shows that LLM systematically diminishes Conservative values while elevating prosocial values such as Benevolence and Universalism. When these comments were read by others, participants opposing same-sex marriage or Islam found human-written comments more aligned with their values, whereas those supportive of these communities found LLM-rewritten versions more aligned with their values. These findings suggest that value homogenization in LLM-mediated prosocial discourse runs the risk of marginalizing conservative viewpoints on value-laden topics and may inadvertently shape the dynamics of online discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16134v1">LLM Prompt Evaluation for Educational Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly common in educational applications, there is a growing need for evidence-based methods to design and evaluate LLM prompts that produce personalized and pedagogically aligned out-puts. This study presents a generalizable, systematic approach for evaluating prompts, demonstrated through an analysis of LLM-generated follow-up questions in a structured dialogue activity. Six prompt templates were designed and tested. The templates incorporated established prompt engineering patterns, with each prompt emphasizing distinct pedagogical strategies. The prompt templates were compared through a tournament-style evaluation framework that can be adapted for other educational applications. The tournament employed the Glicko2 rating system with eight judges evaluating question pairs across three dimensions: format, dialogue support, and appropriateness for learners. Data was sourced from 120 authentic user interactions across three distinct educational deployments. Results showed that a single prompt related to strategic reading out-performed other templates with win probabilities ranging from 81% to 100% in pairwise comparisons. This prompt combined persona and context manager pat-terns and was designed to support metacognitive learning strategies such as self-directed learning. The methodology showcases how educational technology re- searchers can systematically evaluate and improve prompt designs, moving beyond ad-hoc prompt engineering toward evidence-based prompt development for educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16130v1">Replicating Human Motivated Reasoning Studies with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Motivated reasoning -- the idea that individuals processing information may be motivated to reach a certain conclusion, whether it be accurate or predetermined -- has been well-explored as a human phenomenon. However, it is unclear whether base LLMs mimic these motivational changes. Replicating 4 prior political motivated reasoning studies, we find that base LLM behavior does not align with expected human behavior. Furthermore, base LLM behavior across models shares some similarities, such as smaller standard deviations and inaccurate argument strength assessments. We emphasize the importance of these findings for researchers using LLMs to automate tasks such as survey data collection and argument assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v3">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v2">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15209v2">Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted for publication in ACM CHI 2026
    </div>
    <details class="paper-abstract">
      We investigate intelligent personal assistants (IPAs) accessibility for deaf and hard of hearing (DHH) people who can use their voice in everyday communication. The inability of IPAs to understand diverse accents including deaf speech renders them largely inaccessible to non-signing and speaking DHH individuals. Using an Echo Show, we compare the usability of natural language input via spoken English; with Alexa's automatic speech recognition and a Wizard-of-Oz setting with a trained facilitator re-speaking commands against that of a large language model (LLM)-assisted touch interface in a mixed-methods study. The touch method was navigated through an LLM-powered "task prompter," which integrated the user's history and smart environment to suggest contextually-appropriate commands. Quantitative results showed no significant differences across both spoken English conditions vs LLM-assisted touch. Qualitative results showed variability in opinions on the usability of each method. Ultimately, it will be necessary to have robust deaf-accented speech recognized natively by IPAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v2">TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures reasoning models not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly release TruthTensor at https://truthtensor.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16034v1">Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs (including GPT-OSS-20B and GLM-4) confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06299v4">How malicious AI swarms can threaten democracy: The fusion of agentic AI and LLMs marks a new frontier in information warfare</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 5 Pages, This is the author's version of the work. It is posted here by permission of the AAAS for personal use, not for redistribution. The definitive version was published in Science on January 22, 2026, DOI: 10.1126/science.adz1697
    </div>
    <details class="paper-abstract">
      Advances in AI offer the prospect of manipulating beliefs and behaviors on a population-wide level. Large language models and autonomous agents now let influence campaigns reach unprecedented scale and precision. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, a disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus efficiently. By adaptively mimicking human social dynamics, they threaten democracy. Because the resulting harms stem from design, commercial incentives, and governance, we prioritize interventions at multiple leverage points, focusing on pragmatic mechanisms over voluntary compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16027v1">Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      The rise of live streaming has transformed online interaction, enabling massive real-time engagement but also exposing platforms to complex risks such as scams and coordinated malicious behaviors. Detecting these risks is challenging because harmful actions often accumulate gradually and recur across seemingly unrelated streams. To address this, we propose CS-VAR (Cross-Session Evidence-Aware Retrieval-Augmented Detector) for live streaming risk assessment. In CS-VAR, a lightweight, domain-specific model performs fast session-level risk inference, guided during training by a Large Language Model (LLM) that reasons over retrieved cross-session behavioral evidence and transfers its local-to-global insights to the small model. This design enables the small model to recognize recurring patterns across streams, perform structured risk assessment, and maintain efficiency for real-time deployment. Extensive offline experiments on large-scale industrial datasets, combined with online validation, demonstrate the state-of-the-art performance of CS-VAR. Furthermore, CS-VAR provides interpretable, localized signals that effectively empower real-world moderation for live streaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16023v1">Timbre-Aware LLM-based Direct Speech-to-Speech Translation Extendable to Multiple Language Pairs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Direct Speech-to-Speech Translation (S2ST) has gained increasing attention for its ability to translate speech from one language to another, while reducing error propagation and latency inherent in traditional cascaded pipelines. However, existing direct S2ST systems continue to face notable challenges, including instability in semantic-acoustic alignment when parallel speech data is scarce, difficulty in preserving speaker identity, and limited multilingual scalability. In this work, we introduce DS2ST-LM, a scalable, single-stage direct S2ST framework leveraging a multilingual Large Language Model (LLM). The architecture integrates a Whisper speech encoder, a learnable projection module, a Qwen2-0.5B LLM, and a timbre-controlled vocoder. We construct GigaS2S-1000, a 1000-hour bilingual corpus by extending the GigaST dataset with high-fidelity synthetic target speech, and show that this synthetic data alleviates data scarcity to some extent. We investigate two semantic token generation strategies: speech-derived S3 tokens and text-derived tokens generated by a pre-trained LLM, and analyze their impact on training stability and semantic consistency. We further evaluate three projection architectures (Linear, Conv1D-Linear, and Q-Former) and observe that while higher-capacity projectors converge faster, the simple Linear projector achieves higher performance. Extensive experiments demonstrate that DS2ST-LM outperforms traditional cascaded and ST (Qwen-Audio) + TTS baselines across both lexical (BLEU, METEOR) and semantic (BLEURT, COMET) metrics, while extending to multiple language pairs, including French, Spanish, German, Hindi, Bengali, and Urdu. Furthermore, we incorporate timbre-aware speech synthesis to preserve speaker information, enabling DS2ST-LM to surpass prior direct S2ST systems in both speaker similarity and perceptual naturalness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.06518v3">Medal Matters: Probing LLMs' Failure Cases Through Olympic Rankings</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 COLM 2025 ORIGen Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, yet their internal knowledge structures remain poorly understood. This study examines these structures through the lens of historical Olympic medal tallies, evaluating LLMs on two tasks: (1) retrieving medal counts for specific teams and (2) identifying rankings of each team. While state-of-the-art LLMs excel in recalling medal counts, they struggle with providing rankings, highlighting a key difference between their knowledge organization and human reasoning. These findings shed light on the limitations of LLMs' internal knowledge integration and suggest directions for improvement. To facilitate further research, we release our code, dataset, and model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12061v2">Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Under Review for ACL 2026
    </div>
    <details class="paper-abstract">
      Dialogue Act (DA) annotation typically treats communicative or pedagogical intent as localized to individual utterances or turns. This leads annotators to agree on the underlying action while disagreeing on segment boundaries, reducing apparent reliability. We propose codebook-injected segmentation, which conditions boundary decisions on downstream annotation criteria, and evaluate LLM-based segmenters against standard and retrieval-augmented baselines. To assess these without gold labels, we introduce evaluation metrics for span consistency, distinctiveness, and human-AI distributional agreement. We found DA-awareness produces segments that are internally more consistent than text-only baselines. While LLMs excel at creating construct-consistent spans, coherence-based baselines remain superior at detecting global shifts in dialogue flow. Across two datasets, no single segmenter dominates. Improvements in within-segment coherence frequently trade off against boundary distinctiveness and human-AI distributional agreement. These results highlight segmentation as a consequential design choice that should be optimized for downstream objectives rather than a single performance score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.03592v4">English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 8 pages, 6 figures, v2
    </div>
    <details class="paper-abstract">
      For consumer usage of locally deployed LLMs, the GGUF format and k\_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to yielded non-significant results indicating that current quantization practices do not disproportionately harm multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06094v3">ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Project page: https://conlangcrafter.github.io
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages -- phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We construct a novel, scalable evaluation framework for this task, evaluating metrics measuring consistency and typological diversity. Automatic and manual evaluations demonstrate ConlangCrafter's ability to produce coherent and varied conlangs without human linguistic expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16602v2">Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12365v3">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10978v2">Does LLM Focus on the Right Words? Mitigating Context Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Context Bias, whereby the model over-relies on auxiliary tokens, such as task descriptions and prefix-generated tokens, while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating context bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates context bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. The code is available at https://github.com/WANGBohaO-jpg/GDRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15879v1">Evaluating and Achieving Controllable Code Completion in Code LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Code completion has become a central task, gaining significant attention with the rise of large language model (LLM)-based tools in software engineering. Although recent advances have greatly improved LLMs' code completion abilities, evaluation methods have not advanced equally. Most current benchmarks focus solely on functional correctness of code completions based on given context, overlooking models' ability to follow user instructions during completion-a common scenario in LLM-assisted programming. To address this limitation, we present the first instruction-guided code completion benchmark, Controllable Code Completion Benchmark (C3-Bench), comprising 2,195 carefully designed completion tasks. Through comprehensive evaluation of over 40 mainstream LLMs across C3-Bench and conventional benchmarks, we reveal substantial gaps in instruction-following capabilities between open-source and advanced proprietary models during code completion tasks. Moreover, we develop a straightforward data synthesis pipeline that leverages Qwen2.5-Coder to generate high-quality instruction-completion pairs for supervised fine-tuning (SFT). The resulting model, Qwen2.5-Coder-C3, achieves state-of-the-art performance on C3-Bench. Our findings provide valuable insights for enhancing LLMs' code completion and instruction-following capabilities, establishing new directions for future research in code LLMs. To facilitate reproducibility and foster further research in code LLMs, we open-source all code, datasets, and models.
    </details>
</div>
