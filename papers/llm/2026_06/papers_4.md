# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17510v2">OmniDroneX: An LLM-Assisted Holistic Drone-as-a-Service Ecosystem</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 This manuscript is a full version of a paper accepted in shortened form by IEEE International Conference on Joint Cloud Computing
    </div>
    <details class="paper-abstract">
      Despite rapid advances in UAV technologies, current deployments remain limited due to several gaps in UAV systems research. To address these challenges, we propose OmniDroneX, a unified Drone-as-a-Service ecosystem, in which drones are transitioned from fixed function platforms into dynamically composable entities that can be integrated with external infrastructures to offer omni-capabilities. OmniDroneX bridges low-level physical primitives with high-level mission intent through a unified vendor-agnostic interface (libUAV) and a formal physical-service abstraction model (PT-SOA). A core innovation is the diverse application of large language models (LLMs) across multiple layers of the OmniDroneX architecture. LLMs are used to assist in identifying and formalizing primitive device functions and abstract service definitions, supporting automated service composition and workflow generation, and enabling interactive, natural-language mission specification and refinement. OmniDroneX also incorporates important categories of composition techniques that are essential in dynamic UAV systems, including physical layer composition for drone capability augmentation, as well as spatiotemporal, functional, collaborative, exception-aware, and QoS-based service compositions. Collectively, these features allow OmniDroneX to serve as a foundation for scalable, resilient, and self-evolving UAV ecosystems operating in complex and dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26557v3">MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 ICML MemFM 2026 Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18673v1">Understanding and Mitigating Prompt Leaking Attacks in Real-World LLM-Based Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Accepted at ACM CCS 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based applications rely on system prompts to encode core logic and developer-defined constraints, making these prompts important intellectual property. However, system prompts are vulnerable to prompt leaking attacks. Although prior work has shown such attacks in controlled settings, their prevalence, causes, and defenses in real-world deployments remain unclear. This paper presents a systematic study of prompt leaking in real-world LLM-based applications. We measure 1,200 applications across six major commercial platforms and find that over 80% of deployments leak system prompts under realistic adversarial queries, sometimes exposing sensitive information such as third-party API keys. We also show that existing defenses often fail to prevent leakage without degrading usability. To explain these failures, we conduct an attention-level mechanistic analysis and identify attention drift, where query-key alignment bias and softmax amplification cause LLMs to progressively ignore defensive constraints. Guided by this insight, we propose AREA, a practical defense that re-anchors the model's attention using an optimizable soft prompt. Experiments and real-world case studies show that AREA matches the leakage resistance of state-of-the-art defenses while improving average usability by over 33% and reducing optimization overhead by nearly 3x. Our responsible disclosure led two affected vendors to classify these leaks as medium-severity vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13681v2">EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align their knowledge, skills, and behavior with changing environments and updated task conditions. To address this gap, we introduce EvoArena, a benchmark suite that models environment changes as sequences of progressive updates across terminal, software, and social domains. We further propose EvoMem, a patch-based memory paradigm that records memory evolution as structured update histories, enabling agents to reason about environmental evolution through changes in their memory. Experiments show that current agents struggle on EvoArena, achieving an average accuracy of 39.6% across evolving terminal, software, and social-preference domains. EvoMem consistently improves performance, yielding an average gain of 1.5% on EvoArena and also improving standard benchmarks such as GAIA and LoCoMo by 6.1% and 4.8%. Beyond individual tasks, EvoMem further improves chain-level accuracy by 3.7% on EvoArena, where success requires completing a consecutive sequence of related evolutionary subtasks. Mechanistic analysis shows that EvoMem improves evidence capture in the memory, indicating better preservation of complete evolving environment states. Our results highlight the importance of modeling evolution in both evaluation and memory for reliable agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18656v1">The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Warning: This paper studies stereotypes and biases, and contains potentially disturbing examples, used for illustration purposes only. Our findings should not be interpreted as an argument against alignment. Instead, this paper highlights the need for principled approaches to more advanced alignment. Alignment aims to ensure that large language models (LLMs) behave safely and reliably, including by avoiding unsafe inferences. However, we show that such safety-oriented behaviors can misfire: models may reject warranted conclusions even when they are explicitly supported by context. We call this failure mode misfired alignment, where alignment-induced changes cause LLMs to override explicit evidence. To quantify this phenomenon, specifically on stereotype-related alignment, we introduce VETO, a benchmark consisting of 2,032 BBQ-derived contrastive pairs, and define a new metric, Misfired Alignment Rate (MAR), which measures on a 0 to 100 scale how often a model fails on a stereotype-related question but succeeds on its contrastive counterpart. We benchmark 25 LLMs on VETO, and show that all LLMs, including the most recent ones, exhibit non-trivial (4.7 to 18.9%) MARs while all human participants achieve 0.0% MAR. Controlled priming experiments further show that alignment-induced cues can substantially amplify MAR across LLMs, indicating that these failures are not merely artifacts of individual examples but can be induced by safety-related framing. Mechanistic analyses on open-weight LLMs reveal late-layer suppression of evidence-supported answers, and comparisons between instruct and base LLMs suggest that this suppression emerges after instruction training. These findings show that current alignment methods can overgeneralize surface-level safety cues, to the point of overriding objective evidence, motivating more work on alignment objectives that better preserve contextual grounding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18519v1">As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agriculture</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Though robotic systems are now being commercialized and deployed in various industries, many of these systems are highly specialized and often require an advanced skill set to operate and ensure they perform as instructed. To mitigate this problem, we recently introduced a mission planner leveraging LLMs to synthesize mission plans in precision agriculture based on mission descriptions provided in natural language. While the system demonstrates impressive performance, it also suffers from the inherent ambiguities of natural language. In this paper, we extend our system to address this issue by introducing multiple feedback loops in the planning architecture that leverage linear temporal logic (LTL) to ensure the mission planning system meets the specifications formulated by the user while still using natural language. To mitigate potential bias, this is achieved by using two different commercial LLMs in charge of the specification and verification subtasks. Through extensive experiments, we highlight the strengths and limitations of integrating mission verification into a fully autonomous pipeline, particularly regarding an LLM's ability to generate valuable LTL formulas, and show how our proposed implementation addresses and solves these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09905v2">The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 19 pages 5 figures
    </div>
    <details class="paper-abstract">
      When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human-validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user-independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion reasoning and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models' emotional reasoning. These results highlight a key challenge for memory-enhanced AI: systems designed for personalization may reinforce social inequalities. To mitigate these disparities, we curate a general-purpose preference dataset designed to reduce demographic profiles' influence on emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18453v1">LLM Parameters for Math Across Languages: Shared or Separate?</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 5 pages. Accepted at ACL Student Research Workshop (SRW) 2026. Code: https://github.com/luisavictor/math-across-languages Translated Datasets: https://huggingface.co/math-across-languages Webpage: https://math-across-languages.github.io
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit substantial cross-lingual variation in mathematical reasoning performance, but it remains unclear whether these differences reflect language-specific parameters or a shared mechanism that manifests differently by language. We present a cross-lingual mechanistic analysis of mathematical reasoning in LLMs, enabling us to localize and compare model parameters that support mathematical reasoning across languages. We find that the extracted math-associated parameters exhibit partial cross-lingual overlap, with the strongest overlap concentrated in intermediate model layers. We further observe that English consistently produces the largest set of math-relevant parameters, whereas lower-resource languages reveal smaller sets of relevant parameters. These results suggest that math-related behavior in multilingual LLMs is neither fully language-invariant nor fully language-specific, but instead exhibits partial cross-lingual parameter overlap with systematic language-dependent differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18431v1">Beyond Prediction: Tail-Aware Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM serving exhibits extreme length variability, making size-based scheduling difficult in practice. Recent LLM schedulers approximate SJF/SRPT using predicted decode lengths or ranks and primarily report mean-centric metrics such as TTFT and TBT. We show that these prediction-driven policies can be fragile under distribution shifts, bursty arrivals, and GPU memory pressure, while offering limited control over the tail latency (P90-P99) that dominates user experience, even with perfect decode-length knowledge. We introduce a distribution-aware, prediction-free scheduling framework that replaces explicit length prediction with soft priority boosting driven by lightweight statistical signals. Our design co-optimizes scheduling and cache-aware preemption to account for memory-coupled decode dynamics across workload mixes. Evaluated on production and open-source traces, our method reduces P99 TTLT by up to 35-50% relative to SRPT with perfect length knowledge and reduces TTFT by 34-47% across workloads, including reasoning-heavy and chat-heavy tasks. These results demonstrate a robust alternative for optimizing tail latency in online LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18405v1">Evaluating the Effectiveness of LLMs in Aiding Compliance Testing of PKCS#1-v1.5</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Testing implementations of binary protocols for specification compliance requires inputs that satisfy both structural and semantic constraints. Purely random generation and primitive mutations are often insufficient for exploring semantically meaningful behaviors in protocols that rely on Type-Length-Value (TLV) encoding, yet domain-specific compliance testing tools require deep protocol expertise and significant manual effort to construct. This work investigates whether grammar-level mutation combined with LLM-based code synthesis can serve as a viable, more generalizable approach to specification compliance testing. We evaluate the approach on PKCS#1 v1.5 signature verification -- a widely deployed TLV-encoded standard with a formally verified testing oracle (Morpheus) -- across 48 cryptographic library implementations. We reproduced 10 of 13 non-trivial specification violation categories previously identified by Morpheus, including all 5 signature forgery categories, and discovered 1 previously unreported discrepancy. We found that LLM hallucination -- occurring in 82.5% of generated scripts -- is the primary factor limiting effectiveness, not the mutation strategies. We identify five distinct hallucination types and show that their distribution varies systematically across mutation categories: structural mutations are implemented with 13.3% fidelity while constraint mutations achieve 30.3% correctness but suffer the highest rate of mutations being fully ignored (8.1%). These findings reveal a striking gap between operational reliability (99.8%) and semantic fidelity (17.5%), providing actionable guidance on when LLM-based code synthesis can be trusted in specification-driven testing pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18388v1">LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction matters because fixed schedules commit all parameters to fixed trajectories and therefore cannot express the non-stationary exploration-exploitation tradeoffs that regularization must track; the principle provides actionable design rules for multi-stage training. We discover this through LLMZero, a system where LLM agents search over training trajectories via tree search, diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across 4 diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% relative and over grid search by 6% to 15% relative, consistently outperforming random search and the skill-based agent. The structural principle transfers across tasks, providing an explanation for why discovered strategies take qualitatively different forms yet share similar parameter dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.25065v2">Vulcan: Instance-specialized, Verifiable Systems Heuristics Through LLM-driven Search</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Systems resource management tasks rely primarily on hand-designed heuristics. However, growing hardware heterogeneity and workload diversity require heuristics specialized to particular deployment instances, making manual design expensive and difficult to scale. In this paper, we explore how to synthesize systems heuristics using LLMs. The main challenge is ensuring that generated heuristics execute safely, integrate correctly with the surrounding system, and still achieve strong performance. We propose Vulcan, a framework that identifies LLM-friendly interfaces that isolate core decision logic from the rest of the implementation. With Vulcan, LLM-generated code is restricted to simple stateless decision functions, while trusted runtime abstractions provide rich derived statistics for meaningful policy exploration without system-integration bugs. To ensure execution safety, LLMs synthesize heuristics in a restricted language, Anvil, that guarantees important properties by construction. We evaluate Vulcan across three well-studied domains and demonstrate up to 4.9x higher savings for spot-VM scheduling, up to 2x lower miss ratios for cache eviction, and up to 10% higher application performance for tiered-memory systems, while ensuring execution safety throughout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18166v1">Evaluating Open-Source LLMs for Multi-Label ATT&CK Technique Classification on CTI Reports</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Classifying Cyber Threat Intelligence (CTI) using MITRE Adversarial Tactics, Techniques, and Common Knowledge (ATT&CK) is essential for proactive defense, but historically required extensive human effort. Pre-Large Language Model (LLM) automation sped up this process, but could not resolve the complex language and multi-step attack patterns found in unstructured CTI reports. LLMs addressed previous limitations by using contextual reasoning to understand unstructured text. However, current evaluations rely on simplified, single-technique sentences that ignore the complexity of real-world CTI reports, which often leads to inflated performance results. Consequently, the baseline performance of open-source LLMs on complex unstructured CTI reports remains unevaluated. To address this gap, we constructed a ground-truth dataset of 2,076 human-annotated sentences (1,281 technique-positive, 795 negative) from 83 complex unstructured CTI reports. These sentences were mapped to 114 unique ATT&CK techniques using a six-phase annotation process, achieving \k{appa} = 0.68 inter-annotator agreement. Using this dataset, we evaluated seven open-source LLMs ranging from 8B to 236B parameters across prompt strategy and temperature configurations. The highest-performing LLM achieved a micro-averaged F1 score of 0.22, establishing the empirical baseline for multi-label ATT&CK classification on complex unstructured CTI. Parameter size showed a statistically significant positive correlation with F1 score. Prompt strategy and temperature produced no statistically significant gains across model configurations. These results indicate that current open-source LLMs are insufficient for production-grade ATT&CK classification. The dataset, benchmark, and findings provide a reproducible foundation for future CTI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.00826v3">LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Cryptocurrency portfolio management requires the fusion of heterogeneous multi-modal signals, including structured price and on-chain time series, unstructured news text, and technical indicators, under high-volatility and real-time constraints. While deep learning approaches show predictive capability, their opacity limits practical adoption, and single large language model (LLM) agents struggle to process the breadth of modality-specific inputs needed for robust decision-making. We propose a multi-agent system (MAS) framework in which three modality-specialised agents, a Crypto Agent for market dynamics, a News Agent for weekly news sentiment, and a Trading Agent for signal fusion and portfolio execution, decompose the task across three communication architectures: hierarchical, collaborative, and debate. We evaluate four capability configurations: zero-shot, chain-of-thought (CoT), retrieval-augmented generation (RAG), and skill-augmented. In a 52-week backtest over calendar year 2025 across the top 15 L1 blockchain native cryptocurrencies by market capitalisation as of January 2025, the best configuration, Hierarchical (Skill), achieves a cumulative return of 133.52% and a Sharpe ratio of 1.502, outperforming single-agent variants, passive benchmarks, and deep learning baselines. An ablation study identifies the Crypto Agent as the most critical component, with its removal reducing cumulative return by 42.57 percentage points. A cross-model comparison further shows that MAS outperforms the single-agent baseline under GPT-4o, GPT-5, and Claude Sonnet 4.5, suggesting that the benefit of multi-agent coordination is model-agnostic. Unlike black-box deep learning models, every portfolio decision is traceable to explicit agent reasoning, offering an interpretable and effective approach to multi-modal cryptocurrency portfolio management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18129v1">Towards Understanding and Measuring COGNITIVE ATROPHY in LLM Behaviour</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Recent incidents involving LLMs used for mental-health support reveal a critical evaluation gap: surface-level safety scores do not capture how models behave across realistic, emotionally sensitive interactions over time. Existing benchmarks measure knowledge, safety, or static response quality, but miss whether LLM interactions help users keep reflecting, coping, and making decisions themselves. We formalize this missing dimension as COGNITIVE ATROPHY, a process-level behavioural measure in AI-mediated mental-health support distinct from safety and helpfulness. To measure it, we introduce COGNITIVE ATROPHY BENCH, a clinically grounded benchmark built from 1,576 fully human-generated counseling conversations, 15,680 turns, and 42,230 responses from five LLMs. Three clinical and neuropsychology experts developed a 20-attribute schema spanning user context, response behaviour, and global risk flags; six trained clinical reviewers applied it with span-grounded evidence, producing 5,324 reviewer judgments. We further introduce the User-Input Risk Index (UIRI), the Cognitive Atrophy Risk Index (ARI), and trajectory summaries. Across five LLMs, models show a consistent moderate-to-high level of atrophy-aligned behaviour across single and multi-turn settings. While models generally respond to overt safety cues, they adapt less reliably when users seek solutions or decisions. The dominant recurring patterns are directive advice, problem-solving, recommendation responses, topic shifts, and forms of validation that may reinforce dependence rather than reflection. Our work makes COGNITIVE ATROPHY measurable and provides a foundation for auditing model behaviour in sensitive LLM conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18063v1">When LLMs Analyze Scars: From Images to Clinically-Meaningful Features</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Medical image classification faces a fundamental dilemma: while deep learning models achieve remarkable performance at scale, real-world clinical scenarios often suffer from severe data scarcity due to annotation costs, privacy constraints, and disease rarity. This challenge is particularly pronounced in pathological scar classification, where differentiating keloids from hypertrophic scars requires subtle expert knowledge and labeled images are extremely limited. We propose a novel paradigm that repositions large language models (LLMs) as knowledge-driven feature engineers rather than end-to-end classifiers. We call this framework ScaFE (Scar Feature Engineering). Our key insight is that LLMs encode rich medical knowledge that can be externalized as executable feature extraction code, enabling the transformation of high-dimensional images into low-dimensional, clinically interpretable representations. Specifically, we prompt an LLM with established scar assessment criteria to generate deterministic Python code that extracts features aligned with clinical scoring systems such as the Vancouver Scar Scale. Our approach offers three key advantages: (1) data efficiency, achieving robust performance with limited training samples by decoupling knowledge acquisition from statistical learning; (2) privacy preservation, as raw images are processed locally without exposure to external LLMs; and (3) interpretability, through explicit features grounded in clinical reasoning. Extensive experiments on scar classification demonstrate that our method consistently outperforms end-to-end deep learning baselines or using LLMs as black-box classifiers under limited data conditions, establishing a promising direction for integrating LLMs into data-efficient and clinically transparent medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18062v1">Security and Privacy Prompts in the Wild: What Users Ask LLMs and How LLMs Respond</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used to fulfill users' information needs; users ask LLMs about the weather, pose educational questions, and consult them for legal assistance. One particularly understudied area is digital security and privacy (S&P), where users may seek LLMs' help on how to secure their online accounts or protect their computers from cyber attacks. To the best of our knowledge, no prior study has collected or analyzed the S&P questions users ask LLMs; prior research on LLM response quality relied on expert-authored S&P misconceptions or FAQs rather than user queries. Drawing from WildChat, a dataset of 3.2M user-LLM conversations collected in the wild, our study identifies 14,727 S&P prompts and categorizes them into nine categories covering a wide range of S&P topics. From the S&P prompts, we sampled 450 and performed a thematic analysis to characterize the S&P questions users ask LLMs. Separate from the thematic analysis, we curated 270 advice-seeking S&P prompts, where users ask for recommendations, guidance, or specific S&P information. We measured LLM response quality and consistency when posing the prompt to LLMs 10 times. We found that commercial LLMs outperform open-weight models (GPT 5.5 provided "good enough" responses on 98% of prompts; Llama 4 on 47%). However, among prompts that received high-quality responses on average, commercial models sometimes produce contradictory responses across runs, risking confusing or misleading users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18051v1">Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compose</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on external skills -- reusable tool specifications -- but real-world tasks often require composing multiple skills, not just selecting one. We formalize this as the Compositional Skill Routing problem: given a complex user query and a large skill library, decompose the query into atomic sub-tasks, retrieve the appropriate skill for each sub-task, and compose an executable plan. We present SkillWeaver, a decompose-retrieve-compose framework combining an LLM task decomposer, a bi-encoder skill retriever with FAISS indexing, and a dependency-aware DAG planner. To support evaluation, we introduce CompSkillBench, a benchmark of 300 compositional queries over 2,209 real MCP server skills spanning 24 functional categories, sourced from the public MCP ecosystem. Our experiments reveal that task decomposition quality is the primary bottleneck: standard LLM decomposition reaches only 34.2% category recall at the step level. To address this, we propose Iterative Skill-Aware Decomposition (SAD), a retrieval-augmented feedback loop that iteratively aligns decomposition with available skills. SAD improves decomposition accuracy from 51.0% to 67.7% (+32.7%, Wilcoxon p < 10^-6) in a single iteration; DA-conditioned analysis confirms that correct granularity is the prerequisite for effective retrieval (CatR@1 rises from 34% to 41% when DA=1). SkillWeaver reduces context window consumption by over 99%, and transfer experiments confirm generalization (+35.6% relative DA gain even when target categories are absent from the retrieval pool).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23243v3">Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      We evaluate whether frontier LLMs are ready for cybersecurity through a dual-mode benchmark: white-box function-level vulnerability detection (VulnLLM-R, across C/Java/Python) and black-box web application security testing (five production-style applications with 118 ground-truth vulnerabilities across 20+ CWE families, which we will open-source). We test six frontier models (GPT-5.4, Codex~5.3, Claude Opus~4.7, Sonnet~4.6, Gemini~3.1~Pro and Gemini~3~Flash) and two domain-specialized models across four testing paradigms. Our findings are sobering: (1)~every frontier model produces 10-50% false positive rates in white-box detection, systematically over-predicting vulnerabilities; (2)~in black-box testing, frontier models achieve only 4-8% ground-truth coverage, improving to just 10-19% even with external security tools (Playwright MCP, Burp Suite MCP); (3)~structured penetration-testing methodology encoded in domain-specialized agents raises per-family detection above 50%, demonstrating that methodology, not scale, is the primary lever; and (4)~a domain-specialized defense model achieves the highest precision (0.904) and lowest false positive rate (9.7%) among all models, on a single GPU. We identify the absence of structured security testing traces end-to-end request/response sequences, failure-heavy data, and multi-step attack chains as the fundamental training data bottleneck, and propose self-play security testing as a data generation strategy. Our results make the case for vertical foundation models purpose-built for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18037v1">ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents increasingly use the Model Context Protocol (MCP) to answer from heterogeneous evidence sources, including search, APIs, databases, clinical records, and formulary tools. Standard factuality metrics usually test whether an answer is supported by pooled evidence, missing a provenance-sensitive failure mode: a claim may be supported somewhere while being attributed to the wrong source. We call this cross-source conflation. We introduce ProvenanceGuard, a source-aware verifier for MCP-grounded answers. It consumes captured MCP traces with stable tool IDs, source IDs, and raw outputs; decomposes answers into atomic claims; routes claims to source-specific evidence; checks support with NLI and a token-alignment proxy; compares stated attribution with the routed source; and returns per-claim verdicts plus an answer-level allow/block decision. Blocked answers can be repaired with retrieval-augmented answer revision and re-verified. We evaluate on 281 medical-domain MCP-agent traces. A 266-trace adjudicated subset yields 2,325 LLM-assisted claim labels split by trace; 361 held-out labels are human-verified. On the 40-trace held-out split, ProvenanceGuard achieves block F1 0.802 and source accuracy 0.858 over 260 source-eligible claims, outperforming source-blind baselines that do not emit claim-to-source IDs. On a harder multi-source benchmark it reaches block F1 0.846, while source-plus-relation accuracy drops to 0.229, showing that exact source ownership remains difficult with semantically close sources. Repair-and-reverify resolves all blocked answers in the full trace set, often via conservative fallback. In 50 controlled clinical conflation probes, ProvenanceGuard detects all injected attribution swaps with no retained wrong attribution. These results show that source attribution is an independent axis for factuality verification in MCP-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18030v1">ParaTutor: LLM Mediated Parent Child Tutoring through Role Separated Scaffolding Interface in Real Time</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Parent child tutoring is a collaborative learning setting with asymmetric roles, where parents guide children s problem solving while children engage in understanding and reasoning. However, most LLM based learning systems are designed for either single users or symmetric collaboration, leaving parent child tutoring with distinct instructional roles underexplored. Through a formative study, we find that effective parent child tutoring depends on preserving these distinct roles, with parents guiding the learning process and children remaining actively engaged in reasoning. We also identify recurring challenges when parents struggle to understand problem structure, lack sufficient knowledge to provide support, or encounter communication difficulties that disrupt shared understanding. To address these challenges, we present ParaTutor, a scaffolding system that provides different forms of support to parents and children. ParaTutor supports parents with guidance for tutoring and provides children with visual grounding for problem solving. We evaluate ParaTutor with 23 parent child dyads (children aged 10 to 12) under four tutoring conditions that vary how LLM assistance is delivered. Results show that generic LLM assistance tends to reduce the parent s role in tutoring, whereas ParaTutor better preserves parent led support and sustains children s participation in reasoning. These findings suggest that in multi users learning, the value of LLM support depends not only on model capability but also on how support is distributed across users with different roles. Our work contributes design implications for LLM systems that support family learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18005v1">LLM Consumer Behavior Theory: Foundations of a Novel Research Field</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous agents that make consumption decisions on behalf of users. This shift raises fundamental questions for consumer theory, which has traditionally modeled humans as the primary decision-makers. In this paper, we introduce LLM Consumer Behavior Theory, a new field of study concerned with analyzing consumer behavior in agentic markets. Drawing on classical and behavioral economics alongside recent advances in Natural Language Processing, we formalize how human preferences are reflected and acted upon by LLM-based agents, and how agent-level decisions aggregate into market demand. We unify previously fragmented literature on LLM decision-making, human behavior simulation, and preference elicitation under a common economic lens, highlighting where assumptions, such as rationality and heterogeneity, may fail in agentic markets. Rather than providing empirical validation, this paper outlines the scope of LLM consumer behavior and identifies open research questions related to alignment, preference representation, and market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17986v1">ShellGames: Speculative LLM-Driven SSH Deception</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Cyber deception and Moving Target Defense are promising strategies that aim to disrupt adversaries by increasing uncertainty. However, sustaining long-lived, credible interactive sessions with adversaries remains an open challenge. Large Language Models (LLMs) offer a promising path toward more dynamic deception systems, but suffer from key limitations that fundamentally limit their applicability, including: lack of persistent state, output inconsistencies, hallucinations, latency, and susceptibility to behavioral subversion that may reveal the deception. We propose ShellGames, an SSH shell simulator based on LLM designed to address these limitations. ShellGames combines five complementary techniques: (i) Automatic Chain-of-Thought and few-shot learning to improve correctness; (ii) memory management to maintain system state coherency; (iii) speculative command execution to reduce response latency; (iv) smart routing of complex interactive commands to a sandboxed environment; and (v) subversion detection leveraging the constrained input-output domain of shell environments. To enable systematic evaluation, we introduce a standardized benchmarking protocol and dataset spanning correctness, consistency, state tracking, and robustness tasks. ShellGames achieves $0.898$ command accuracy on correctness ($+5.3pp$ over baselines), $0.918$ sequence-level accuracy on consistency ($+36pp$), $0.98$ state tracking accuracy ($+18.3pp$), and $0.95$ accuracy on robustness ($+37pp$). A user study with $n=20$ participants confirms that ShellGames achieves realism comparable to a real shell under free exploration and outperforms traditional honeypots on perceived command coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17952v1">SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) architectures enable scaling LLM parameters under a fixed inference budget by activating only a small subset of experts via top-$k$ routing. While this preserves causality and suits autoregressive language models, the discrete top-$k$ operator is not differentiable, forcing a fixed number of active experts per input and resulting in inefficient use of computation. We propose SoftMoE, which replaces discrete routing with a truncated soft top-$k$ LapSum relaxation, allowing gradient-based optimization of expert routing. We further parameterize the mean number of active experts per layer and impose a global budget constraint, enabling the model to learn how to allocate expert capacity across layers. SoftMoE remains fully compatible with autoregressive modeling and achieves performance comparable to or better than sparse MoE on language modeling and downstream tasks, while activating significantly fewer experts. Notably, the learned allocation is highly non-uniform, with later layers activating more experts. The source code is publicly available$^\dagger$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17949v1">RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Heterogeneous LLM serving stacks split scheduling into two layers that optimize in isolation: model routers pick a model from quality and cost signals while ignoring instance load, and serving load balancers optimize queues while ignoring quality. We present RouteBalance, a serving-aware scheduling layer that fuses both into a single online assignment over concrete model instances, jointly trading off quality, latency, and cost. A batched in-process predictor stack and dead-reckoned instance state keep the joint decision cheap on the request hot path ($\approx$32 ms at 12 req/s). On a 13-instance, 28-GPU heterogeneous cluster serving four model sizes, a single deployed RouteBalance stack traces the upper region of the three-way quality-cost-throughput frontier. Sweeping one weight vector reaches both the highest routing-decision quality (DeepEval $0.419$, $+0.013$ over the strongest baseline, $95\%$ CI $[{+}0.005,{+}0.022]$; the ordering holds when a second judge re-scores the actually served text) and, at its cost-priority corner, per-request cost that ties the cheapest baseline. With router engineering equalized against concurrent-scoring baseline variants we build, its balanced preset serves at $2.8$ s and $30$ req/s, leading $2.6$ to $4.1\times$ ahead of enhanced BEST-Route at high load. (Deploying those routers as published, one serial scoring call per request, makes them collapse $23\times$ under load, a deployment-architecture effect we isolate separately, not the routing result.) A four-arm isolation shows the benefit follows from pricing latency at model-selection time; the learned predictors contribute calibration and SLO headroom rather than the headline frontier. Code: https://github.com/AKafakA/route-balance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06116v6">The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Generative AI models reproduce the human biases in their training data and further amplify them through mechanisms such as mode collapse. The loss of diversity produces homogenization, which not only harms the minoritized but impoverishes everyone. We argue homogenization should be a central concern in AI safety. To meaningfully characterize homogenization in Large Language Models (LLMs), we introduce a framework that allows stakeholders to encode their context and value system. We illustrate our approach with an experiment that surfaces gender bias in an LLM (Claude 3.5 Haiku) on an open-ended story prompt. Building from queer theory, we formalize homogenization in terms of normativity. Borrowing language from feminist theory, we introduce the concept of xeno-reproduction as a class of tasks for mitigating homogenization by promoting diversity. Our work opens a collaborative line of research that seeks to understand and advance diversity in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16337v2">Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Predictive modeling for clinical tabular data is central to clinical decision support and therefore requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning-beyond-gradients paradigm for clinical tabular prediction. Instead of relying on neural network weight updates, MHL uses a large language model (LLM)-driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable decision system. The resulting model is expressed not as opaque parameters, but as versioned pure-Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17915v1">Trustworthy Self-Composable Big-Data-as-a-Service: An LLM-Orchestrated Multi-Agent Framework for Automated Data Engineering, AutoML, MLOps Deployment, and Drift-Aware Lifecycle Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Big-Data-as-a-Service (BDaaS) platforms require re liable automation across data ingestion, cleaning, feature engi neering, model development, deployment, and post-deployment monitoring. However, existing LLM-based data science agents and AutoML systems mainly focus on isolated workflow stages, leaving limited support for lifecycle-level orchestration, artifact governance, human oversight, and drift-aware adaptation. This paper proposes a trustworthy self-composable BDaaS frame work based on LLM-orchestrated multi-agent collaboration. The proposed architecture decomposes the BDaaS lifecycle into specialized agents for data ingestion, data cleaning, feature engineering, AutoML training, model evaluation, MLOps de ployment, monitoring, and drift detection. A central LLM or chestration layer coordinates agent execution, validates interme diate outputs, manages workflow context, and enables dynamic workflow composition. The framework also incorporates shared artifact governance, reproducibility support, human-in-the-loop checkpoints, and drift-aware feedback loops. A prototype-based evaluation is conducted using controlled tabular benchmark datasets with missing values, categorical variables, outliers, class imbalance, and simulated covariate drift. Compared with manual ML, AutoML-only, and single-agent LLM baselines, the pro posed multi-agent BDaaS pipeline achieves competitive predictive performance while improving lifecycle-level reliability, including workflow completion, artifact traceability, deployment readiness, reproducibility, and drift recovery. The results suggest that LLM-orchestrated multi-agent systems can extend conventional AutoML toward trustworthy, adaptive, and production-oriented BDaaS lifecycle automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10139v2">TACOMORE: Exploring a replicable prompting protocol for LLM-assisted corpus analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      As corpus linguistics continues to scale, researchers are facing a growing methodological bottleneck: while computational tools can easily count billions of words, the qualitative interpretation of these data remains a slow and labor-intensive human task. Large Language Models (LLMs) offer a promising way to automate this process, yet their integration into the field is often hindered by concerns over black-box unpredictability and a lack of replicability. This study introduces TACOMORE, a structured prompting framework designed to transform ad-hoc AI interactions into a standardized linguistic protocol. Built upon four foundational principles (Task, Context, Model, and Replicability), the framework guides LLMs to move beyond generic probability prediction to anchoring their reasoning in the specific co-occurrence patterns of a target corpus. We applied this framework to three core corpus tasks, i.e., the analysis of keywords, collocates, and concordances, using an open corpus of COVID-19 research abstracts. After testing three LLMs, we found that while structured prompting improves accuracy and replicability, inherent limitations regarding hallucination persist. This research offers a critical lens into the role of LLMs in corpus linguistics, highlighting their potential as complementary tools while emphasizing the irreplaceable role of human validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02721v2">Blueprint First, Model Second: A Framework for Deterministic LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 12 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      While powerful, the inherent non-determinism of large language model (LLM) agents limits their application in structured operational environments where procedural fidelity and predictable execution are strict requirements. This limitation stems from current architectures that conflate probabilistic, high-level planning with low-level action execution within a single generative process. To address this, we introduce the \textsc{Source Code Agent} framework, a new paradigm built on the ``Blueprint First, Model Second'' philosophy that decouples workflow logic from the generative model. An expert-defined operational procedure is first codified into a source code-based Execution Blueprint, which is then executed by a deterministic engine. The LLM is strategically invoked as a specialized tool to handle bounded, complex sub-tasks within the workflow, but never to decide the workflow's path. We evaluate on the TravelPlanner benchmark for constraint-aware travel planning. The \textsc{Source Code Agent} achieves a 35.56\% final pass rate, a 97.6\% improvement over the state-of-the-art ATLAS baseline (18.00\%) on the same Claude-Sonnet-4 backbone. Critically, it reduces constraint violations by 96.0\% (11 vs 275) while improving execution efficiency by 27.1\% (10.2$\pm$0.7 steps vs 14.0). Two production incident-diagnosis deployments and additional results on ScienceWorld and ALFWorld confirm that the architecture transfers beyond travel planning to procedurally well-defined, constraint-intensive workflows. Our work enables the verifiable and reliable deployment of autonomous agents in applications governed by strict procedural logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30036v2">Teaching Values to Machines: Simulating Human-Like Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 We had some disagreement regarding proper attribution; we hope to resolve it soon and upload the paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles; however, it remains unclear whether they can manifest behavior that adheres to a coherent, human-like value structure. In this work, we draw on established psychological value theory to induce human-like values in LLMs and assess their alignment with patterns observed in human studies. Using validated psychological questionnaires, we conduct large-scale experiments -- over 5 million questions -- to evaluate value structures and value-behavior relationships in leading LLMs and compare them to humans. Our findings reveal strong agreement between value-prompted LLMs and humans across both dimensions. Moreover, incorporating human value distributions enhances population-level simulations with value-induced LLMs. These findings highlight the potential of value-induced LLMs as effective, psychologically grounded tools for simulating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17838v1">Environment-Grounded Automated Prompt Optimization for LLM Game Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM agents in interactive environments are highly sensitive to their prompts, yet prompt engineering remains a manual, task-specific process. We introduce an automated prompt optimization framework for LLM agents that decomposes the observation-to-action pipeline into a goal-conditioned descriptor agent and an action selection agent, and iteratively refines each module's prompt through an LLM-driven evolutionary loop guided by environment returns. We propose a behavior analyzer to attribute episode outcomes to specific prompt components, and a mutator to propose targeted revisions to the prompt, before validating them through environment rollouts. We evaluate on all five BabyAI tasks in the BALROG benchmark, comparing our pipeline against BALROG's RobustCoTAgent under both plain and guided prompt initializations. Optimization improves performance consistently across tasks and conditions, without requiring updates to the model weights. On PutNext, a multi-step coordination task where the RobustCoTAgent achieves 0% success, our framework reaches up to 72.5% success rate using the same underlying LLM with optimized prompts. These results suggest that a multi-agent framework, combined with automatic prompt optimization, enhances LLMs without the need for fine-tuning or extensive human supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09004v2">LATTEArena: An Evaluation Framework for LLM-powered Tabular Feature Engineering (Extended Version)</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Feature engineering remains a cornerstone of tabular data analysis, and Large Language Models (LLMs) have emerged as a promising paradigm for its automation, giving rise to LLM-powered Automated Tabular Feature Engineering (LATTE). However, the field lacks standardized, cost-aware evaluation platforms, and the combinatorial explosion of design choices obscures true algorithmic progress. To bridge these gaps, we systematically deconstruct 15 representative LATTE methods into a unified 6-dimensional taxonomy. Based on this abstraction, we introduce LATTEArena, a standardized, modular, and extensible benchmarking framework that decouples monolithic pipelines into reusable execution blocks. By distilling the massive combinatorial space, we evaluate 24 core LATTE configurations across 7 research questions. Our head-to-head benchmarking goes beyond predictive accuracy to quantify token efficiency and execution robustness, yielding 17 empirical findings on cost-effectiveness trade-offs. Furthermore, we provide 3 concrete recommendations for optimal real-world deployment. By enabling controlled component-level comparisons, LATTEArena shifts the paradigm from ad-hoc prompt engineering to systematic context management. All code, datasets, and over 4,000 execution logs are publicly available to foster a dynamic, community-driven benchmark. Our framework, leaderboard, and all artifacts are hosted on the LATTEArena project website at https://goodenhak.github.io/LATTEArena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17787v1">LUMEN: Coordinated Failure Recovery for Distributed LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) serving clusters distribute inference requests across multiple worker processes on different GPUs, but failures are prevalent at scale. When a worker fails, the cluster simultaneously loses the failed worker's GPU-resident key-value (KV) caches and serving capacity, leaving surviving workers to absorb the redirected traffic while re-running interrupted requests from scratch. Existing fault-tolerant systems either restart interrupted requests from scratch or restore KV caches from checkpoints stored on a fixed neighboring worker, but both approaches route recovery work without considering current cluster load and leave the recovering worker idle during model reload. We present LUMEN, a fault-tolerant LLM serving system that treats recovery as a load-aware coordination problem across three decision points: checkpoint placement before failures, interrupted-request distribution at failure time, and serving capacity restoration during model reload. We evaluate LUMEN using both prototype experiments and large-scale simulations and demonstrate significant improvements in serving and recovery times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17735v1">Shattering the Autoregressive Curse: Dynamic Epistemic Entropy Orchestrated Erasable Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Although reinforcement learning (RL) has expanded the cognitive boundaries of large language models (LLMs), it often remains vulnerable to the autoregressive curse in long-horizon logical reasoning: small epistemic perturbations introduced early in generation can propagate irreversibly along the Markov decision process flow, triggering cascading failures that drive the reasoning trajectory toward collapse. To overcome this autoregressive cascade, in which a single early mistake can compromise all subsequent reasoning steps, we propose dynamic epistemic entropy orchestrated erasable reinforcement learning ($\text{E}^3\text{RL}$). $\text{E}^3\text{RL}$ eliminates reliance on external signals by grounding the model's endogenous local autoregressive cross-entropy as an intrinsic coordinate of epistemic uncertainty. By introducing segment-level adaptive dynamic thresholds and advantage allocation, $\text{E}^3\text{RL}$ enables the model to precisely excise localized logical defects while reusing historical key-value (KV) cache streams, thereby endowing the reasoning process with a self-healing capability. We train $\text{E}^3\text{RL}$ on the DeepMath-103k dataset. Experimental results show that $\text{E}^3\text{RL}$ reshapes the exploration efficiency of long-sequence reasoning and improves sample efficiency while maintaining linear memory overhead. On mathematical reasoning benchmarks such as AIME, $\text{E}^3\text{RL}$ achieves substantial performance gains, with the 4B and 8B parameter models surpassing previous state-of-the-art (SOTA) results by 5.349\% and 6.514\%, respectively. These findings suggest that $\text{E}^3\text{RL}$ shatters the autoregressive curse in long-sequence reasoning and establishes a theoretical and systems-level foundation for the next generation of self-healing artificial general intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16591v2">SING: Synthetic Intention Graph for Scalable Active Tool Discovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on agent harnesses that manage context, tools, and multi-turn execution, making tools a central interface for acting in realistic digital environments. As harness-connected tool ecosystems expand to hundreds or thousands of APIs, services, and task-specific skills, exhaustive tool schema injection becomes costly and imposes a closed-world assumption that limits agents to a predefined static inventory. Retrieval-augmented tool selection offers a natural alternative, but existing one-shot retrieval methods often fail to align isolated tool descriptions with the agent's true task intention, especially in long-horizon tasks where required capabilities emerge through decomposition, observations, and newly induced subgoals. We propose SING, an intention-aware active tool discovery framework that builds an intention-tool graph linking user intentions, tool capabilities, and tool collaboration patterns, and dynamically retrieves tools according to evolving task states. Using a unified corpus of 7,471 tools, we evaluate SING on three real-world tool-use benchmarks. SING improves Global Recall@5 by up to 59.8% and downstream success rate by up to 28.9% over baselines, while reducing full-corpus tool-schema exposure by 99.8%, demonstrating that intention-aware graph structure enables more accurate and context-efficient tool discovery in large-scale agentic ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28251v3">DiffAttn: Diffusion-Based Drivers' Visual Attention Prediction with LLM-Enhanced Semantic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Drivers' visual attention provides critical cues for anticipating latent hazards and directly shapes decision-making and control maneuvers, where its absence can compromise traffic safety. To emulate drivers' perception patterns and advance visual attention prediction for intelligent vehicles, we propose DiffAttn, a diffusion-based framework that formulates this task as a conditional diffusion-denoising process, enabling more accurate modeling of drivers' attention. To capture both local and global scene features, we adopt Swin Transformer as encoder and design a decoder that combines a Feature Fusion Pyramid for cross-layer interaction with dense, multi-scale conditional diffusion to jointly enhance denoising learning and model fine-grained local and global scene contexts. Additionally, a large language model (LLM) layer is incorporated to enhance top-down semantic reasoning and improve sensitivity to safety-critical cues. Extensive experiments on four public datasets demonstrate that DiffAttn achieves state-of-the-art (SoTA) performance, surpassing most video-based, top-down-feature-driven, and LLM-enhanced baselines. Our framework further supports interpretable driver-centric scene understanding and has the potential to improve in-cabin human-machine interaction, risk perception, and drivers' state measurement in intelligent vehicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18897v3">Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM-powered agents execute tasks through a sequential loop of model generation and tool execution. Today's serving systems serialize this loop, leaving tool latency exposed on the task critical path. This paper presents PASTE, a tool-aware agent-serving system that predicts concrete future tool invocations from recurring agent patterns and executes them speculatively while the LLM is still generating. PASTE isolates speculative results until confirmed by the LLM and jointly schedules tool execution and returning LLM sessions to avoid shifting bottlenecks to the GPU. Across deep research, coding, and scientific-agent workloads, PASTE reduces average task completion time by 43.5% and lowers observed tool latency by 1.8x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14517v2">From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      LLM-based guardrails have emerged as a highly effective defense against prompt injection and jailbreak attacks in autonomous agents. However, we reveal that the very reasoning and task-following capabilities enabling this protection introduce a novel vulnerability: attackers can inject crafted data to trap the guardrail in extended reasoning loops, effectuating a systematic denial-of-service (DoS) attack. To systematically expose this threat, we design a beam-search optimization framework that crafts natural-language payloads to maximize guardrail reasoning length, utilizing an LLM proposer guided by a strategy bank. Based on the observation of guardrail's schema-following nature, we also provide another attack framework driven by mechanism-aware structural mutations with less computational load. The attack efficacy is systematically evaluated in two parts. First, in standalone evaluations, the attack generalizes across diverse guardrail architectures, safety templates, and agent benchmarks. Payloads optimized on a single open-source surrogate successfully transfer to eight leading model backbones (e.g., Claude, GPT, Gemini, DeepSeek, and Qwen), achieving a 13--63$\times$ token amplification. Second, in end-to-end real-world agent deployments (web, desktop, code, and multi-agent systems), the attack reveals up to a 148$\times$ latency amplification. We show that a single poisoned document can saturate shared guardrail infrastructures, effectively starving co-located agents and paralyzing the entire system. By uncovering this availability flaw, our work underscores the urgent need to develop cost-bounded, reasoning-robust guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17707v1">Do Generative Recommenders Deepen the Information Cocoon? A Closed-Loop Simulation with LLM-powered User Simulators</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Recommender systems alleviate information overload, yet repeated feedback between recommendations and user interactions can reinforce existing preferences and narrow users' exposure, forming information cocoons. While this phenomenon has been widely studied in traditional sequential recommendation, its impact on generative recommendation remains unclear. By replacing atomic item IDs with Semantic ID (SID) sequences, generative recommenders introduce a different recommendation mechanism whose role in information cocoon formation is not yet understood. To investigate whether generative recommenders deepen information cocoons, we propose \textsc{RecLoop}, a closed-loop simulation framework with LLM-driven user agents. We compare two generative recommenders and two traditional sequential baselines on two Amazon datasets across multiple feedback cycles. In addition to standard exposure-level metrics, we introduce \emph{Code-Space Structural Cocoon}, a model-level metric that measures concentration in the generated SID space. Experimental results show that generative recommenders are generally less prone to exposure-level cocoon formation than traditional baselines, preserving broader exposure diversity and slowing cross-user homogenization. However, feedback loops can still induce concentration within the generated SID space. We further find that cocoon severity depends strongly on tokenization strategy and model scale: collaborative-signal tokenization produces stronger cocoon effects than semantic tokenization, whereas larger models maintain greater code-space diversity and better retain access to niche content. These findings suggest that information cocoons in generative recommendation are shaped not only by recommendation behavior, but also by item tokenization and model capacity. Our code is available at https://github.com/Dregen-Yor/RecLoop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17688v1">LLMs Infer Cultural Context but Fail to Apply It When Responding</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 9 pages, 7 figures, 2 tables (24 pages, 12 figures, 8 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Recent work has shown that LLMs overrepresent dominant cultures, particularly Western ones, while marginalizing others. We investigate whether this affects models' ability to generate culturally adapted responses by evaluating their use of local measurement units based on the user's perceived cultural background. We introduce Cultural and Pragmatic Response Inference (CAPRI), a dataset of conversations with varying levels of cultural cues. Experiments with state-of-the-art LLMs show that models can infer cultural background and recall relevant conventions, but often fail to utilize the information to adapt their answers to the relevant cultural conventions, unless explicitly prompted to perform the tasks sequentially. We further evaluate adaptation to the interpretation of time and quantity expressions, two subjective language grounding dimensions that are affected by culture. We find that models increasingly adapt their answers as cultural cues accumulate, but their priors are not culture-neutral, sometimes aligning with the model's country of origin. Overall, CAPRI provides a resource for future research aimed at narrowing the gap between cultural knowledge and culturally adaptive language generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17683v1">Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have greatly advanced the functional correctness of automated code translation systems, the runtime efficiency of translated programs has received comparatively little attention. With the waning of Moore's law, runtime efficiency has become increasingly important for program quality, alongside functional correctness. Our preliminary study reveals that LLM-translated programs often run slower than human-written ones, and this issue cannot be remedied through prompt engineering alone. Therefore, our work proposes SwiftTrans, a code translation framework comprising two key stages: (1) Multi-Perspective Exploration, where MpTranslator leverages parallel in-context learning (ICL) to generate diverse translation candidates; and (2) Difference-Aware Selection, where DiffSelector identifies the optimal candidate by explicitly comparing differences between translations. We further introduce Hierarchical Guidance for MpTranslator and Ordinal Guidance for DiffSelector, enabling LLMs to better adapt to these two core components. To support the evaluation of runtime efficiency in translated programs, we extend existing benchmarks, CodeNet and F2SBench, and introduce a new benchmark, SwiftBench. Experimental results across all three benchmarks show that SwiftTrans achieves consistent improvements in both correctness and runtime efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17682v1">From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning pipelines for Large Language Model (LLM) training often rely on manually redesigned environments between stages, requiring practitioners to heuristically infer which configuration will best improve the current policy. To automate this process, we propose the LLM-as-Environment-Engineer framework in which the current policy model analyzes failure trajectories together with contextual information and proposes modifications to the next-stage training environment configuration. We also introduce MAPF-FrozenLake, a controllable testbed whose generator exposes multi-dimensional environment configurations, making it suitable for studying and benchmarking environment redesign. On this testbed, we condition the environment engineer on structured summaries of policy behavior, failure cases, and environment statistics, from which it produces the configuration for the next training stage. With Qwen3-4B as the backbone, our framework achieves the strongest aggregate performance on our benchmarks, outperforming larger proprietary LLMs (e.g., GPT, Gemini) and fixed-environment training baselines. We further analyze which forms of context are most effective, finding that successful environment updates rely on failure evidence and preserve configurations that already work. Interestingly, the current RL checkpoint serves as a better environment engineer than the original base model, suggesting that policy learning improves the model's ability to diagnose its remaining weaknesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17666v1">FacProcessTwin: An LLM-Based System for Process Twin Development</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Process twins provide real-time representations of entire production processes. By capturing how process steps interact, rather than monitoring a single machine in isolation as an asset-based digital twin does, they have the potential to drive efficiency gains across the whole process. However, developing a process twin is costly. It requires accurately modelling the entire production process: its process steps, the equipment and product-specific settings each step uses, and its process variations. The resulting model must then be bound to live operational data. We present FacProcessTwin, a system that leverages a large language model (LLM) to reduce this development time, building a process twin from a plant's process documentation and natural-language input from an operator. FacProcessTwin generates this complete process model and then automatically binds its process steps to live operational data. The generated model and its data bindings are rendered as an interactive process diagram through which manufacturing personnel can monitor and correct the system's autonomous decisions, such as resolving uncertainty at safety-critical binding steps. We evaluate FacProcessTwin through a real-world case study of an Australian food manufacturer, covering 16 production process flows that span chilled, frozen, and aseptic shelf-stable product categories and include process variations within the same product. The results show that FacProcessTwin generates these process models accurately (a mean F1 of 95.2% against ground truth) and builds each twin in roughly a sixth of the manual time. Its human-in-the-loop governance then keeps the safety-critical bindings correct: at ambiguous tags where a single-pass baseline silently mis-binds 75.0% of the time, FacProcessTwin defers to the operator and mis-binds none.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17648v1">From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Standard accuracy metrics cannot explain why LLMs handle variable tracking but fail on semantically equivalent loops. We study an internal lifecycle of code reasoning in which models first brew the answer, making it linearly recoverable many layers before it becomes self-decodable, and then diverge into one of four resolution outcomes: Resolved, Overprocessed, Misresolved, or Unresolved. Understanding this lifecycle matters because similar task accuracies can mask fundamentally different failure modes that surface-level evaluation cannot detect. We introduce a dual diagnostic framework pairing layer-wise linear probing with Context-Stripped Decoding (CSD) and apply it to six code-reasoning task families across 16 models spanning Qwen, Llama, and DeepSeek architectures. All four outcomes carry substantial mass in every task family: overall Resolved is only 41.5%, with multiple tasks below 30%. Controlled sweeps over structure, depth, and operators expose task-specific failure bottlenecks: Function Call Resolved plunges from 61.1% to 2.5% as call depth increases from one to three. Across architectures and scales, the brewing scaffold remains stable, with normalized brewing duration 24-42% across all 16 models, while resolution success varies with capability. This indicates that the scaffold is a stable empirical regularity across the tested decoder-only Transformer families, whereas resolution success covaries with capability, scale, and training. Code: https://github.com/euyis1019/llm-brewing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17188v3">LLM-Aided Joint Secrecy Precoding and Trajectory for RSMA-Based Heterogeneous UAV Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      This paper investigates secure communications in rate-splitting multiple access (RSMA) enabled heterogeneous UAV networks, where multiple UAVs collaboratively serve ground terminals in the presence of eavesdroppers. By jointly considering secrecy rate maximization and propulsion energy consumption minimization, we formulate a multi-objective optimization problem involving UAV trajectory design, service association, power allocation, and secrecy precoding under mobility, collision-avoidance, service-capacity, and communication constraints. The formulated problem is highly non-convex due to the coupling among UAV trajectories, RSMA transmission variables, and secrecy constraints.To address the resulting non-convex and highly coupled optimization problem, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (D.C.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates LLM-generated expert heuristic policy, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17634v1">Prompt Perturbation for Reliable LLM Evaluation over Comparison Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 42 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) is important for understanding their capabilities, comparing competing systems, and supporting the deployment of reliable models in practice. For open-ended tasks, pairwise evaluation has become a popular paradigm, in which two responses to the same prompt are compared and the resulting judgments are aggregated into an overall ranking. A central challenge of this paradigm is intransitivity: the induced comparison outcomes may fail to support any coherent global ranking. For example, one may observe cyclic preferences such as $A \succ B \succ C \succ A$, or inconsistencies involving ties such as $A \equiv B\equiv C\neq A$. Such contradictions make the resulting leaderboard unstable and challenging to interpret. In this paper, we propose a prompt perturbation framework for improving the consistency of pairwise LLM evaluation. Our approach generates perturbed variants of each prompt, uses the resulting comparison graphs to identify and filter out structurally inconsistent comparison patterns, and then applies standard ranking methods to the filtered comparisons. A key feature of the proposed framework is that graph-level structural consistency is incorporated explicitly into the evaluation pipeline before ranking aggregation. This provides a simple and principled way to reduce cyclic inconsistencies and improve the reliability of LLM rankings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01973v3">Learn-To-Learn on Arbitrary Textual Conditioning: A Hypernetwork-Driven Meta-Gated LLM</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Accepted by ICML2026
    </div>
    <details class="paper-abstract">
      Conventional LLMs may suffer from corpus heterogeneity and subtle condition changes. While finetuning can create the catastrophe forgetting issue, application of meta-learning on LLMs is also limited due to its complexity and scalability. In this paper, we activate the meta-signal of $β$ within the SwiGLU blocks, resulting in a meta-gating mechanism that adaptively adjusts the nonlinearity of FFN. A hypernetwork is employed which dynamically produces $β$ on textual conditions, providing meta-controllability on LLMs. By testing on different condition types such as task, domain, persona, and style, our method outperforms finetuning and meta-learning baselines, and can generalize reasonably on unseen tasks, condition types, or instructions. Our code can be found in https://github.com/AaronJi/MeGan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01904v3">Combating Data Laundering in LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 29 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Post-hoc unauthorized-training data detection for large language models (LLMs) typically assumes a query-with-originals regime: rights holders query a target LLM with raw proprietary data and assess whether the model assigns them stronger memorization-based detection signals, e.g., higher confidence or lower loss, than held-out non-training reference texts. We show that this regime becomes brittle under data laundering, where the target LLM is trained on semantics-preserving but stylistically or structurally transformed surrogates of proprietary data to obfuscate provenance. Since training-time exposure occurs in the laundered form, memorization signals may no longer appear on the originals, collapsing the candidate-reference signal separation that standard detectors rely on. We counter this threat by studying laundering-aware detection with raw proprietary data, a held-out reference corpus, and query access to the target LLM, while the laundering transformation is undisclosed. Since exact recovery of the laundered corpus is infeasible, we infer a detection-useful synthesis process via an auxiliary LLM that maps originals into training-like queries. To make this search tractable, we introduce Synthesis Data Reversion (SDR), which constrains the unbounded space of natural-language transformations through a goal-details abstraction: a high-level transformation goal, e.g., "lyrical rewriting", and fine-grained details, e.g., "with vivid imagery". SDR identifies the most likely goal and iteratively refines details so synthesized queries elicit stronger target-model detection signals. Evaluated on the MIMIR benchmark against diverse laundering practices and target LLM families (Pythia, Llama2, and Falcon), SDR consistently restores detection signals, offering a practical auditing layer against data laundering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17612v1">PracRepair: LLM-Empowered Automated Program Repair Inspired by Human-Like Debugging Practices</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      As software systems grow in scale and complexity, debugging and repair remain costly and time-consuming. Large language models (LLMs) have advanced automated program repair (APR), but existing LLM-based APR approaches still largely rely on static or retrieved context, error messages, and coarse-grained validation outcomes. As a result, they underutilize dynamic information for failure understanding and repair, including failure-execution dynamics and patch-validation dynamics. Effectively leveraging such information, however, is challenging: failure-execution traces are large and noisy, raw static-dynamic context is not self-explanatory, and patch-validation dynamics are often reduced to coarse feedback. To address these challenges, we propose \textsc{PracRepair}, a fully automated LLM-based APR framework inspired by human-like debugging practices. \textsc{PracRepair} constructs an on-demand static-dynamic context from buggy programs and failure executions, performs question-driven failure diagnosis to formulate explicit repair hypotheses, and iteratively refines candidate patches using validation diagnostics and trace-level behavioral changes. Experimental results on Defects4J V1.2 and V2.0 show that \textsc{PracRepair} consistently outperforms state-of-the-art baselines. Specifically, under GPT-3.5, \textsc{PracRepair} correctly fixes 139/136 bugs on Defects4J V1.2/V2.0, while under GPT-4o it further improves to 162/171. Moreover, \textsc{PracRepair} generalizes effectively to RWB (Real-World Bugs), achieving the best performance across multiple foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17609v1">The Benchmark Illusion: Pruned LLMs Can Pass Multiple Choice but Fail to Answer</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Compressing large language models reduces memory use and inference cost, but it can also create failures that standard benchmarks miss. A pruned model may still perform well on multiple-choice evaluations, yet fail to answer the same question in open generation. We ask what pruning changes: does it erase the correct answer, or does it make the answer harder to produce as the top output? We study this question with multilingual question answering, tracking the same questions before and after pruning. We find a benchmark illusion. Under high-sparsity pruning, especially Wanda, models often fail in greedy open generation while still selecting the correct answer under multiple-choice scoring. In these recognition-only errors, the answer is usually not gone, but demoted: it often reappears with beam search, sampling, or one in-context example. Overall, multiple-choice benchmarks can overstate the usability of compressed LLMs, creating an evaluation blind spot. Compressed models should be tested on what they can produce, not only on what they can recognize.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25652v2">A Two-Phase Stability Study of LLM Judges and Bar Council Examiners on Thai Bar-Exam Free-Form Essays</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Free-form legal essay evaluation in NLP treats expert inter-rater stability as a single ceiling number, and treats LLM-judge agreement with that ceiling as evidence of judge stability. We test both assumptions on the Thai bar examination through an identical-inputs protocol: three Bar Council-trained examiners (A, B, C) and a 26-LLM judge panel score the same 15 cross-graded answers from the same four inputs (question, official Bar Council grading regulation, gold answer, candidate answer). The headline finding is asymmetric. On 10 of 15 cells where the rubric prescribes both axes, all 29 raters converge in a tight band: panel agreement is universal. On the remaining 5 cells where the rubric does not prescribe how to grade a correct final answer that omits a decisive statutory citation, the human panel splits between two coherent readings (B/C majority at the upper rubric band, score 6-8; A minority at the lower band, score 1-2). The LLM judge population does not split symmetrically: 22 of 26 LLMs score in or near B/C's contested band, 3 sit in the regulation-silent middle gap, and only 1 (GPT-5.4 Nano) approaches A's band without consistently scoring within it. Zero LLMs in our 26-judge panel reproduce the minority human reading on the contested cells. The B/C-direction cluster spans every model size, vendor, and price tier we tested. An instrumented three-LLM anchor sub-panel (Claude 4.6 Opus, Gemini 3.1 Pro, GPT-5.4 Pro) carries determinism probes, input ablations, and bootstrap CIs, and reaches anchor panel $α= 0.77$ on the 15 cells against human-panel $α= 0.36$. The high LLM-panel $α$ reflects systematic convergence on the majority reading rather than balanced reproduction of both readings; a benchmark that selects its LLM judge by maximising agreement with a human reference panel will inherit this asymmetry by construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17588v1">Understanding LLMs in Title-Abstract Screening: From Disagreements to Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 14 pages + references. Accepted for publication in the 52nd Euromicro Conference on Software Engineering and Advanced Applications (SEAA 2026)
    </div>
    <details class="paper-abstract">
      Several studies have examined the use of large language models (LLMs) for title-abstract screening in systematic reviews (SRs), reporting mixed accuracy. However, questions of reliability remain largely unaddressed. In this study, we go beyond quantitative LLM-human agreement metrics and qualitatively investigate how and why LLMs fail. We also propose actionable recommendations. We analyzed disagreements between LLMs and researchers across six software engineering SRs and over 1,000 primary study papers. For each SR, papers were screened independently by human experts and LLMs in zero-shot mode, resulting in Kappa values ranging from 0.52 to 0.77. Qualitative analysis suggests that human-LLM disagreement results from recurring, identifiable causes, such as boundary ambiguity in key terms, keyword overemphasization, and incorrect topic inference. Based on these findings, we propose recommendations such as validating semantic understanding before deployment, running multiple LLMs, and focusing validation efforts on borderline cases. Future studies are needed to validate the impact of our recommendations, and community efforts are needed to develop normative guidelines on LLM usage in SRs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15121v2">When Cognitive Graphs Meet LLMs: BDEI Cognitive Pathways for Panic Emotional Arousal Prediction</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Predicting individual panic emotional arousal timing before manifestation is essential for proactive emergency intervention. Existing methods incorporate cognitive elements but none explicitly model the emotional arousal process, making them ill-suited for emotional arousal timing prediction. We argue that grounding prediction in appraisal emotion theory is necessary because it explicitly models this process, but three problems must be solved. (1) Appraisal theory posits that emotion arises from simultaneous evaluation across multiple threat dimensions, yet no prior work fuses these inputs into risk perception. (2) Existing cognitive models lack an Emotion node, decoupling threat appraisal from emotional arousal and forcing emotions to be inferred indirectly from behaviors. (3) Given their generalizable cognitive reasoning, current approaches adopt LLMs as the primary decision-maker, yet overlook the fragility and hallucination-proneness of their outputs. To address these issues, we introduce PanicCognitivePath (PCP), a framework that addresses all three. A Psychological Safety Distance (PSD) model, grounded in psychological distance theory, maps four-domain signals into a unified risk metric as the entry condition for subsequent cognitive reasoning. An explicit Emotion node grounded in appraisal emotion theory is introduced into BDI, forming a Belief-Desire-Emotion-Intention (BDEI) pathway. Agents whose risk metric exceeds the PSD threshold enter this pathway, coupling threat appraisal directly to emotional arousal. The BDEI pathway governs all state transitions while the LLM is confined to parameter estimation for the Belief-to-Desire transition, confining hallucinations to a single step and preventing error propagation. Experiments on Hurricane Sandy show PCP improves arousal timing accuracy by 10.68% over baselines, reduces peak count error to 7.07%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17581v1">Visored: A Controlled-Natural-Language Prover for LLM-Generated Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      We present a dependent-type-based prover designed around the way LLMs (and humans) tend to write mathematics, complementing existing systems such as Lean and Rocq. Its core design choices are a surface that imitates mathematical natural language and a rule-driven automation layer that closes the routine steps a textbook would omit, so that an accepted proof can be re-emitted as a checked Lean file. Early experiments suggest that, even without any prover-specific training data, LLMs can learn to use it effectively on the miniF2F benchmark. Lean output excerpts: https://github.com/xiyuzhai-husky-lang/visored/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17579v1">LLM Features Can Hurt GNNs: Concatenation Interference on Homophilous Graph Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 29 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Adding LLM-generated node features to graph neural networks (GNNs) is widely reported to improve accuracy on standard benchmarks. We document a contrasting observation: when LLM features are introduced through pure input concatenation (rather than joint training, distillation, or prompt-conditioning), they can systematically degrade accuracy on the same homophilous benchmarks where end-to-end LLM pipelines succeed. With an MLP backbone on the Planetoid public split and bag-of-words original features, concatenating SBERT-encoded GPT-4o-mini TAPE features reduces PubMed test accuracy by -17.0 +/- 0.3 pp and Cora by -4.3 +/- 0.6 pp (CiteSeer -0.6 +/- 0.8 pp, within seed noise). The drop attenuates as we relax each condition (GCN / GCNII / GAT backbones, random splits, smaller encoders) and reverses on medium-homophily WikiCS (+4.4 pp) and ogbn-arxiv (+11.7 pp). To predict when concatenation helps versus hurts, we report a simple measure of LLM-alone discriminability, Delta_sig. Across 9 datasets Delta_sig correlates with the concatenation cost more strongly than homophily at point estimate (r^2 = 0.38 vs. 0.06; N=9, bootstrap CIs overlap). The bootstrap-best change-point is tau = 13.8 pp, and the rule "Delta_sig <= tau predicts non-positive concat cost" classifies 7/9 datasets correctly; since 60% of bootstrap samples place tau in [5, 30] pp, we treat Delta_sig as an interpretive lens rather than a precision filter. A dimension-controlled ablation on PubMed places the LLM-feature drop between same-source PCA (-2.3 pp) and same-dim Gaussian noise (-37.3 pp), ruling out dimensionality and weight-decay artifacts. Nine PubMed configurations fit a power law |Delta_concat| proportional to (sqrt(d_l/n))^1.31 with r^2 = 0.97; the low-Delta_sig, small-n corner is exactly where the headline -17 pp PubMed deficit appears.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17573v1">Cordon: Semantic Transactions for Tool-Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Tool-using LLM agents are shifting the unit of computation from explicit human-issued commands to model-driven tasks with stateful consequences. Yet today's agent runtimes still expose tools as isolated RPCs. This interface gives runtimes a convenient integration point, but it lacks a task-scoped execution boundary for commit, rollback, recovery, and audit across multi-step agent workflows. We argue that this mismatch calls for a runtime containment boundary rather than another per-call guardrail. This paper introduces Cordon, a transactional runtime system for staging and validating irreversible agent effects before commit. A semantic transaction is a task-level execution boundary that binds tool intents and runtime-tracked result lineage to reversible local state, staged external effects, delegated authority, and audit metadata. Cordon implements this abstraction with a transaction manager that tracks derived result objects, executes reversible mutations in shadow state, stages outward-facing actions in an effect outbox, and records recovery metadata. The runtime then validates the composed execution flow before it commits state or releases external effects. Our evaluation across adversarial and benign workflows shows that Cordon exposes cross-step violations missed by existing defenses. It also reduces irreversible-effect failures while preserving benign task completion with modest approval and latency overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17546v1">SEAGym: An Evaluation Environment for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Self-evolving LLM-based agents improve mainly by changing their agent harness: the structured execution layer around a base model, including prompts, memory, tools, middleware, runtime state, and the model-tool interaction loop. Existing evaluations often reduce this process to isolated task scores or a single sequential curve, obscuring whether an update produces reusable improvement, overfits recent tasks, increases cost, or harms older behavior. We introduce SEAGym, an evaluation environment for measuring agent harness updates across training, validation, test, replay, and cost records. SEAGym turns Harbor-compatible benchmarks into dynamic self-evolution task sources with train batches, frozen update-validation, held-out ID and OOD transfer views, replay diagnostics, and saved snapshot and metric records. Instantiating SEAGym on Terminal-Bench 2.0 and HLE, we compare ACE, TF-GRPO, and AHE under a shared epoch/batch protocol. The results show that these evaluation views provide complementary signals about the evolution process: frequent updates may fail to improve held-out performance, useful intermediate snapshots may collapse later, and source diversity and model backend can affect harness reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17536v1">OmniDrive: An LLM-Choreographed Multi-Agent World Model with Unified Latent Co-Compression for Multi-View Driving Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Generative world models for autonomous driving face two unresolved tensions: heterogeneous control injection, where free-form language, HD-maps, trajectories, and camera poses reside in incompatible representational spaces, and post-hoc cross-view fusion, where per-camera latents fail to encode global 3-D geometry. We trace both to a single root cause: the absence of a shared symbolic interlingua aligning language, geometry, and pixels at the latent-token level. We present DRIVE-CHOREO, an LLM-choreographed multi-agent world model that recasts controllable multi-view video generation as latent choreography. Three Qwen2.5-VL agents - a Director parsing user intent into a structured WorldScript, a Cartographer grounding it into spatially-anchored layout tokens, and an Auditor feeding cross-view critiques back as auxiliary supervision - jointly author a single position-aware token sequence. This sequence is co-compressed with the multi-view video via a view-time permutation that enforces inter-camera geometry within the convolutional receptive field of a 3-D VAE. On nuScenes, DRIVE-CHOREO sets new state-of-the-art multi-view consistency and BEV mAP (21.6) with competitive FVD (45.7); a detector trained purely on our synthetic data gains +2.4 NDS on the real validation split, validating downstream utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04574v2">FeedEval: Pedagogically Aligned Evaluation of LLM-Generated Essay Feedback</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Going beyond the prediction of numerical scores, recent research in automated essay scoring has increasingly emphasized the generation of high-quality feedback that provides justification and actionable guidance. To mitigate the high cost of expert annotation, prior work has commonly relied on LLM-generated feedback to train essay assessment models. However, such feedback is often incorporated without explicit quality validation, resulting in the propagation of noise in downstream applications. To address this limitation, we propose FeedEval, an LLM-based framework for evaluating LLM-generated essay feedback along three pedagogically grounded dimensions: specificity, helpfulness, and validity. FeedEval employs dimension-specialized LLM evaluators trained on datasets curated in this study to assess multiple feedback candidates and select high-quality feedback for downstream use. Experiments on the ASAP++ benchmark show that FeedEval closely aligns with human expert judgments and that essay scoring models trained with FeedEval-filtered high-quality feedback achieve superior scoring performance. Furthermore, revision experiments using small LLMs show that the high-quality feedback identified by FeedEval leads to more effective essay revisions. We release our code and curated datasets at: https://github.com/BBeeChu/FeedEval.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17514v1">Unlocking LLM Code Correction with Iterative Feedback Loops</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 22 pages, 14th Computing Conference 2026
    </div>
    <details class="paper-abstract">
      Large Language Models have shown remarkable capabilities in code generation. However, most existing evaluations focus only on single-attempt accuracy and overlook the iterative refinement process that is central to real-world programming. This study presents a systematic investigation of LLMs' ability to rectify their own code through execution feedback. Using real-world programming problems across four models and two major programming languages, this study evaluates performance using iterative refinement framework where LLMs receive compiler error messages and testcase feedback after each attempt. This study introduces metrics to evaluate code failures, analyze rectification patterns, and compare the effectiveness of reasoning and non-reasoning models, offering actionable insights into both the understanding and practical application of feedback loops in LLM-driven code generation systems. Results show that reasoning models consistently improve over iterations, substantially outperforming non-reasoning models in leveraging feedback, while syntactic and runtime errors are far more tractable than logical or algorithmic failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17507v1">LLM-as-Judge in Education: A Curriculum-Grounded Marking Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Generative AI and large language models (LLMs) are increasingly applied to question generation and automated assessment. However, deploying LLMs in preparation for high-stakes exams requires more than prompt engineering; it demands software pipelines that systematically ground model outputs in authorised curriculum artefacts and marking guidelines issued by education authorities. This paper presents a curriculum-grounded, configurable LLM-as-Judge pipeline for question-level marking, co-developed with an industrial partner, to support exam preparation for university admission. The pipeline identifies the relevant topics, subtopics, and cognitive demand of a question, and assembles verifiable and authorised context to support LLM judgement. Curriculum intent is operationalised through concrete syllabus artefacts, including prescribed verbs and outcomes, performance band descriptors, glossary definitions, and marking-guideline principles. A staged LLM workflow is employed to first generate question-specific rubrics, capturing structured expectations of performance, and then derive and evaluate marking criteria used to allocate marks to student responses. This design improves consistency, transparency, and alignment with official marking practices. Preliminary evaluation shows that the proposed LLM-as-Judge pipeline delivers marking outcomes comparable to human tutors, while yielding justifications that are more traceable to authorised curriculum artefacts and marking standards. The pipeline has also been integrated into an online study platform, where early deployment data provide initial insights into operational usage and manual overrides.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17506v1">Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 20 pages, 13 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we find that our task evades safety guardrails by surfacing bias in model judgment. It varies systematically across target groups, reflects implicit social maps, and shows how models are still triggered by demographic labels. Our work points to the need for LLM bias evaluation in judgment tasks and broadly, for more theoretically grounded approaches to bias evaluation in NLP. We release our code and model responses at https://github.com/uofthcdslab/second-order-bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18003v2">BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 ACL 2026; Project Page at https://bad-scientist.github.io/
    </div>
    <details class="paper-abstract">
      The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17489v1">Online LLM Selection via Constrained Bandits with Time-Varying Demand</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 11 pages, 3 figures with multiple subfigures, 1 table, submitted for possible journal publication
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in edge-cloud inference systems to handle diverse user tasks with heterogeneous accuracy, latency, and cost profiles. Selecting the appropriate LLM for each incoming task is critical for ensuring service quality and efficient resource utilization. However, model heterogeneity, stochastic and unknown performance characteristics, and time-varying task demands make static selection strategies inadequate. Real-world deployments often impose hard resource budgets such as monetary expenditure limits, along with soft service-level requirements such as latency guarantees. These constraints introduce additional challenges for online decision-making. We formulate this problem as a constrained stochastic bandit learning task, where the learner sequentially selects models under both packing-type (hard) and covering-type (soft) constraints, while adapting to time-varying task demand. The learner operates without access to the underlying reward, cost, or latency distributions and must rely on partial feedback. We develop a novel online learning algorithm that leverages confidence-bound estimates and demand predictions to balance reward maximization with long-term constraint satisfaction. We provide theoretical guarantees showing sublinear regret and sublinear covering constraint violations compared to an offline benchmark with full information. Experimental results on synthetic workloads demonstrate the effectiveness and robustness of our approach in dynamic, resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04990v3">From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are evolving from passive text generators into autonomous systems capable of planning, tool use, retrieval, memory access, environmental interaction, and multi-agent collaboration. These capabilities expand agent autonomy, but also make agent behavior harder to verify, debug, and audit. Final-answer accuracy alone cannot explain how an output was produced, which evidence supported each claim, whether tool calls were justified, how memory influenced later decisions, or where failures originated. This survey examines evidence tracing and execution provenance as foundations for process-level accountability in trustworthy LLM agents. We define execution provenance as the typed graph of an agent execution and evidence tracing as its projection onto evidence-support relations. This perspective connects retrieval grounding, claim support, tool-use safety, memory lineage, observability, debugging, audit, and recovery within a unified framework. We introduce a taxonomy covering trace sources, evidence and execution units, provenance relations, tracing granularity and timing, representation forms, and trust functions. We then review key methodological directions, including provenance representation, evidence attribution, tool-use provenance, runtime guardrails, provenance-bearing memory, observability, and failure diagnosis. Finally, we discuss benchmarks, datasets, metrics, and open challenges for building provenance-aware, auditable, and recoverable agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17478v1">Decoding Hidden Deception in Reasoning LLMs: Activation Explainers for Deception Auditing</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      As LLMs acquire stronger reasoning capabilities, deceptive behavior becomes an increasingly serious safety concern. Existing deception monitors either score visible transcripts or derive scalar probe scores from representation vectors, leaving little inspectable evidence about why a response is suspicious. We introduce STATEWITNESS, an activation explainer for deception auditing. A separate decoder reads a target model's hidden states, then answers natural-language queries or emits structured reports about them. We evaluate STATEWITNESS on two target reasoning LLMs across seven deception datasets. STATEWITNESS reaches 0.916 mean AUROC, a relative gain of 11.6% over the best black-box text monitor and 25.0% over the best activation-probe baseline under the same evaluation protocol. When combined with existing monitors, STATEWITNESS reduces missed deceptive examples in simple threshold ensembles. Beyond scalar detection, the decoder returns query-level answers, schema reports, and token- or sentence-level evidence traces for human inspection. We view this interface as a potential building block for broader interpretability and alignment tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17461v1">AUTOGATE: Automated Clock Gating via Toggling-Aware LLM-based RTL Rewriting</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 9 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Fine-grain clock gating (FGCG) is among the most effective techniques for reducing dynamic power, yet current FGCG optimization flows remain largely manual. Recent LLM-based RTL optimization approaches remain limited by two key drawbacks: (1) the inability to process long waveform traces spanning millions of cycles, and (2) the difficulty of scaling optimization to large hierarchical codebases while preserving correctness. In this work, we present AUTOGATE, the first agentic framework for industry-grade RTL power optimization, enabling workload-aware clock-gating optimization across large hierarchical codebases. AUTOGATE introduces a Machine Learning (ML)-LLM co-design that bridges waveform-level analysis and RTL rewriting. Specifically, we design an ML-based clustering algorithm that distills raw toggling traces into compact, structured representations that guide LLM-based RTL rewriting. This enables accurate identification and application of clock-gating opportunities without requiring LLMs to directly process raw waveform data. To enhance scalability, AUTOGATE employs a hierarchical multi-agent architecture that decomposes large designs into independently optimizable modules, enabling coordinated optimization across deep design hierarchies. We evaluate AUTOGATE on a diverse set of designs ranging from small RTL designs to large industrial-grade codebases. Experimental results show that AUTOGATE consistently reduces dynamic power relative to baselines. Across the small-design suite, AUTOGATE reduces dynamic power by 49.31% on average. On industry-scale designs, it achieves 19.34% and 7.96% dynamic power reductions on NVDLA and BlackParrot, respectively, and up to 6.86% on highly optimized proprietary production designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17459v1">Can LLMs Be CEOs? Benchmarking Strategic Resource Reallocation with Multi-Role Agent Simulation</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Evaluating the decision-making capabilities of large language models (LLMs) is a growing research priority, yet existing benchmarks focus on isolated cognitive tasks such as reasoning, knowledge retrieval, and economic rationality in stylized settings. These evaluations overlook the defining challenge of real executive decision-making: integrating conflicting recommendations from specialized stakeholders under information asymmetry, organizational constraints, and temporal dependencies. We introduce \textsc{CEO-Bench}, a multi-agent benchmark that evaluates LLMs on CEO-level strategic resource reallocation -- the process of redirecting capital across business units in a multi-round, constraint-rich organizational environment. In \textsc{CEO-Bench}, LLM agents receive conflicting advice from four role-conditioned C-suite advisors (CFO, CTO, COO, CMO), each with private signals and distinct priorities, and must synthesize these into a concrete allocation plan evaluated along four dimensions: role integration, conditional boldness, history-sensitive judgment, and plan validity. Experiments across five frontier models on 13 scenarios reveal that all models achieve high structural validity but diverge sharply on strategic calibration -- the hardest capability layer. We identify systematic failure modes including single-advisor capture, conservative default under ambiguity, and historical amnesia, and uncover a structural integration-boldness tradeoff: models that engage more deeply with conflicting perspectives tend to produce less decisive action. These findings delineate the current capability boundary of LLMs as organizational decision-makers and inform the design of future AI-assisted executive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17443v1">Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 16 pages, 4 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming a major way for consumers to find products, but we do not yet understand how brands compete in this new channel. We study brand dynamics in LLM recommendations using skincare products -- a category where consumers cannot easily judge quality before buying and must rely on brand reputation -- across three commercial LLMs (GPT-4o-mini, Claude Sonnet, Gemini 3 Flash), with a robustness check on search goods. In three experiments, we find: (1) a Conditional Monopoly where well-known brands get recommended 100% of the time (IAI = 10.0) when all products have the same specifications, but this dominance disappears with less than a +0.1-star rating advantage for a competitor; (2) authority-style marketing language, including fabricated clinical-evidence claims, breaks this monopoly at a Bias Surplus Value equal to +0.17 rating points, with each model responding differently; and (3) a social dilemma in multi-brand GEO competition: when all brands adopt the same optimization strategy, individual payoff falls from +0.802 to +0.007 in our payoff proxy, and non-participating brands receive zero recommendations in our tests. Our results suggest that generative engine optimization (GEO) should be studied not only as a security risk, but also as an emerging marketing practice that shapes market competition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17421v1">Bifrost: Hybrid TEE-FHE Inference for Privacy-Preserving Transformer and LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Cloud-hosted transformer and large language model (LLM) inference creates a direct confidentiality problem: user prompts may contain sensitive code, business data, personal information, or regulated documents, yet remote serving exposes intermediate state to the cloud software stack and accelerator runtime. Fully homomorphic encryption (FHE) keeps accelerator-side execution ciphertext-only, but end-to-end LLM inference remains expensive because linear layers are interleaved with non-linear, cache-state, and refresh-sensitive operators. CPU trusted execution environments (TEEs) can execute those operators natively, but a CPU TEE alone does not define how an untrusted accelerator should participate. We present Bifrost, a hybrid TEE-FHE serving architecture in which secrets are provisioned only to an attested CPU TEE, while the accelerator, device memory, driver/runtime stack, and host software remain outside the trusted computing base. Bifrost uses FHE as a secure delegation mechanism for projection and feed-forward linear layers on accelerator-backed CKKS, while non-linear operators, attention-side control logic, KV-state transitions, and decrypt-then-encrypt refresh execute inside the CPU TEE. Bifrost+ further applies a prefill/decode split: prompt-side KV state is built inside the CPU TEE, and only decode-side state enters the hybrid ciphertext path. In an estimator-style comparison matching Euston's methodology, Bifrost reduces projected latency by 9.25x on GPT-2 (1.5B) and 9.91x on LLaMA 3 (8B). In direct CKKS/FHE deployments, Bifrost+ reduces TTFT by 14.6-45.8x on GPT-2 (124M) and 15.3-53.4x on Qwen3 (0.6B). The systems lesson is selective encrypted execution: use FHE only where ciphertext-only accelerator delegation is required, and keep non-linear, refresh, and prompt-side work inside the CPU TEE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03573v2">STRIDE: Post-Training LLMs to Reason and Refine Bio-Sequences via Edit Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-06-16
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Discrete biological sequence optimization often requires goal-directed, parser-valid edits to an existing protein or molecule. Diffusion models support iterative refinement but do not expose a controllable discrete-edit interface, while autoregressive LLMs can be myopic when planning constrained edits over multiple steps. We introduce STRIDE (Sequence Trajectory Refinement via Iterative Discrete Editing), a post-training framework that trains an LLM to emit executable INSERT/DELETE/REPLACE trajectories for variable-length refinement. STRIDE first learns Levenshtein-aligned shortest-edit demonstrations, then uses supervised fine-tuning and group-based policy optimization to align trajectories with task rewards while preserving coherent editing. On an oracle-based full-action protein stress test, STRIDE raises success over Vanilla SFT from 42% to 89% and novelty among unique improvements from 47% to 97%. On instruction-conditioned molecular editing, the GSPO-aligned variant improves strict success, controllability, and SMILES validity over the SFT-only STRIDE model (code: https://github.com/daiheng-zhang/STRIDE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19387v1">Interpretable and Verifiable Hardware Generation with LLM-Driven Stepwise Refinement</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in software development. However, they are susceptible to hallucinations, meaning that they can introduce subtle semantic and logical errors. Due to the high stakes in chip design and manufacturing, hardware engineers are still reluctant to rely on LLMs for register-transfer level (RTL) generation. In this paper, we propose a hardware generation framework that combines the creativity and broad knowledge of LLMs with the explainability and mathematical rigor of formal methods. Specifically, we devise a set of transformation rules that cover various design decisions and hardware features. By iteratively applying these rules, an LLM agent can convert a design specification into an RTL program with guaranteed correctness. Experimental results demonstrate the effectiveness and efficiency of the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17312v1">Quantifying Consistency in LLM Logical Reasoning via Structural Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 Published at ICLR 2026 Workshop on Logical Reasoning of Large Language Models. Accepted as best paper
    </div>
    <details class="paper-abstract">
      Large language models can arrive at the same answer through reasoning paths that are unstable, contradictory, or difficult to rank consistently -- a failure mode especially prevalent in multi-step deductive reasoning. Existing methods assess reliability primarily through output dispersion -- measuring how much sampled answers differ -- but this discards a complementary signal: whether the model can consistently rank competing reasoning candidates. We propose structural uncertainty, a consistency-aware framework derived from the stability of self-preference-induced rankings over sampled reasoning solutions. Given a query, we generate multiple candidate solutions and ask the model to judge pairwise preferences among its own outputs. We aggregate self-preferences into ranking distributions via Bradley-Terry modeling with PageRank, and decompose the signal into two entropy-based components: across-trial ranking instability and within-trial candidate ambiguity. Across five LLMs and eight benchmarks, structural signals provide information complementary to answer dispersion: on logical and mathematical reasoning tasks, the combination improves identification of unreliable instances, while on factual retrieval the structural signal collapses toward uniformity, diagnosing a regime boundary where reasoning-level consistency evaluation is uninformative. The two components relate differently to accuracy: within-trial ambiguity correlates positively with correctness -- consistent with settings where multiple plausible solution paths remain competitive -- while across-trial instability correlates negatively, signaling unreliable reasoning. Structural uncertainty is best understood not as a universal confidence estimator, but as a regime-sensitive evaluator of logical reasoning consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17309v1">Abstention-Aware Personalized Object Rearrangement via Uncertainty-Guided LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 Accepted at the 2026 IEEE 35th International Conference on Robot and Human Interactive Communication (RO-MAN 2026)
    </div>
    <details class="paper-abstract">
      Robotic assistance in household environments requires not only predicting where objects should be placed, but also reasoning about when objects should not be placed at all. Existing approaches to personalized object rearrangement primarily focus on placement decisions under the assumption of clean observations and complete actionability, limiting their applicability in realistic, cluttered, and partially erroneous settings. In this paper, we introduce APOLLO, a hybrid framework for abstention-aware personalized object rearrangement that combines a lightweight, personalized embedding model (PEM) with selective large language model (LLM) assistance. PEM is trained for each user-environment pair using a small number of demonstrations, operates entirely on CPU, and produces uncertainty estimates, which are used to selectively invoke LLM-based reasoning only for ambiguous decisions, balancing efficiency, privacy, and reasoning capability. To evaluate this formulation beyond existing benchmarks, we introduce APOR, a synthetic, LLM-generated dataset that captures room-level, multi-furniture environments, diverse organizational profiles, explicit abstention behavior, and noisy partial scene context. Extensive experiments on both PARSEC and APOR provide initial evidence that APOLLO improves over prior LLM-based baselines in controlled benchmark settings while substantially reducing LLM usage. Code is available at https://github.com/PaInt-Lab/APOLLO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17281v1">Are you speaking my languages? On spoken language adherence in multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 7 pages, 3 tables in the main body
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) based Automatic Speech Recognition (ASR) enables seamless multilingual use, models often misidentify the output language, compromising transcription fidelity and downstream application quality. To preserve flexibility and code-switching capabilities, we propose a soft prompting approach that hints at potential spoken languages without strictly constraining the output. We formally define this challenge as a lack of language adherence, introduce a novel metric to quantify violations, and evaluate three mitigation strategies: (1) zero-shot prompting for robust guidance under uncertainty, (2) supervised fine-tuning (SFT) to improve prompt adherence, and (3) Chain-of-Thought (CoT) reasoning to enforce adherence during decoding. We present a comparative analysis of these methods across multiple languages, evaluating effectiveness in reducing the language violation while maintaining overall ASR performance. Finally, we discuss trade-offs to guide strategy selection under various compute constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17279v1">Training LLMs with Reinforcement Learning over Digital Twin Representations for Reasoning-Intensive Surgical VideoQA</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Surgical video question answering requires multi-step reasoning across semantic, spatial, and temporal dimensions. Existing methods architecturally compress videos into discrete token representations and couple visual perception with reasoning. This approach fragments continuous spatial-temporal relationships and has been shown to restrict multi-step reasoning capabilities. We introduce a reinforcement learning (RL) framework that trains large language models (LLMs) to decouple perception from reasoning by operating over digital twin representations constructed from surgical foundation models. Additionally, we introduce hierarchical representations across frame, temporal window, and procedure levels with probabilistic uncertainty estimates. Finally, we propose a novel reward that combines format validation with accuracy assessment through clinical plausibility evaluation and uncertainty-aware calibration for training. To demonstrate the capabilities of this approach, we introduce REAL-Colon-Reason, a colonoscopic benchmark with 2000 question-answer pairs across three complexity levels. We achieve state-of-the-art performance on REAL-Colon-Reason and two existing surgical VideoQA benchmarks REAL-Colon-VQA and EndoVis18-VQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00215v2">Disentangling Perception and Reasoning in Multimodal LLMs via Reward Design</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 24 pages, 15 Figures, 10 Tables
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards has driven major gains in LLM reasoning, and it is intuitive to assume this recipe will transfer well to multimodal models. However, multimodal models do two things: first, perceive what is in an image, then reason about what it implies. Because these stages are graded jointly, it is hard to tell how much room reasoning alone has to grow. We study this on algorithmic visual puzzles, where both components are necessary and show that perception, not reasoning, is the binding constraint. Replacing images with simple textual descriptions raises performance by over 20 points on average for Claude models. We then evaluate six reward designs aimed at inducing visual grounding during reasoning without chain-of-thought supervision. Training Qwen-2.5-VL-7B with GRPO, reward design induces long, structured reasoning with self-reflection and visual references, yielding a 5.56-point gain over the base model. These gains are, however, uneven; no single reward improves all categories, and rewards with verifiable accuracy signals trade out-of-domain transfer for in-domain accuracy. These results point to perception-aware reward design as a path forward, so that signals correct perception at its source rather than the reasoning that inherits its errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18831v3">Adaptive Activation Steering for Efficient LLM Reasoning via Closed-Loop PID Control</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Reasoning LLMs trained with long chain-of-thought often overthink: they spend tokens on redundant reflection and transitions that inflate cost without improving accuracy. Static activation steering (e.g.\ SEAL) suppresses such content with a fixed vector, but applies the same strength regardless of how redundant the current chunk actually is. We describe PID-steering, a training-free, decoding-time method that modulates the steering strength with a PID controller driven by a lightweight chunk-level redundancy classifier. On a subset of GSM8K with DeepSeek-R1-Distill-Qwen-1.5B, the method improves accuracy from 85.7\% to 89.6\% (+3.9 pp) while cutting average output length from 1026 to 790 tokens ($-$23\%). We report it as a small-scale proof of concept rather than a benchmark result.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17253v1">PDAGENT-BENCH: Characterizing, Grounding, and Architecting LLM Agents for VLSI Physical Design</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Large Language Models and vision-language models have shown remarkable success in the front-end design of Very Large-Scale Integrated Circuits, yet their capabilities for VLSI physical design remain significantly underexplored. The primary cause is the lack of standardized benchmarks for evaluating agentic physical design workflows that require high-dimensional, multi-stage optimization under strict design constraints, coordinated interaction with diverse Electronic Design Automation tools, and iterative refinement. This work introduces PDAGENT-BENCH, a comprehensive and multi-dimensional benchmark for evaluating LLM/VLM-based agents across the physical design stack. PDAGENT-BENCH integrates both task-level assessment and workflow-level execution. The benchmark suite contains 353 curated problems that combine conceptual questions with real-world industrial artifacts, with expert-validated references and executable solutions. These tasks cover five key capability dimensions: foundational knowledge, report comprehension, root-cause analysis, script generation, and full-flow implementation. In addition, the benchmark provides a unified, human-aligned agentic physical design workflow framework that enables closed-loop evaluation of holistic physical design in realistic EDA environments. Experiments on 11 state-of-the-art models reveal that while modern LLMs/VLMs perform competitively on conceptual tasks, they remain substantially limited in tool-centric execution (e.g., 42.2% on Innovus script generation) and long-horizon, multi-stage reasoning. Our studies further show that human-skill-enhanced agentic workflows significantly improve end-to-end physical design performance. PDAGENT-BENCH establishes a standardized, reproducible, and realistic evaluation framework for advancing LLM/VLM-driven holistic physical design automation. We will open source the benchmark and framework soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17234v1">Speaking in Self-Assessing Tongues: On the Verbalized Confidence of LLMs in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      The rapid rise in popularity of large language models (LLMs) for translation calls for a thorough study of the reliability of their confidence in their own outputs. Unlike many generation tasks, translation errors and confidence levels can be useful at different levels of granularity (tokens, words, or spans). Unsupervised approaches based on internal signals like predicted probabilities can be misleading because they reflect certainty among alternatives rather than correctness. In addition, they require access to such internal signals. Here, we devise five verbalized methods of extracting an LLM's per-token confidence without those shortcomings and compare their reliability with that of the model's internal signals of certainty. We evaluate reliability using two forms of alignment: fine-grained error detection and calibration. For both, internal and verbalized methods perform similarly, although results vary by model. Interestingly, we find little to no correlation between internal and verbalized methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17213v1">Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Recent advances in multimodal learning, including large language models (LLMs) and vision-language models (VLMs), have demonstrated strong adaptability to natural images. However, extending their use to the medical domain, particularly for volumetric (3D) images, is challenging due to high computational complexity, volumetric dependencies and the semantic gap between visual features and clinical terminology. Naively fine-tuning LLMs on limited medical data often leads to overfitting and clinical hallucination, where linguistic fluency is prioritized over clinical factuality. In this study, we investigate parameter-efficient adaptation strategies for volumetric CT report generation and introduce RAD3D-Prefix, a lightweight diagnostic-prior conditioning framework that minimizes the need for extensive parameter training. This module integrates image embeddings with multi-label diagnostic classification logits, preserving critical clinical details while bridging the semantic gap. By keeping the LLM frozen, our method requires minimal trainable parameters and mitigates the risk of overfitting on small, domain-specific datasets. Through a systematic study spanning LLMs from 96.1M to 1.6B parameters, we find that fine-tuning is most beneficial for smaller LLMs, whereas freezing larger (~1B+ LLMs and training only lightweight projection layers provides a superior trade-off between performance, generalization, and computational efficiency. Across multiple automatic metrics and a clinical reader study, RAD3D-Prefix outperforms comparable parameter-efficient baselines and demonstrates strong out-of-domain generalization while using substantially fewer trainable parameters than fully fine-tuned alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16407v4">Jacobian Scopes: token-level causal attributions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 25 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. Grounded in perturbation theory and information geometry, Jacobian Scopes quantify how input tokens influence various aspects of a model's prediction, such as specific logits, the full predictive distribution, and model uncertainty (effective temperature). Through case studies spanning instruction understanding, translation, and in-context learning (ICL), we demonstrate how Jacobian Scopes reveal implicit political biases, uncover word- and phrase-level translation strategies, and shed light on recently debated mechanisms underlying in-context time-series forecasting. To facilitate exploration of Jacobian Scopes on custom text, we open-source our implementations and provide a cloud-hosted interactive demo at https://huggingface.co/spaces/Typony/JacobianScopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17168v1">RepSelect: Robust LLM Unlearning via Representation Selectivity</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Making large language models (LLMs) deeply forget specific knowledge and values without sacrificing general capabilities remains a central challenge in unlearning. However, current methods are easily reversed by fine-tuning or few-shot prompting, suggesting their forgetting is only shallow. We identify the root cause. Existing methods target representations shared with both the retain set and the subspace recovered by a fine-tuning attacker, making unlearning both disruptive to general capabilities and easy to reverse. We propose RepSelect (Representation Selectivity), isolates forget-set-specific representations by collapsing top principal components of weight gradients before each update, leaving general capabilities intact while limiting what fine-tuning can recover. We evaluate across two forget categories, biohazardous knowledge and abusive tendencies, and four model families spanning dense and Mixture-of-Experts architectures (Llama 3, Qwen 3.5, Gemma 4 E4B, DeepSeek V2 Lite). Compared to five popular baselines (GradDiff, NPO, SimNPO, RMU, UNDIAL), RepSelect achieves a 4-50x larger reduction in post-relearning answer accuracy than the strongest baseline, and is near-perfectly robust to few-shot prompting attacks. Targeting selective representations is thus an important step towards deep and robust LLM forgetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17053v1">Context-Aware RL for Agentic and Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 29 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail when answering requires identifying a small but decisive piece of evidence within a long or complex context, such as a single line in a tool trace or a subtle detail in an image. We propose ContextRL, a context-aware reinforcement learning (RL) method that improves long-horizon reasoning and multimodal performance through an \emph{indirect} auxiliary objective. Instead of supervising only the final answer, ContextRL presents the model with a query, an answer, and two highly similar contexts, and rewards it for selecting the context that supports the query--answer pair, thereby encouraging fine-grained grounding. We construct contrastive context data in two domains: for coding agents, trajectories serve as contexts, yielding 1k pairs built via condition filtering; for multimodal reasoning, images serve as contexts, yielding 7K pairs built via generative editing and similarity search. ContextRL achieves average gains of +2.2% over standard GRPO on 5 long-horizon benchmarks, and +1.8% across 12 diverse visual question answering benchmarks. To disentangle the effect of the proposed objective from that of additional data, we compare against data-augmentation baselines that repurpose the same contrastive contexts as standard query--context--answer examples. These baselines provide little to no improvement, showing that the gains arise from the proposed context-selection objective rather than from the contrastive data alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17024v1">ExpRL: Exploratory RL for LLM Mid-Training</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Sparse reward reinforcement learning (RL) has become a standard tool for improving LLM reasoning, but its success depends critically on the coverage present in the base model. In practice, models are often primed for RL through \emph{mid-training} on curated reasoning traces that teach useful primitive skills such as decomposition, verification, or self-correction. Although effective, this strategy requires manually specifying what the model should learn, and it remains unclear whether such primitive coverage is enough for much harder problems, which require combining these skills into broader solution strategies. We study a more automated approach: \emph{RL-based mid-training} using large corpora of human-written question-answer data. Rather than treating reference solutions as targets to imitate, our method, ExpRL, uses them as \emph{reward scaffolds}: references are hidden from the policy and used only to construct problem-specific grading rubrics for judging on-policy reasoning traces. The policy samples from the original problem prompt, while an LLM judge compares the sampled reasoning trace against the reference solution and assigns outcome-level or process-level dense rewards. This lets ExpRL reinforce partial progress, useful intermediate reductions, and productive reasoning behaviors that sparse final-answer rewards often fail to upweight. On challenging math reasoning tasks, ExpRL yields stronger RL priming than SFT, sparse-reward GRPO, and self-distillation, and provides a better initialization for subsequent sparse-reward RL. Additional mixed-domain experiments further suggest that ExpRL can extend beyond the original math-only setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17016v1">TokenPilot: Cache-Efficient Context Management for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 LightMem Series: Work in Progress
    </div>
    <details class="paper-abstract">
      As LLM agents are deployed in long-horizon sessions, context accumulation drives up inference costs. Existing approaches utilize text pruning or dynamic memory eviction to minimize token footprints; however, their unconstrained sequence mutations alter layouts, introducing prefix mismatches and cache invalidation. This reveals a critical trade-off between text sparsity and prompt cache continuity. To address this, we present TokenPilot, a dual-granularity context management framework. Globally, Ingestion-Aware Compaction acts as a framework harness to stabilize prompt prefixes and eliminate open-world environmental noise at the ingestion gate. Locally, Lifecycle-Aware Eviction monitors the ongoing residual utility of context segments, enforcing a conservative batch-turn schedule to offload content segments only when task relevance expires. Experiments on PinchBench and Claw-Eval under both isolated and continuous modes demonstrate that TokenPilot reduces costs by 61% and 56% in isolated mode, and 61% and 87% in continuous mode, while maintaining competitive performance compared to prior systems. TokenPilot has been integrated into LightMem2 at https://github.com/zjunlp/LightMem2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21027v2">Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 Accepted to the Enterprise AI Agents Workshop @ KDD 2026. The first four authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02493v3">Not What, But How: A Framework for Auditing LLM Responses across Positioning, Generalization, Anthropomorphism, and Maxims</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 34 pages, 19 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly used to answer subjective, information-seeking questions, where users are sensitive to how responses are communicated, not just whether the answers are correct. Existing LLM evaluations for subjective cultural queries largely focus on factual correctness, ignoring how the response is framed. To this end, we introduce FRANZ, an automated FRAmework for respoNse characteriZation to conduct communicative audit of LLM responses along four dimensions: cultural positioning, use of generalizing language, anthropomorphic cues, and adherence to conversational maxims. To enable this evaluation, we contribute SQUARE - a corpus of 376k subjective questions sourced from 57 subreddits, and mapped to 7 countries and 19 question categories. We demonstrate FRANZ's applicability by scoring responses from three open-weight LLMs. We observe that LLMs show statistically significant differences in the frequency with which they employ each response characteristic. Unlike single-dimensional audits, FRANZ reveals that insider positioning and anthropomorphism are positively coupled, with the degree of coupling varying by country, providing a diagnostic lens for identifying framing divergences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01410v2">What LLMs Must Forget to Teach Effectively: A DIY Approach to Premodern Japanese Language Pedagogy</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      We discuss a novel approach to Premodern Japanese Language Pedagogy (PJLP) with potential applications in other languages and fields. The integration of artificial intelligence into education has largely operated as a top-down project, affording minimal agency to everyday users. This dynamic mirrors the broader frontier model ecosystem, which concentrates massive human and financial resources within a few labs. Drawing inspiration from grassroots initiatives such as the DIY and Maker movements, this paper advocates for an approach to AI in Education that fosters instructional and student agency over the pedagogical process. Specifically, we discuss a tutoring framework for textual analysis in the context of a graduate seminar in premodern Japanese literature, as well as a bilingual interactive dictionary and a conversational partner created for a language course in Classical Japanese. Created through prompt engineering as custom instances of a Large Language Model (LLM), these three tools are designed to counteract the tendency of out-of-the-box LLMs to either bypass student effort through over-explanation or misguide learners via hallucinations. To illustrate how this approach can promote active comprehension and pedagogical alignment, we provide transcripts (logs) of actual exchanges, sample instructions (system prompts), and guidance for instructors curious about exploring this approach in a variety of fields (starter kit).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09537v2">STEPS: Semantic Contract-Guided Scheduling for LLM-Assisted Natural Language-Driven Edge AI Services</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Edge user/service scheduling has become a cornerstone of distributed AI systems, determining where and how AI services are executed under limited communication and computing resources. Existing edge scheduling frameworks usually assume that service requirements are given as numerical constraints, such as latency bounds or energy budgets. In practice, users often express service expectations through ambiguous and context-dependent natural language, creating a gap between user intent and scheduling decisions. To bridge this semantic-to-optimization gap, we propose semantic contract-guided edge potential scheduling (STEPS), a natural language-driven scheduling framework that introduces semantic contracts as executable interfaces between user-side semantics and edge-side decision making. In STEPS, a large language model (LLM)-assisted parser interprets natural language requests and extracts semantic service requirements with confidence scores, which are converted into service requirements and semantic uncertainty. Based on this information, STEPS formulates edge scheduling as a contract-guided potential game that jointly determines execution-node selection, computing-resource provisioning, and bandwidth allocation. STEPS further uses feedback signals to support adaptive scheduling under evolving service and network conditions. We characterize the exact potential game structure, establish the existence of a pure-strategy Nash equilibrium, and prove convergence and stability properties of the scheduling and adaptation processes. Extensive experiments show that STEPS improves semantic contract fulfillment, reduces contract-guided service loss, and maintains robust adaptation under ambiguous natural language requests in non-stationary networked AI environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16920v1">Demystifying Variance in Circuit Discovery of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Circuit discovery is a key technique in mechanistic interpretability to pinpoint the model components that are crucial for performing a given task. Although the current state-of-the-art method (EAP-IG) performs well on the metric of (un)faithfulness, it suffers from substantial variability. This includes resampling variance, where the circuit changes when we probe with a new batch of data from the same distribution; rephrasing variance, where the discovered circuit shifts when the prompts are rephrased; and sample-wise variance, where a circuit with low population unfaithfulness exhibits large fluctuations in unfaithfulness across individual samples. This paper studies the roots of these variances. We demonstrate that CEAP, our new circuit discovery method that improves upon EAP-IG with a theoretical guarantee, can substantially lessen resampling variance. We further show that rephrasing variance arises because prompts with different templates tend to activate different circuits in the model. This leads us to argue that it may be challenging to find a comprehensive circuit that explains and controls the model's behavior on a task, which can be expressed in countless templates, suggesting that LLMs may be inherently hard to steer. We show that sparsity, which has been claimed to form more compact and interpretable task circuits, fails to solve this problem. Regarding sample-wise variance, we argue that it is largely benign: extremely poor unfaithfulness scores often stem from how unfaithfulness is defined, rather than from defects in the measured circuits. We show that the magnitude of unfaithfulness is affected by selective contribution scaling, a neural mechanism that accounts for the extremely poor scores sometimes observed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16907v1">Tangram: Hiding GPU Heterogeneity for Efficient LLM Parallelization</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      The scale of LLM training jobs requires parallelization planning over large GPU clusters. Due to different GPU types and interconnects added over time, these GPU clusters are increasingly heterogeneous. Automatic LLM parallelizers can search for parallelization plans but face an exploding search space with heterogeneous GPUs. To make search tractable in heterogeneous GPU clusters, parallelizers often omit types of parallelism (e.g., expert parallelism) or memory-saving techniques (e.g., ZeRO), which results in worse plans. We describe Tangram, a system that enables the use of existing heterogeneity-unaware LLM parallelizers in heterogeneous GPU clusters by decoupling parallelization planning from GPU heterogeneity. For this, Tangram exploits two insights: (1) since bulk purchases result in sets of GPUs with similar compute, memory, and connectivity, Tangram can expose such homogeneous GPU islands to existing parallelizers; and (2) parallelizers commonly first partition models and then parallelize partitions. Tangram can compose such model slices, assigned to GPU islands, into work-balanced pipelines for high throughput. Tangram integrates with existing parallelizers through a narrow API, which relies on the enumeration of model-slice/island pairs. Tangram achieves up to 2.3x higher training throughput than current heterogeneous parallelizers (Metis and Sailor) and scales to large GPU clusters by pruning enumerated plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10574v3">LLM Jaggedness Unlocks Scientific Creativity</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      As artificial intelligence advances, models are not improving uniformly. Instead, progress unfolds in a jagged fashion, with capabilities growing unevenly across tasks, domains, and model scales. In this work, we examine this dynamic jaggedness through the lens of scientific idea generation. We introduce SciAidanBench, a benchmark of open-ended scientific questions designed to measure the scientific creativity of large language models (LLMs). Given a scientific question, models are asked to generate as many unique and coherent ideas as possible, with the total number of valid responses serving as a proxy for creative potential. Evaluating 19 base models across 8 providers (30 total variants including reasoning versions), we find that jaggedness manifests both across models and within models. First, in a cross-task comparison between general and scientific creativity, improvements in general creativity do not translate uniformly to scientific creativity, revealing divergent capability profiles across models. Second, at the prompt level, stronger models do not improve uniformly; instead, they exhibit high variability, with bursts of creativity on some questions and limited performance on others. Third, at the domain level, individual models display uneven strengths across scientific subfields, reflecting fragmented internal capability profiles. Finally, we show that this jaggedness can be harnessed. We explore mechanisms of inference-time compute, knowledge pooling, and brainstorming to combine models effectively and construct meta-model ensembles that outperform any single model. Our results position jaggedness not as a limitation, but as a resource, a structural feature of AI progress that, when understood and leveraged, can amplify LLM-driven scientific creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13681v1">EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align their knowledge, skills, and behavior with changing environments and updated task conditions. To address this gap, we introduce EvoArena, a benchmark suite that models environment changes as sequences of progressive updates across terminal, software, and social domains. We further propose EvoMem, a patch-based memory paradigm that records memory evolution as structured update histories, enabling agents to reason about environmental evolution through changes in their memory. Experiments show that current agents struggle on EvoArena, achieving an average accuracy of 39.6% across evolving terminal, software, and social-preference domains. EvoMem consistently improves performance, yielding an average gain of 1.5% on EvoArena and also improving standard benchmarks such as GAIA and LoCoMo by 6.1% and 4.8%. Beyond individual tasks, EvoMem further improves chain-level accuracy by 3.7% on EvoArena, where success requires completing a consecutive sequence of related evolutionary subtasks. Mechanistic analysis shows that EvoMem improves evidence capture in the memory, indicating better preservation of complete evolving environment states. Our results highlight the importance of modeling evolution in both evaluation and memory for reliable agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13649v1">Operadic consistency: a label-free signal for compositional reasoning failures in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Detecting LLM reasoning failures at inference time without ground-truth labels has motivated a wide range of confidence baselines, including self-consistency, semantic entropy, and P(True), built on within-question sampling and self-evaluation. Operad theory, the formalism for systems built by iterated substitution, suggests a complementary diagnostic: a model's direct answer to a compositional query should agree with the answer it produces by composing a stated decomposition of the same query. We instantiate this idea as operadic consistency (OC), a per-question signal. Across twelve instruction-tuned LLMs (4B to 671B parameters, open-weights and closed-source) on four multi-hop QA datasets, OC is strongly correlated with accuracy on every dataset (Pearson $r \in [0.86, 0.94]$, all $p \leq 0.0004$), and is the only signal we evaluate with $r \geq 0.85$ uniformly across all four datasets. Chain-of-thought self-consistency (CoT-SC; Wang et al., 2023) matches OC on HotpotQA and DROP ($r = 0.93, 0.87$) but drops to $r \approx 0.45$ on MuSiQue and StrategyQA. At the per-question level, OC contributes information beyond CoT-SC and semantic entropy on every dataset (cluster-robust $p \leq 10^{-16}$ for the OC coefficient), and the conclusion is robust to additionally controlling for constructed decomposition-aware baselines ($p \leq 10^{-13}$). The same signal yields selective-prediction improvements (accuracy at fixed coverage) over a tuned CoT-SC baseline at the equal-cost $K = 3$ budget (AUARC lifts of +0.086 to +0.096 and AUROC lifts of +0.092 to +0.164; 95% CIs exclude zero on every cell). On five frontier thinking models, where the decomposition is extracted from the model's own chain of thought, the same equal-cost comparison gives positive selective-prediction point-estimate lift on all 16 (dataset, budget, metric) cells tested, with 95% CIs excluding zero on 12 of the 16.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13634v1">Operads for compositional reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Question decomposition, i.e. breaking a complex query into simpler sub-queries whose answers are composed to produce a final answer, is a widely used strategy for improving LLM reasoning, yet it currently lacks a rigorous mathematical foundation. In this paper, we propose operads, mathematical structures that model many-in, one-out operations and compositions thereof, as a natural framework for describing question decomposition. We define the questions operad $Q$, in which operations correspond to question templates and composition corresponds to substitution of sub-answers, and show how QA models can be interpreted as algebras over $Q$. Beyond reframing existing practice, this operadic perspective points toward new methods, in particular a notion of operadic consistency, which measures whether a QA model's answers agree across the partial collapses of a question decomposition tree. Empirical evaluation of operadic consistency is reported in our companion paper (Bottman, Liu, and Richardson, 2026), which finds it strongly correlated with accuracy across twelve LLMs and four multi-hop QA datasets and outperforming standard temperature-based self-consistency baselines. We argue that operads are the natural mathematical home for question decomposition, and that invariants such as operadic consistency open new directions for analyzing and improving the reliability of multi-step reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13607v1">Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 13 pages main text, 51 pages supplementary text
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) fail to generalize or make haphazard errors in reasoning, it is often taken as evidence that LLMs are not truly reasoning, but rather performing a kind of pattern matching. The implication is that people's behavior does not exhibit the same types of failures because human reasoning uses principled and abstract world models. We evaluate human participants and 25 LLMs on their ability to engage in common-sense reasoning about a variety of everyday situations and observe similar patterns of errors in both people and models. We then identify the set of attention heads driving LLM responses and find that these heads implement a form of pattern-matching. These attention heads allow us to predict seemingly inexplicable reasoning errors in people caused by ostensibly irrelevant prompt details. Taken together, our results suggest that everyday causal reasoning in people and LLMs is more consistent with a form of pattern-matching than with abstract world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31514v3">If LLMs Have Human-Like Attributes, Then So Does Age of Empires II</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Fixed corollary 1, added stat sig
    </div>
    <details class="paper-abstract">
      Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain invariant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion on these attributes requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions. This is regardless of the experimenter's viewpoint on the subject, or whether the outcome shows existence or non-existence. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that Age of Empires II is functionally- and Turing-complete.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07515v2">How reliable are LLMs when it comes to playing dice?</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      We investigate the probabilistic reasoning capabilities of large language models through a controlled benchmarking study on discrete probability problems. We constructed two datasets, respectively a set of standard exercises and a set of counterintuitive exercises, designed to trigger heuristic reasoning, and evaluated 8 state-of-the-art models, each tested with and without Chain-of-Thought prompting. Models achieve an average accuracy of 0.96 on standard problems but only 0.59 on counterintuitive ones. We further provide empirical evidence of token bias: performance drops by over 20% when canonical formulations are replaced by disguised variants. Embedding misleading suggestions in the prompt reduces performance by up to 34%, with no model proving immune. Taken together, the reported findings suggest that current LLMs are not yet genuine probabilistic reasoners, despite their success in advanced mathematical problems.
    </details>
</div>
