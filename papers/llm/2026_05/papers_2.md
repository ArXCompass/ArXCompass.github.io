# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15949v5">ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24472v3">Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Code is available at https://github.com/beanie00/self-distillation-analysis
    </div>
    <details class="paper-abstract">
      Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-1.7B/8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21560v1">AutoMCU: Feasibility-First MCU Neural Network Customization via LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Deploying neural networks on microcontroller units (MCUs) is critical for edge intelligence but remains challenging due to tight memory, storage, and computation constraints. Existing approaches, such as model compression and hardware-aware neural architecture search (HW-NAS), often depend on proxy metrics, incur high search cost, and do not fully bridge the gap between architecture design and verified deployment. This paper presents AutoMCU, a feasibility-first large language model (LLM)-based multi-agent system for automated neural network customization under MCU constraints. Given natural-language task requirements and hardware specifications, AutoMCU iteratively generates structured architecture candidates, filters infeasible designs through vendor toolchain feedback before training, evaluates feasible models under a controlled protocol, and verifies deployability through backend-grounded deployment analysis. AutoMCU includes two key mechanisms: 1) hardware-in-the-loop architecture generation for early elimination of undeployable candidates under RAM and Flash constraints, and 2) state-isolated multi-agent scheduling for stable coordination of proposal, training, evaluation, and deployment stages. Experiments on CIFAR-10 and CIFAR-100 under strict MCU constraints show that AutoMCU achieves competitive accuracy while reducing customization time to about 1--2 hours, compared with hundreds of GPU hours for representative MCU-oriented HW-NAS baselines. Comparisons with ColabNAS and the LLM-based NAS method GENIUS on NAS-Bench-201 further demonstrate the effectiveness and stability of AutoMCU. Real-device deployments on multiple STM32 microcontrollers validate its practical applicability to MCU-scale edge intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21240v1">APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM agents have shown strong performance across a wide range of complex tasks, including interactive environments that require long-horizon decision making. But these agents cannot learn on the fly at test time. Self-evolving agents address this by accumulating memory and reflection across episodes rather than requiring model-weight updates. However, these agents often suffer from exploration collapse: as memory grows, behavior concentrates around familiar high-reward routines, reducing the chance of discovering better alternatives. To address this problem, we propose Autonomous Policy EXploration (APEX), which builds and maintains an explicit strategy space through a strategy map-a directed acyclic graph of milestones with prerequisite dependency edges. In APEX, Fork Discovery expands the map with evidence-grounded unexplored directions, while Policy Selection balances exploration and exploitation during planning. Evaluated on nine Jericho text-adventure games and WebArena, a realistic web interaction benchmark, APEX outperforms all baselines. Extensive ablations validate each component's contribution and demonstrate robustness across diverse settings, demonstrating APEX's effectiveness for sustained exploration in self-evolving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21558v1">From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted@ICML26, 28 pages, 11 figures, 26 tables
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to specialized domains typically incurs high data and computational overhead. While prior efficiency efforts have largely treated data selection and parameter-efficient fine-tuning as isolated processes, our empirical analysis suggests they may be intrinsically coupled. We posit the Strong Map Hypothesis: a sparse subset of attention heads plays a dominant role in task-specific adaptation, acting as keys that unlock specific data patterns. Building on this observation, we propose From Parameters to Data (P2D), a unified framework that leverages these task-sensitive attention heads as a dual compass for both sample mining and structural pruning. To rigorously quantify the total pipeline cost, we introduce the Alignment Efficiency Ratio (AER) metric for both selection latency and training time. Mechanistically, P2D identifies critical heads via a lightweight proxy and uses them as a functional filter to curate high-affinity data, establishing a synergistic pipeline. Empirically, by updating merely 10% of attention heads on 10% of the data, P2D achieves an 8.3 pp performance gain over strong baselines and delivers a 7.0x end-to-end time speedup. These results validate that precise parameter-data synchronization eliminates redundancy, offering a new paradigm for efficient alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21227v1">Do LLMs Know What Luxembourgish Borrows? Probing Lexical Neology in Low-Resource Multilingual Models</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to Neollm colocated with LREC2026, Three figures and three tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for writing assistance in small contact languages, yet it is unclear whether they respect community norms around lexical borrowing and neology. We introduce LexNeo-Bench, a 3{,}050-instance token-level benchmark derived from LuxBorrow, a large-scale Luxembourgish news corpus, where target tokens are labelled as native or as French, German, or English borrowings. Using this benchmark, we probe three multilingual LLMs across 34 prompt settings on two tasks: borrowing type classification and a binary lexical-innovation proxy (borrowing versus native). Without external context, models perform only slightly above chance on borrowing classification, so we construct a linguistic knowledge graph that encodes donor language, morphological patterns, and lexical analogues, and inject instance-specific subgraphs into the prompt. Knowledge-graph prompts raise borrowing classification accuracy from 25 -- 35\% up to 71 -- 81\% and largely close the gap between small and large models, while leaving neology detection difficult and sensitive to few-shot design. Our results show that lexicon-aware prompting is highly beneficial for robust borrowing judgments in low-resource contact languages and that lexical resources can serve as structured context for LLM evaluation. This study was carried out within the ENEOLI COST Action and examines borrowing as a form of lexical innovation in multilingual Luxembourgish data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21217v1">Federated LoRA Fine-Tuning for LLMs via Collaborative Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Low-rank adaptation (LoRA) has emerged as a powerful tool for parameter-efficient fine-tuning of large language models (LLMs). This paper studies LoRA under a federated learning setting, enabling collaborative fine-tuning across clients while preserving parameter efficiency. We focus on a highly heterogeneous regime in which clients share only partial structure and a substantial subset may be contaminated. We propose Collaborative Low-rank Alignment and Identifiable Recovery (CLAIR), a contamination-aware framework that relies only on preliminary local estimators. Its formulation applies broadly, from linear regression to neural network and LLM modules, whenever local adaptation can be represented by matrix-valued updates. CLAIR recovers the shared LoRA subspace and detects contaminated clients via a structured low-rank plus block-sparse decomposition. We prove exact recovery of the shared LoRA subspace in the noiseless case, stable recovery under preliminary estimation error, and consistent collaborative-set recovery under mild separation conditions. We further quantify the gain from CLAIR refinement: it reduces off-subspace estimation error through cross-client averaging while preserving client-specific variation within the shared LoRA subspace, thus improves over local fine-tuning whenever this oracle gain outweighs the costs of subspace estimation and benign-client heterogeneity. Empirically, we demonstrate the benefits of CLAIR by fine-tuning a Transformer architecture on a text-copying task. The results show accurate contamination detection and improved benign-client performance compared with local fine-tuning and non-robust federated averaging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12120v3">LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 ICML 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance and generalization. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data determines the scaling trend. In contrast, model size, optimization hyperparameters, tokenizer and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, generally have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2305.12138v5">Exploring Code Analysis: Zero-Shot Insights on Syntax and Semantics with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Code analysis is fundamental in Software Engineering, supporting debugging, optimization, and security assessment. Human developers approach it through syntax parsing, static semantics inference, and dynamic reasoning. Traditional tools are effective but limited by language specificity and weak cross-language generalization. Large language models (LLMs) are promising for code tasks, yet their capabilities for fundamental code analysis remain underexplored. We structure our study around three aspects aligned with human practices: syntax parsing, static semantics inference, and dynamic reasoning. We evaluate 21 state-of-the-art LLMs across nine tasks in four languages (C, Java, Python, Solidity), including AST generation, CFG construction, data dependency, taint analysis, and flaky test reasoning. We apply a three-layer evaluation protocol (automated metrics, expert adjudication, consistency validation) to 3,124 code samples, achieving high inter-rater reliability (Cohen's kappa = 0.844-0.936) and strong human-machine agreement (Gwet's AC1 = 0.500-0.727, F1 = 0.791-0.882). While the best LLMs excel in syntax parsing (AST 90%+, expression matching 84-100%) and show promise in static analysis, their dynamic reasoning remains limited (<70%) with high data-shift sensitivity (per-project F1 varying 0-1.0). This hierarchy holds across model families and scales, suggesting fundamental rather than transient limitations. These findings show how LLMs complement traditional analyzers: they offer cross-language generalization but non-deterministic outputs needing validation, while traditional tools give deterministic guarantees but need language-specific configuration. We contribute a validated evaluation framework with comparison against traditional analyzers (Tree-sitter, Soot, Joern) and task-specific applicability tiers. Benchmark: https://github.com/mathieu0905/llm_code_analysis.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14896v2">DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 2 figures, 2 tables. The revised version includes McNemar's paired statistical analysis, Wilson confidence intervals, expanded methodological clarifications, a revised discussion of evidence retrieval, improved reproducibility details, and updated limitations
    </div>
    <details class="paper-abstract">
      In our study, we evaluated large language model (LLM) performance on pharmacy licensure-style question-answering tasks and developed an external knowledge integration method to improve accuracy. We benchmarked ten LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset, measuring baseline accuracy without modification. Baseline performance ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores, while smaller open-source models showed substantially lower performance. We then developed DrugRAG, a three-step retrieval-augmented generation (RAG) pipeline that retrieves structured, evidence-based drug information and augments model prompts with contextual pharmacological evidence, operating externally and requiring no changes to model architecture or parameters. DrugRAG increased accuracy across all five evaluated models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61.0% to 71%, Llama 3.1 8B: 46% to 67%). McNemar analyses demonstrated statistically significant paired improvements primarily in smaller and mid-sized open-source models. These findings demonstrate that integrating structured external drug knowledge via DrugRAG can improve LLM performance on pharmacy-focused question-answering tasks without modifying the underlying models, providing a practical pipeline for enhancing evidence-based pharmacy-focused AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07926v2">AgentEscapeBench: Evaluating Out-of-Domain Tool-Grounded Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      As LLM-based agents increasingly rely on external tools, it is important to evaluate their ability to sustain tool-grounded reasoning beyond familiar workflows and short-range interactions. We introduce AgentEscapeBench, an escape-room-style benchmark that tests whether agents can infer, execute, and revise novel tool-use procedures under explicit long-range dependency constraints. Each task defines a directed acyclic dependency graph over tools and items, requiring agents to invoke real external functions, track hidden state revealed incrementally, propagate intermediate results, and submit a deterministically verifiable final answer. AgentEscapeBench includes 270 instances across five difficulty tiers and supports fully automated evaluation. Experiments with sixteen LLM agents and human participants show that performance drops sharply as dependency depth increases: humans decline from 98.3% success at difficulty-5 to 80.0% at difficulty-25, while the best model drops from 90.0% to 60.0%. Trajectory analysis attributes model failures mainly to breakdowns in long-range state tracking, clue adherence, and intermediate-result propagation. These findings suggest that current agents can often handle local tool use but still struggle with deep contextual dependencies. We hope AgentEscapeBench can serve as a diagnostic testbed for measuring current agent capabilities and informing future training efforts toward more robust general-purpose reasoning, action, and adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17631v4">Time-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted at IJCNN 2026
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting, but this progress is still met with skepticism about whether LLMs are truly useful for this task. To address this, we propose Time-Prompt, a framework for activating LLMs for time series forecasting. Specifically, we first construct a unified prompt paradigm with learnable soft prompts to guide the LLM's behavior and textualized hard prompts to enhance the time series representations. Second, to enhance LLM' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve fusion of temporal and textual data. Finally, we efficiently fine-tune the LLM's parameters using time series data. Furthermore, we focus on carbon emissions, aiming to provide a modest contribution to global carbon neutrality. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that Time-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21082v1">AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted in ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based agents have demonstrated proficiency in multi-step interactions with graphical user interfaces (GUIs). While most research focuses on improving single-task performance, practical scenarios often involve repetitive GUI tasks for which invoking LLM reasoning repeatedly, i.e., the ReAct paradigm, is inefficient. Prior to LLMs, traditional Robotic Process Automation (RPA) offers runtime efficiency but demands significant manual effort to develop and maintain. To bridge this gap, we propose AutoRPA, a framework that automatically distills the decision logic of ReAct-style agents into robust RPA functions. AutoRPA introduces two core innovations: (1) A translator-builder pipeline, where a translator agent converts hard-coded ReAct actions into soft-coded procedures, and a builder agent synthesizes robust RPA functions via retrieval-augmented generation over multiple trajectories; (2) A hybrid repair strategy during code verification, combining RPA execution with ReAct-based fallback for iterative refinement. Experiments across multiple GUI environments demonstrate that RPA functions generated by AutoRPA successfully solve similar tasks while reducing token usage by 82% to 96%, significantly improving runtime efficiency and reusability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21063v1">APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Typical LLM responses tend to follow a default style, even though users often have distinct preferences regarding tone, verbosity, and formality that they do not explicitly state in their prompts. Evaluating whether personalization methods can adapt to these implicit preferences is challenging, since users typically provide prompts rather than reference responses, style preferences are not factually verifiable, and reference-free LLM judges may conflate personalization with general response quality. To address these challenges, we introduce the Arbitrary Preference Mapping (APM) benchmark, which decouples user attributes (e.g. enthusiastic) from response principles (e.g. persuasive) via a hidden, randomized mapping $\mathbf{C}$ that maps user attributes to preferences about response traits. Because $\mathbf{C}$ carries no semantic content and is resampled across runs, models cannot exploit stereotypical associations and must infer preferences from conversation history. Using this unbiased evaluation methodology, we adapt retrieval-augmented, prompt-optimization, and routing personalization methods and evaluate them on Llama-3.1-8B and Qwen-3.5-27B. Our results show that routing is the most reliable approach, while RAG only improves with the stronger base LLM, and soft prompt optimization fails to improve significantly over a non-personalized baseline. Our extensive evaluation reveals that in this realistic setting, personalization remains challenging, but our adapted methods show promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21049v1">Cross-lingual robustness of LLM-brain alignment and its computational roots</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) reliably predict neural activity during language comprehension and transformer depth has been interpreted as mirroring hierarchical cortical organization. However, it remains unclear whether such alignment extends to subcortical regions, overlaps spatially across languages, and what the computational roots of such alignment are. Here, we used a multilingual, whole-brain encoding framework to examine brain-LLM alignment across three typologically distinct languages: Mandarin, English, and French during naturalistic story listening. Our results show that across languages, transformer-based models predicted activity in a distributed landscape spanning widely distributed cortical functional networks like limbic, ventral attention, default mode network, and subcortical structures. Spatial alignment patterns showed substantial cross-linguistic overlap and remained largely stable across model layers, with limited layer progression consistent with functional cortical hierarchies. Contrary to previous evidence, contextual embeddings did not outperform static embeddings. To test candidate computational explanations, we examined whether layer-wise brain scores reflect surprisal and intrinsic dimensionality, and thereby predictive processing and information compression. Neither of these two computational metrics mirrored neural alignment profiles. Our findings suggest that brain-LLM alignment is spatially robust and cross-linguistically stable but not explainable from predictive uncertainty or representational geometry. Rather than directly reflecting shared hierarchical computation, neural predictivity may primarily arise from distributed lexical-semantic correspondences that generalize across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18309v2">From Program Slices to Causal Clarity: Evaluating Faithful, Actionable LLM-Generated Failure Explanations via Context Partitioning and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 10 pages, 5 figures, 5 tables. Accepted to EASE 2026 (EQUISA workshop), Glasgow, United Kingdom
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based debugging systems can generate failure explanations, but these explanations may be incomplete or incorrect. Misleading explanations are harmful for downstream tasks (e.g., bug triage, bug fixing). We investigate how explanation quality is affected by various LLM context configurations. Existing work predominantly treats LLM-generated failure explanations as an ad hoc by-product of debugging or repair workflows, using generic prompting over undifferentiated artifacts such as code, tests, and error messages rather than targeting explanations as a first-class output with dedicated quality assessment. Consequently, existing approaches provide limited support for assessing whether these explanations capture the underlying fault-error-failure mechanism and for actionable next steps, and most techniques instead prioritize task success (e.g., patch correctness or review quality) over the explicit causal explanation quality. We systematically vary the debugging information to study how distinct context compositions affect the quality of LLM-generated failure explanations. Across 93 context configurations on real bugs and three economically viable models (gpt-5-mini, DeepSeek-V3.2, and Grok-4.1-fast), we evaluate explanations with six criteria and validate the LLM-as-a-judge scores against human ratings in a user study. Our results indicate that explanation quality is causally affected by context composition. Evidence-rich, failure-specific artifacts improve causal and action-oriented quality, whereas overly large contexts tend to yield vague explanations. Higher explanation-score quartiles are associated with higher downstream repair pass rates and, for some models, with fixes that are closer to the reference minimal fixes. In contrast, low-score quartiles can even underperform the no-explanation baseline. Reproduction package is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25898v3">On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      LLM-assisted modeling holds the potential to rapidly build executable Digital Twins of complex systems from only coarse descriptions and sensor data. However, resilience to LLM hallucination, human oversight, and real-time model adaptability remain challenging and often mutually conflicting requirements. We present three critical design principles for integrating resilience and oversight into such workflows, derived from insights gained through our work on FactoryFlow - an open-source LLM-assisted framework for building simulation-based Digital Twins of manufacturing systems. First, orthogonalize structural modeling and parameter fitting. Structural descriptions (components, interconnections) are LLM-translated from coarse natural language to an intermediate representation (IR) with human visualization and validation, which is algorithmically converted to the final model. Parameter inference, in contrast, operates continuously on sensor data streams with expert-tunable controls. Second, restrict the model IR to interconnections of parameterized, pre-validated library components rather than monolithic simulation code, enabling interpretability and error-resilience. Third, and most important, is to use a density-preserving IR. When IR descriptions expand dramatically from compact inputs hallucination errors accumulate proportionally. We present the case for Python as a density-preserving IR : loops express regularity compactly, classes capture hierarchy and composition, and the result remains highly readable while exploiting LLMs strong code generation capabilities. A key contribution is detailed characterization of LLM-induced errors across model descriptions of varying detail and complexity, revealing how IR choice critically impacts error rates. These insights provide actionable guidance for building resilient and transparent LLM-assisted simulation automation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21027v1">Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 The first four authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10726v2">PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rely on optimizations like Automatic Prefix Caching (APC) to accelerate inference. APC works by reusing previously computed states for the beginning part of a request (prefix), when another request starts with the same text. While APC improves throughput, it introduces timing side channels: cache hits are faster than misses, creating observable latency differences. In multi-tenant systems, attackers can exploit these differences to infer sensitive information, e.g., by incrementally reconstructing another user's request by observing hit/miss patterns. Current defenses take a sledgehammer approach: they disable APC and cache sharing, isolating users, and sacrificing efficiency for regular users. This paper presents PrefixWall, a system that secures multi-tenant LLM serving systems against APC side channels without sacrificing performance and efficiency. PrefixWall monitors cache reuse across users, flags suspicious sharing, and selectively isolates prefixes, restricting their reuse only when necessary. Evaluation shows that PrefixWall enables up to 70% higher cache reuse and 30% lower inference latency compared to existing defenses that isolate users. PrefixWall's lightweight design demonstrates how security in LLM serving does not have to come at the cost of unnecessarily reduced performance or unbearable overheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15104v2">From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demonstrates that degradations most often reflect misunderstandings of argument values in the speech. Considering real-world deployment scenarios, we further report text-only results, an ambiguity-based reformulation stress test, and a reference-free LLM-as-judge protocol validated against human preferences. Notably, we find that open-source Qwen3 judges with at least 8B parameters exceed 80% agreement with proprietary judges, supporting privacy-preserving evaluation. Overall, our framework provides a verifiable and reproducible first-stage diagnostic that complements purpose-built audio corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14444v3">A Free Lunch in LLM Compression: Revisiting Retraining after Pruning</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Post-training pruning can substantially reduce LLM inference costs, but it often degrades quality unless the remaining weights are adapted. Since global retraining is expensive at LLM scale, recent work has largely focused on increasingly sophisticated pruning criteria that aim to select better sparsity patterns without adaptation. We revisit this trade-off through local reconstruction: after pruning, we adapt one subset of the model parameters at a time on a calibration set, training it to match the corresponding intermediate activations of the dense model. We evaluate local reconstruction across model families and scales, up to 72B parameters, and establish three main findings. First, local reconstruction is an effective adaptation mechanism for LLMs: it matches post-pruning retraining while using over an order of magnitude less data and compute, even when using PEFT techniques. Second, reconstruction exhibits a broad "free-lunch" regime in granularity, i.e., the reconstruction parameter window: as long as the reconstructed region contains at least a nonlinear submodule, final quality is largely insensitive to the window size, allowing granularity to be chosen primarily based on memory constraints. In contrast, reconstructing individual matrices, despite being the natural approach often proposed in the literature, consistently underperforms, as small matrix-level errors accumulate into larger activation drift. Lastly, reconstruction reduces the relative importance of the pruning criterion: performance gaps between sophisticated criteria and simple baselines shrink with model scale, making simple methods competitive again. Overall, our results challenge the prevailing view that post-pruning adaptation is impractical for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20923v1">Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Distributed LLM agent workflows should not be monitored as if they produced a single sequential log. In an asynchronous execution, a decision can only depend on events that are causally visible to the lifeline that makes it: an event that appears earlier in some log may still be unknown locally. We extend the ZipperGen agent-workflow framework with Causal Past Logic (CPL), a small past-time temporal logic for guards in conditionals and while loops. In addition to standard past-time modalities such as previous and since, a guard can inspect the latest causally visible event of another lifeline and selected variables stored there. The formula is a source-level guard: it is evaluated online by the owner lifeline and can influence control flow at runtime. We give a vector-clock monitor with latest-value views and prove that the locally computed monitor value coincides with the denotational semantics of the guard at the current event. Thus runtime verification becomes part of the coordination language itself, rather than a post-hoc check over an execution log.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07731v2">Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06139v2">Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a standard approach for large language models (LLMs) post-training to incentivize reasoning capacity. Among existing recipes, group-based policy gradient is prevalent, which samples a group of responses per prompt and updates the policy via group-relative advantage signals. This work reveals that these optimization strategies share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose Listwise Policy Optimization (LPO) to explicitly conduct the target-projection, which demystifies the implicit target by restricting the proximal RL objective to the response simplex, and then projects the policy via exact divergence minimization. This framework provides (i) monotonic improvement on the listwise objective with bounded, zero-sum, and self-correcting projection gradients, and (ii) flexibility in divergence selection with distinct structural properties through the decoupled projection step. On diverse reasoning tasks and LLM backbones, LPO consistently improves training performance over typical policy gradient baselines under matched targets, while intrinsically preserving optimization stability and response diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01712v2">FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 26 pages, 6 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models for vertical domains remains labor-intensive, requiring practitioners to curate data, configure training, and iteratively diagnose model behavior. Despite growing interest in autonomous machine learning and language agents, end-to-end LLM fine-tuning has not been systematically studied as an interactive agent task. We introduce FT-Dojo, an interactive benchmark environment for autonomous LLM fine-tuning, comprising 13 tasks across 5 domains. Rather than a new collection of static datasets, FT-Dojo standardizes a task interface, shared raw-data repository, sandboxed execution environment, structured feedback protocol, and held-out evaluation procedure. We further develop FT-Agent, a fine-tuning-oriented autonomous framework that uses structured iteration planning, fail-fast validation, and multi-level feedback analysis to refine data and training strategies. Experiments show that FT-Agent provides a strong initial baseline, achieving the best performance on 10 out of 13 tasks, with additional controlled comparisons against frontier agents, open-source planning backbones, and multi-run statistics supporting the main findings. Case studies show that agents can recover from failures through cumulative learning, while still exposing limitations in causal diagnosis and long-horizon planning. The implementation is available at https://github.com/microsoft/rd-agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19075v3">Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically require retraining for each LLM backbone due to architectural dependencies. To address these challenges, we propose Universal Reasoner (UniR)-a modular, composable, and plug-and-play reasoning module that can be used with larger frozen LLMs to provide specialized reasoning capabilities with a shared or aligned token space. Specifically, UniR decomposes the reward into a standalone reasoning module trained in a decoupled manner using verifiable rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR is combined with frozen LLMs at inference time by simply adding its output logits to those of the backbone. This additive structure enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Furthermore, UniR demonstrates weak-to-strong generalization, where reasoning modules trained on smaller models effectively guide much larger LLMs in the same model family, and generalize across domains such as in vision language models and medical reasoning. Experiments on mathematical reasoning and machine translation show that UniR surpasses existing fine-tuning methods. Code is open-sourced at https://github.com/hangeol/UniR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08636v2">InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 InternBootcamp Tech Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20863v1">PlexRL: Cluster-Level Orchestration of Serviceized LLM Execution for RLVR</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has recently unlocked strong reasoning capabilities in large language models (LLMs), triggering rapid exploration of new algorithms and data. However, RLVR training is notoriously inefficient: long-tailed rollouts, tool-induced stalls, and asymmetric resource requirements between rollout and training introduce substantial idle time that cannot be eliminated by job-local optimizations such as synchronous pipelining, asynchronous rollout, or colocated execution. We argue that this inefficiency is structural. While idle gaps are unavoidable within individual RLVR jobs, they are largely anti-correlated across jobs and therefore exploitable at the cluster level. Leveraging this observation, we present PlexRL, a cluster-level runtime for multiplexing unified LLM services across RLVR jobs. By centrally managing model placement, state transitions, and function-level scheduling under strict affinity constraints, PlexRL time-slices LLM execution across jobs to fill otherwise idle periods without expensive model migration. Our implementation and evaluations demonstrate that PlexRL significantly improves effective cluster capacity and reduces user GPU hour cost by maximum 37.58% while preserving algorithmic flexibility and introducing minimal per-job overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18586v3">TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 17 figures, 3 tables, 2 algorithms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in complex multi-agent applications that rely on external function calls. This workload creates severe performance challenges for the KV Cache: spatial contention leads to the eviction of critical agents' caches and temporal underutilization leaves the cache of agents stalled on long-running function calls idling in GPU memory. We present TokenCake, a KV-Cache-centric serving framework that bridges this gap by co-optimizing scheduling and memory management through an agent-aware design. TokenCake's Temporal Scheduler employs an event-driven, opportunistic policy to proactively offload idle KV Caches during function calls and uses predictive uploading to hide data transfer latency. TokenCake's Spatial Scheduler uses dynamic memory partitioning, guided by a hybrid priority metric combining graph structure and runtime state, to reserve GPU memory for critical-path agents. Our evaluation on representative multi-agent benchmarks shows that TokenCake reduces end-to-end latency by over 47.06% and improves effective GPU memory utilization by up to 16.9% compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08292v5">Do LLMs Triage Like Clinicians? A Dynamic Study of Outpatient Referral</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Outpatient referral (OR) is a core clinical workflow that assigns patients to hospital departments under incomplete and evolving information, yet it is commonly simplified as a static classification problem despite being inherently interactive in practice. In this work, we study outpatient referral as a dynamic process driven by information acquisition and uncertainty reduction. We analyze both static scenarios based on fixed patient information and dynamic scenarios involving multi-turn dialogue, to test whether large language models (LLMs) improve referral outcomes through better prediction or more effective questioning. Our findings show that LLMs offer limited advantages over traditional classifiers in static referral accuracy, but consistently outperform them in dynamic settings by asking discriminative follow-up questions that reduce uncertainty over candidate departments. These results suggest that the primary value of LLMs in outpatient referral lies not in static prediction, but in supporting interactive, uncertainty-aware clinical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20833v1">MemGym: a Long-Horizon Memory Environment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Memory is a central capability for LLM agents operating across long-horizon tasks. Existing memory benchmarks predominantly evaluate retention of personalized information in multi-turn chat scenarios, overlooking the dynamic memory formation that occurs during extended agent execution. Consequently, the memory systems they produce transfer poorly to realistic agentic environments, such as coding and web navigation. We present MemGym, a benchmark for agentic memory that unifies existing agent gyms and in-house memory-grounded pipelines behind one memory-reasoning interface. MemGym spans five evaluation tracks grouped into four agentic regimes: tool-use dialogue (tau2-bench), multi-turn deep-research search (MEMGYM-DR), coding (SWE-Gym and MEMGYM-CODEQA), and computer use (WebArena-Infinity). MemGym reports memory-isolated scores that decouple memory performance from reasoning, retrieval, and tool-use ability, so memory strategies can be ranked without those confounders. Our synthetic pipelines for MEMGYM-CODEQA and MEMGYM-DR are length-controllable, ablation-verified at every stage, and tightly aligned with downstream scenarios. To make evaluation on coding environments academically tractable, we train MemRM, a lightweight reward model (Qwen3-1.7B fine-tuned with QLoRA) that scores compression quality as a fast scalar read in place of full Docker rollouts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14654v2">Beyond Words: Multimodal LLM Knows When to Speak</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Project page: https://github.com/lzk901372/MM-When2Speak
    </div>
    <details class="paper-abstract">
      Chatbots via large language models (LLMs) generate fluent responses but often struggle with when to speak, especially for brief, timely listener reactions during ongoing dialogue. We present a multimodal strategy for LLMs, which leverages synchronized video, audio, and text cues to improve conversational timing awareness. The strategy reformulates response timing as a dense response-type prediction task, enabling an agent to decide whether to remain silent, produce a short reaction, or start a full response under streaming constraints. Therefore, we introduce a curated multimodal dataset from real-world dyadic conversational videos with temporally aligned modalities and fine-grained reaction type annotations. Moreover, we design a multimodal strategy, MM-When2Speak, with a multimodal integration module on top of an LLM backbone. Experiments across various modality settings and strong LLM baselines show that MM-When2Speak achieves up to a 3x improvement in response type prediction performance, highlighting the importance of multimodal perception for natural and engaging conversational interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19537v2">The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Progress in LLMs is increasingly measured through standardized benchmarks, where state-of-the-art improvements are often separated by fractions of a percentage point. At the same time, the computational cost of evaluating modern LLMs has driven widespread adoption of specialized inference backends, software systems that execute trained models efficiently at inference time. While critical for scalability, system-level optimizations, such as custom CUDA kernels and reduced-precision arithmetic, can alter token probabilities and introduce non-determinism, possibly cascading into divergent generation. In this work, we first survey the inference landscape, identifying 200 distinct engines, and analyze 35,000 ML publications, finding that the specific inference stack is rarely reported despite this widespread diversity. We then present a systematic empirical study of how inference backends affect LLM benchmark results. Holding model weights, decoding parameters, and hardware constant, we evaluate five widely used inference engines, including vLLM, SGLang, and llama$.$cpp, across multiple open-weight models and established benchmarks. We show that the choice of backend alone can shift benchmark scores by up to 16.6 percentage points and induce high rates of output disagreement. By isolating backend optimizations and tracing the execution pipeline, we find this divergence is driven by system-level optimizations like prefix caching and CUDA graphs, custom kernels, and engine-specific defaults in logit processing. Our findings identify the inference backend as a previously unreported but consequential hyperparameter in the evaluation of LLM and advocate standardized reporting of inference stacks to improve the reproducibility and interpretability of benchmark comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20815v1">GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 9 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data is complex and strictly regulated, reliance on cloud-based large language models (LLMs) introduces challenges in cost, latency, and compliance. In this work, we present a systematic evaluation of GraphRAG for EHR schema retrieval using locally deployed open-source LLMs. We implement the Microsoft GraphRAG pipeline on real-world EHR schema documentation and benchmark four models, including Llama 3.1 (8B), Mistral (7B), Qwen 2.5 (7B), and Phi-4-mini (3.8B), each deployed via Ollama on a single consumer GPU (8 GB VRAM). We evaluate indexing efficiency, knowledge graph construction, query latency, answer quality, and hallucination under both global and local retrieval modes. Our results reveal substantial differences: Llama 3.1 produces the richest knowledge graph (1,172 entities), Qwen 2.5 achieves the best answer quality (3.3/5), Phi-4-mini fails to complete the pipeline due to structured-output errors, and Mistral exhibits degenerate repetition behavior. We further show that GraphRAG exhibits a practical capacity threshold, where models below approximately 7B parameters fail to reliably produce valid structured outputs and cannot complete the pipeline. In addition, indexing and answer quality are decoupled across models, and local retrieval consistently outperforms global summarization in both latency and factual grounding, with reduced hallucination. These findings demonstrate that GraphRAG is feasible on consumer hardware while highlighting the importance of model selection and retrieval design for robust deployment in regulated settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10807v3">LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted for 2026 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) and hardware security is rapidly reshaping the semiconductor industry. While LLMs offer unprecedented capabilities in generating Register Transfer Level (RTL) code, automating testbenches, and bridging the semantic gap between high-level specifications and silicon, they simultaneously introduce severe vulnerabilities. This comprehensive review provides an in-depth analysis of the state-of-the-art in LLM-driven hardware design, organized around key advancements in EDA synthesis, hardware trust, design for security, and education. We systematically expand on the methodologies of recent breakthroughs -- from reasoning-driven synthesis and multi-agent vulnerability extraction to data contamination and adversarial machine learning (ML) evasion. We integrate general discussions on critical countermeasures, such as dynamic benchmarking to combat data memorization and aggressive red-teaming for robust security assessment. Finally, we synthesize cross-cutting lessons learned to guide future research toward secure, trustworthy, and autonomous design ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20809v1">Refining and Reusing Annotation Guidelines for LLM Annotation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 14 pages, 7 figures. Accepted to the ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable performance on zero-shot annotation tasks, they often struggle with the specialized conventions of gold-standard benchmarks. We propose the systematic reuse and refinement of annotation guidelines as an alignment mechanism, introducing an iterative moderation framework that simulates the early phases of annotation projects. We evaluate three hypotheses: (1) the efficacy of guideline integration, (2) the advantage of reasoning optimized models, and (3) the viability of moderation under minimal supervision. Testing across biomedical NER tasks (NCBI Disease, BC5CDR, BioRED) with three LLM families (GPT, Gemini, DeepSeek), our results empirically confirm all three hypotheses. While the iterative moderation framework shows good potential in effectively refining guidelines, our analysis also reveals substantial room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20767v1">The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show potential as simulators of human behavior, offering a scalable way to study responses to interventions. However, because LLMs are trained largely on observational data, interventions in experiments with LLM-simulated synthetic users can induce unintended shifts in latent user attributes, causing user drift where the implicit simulated population differs across treatment conditions, potentially distorting effect estimates. We formalize the confounding or selection bias that can arise due to user drift and show how intervention-dependent shifts can inflate or attenuate observed differences in user responses under intervention. To diagnose confounding, we propose using negative control outcomes--attributes that should remain invariant under intervention--to identify distribution shifts across intervention conditions, providing evidence of user drift. To mitigate drift, we study adjusting the persona specification by eliciting additional confounders, finding that targeted, setting-relevant confounders can substantially reduce bias across survey-style and multi-turn agent evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20759v1">Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety-Utility Tradeoffs in Graph-Context LLM Defenders</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Single-turn safety evaluation is a poor proxy for real fraud defense, where attackers escalate across multiple rounds. This paper evaluates fraud defenders under replay and adaptive multi-round attacks and measures when a defender refuses, not just whether it eventually refuses. On a frozen multi-round suite built from Fraud-R1, graph-context defenders improve early safe refusal relative to text-only baselines under both replay and adaptive fraud pressure, but they also produce substantially more benign over-refusal. Direct probing of the trained graph encoder, together with paired shuffle-risk ablations on both fraud and benign sides replicated across two seeds on the Qwen-1.5B backbone, localises this cost to how the defender LLM consumes structured context rather than to graph-encoder quality: the encoder cleanly separates fraud from benign, while the LLM responds primarily to the presence of structured graph fields and only secondarily, and asymmetrically, to risk-score magnitude. Temporal graph context is directionally stronger than static and significantly better grounded, but is not yet conclusively superior on the main refusal metrics. The contribution is evaluative and measurement-oriented: robust fraud assessment must be multi-round, must report refusal timing, must account for benign false positives alongside fraud-side safety gains, and must localize observed costs to the graph signal or to how the LLM consumes it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20734v1">An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      A large language model (LLM) agent that sends messages can leak data inside them. Destination allowlists and content scanners do not police whether an otherwise-benign payload is itself a covert channel: a compromised agent encodes bits in zero-width characters, homoglyphs, whitespace, base64, JavaScript Object Notation (JSON) key ordering, message timing or size -- and, in binary egress, in least-significant-bit (LSB) pixel planes, per-image mean luminance, inter-image sequence permutation, ultrasonic tones, or audible-band sonified data. Our egress reference monitor has three contributions. (i) A text pipeline of ten capacity-reducing stages, a per-sink leaky-bucket capacity ledger, and a staged posture that enforces lossless stages from day one. (ii) Two media scramblers (a Fourier-domain audio band-limiter and a red-green-blue (RGB) image bit-depth and mean-luminance bucketer) gated by a boot-time cryptographic legitimacy attestation: an auditor publishes at boot the trusted Ed25519 keys and {kind, data-class} pairs; only payloads with a verifying signature for an authorized class are exempt. The attestation sidesteps the intractable content-based discrimination between real media and data sonified or rasterized as a carrier; unsigned media is suspect by default; a content-addressed canonicalizer closes the inter-image permutation channel. (iii) Residual capacity is the Miller--Madow corrected mutual information between embedded and recovered bits (zero when destroyed), measured by an adversarial ensemble of fifteen working encoders across text, image and audio. The reference implementation drives residual capacity to zero on every destroyable channel and to a stated bound on the one (per-image mean luminance) that cannot be destroyed without ruining the image.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20706v1">Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 19 pages, 11 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Running language models in the browser presents a unique opportunity to build efficient, private, and portable AI applications, but requires contending with constrained memory availability and heterogeneous hardware targets. To realize this opportunity, we present Llamas on the Web (LlamaWeb), a WebGPU backend for llama$.$cpp that enables memory-efficient and performance-portable LLM inference across a wide range of model weight formats in the browser. Our design significantly reduces memory overhead through static memory planning and efficient model loading, addresses cross-device variability through a tunable kernel library, and introduces templated GPU kernels that support performant implementations of numerous quantization formats, enabling broad model support and extensibility to new formats. We evaluate LlamaWeb on 16 devices from 8 vendors, collecting data from 10 language models and four model weight formats. We compare LlamaWeb against existing browser-based LLM frameworks and find that LlamaWeb requires 29-33% less memory across several combinations of device, browser, and operating system. We also evaluate LlamaWeb's performance against these frameworks and find that it increases decode throughput by 45-69% across four GPUs from separate vendors. In addition, we compare LlamaWeb's performance against other llama$.$cpp backends, where it is competitive with and even beats vendor-specific backend performance on some devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21537v1">Articulate but Wrong: Self-Review Failures in LLM-Based Code Modernization</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 11 pages, 6 figures, 2 tables. Corpus, oracle, output extractor, prompts, harness, self-review probe, and all 1,980 + 1,979 raw model outputs released as supplementary material at https://zenodo.org/records/20300861
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used to migrate legacy code to modern stacks. We ask a deceptively simple question: when an LLM modernizes legacy code, can the same model be relied upon to recognize when its own output silently changes observable behavior? We run 1,980 real modernization calls across 11 production LLMs from 7 distinct families on a balanced 60-snippet legacy-Python-2 corpus, evaluate every output with a type-strict behavioral oracle, and then ask each model to judge whether its own output preserves behavior. We report four findings. (1) Semantic-preservation drift is prevalent and sharply separable from a cleanly-controlled baseline: semantic-trap snippets drift in 39.7% of attempts versus 7.0% on benign-control code that requires no real modernization (+32.7 percentage points; n=660 each). (2) Drift concentrates on specific snippets that fail across models: pairwise model agreement on which snippets are hard is high (mean Pearson r=0.52), and a small core of numeric-semantics snippets fails for nearly every model and every prompt phrasing. (3) Self-review by the producing model is not a reliable safety net: across all semantic drift cases, 31.7% are silently endorsed by the same model that produced them (83/262), and the per-model self-miss rate is strongly bimodal -- ranging from 0% on five models to 100% on one widely deployed model -- with several models explicitly articulating the very Py2/Py3 semantic distinction that broke their output, then declaring behavior preserved. (4) Drift rate is non-monotone in model capability and price: per-model rates range 5.6%-46.7% and do not track model capability cleanly, indicating the failure is task-structural rather than driven by model scale. All code, prompts, the 60-snippet corpus, the behavioral oracle, the output extractor, and the raw model outputs are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20641v1">Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 20 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Inference optimization is a vital technique for deploying LLMs at scale. Compilation is the most widely adopted optimization technique for LLMs. While it assumes semantic equivalence between the original and compiled graphs, we first uncover its numerical side effects can be maliciously exploited to implant stealthy backdoors in LLMs. We propose a unified optimization-triggered attack framework comprising two complementary strategies. Without any modification to the compiler or hardware, one strategy flips predictions for specific inputs only when the model is compiled, while the other uses a universal trigger that remains dormant under uncompiled execution but hijacks arbitrary inputs once compilation optimization is applied. Both attacks bypass standard safety evaluations run without compilation. We empirically demonstrate that these optimization-triggered backdoors achieve attack success rates averaging 90% across four mainstream open-source LLMs and four tasks, while clean accuracy is preserved at nearly 100% under all settings. Our findings reveal a novel attack surface at the intersection of optimization and security in the LLM deployment pipeline, and we investigate practical defenses to mitigate this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06007v2">MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to the ACL 2026 Demo Track. Camera-ready version. 10 pages, 6 figures. Code and documentation are available at: https://github.com/BUPT-GAMMA/MASFactory
    </div>
    <details class="paper-abstract">
      Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents or sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components, skill support, multimodal message handling, and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (https://github.com/BUPT-GAMMA/MASFactory, licensed under Apache-2.0) and video demonstration (https://youtu.be/ANynzVfY32k) are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10787v2">ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      Current LLM agents are proficient at calling isolated APIs but struggle with the "last mile" of commercial software automation. In real-world scenarios, tools are not independent; they are atomic, interdependent, and prone to environmental noise. We introduce $\textbf{ComplexMCP}$, a benchmark designed to evaluate agents in these rigorous conditions. Built on the Model Context Protocol (MCP), $\textbf{ComplexMCP}$ provides over 300 meticulously tested tools derived from 7 stateful sandboxes, ranging from office suites to financial systems. Unlike existing datasets, our benchmark utilizes a seed-driven architecture to simulate dynamic environment states and unpredictable API failures, ensuring a deterministic yet diverse evaluation. We evaluate various LLMs across full-context and RAG paradigms, revealing a stark performance gap: even top-tier models fail to exceed a 60% success rate, far trailing human performance 90%. Granular trajectory analysis identifies three fundamental bottlenecks: (1) $\textbf{tool retrieval saturation}$ as action spaces scale; (2) $\textbf{over-confidence}$, where agents skip essential environment verifications; and (3) $\textbf{strategic defeatism}$, a tendency to rationalize failure rather than pursuing recovery. These findings underscore the insufficiency of current agents for interdependent workflows, positioning $\textbf{ComplexMCP}$ as a critical testbed for the next generation of resilient autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17164v2">Charon: A Unified and Fine-Grained Simulator for Large-Scale LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted by MLSys 2026
    </div>
    <details class="paper-abstract">
      Deploying large-scale LLM training and inference with optimal performance is exceptionally challenging due to a complex design space of parallelism strategies, system optimizations, and hardware configurations. Accurate and rapid performance simulation is critical for guiding optimization efforts and system studies by validating "what-if" Hooker Figure hypotheses. To address this, we introduce Charon, a unified, modular, and fine-grained simulator for accurately predicting LLM performance. Experiments show Charon achieves high accuracy across different models and configurations, with an overall prediction error consistently under 5.35%, and even under 3.74% for training with a large-scale GPU cluster. In a practical inference deployment case, Charon discovered a configuration that improved system throughput over an engineering-tuned baseline, demonstrating its significant real-world value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17694v2">Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 ACL 2026 (main)
    </div>
    <details class="paper-abstract">
      Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) language coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio-cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20519v1">Codec-Robust Attacks on Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Prior attacks on Audio Large Language Models (Audio LLMs) demonstrated that carefully crafted waveform-domain perturbations can force targeted adversarial outputs. As a defense mechanism against these attacks, real-world codec compression preprocessing has been studied to both detect and remove the perturbations. Yet no existing attack has demonstrated robustness against these compressions. We introduce CodecAttack, which optimizes a perturbation in a neural audio codec's continuous latent space rather than directly perturbing the audio waveform. We show that the codec's compression channel, which discards waveform perturbations, transmits perturbations crafted in its own latent space. To further harden the attack across real-world compression channels, we apply multi-bitrate straight-through Expectation-over-Transformation (EoT), all without modifying the target model. Across three realistic Audio LLM deployment scenarios and three target models, CodecAttack achieves an average 85.5% target-substring attack success rate (ASR) on Opus at moderate bitrates, while the waveform baseline trained with identical EoT hardening does not exceed 26% at any bitrate. The attack transfers to held-out codecs, reaching up to 100% ASR on MP3 and 84% on AAC-LC without retraining. A per-band energy analysis shows that the latent perturbation concentrates below 4kHz, exactly where codecs allocate the most bits, while the waveform baseline spreads into higher frequencies that codecs discard. These results demonstrate that lossy compression is not a reliable defense against adversarial audio and that codec-aware attacks pose a practical threat to deployed Audio LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16517v2">Customizing an LLM for Enterprise Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Enterprise software development is a continuous evolutionary process, characterized by incremental additions, architectural revisions, production deployments and rigorous maintenance. These activities generate valuable data that modern LLMs could be finetuned on, to unlock additional tool possibilities for enterprise software engineering. While frontier LLMs are already very capable, this form of customization offers a compelling path for enterprise-specific optimization. We introduce Gemini for Google (GfG)}, an adaptation of Gemini specialized for Google's internal software engineering ecosystem. This paper details the model's end-to-end development, from curating a trillion-token proprietary dataset to implementing a mid-training strategy that mitigates catastrophic forgetting. In a large-scale blind A/B study across 29,000 developers, Gemini for Google significantly outperformed baselines: reducing the mean number of iterations per turn by 23\%, and increasing code survival rates by about 17%. Beyond metrics, we provide a comprehensive blueprint for enterprise model adaptation, covering: (1)The extraction of high-value signals from software engineering data, (2)Data preparation strategies, (3)Full-stack model tuning (continued pre-training and post-training), and (4)The deployment of downstream applications. We believe this methodology offers a replicable path for other organizations to unlock the full potential of their internal engineering data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20485v1">ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      As autonomous agents increasingly execute end-to-end tasks under fixed monetary budgets, the pressing open question shifts from whether the budget is respected, to how to spend it effectively. Existing budget-aware methods typically control reasoning step-by-step within a single agent, or learn resource allocation policies via RL. None address how to split a budget across the composing phases of a multi-agent pipeline at inference time. We propose ZEBRA, a zero-shot framework that reduces multi-phase budget allocation to a continuous nonlinear knapsack problem: an LLM controller estimates per-phase utility curves, and a water-filling search on the Lagrange multiplier returns the per-phase split. Additive and multiplicative aggregations are unified under the same solver. On a $150$-task APPS coding benchmark, both ZEBRA variants outperform LLM-direct (budget allocation directly by an LLM) on every aggregate metric. At a budget of $α= 0.5$ of the unconstrained spend, ZEBRA recovers $94.4\%$ of unconstrained quality, versus $88.1\%$ for LLM-direct. The advantage is statistically significant and transfers beyond coding: on a $3$-phase HotpotQA pipeline, ZEBRA beats LLM-direct by $14.3$pp, with allocations empirically robust to curve-estimation noise. On HotpotQA, ZEBRA arrives at a different budget split (near-balanced) compared to the APPS one (skewed towards a refinement phase), showing adaptation to the pipeline structure. More broadly, we show that lightweight algorithmic guidance at inference time can improve the economic behavior of autonomous multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00086v2">Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p<0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment, pending validation on larger corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20449v1">LLM Pretraining Shapes a Generalizable Manifold: Insights into Cross-Modal Transfer to Time Series</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Can language-pretrained transformers become effective time-series forecasters, and why? In this paper, we show that cross-modal transfer arises because language pretraining preconditions time series training with a reusable manifold. A linear probe on frozen LLM states decodes realistic time-series trajectories without paired supervision, and retrieval in this projected space yields competitive forecasts, showing that structure and dynamics exist before finetuning. Pretrained initialization also improves optimization, producing coherent gradients and a highly anisotropic loss landscape unlike random initialization. Finetuning then acts as low-dimensional alignment, reusing existing directions rather than learning temporal primitives from scratch, as evidenced by low-rank updates, subspace alignment, and shared features for periodicity, trend, and repetition. Together, these results support a geometric account of LLM-to-time-series transfer: language pretraining builds the manifold, and finetuning projects numerical dynamics onto task-relevant directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07832v2">rePIRL: Learn PRM with Inverse RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Process rewards have been widely used in deep reinforcement learning to improve training efficiency, reduce variance, and prevent reward hacking. In LLM reasoning, existing works also explore various solutions for learning effective process reward models (PRM) with or without the help of an expert policy. However, existing methods either rely on strong assumptions about the expert policies (e.g., requiring their reward functions) or suffer intrinsic limitations (e.g., entropy collapse), resulting in weak PRMs or limited generalizability. In this paper, we introduce rePIRL, an inverse RL-inspired framework that learns effective PRMs with minimal assumptions about expert policies. Specifically, we design a dual learning process that updates the policy and the PRM interchangeably. Our learning algorithm has customized techniques to address the challenges of scaling traditional inverse RL to LLMs. We theoretically show that our proposed learning framework can unify both online and offline PRM learning methods, justifying that rePIRL can learn PRMs with minimal assumptions. Empirical evaluations on standardized math and coding reasoning datasets demonstrate the effectiveness of rePIRL over existing methods. We further show the application of our trained PRM in test-time training, test-time scaling, and providing an early signal for training hard problems. Finally, we validate our training recipe and key design choices via a detailed ablation study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20410v1">Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 24 pages, 6 figures, including appendix. Accepted at the ICLR 2026 Workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems. Submitted to COLM 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in socially sensitive settings despite substantial documentation that they encode gender biases. Chain-of-Thought (CoT) prompting has been proposed as a bias-mitigation approach. However, existing evaluations primarily focus on changes in LLM benchmark performance, providing limited insight into whether apparent bias reductions reflect meaningful changes in a model's internal mechanisms. In this work, we investigate how CoT prompting affects gender bias in LLMs, combining benchmark-based evaluation with mechanistic interpretability techniques and reasoning chain failure analysis. Our results confirm a stereotypical bias present in LLM outputs across benchmarks, showing that CoT prompting does not consistently reduce the bias gap. Mechanistic analyses reveal that although CoT balances biased behavior in certain attention head clusters, gender bias remains embedded in hidden representations, indicating only superficial mitigation. Inspection of reasoning chains further suggests that these improvements stem from memorization and familiarity with the dataset rather than genuine understanding of bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20402v1">Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      MXFP4 arithmetic can dramatically accelerate reinforcement learning (RL) post-training of large language models (LLMs), yet the quantization error introduces severe accuracy degradation. Existing work treats the quantization error as a monolithic noise term, missing the distinct mechanisms upon interpreting how quantization error damages training. We prove an exact three-way decomposition of quantization error and show how each component dominates a distinct RL training pathway. Our theoretical and empirical analysis decomposes the MXFP4 quantization error into three additive components: "scale bias" from power-of-two rounding, "deadzone truncation" from zeroing small values, and "grid noise" from rounding to the nearest 4-bit grid. Each component dominates a distinct RL failure mode: scale bias accumulates multiplicatively through the backward pass, affecting gradient accuracy; deadzone truncation degrades rollout quality; and grid noise raises the policy's entropy. We combine corrections that are RL failure mode-targeted but not component-exclusive: Macro-block scaling to reduce scale bias, Outlier Fallback recovers deadzone entries, but also partially reduces scale bias induced error, and Adaptive Quantization Noise (AQN) for controlling the policy entropy. On Qwen2.5-3B dense and Qwen3-30B-A3B-Base mixture-of-experts model, the targeted corrections recover BF16 accuracy to within 0.7% and 3.0% respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20351v1">Refusal Evaluation in Coding LLMs and Code Agents: A Systematic Review of Thirteen Malicious-Code Prompt Corpora (2023-2025)</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 30 pages, 6 figures, 2 tables. PRISMA-style systematic review covering thirteen publicly released refusal corpora (AdvBench, CyberSecEval family, RMCBench, RedCode, MCGMark, JailbreakBench, CySecBench, MalwareBench, CIRCLE, MOCHA, ASTRA, Scam2Prompt, JAWS-Bench)
    </div>
    <details class="paper-abstract">
      The evaluation of large language model refusal on malicious-coding tasks now spans at least thirteen publicly released prompt corpora (AdvBench, the CyberSecEval family, RMCBench, RedCode, MCGMark, JailbreakBench, CySecBench, MalwareBench, CIRCLE, MOCHA, ASTRA, Scam2Prompt / Innoc2Scam-bench, and JAWS-Bench), each constructed under a different protocol, released under different licensing terms, and validated (or not) against different inter-rater reliability standards. Existing surveys treat code security, jailbreak taxonomy, or vulnerability detection as the central object and mention these corpora only in passing. This paper reverses that framing: it treats the prompt datasets themselves as the unit of analysis. Following a PRISMA-style protocol, we specify a search strategy, screen the recent literature on coding-LLM refusal evaluation, apply a uniform extraction template to each in-scope corpus, and synthesize the resulting catalogue along construction methodology, prompt-construction taxonomy (modality, turn structure, elicitation style), reproducibility and licensing, and malware-category coverage. The synthesis surfaces three recurring methodological gaps: the absence of human-annotator baselines against which LLM-judge labels can be calibrated, the absence of cross-corpus comparability with refusal-rate statistics measuring non-equivalent constructs, and the fragmentation of malware-category taxonomies, with no canonical schema spanning the thirteen in-scope corpora. The review concludes with proposed methodological directions for next-generation corpora, including pre-registration of inclusion criteria, vendor-diverse multi-judge validation, Fleiss' kappa with bootstrap CI as the reliability baseline, and a candidate canonical taxonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11401v5">FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20173v1">A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 25 pages, 2 figures, 6 tables. Companion repo at https://github.com/vasundras/agent-runtime-patterns
    </div>
    <details class="paper-abstract">
      Production LLM agents combine stochastic model outputs with deterministic software systems, yet the boundary between the two is rarely treated as a first-class architectural object. This paper names that boundary the stochastic-deterministic boundary (SDB): a four-part contract among a proposer, verifier, commit step, and reject signal that specifies how an LLM output becomes a system action. We argue that the SDB is the load-bearing primitive of production agent runtimes. Around this primitive, we organize agent runtime design into three concerns: Coordination, State, and Control. We present a catalog of six runtime patterns that compose the SDB differently across conversational, autonomous, and long-horizon agents: hierarchical delegation, scatter-gather plus saga, event-driven sequencing, shared state machine, supervisor plus gate, and human in the loop. For each pattern, we trace its lineage to distributed-systems concepts and identify what changes when the worker is stochastic. The paper contributes a five-step methodology for selecting runtime patterns, a diagnostic procedure that maps production failures to pattern weaknesses, and a failure mode called replay divergence, in which LLM-based consumers of a deterministic event log produce different downstream outputs under model-version or prompt changes. A stylized reliability decomposition separates per-call model variance from architectural momentum, motivating the claim that as model variance decreases, pattern choice and SDB strength become increasingly important levers for long-run reliability. We apply the methodology to five workloads and provide one runnable reference implementation for a 90-day contract-renewal agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20315v1">Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck in long-context, multi-turn inference. In this work, we propose Mix-Quant, a simple and effective phase-aware quantization framework for fast agentic inference. We first investigate FP4 quantization in agentic LLM workflows and observe that quantizing the entire inference process can incur significant performance degradation. In contrast, the prefilling stage exhibits substantial quantization redundancy and can therefore be quantized with minimal accuracy loss, despite being the dominant source of computation. Based on this insight, we apply high-throughput NVFP4 quantization to the prefilling phase while preserving BF16 precision for decoding. By decoupling prefilling acceleration from decoding quality, Mix-Quant combines phase-aware algorithmic quantization with hardware-efficient NVFP4 execution to alleviate the inference bottleneck in LLM agents. Extensive experiments across long-context and agentic benchmarks demonstrate that Mix-Quant largely preserves task performance while delivering significant efficiency improvements, achieving up to a 3x speedup during prefilling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03645v2">LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16559v5">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It takes a first step towards engineering automation using LLMs. Technically, it contributes to the community in two aspects:(1) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (2) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions. On nine frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20087v1">ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 53 pages, 23 figures, 4 tables. Project website: https://thoughttrace-project.github.io/
    </div>
    <details class="paper-abstract">
      Conversational AI has now reached billions of users, yet existing datasets capture only what people say, not what they think. We introduce ThoughtTrace, the first large-scale dataset that pairs real-world multi-turn human--AI conversations with users' self-reported thoughts: their reasons for sending prompts and reactions to assistant responses. ThoughtTrace comprises 1,058 users, 2,155 conversations, 17,058 turns, and 10,174 thought annotations collected across 20 language models. Our analysis shows that ThoughtTrace captures long-horizon, topically diverse interactions, and that thoughts are semantically distinct from messages, difficult for frontier LLMs to infer from context, diverse in content, and tied to conversation stages. We further demonstrate the utility of thoughts for downstream modeling. First, thoughts improve user-behavior prediction as inference-time context. Second, thought-guided rewrites provide fine-grained alignment signals for training personalized assistants. Together, ThoughtTrace establishes user thoughts as a new data modality for studying the cognitive dynamics behind human--AI interaction and provides a foundation for building assistants that better understand and adapt to users' latent goals, preferences, and needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20072v1">Probing Embodied LLMs: When Higher Observation Fidelity Hurts Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Submitted to From Animals to Animats: The 18th International Conference on the Simulation of Adaptive Behavior (SAB)
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly proposed as cognitive components for robotic systems, yet their opaque decision processes make it difficult to explain success or failure in closed-loop embodied tasks. Following an empirical AI methodology, we study embodied LLM agents behaviorally by varying the information available to the agent and measuring the resulting changes in behavior. Using the Lockbox, a sequential mechanical puzzle with hidden interdependencies, we evaluate LLMs across RGB, RGB-D, and ground-truth symbolic observations in a physical robotic setup and use controlled simulation to probe the resulting behavior. Counterintuitively, agents perform best under raw RGB input and worst under perfect ground-truth observations. In simulation, we probe this effect by randomly flipping perceived action outcomes and find that moderate noise improves performance, peaking at a 40% flip probability with a 2.85-fold success rate increase over the noise-free baseline. Further analysis links this gain to a reduction in repetitive action loops. These findings suggest that success rates alone are insufficient for evaluating LLMs, as measured performance may reflect the interaction between perceptual errors and reasoning failures rather than robust problem solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20055v1">Towards LLM-Assisted Architecture Recovery for Real-World ROS~2 Systems: An Agent-Based Multi-Level Approach to Hierarchical Structural Architecture Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Explicit software architecture models are essential artifacts for communicating, analyzing, and evolving complex software-intensive systems. In ROS~2-based robotic systems, however, structural (de-)composition and integration semantics are often only implicitly encoded across distributed artifacts such as source code and launch files, making recovery of hierarchical architecture particularly difficult. Existing approaches mainly focus on node-level entities and communication wiring, while providing limited support for recovering hierarchical structural (de-)composition across multiple abstraction levels. In this paper, we extend our previously proposed blueprint-guided LLM-assisted architecture recovery pipeline for ROS~2 systems through two major enhancements: (1) refined prompting to improve the consistency and controllability of architecture synthesis, and (2) a staged recovery strategy based on multi-level intermediate architectural representations that incorporate the atomic ROS node list and launch file dependencies, thereby enabling structurally constrained reconstruction across multiple abstraction levels. The approach is evaluated on a real-world automated product disassembly system based on cooperative robotic arms and heterogeneous ROS~2 artifacts. Compared to our previous work, the considered case study exhibits substantially higher integration complexity and richer functionality. The results demonstrate improved structural consistency, scalability, and robustness of architecture recovery, while also revealing remaining challenges related to dynamic integration semantics in large-scale ROS~2 systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09063v3">Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review, For questions or model-evaluation requests, contact $guijin.son@snu.ac.kr$
    </div>
    <details class="paper-abstract">
      Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20035v1">Stage-adaptive Token Selection for Efficient Omni-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Code Link: https://github.com/xxayt/SEATS
    </div>
    <details class="paper-abstract">
      Omni-modal large language models (om-LLMs) achieve unified audio-visual understanding by encoding video and audio into temporally aligned token sequences interleaved at the window level. However, processing these dense non-textual tokens throughout the LLM incurs substantial computational overhead. Although training-free token selection can reduce this cost, existing methods either focus on visual-only inputs or prune om-LLM tokens only before the LLM with fixed per-modality ratios, failing to capture how cross-modal token importance evolves across layers. To address this limitation, we first analyze the layer-wise token dependency of om-LLMs. We find that visual and audio dependencies follow a block-wise pattern and gradually weaken with depth, indicating that many late-layer non-textual tokens become redundant after cross-modal fusion. Motivated by this observation, we propose SEATS, a training-free, stage-adaptive token selection method for efficient om-LLM inference. Before the LLM, SEATS removes spatiotemporal redundancy via attention-weighted diversity selection. Inside the LLM, it progressively prunes tokens across blocks and dynamically allocates the retention budget from temporal windows to modalities using query relevance scores. In late layers, it removes all remaining non-textual tokens once cross-modal fusion is complete. Experiments on Qwen2.5-Omni and Qwen3-Omni demonstrate that SEATS effectively improves inference efficiency. Retaining only 10% of visual and audio tokens, it achieves a 9.3x FLOPs reduction and a 4.8x prefill speedup while preserving 96.3% of the original performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19999v1">LLM Benchmark Datasets Should Be Contamination-Resistant</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      Benchmark datasets are critical for reproducible, reliable, and discriminative evaluation of LLMs. However, recent studies reveal that many benchmark datasets are included in pretraining corpora, i.e., $\textit{contaminated}$, which diminishes their value as reliable measures of model generalization. In this paper, we argue that benchmark datasets should be $\textit{contamination-resistant}$, i.e., $\textit{unlearnable}$, but support $\textit{inference}$. To accomplish this, we first highlight the wide prevalence of benchmark dataset contamination and outline the properties of contamination-resistant datasets. Second, we highlight how the asymmetry between the inference and training pipelines in the Transformer architecture can be leveraged to support contamination-resistance. Third, we outline mathematical advancements to make these datasets interoperable across various LLM architectures. Based on the above, we call on the community to ensure the reliability of LLM benchmarking by: (i) advancing novel contamination-resistant methodologies, (ii) developing supporting methods and platforms, and (iii) adopting contamination-resistant benchmarks into existing evaluation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19952v1">Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      To enable reliable long-term interaction, LLM agents require a memory system that can faithfully store, efficiently retrieve, and deeply reason over accumulated dialogue history. Most existing methods adopt an extracted fact based paradigm: handcrafted static prompts compress raw dialogues into atomic facts, which are then stored, matched, and injected into downstream reasoning. Nevertheless, such fact-centric designs inevitably discard fine-grained details in original dialogues and fail to support deep reasoning over scattered isolated facts. Moreover, static prompts cannot maintain consistent extraction granularity across diverse dialogue styles. To address these limitations, we propose TriMem, which maintains three coexisting representation granularities, including raw dialogue segments anchored by source identifiers for storage fidelity, extracted atomic facts for efficient memory retrieval, synthesized profiles that aggregate dispersed facts into holistic semantic understanding for deep reasoning. We further adopt TextGrad-based prompt optimization, which iteratively refines extraction and profiling prompts via response quality feedback, achieving lifelong evolution without any parameter updating. Extensive experiments on LoCoMo and PerLTQA across multiple LLM backbones demonstrate that TriMem consistently outperforms strong memory baselines. The code is available at https://TMLR-TriMem.github.io .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19936v1">What Are LLMs Doing to Scientific Communication? Measuring Changes in Writing Practices and Reading Experience</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      Has the style of scientific communication changed due to the growing use of large language models in the writing process? We address this question in the domain of Natural Language Processing by leveraging two data resources we create: a naturalistic corpus of over 37,000 papers from the ACL Anthology (2020-2024); and a synthetic dataset of 3,000 human-written passages and their LLM-generated improvements. We first implement a series of diachronic lexical analyses, showing that both word frequency and usage contexts have changed significantly over time, indicating semantic specialization in some cases and generalization in others. Broadening our perspective, we then model a range of more complex stylistic features and find that LLM-modified texts more frequently contain certain syntactic constructions, more complex and longer words and a lower lexical diversity. Finally, we connect these changes in writing practices to subjective reading experience through a pilot annotation study with 20 domain experts. They overall rate LLM-improved texts as more understandable and exciting, but also express negative qualitative attitudes towards LLMs, highlighting the strongly subjective effect of AI-assisted writing on reading experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19932v1">PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly operate over long and recurring external contexts, like document corpora and code repositories. Across invocations, existing approaches preserve either the agent's trajectory, passive access to raw material, or task-level strategies. None of them preserves what we argue is most needed for repeated same-context workloads: reusable orientation knowledge (e.g., what the context contains, how it is organized, and which entities, constants, and schemas have historically been useful) about the recurring context itself. We introduce PEEK, a system that caches and maintains this orientation knowledge as a context map: a small, constant-sized artifact in the agent's prompt that gives it a persistent peek into the external context. The map is maintained by a programmable cache policy with three modules: a Distiller that extracts transferable knowledge from inference-time signals, a Cartographer that translates it into structured edits, and a priority-based Evictor that enforces a fixed token budget. On long-context reasoning and information aggregation, PEEK improves over strong baselines by 6.3-34.0% while using 93-145 fewer iterations and incurring 1.7-5.8x lower cost than the state-of-the-art prompt-learning framework, ACE. On context learning, PEEK improves solving rate and rubric accuracy by 6.0-14.0% and 7.8-12.1%, respectively, at 1.4x lower cost than ACE. These gains generalize across LMs and agent architectures, including OpenAI Codex, a production-grade coding agent. Together, these results show that a context map helps long-context LLM agents interact with recurring external contexts more accurately and efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11768v2">Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19915v1">LLM Agents Make Collective Belief Dynamics Programmable: Challenges and Research Directions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Classical models of opinion dynamics assume human participants with bounded rationality and limited coordination. The rise of LLM-based agents introduces a qualitative shift: agents can now participate in online discussions at scale, maintain consistent persuasion strategies, and coordinate systematically. This paper argues that LLM agents make collective belief dynamics programmable, enabling deliberate steering of population-level beliefs. We term this emerging problem programmable collective belief control. Through controlled multi-agent simulations, we provide proof-of-concept evidence that coordinated AI agents can induce measurable belief shifts that stabilize within a few interaction rounds. We identify four structural properties (indistinguishability, persistence, contextuality, and configurability) that make detection and defense fundamentally difficult. Based on these findings, we outline a research agenda spanning theoretical foundations for adversarial belief dynamics, operational methods for system-level detection and intervention, and simulation infrastructure for scalable experimentation. Our goal is not to present a complete solution, but to articulate why this problem demands urgent attention and to provide a conceptual foundation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19901v1">Can LLMs Produce Better Object-Oriented Designs than Human-Involved Development?</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Background: Large Language Models (LLMs) are increasingly used for code generation. However, their ability to generate multi-class projects that require object-oriented design (OOD) remains unclear, especially relative to projects developed with human involvement. Aims: The primary objective of this study is to compare OOD quality in projects from three authorship conditions: PreAI (human-involved projects produced before widespread LLM use), PostAI (human-involved projects produced after widespread LLM use), and PureAI (projects generated end-to-end by contemporary LLMs). Method: We conducted a comparative case study on a postgraduate Java assignment. Two offerings of the same assignment were selected as the PreAI and PostAI datasets. PureAI projects were generated using three contemporary LLMs. We analyzed OOD quality using project-level OOD metrics, code smell density, and domain modeling. Results: Relative to human-involved projects, PureAI projects show lower code smell density and generally appear simpler in terms of total size, complexity, and coupling. However, this is consistent with oversimplification, as it is associated with missing abstractions and weaker responsibility separation. PostAI is closer to PureAI than PreAI on many OOD measures and also shows tendencies toward oversimplification. Conclusions: Our findings indicate that appropriate human guidance on object-oriented decomposition and responsibility assignment remains important when LLMs are used for object-oriented design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13793v2">An LLM-Based System for Argument Mining</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Arguments are a fundamental aspect of human reasoning, in which claims are supported, challenged, and weighed against one another. We present an end-to-end large language model (LLM)-based system for reconstructing arguments from natural language text into abstract argument graphs. The system follows a multi-stage pipeline that progressively identifies argumentative components, selects relevant elements, and uncovers their logical relations. These elements are represented as directed acyclic graphs consisting of two component types (premises and conclusions) and three relation types (support, attack, and undercut). We conduct two complementary experiments to evaluate the system. First, we perform a manual evaluation on arguments drawn from an argumentation theory textbook to assess the system's ability to recover argumentative structure. Second, we conduct a quantitative evaluation on benchmark datasets, allowing comparison with prior work by mapping our outputs to established annotation schemes. Results show that the system can adequately recover argumentative structures and, when adapted to different annotation schemes, achieve reasonable performance across benchmark datasets. These findings highlight the potential of LLM-based pipelines for scalable argument mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19798v1">Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      As Socially Interactive Agents (SIAs) become increasingly integrated into daily life, the ability to calibrate user trust to an agent's actual capabilities would help ensure appropriate usage of these agents. In this paper, we explore the capacity of Large Language Models (LLMs) to generate multimodal behaviors (verbal, vocal, gestural, and facial expression modalities) that reflect varying levels of ability and benevolence, two key dimensions of trustworthiness. We propose a novel method for automatically generating behaviors aligned with specific levels of these traits, a first step towards enabling nuanced and trust-calibrated interactions. By analyzing a large dataset of multimodal transcripts generated by LLMs, we demonstrate that GPT-5.4 is able to produce coherent behavior across different modalities (text, intonation, facial expression, and gesture). Using Random Forest feature importance analysis, we show that the generated behaviors align with theoretical expectations for ability and benevolence. However, we also find that when gender is specified in the prompt, LLMs tend to reproduce societal gender stereotypes, associating male agents' behaviors with high ability and female agents' behaviors with high benevolence. To validate our approach, we conducted a user study on Prolific using a within-subjects design. Participants perceived different levels of ability and benevolence in the generated behaviors align with the intended instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19782v1">Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in their structure, it is difficult to evaluate which components contribute the most, and when and how this exploration may fail. We answer these questions through three controlled experiments. Our findings: (1) In pure black-box optimization, LLMs act as greedy optimizers. (2) In zero-shot kernel generation, providing explicit input-size information has no measurable effect, models converge to the same kernel parameters regardless of size or temperature, as though the size instruction were invisible. Moreover, when tasked to perform kernel optimization for uncommon kernel sizes, performance sharply degrades regardless of the language used. (3) In feedback-loop kernel optimization, CUDA improves monotonically under iterative feedback, while TVM IR actively degrades, which demonstrates that kernel optimization degrades when models operate with low-density language. Our results conclude that LLMs in code optimization tasks highly depend on pretrained priors rather than provided feedback or agentic structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19743v1">EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 26 pages, 10 figures, to be published at IDETC 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions: (1) a workflow benchmark with seven prompt styles targeting distinct cognitive demands-including direct tool use, semantic disambiguation, conditional branching, and working-memory tasks; (2) a Retrieval-Augmented Generation (RAG) benchmark with gated scoring isolating retrieval contributions to parameter selection; and (3) an High Performance Computing (HPC) benchmark evaluating end-to-end ML training orchestration on a SLURM cluster. Alongside the benchmark we present EngiAI, a Multi-Agent System (MAS) reference implementation built on LangGraph that operationalizes the benchmark by coordinating seven specialized agents through a supervisor architecture, unifying topology optimization, document retrieval, HPC job orchestration, and 3D printer control. Across four LLM backends and two EngiBench problems, proprietary models achieve 96-97% average task completion on Beams2D, while open-source 4B-parameter models reach 55-78%, with clear generational improvement. Conditional branching proves most challenging, with task completion dropping to 20-53% for the conditional style on Photonics2D. RAG gating confirms near-perfect retrieval-augmented scores ($\approx 1.0$) versus near-zero without retrieval, validating the evaluation design. On HPC orchestration, one model completes all pipeline steps in 100% of runs while another drops to 50%, revealing that multi-step instruction following degrades over long-running workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07066v2">2.5-D Decomposition for LLM-Based Spatial Construction</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Autonomous systems that build structures from natural-language instructions need reliable spatial reasoning, yet large language models (LLMs) make systematic coordinate errors when generating three-dimensional block placements. We present a neuro-symbolic pipeline based on \emph{2.5-D decomposition}: the LLM plans in the two-dimensional horizontal plane while a deterministic executor computes all vertical placement from column occupancy, eliminating an entire class of errors. On the Build What I Mean benchmark (160 rounds), GPT-4o-mini with this pipeline achieves 94.6\% mean structural accuracy across 12 independent runs, within 3.0 percentage points of the 97.6\% ceiling imposed by architect-agent errors that no builder-side improvement can address. This outperforms both GPT-4o at 90.3\% and the best competing system at 76.3\%. A controlled ablation confirms that 2.5-D decomposition is the dominant contributor, accounting for 50.7 percentage points of accuracy. The pipeline transfers directly to edge hardware: Nemotron-3 120B running locally on an NVIDIA Jetson Thor AGX matches the cloud result at 94.5\% with no prompt modifications. The underlying principle, removing deterministic dimensions from the LLM's output space, applies to any autonomous construction or assembly task where gravity or other physical constraints fix one or more degrees of freedom. A transfer experiment on 500 IGLU collaborative building tasks confirm the effect generalizes beyond the primary benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01482v2">Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04588v3">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19660v1">OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache compression framework for X-LLMs (i.e., text-only, multi-modal, and omni-modal LLMs). Advancing the per-channel paradigm, OScaR employs Canalized Rotation followed by Omni-Token Scaling to mitigate TNI-induced sequence-dimensional variance both effectively and efficiently, further supported by our optimized system design and CUDA kernels. Extensive evaluations across X-LLMs show that OScaR consistently outperforms existing methods and achieves near-lossless performance under INT2 quantization, establishing it as a robust, low-complexity, and universal framework that defines a new Pareto front. Compared with the BF16 FlashDecoding-v2 baseline, our OScaR implementation achieves a notable up to 3.0x speedup in decoding, reduces memory footprint by 5.3x, and increases throughput by 4.1x. The code for OScaR is publicly available at https://github.com/ZunhaiSu/OScaR-KV-Quant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20295v1">Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on mobile devices, where Neural Processing Units (NPUs) necessitate fully static quantization for optimal inference efficiency. However, existing post-training quantization (PTQ) methods predominantly rely on dynamic activation quantization, rendering them incompatible with NPU hardware constraints. To bridge the gap between high-fidelity PTQ and NPU-constrained inference, we propose Quant.npu, a integer-only fully static quantization framework. It incorporates learnable quantization parameters and rotation matrices, enabling low-bit activation-weight quantization without runtime quantization parameters re-computation. Crucially, we identify that initialization and selective optimization of quantization parameters is pivotal for optimization stability, as improper initialization and naive joint optimization induce gradient instability that disrupts the optimization of rotation matrices. To address this, we propose a rotation-and-bit-width-aware initialization tailored to diverse activation profiles and a distribution-aware selective optimization (two-stage quantization pipeline) tailored to rotated and unrotated tensors. Furthermore, we introduce a sensitivity-guided adaptive mixed-precision scheme to balance accuracy with inference efficiency. Extensive experiments on real-world mobile NPUs demonstrate that Quant.npu achieves comparable accuracy to state-of-the-art methods, while reducing inference latency by up to 15.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17839v3">How do LLMs Compute Verbal Confidence</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed -- just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents -- token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B (across TriviaQA, BigMath, and MMLU), Qwen 2.5 7B, and the reasoning model Magistral Small 24B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19627v1">How Helpful is LLM Assistance in Network Operations? A Case Study at a Large Demonstration Network</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      This paper reports on a real-world case study in which over 100 network engineers assessed how a Large Language Model (LLM) can assist in building and operating a network. The versatility of LLMs has accelerated their adoption across a wide range of domains, and assisting network operations is one such promising application. LLMs are probabilistic models, unlike deterministic protocols and configurations; therefore, clarifying their capabilities -- how and to what extent LLMs can help in network operations -- is a crucial step toward adopting LLMs. To offer practical insights into this issue, we conducted an extensive experiment on a large demonstration network built for a public exhibition, consisting of 21 racks with heterogeneous network devices. In the experiment, a total of 105 network engineers used an LLM-based chatbot while building and operating the network. The chatbot was equipped with three external functions: retrieval-augmented generation for domain-specific knowledge, CLI control of network devices running on the network, and access to a ticket system. The participants gave evaluations for the chatbot's responses on a best-effort basis. Analysis of the chat histories shows that 68.1% of the evaluations were positive, indicating a quantitative baseline of the LLM's helpfulness in network operations. Our results also demonstrate that understanding the capabilities of the chatbot is important for eliciting better responses. Moreover, we provide detailed use case analyses while sharing actual user--chatbot interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17726v3">Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19604v1">Formal Skill: Programmable Runtime Skills for Efficient and Accurate LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly act inside real workspaces, where tools and skills determine whether model reasoning becomes reliable action. Existing skills remain largely informal: Markdown skills and instruction packs encode procedures as long natural-language documents, while function calling, Model Context Protocol (MCP) servers, and framework tools structure individual actions but usually leave workflow state, policy enforcement, and completion discipline outside the skill itself. We introduce Formal Skill, a runtime-native abstraction that represents reusable capability with JSON metadata and action schemas, reliable Python executors, hook-governed control logic, Formal Skill routing, and skill-local runtime state. By moving reusable procedure from repeated prompt text into executable state machines and hook policies, Formal Skill gives agents a token-efficient and enforceable control surface. We implement the abstraction in FairyClaw, an open-source event-driven runtime for executable, observable, and composable Formal Skills. On Harness-Bench, FairyClaw obtains highly competitive average scores while using substantially fewer tokens, with especially strong results on tasks that expose the role of Formal Skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19597v1">LLMEval-Logic: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on natural-language logical reasoning is essential because rule-governed tasks require conclusions to follow strictly from stated premises. Many existing logical-reasoning benchmarks are generated by templating natural-language items from sampled formulas, provide only coarse or unaudited formal annotations, and are now quickly saturated by frontier reasoning models. We present LLMEval-Logic, a Chinese logical reasoning benchmark built from realistic situational scenarios. Its pipeline forward-authors and expert-audits natural-language items together with their reference formalizations, verifies annotated answers with Z3, constructs expert rubrics for natural-to-formal grading, and hardens selected items through a closed-loop adversarial workflow. The benchmark is released in two paired subsets: a 246-item Base subset shipped with 1,400 expert-developed rubric atoms, and a 190-item Hard subset with 938 multi-step sub-questions over closed model spaces. Evaluating 14 frontier LLMs on LLMEval-Logic reveals substantial gaps in current models: the best model reaches only 37.5% Hard Item Accuracy, and even with reference symbols the highest joint Z3+Rubric formalization score among evaluated models reaches only 60.16%. Our benchmark is publicly available at https://github.com/llmeval/LLMEval-Logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19595v1">A novel YOLO26-MoE optimized by an LLM agent for insulator fault detection considering UAV images</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The inspection of electrical power line insulators is essential for ensuring grid reliability and preventing failures caused by damaged or degraded insulation components. In recent years, Unmanned Aerial Vehicles (UAVs) combined with deep learning-based vision systems have emerged as an effective solution for automating this process. However, insulator fault detection remains challenging due to small defect regions, heterogeneous fault patterns, complex backgrounds, and varying imaging conditions. To address these challenges, this paper proposes an optimized YOLO26-MoE, a novel object detection architecture that integrates a sparse Mixture-of-Experts (MoE) module into the high-resolution branch of the YOLO26 detector. The proposed modification enables adaptive feature refinement for subtle and diverse fault patterns while preserving the efficiency of a one-stage detection framework. Hyperparameter optimization, final training, and evaluation were coordinated through a tool-augmented Large Language Model (LLM) agent. The proposed model achieved 0.9900 mAP@0.5 and 0.9515 mAP@0.5:0.95, outperforming the latest YOLO versions. These results demonstrate that the proposed model provides an effective and reliable solution for UAV-based insulator fault detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19593v1">Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 The 2026 Mediterranean Artificial Intelligence and Networking Conference (MAIN 2026)
    </div>
    <details class="paper-abstract">
      Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU memory constraints where partial CPU-GPU offloading and preemption become necessary. While existing systems primarily optimize throughput for a single model, comparatively little work addresses multi-model scheduling under these conditions. In this paper, we present an empirical study of how different LLMs behave across hardware platforms, focusing on the performance implications of layer offloading and preemption. We show that offloading leads to strongly non-linear and model-dependent degradation in decode throughput, with smaller models exhibiting sharper sensitivity to reduced GPU residency. We further demonstrate that preemption incurs substantial overhead, largely dominated by model state reload rather than key-value cache transfer, and that this cost varies significantly across models and hardware platforms. Additionally, we highlight the role of sequence length and interconnect bandwidth in amplifying data movement and execution inefficiencies. Based on these findings, we identify a set of key features that future schedulers must consider, including model-specific offloading sensitivity, workload characteristics, and the cost structure of preemption and data transfer. These insights provide guidance for the design of next-generation LLM serving systems capable of efficiently managing heterogeneous, multi-model workloads with hybrid CPU-GPU execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19576v1">Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom--LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)--yet the underlying mechanism has not been isolated. We provide (1) a reproducible trigger: ablations that isolate drift--one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) trace-level diagnostics: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a verified fix: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09997v2">GraphInstruct: A Progressive Benchmark for Diagnosing Capability Gaps in LLM Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 44 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Graph-structured data underpins applications from citation analysis and social-network modeling to molecular design and knowledge-graph construction, and Large Language Models (LLMs) are increasingly used as prompt-driven graph synthesizers. Classical graph-generation reviews catalog deep generative models and their evaluation primitives, but predate the LLM era and provide no foundation for evaluating instruction-following graph synthesis. Recent LLM-era benchmarks evaluate models along graph-type or task-domain axes; such organizations, however, average over structural complexity and cannot localize where in the complexity spectrum an LLM breaks down. To close this diagnostic gap, we introduce GraphInstruct, a progressive-complexity benchmark that stratifies LLM graph generation into six complexity levels and five evaluation dimensions, paired with 800 hand-authored instructions, 1,582 algorithmically synthesized reference solutions, and a 12-LLM capability evaluation across 45 (model, strategy) configurations. We find that discriminative power peaks at multi-constraint composition rather than reasoning depth, that no single prompting strategy dominates across levels or model families, and that domain-semantic constraints remain iteration-invariant under all tested methods -- pointing to retrieval rather than additional compute as the next research frontier. Atop the benchmark, a verification-guided iterative framework with constraint-aware adaptive prompting consistently surpasses the prompt-engineering ceiling on tested target models, demonstrating that the benchmark's fine-grained signals drive method development. Data, code, and reproducibility artifacts are released alongside the paper at https://github.com/AI4DataSynth/GraphInstruct_formal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22202v3">Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 27 pages, 1 figure, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now play a central role in code generation, yet they continue to hallucinate, frequently inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite growing awareness of these risks, there is limited understanding of how library hallucinations manifest under realistic usage conditions. To fill this gap, we present the first systematic study of how user-level prompt variations influence library hallucinations in LLM-generated code. Across seven diverse LLMs, we analyse library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries), examining the effects of realistic developer language and controlled user mistakes, including misspellings and fabricated libraries or members. Our findings expose systemic vulnerabilities: one-character misspellings trigger hallucinations in up to 26% of tasks; fabricated library names are accepted in up to 99%; and time-based prompts induce hallucinations in up to 85%. Grounded in the highest-risk prompts identified in our study, we introduce LibHalluBench, a benchmark that enables a systematic and reproducible evaluation of these library hallucinations. Our findings underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their downstream risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19529v1">Generative-Evaluative Agreement: A Necessary Validity Criterion for LLM-Enabled Adaptive Assessment</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 BEA 2026
    </div>
    <details class="paper-abstract">
      When the same LLM generates assessment items, simulates student responses, and scores them, the validation loop is self-referential. We introduce Generative-Evaluative Agreement (GEA), a validity criterion measuring whether an LLM's scoring function recovers the skill levels its generative function was instructed to produce. In the first direct measurement of GEA on a two-stage adaptive assessment, the model recovers roughly half the intended variance r = 0.698 with systematic positive bias. GEA is strong r > 0.7 for syntactically verifiable skills but near zero for design-level skills, and low-skill overestimation inflates scores near the routing threshold. We argue that granular, skill-decomposed rubrics are the principal proposed mechanism for strengthening GEA and outline complementary mitigations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18474v2">Prompt2Fingerprint: Plug-and-Play LLM Fingerprinting via Text-to-Weight Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      The widespread deployment and redistribution of large language models (LLMs) have made model provenance tracking a critical challenge. While existing LLM fingerprinting methods, particularly active approaches that embed identity signals via fine-tuning, achieve high accuracy and robustness, they suffer from significant scalability bottlenecks. These methods typically treat fingerprint injection as an independent, one-off optimization task rather than a reusable capability, necessitating separate, resource-intensive training for every new identity. This incurs prohibitive computational costs and deployment delays. To address this, we propose Prompt2Fingerprint (P2F), the first framework that reformulates fingerprinting as a conditional parameter generation task. By leveraging a specialized generator, P2F maps textual descriptions directly to low-rank parameter increments in a single forward pass, enabling plug-and-play LLM fingerprint injection without further model retraining. Our experiments demonstrate that P2F maintains high fingerprint accuracy, harmlessness, and robustness while significantly reducing computational overhead, offering a scalable and instant solution for LLM ownership management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19518v1">BLINKG: A Benchmark for LLM-Integrated Knowledge Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Generating Knowledge Graphs (KGs) remains one of the most time-consuming and labor-intensive tasks for knowledge engineers, as they need to identify semantic equivalences between input data sources and ontology terms. While declarative solutions (e.g., RML, SPARQL-Anything) have helped to generalize this process, aligning input schema elements with ontology terms still involves intricate transformations and requires considerable manual effort. With the advent of Large Language Models (LLMs), there is growing interest in leveraging their capabilities to assist KG engineers. Although some studies have explored using LLMs to automate KG construction, there is still no standardized framework for assessing how effectively they establish correspondences between data schemes and ontology concepts. Therefore, in this paper, we propose BLINKG, a benchmark designed to evaluate the mapping capabilities of LLMs in constructing KGs from heterogeneous data sources. The benchmark includes a set of scenarios with increasing complexity, based on real-world use cases. We conduct an extensive experimental evaluation of several stateof-the-art LLMs using BLINK and observe that they already offer promising solutions. However, their performance remains limited in complex scenarios. Thanks to this benchmark, we can already assess the current capabilities of LLMs for KG construction. Additionally, we define a set of requirements for achieving (semi)automated (LLM-driven) KG construction, opening new research lines in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19481v1">C2CServe: Leveraging NVLink-C2C for Elastic Serverless LLM Serving on MIG</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Modern LLM serving is increasingly serverless in shape: large model catalogs, long-tail invocations, and multi-tenant demand. Existing GPU serving systems face a tradeoff: dedicated-GPU allocation wastes scarce HBM under sparse traffic, while GPU time sharing places model initialization and weight loading on the cold-start path. Spatial GPU sharing such as multi-instance GPU (MIG) provides isolation and accounting, but each slice has too little HBM for modern LLM weights. We observe that high-bandwidth CPU--GPU interconnects, such as NVLink-C2C (C2C) in NVIDIA GH200 and GB200 Superchips, change the memory constraint: model weights can reside in CPU memory and be streamed on demand to MIG instances, shifting model residency from scarce HBM to abundant host memory. Leveraging this capability, we present C2CServe, a request-granularity serverless LLM serving system that allows MIG instances to switch models across requests without reloading weights into HBM. C2CServe introduces HybridGEMM, a heterogeneous-memory-aware GEMM kernel that adapts data access patterns to balance HBM and C2C bandwidth across MIG partitions using a single tuning knob. To mitigate shared-C2C contention, C2CServe further uses a hierarchical scheduler that coordinates model placement, input chunking, and kernel selection with online feedback control. On GH200, C2CServe reduces cold-start latency by up to 7.1x for dense models and 4.6x for MoE models compared with state-of-the-art serverless LLM serving systems, while maintaining over 95\% TTFT and TPOT attainment under C2C contention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11767v3">TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using state-based feedback. This improves rollout quality and stabilizes learning while remaining compatible with standard policy gradient optimizers, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a modest, one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a modular and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19433v1">Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in complex reasoning tasks via long chain-of-thought (CoT), yet their immense computational overhead hinders real-world deployment. LLM reasoning distillation addresses this by transferring reasoning capabilities from formidable teacher models to compact student models. However, existing distillation paradigms face a fundamental dilemma. Typical off-policy distillation strictly utilizes teacher-generated golden trajectories, suffering from an exposure bias due to the mismatch between training distributions and student-generated inference contexts, which leads to error cascades in long CoT reasoning. To address this, on-policy distillation allows students to explore their own trajectories, but we demonstrate that it inherently introduces a reciprocal reversed exposure bias: the teacher model also struggles to provide positive guidance when conditioned on student-generated sub-optimal contexts. To resolve this dual exposure biases problem, we propose Monitoring Trajectories and Backtracking when it strays (MOTAB), a new LLM reasoning distillation pipeline. Specifically, MOTAB dynamically monitors the student's on-policy generation against an adaptive safety boundary. When the generation strays and exceeds this threshold, MOTAB backtracks to the last safe state and leverages teacher intervention to correct the course. This approach inherently tolerates minor student errors to mitigate exposure bias, while preventing sub-optimal contexts to circumvent reversed exposure bias. Extensive experiments on the LIMO-v2 and AceReason datasets demonstrate that MOTAB effectively alleviates the dual exposure biases, yielding a roughly 3% average performance improvement in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20286v1">Adaptive Probe-based Steering for Robust LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 19 pages, 13 figures, accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the potential of contrastive steering for jailbreaking Large Language Models (LLMs). However, existing methods rely on limited and inherently biased contrastive prompts and require laborious manual tuning of steering strength, limiting their robustness and effectiveness. In this paper, we leverage the idea of model extraction to guide the learned steering vectors to approximate the ideal one and propose tuning the steering strength adaptively based on contrastive activations' statistics. Experiments demonstrate that our method notably improves the effectiveness and robustness of probe-based steering, without any extra contrastive prompts or laborious manual tuning. Being an attack paper, this paper focuses on revealing the breakdown of fortified LLMs, raising the average harmfulness score from 6\% to 70\%. Our code is available at https://github.com/fhdnskfbeuv/adaptiveSteering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20285v1">Introspective X Training: Feedback Conditioning Improves Scaling Across all LLM Training Stages</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      We tackle the question of how to scale more efficiently across the many, ever-growing stages of current LLM training pipelines. Our guiding intuition stems from the fact that the dynamics of later stages of the pipeline, e.g. post-training, can be used to inform earlier stages such as pre-training. To this end, we propose Introspective Training (or IXT), inspired by offline reward-conditioned reinforcement learning and applicable to any stage of training. IXT uses a thinking reward model to annotate data with natural language critique based feedback, enabling quality aware training from the earliest stages of the pipeline. Models are then trained by prefix-conditioning the data with the generated feedback -- ensuring that not all tokens are treated equally starting much earlier in training than usual. Comprehensive experiments on 7.5-12B transformer-based dense LLMs trained from scratch all the way up to 18 Trillion tokens seen show that our method: bends scaling curves resulting in up to 2.8x more compute efficiency generally; and reaches performance levels unachievable for models trained otherwise in domains such as math and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05431v4">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Organizing large-scale patent corpora according to classification schemes is a core information management task that determines the accuracy and efficiency of prior art retrieval, technology knowledge discovery, and intellectual property decision-making. Recent approaches distill natural language rationales generated by large language models (LLMs) into compact student models, yet logical errors, label mismatches, and taxonomy misalignments inherent in these rationales are indiscriminately absorbed during training, undermining classification reliability and propagating errors throughout downstream information processes. Rather than correcting such errors post-hoc, we propose Self-Filtered Distillation (SFD), which embeds quality assurance directly into the learning process by reinterpreting LLM-generated rationales as trust indicators rather than ground-truth supervision. SFD integrates three unsupervised signals into a unified trust score that dynamically modulates each training instance's contribution: Self-Consistency, which quantifies agreement among independently generated rationales; Class Entailment Alignment, which evaluates semantic coherence between a rationale and its assigned CPC class definition; and LLM Agreement Scoring, which assesses external plausibility through an independent verifier. On the USPTO-2M benchmark comprising over two million patents, SFD achieves up to 38.7\% relative improvement in Macro-F1 across four student architectures, and the strong correlation between trust scores and expert judgments ($r = 0.685$) confirms that the framework provides not only accurate predictions but also decomposable confidence semantics that enable auditable and self-documenting classification outcomes for large-scale patent knowledge organization.
    </details>
</div>
