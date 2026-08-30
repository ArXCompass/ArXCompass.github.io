# llm - 2026_08

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.21156v2">Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      LLMs have evolved from language generators to autonomous agents capable of complex, long-horizon tasks. This evolution has produced paradigms including Prompt Engineering to elicit model capabilities, Context Engineering to manage information access, Harness Engineering to organize external tools and resources, and Loop Engineering to support continual reflection and self-improvement. Yet as tasks grow more complex, individual intelligence faces a fundamental limit: many tasks require heterogeneous expertise, interdependent subtasks, parallel execution, independent verification, and persistent state, exceeding any single agent's organizational capacity. Augmenting one agent's capabilities or context cannot resolve this architectural mismatch; intelligence must instead be distributed across specialized agents and organized at the system level. We call this System Intelligence: an agent system's ability to organize and coordinate multiple intelligent components into a coherent, adaptive whole pursuing a shared objective. Achieving it requires more than adding agents; it demands explicit structures to organize work, coordinate heterogeneous agents, and maintain evolving execution states. We introduce Graph Engineering, an emerging paradigm for next-generation agent systems. Unlike prior paradigms that mainly optimize individual interactions or agent-level behavior, Graph Engineering constructs explicit, dynamic, evolving graph structures representing tasks, agents, and system states. These abstractions provide a unified foundation for organizing complex objectives, orchestrating heterogeneous agents, modeling system dynamics, and enabling scalable agent evolution. We systematically review the principles, methodologies, and applications of Graph Engineering for LLM agents. Related papers, open-source data, and projects are collected at https://github.com/DEEP-JLU/Awesome-Graph-Engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25657v1">Narcissus: Program Synthesis Using Context-Aware LLM Approximations</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at programming, but not when the task fixes the target language: prompted with a grammar rare in their training data, their programs usually break the grammar or fail the given specification. Enumerative synthesizers search the space of syntactically correct programs systematically guided by LLMs; the state of the art guides them by approximating LLM proposals into rule frequencies, which loses where each construct belongs and prunes every rule the proposals miss, exactly when the proposals are wrong. We present Narcissus, a synthesizer that keeps the proposals as syntax trees and scores each expansion of a candidate program in its context: does a proposal with the same surrounding structure continue the same way, and does the expansion rebuild a fragment the proposals repeat? A regularization term keeps every rule reachable, so wrong proposals delay the solution but cannot hide it. Across five domains and two search backends, Narcissus beats static guidance at every budget and consistently outperforms re-prompting the LLM to fix its own proposals; it reaches proposal-like programs an order of magnitude sooner and solves $40\%$ of ARC tasks where the raw proposals solve $13\%$, all without a single LLM call during search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01975v2">Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Submitted to the proceedings of the deRSE26 conference
    </div>
    <details class="paper-abstract">
      We consider LLM-based algorithm development through a case study on contractionorder optimisation for tensor networks with OpenEvolve. We pay particular attention to the choice of the LLM as well as design choices such as evaluation metric and test instances. Our results highlight both the promise of verifier-guided evolutionary coding agents for algorithm development/improvement and the continuing importance of evaluation, validation, and interpretation -- and corresponding challenges -- by the human scientist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23022v2">Scorpio: Serving Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Accepted at WISE 2026 (27th International Conference on Web Information Systems Engineering), Special Track on Innovative Web Technologies. 15 pages. This version has been accepted for publication after peer review but is not the Version of Record; the Version of Record will appear in the Springer LNCS proceedings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) serving increasingly underpins online Web services such as conversational agents, Web search, and programming assistants, where requests carry heterogeneous Service Level Objectives (SLOs) such as Time to First Token (TTFT) and Time Per Output Token (TPOT). Existing LLM serving systems prioritize maximum throughput and treat all requests uniformly, which leads to suboptimal SLO attainment. This paper introduces Scorpio, an SLO-oriented LLM serving system designed to maximize system goodput and SLO attainment for workloads with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection. Scorpio features a TTFT Guard, which employs least-deadline-first reordering and rejects unattainable requests, and a TPOT Guard, which utilizes a VBS-based admission control and a novel credit-based batching mechanism. Both guards are supported by a predictive module. Evaluations demonstrate that Scorpio improves system goodput by up to 14.4x and SLO adherence by up to 46.5% under high load compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24625v2">APPA: Recoverable Information-Flow Control for Real-World LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 21 pages, 3 figures. v2: Major revision with updated title, MCP protocol gateway architecture, 3-model empirical benchmark (6,600 episodes), recovery ablations, gradual security typing, and complete formal safety proofs
    </div>
    <details class="paper-abstract">
      LLM agents deployed in practical workflows routinely mix private context, untrusted tool and web outputs, and external side effects. While information-flow control (IFC) provides structural defenses against prompt injection, data exfiltration, and confused-deputy attacks, conventional IFC relies on monotone taint tracking that either over-blocks benign operations or permanently strands downstream execution once an agent ingests unvetted data. We present APPA (Agentic Permissions Policy Algebra), which turns agent IFC from an abort-only barrier into a policy-governed recovery system. APPA enforces a dual-phase reference monitor at tool dispatch and protocol gateways (e.g., Model Context Protocol): before tool execution, it prospectively evaluates composite label restrictions and workflow history; upon completion, it validates realized outputs before context admission. For incremental rollout across unannotated tools, APPA incorporates gradual security typing with bounded cast resolution. To inspect untrusted data without poisoning primary agent context, APPA introduces on-demand trajectory confinement: disposable child branches absorb taint locally and exit through shape-bounded channels (attest-schema) with exact parent-label and transcript preservation, avoiding permanently partitioned multi-agent infrastructure. We prove core safety invariants: no-laundering gradual resolution, branch boundary isolation, and recovery containment against prompt-injected models. Across 6,600 controlled benchmark episodes spanning OWASP AgentThreatBench and enterprise workflows (Bench-Corp), APPA sustains 64.2-91% utility with zero observed attacks across 1,320 guarded episodes, establishing a practical defense for deployed tool-using agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25605v1">From Specialization to Generalization: Instruction-tuned LLMs for Robust Harmful Content Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive performance across a wide range of general NLP tasks; however, their effectiveness in sensitive domains, such as hate speech detection, remains less clear. Prior studies comparing prompted LLMs with state-of-the-art encoder-based models (e.g., BERT variants (Roy et al., 2023; Dönmez et al., 2024)) have shown only marginal gains, suggesting that LLMs may not excel in hate speech detection or mitigation. In this work, we revisit this question through the lens of instruction tuning. By thoroughly unifying 36 English hate speech datasets spanning multiple labeling schemes, we fine-tune a generalist LLM, based on Qwen3 (Qwen Team, 2025), specifically for hate speech mitigation. Our results demonstrate not only state-of-the-art performance on in-domain benchmarks but also substantial improvements in cross-domain and cross-lingual generalization--areas where encoder-based specialist classifiers often struggle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25570v1">Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Hardware kernel optimization requires repeated compilation, correctness testing, profiling, and revision. LLM agents can automate parts of this process, and stronger foundation models, longer context windows, and longer execution horizons have improved optimization within individual tasks. These advances alone do not enable an agent to learn from completed optimization runs. Existing kernel-optimization agents seldom preserve a decision, its observed execution feedback, and the later decisions that use that evidence. Retaining every prior trajectory is also impractical because an expanding history competes with the current task for context. We present KOPE, an experience-driven framework for hardware kernel optimization. KOPE records optimization trajectories with correctness and performance feedback in Experience Graph Memory, then uses Active Context Management and Injection to retrieve relevant experience under a fixed token budget. The graph retains decision order, observed outcomes, and alternative branches, allowing evidence collected on the target hardware to inform later optimization steps and tasks. Under the same GLM-5.2 setting, the geometric mean of KOPE's per-operator speedups is $1.54\times$ that of CANNBot, the strongest competing baseline. In a complete 53-operator ablation, Active Context Management and Injection raises pass rate from 60.0\% to 84.6\%, increases the evaluator-reported positive-field geometric mean from 0.0382 to 0.0661, and reduces optimization token consumption from 15.9B to 1.113B tokens relative to passive agent-led context construction. Enabling Experience Graph Memory raises full-suite pass rate from 55.2\% to 84.6\% and yields a $1.43\times$ geometric-mean speedup on valid timing comparisons. These results support continual optimization through external experience while the foundation model remains fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18098v2">VDAR-Router: Adaptive LLMs Routing via Verbalized Query Difficulty Analysis Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Accepted by EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used in practical systems, making efficient model selection important for reducing deployment cost. LLM routing has emerged as a practical solution for allocating each input query to an appropriate model under a desired cost-performance trade-off. Existing routing methods often estimate model suitability from the surface semantics or embedding similarity of the input query. However, such methods may ignore the underlying difficulty of a query, leading to suboptimal routing decisions. To address the challenge, we propose VDAR-Router, a difficulty-aware retrieval-based routing framework. For each input query, VDAR-Router first generates an explicit difficulty analysis. It then retrieves historical examples with similar difficulty profiles. Based on the retrieved records, it estimates candidate model suitability and selects the model using a reward function that considers both performance and cost. Experiments on three datasets show that VDAR-Router consistently achieves better cost-performance trade-offs than existing baselines. These results demonstrate the effectiveness of difficulty-aware retrieval for training-free LLM routing. Case studies further show that explicit query analysis helps retrieve more relevant examples and supports more reliable routing decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25546v1">An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 11 pages, 10 figures, 7 tables
    </div>
    <details class="paper-abstract">
      LLM-based recommendation has scaled along model capacity and sequence length, yet each position encodes only text, semantic IDs, or a few categorical features, discarding rich user, item, context, and outcome signals available at each event. Under autoregressive modeling, this yields weak queries at each position and, since each position becomes context for the next, the degradation compounds across the sequence. We propose an event-centric paradigm that represents each interaction by its full temporal snapshot, and identify a new scaling dimension we term snapshot resolution: the amount of information encoded per event. To efficiently scale snapshot resolution, we introduce AMBER (Autoregressive Modeling via Bottlenecked Event Representation), which compresses each temporal snapshot into a compact Event Token, a new LLM input modality. The representation is learned end-to-end, while Event Tokens are pre-computed and cached for serving, decoupling snapshot resolution from real-time serving compute. On industrial-scale ranking and retrieval benchmarks, AMBER advances the compute-quality Pareto frontier relative to alternative recommendation paradigms. At sufficient capacity, a single unified tokenizer even outperforms dedicated per-entity tokenizers, demonstrating positive transfer across structurally different entity types. AMBER's Event Tokens also transfer across model architectures: when integrated into a heavily optimized non-LLM ranker as serving-time historical features, they yield statistically significant improvements. Further scaling Event Tokenizer capacity provides additional improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25523v1">TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Prefix caching introduces a fundamental tradeoff in multi-agent large language model (LLM) serving: retaining a long system-prompt key-value (KV) cache for an agent accelerates future calls, yet it reduces the GPU memory available for batching concurrent requests. In multi-stage workflows, existing schedulers tend to prioritize either immediate prefix locality or overall workflow progress. However, under a shared KV cache budget, optimizing either objective in isolation can prolong tasklevel job completion time (JCT) through downstream delays or frequent prefix replacement. To strike a balance, we here propose TOPAS, a Task-Oriented Prefix-Aware Scheduler that jointly decides which agent prefixes to keep in the cache and which requests to schedule for execution. TOPAS scores candidate post-decision states by trading off the expected reduction in each task's longest remaining service path against the near-term benefit of downstream prefix reuse, accounting for the costs of prefix movement and preemption. A task-level aging mechanism is also incorporated to prevent starvation. We implement TOPAS within the SGLang framework and assess its performance on three synthetic DAGs and two MetaGPT software-development workflows. Compared with the best performing baseline for each workload and metric, TOPAS reduces the mean/p99 JCT by up to 39.8%/49.4% on the synthetic workloads, while lowering mean JCT by 9.8% on MetaGPT-SOP and mean/p99 JCT by 22.0%/26.6% on MetaGPT-TL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15557v6">SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Agent skills, SKILL files that package reusable procedural knowledge for an LLM agent, are a popular mechanism for extending agent capabilities. Public repositories now host them in large and growing numbers, yet these artifacts are fragmented, redundant, and uneven in quality, and their value in practice is unclear. A core question remains open, namely how to consolidate this open-source SKILL ecosystem into a single usable corpus, and what bounds its benefit on real-world agent tasks. We present SkillCorpus, a framework that aggregates, curates, matches, and evaluates the open skill ecosystem at scale. It filters ~821,000 crawled skills through a multi-stage pipeline into 96,401 skills organised by a 16-class taxonomy and three quality facets (utility, robustness, safety), and pairs them with a fine-tuned retrieval-and-selection stack that matches task-relevant skills. We evaluate end-to-end across three benchmarks (SkillsBench, GDPVal, QwenClawBench), two harnesses, and two open backbones with a frontier robustness check. Integrating SkillCorpus yields consistent gains across all three benchmarks, largest on SkillsBench (+7.5 pp). An operational analysis traces the gains to a coverage boundary and a harness boundary. SkillCorpus is, to our knowledge, the first end-to-end account of when a curated, retrieval-served community corpus improves real agent tasks, and where it does not. The dataset, models, and code are available at https://github.com/EverMind-AI/SkillCorpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17648v2">From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Standard accuracy metrics cannot explain why LLMs handle variable tracking but fail on semantically equivalent loops. We study an internal lifecycle of code reasoning in which models first brew the answer, making it linearly recoverable many layers before it becomes self-decodable, and then diverge into one of four resolution outcomes: Resolved, Overprocessed, Misresolved, or Unresolved. Understanding this lifecycle matters because similar task accuracies can mask fundamentally different failure modes that surface-level evaluation cannot detect. We introduce a dual diagnostic framework pairing layer-wise linear probing with Context-Stripped Decoding (CSD) and apply it to six code-reasoning task families across 16 models spanning Qwen, Llama, and DeepSeek architectures. All four outcomes carry substantial mass in every task family: overall Resolved is only 41.5%, with multiple tasks below 30%. Controlled sweeps over structure, depth, and operators expose task-specific failure bottlenecks: Function Call Resolved plunges from 61.1% to 2.5% as call depth increases from one to three. Across architectures and scales, the brewing scaffold remains stable, with normalized brewing duration 24-42% across all 16 models, while resolution success varies with capability. This indicates that the scaffold is a stable empirical regularity across the tested decoder-only Transformer families, whereas resolution success covaries with capability, scale, and training. Code: https://github.com/euyis1019/llm-brewing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05158v3">Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 16 pages, 9 figures, 12 tables. Published in Proceedings of ACL 2026 (Volume 1: Long Papers), pages 11480-11497. Demo and API: https://witness-ai-jpt-ner.hf.space/
    </div>
    <details class="paper-abstract">
      Large language models encode extensive world knowledge valuable for zero-shot named entity recognition. However, their causal attention mechanism, where tokens attend only to preceding context, prevents effective token classification when disambiguation requires future context. Existing approaches use LLMs generatively, prompting them to list entities or produce structured outputs, but suffer from slow autoregressive decoding, hallucinated entities, and formatting errors. We propose Just Pass Twice (JPT), a simple yet effective method that enables causal LLMs to perform discriminative token classification with full bidirectional context. Our key insight is that concatenating the input to itself lets each token in the second pass attend to the complete sentence, requiring no architectural modifications. We combine these representations with definition-guided entity embeddings for flexible zero-shot generalization. Our approach achieves state-of-the-art results on zero-shot NER benchmarks, surpassing the previous best method by +7.9 F1 on average across CrossNER and MIT benchmarks, being over 20x faster than comparable generative methods. Interactive demo and access to the model are available at https://witness-ai-jpt-ner.hf.space/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07054v3">Sell More, Play Less: Benchmarking LLM Realistic Selling Skill</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Sales dialogues require multi-turn, goal-directed persuasion under asymmetric incentives, which makes them a challenging setting for large language models (LLMs). Yet existing dialogue benchmarks rarely measure deal progression and outcomes. We introduce SalesLLM benchmark, a bilingual (ZH/EN) benchmark derived from realistic applications covering Financial Services and Consumer Goods, built from 30,074 scripted configurations and 1,805 curated multi-turn scenarios with controllable difficulty and personas. We propose a fully automatic evaluation pipeline that combines (i) an LLM judge for sales-process progress, and (ii) fine-tuned BERT classifiers for end-of-dialogue buying intent. To improve simulation fidelity, we train a user model, CustomerLM, with SFT and DPO on 8,000+ crowdworker-involved sales conversations, reducing role inversion from 17.44% (GPT-4o) to 8.8%. SalesLLM benchmark scores correlate strongly with human ratings (average Pearson r=0.86; inter-annotator agreement Krippendorff's alpha=0.86, 500 annotations per model). Across 15 mainstream LLMs, in Chinese the strongest models are competitive with typical (junior-to-intermediate) human salespeople -- not with sales experts -- while weaker ones fall below this baseline, and cross-lingual consistency remains poor. SalesLLM benchmark serves as a scalable benchmark for outcome-oriented sales agents. Our code and data are released at https://github.com/Bairong-Xdynamics/Benchmarking-LLM-Realistic-Selling-Skill
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25431v1">Here is a GIFT: Enforcing User Data Isolation in LLM Serving via GPU Information Flow Tracking</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      LLM serving frameworks process large volumes of user data--often containing sensitive information--on shared infrastructure. Ensuring isolation between users who share the same serving framework (on CPUs) and LLM operators (on GPUs) is critical for privacy protection. This paper presents GIFT, a GPU Information Flow Tracking system that enforces user data isolation in LLM serving with minimal overhead. Moreover, the design of GIFT is non-intrusive and allows CPU-side serving frameworks to evolve freely. It rests on two key insights. First, encryption-as-isolation leverages the observation that CPU components only orchestrate data flow, not content manipulation; thus, per-user encryption can provide isolation without modifying serving logic. Second, GPU kernels exhibit limited and predictable information flows, enabling static flow analysis. GIFT precomputes information flow rules for each kernel and uses decoupled flow tracking, avoiding instrumentation or GPU stalls. Furthermore, we extend GIFT to GIFT-CC, which integrates confidential computing to protect against untrusted operating systems and hypervisors (LLM service providers). Implemented on vLLM and DistServe, GIFT and GIFT-CC enforce user data isolation with a 4-10.7% throughput overhead while maintaining the same latency level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25429v1">Distance Is Not Enough: Forget-Retain Alignment Gap Predicts LLM Relearning Robustness</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Accepted to EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to make a model forget specific data, yet unlearned LLMs often fail to stay unlearned: brief fine-tuning can revive removed knowledge. Existing robustness predictors rely on global weight-space displacement, but distance alone can be misleading when random or destructive updates collapse performance. We argue that relearning robustness depends on update structure: robust unlearning should affect forget-critical weights while sparing retain-critical ones. We introduce the Forget-Retain Alignment Gap (FRAG), a training-free predictor that scores an update's forget-retain alignment without running a relearning attack, and separates selective from dense updates more reliably than global distance. Building on the forget-critical, retain-sparing principle, Forget-Retain Pruning (FRP) improves relearning robustness. Our results suggest that weight selectivity better explains robustness than distance alone. Code is available at https://github.com/Yi1-Chen/FRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06346v3">Ban&Pick: Enhancing Performance and Efficiency of MoE-LLMs via Smarter Routing</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 26 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency at inference. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban further dynamically prunes redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban\&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08636v4">InternBootcamp: Boosting LLM Reasoning with Verifiable Task Scaling</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. With these bootcamps, we further establish Bootcamp-Eval, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves stateof-the-art results on Bootcamp-Eval and excels on other established benchmarks. In particular, we validate that consistent performance gain come from including more training tasks, namely task scaling, over two orders of magnitude, offering a promising route towards capable reasoning generalist. All data and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00991v2">SCHEDBench: A Benchmark for Evaluating LLM Constraint Faithfulness in Natural-Language Combinatorial Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      This paper introduces SCHEDBench, a natural-language benchmark for evaluating combinatorial scheduling constraint faithfulness under surface-form variation. Grounded in canonical scheduling instances and solver-derived feasibility and optimality, SCHEDBench assesses whether large language models (LLMs) generate schedules with the same constraint-feasible behavior across varied natural-language (NL) surface forms. SCHEDBench spans 1,132 instances across job-shop scheduling problems (JSP), single and multi-mode resource-constrained project scheduling problems (RCPSP), nurse rostering/scheduling, and curriculum timetabling problems of varying difficulty. Instances are templated into natural language problems using domain-specific templates, themed entities, lexical-syntactic template rephrasing, and constraint-level surface-form variation, with reference solutions verified for feasibility and objective optimality. Across thirteen frontier and open-weight LLMs, we find that models are not reliably invariant to semantically equivalent renderings of the same scheduling problem. Surface-form variation reduces feasibility and induces above-noise shifts in per-instance hard-constraint violations on matched instances. Among the tested isolated axes, constraint reordering yields the clearest above-noise sensitivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06087v3">LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling, and composition. On ALFWorld and Search-QA, LatentSkill outperforms the corresponding in-context skill baseline while using substantially fewer prefill tokens: it improves ALFWorld success by 21.4 and 13.4 points on the seen and unseen splits with 63.9% fewer prefill tokens on average, and improves Search-QA exact match by 3.0 points while using 71.8% fewer tokens per step. Further analysis shows that generated skill LoRAs form a structured semantic geometry, can be continuously modulated via the LoRA scaling coefficient, and can be composed through parameter-space arithmetic when skill components are aligned. These findings suggest that weight-space skills provide an efficient, modular, and less exposed substrate for extending LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25981v3">AgentDiff: Meaning-Bearing Rewrites Trigger Deeper Divergence than Presentation Changes in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      LLM agents should respond to what an input means, not how it is presented. We show that they do not treat these two kinds of variation equally. AgentDiff measures the difference between answer changes caused by meaning-bearing rewrites (paraphrase and synonym substitution) and presentation changes (reordering, formatting, and distractors), while matching perturbation severity. Across 68 model--benchmark--scaffold cells spanning ten LLMs from seven architecture families, three benchmarks, 1{,}530 original questions, and approximately 11{,}150 variants, meaning-bearing rewrites produce a $+19.69$ percentage-point higher inconsistency rate (paired $t=9.58$, $p<0.0001$; 64/68 cells positive). The result is stable under four severity proxies ($+18.9$ to $+20.9$~pp, all $p<0.0001$) and remains $+11.10$~pp on the 48 cells outside the qwen family. A fully held-out qwen2.5-14B-Instruct evaluation then tests the structure of this effect on 1{,}800 new trajectories: the pre-registered capable-and-tractable partition is positive in 3/4 held-out cells and remains sharply separated after pooling (Welch $t=3.81$, $p=9.6\!\times\!10^{-4}$). Trace analysis explains how the difference propagates. Meaning-bearing rewrites preserve the first action but reduce thought similarity from step 2 onward by $5.6$--$10.5$ points and extend the resulting cascade by $0.17$ steps (paired $t=7.69$, $p=2.5\!\times\!10^{-14}$), a pattern we call \emph{stealth divergence}. AgentDiff therefore establishes a reproducible directional robustness gap, identifies the regime in which it is most reliable, and connects final-answer inconsistency to a distinct trajectory-level signature. Code, perturbations, trajectories, and analysis scripts are released for review at https://anonymous.4open.science/r/agentdiff-emnlp-0BB4/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25379v1">Adaptive Triggering for Bias Correction in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 10 pages, 6 figures, Under review
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting can expose and amplify demographic stereotypes within an LLM's intermediate reasoning and create a failure mode that final-answer debiasing alone cannot address. Mitigating such bias during generation presents a fundamental timing problem: intervening too late allows biased reasoning to propagate, while unnecessarily intervening can disrupt otherwise correct reasoning. Existing approaches largely avoid this decision by either evaluating completed reasoning chains post hoc or intervening at predetermined steps, leaving open when a developing reasoning trajectory provides sufficient evidence to warrant correction. We formulate this decision as an online change-point detection problem. A per-step bias signal updates a CUSUM statistic and a targeted correction is injected only when accumulated evidence crosses a detector-specific threshold calibrated on held-out data. We instantiate the framework with a white-box signal derived from next-token probabilities and a black-box signal obtained from an LLM judge, enabling deployment with both open-weight and hosted models. On gpt-4o-mini adaptive black-box triggering recovers most of the disambiguated-context accuracy lost under fixed-interval intervention while requiring substantially fewer interventions. That result holds even with an independent judge. Across six open-weight models, the white-box signal improves ambiguous-item accuracy on all six but reduces disambiguated-item accuracy on five because it cannot distinguish unsupported stereotype reliance from correct, stereotype-congruent evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17118v2">MODE: Modality-Decomposed Expert-Level Mixed-Precision Quantization for MoE Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts Multimodal Large Language Models (MoE-MLLMs) offer remarkable performance but incur prohibitive GPU memory costs, making compression essential. Among PTQ methods, expert-level mixed-precision quantization has proven effective for MoE-LLMs, yet suffers notable degradation on MoE-MLLMs due to two overlooked biases in expert importance estimation. (1) At the cross-modal level, the numerical dominance of vision tokens causes expert selection frequency to be dominated by vision tokens, masking experts that are critical to the text modality; (2) at the intra-vision level, the large proportion of redundant vision tokens further skew frequency statistics, obscuring experts critical for informative visual content. To bridge gaps, we propose MODE, a modality-decomposed expert-level mixed-precision quantization framework for MoE-MLLMs that decomposes expert selection frequency by modality, filters redundant vision tokens to obtain denoised visual frequency, and further evaluates quantization sensitivity per modality as a complementary signal to frequency-based estimation. These signals are integrated into an Integer Linear Programming formulation to assign per-expert bit-widths under a given budget. Extensive experiments show that MODE is particularly well-suited for MoE-MLLMs, limiting average performance loss to within 2.9% at W3A16, with larger gains at the extreme 2-bit setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25358v1">Where vs What: Decomposing Structural and Content Failures in LLM-Generated Structured Outputs</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Structured outputs such as JSON and tables are central to modern LLM-based systems, yet generation failures are evaluated monolithically, conflating two distinct error modes: placement errors (correct values at wrong positions) and value errors (wrong values at intended positions). We introduce Structure-Content Decomposition (SCD), a framework that independently measures structural fidelity and content accuracy. Applying SCD to nested JSON and table tasks across six models (7B to frontier), we uncover a consistent phenomenon: structural fidelity degrades earlier and more sharply than content accuracy as complexity increases. At the highest complexity, even DeepSeek-V4-Flash (with reasoning) misplaces 35% of recalled values, while Qwen2.5-7B misplaces 74%. Controlled ablations suggest that this pattern is associated with reliance on semantic shortcuts rather than topological understanding of output structure. Based on these findings, we propose SA-RLVR, converting SCD metrics into verifiable rewards for reinforcement learning via GRPO. SA-RLVR successfully optimizes structural addressing across distinct topologies: it lifts JSON Value Placement Accuracy (VPA) from 26% to 63% while generalizing to held-out schemas; moreover, it consistently drives VPA improvements in the table domain, demonstrating that structure-aware rewards can directly enhance multi-domain structural positioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04424v4">Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Accepted at EMNLP 2026 Main; webpage at https://yao-dou.github.io/gavel/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now support contexts of up to 1M tokens, but their strengths and weaknesses on complex long-context tasks remain unclear. To study this, we focus on multi-document legal case summarization, where a single case often spans many documents exceeding 100K tokens. We systematically evaluate 12 frontier LLMs with Gavel, which consists of Gavel-Ref, a reference-based evaluation framework with checklist, residual-fact, and writing-style evaluations, and Gavel-Agent, a reference-free agent for evaluating factual coverage directly from source documents. Our results show that current models are more prone to omitting key information than hallucinating. They all perform well on simple checklist items, such as filing date, but struggle with rare and complex items, such as settlements. Performance also declines as case length increases. To meta-evaluate Gavel, we collect 160 hours of human annotations. Gavel-Agent reduces token usage by at least 36% compared to end-to-end and chunk-by-chunk methods while achieving competitive performance. Gavel-Agent also generalizes to the medical domain, performing the best with at least 77% fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.14480v3">Lower-Resource, Higher Scores: Language Bias in LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      LLM evaluators (trained reward models and prompted LLM-as-a-Judge) are routinely validated via pairwise accuracy. In a multilingual setting, this operates under the premise that high pairwise accuracy implies reliable, language-neutral scoring. We show that this assumption does not hold. We conduct experiments with semantically identical instruction-response pairs across 23 languages, and find that multilingual evaluators assign significantly different scores to different evaluation languages. The bias is statistically significant and consistent across eight open-weight evaluators of different architectures and training paradigms, persists in frontier judges, and is strongly correlated with language resource level: lower-resource languages are scored more generously. Meanwhile, these biases are invisible to pairwise accuracy: evaluators achieve above 90% pairwise accuracy, yet have up to 43% difference in acceptance rate across languages under a global decision threshold, meaning, for instance, that harmful content in lower-resource languages is more likely to pass safety filters. Per-language thresholds would require language identification, which can be defeated by code-switched prompts. We then investigate why lower-resource languages receive higher rather than lower scores, and we find that model uncertainty is linked with the effect: models tend to give higher scores when less confident, both under negative log-likelihood and under token-free uncertainty measures; however, language identity remains a significant predictor after controlling for uncertainty, and the bias cannot be explained away by content difficulty alone, but is a structural, language-level misalignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24650v2">Simthesizer: An Agent-Driven Simulation Framework for LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      System-level simulation is an essential tool for exploring the rapidly expanding design space of LLM serving systems, where real deployments remain costly and often infeasible. However, modern LLM serving now evolves faster than human-driven simulator development can track, and emerging workloads and mechanisms, from agentic workflows to disaggregated serving, no longer fit the monolithic simulation pipeline that existing simulators assume. Each new mechanism therefore demands an invasive rewrite, leaving a widening development gap between deployed serving systems and the simulators that model them. To close this gap, we present Simthesizer, a framework that realizes agent-driven simulator development. Simthesizer introduces a composable simulator infrastructure that uniformly expresses the complete serving workflow, including the control decisions that coordinate it, and realizes it as a unified dynamic graph in Simthesizer simulator. Synthesizer agent, a harnessed coding agent, then lowers natural-language feature requests onto this abstraction under simulator-specific guardrails and fidelity validation, evolving one shared simulator instead of building a new one for every feature. Under the same coding agent and harnesses, extensions built on Simthesizer follow a vLLM-based real system with 2.51% average throughput error, versus 6.03% for extensions built on existing simulators. On identical workloads, Simthesizer also simulates up to 284.96x and 23.19x faster than two state-of-the-art simulators, LLMServingSim2.0 and Vidur, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24001v2">Diverse by Reasoning: Harnessing the Wisdom of LLM Crowds for Future Prediction</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 13 pages, 4 figures, 5 tables. Submitted to IEEE BigData 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for future prediction, motivating the use of multiple models as a wisdom-of-the-crowd mechanism. However, simply increasing crowd size does not guarantee effective diversity, as different LLMs may exhibit redundant behaviors. We propose a behavior-aware framework for constructing diverse LLM crowds. The framework characterizes models using their reasoning traces on independent development tasks, clusters models by behavioral similarity, and selects representatives for collective prediction. We evaluate 25 LLMs using seven development benchmarks for behavioral diversity modeling and two future-prediction benchmarks for evaluating diverse crowds' performance. Our results show that crowd composition can matter more than crowd size: a three-model medoid crowd based on K-means++ behavioral clustering outperforms conventional voting over all 25 models on both prediction benchmarks, while reducing model calls by 88% and inference cost by approximately 80%. The results further suggest that representative behavioral diversity, rather than simply maximizing diversity, is important for constructing effective LLM crowds
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02764v2">Netflix Artwork Personalization via LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Pluralistic Alignment @ ICML 2026 Workshop; 6 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated success in various applications of user recommendation and personalization across e-commerce and entertainment. On many entertainment platforms such as Netflix, users typically interact with a wide range of titles, each represented by an artwork. Since users have diverse preferences, an artwork that appeals to one type of user may not resonate with another with different preferences. Given this user heterogeneity, our work explores the novel problem of personalized artwork recommendations according to diverse user preferences. Similar to the multi-dimensional nature of users' tastes, titles contain different themes and tones that may appeal to different viewers. For example, the same title might feature both heartfelt family drama and intense action scenes. Users who prefer romantic content may like the artwork emphasizing emotional warmth between the characters, while those who prefer action thrillers may find high-intensity action scenes more intriguing. Rather than a one-size-fits-all approach, we conduct post-training of pre-trained LLMs to make personalized artwork recommendations, selecting the most preferred visual representation of a title for each user and thereby improving user satisfaction and engagement. Our experimental results with Llama 3.1 8B models (trained on a dataset of 110K data points and evaluated on 5K held-out user-title pairs) show that the post-trained LLMs achieve 3-5\% improvements over the Netflix production model, suggesting a promising direction for granular personalized recommendations using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25321v1">LLMscope: Extracting LLM Assets from Edge AI Chips via Optical Probing</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      The move of LLM inference to edge AI accelerators introduces new physical vulnerabilities. During execution, model parameters and intermediate inference states are repeatedly loaded into and processed on the chip, making them suscep- tible to physical side-channel attacks. In this work, by deploying laser voltage imaging, we show that one can extract LLM assets during inference, namely embeddings, attention, and quantized MLP weights, activations, and other inference states, from localized memories and compute subcircuits. To validate our claims, we perform an attack on an FPGA-based LLM accelerator. Since such accelerators reuse the same buffers and compute subcircuits across addresses, tiles, modules, and layers, reading asset values comes down to probing different memories during inference. We demonstrate full recovery of the targeted values; however, we also establish a methodology to recover asset values even if some weights or bits remain unread. We further derive lower bounds that relate imaging effort to asset dimensions and show that even direct recovery scales linearly with the size of the targeted asset
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.14684v2">Mitigating Rubric Interference in LLM Judges via On-Policy Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      LLM judges increasingly evaluate responses against fine-grained rubric checklists. When a sample requires multiple rubrics, current methods typically assess each in a separate inference call. Evaluating all rubrics in a single pass is a natural alternative with greater efficiency, but we find that it introduces rubric interference: the verdict on one rubric shifts depending on which other rubrics are co-present. In a preliminary study, only one-third of samples receive fully consistent verdicts when evaluated under rubric sets of varying composition. We develop a measurement framework that probes interference through four controlled operations: rubric set expansion, subsetting, reordering, and noise injection. To mitigate interference without external supervision, we propose Self-Anchored Rubric Alignment (SARA). SARA uses a model's own single-rubric judgments as stable anchors and aligns multi-rubric reasoning with these anchors through on-policy self-distillation. We validate SARA on three datasets (HealthBench, FLASK, ResearchQA) and two model families (Qwen3, Llama-3.1). SARA consistently improves evaluation consistency while maintaining agreement with both base models and GPT-4.1 as a reference judge. Furthermore, the learned consistency transfers across datasets, confirming that SARA teaches a general capability rather than fitting dataset-specific patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02585v3">Mitigating LLM biases toward spurious social contexts using direct preference optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 COLM 2026, main text 10 pages
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used for high-stakes decision-making, yet their sensitivity to spurious context can introduce harmful biases. This is a critical concern when models are deployed for tasks like evaluating teachers' instructional quality, where biased assessment can affect teachers' professional development and career. We investigate model robustness to spurious social contexts about teachers using the largest publicly available dataset of U.S. classroom transcripts (NCTE) paired with expert evaluation scores. Evaluating seven frontier and open-weight models across seven categories of spurious contexts -- including teacher experience, education level, demographic identity, and sycophancy-inducing framings -- we find that irrelevant contexts can shift model-generated ratings by up to 1.48 points on a 7-point scale. Prompt-based mitigations and popular post-training methods, such as Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), fall short of addressing the issue. We propose Debiasing-DPO, which combines contrastive reasoning-augmented DPO with SFT on expert labels to reduce the effect of spurious context while avoiding mode collapse. Applied to Llama and Qwen Instruct models of 3-8B parameters, Debiasing-DPO reduces bias by 84% and improves predictive accuracy by 52% on average across models. Our findings from the educational dataset highlight that stronger models may exhibit greater sensitivity despite higher accuracy, and Debiasing-DPO can improve both accuracy and robustness in prompt-based prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24637v2">Thermal Tuning Overhead in Wafer-Scale Optical Interconnects for LLM MoE Training: A Cross-Layer Analysis and Ferroelectric-Based Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      The rapid scaling of large language models (LLMs), particularly mixture-of-experts (MoE) architectures, has intensified interconnect demands because expert-parallel execution is communication-intensive. Wafer-scale optical interconnects based on dense wavelength-division multiplexing (DWDM) offer a promising path to higher bandwidth; however, conventional microring-resonator (MRR)-based links rely on thermo-optic tuning and are therefore vulnerable to workload-induced thermal fluctuations. In this work, we present a cross-layer analysis of wafer-scale optical interconnects for MoE workloads that combines workload profiling, packet-level network simulation, and transient thermal analysis. We implement a wafer-scale topology in the ht-sim simulator and construct an Ansys thermal model of a 3D-integrated GPU/EIC/PIC stack. Our results show that transient temperature variations can exceed the tracking capability of conventional thermo-optic control loops and thereby introduce repeated tuning stalls during communication phases. The stall durations injected into the network simulation are derived directly from the thermal model rather than assumed. We further evaluate a ferroelectric-based electro-optic tuning mechanism that removes the continuous thermal-tuning requirement. In a four-layer proxy simulation across three MoE models, eliminating the tuning stalls yields speedups of 2.7x for Mixtral 8x7B, 3.8x for Qwen-MoE 14.3B, and 3.3x for LLaMA-MoE 6.7B relative to the thermo-optic case. These results indicate that minimizing photonic tuning latency is important for realizing the performance potential of optical interconnects in large-scale AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25291v1">InsightSR: Refining Symbolic Regression Search Spaces via Parallel Semantic and Structural LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Symbolic regression (SR) seeks to discover parsimonious mathematical laws from observational data, yet conventional approaches often struggle with the vast combinatorial search space of physically meaningful expressions. We present InsightSR, a framework that embeds Large Language Models (LLMs) as a guiding layer around the PySR genetic programming engine. Rather than relying on LLMs to generate expressions directly, InsightSR uses LLMs to progressively transform the search space itself through two complementary pathways: a Semantic Seed Pathway that proposes dimensionally consistent functional skeletons, and a Structural Feature Pathway that recommends nonlinear feature transformations. These transformations accumulate over iterations, broadening the input space and shifting the symbolic search from constructing deep expression trees over raw variables to assembling shallow trees over a rich, semantically informed feature set. A post-generation feedback loop evaluates candidates, categorizes features by their empirical utility, and refines the guidance for the next iteration, transforming the discovery process from open-ended generation into iterative, self-correcting refinement. Across three benchmarks, InsightSR achieves a 95% exact recovery rate on the Feynman benchmark and 80.18% accuracy on the LLM-SRBench LSR-Transform task, substantially outperforming state-of-the-art genetic programming and neural-symbolic methods while maintaining strong out-of-distribution generalization on real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25277v1">Routed Graph Handoff: Adaptive Format Selection for Multi-Agent LLM Delegation</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 Accepted in EMNLP 2026
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems coordinate through natural-language messages that consume 40--60\% of their token budget. Replacing these with structured graphs reduces cost but fails on tasks requiring adaptive reasoning. We propose \textbf{Routed Graph Handoff}, where a lightweight LLM router (155 tokens, 0.15\% overhead) selects between a typed dependency graph and natural language for each delegation. On four benchmarks (1,050+ trajectories), the routed system matches or exceeds NL-only on every task: \textbf{+12.7\,pp} on $τ$-retail at 3.2$\times$ compression ($p{<}0.01$), \textbf{+8.7\,pp} on BrowseComp at 2.2$\times$ compression ($p{<}0.05$), and parity on BFCL and AppWorld. Without the router, graph-only delegation regresses 14.6\,pp on AppWorld; the router eliminates this at near-zero cost. A graph-aware executor prompt is required: the same schema without interpretation guidance yields no gain. An oracle analysis reveals 8.6\,pp of additional headroom, motivating execution-time adaptive routing as future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25276v1">Groundhog Bit-Flip Attack: Seeding Infinite Generation Loops in Mixture-of-Experts LLMs through Bit Flips</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 9 pages, 3 figures; Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures enable scalable and efficient large language models (LLMs) by selectively activating expert sub-networks through a routing mechanism. However, this adaptive design introduces a new attack surface: specific experts become disproportionately correlated with certain tokens (e.g., end-of-sequence), allowing adversaries to manipulate model behavior via lightweight perturbations. In this work, we present \textbf{Groundhog Bit-Flip Attack (GBFA)}, the first bit-flip-based \textit{ Denial-of-Wallet availability attack} against MoE-based LLMs. By identifying and flipping routing-layer bits associated with related expert activations, we demonstrate that GBFA substantially extends the decoding token usage across three different LLM modes: conversational, reasoning, and agentic tasks, while largely preserving semantic fidelity. Across four main real-world MoE-based LLMs, manually deactivating on average fewer than \textbf{4 experts} drives average output inflation to $\mathbf{5912\%}$, with the majority of test samples reaching max tokens. These results reveal a robustness vulnerability of MoE architectures to bit flip, and highlight the potential of GBFA as an availability attack against LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10018v2">Helping the Helper: LLM-Assisted Problem Articulation for Older Adults Seeking Technology Support</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Older adults often struggle to articulate technology support needs due to unfamiliar technical terminology and age-related cognitive changes. We explore how large language models (LLMs) can facilitate this problem articulation process. Through a diary study (n = 27), we identified four communication barriers in older adults' queries: verbosity, incompleteness, over-specification, and under-specification. To mitigate these barriers, we developed an LLM pipeline that clarifies context and paraphrases unstructured queries. LLM-rephrased queries significantly improved automated solution accuracy (69% vs. 35%). Furthermore, younger adults (n = 48) acting as technology helpers understood LLM-rephrased queries better (93.7% vs. 65.8%) and reported greater ease in providing support. Older adults (n = 34) also found the resulting solutions highly actionable (94.7%). Finally, we contribute the first synthetic dataset of older adults' technology assistance queries (STAQ). This work demonstrates how LLMs can improve technology support seeking for older adults by addressing age-related communication barriers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25267v1">Mitigating LLM sycophancy with RL-based fine-tuning: Bayesian Truth Serum approach</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently exhibit \emph{sycophancy}: they adapt their answers to a user's stated beliefs or preferences instead of reporting what they hold to be true, which lowers factual accuracy and can amplify misinformation. This paper proposes a methodology for mitigating sycophancy that employs the Bayesian Truth Serum (BTS), a peer-prediction mechanism, as the reward in Group Relative Policy Optimization (GRPO) to fine-tune an LLM. BTS pays an answer for being \emph{surprisingly common}, that is, more frequent among respondents than those respondents themselves predicted. We treat a group of responses from a model for one question as those respondents, so the reward is a function of the model's own outputs and fine-tuning needs neither labels nor preference annotations. We prove that in the large-group limit a sycophantic response earns strictly lower expected reward than an honest one. We also prove that if the entire group agrees in advance on a symmetric answering rule, it cannot earn a higher information score than under truthful reporting. On our true/false benchmark the reference model's answer-flip rate under user pressure decreases from 23% to 4%, and its accuracy under that pressure increases from 80% to 93%. Our reward outperforms SMART and is comparable to synthetic-data fine-tuning and to pinpoint tuning, all three of which train on labels. It spends considerably more compute in exchange, which makes it suitable when labeled data is scarce. Peer Truth Serum, which also pays a premium for a rare answer but elicits no prediction report, reproduces the effect. A peer-prediction reward computed inside a single GRPO group therefore reduces sycophancy without labels, and comparing mechanisms suggests that the premium paid for a rarer answer drives the effect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19294v6">Maximizing Mutual Information Between Prompt and Response Improves LLM Performance With No Additional Data</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new data is expensive to collect. Moreover, true intelligence goes far beyond verifiable tasks. Therefore, we need self-improvement frameworks that are less dependent on external signals and more broadly applicable to both verifiable and non-verifiable domains. We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization to learn from this paired data maximizes pointwise mutual information *under the base LLM* between prompts and model responses. Experiments with with 1-7B parameter Llama and Qwen instruct models show that MIPO achieves 3-16% gains (and 51% increase for Qwen2.5-1.5B-Instruct) on personalization compared to prompting baselines. Surprisingly, MIPO can also be useful in verifiable domains, such as math and multiple-choice question answering, yielding 1-20% gains *without any additional data or external supervision*. These results suggest a promising direction for self-improvement using intrinsic signals derived from contrastive data pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25245v1">The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion</a></div>
    <div class="paper-meta">
      📅 2026-08-26
      | 💬 12 pages, 4 figures, and 2 tables. To appear in the Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)
    </div>
    <details class="paper-abstract">
      LLM-generated search queries are widely used to augment IR evaluation, yet they may contain concepts that presuppose answer-side document knowledge, violating the information-access boundary of pre-search users. Existing validation metrics, including overlap, diversity, and effectiveness, cannot distinguish rare human-tail variation from candidate answer-side intrusion. We introduce concept provenance, a framework that assigns query concepts to backstory-supported, human-central, human-tail, and candidate answer-side zones, operationalizing a boundary that retrieval metrics alone cannot detect. Applying concept provenance to 77,004 queries across 100 UQV100 topics, 8 LLMs, and 5 prompt conditions with two extraction pipelines, we obtain a cross-pipeline token-HCIR Spearman rho of 1.0 over five condition means. Candidate answer-side concepts constitute 7.40 percent of non-generic concepts and appear in 97 of 100 topics, with topic explaining approximately 67 percent of variance. Human validation yields 68.2 percent relaxed precision, revealing two mechanisms: knowledge intrusion at 45.5 percent and deployment intrusion at 45.0 percent. Diagnostic probes show disproportionate localized retrieval effects, with deletion effect size d = -0.47 compared with d = -0.34 for random deletion, but these concepts explain less than 2 percent of aggregate evaluation variance. Concept provenance therefore serves as a boundary-compliance diagnostic rather than an evaluation-shift predictor. Under the tested conditions, no prompt condition eliminates intrusion; post-generation concept-provenance selection achieves 99 percent elimination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07392v2">Online Pandora's Box for Contextual LLM Cascading</a></div>
    <div class="paper-meta">
      📅 2026-08-26
    </div>
    <details class="paper-abstract">
      Motivated by Large Language Model (LLM) cascading, we propose an online contextual Pandora's Box model for adaptively querying and selecting LLM APIs. In each period, a decision-maker observes a request context and faces a two-phase decision problem. In the query phase, the decision-maker sequentially queries APIs, where each query reveals a generated output and the decision-maker incurs an (output-dependent) cost. In the selection phase, the decision-maker selects one of the generated outputs to deploy and observes only the downstream reward of the deployed output. This output-mediated feedback structure differs from classical online contextual Pandora's Box models, in which opening a box directly reveals its reward. Rather than estimating the full conditional output and cost distributions of each API, we directly model the reservation index and develop a learning approach for the query phase. Specifically, we impose a parametric structure on the contextual reservation index functions induced by the classical Weitzman's policy. Our policy combines generalized method of moments (GMM) type estimation of these reservation indices with UCB-style confidence bounds for both these indices and the shared output-level reward evaluator. Under regularity conditions, we prove that the resulting policy achieves dimension-dependent $\widetilde O(\sqrt T)$ cumulative regret over a horizon of $T$ periods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25236v1">Rare Diseases, Common Dilemmas: LLMs Prioritize Equal Resource Distribution over Patient Benefit in Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Clinical decision-making often involves prioritizing ethical values, such as beneficence, non-maleficence, respecting a patient's autonomy, and justice. Recent work has begun to assess how large language models (LLMs) make such subjective, value-laden clinical judgments. However, evaluations of LLM decision-making in rare disease care contexts, where ethical tensions are ubiquitous and where scarce prior information likely impacts LLM behavior, are still lacking. Here, we present a benchmark of 208 clinically grounded rare disease vignettes, each of which presents genuine, high-stakes conflicts. When prompting 11 state-of-the-art LLMs to choose between clinically defensible yet ethically conflicting next steps embedded within these vignettes, we found that all evaluated models consistently prioritized justice over other core bioethical principles. Specifically, models overwhelmingly favor equal resource allocation over need-based considerations, indicating LLMs' limited responsiveness to differences in clinical severity or situational context. We also identify a strong authority-framing effect: models favor justice in committee-based contexts and shift toward beneficence and autonomy only when final decisions are framed as being made by clinicians or patients respectively. Our work suggests that institutional pressures surrounding rare disease resource utilization may be silently reflected in LLM-based decision support systems, with finer ethical considerations disregarded.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25220v1">FLARE: Verifying MILP Reformulations with LLM-Based Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Mixed-Integer Linear Programming (MILP) is a fundamental tool for combinatorial optimization with extensive real-world applications. A central challenge is designing computationally efficient MILP formulations. Large Language Models (LLMs) offer new opportunities to automate the modeling process, from deriving formulations to strengthening them. Reliable automation requires robust methods for verifying that proposed formulations preserve the underlying optimization problem. However, existing approaches evaluate formulations numerically and fail to reason about general problem instances. We resolve this limitation by introducing a constructive definition of MILP reformulation that can be formalized in Lean and machine-checked. We develop FLARE (Formulation-Level Automated Reformulation Evaluation), a method that uses an LLM-based agent and the Lean proof assistant to verify proposed reformulations against a reference formulation. To evaluate our approach, we introduce FormulationBench, a challenging dataset of 20 problems and 109 formulations. FLARE outperforms existing methods, with 100% accuracy on the NP-hard subset of FormulationBench. Furthermore, FLARE produces a machine-checkable certificate for every reformulation it accepts. For cases where formal guarantees are not necessary, we introduce FLARE-NL, a fast and cheap LLM proxy that matches FLARE's accuracy but produces no certificate. These methods enable reliable verification in automated optimization modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25217v1">LLM-Driven, Datasheet-Aware Automated Hardware Compatibility Verification for Early-Stage, Pre-Schematic Embedded System Design</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      We present an LLM-driven, datasheet-aware framework for early-stage hardware compatibility verification that identifies documentation-level interface incompatibilities based on hardware datasheets and high-level component connectivity descriptions. It does not require, and can therefore be used, before detailed schematic simulation and implementation. We view trustworthy LLM-assisted design automation not as directly generating answers from documents, but as transforming engineering information through traceable verification stages. Given hardware datasheets and high-level component connectivity descriptions, the framework constructs a design graph that captures device connectivity and shared interaction domains, retrieves only the engineering properties required by explicit, domain-oriented verification criteria , and generates deterministic scripts for compatibility evaluation. By decomposing compatibility analysis into modular stages and preserving intermediate results, the framework reduces context overhead, improves transparency and tractability, enables scaling, and avoids reliance on LLMs for numerical computation. Evaluated on seven embedded-system designs comprising 34 datasheets, our framework achieves 97.5% compatibility-verification accuracy and an 8.6 times reduction in input context size compared with ``upload-and-query'' workflows. These results demonstrate the feasibility of LLM-assisted, specification-based hardware compatibility verification at an early design stage, as well as the need for, and substantial benefits of, modular task decomposition, formalized verification criteria, and task-aware compact context construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21852v4">A Comedy of Estimators: On KL Regularization in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25210v1">Bolt-on, Verifiable Provenance for LLM-Powered Data Processing</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful tools for processing data. However, LLMs are also complex black-boxes, returning answers to queries on data, without any indication for where the answer came from or whether it is trustworthy. We introduce the notion of provenance for data processing with LLMs. While existing heuristics (such as embedding similarity or directly asking an LLM) could provide some hints for where the answer was derived, they provide no guarantees that the answer can be derived using the identified provenance, and indeed, are often incorrect. Instead, we propose the notion of verifiable provenance wherein we identify a subset of the input text that reproduces the same (or equivalent) answer as that on the complete text, and introduce the notion of minimality, where the verifiable provenance is as small as possible. To identify such a provenance, a naive solution would require checking all possible subsets of the source data with the LLM, which is prohibitively expensive. We present BLIP, a bolt-on framework for efficiently inferring a small-sized verifiable provenance for any LLM-powered data processing task, with any LLM. As part of BLIP, we introduce eight strategies, each guaranteed to find a minimal verifiable provenance, as well as an adaptive strategy that combines their strengths to reduce cost further. We further extend BLIP to produce multiple minimal verifiable provenances. Experiments on seven datasets show that the provenance generated by BLIP is always guaranteed to reproduce the answer - achieving over 30% higher accuracy than the best-performing baseline with a comparable provenance size. Moreover, BLIP incurs a low cost, comparable to the original query on the original data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25198v1">Tunable Tool-Call Rates in LLM Agents via Representation Steering</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Deciding whether to call a tool is a core competence of an LLM agent, and a costly one to get wrong: needless calls add latency, accrue cost, and may trigger irreversible side effects, while missing calls leave the model confidently wrong on questions it could only answer through tool-calls. Models manage this balance poorly, both over-using and under-using tools. Existing methods such as post-training and prompt engineering are expensive and difficult to modify at inference time. We show that whether an instruction-tuned model calls a tool can be controlled by a single linear direction in its residual stream, extracted without any training from the model's own tool-use preference signal and turned into an inference-time intervention with no prompt change. Adding the direction with strength $α$ moves the call rate monotonically from near $0\% $ to over $90\%$ while keeping calls well-formed. The steering works in both directions: dialing it down suppresses calls, and dialing it up induces new calls that land precisely on the questions the model cannot answer from its own knowledge. We also show that the direction generalizes to unseen tools with strength comparable to each tool's own direction and without favoring any specific tool choice. With live tool execution, a single sweep of the steering traces a cost/accuracy Pareto frontier and nearly doubles open-domain QA accuracy ($0.29 \! \rightarrow \! 0.56$); the same recipe transfers across a diverse range of models spanning dense, MoE, and multimodal architectures, without any training. Our code is publicly available at https://github.com/YuqiChen4188/Steering-Tool-Use-Propensity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01409v2">When Retrieval Helps and Distracts: Evaluating Evidence-Generating LLMs for Biomedical Claim Verification</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Biomedical fact-checking systems must do more than predict whether a claim is supported, contradicted, or unaddressed: they should also produce evidence that is faithful, complete, and useful for verification. We study this evidence-generation setting on CARE-XAI, a unified benchmark spanning five biomedical and health fact-checking sources. We compare base instruction LLMs, PubMed retrieval-augmented LLMs, fine-tuned LLMs, label-only LLMs, and biomedical encoder classifiers under a shared evaluation protocol. Biomedical classifiers remain strongest for verdict-only prediction, while fine-tuned LLMs are the strongest evidence-generating systems. PubMed retrieval is mixed: it helps PubMed-aligned sources such as PubMedQA and SciFact, but can distract models on broader public-health claims. We introduce Bio-GRACE, a gold-reference-normalized diagnostic for measuring whether retrieved evidence recovers the decision benefit of reference evidence. Bio-GRACE shows that retrieval utility is source-dependent, motivates selective retrieval, and exposes why retrieval recall and lexical evidence overlap are insufficient for biomedical fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25188v1">Transforms for LLM Quantization: The Great Inversion and Format Co-Design</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Most competitive 4-bit LLM research pipelines now open the same way: apply a linear, function-preserving transform (rotation, scaling, permutation, non-orthogonal affine) so the outlier mass sits more favorably against the group scales, and only then round. Yet we are aware of no survey dedicated to this transform stage, and its literature is quietly re-deriving an older theory. We identify and formalize the principle that organizes it, the Great Inversion: allocation-flexible coding rewards energy concentration, whereas the grouped shared-scale quantization a deployed matrix instruction performs rewards within-group flattening. Classical transform coding (1963: decorrelate, allocate bits, quantize) spends different bits per coordinate at a fixed total rate; for a Gaussian source at high rate the Karhunen-Loeve transform's concentration minimizes distortion. A deployed operand tile instead carries one absolute-maximum scale per group and equal bits everywhere, with no allocation; on a uniform grid that objective rewards flattening, approached by Hadamard incoherence. We prove that opposition under within-group majorization: the prescriptions point in opposite directions, each backed by a proof against its own objective, and for a generic spectrum no optimality guarantee transfers. A second axis is the number format: the non-uniform FP4 grid makes flattening buy less, MXFP4's power-of-two block scale still rewards a rotation confined to that block, and NVFP4's mantissa-carrying scale largely removes that pull, so the target pole depends jointly on allocation regime and format. We survey 200 works to a June 2026 cutoff; classify 43 transform methods by structure, data-awareness, searched-versus-constructed, and runtime cost; record, where reported, how they compose with GPTQ rounding; distill a first-choice guide by deployment regime; and close with the open problems it exposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25158v1">FuzzingBrain-Bench V1: Evaluating Open-Ended Bug Discovery by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 21 pages, 12 figures, 7 tables. Ze Sheng and Aleksandar Kezic contributed equally. Benchmark corpus and harnesses: https://github.com/fuzzingbrain/FuzzingBrain-Bench Counts verified against the built PDF and sources: 21 pages, 12 figure environments, 7 tables
    </div>
    <details class="paper-abstract">
      Evaluating the ability of large language models (LLMs) to discover software bugs is increasingly important. Existing benchmarks typically evaluate this capability by asking the model to generate a proof-of-concept input that triggers a predefined target vulnerability. However, this setup may overlook valid crashes discovered by the model when they do not match the predefined target. As a result, the evaluation may not reflect the model's real capability. We present FuzzingBrain-Bench, a benchmark for assessing AI models' ability to discover bugs in open-source software. Models are given an open-source project and a sanitizer-instrumented harness in a self-contained Docker image. Their goal is to generate inputs that trigger as many distinct crashes as possible through the harness. A model's performance on each challenge is scored based on the number of distinct crash signatures it produces, capped at a predefined maximum and weighted by a difficulty coefficient. FuzzingBrain-Bench V1 consists of 77 challenges drawn from 43 open-source projects, with 36 C, 32 C++, and 9 Java/JVM challenges. We evaluate Claude Haiku 4.5, Claude Sonnet 4.6, and Claude Opus 4.8 on the full benchmark. Claude Opus 4.8 performs best, triggering crashes in 60 of 77 challenges and achieving a score of 196 out of 579. None of the three models triggers a crash in 13 challenges. The FuzzingBrain-Bench corpus and harnesses are publicly available at https://github.com/fuzzingbrain/FuzzingBrain-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25152v1">Belief Cascades Drive Persuasion in LLM Agent Networks</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems increasingly debate answers, coordinate research, simulate users, and mediate information flows, making agent-to-agent persuasion a basic but undermeasured capability. We introduce a controlled testbed for studying how goal-directed persuaders shift elicited stances in networks of LLM agents grounded in real-world ego-network topologies. Across four LLM backbones, five graphs, and 55 policy statements, we find that persuasion dynamics depend on the interaction between topology, competition, topic, and model prior. Additionally, we show that direct exposure reliably predicts next-round stance change in competing runs, and peer relays carry smaller but measurable influence, showing that agents not assigned to persuade can still transmit persuasive force. Finally, analyzing post text alone misses important movement: planned strategies are only partly realized in executed messages, action choices can diverge from message content, and persuadees rarely state the stance shifts detected by probes. These results argue for evaluating multi-agent persuasion as a trajectory- and exposure-level process, using belief probes, exposure provenance, and action logs to identify who influenced whom and whether visible language reflects underlying stance movement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01120v3">New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 *Jay Bhan and Nicole Nobili contributed equally to this work as first authors, and their order was determined via coin flip
    </div>
    <details class="paper-abstract">
      The Zarankiewicz number $\textbf{Z}(m, n, s, t)$ is the maximum number of edges in a bipartite graph $G_{m, n}$ such that there is no complete $K_{s, t}$ bipartite subgraph. We determine for the first time the exact values of three Zarankiewicz numbers: $\textbf{Z}(11, 21, 3, 3)=116$, $\textbf{Z}(11, 22, 3, 3)=121$, and $\textbf{Z}(12, 22, 3, 3)=132$. We further establish lower bounds for 41 more Zarankiewicz numbers, including several that are within one edge of the best known upper bound, and we match the established value in four more closed cases. Our results are obtained using OpenEvolve, an open-source evolutionary algorithm based on Large Language Models (LLMs) that iteratively improves algorithms for generating mathematical constructions by optimizing a reward signal which we tailored for this specific problem. These findings provide new extremal graph constructions and demonstrate the potential of LLM-guided evolutionary search to contribute to mathematical research. In addition to presenting the resulting constructions, we report the generation algorithms produced, describe the relevant implementation details, and provide our computational costs. Our costs are remarkably low, at less than \$30 for each Zarankiewicz parameter combination, showing that LLM-guided evolutionary search can be an inexpensive, reproducible, and accessible tool for discovering new combinatorial constructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11051v2">OpenSanctions Pairs: Large-Scale Entity Matching with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We release OpenSanctions Pairs, the first large-scale public benchmark for entity matching on sanctions and OSINT data. The dataset includes 755,540 expert-labeled pairs over 1 million entities, aggregated from 293 source datasets across 45 jurisdictions. It captures real-world diversity in compliance data, spanning multiple languages and writing systems (e.g., Latin, Cyrillic, Arabic), inconsistent structure, and time-varying provenance, and is substantially more heterogeneous than prior entity matching benchmarks. As baselines, we evaluate the production rule-based matcher (nomenklatura RegressionV1) alongside open- and closed-source LLMs in both zero- and few-shot settings, each tested with and without MIPROv2 prompt optimization to control for prompt sensitivity. The rule-based baseline reaches 91.3\% F1; GPT-4o achieves the best result at 99.0\% F1, and a locally deployable open-source model (DeepSeek-R1-Distill-Qwen-14B) achieves 98.2\% F1. The rule-based baseline and LLMs fail in complementary ways: rules over-match, while LLMs struggle with cross-script transliteration. These results suggest that pairwise matching performance is approaching a practical ceiling and shift attention toward pipeline components such as blocking, clustering, and uncertainty-aware review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25085v1">MTDiag: A Multi-Turn Diagnostic Dataset Towards Clinically Meaningful LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 20 pages, published in the SIGDIAL 2026 conference proceedings, see https://aclanthology.org/2026.sigdial-1.58/
    </div>
    <details class="paper-abstract">
      Clinical diagnosis is fundamentally interactive and incremental, yet the dominant paradigm for evaluating Large Language Models (LLMs) in medicine remains static QA benchmarks or template-based dialogues. These benchmarks say little about whether a model can serve as a diagnostic agent in a dynamic clinical encounter, with LLMs showing significant accuracy and reliability degradation in multi-turn settings. To address this issue, we present MTDiag, a large multi-turn diagnostic dialogue dataset constructed from three heterogeneous sources: DDXPlus, MIMIC-IV, and published case reports (AJCR), covering common ED presentations as well as long-tail rare and atypical conditions. All cases are normalized into a canonical schema anchored in the most comprehensive and widely-adopted medical knowledge bases (UMLS concept identifiers, with ICD-10 diagnosis codes). We release the schema, a UserLM-8B-based utterance-generation pipeline, and the physician-validated dataset that converts structured clinical evidence into natural-language utterances. Importantly, we introduce and motivate clinical knowledge-grounded metrics for evaluating LLMs as diagnostic agents, beyond diagnostic accuracy, for the task of multi-turn differential diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25068v1">SHIFT-LLM: Distribution Shift Correction in Depth-Pruned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Depth pruning removes entire Transformer blocks to reduce the inference cost of large language models, but disrupts the hidden-state distributions expected by downstream layers, leading to significant accuracy loss. We introduce SHIFT-LLM, a training-free post-pruning correction framework that inserts a Linear Residual Adapter (LRA) at each pruning site. Each LRA preserves the identity pathway of the original residual block and adds a lightweight affine residual correction. This correction is calibrated via closed-form least-squares regression on a small held-out set, without gradient computation, to approximate the missing residual update produced by the pruned block. Together with the preserved identity pathway, the resulting LRA output approximates the hidden state produced by the original block, thereby mitigating the distributional mismatch introduced by layer removal while avoiding the expensive attention and feed-forward computations of the removed blocks. The resulting LRAs support low-rank factorization and exact merging across consecutive pruned layers for additional compression, and combine naturally with parameter-efficient fine-tuning for further recovery beyond fine-tuning the pruned model alone. Experiments on five model families, six layer-selection criteria, and seven zero-shot benchmarks show that SHIFT-LLM consistently recovers accuracy lost to depth pruning across most configurations, achieving gains up to +15.7 points on Llama-3.1-8B-Instruct while requiring only a few hundred calibration samples and no gradient computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25062v1">FLINT: Efficiently Leveraging High Bandwidth Flash for Capacity-Scalable LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      LLM inference is increasingly constrained by accelerator memory capacity rather than compute throughput. This constraint is especially acute in single-accelerator and small-node inference systems, where limited on-package memory capacity restricts the size of deployable models. HBF is an emerging 3D-stacked NAND flash technology that provides multi-terabyte near-accelerator capacity, making it a promising capacity tier for storing LLM weights. However, existing HBF-based proposals face three adoption challenges: they (1) rely on coarse-grained static prefetching for LLM weights aiming to hide the microsecond-level read latency of the NAND flash device while maximizing HBF's read throughput, (2) expose NAND flash management tasks (e.g., refresh operations) to the accelerator-visible critical inference path, and (3) miss optimization opportunities to specialize and optimize the flash-management mechanisms to the workload behavior. Our goal is to design an efficient HBF substrate that integrates HBF as a memory-capacity tier alongside HBM while addressing these three challenges. To this end, we propose FLINT, a workload-driven HBF substrate for capacity-scalable LLM inference. FLINT introduces three mechanisms: (1) a hardware burst-buffer controller that dynamically coalesces and pipelines HBF reads aiming to utilize existing NAND flash buffers while sustaining high HBF bandwidth, (2) a phantom-plane refresh mechanism, which removes refresh from the critical inference path by moving refresh-related NAND flash operations outside the read foreground back via low-cost resource duplication, and (3) a read-only FTL, which replaces SSD-class support for arbitrary writes with a compact table that translates logical weight bursts to physical HBF locations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25053v1">Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Accepted at the IEEE International Symposium on Workload Characterization (IISWC 2026)
    </div>
    <details class="paper-abstract">
      Edge LLM deployment is shaped by more than model size and precision: inference backend, hardware platform, memory traffic, and power management all affect latency and efficiency. We present Hydra, a common-schema, phase-aware workload characterization framework for LLM inference on edge SoCs. Hydra instruments HuggingFace Transformers and llama.cpp with a shared per-prompt timing schema and fuses those records with hardware telemetry, enabling a multi-dimensional characterization of performance, system-resource utilization, and efficiency across prefill and decode phases. Using Hydra, we evaluate three consecutive edge System-on-Chip (SoC) generations (AGX Xavier, AGX Orin, and AGX Thor), 13 instruction-tuned LLMs from seven families, five execution formats, and consider input/output-length sensitivity. The resulting artifact contains roughly 107K per-prompt records and is publicly released with Hydra. Our analysis shows that aggregate latency alone hides key deployment effects: backend structure changes where latency is introduced, quantization reduces memory traffic and energy but does not predict power monotonically, and SoC generation changes how utilization and efficiency should be interpreted. By connecting phase-level timing with system-resource utilization and efficiency metrics, Hydra enables reproducible, phase-aware characterization of edge LLM inference. Hydra's source code and the collected per-prompt trace corpus are available open-source at: https://github.com/amirtaherin/hydra
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25039v1">LifePlanner: Evaluating LLM Agents for Geo-spatial Planning with Social Media Data</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 25 pages, 7 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Geo-spatial planning, like trip design, is a realistic testbed for LLM agents because it requires grounded tool use, noisy evidence retrieval, and multi-constraint reasoning. Most benchmarks, however, only provide clean geospatial data and tools, missing the open-ended social signals that people use in daily planning. We introduce LifePlanner, a benchmark that enriches map data with large-scale local social media posts and provides access through an MCP toolset. LifePlanner provides an evaluation suite spanning four task categories and three difficulty levels. Experiments show frontier LLMs perform well on simple retrieval but degrade sharply on complex planning, with the Pass Rate dropping to 40.2%. Results show that failures mainly stem from incomplete evidence acquisition from such a large multimodal database, imprecise tool use, and weak constraint integration rather than model size or reasoning length, suggesting that future progress requires effective grounded planning instead of scaling alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19760v2">Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 52 pages, 6 figures. Pre-registered; frozen analysis plans and prompts included in the appendices
    </div>
    <details class="paper-abstract">
      Audited against policy-conditional ground truth from executed replay in a single-agent tool environment (ALFWorld), none of the step-level credit signals we audit -- LLM-judge scores, outcome-conditioned logprob ratios, or the policy's own confidence -- shows reliable incremental fidelity beyond its own marginal-matched shuffled control. Correcting for replay-target reliability leaves implicit fidelity bounded near zero and judge fidelity inconclusive at the achieved target reliability. Existing evaluations grade these signals against annotated step *correctness*; we audit them against step *contribution* -- what re-sampling the policy's own alternatives at each decision point and rolling forward changes about the outcome -- and they come apart. The ground truth is structured: 30.5% of decision points where it is defined exhibit a nonzero replay contrast at the achieved sampling resolution, and measurability is model-dependent -- the fraction of points with no policy-supported counterfactual differs twofold (13.1% vs. 26.8%) between two similar-scale policies. The failure mode is identifiable: implicit credit echoes the policy's fluency (median rank correlation +0.75, replicating at +0.70 in a second family under a corrected instrument), while outcome conditioning adds no causal information (partial correlation -0.004, Qwen). A confidence-only router recovers pivotal steps at chance level, but cuts judge cost by 13.1% per turn (14.0% per trajectory). In a seven-arm pre-registered training experiment, no arm reliably outperforms the untrained policy, and the checkpoints' apparent instrument signature is statistically consistent with mediation by effective training dose in this design -- sparser credit retains fewer examples, an order-of-magnitude spread in optimizer steps -- not credit content. Comparisons of credit rules must match effective sample size, or they measure dose, not credit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24857v1">Prompt Structure Redistributes, Not Reduces: An Empirical Analysis of Security-Weaknesses in LLM-Generated Python Code</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Accepted at CASCON 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly generate code from natural-language prompts, making prompt engineering a key mechanism for shaping the security of generated software. Structured and security-oriented prompts are widely used to encourage safer code, yet their effects extend beyond whether detected weaknesses are simply present or absent. Using 424 security-sensitive Python tasks, we generate solutions with GPT-4o and LLaMA 3.1-8B under five prompt variants that progressively add structural and security guidance, and evaluate them with Bandit and CodeQL along two axes: generation compliance and security weakness prevalence, severity, and CWE distributions. Structured prompting substantially reduces refusals (e.g., GPT-4o invalid outputs drop from 338 of 424 to 37-52), enabling large-scale analysis, but security-oriented refinements do not consistently reduce overall weakness prevalence. For GPT-4o, stronger prompts primarily redistribute risk: high-severity findings fall (20.8% to 13.6%) while low-severity findings rise (32% to 43.5%); LLaMA shows weaker, less consistent shifts. We also observe security-driven semantic drift, where stricter prompts silently remove or rewrite explicitly requested unsafe constructs. Overall, prompt structure improves compliance but is an unreliable substitute for robust security controls in LLM-assisted development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24825v1">A Dual-Dimensional LLM Framework for Automated Item Incidental Content Similarity Analysis in Large-Scale Assessments</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapid expansion of large-scale assessments and the growing adoption of automatic item generation have intensified concerns about incidental content redundancy, where construct-irrelevant elements such as wording or contextual framing become unintentionally repetitive across items. Traditional similarity metrics like BLEU or cosine similarity, often fail to capture the nuanced structural and semantic layers that drive perceived redundancy simultaneously. This study proposes a dual-dimensional framework for Automated Item Similarity Analysis (AISA) powered by Large Language Models (LLMs), operationalizing similarity through Structured Decomposition and Semantic Relatedness. Psychometric validation indicates that LLM-derived metrics align more closely with indicators of construct-irrelevant local dependence and yield more coherent item parameter groupings than traditional text-based measures. The framework is further evaluated through its application in Computerized Adaptive Testing (CAT). Simulations reveal that incorporating LLM-based similarity constraints into item selection improves estimation stability and reduces bias with minimal efficiency trade-offs, outperforming constraints based on conventional metrics. These findings highlight the potential of LLM-powered AISA to support scalable bank curation, content-aware test assembly, and experience-sensitive adaptive testing across diverse assessment contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24824v1">Constrained Entity Selection under Partial Knowledge for LLM-Based Knowledge Graph QA</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 8 pages, 2 tables. Submitted manuscript. Revised version published in KI 2026: Advances in Artificial Intelligence, LNCS 16830
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used for knowledge graph question answering (KGQA), but can fail to correctly ground answers in the underlying graph. Current approaches to LLM-based KGQA either rely on full semantic parsing into executable queries such as SPARQL, which is brittle in practice due to complex schemas or incompleteness of real-world KGs, or on LLM-reasoning and answer generation over KGs, which can be more robust but lacks formal guarantees. In this work, we study a complementary setting in which \emph{candidate} answers are generated by an LLM-based system and subsequently verified using lightweight symbolic constraints derived from the question. We introduce \emph{Constrained Entity Selection under Partial Knowledge (CES-PK)}, a problem formulation that focuses on eliminating invalid answers and providing symbolic support for valid ones without requiring construction of executable logical forms. To account for incomplete KGs, we employ a three-valued constraint semantics (\emph{satisfied, violated, unknown}) that avoids incorrect rejections under open-world assumptions. To demonstrate the effects of our method, we instantiate this framework over the Hetionet biomedical knowledge graph and evaluate the impact of type, relation, and exclusion constraints. Experiments show that precision improves by filtering invalid candidates, while recall is preserved due to retaining candidates whose constraints are not explicitly violated. Satisfied constraints provide additional positive symbolic evidence to rank remaining candidates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24807v1">Automatic Model Card Generation Using an LLM</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Model cards are structured documents that summarize key information about machine learning models to improve transparency, usability, and accountability. However, they often lack a consistent structure, and many models provide no model cards, making comparison and interpretation difficult. This paper presents two contributions. First, we propose MCTidy, an LLM-based approach that reorganizes existing model cards into a standardized template to improve clarity and comparability. Second, we introduce MCGenie, an LLM-based system that generates model cards directly from model repository data. We apply MCTidy to 48 Hugging Face model cards and evaluate information retention, section alignment, hallucination, and stability. Our findings show high information retention with minimal textual loss, accurate section assignment, rare hallucinations primarily in descriptive sections, and strong stability across runs. We assess MCGenie by generating model cards for the same 48 models and assessing semantic similarity, factual correctness, and sensitivity to input resources. The generated model cards achieved high semantic similarity (mean around 0.9); over half were fully correct, and most remaining errors were minor. Generation quality depended strongly on the availability of supporting resources, particularly associated papers. Overall, our findings demonstrate the potential of LLM-based methods to enable scalable, standardized model card documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19743v3">EngiAI: Capability-Based Evaluation of Tool-Connected LLM Agents for Engineering Design</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 42 pages, 17 figures, 15 tables. Extended version of work presented at the ASME 2026 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC-CIE 2026), Houston, Texas, USA, August 23-26, 2026. Submitted to the ASME Journal of Mechanical Design
    </div>
    <details class="paper-abstract">
      Engineering-agent systems are proliferating, but differences in tasks, tools, and success criteria make demonstrations difficult to compare and failures difficult to diagnose. We introduce a capability-based evaluation framework for tool-connected engineering agents. The framework separately evaluates workflow execution, retrieval-assisted parameter selection, high-performance computing (HPC) orchestration, and training-code authoring using execution traces and resulting engineering artifacts. We evaluate four LLM backends on the EngiBench Beams2D and Photonics2D problems using EngiAI, a LangGraph reference implementation. On Beams2D, the two proprietary models complete 96-97% of workflow tasks, compared with 55-78% for the two open-source models. Workflows requiring tool-based decision-making perform worse on Photonics2D, reaching 20-53% completion. Indexed retrieval improves parameter selection. For HPC orchestration, Gemini-3-Flash completes every tested pipeline, whereas GPT-5-mini completes 50-70% through the final evaluation step. When the agents must write the training code themselves, they fill small gaps reliably but diverge as larger regions are left open. In the open-synthesis tier, both proprietary models select conditional variational autoencoders instead of the reference conditional Generative Adversarial Network (cGAN); Gemini-3-Flash achieves lower maximum mean discrepancy than the reference in seven of ten Beams2D trials. The results support capability-specific evaluation, with separate scores for distinct skills, rather than assessment through successful end-to-end demonstrations alone. The framework provides common evaluation dimensions for identifying failure mechanisms and structuring comparisons of models, agent architectures, and tool interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10059v3">Strategic Exploitation in LLM Agent Markets: A Simulation Framework for E-Commerce Trust</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Agent-based modeling (ABM) has long been used in economics to study human behavior, and large language model (LLM) agents now enable new forms of social and economic simulation. While prior work has discovered strategic deception by LLM agents in financial trading and auction markets, e-commerce remains underexplored despite its distinctive information asymmetry: sellers privately observe product quality, whereas buyers rely on advertised claims and reputation signals. We introduce TruthMarketTwin, a controlled simulation framework for studying LLM-agent behavior in e-commerce markets. The framework is one of the first to model bilateral trade under asymmetric information sharing, where agents make strategic listing, purchasing, rating, and recourse-related decisions to optimize seller profit and buyer utility. We find that LLM agents released into traditional markets autonomously exploit weaknesses in reputation-based governance, while warrant enforcement reduces deception and reshapes strategic reasoning. Our results position LLM-agent simulation as a tool for studying institution-governed autonomous markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24658v1">Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Scaling test-time reasoning has substantially improved the problem-solving ability of large language models (LLMs), but standard autoregressive decoding still executes long reasoning traces sequentially, creating severe latency for difficult tasks (up to days and weeks). Parallel reasoning offers a natural remedy. However, prior systems primarily focus on Subtask Parallelism, where the model learns to decompose a high-level task into smaller chunks that can be solved independently. This approach overlooks another pervasive form of parallelism: Trial Parallelism, where multiple speculative attempts explore, verify, and aggregate competing hypotheses in parallel. In this paper, we introduce Parason, which reveals and learns both forms of parallelism in LLM reasoning. Our analysis identifies Trial Parallelism as the majority of parallelizable reasoning computation (65.5% in DeepSeek-V4's reasoning steps in HLE), and it becomes increasingly dominant on hard problems. Guided by this taxonomy, Parason converts sequential reasoning traces into structured parallel trajectories with a context-free grammar, then trains models with Parallelism-Aware Group Relative Policy Optimization (PA-GRPO), whose reward jointly balances accuracy, latency, and the two parallelism ratios. At inference time, Parason executes the learned parallel structure through tool calls, translating theoretical savings to real-world wall-clock acceleration. Experiments on mathematical reasoning benchmarks including AIME24 and AIME25 show that Parason achieves an average acceleration about 1.7$\times$ while maintaining competitive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24641v1">Aging of Prompt Engineering Techniques Across LLM Versions</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Accepted at ICSME 2026
    </div>
    <details class="paper-abstract">
      Prompt engineering and prompt engineering techniques (PETs) have become an integral part of software engineering for AI systems. However, new LLMs are released frequently and it remains unclear how the effectiveness of prompt engineering techniques changes across successive generations of Large Language Models (LLMs). To this end, we conduct a partial replication of the study by Khojah et al. (2025). We evaluate five techniques - Zero-Shot, Few-Shot, Chain-of-Thought (CoT), Contrastive Chain-of-Thought (CCoT), and an adapted version of Program-of-Thought (PoT) - on six instruction-tuned models grouped into three version pairs: GPT-3.5-Turbo/GPT-4o, Qwen2 7B Instruct/Qwen2.5 7B Instruct, and Mistral-7B-Instruct/Mistral-Large. We use a cleaned subset of the CodePromptEval dataset with 218 context-rich Python functions and 19,620 total generations assessed via pass@k-based functional correctness to evaluate model pairs on function-level code generation tasks. We show that prompt engineering "ages" in a model-family-specific way: Newer GPT models exhibit diminishing or even negative marginal gains from structured prompting, suggesting that instruction-following and reasoning scaffolds are increasingly internalized, whereas Qwen models continue to benefit substantially from Few-Shot and CCoT. Mistral models show mixed behavior with persistent gains from CCoT but attenuated benefits from CoT and PoT. Our results imply that effective prompting strategies must be adapted per model family and generation rather than transferred unchanged. This motivates future work on adaptive, model-aware prompting and broader, multi-dimensional code quality evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08626v3">How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 EMNLP 2026 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at semantic understanding, yet their ability to reconstruct internal structure from scrambled inputs remains underexplored. Sentence-level restoration is difficult to evaluate automatically because scrambled sentences often admit multiple valid reorderings. We introduce OrderProbe, a deterministic benchmark for structural reconstruction using fixed four-character expressions in Chinese, Japanese, and Korean, which have a unique canonical order and thus support exact-match scoring. We further propose a diagnostic framework that evaluates models beyond recovery accuracy, including Semantic Accuracy, Logical Validity, Structural Consistency, Robustness, and Information Density. Experiments on twelve widely used LLMs show that structural reconstruction remains difficult even for frontier systems: zero-shot recovery frequently falls below 35%. We also observe a consistent gap between meaning-oriented generation and exact structural reconstruction, suggesting that structural robustness is not an automatic byproduct of semantic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11668v3">HBF Sucks? A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      A faster storage device should make serving faster. We find the opposite. High-Bandwidth Flash (HBF) stacks NAND behind a wide, package-local interface, promising flash-scale capacity with far lower read latency and higher bandwidth than an SSD. The obvious move is to keep an SSD-style Mooncake KV-offloading stack and swap in HBF underneath. We built that system and measured it: an extended TokenSim, four complete two-hour Qwen-Bailian production traces, five dense and mixture-of-experts models, and H100/B200 profiles. The upgrade backfires. Average end-to-end latency rises 2--5.5$\times$ and maximum SLO goodput falls 1.1--2.7$\times$ across H100 and B200, so the faster device yields a slower system. A cost-benefit model explains the paradox: a faster far tier pays off only when read I/O is the bottleneck, reads outweigh writes, and delivered bandwidth is sustainable. Transient KV violates all three at once. Buying flash through the package costs GPU near-tier capacity and bandwidth, while HBF's own read/write latency barely matters: scaling it 3.75$\times$ moves latency less than 1\%. Worse, the two-tier hierarchy keeps reuse in the near tier and hands HBF a relentless write-heavy stream. Writes outnumber reads on every trace, so a 3D-ICE model shows the stack hits its thermal limit well below peak bandwidth, and a TLC tier wears out sooner than the SSD pool it replaced. The device is fine; the drop-in deployment is not. HBF sucks as an SSD replacement for transient KV, but earns its place in LLM serving when used selectively with reuse-aware placement, write budgeting, and thermal coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.05109v7">A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 20 pages, 11 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Translating users' natural language queries (NL) into SQL queries (i.e., Text-to-SQL, a.k.a. NL2SQL) can significantly reduce barriers to accessing relational databases and support various commercial applications. The performance of Text-to-SQL has been greatly enhanced with the emergence of Large Language Models (LLMs). In this survey, we provide a comprehensive review of Text-to-SQL techniques powered by LLMs, covering its entire lifecycle from the following four aspects: (1) Model: Text-to-SQL translation techniques that tackle not only NL ambiguity and under-specification, but also properly map NL with database schema and instances; (2) Data: From the collection of training data, data synthesis due to training data scarcity, to Text-to-SQL benchmarks; (3) Evaluation: Evaluating Text-to-SQL methods from multiple angles using different metrics and granularities; and (4) Error Analysis: analyzing Text-to-SQL errors to find the root cause and guiding Text-to-SQL models to evolve. Moreover, we offer a rule of thumb for developing Text-to-SQL solutions. Finally, we discuss the research challenges and open problems of Text-to-SQL in the LLMs era. Text-to-SQL Handbook: https://github.com/HKUSTDial/NL2SQL_Handbook
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24570v1">EviDx: Evidence-Aware Active Diagnosis with Scaffolded LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Clinical diagnosis is an active evidence-seeking process in which clinicians acquire evidence, update competing hypotheses, and decide when the available evidence is sufficient for diagnosis. Yet many medical diagnosis systems built around large language models (LLMs) still formulate diagnosis as static case-to-answer prediction, with limited support for evidence acquisition. Agentic LLMs offer a dynamic alternative through tool use and intermediate diagnostic trajectories, but existing systems often under-specify how patient evidence should be exposed, scaffolded, and controlled at runtime. We introduce EviDx, an evidence-aware active diagnosis framework that pairs patient-specific diagnostic environments with a clinical diagnostic scaffold and an observer-guided runtime harness. In EviDx, $\mathcal{E}$-Synthesis constructs interactive environments from raw clinical cases; the scaffold organizes role-specialized agents, evidence tools, and evolving evidence states; and the harness regulates diagnostic termination by tracking uncertainty and evidence coverage. A 3-level evaluation pyramid assesses execution robustness, reasoning dynamics, and diagnostic outcomes. Experiments show that EviDx improves diagnostic performance and process stability while revealing model-dependent capability boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24569v1">When "Must" Becomes "Maybe": Constraint Weakening in LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents coordinate complex tasks through multi-role and multi-stage workflows. Upstream state is repeatedly transformed into intermediate language artifacts, such as summaries, plans, tickets, memories, and handoff notes, from which downstream components act. For action-constraining state, topical retention is insufficient: an artifact may mention an unresolved condition while changing it from a requirement that must be resolved before execution into information that may merely inform the next action. We study this action-binding role as operational state preservation. Safety blockers provide a controlled instance because each source state has an explicit prerequisite, authority, fallback, and execution consequence. We condition on correct upstream identification, vary the handoff transformation, and evaluate an executor restricted to the resulting artifact. Across 1,296 controlled synthetic episodes, direct-handoff controls preserve every blocker, whereas compression, plan assimilation, convergence, ownership deferral, and precedent substitution repeatedly turn binding state into caveats or non-binding considerations. Normal handoff compression produces 100.0% deactivation and 54.2% forbidden action. Restoring all four state fields raises preservation to 100.0% and reduces forbidden action to 0.0%. Fixed-artifact interventions further separate preservation from containment: downstream verification eliminates forbidden action while artifact deactivation remains 95.3%. These results identify a state-transmission failure between information extraction and action. Handoff transformations can retain state content while weakening its constraints on downstream action. Semantic availability does not guarantee operational preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28791v2">Skill-Conditioned Gated Self-Distillation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Accepted by EMNLP 2026 Findings. Code is available at https://github.com/walawalagoose/SGSD
    </div>
    <details class="paper-abstract">
      On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19435v2">Ad Insertion in LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Sustainable monetization of large language models (LLMs) remains a critical open challenge. Traditional search advertising, which relies on static keywords, fails to capture the fleeting, context-dependent user intent---the specific information, goods, or services a user seeks---embedded in conversational flows. Beyond the standard goal of social welfare maximization effective LLM advertising requires contextual coherence (aligning ads semantically with transient user intent), computational efficiency (avoiding user-facing latency), and adherence to ethical and regulatory standards, including privacy preservation and explicit ad disclosure. Although recent solutions have explored bidding at the token and query levels, neither category holistically satisfies these constraints. We propose a framework that resolves these tensions through two decoupling strategies. First, we decouple ad insertion from response generation to facilitate pre-screening and explicit disclosure. Second, we decouple bidding from specific user queries by using ``genres'' (high-level semantic clusters) as a proxy. This allows advertisers to bid on stable categories rather than sensitive real-time responses, reducing computational burden and privacy risks. Applying the VCG auction mechanism to this genre-based framework provides approximate guarantees for dominant-strategy incentive compatibility (DSIC), individual rationality (IR), and social welfare. In synthetic experiments with $10^5$ advertisers and 100 candidate slots, VCG clears in approximately 1.25 seconds on a consumer-grade laptop. Finally, we introduce an ``LLM-as-a-Judge'' metric for estimating contextual coherence. Its predictions correlate with mean human ratings at Spearman's $ρ\approx 0.66$ and have a higher correlation with the leave-one-out group mean than 29 of 36 individual raters (80.6\%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24509v1">PeakBench: Benchmarking Resource-Aware Tool Invocation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      LLM agents increasingly solve tasks by invoking multiple tools, where parallel execution is essential for low latency but difficult to manage safely. Existing agent benchmarks primarily evaluate tool selection, argument generation, and end-to-end success under mostly serial execution, largely overlooking valid parallelization and resource-constrained scheduling. This missing scheduling dimension creates a practical failure mode: serial execution is safe but slow, while resource-agnostic parallel execution is fast but prone to avoidable resource overflows. To address this gap, we introduce PeakBench, a benchmark of executable multi-tool workflows with execution-grounded dependency annotations and measured resource profiles. A central challenge in evaluating such workflows is attribution: failures and inefficiencies may arise from incorrect dependency planning, poor resource-constrained scheduling, or both. PeakBench addresses this challenge with a two-part evaluation framework that disentangles logical planning from physical scheduling, with dedicated metrics for each dimension. Using this framework, we show that strong logical planning does not reliably translate into safe or efficient execution under resource constraints. We further show that exposing resource information can reduce avoidable overflows and improve resource utilization, making PeakBench a useful testbed for diagnosing resource-aware agent behavior. Code is available at https://github.com/Czzzk/Staggering-the-Peaks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24498v1">SeriCrypt: An LLM-Driven Context-Aware Serialization Framework for Cryptographic Protocols</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Constructing syntactically correct and cryptographically valid message sequences is essential for protocol state machine learning, conformance testing, and fuzzing. Unlike plaintext protocols, cryptographic protocols involve complex cross-message state dependencies and cryptographic computation constraints. Existing automated approaches predominantly target text-based or plaintext protocols, leaving cryptographic message construction largely manual. We present SeriCrypt, an LLM-driven, context-aware serialization framework for cryptographic protocols. It employs a large language model to extract field constraints, state dependencies, and cryptographic computation rules from unstructured protocol specifications into a unified structured intermediate representation, formally characterized by a domain-specific language for cryptographic protocols (CDSL). A protocol-agnostic execution engine parses CDSL declarations, automating field value resolution, cryptographic primitive invocation, and byte-stream serialization. As case studies in protocol security testing, we use the framework to construct violation messages targeting specification-defined security constraints and to support protocol fuzzing, evaluating it on mainstream implementations of TLS 1.2/1.3, IKEv1/v2, SSH, and TLCP. SeriCrypt generated message sequences accepted by all evaluated implementations and completed handshakes in every scenario. Security constraint testing revealed five specification violations, and fuzzing reached deeper protocol states with higher code coverage than mainstream fuzzers under the same time budget, demonstrating the framework's practical value for cryptographic protocol security testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24492v1">When Do Supervised UQ Ensembles Improve LLM Hallucination Detection? A Robustness Study</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods are widely used for hallucination detection in large language models (LLMs) in closed-book settings where ground-truth evidence is unavailable at inference time. Prior work has proposed combining UQ signals via learned ensembles, but empirical investigations into the robustness of these ensembles are limited. We study a supervised ensembling framework that trains a classifier over heterogeneous UQ-based scorer outputs on a small, domain-specific dataset of labeled LLM responses, then applies it to out-of-sample hallucination classification without retrieval, tools, or reference documents. Across four LLMs, nine datasets, and three generation regimes (short-form QA, long-form generation, and code generation), we provide a systematic robustness analysis along three axes: sample efficiency, in-domain dataset transfer, and generation regime dependence. We find that supervised ensembles outperform the best individual scorer in 30 of 32 settings, with gains realized from as few as 100 labeled instances. Ensembles retain most of their advantage in cases of in-domain transfer under distribution shift, outperforming the best non-ensemble scorer in 23 of 28 transfer settings. Sampling-based black-box ensembles are nearly as effective as full ensembles, while single-generation white-box ensembles offer limited benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.17535v2">Olapa-MCoT: Enhancing the Chinese Mathematical Reasoning Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 10 pages, 1 figures
    </div>
    <details class="paper-abstract">
      In the past two years, the outstanding performance of ChatGPT in multilingual and multitasking has led to large language models (LLMs) attracting widespread attention. However, restricted by expensive costs, many studies have to focus on the ability of only one major language. How can we quickly improve the model's capabilities in new languages without reducing its original capabilities under limited data and computing power? In this work, we focus on improving the Chinese mathematical reasoning capability based on Llama-2-13B, which is weak in Chinese mathematical reasoning. We proposed the Mathematical Chain of Thought method (Olapa-MCoT). First, we propose Similarity RRHF (SimRRHF), which adds the constraint of model optimization direction by introducing similarity loss based on RRHF. Furthermore, the novelty Incorrect Data Relearning (IDRL) method is designed, which improves the model's ability to learn difficult knowledge. The experiment achieves significant performance, with the accuracy of Chinese mathematical reasoning up to 50%, a 36% rise compared to Llama-2-13B-chat. In addition, the accuracy of English reasoning ability also increased by nearly 4%. It is worth mentioning that our method can be applied to any language major LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24419v1">A Judge Should Know What Changed:Construct Validity for LLM-as-a-Judge Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 39 pages, 10 figures, 11 tables
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge evaluation is usually assessed by agreement and robustness to surface perturbations, but reliability does not establish construct validity. We formalize construct validity for an evaluator as a two-dimensional profile: invariance S, the probability that a verdict is unchanged under construct-preserving edits, and construct sensitivity R, the probability that it changes under minimal construct-changing edits. We show that S and R are independent and that no scalar summary preserves all relevant comparisons. We measure the profile across 7 judges and 4 domains using 7 construct-changing intervention types and 5 register-only controls, with intervention direction determined by human annotators and generation, verification, and judging assigned to disjoint model families. At matched invariance S >= 0.90, judges average S = 0.945 but R = 0.319. Sensitivity also differs between scope and strength edits: R_scope = 0.383 versus R_strength = 0.262, a +0.121 gap with the same sign for all 7 judges. We further audit five public label sets and find that surface-only predictors reproduce 55%-67% of labels in paired mode, including 67.4% of MT-Bench human votes. These results show that high judge agreement can coexist with weak sensitivity to changes in the construct being evaluated, motivating joint reporting of invariance and sensitivity and auditing the validation set itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02041v2">EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are predominantly trained on English-centric data, resulting in uneven performance for smaller languages. We study whether continued pretraining (CPT) can improve Estonian capabilities in multilingual LLMs while preserving English and general reasoning performance. Using Llama 3.1 8B and Apertus 8B as base models, we apply CPT with Estonian-enriched multilingual replay, followed by mostly English supervised fine-tuning, preference optimization, and chat vector merging. Evaluation on Estonian benchmarks, targeted pairwise human evaluation, and an Estonian Chatbot Arena-style setup shows consistent improvements in Estonian language competence, reasoning, translation, and instruction-following. Although Apertus exhibits stronger Estonian capabilities before adaptation, the more English-centric Llama model achieves substantially larger gains after adaptation. While some English capabilities regress relative to the original instruction-tuned models, chat vector merging substantially restores English instruction-following and reasoning performance. These findings suggest that CPT with balanced multilingual replay and lightweight post-training alignment can substantially improve single-language capabilities in multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24363v1">Rethinking Semantic Alignment in LLM-Enhanced Collaborative Filtering: A Spectral Decoupling Approach</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Recent advances in LLM-enhanced recommendation commonly align semantic representations with collaborative embeddings in a shared space, yet how alignment affects LLM-encoded information remains unclear. In this work, we revisit LLM-enhanced recommendation from a spectral perspective and show that collaborative and semantic signals benefit from different spectral parts. While collaborative representations are dominated by smooth low-frequency components due to user-item homophily, semantic embeddings contain useful non-principal singular components. Through component-wise evaluation and training-dynamics analysis, we find that alignment increasingly concentrates learned representations in dominant collaborative and principal semantic subspaces, reducing overlap with non-principal semantic components. Controlled comparisons show that non-principal components provide inconsistent gains under alignment but consistently improve performance through component-level decoupling, while full prediction-level decoupling achieves the best overall performance. These results indicate that alignment fails to effectively exploit complementary non-principal semantic information. Motivated by these findings, we propose UniSpecRec (Unifying Spectral Signals for Recommendation), which applies signal-specific spectral filtering while preserving collaborative and semantic representations in their respective spaces. UniSpecRec combines their predictions without cross-space alignment or additional trainable parameters. Extensive experiments demonstrate its effectiveness, efficiency, and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24314v1">Benchmarking LLM Judges for Voice-Agent Evaluation: Reliability, Calibration, and Human Oversight</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Extends LLM-as-a-Judge to voice agents across telecom and retail, testing GPT-4.1 and GPT-5 against human raters across 10 safety and efficiency metrics. A correlation-based calibration analysis reveals domain-dependent reliability and identifies Recovery Turn Count and safety-recall metrics as unreliable for fully automated judging
    </div>
    <details class="paper-abstract">
      Evaluating conversational voice agents at scale re- quires reliable assessment methods that capture both observ- able interaction quality and the contextual judgment typically provided by human evaluators. We investigate LLM-as-a-Judge evaluation by comparing human judgments with GPT-4.1 and GPT-5 on telecom and retail voice-agent conversations, across conversational quality and safety dimensions. The same interac- tions are scored under three evaluation configurations, p0, p1, and p2, to test whether automated judgments are sensitive to the evaluation setup and whether observed patterns generalize across configurations and judge models. Beyond aggregate agreement, we examine metric-level correlations, evaluator consistency, and systematic human-LLM disagreement to identify which conver- sational attributes can be judged reliably by automation and which remain sensitive to interpretation and context. Effective voice-agent evaluation is also shaped by pipeline-level factors such as speech generation, streaming, and error propagation across ASR, reasoning, and tool-calling stages, motivating our focus on comparing how human and LLM judges score the same interactions end to end. Our results show that LLM- based evaluation can serve as an effective component of large- scale voice-agent assessment, but that its reliability is metric- and configuration-dependent rather than uniform. This pro- vides an empirical framework for identifying which metrics suit automated evaluation and supports hybrid pipelines in which LLM judges handle scalable assessment while human evaluators remain engaged for metrics that demand contextual interpretation and higher-confidence judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24297v1">When AI "Works," When Does Help Begin?: Intergenerational Support Around Older Adults' LLM Usage</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 5 pages, 1 table, Accepted by CSCW 2026 workshop Growing Up (and Old) with AI: Co-Constructing the Future for Family-Centered AI
    </div>
    <details class="paper-abstract">
      LLMs are becoming part of everyday life, including for older adults (OAs). OAs often learn digital technologies with younger family members, who have traditionally served as "warm experts" providing trusted and personalized operational help. LLMs expand this role: family supporters may also help OAs judge appropriate uses, consider what information to disclose, assess the credibility of outputs, and decide when AI-generated advice is safe to act on. We conducted a formative qualitative study with six OAs and seven younger adults (YAs), using semi-structured interviews and scenario-based think-aloud activities. OA participants described using LLMs to lighten their recurring reliance on family, while preserving family as a selectively invoked support channel. However, because LLMs rarely produced visible operational breakdowns, YAs had limited signals for when support was actually needed. Instead, YAs relied on OAs' partial disclosures and negotiated intervention through general warnings and self-imposed action boundaries. As a result, family support often solved an immediate problem without leaving reusable calibration knowledge for future use. Based on these findings, we propose design implications for intergenerational LLM support (e.g., consentful help requests, learning-oriented family support that preserves OA task ownership).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24271v1">Observability and Fault Injection for LLM-Based Multi-Agent Systems in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Large Language Model-based multi-agent systems are increasingly explored for software engineering tasks, but they remain difficult to inspect, debug, and evaluate under controlled failures. We present llmmas-otel, a lightweight and framework-agnostic tool that combines OpenTelemetry-based distributed tracing with fault injection for LLM-based multi-agent systems in software engineering workflows. The tool instruments agent executions with trace-aligned telemetry across workflow phases, agent steps, inter-agent communication, tool calls, and LLM invocations, and supports targeted fault injection at selected interaction points. This makes it possible to compare baseline and faulty executions in a reproducible way and inspect the effects through aligned traces and run artifacts. We describe the motivation, architecture, implementation, current capabilities, and initial validation of the tool on a minimal demo workflow and a real LLM-based multi-agent system for software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24269v1">Towards LLM-Enhanced Android Taint Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Taint analysis is a fundamental technique for detecting sensitive data leaks in Android apps. However, traditional static tools, such as FlowDroid, still face well-known challenges due to the complexity of accurately modeling the Android framework. In this paper, we investigate whether off-the-shelf Large Language Models (LLMs) can effectively reason about taint flows in Android apps. Our preliminary approach relies on an agentic interaction strategy, enabling the LLM to iteratively explore code and reason about data flows. We conduct an initial evaluation on the DroidBench benchmark against FlowDroid, where our approach outperforms the baseline: Gemini-3 Flash achieves an F1-score of 0.96, compared to 0.55 for FlowDroid. In particular, we observe improvements in challenging categories such as inter-component communication (0.95 vs. 0.17), implicit flows (0.94 vs. 0.00), and reflection (1.00 vs. 0.50), where FlowDroid typically struggles. On a small set of real-world apps, the LLM-based approach also identifies additional potential data leaks not reported by FlowDroid. These preliminary findings suggest that LLM reasoning may effectively complement traditional static taint analysis, motivating future research on hybrid LLM-enhanced taint analysis pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24252v1">SA-Bench: Evaluating Semantic Alignment in LLM-Based Paper Reproduction</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      LLM agents can generate paper reproduction code, yet often produce scientifically unfaithful implementations. We define this failure mode as semantic drift, where generated code silently diverges from the paper's specifications. We introduce SemanticAlign-Bench(SA-Bench), a diagnostic benchmark covering 30 papers from ICLR, ICML and NeurIPS 2025. For each paper, we decompose its specifications into atomic and verifiable implementation claims, which we call Semantic Alignment Units (SAUs) and evaluate repositories along four diagnostic dimensions spanning numerical, methodological, protocol and ordering drift. In total, we construct 1,491 SAUs across five ML domains and evaluate 12 generator configurations (4 models $\times$ 3 scaffolds). Even the strongest configuration (Claude+PaperCoder) achieves a mean SAU score of only 0.301 out of 1.0, with an overall mean of 0.221 across 360 evaluations. A failure taxonomy reveals that agents attempt most requirements but implement them incorrectly, with implementation mismatch and stubs accounting for the majority of zero-scored claims. Our analysis further indicates that scaffolds optimized for executability provide limited leverage for scientific reproduction; narrowing the gap requires scaffolds that prioritize semantic specification verification. The benchmark, annotations and evaluation pipeline are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24231v1">RecurSE: Bounded Recursive Self-Evaluation for LLM Rubric Judges</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      LLM-as-judge is essential for evaluating open-ended text and steering post-training, yet improving the judge itself typically relies on expensive annotations, reward models, or distillation from stronger teachers. In this work, we eliminate external gold supervision from the RL training reward: the model's own evaluative capability generates learning signals for its optimization -- a closed-loop setting of bounded recursive self-improvement (RSI) termed Recursive Self-Evaluation (RecurSE). We study two central questions: when can self-improvement occur, and when must it stop? First, RecurSE pairs a trainable judge evaluating candidate responses under per-rule rubrics (Pass 1) with a synchronized policy-copy checker that audits the judge's reasoning against meta-rubrics to supply a scalar process reward (Pass 2). To enable learning, interface decoupling structurally isolates the checker's scalar score from the judge's verdict tokens, eliminating a degenerative token-copying shortcut that inflates self-assigned rewards. Second, because unanchored recursive learning is inherently bounded, Pairwise Advantage Validity (PAV) serves as an unbiased validation monitor that jointly tracks judge accuracy and checker fidelity to reliably identify the optimal early-stopping window. Across Qwen3.5-9B, Gemma-4-E4B-it, and Qwen3.6-27B, RecurSE achieves consistent generalization gains across held-out medical, pairwise, summarization, and professional benchmarks. Ablations demonstrate that synchronized judge-checker co-evolution outperforms frozen checkers, external meta-judges, self-consistency, and scaled teacher distillation. Furthermore, preference pairs curated by our judge effectively enhance downstream policy alignment. Bounded RSI for LLM-as-judge is thus viable when self-produced reward validity is explicitly decoupled and monitored.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06268v2">MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 19 pages, 7 figures, 18 tables. Accepted to Findings of EMNLP 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly integrated into clinical workflows. However, prompt injection attacks can steer these systems toward clinically unsafe or misleading outputs. We introduce the Medical Prompt Injection Benchmark (MPIB), a dataset-and-benchmark suite for evaluating clinical safety under both direct prompt injection and indirect, RAG-mediated injection across clinically grounded tasks. MPIB emphasizes outcome-level risk via the Clinical Harm Event Rate (CHER), which measures high-severity clinical harm events under a clinically grounded taxonomy, and reports CHER alongside ASR$_2$ to distinguish moderate-or-worse from high-severity outcomes. The benchmark comprises 9,697 curated instances constructed through multi-stage quality gates and clinical safety linting. Evaluating MPIB across a diverse set of baseline LLMs and defense configurations, we find that ASR$_2$ and CHER$_3$ can diverge substantially, and that observed rates vary depending on whether adversarial instructions appear in the user query or in retrieved context. The MPIB dataset and evaluation code are available on Hugging Face and GitHub, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24228v1">Evaluating Multiple LLM Generations with Validated Task Coverage</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Many LLM applications are most useful when they provide several candidate outputs for comparison, validation, or combination. Predominant evaluation settings, however, still focus on individual outputs or reduce multiple samples to a single success or selected answer. This can miss whether the outputs include several genuinely different useful results. We introduce VTC-Bench, a five-domain benchmark for this setting, together with Validated Task Coverage (VTC) as its core evaluation quantity. The benchmark is built from carefully selected real-data tasks where both output quality and task-relevant distinctness can be checked automatically and reproducibly, without model-based judges. VTC measures how many distinct useful results are obtained within $k$ attempts. Across multiple models and inference settings, the benchmark leads to different conclusions from conventional evaluation: configurations that look strongest from single-draw quality are not necessarily those with the best coverage, and simple measures of output variation do not reliably recover task-relevant coverage. These results show that finite candidate sets can be evaluated directly as objects of interest, revealing differences in model behavior that are not apparent from conventional per-output evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14646v2">ALT4Decompile: Inferring C-aligned Abstract Loop Tree for LLM-Based Binary Decompilation</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Decompilation refers to the process of recovering high-level (C) language code from low-level (assembly) code. Recent Large Language Model (LLM)-based methods can generate re-executable code but struggle with the enormous syntax gap between assembly and C languages. Loop structures in C are fragmented into complex jumps within assembly languages, which poses a challenge to LLMs decompiling assembly language, which is mainly pre-trained on C. To bridge this gap, we propose ALT4Decompile, which refactors assembly to closely resemble C language structure, thereby preventing hallucinations in LLMs caused by complex jumps. Specifically, we construct a C-aligned \underline{\textbf{A}}bstract \underline{\textbf{L}}oop \underline{\textbf{T}}ree (ALT) based on specific jump patterns of Assembly that explicitly aggregates fragmented Assembly blocks into corresponding high-level loop structures (e.g., nested loops). Finally, we fine-tune an LLM adapted to ALT to generate decompiled code and then improve its output by fixing specific errors and restoring symbols. Evaluated against 12 baselines (including rule-based SAILR and LLM-based LLM4Decompile) on Decompile-Eval, MBPP, and ExeBench, ALT4Decompile achieves state-of-the-art correct re-executable results (even under four commonly used obfuscation techniques): i.e., a 70.4\% test case pass rate on Decompile-Eval, a 10.6\% improvement over prior best work. Furthermore, analyses on real-world software and a user study further show its practicability in understanding binary functions. The datasets and source code used in this paper are available at https://github.com/wang-yongpan/ALT4Decompile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24191v1">'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Urdu, the world's tenth most spoken language with 246 million speakers, remains almost entirely absent from mainstream LLM safety evaluation and nine years of WOAH proceedings. To investigate whether this absence has measurable consequences for content moderation reliability, five large language models, GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash, Qwen-2.5, and Llama-3.1, were tested across six datasets spanning Nastaliq Urdu, Roman Urdu, English, and code-switched Urdu-English. Across the five Urdu-script datasets, label instability between original-script and English-translation classification ranged from 15.9% (Gemini 2.5 Flash) to 31.6% (Qwen-2.5), with a 'Missed-in-Urdu' rate, content flagged as harmful in English translation but passed as normal in the original script, ranging from 2.4% to 9.9% (median 4.3%). A complete enumeration of all 205 papers across nine ALW/WOAH editions via the ACL Anthology API confirms zero dedicated Urdu papers across the entire period. Results indicate that current LLMs provide uneven safety assurance across Urdu's script varieties, with smaller open-weight models showing substantially higher instability and missed-harm rates than frontier closed models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24156v1">LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Industrial actor--critic methods usually represent continuous actions as anonymous numerical coordinates. They must therefore learn from limited interactions which process variables each action affects, in which direction, and after what delay. Fixed industrial documents already describe part of these relations, but their open-text statements neither represent the current operating condition nor directly fit a numerical policy. This article presents LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes (LCAE), which uses a large language model before training to normalize fixed documents into a frozen action--observation--direction--delay relation basis. Recent numerical action--response history then modulates the current strength of each relation, while the evaluated action forms a state-conditioned nonlinear action-effect field in the same basis. The critic evaluates actions through this field, and the actor uses the same relation gains to generate actions, making document semantics part of maximum-entropy policy learning. Neither the LLM nor the embedding model runs online during training or deployment; the deployed policy uses only frozen semantic artifacts and visible numerical history. The method states a falsifiable hypothesis: when documented relations are correct and recent history reflects their contextual strength, this action representation should provide a more useful decision bias than raw action coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24145v1">TrustDABench: Benchmarking Reliability and Robustness of LLMs for Structured Data Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Code&Data: https://github.com/Skyorca/TrustDABench
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to analyze spreadsheets, CSV files, and other structured data, but producing a correct-looking answer is not the same as producing a trustworthy analysis. A trustworthy result should be supported by a valid path from the user question to the relevant data evidence. This requirement creates two diagnostic questions: whether an LLM can refuse to answer or ask for clarification when such a path does not exist, and whether it can preserve the correct analysis when the same evidence is expressed in different table forms. We introduce TrustDABench, a benchmark that operationalizes these questions as reliability and robustness. Starting from the evidence-path view, we derive 19 perturbation operators and instantiate them through an Agentic-LLM-based generation framework. TrustDABench contains 2,340 human-verified perturbed instances, and we evaluate eight representative LLMs. The results show substantial headroom: the best reliability result is only 24.21% average MRS, achieved by GPT-5.5, while the best robustness result still has 9.10% average ASR, achieved by Claude-Sonnet-5. The failures are systematic: models rarely detect conflicting evidence, often continue along executable but unsupported analysis paths, and remain sensitive to perturbations that change observation boundaries or cross-table relations. These findings suggest that stronger evidence-boundary recognition and representation-invariant reasoning are still needed for reliable structured-data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.26195v1">Cost-Utility Alignment in LLM Agent Trajectories:Profiling,Attribution,Diagnosis,Adaptation,and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 12pages,3 figures
    </div>
    <details class="paper-abstract">
      LLM agents execute tasks through multi-step trajectories that accumulate cost in tokens, latency, monetary fees, and environmental risk while producing utility only at the aggregate task level. Prior surveys address inference optimization, agent capabilities, or evaluation in isolation, leaving practitioners without principled tools to determine whether a trajectory's resource expenditure is justified by its task contribution. We address this gap by developing a trajectory-centric cost-utility alignment framework that treats resource consumption and task contribution as dual ledgers over the same execution, organized around five analytical stages: cost profiling, utility attribution, misalignment diagnosis, targeted adaptation, and evaluation. Utility attribution is central to this structure: rather than relying on aggregate outcomes, it organizes contribution methods by evidential strength, from process proxies and information dependency to counterfactual replay, supplying the causal evidence that grounds diagnosis and guides adaptation. Using this framework, we analyze recent agent systems, attribution methods, and evaluation protocols covering efficiency, reliability, and economic value, as well as five forms of misalignment spanning cognitive and context use, external interaction, recovery-loop control, resource-capability allocation, and multi-agent coordination, together with their targeted adaptations. The result is a closed analytical loop connecting the cost side of agent execution to its utility side, providing a structured basis for resource-aware agent design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24113v1">Structured Frequency-Domain Evidence for LLM-Based Time-Series Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Time-series anomalies can appear not only as pointwise deviations but also as changes in recurring temporal structure, such as shifted periodicity or localized oscillatory fluctuations. However, existing LLM-based time-series anomaly detection methods mainly expose time-domain evidence through indexed values, plots, or de-seasonalized representations, leaving spectral structure implicit. We propose an evidence-augmented zero-shot TSAD framework that preserves indexed de-seasonalized observations while adding compact frequency-domain evidence computed with the Fast Fourier Transform (FFT). The evidence is constructed at two resolutions: global frequency-domain evidence summarizes sequence-level periodic context, while local frequency-domain evidence captures time-localized spectral departures. Experiments on AnomLLM with InternVL2-LLaMA3-76B, Qwen2.5-VL-72B-Instruct, Gemini-2.5-Flash, and GPT-4o, together with evaluation on the TSB-AD-U subset, show that explicit frequency-domain evidence improves LLM-based TSAD baselines. These results suggest that frequency-domain evidence can complement indexed and de-seasonalized time-domain inputs for zero-shot LLM-based TSAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24087v1">Knowing When to Ask for Help: Bayesian Self-Escalation in Hierarchical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 21 pages, 5 figures. Code and data: https://github.com/nadeem-shaikh/llm-self-escalation . Associated record: https://doi.org/10.5281/zenodo.21330787
    </div>
    <details class="paper-abstract">
      Current LLM agent systems decide delegation before reasoning begins (a router picks a model) or after a response is complete (a verifier scores it and may retry). We study a third regime: an agent that recognises, during its own reasoning, that it is unlikely to succeed and transfers control to a stronger model. We formulate intra-generation delegation as a Bayesian optimal-stopping problem over a learned competence posterior -- an online estimate of the agent's eventual task success whose sufficient statistics are learned from labelled trajectories, not read off raw entropy. We derive the myopic escalation threshold in closed form, characterise the optimal policy via dynamic programming, and prove that the optimal policy is a time-varying threshold with no shape assumption on the raw signal. We further prove exponential separation of the oracle belief at the Chernoff-information rate of the signal, a regret bound governed by the calibration of the posterior, and a finite-sample guarantee: with n labelled calibration trajectories the deployed plug-in policy's regret decays as 1/sqrt(n). A controlled simulation study confirms each prediction of the theory, including the predicted 1/sqrt(n) rate. We additionally report a real-model validation on a Qwen2.5-Coder 1.5B->7B code cascade (MBPP, 257 tasks), confirming two of three pre-registered predictions: the escalation frontier dominates post-hoc routing at equal cost, and the cumulative competence belief's discrimination rises over generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24086v1">EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as code agents for scientific and engineering analysis, but their ability to analyze raw physical-layer measurements remains untested. We introduce \textbf{EMRB} (\textbf{E}lectro\textbf{m}agnetic \textbf{R}easoning \textbf{B}enchmark), which evaluates whether LLMs can analyze raw I/Q data by writing and running code. EMRB contains 200 problems across five difficulty levels and 27 question types, from signal detection to OFDM design, generated from 11 signal types with verified ground truth. Unlike benchmarks built on preprocessed features or structured tables, EMRB provides only the raw capture; the quantities each question refers to must first be discovered through code. We evaluate 14 LLMs spanning proprietary, open-weight, and reasoning-oriented families. Scores range from 24.1\% to 78.9\%, with the mean dropping from 84.9\% on basic measurement to 21.2\% on system design. We also propose \textbf{ReconPilot}, a structured method that separates signal reconnaissance, targeted analysis, and self-verification. Across three backbones, ReconPilot raises the overall score by 3.8 to 17.6 points and improves 13 of 15 backbone-level combinations tested. All data and code are publicly released in \href{https://github.com/mingxuZhang2/EMRB}{\textcolor{blue}{our GitHub repository}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.20073v3">A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that requires generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without task-specific fine-tuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both numerical solutions and detailed reasoning rationales. By grounding all explanations in verified computational outputs, STReason inherently suppresses factual hallucinations common in direct-prompt LLM approaches. To enable rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results demonstrate that STReason significantly outperforms advanced LLM baselines across all reasoning metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29717v3">Optimizing Expert-Designed Crystal Graph Networks for Band-Gap Prediction with an Autonomous LLM Research Loop</a></div>
    <div class="paper-meta">
      📅 2026-08-25
    </div>
    <details class="paper-abstract">
      Predicting a material's properties from its structure is a central, fast-advancing problem in computational materials science. A decade of work has produced standard public benchmarks and many published machine-learning models for the task (Dunn et al., 2020). The task's fixed metric and these baselines make it a natural setting for autonomous agent research (Karpathy, 2026). On the MatBench band-gap benchmark ($>$100k crystals), a general-purpose coding agent autonomously built the most accurate model trained without external pretraining, ahead of all seventeen expert-designed models reported for the task. A closer analysis shows it reached this by implementing known methods: either already standard in crystal neural-network models, or borrowed from other areas of machine learning. The contributing implementations include element-pair features on each message-passing edge and a crystal space-group embedding. The work not only demonstrates that LLM-agent autonomous research can optimize an expert-designed machine learning model for material property prediction, but also investigates the limitations of such autonomous research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24080v1">When Less Is More: An Empirical Study of Minimal Responses in Counseling Dialogues and the Behavior of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-25
      | 💬 Camera-ready version. Accepted to EMNLP 2026 Main Conference
    </div>
    <details class="paper-abstract">
      In psychological counseling, effective support is not always delivered through long, information-rich responses. Minimal responses, such as backchannel cues and concise empathic statements, help convey attentive listening, express empathy, and encourage clients to continue expressing themselves. However, existing counseling dialogue systems and evaluation frameworks often favor explicit, content-rich replies, overlooking the interactional value of brief counselor utterances. This paper presents a systematic cross-lingual analysis of minimal responses across multiple counseling dialogue datasets. We develop a two-stage filtering method based on utterance length and content, followed by contextual verification using a large language model (LLM). Our analysis shows that minimal responses are common in human-collected datasets but substantially underrepresented in LLM-generated ones. We further evaluate current LLMs in manually curated dialogue contexts where human counselors used minimal responses. The results show that strong commercial LLMs are capable of generating minimal responses when explicitly instructed, but still struggle to determine when such responses are appropriate. Counseling-specific models trained on synthetic data perform particularly poorly, tending instead to produce longer and more information-rich responses. Moreover, LLM-based response-quality evaluation may undervalue minimal responses, even when they are interactionally appropriate.
    </details>
</div>
