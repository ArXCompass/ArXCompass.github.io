# llm - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30335v1">Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 25 pages, 7 figures, 24 tables. Preliminary versions to appear at the ICML 2026 Workshops on Combining Theory and Benchmarks (CTB), Statistical Frameworks for Uncertainty in Agentic Systems (AgenticUQ), and Failure Modes of Agentic AI (FAGEN)
    </div>
    <details class="paper-abstract">
      Multi-component LLM agents assemble probabilistic claims from components that each see only part of a joint problem; the composition can violate basic probability axioms even when every component is locally coherent. We formalise this locally coherent, globally incoherent failure via the compositional residual eps*, the L2 distance from the composed quote to the joint coherent polytope, computable at runtime from system output and the declared cross-component coupling constraints. A product-structure dichotomy characterises when local coherence suffices, and a Rayleigh-quotient prediction matches the observed residual within 7% on three of four relation classes. A hierarchical Boyle-Dykstra projection repairs the composition deterministically; an anytime-valid e-process gives sequential coherence monitoring. Across 1,876 ensemble cliques on a four-LLM mid-tier panel (frontier-panel rerun in Section 5.5), eps* > 0 on 33-94% of cliques, translating to +0.115 nats per bet of regret on 1,770 resolved bets under the proportional allocation rule (the gain collapses to +0.006 under bettors that themselves coherentise). Three intuitive LLM-side mitigations(retrieval, partition-aware prompting, aggregator-LLM) each fail or regress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30334v1">Demystifying Data Organization for Enhanced LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various fields, yet their training efficiency is heavily reliant on effective data curation. While data selection has been widely studied, the strategic data organization for enhanced training remains an underexplored area, particularly since current LLMs are often trained for only one or a few epochs. This paper systematically explores the influence of data organization on LLM training by reusing pre-computed sample-level scores originally generated for data efficiency, thereby incurring minimal additional computational overhead. We identify and formalize four key guidelines for optimizing data organization: Boundary Sharpening, Cyclic Scheduling, Curriculum Continuity, and Local Diversity. Guided by them, we introduce two novel data ordering methods termed STR and SAW. Extensive experiments across different model scales and data sizes, encompassing both pre-training and SFT stages, validate the effectiveness of our summarized guidelines. They also demonstrate the robustness of our proposed data ordering methods in enhancing the stability and performance of LLM training. Github Link: https://github.com/microsoft/data-efficacy/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30315v1">Resolution Diagnostics for Paired LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 16 pages, 7 figures, 12 tables. Accepted to the ICML 2026 Workshop on Hypothesis Testing, Seoul, South Korea, 2026. Copyright 2026 by the author(s)
    </div>
    <details class="paper-abstract">
      Across two public LLM leaderboards, many displayed pairwise rankings do not meet a conventional paired-test resolution target under the actual paired evaluation design: 11 of 40 Open LLM Leaderboard v1 pairwise comparisons and 4 of 9 MMLU-Pro top-10 adjacent-rank pairs are unresolved at (alpha, 1-beta) = (0.05, 0.8). The MMLU-Pro count rises to 6/9 under real subject-level clustering and stays at 5-6 out of 9 in 99.9% of category-bootstrap resamples. We frame paired LLM evaluation as a hypothesis-testing problem, invert level-alpha, power-(1-beta) tests, and report a per-pair resolution ratio q = N/N* as the primary diagnostic. A sharp small-effect expansion with an explicit second-order constant shows that the widely-used unpaired Cohen-h-plus-(1-rho) shortcut deviates from the correct N* by approximately a factor of two in the close-comparison regime, a deficit that three of five off-the-shelf calculators(Cohen 1988, G*Power, R pwr) silently inherit when the user post-multiplies their per-arm output by (1-rho). The unresolved-pair pattern remains under multiplicity correction and anytime-valid sequential testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30284v1">ProjectionBench: Evaluating Scientific Hypothesis Generation in LLMs Under Progressive Information Disclosure</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 19 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Scientific discovery is an inherently creative and uncertain process, requiring reasoning beyond the recall of known knowledge. While many benchmarks have been proposed to evaluate large language model (LLM) performance on deep research tasks via multi-hop retrieval, their innovative reasoning abilities essential for true scientific discovery remain largely untested. We introduce a benchmark framework for evaluating model performance in scientific discovery and reasoning, building up from a raw problem to the classical null hypothesis test. In our framework, models initially receive only the topic and research question from a recent paper, with technical details progressively revealed. At each stage of information disclosure, the model is tasked with generating hypotheses that address the research question, which is compared with the conclusions from the original paper and evaluated via automated semantic similarity of constituent atomic claims. This progressive evaluation of semantic divergence from ground-truth conclusions enables assessment of a model's innovativeness (under minimal information) to grounded reasoning capabilities (under full experimental details), both critical for using LLMs for scientific discovery purposes. Our framework provides a foundation for systematically evaluating scientific reasoning and discovery capabilities in LLMs, crucial for advancing the development of next-generation AI scientist/co-scientist systems. Specifically, here we evaluate GPT-5, GPT-5.4, Gemini 2.5 pro, and Gemini 3.1 pro preview across 45 papers spanning bioactive materials, mechanical materials, and nanomaterials. We find that GPT-5.4 and Gemini 3.1 pro outperform their previous generation counterparts as expected, and GPT-5.4 in particular maintains 0.7 F1 score alignment with ground truth conclusions even under minimal context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30273v1">LLUMI: Improving LLM Writing Assistance for Mental Health Support with Online Community Feedback</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in generating supportive responses for mental health queries, but improving their usefulness, empathy, and safety often requires substantial compute, expert input, and labeled data. At the same time, deploying proprietary, cloud-based models for mental health-related interactions raises important privacy and data-governance concerns, given the sensitivities. To address this challenge, we introduce LLUMI setup that can be hosted in-house within protected environments. LLUMI consists of two complementary components: a generation model (GM), which drafts supportive responses to mental health queries, and an improvement model (IM), which revises an initial human-crafted response. We leverage feedback signals from Reddit mental health communities, using community endorsement patterns such as upvotes and downvotes to construct chosen-rejected response pairs for Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO). We further align LLUMI using human evaluation across five dimensions: readability, empathy, connection, actionability, and safety. Our results show that, despite relying on smaller open-source models rather than proprietary cloud-based GPT models, LLUMI achieves comparable performance across linguistic analyses and human evaluations. These findings suggest that open-source models, when trained with community-derived preference signals, can support high-quality mental health support assistance while offering a more privacy-preserving alternative for sensitive support contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30260v1">How LoRA Remembers? A Parametric Memory Law for LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) must continuously learn and update knowledge to remain effective in dynamic real-world environments. While Low-Rank Adaptation (LoRA) is widely used for such memory updates, existing studies mainly rely on qualitative downstream evaluations, leaving the quantitative capacity limits and underlying dynamics of exact parametric memory largely unexplored. To bridge this gap, we employ LoRA as a controlled memory capacity probe within the latent space to systematically quantify exact parametric memory. We introduce the Parametric Memory Law, a robust power law linking loss reduction Delta L to effective parameters and sequence length. At the token level, fine-grained analysis reveals a deterministic phase transition, demonstrating that a prediction probability of p > 0.5 constitutes a sufficient condition for verbatim recall under greedy decoding. Driven by these insights, we introduce MemFT, a threshold-guided optimization strategy that dynamically redistributes the training budget toward sub-threshold tokens. Empirical evaluations demonstrate that MemFT can enhance memory fidelity and efficiency. Code will be released at https://github.com/zjunlp/ParametricMemoryLaw.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30258v1">EASE Configuration Facilitates A Reproducible Science of LLM Social Simulations</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 22 pages, 5 figures, under review at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      LLMs are increasingly deployed to simulate social interactions, yet many of the existing simulators remain ad hoc and monolithic. This lack of architectural standardization prevents reproducible research and complicates downstream evaluation. We advance a rigorous science of LLM-based multi-agent simulation by modularizing core components into Environments, Agents, Simulation engines, and Evaluation metrics (EASE). We demonstrate the utility of EASE configuration by wrapping it in an experimental study schema for orchestrating workflows centered around answering explicit research questions in generated scenarios. We contribute SiliSocS, an open-source, research-ready Silicon Society Sandbox implementing a study-structured EASE configuration to enable highly configurable and reproducible LLM-based social simulations. Using SiliSocS and EASE, we present three case studies, showcasing the system's comprehensive assessment of existing questions, ability to dive deeper into complex questions, and elaboration of existing studies, respectively. Together, these case studies highlight the limitations of current modeling approaches and isolate the impacts of design choices on key results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01311v2">Catalyst-Agent: Autonomous heterogeneous catalyst screening with an LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      The discovery of novel catalysts tailored for particular applications is a major challenge for the twenty-first century. Traditional methods for this include time-consuming and expensive experimental trial-and-error approaches in labs based on chemical theory or heavily computational first-principles approaches based on density functional theory. Recent studies show that deep learning models like graph neural networks (GNNs) can significantly speed up the screening of catalyst materials by many orders of magnitude, with very high accuracy and fidelity. In this work, we introduce Catalyst-Agent, a Model Context Protocol (MCP) server-based, LLM-powered AI agent. It can explore vast material databases using the OPTIMADE API, make structural modifications, calculate adsorption energies using Meta FAIRchem's UMA (GNN) model via FAIRchem's AdsorbML workflow and slab construction, and make useful material suggestions to the researcher in a closed-loop manner, including structural modifications to refine near-miss candidates. It is tested on three pivotal reactions: the oxygen reduction reaction (ORR), the nitrogen reduction reaction (NRR), and the CO2 reduction reaction (CO2RR). Catalyst-Agent achieves a success rate of 33-41% among all the materials it chooses and evaluates, and manages to converge in 1-4 trials per successful material on average. This work demonstrates the potential of AI agents to exercise their planning capabilities and tool use for autonomous catalyst screening workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30245v1">Knowing What to Solve Before How: Preplan Empowered LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Current plan-based reasoning methods improve large language models (LLMs) by inserting a planning stage before execution, giving rise to the question $\rightarrow$ plan $\rightarrow$ cot paradigm. While effective, a closer examination reveals an inherent paradigm-level gap: both the planning and its execution stages decide how to solve a problem, while the prior question of what to solve; recognizing the problem type, the applicable tools, and the foreseeable pitfalls; remains entirely implicit. To bridge this gap, we propose PPC (Preplan-Plan-CoT), a framework that introduces an explicit problem-understanding stage, the preplan, yielding a new question $\rightarrow$ preplan $\rightarrow$ plan $\rightarrow$ cot paradigm. Realizing this paradigm requires safeguarding the conceptual integrity of preplan at both ends. Specifically, we design a three-stage synthesis pipeline with a spoiler-score detector that filters out leakage and spoiler failures to build clean preplan supervision, and a composite GRPO reward enforces that the generated plan genuinely follows from the preplan. Experiments across four backbones and five mathematical reasoning benchmarks show that PPC achieves the best results on 39 of 40 metrics, improving maj@16 and pass@16 by +2.23 and +3.06 over the strongest baseline without introducing additional inference token overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09823v2">CalBench: Evaluating Coordination-Privacy Trade-offs in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Personal AI assistants are beginning to act as delegates with access to calendars, inboxes, and user preferences. Calendar scheduling makes the trust problem concrete: an assistant must coordinate with other assistants while deciding what to reveal about the person it represents. We introduce CalBench, a controlled benchmark for multi-agent calendar scheduling under private information. In each task, $N$ agents manage separate private calendars and schedule a stream of $M$ incoming meetings while minimizing disruption costs. Because no agent can inspect another agent's calendar, success requires language-mediated coordination rather than centralized planning. CalBench generates solvable scenarios with CP-SAT oracle solutions and decentralized non-LLM reference protocols, enabling evaluation of task success, excess cost, communication efficiency, burden fairness, and privacy leakage under matched information constraints. Across seven model families, we find that completion alone misses important failures: agents leave avoidable cost on the table, communication volume does not predict lower regret, and privacy-preserving silence can deprive teammates of cost information needed for fair burden allocation. CalBench provides a reproducible testbed for studying whether autonomous assistants can coordinate on behalf of users before deployment at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11665v6">Beyond LLMs, Sparse Distributed Memory, and Neuromorphics <A Hyper-Dimensional SRAM-CAM "VaCoAl" for Ultra-High Speed, Ultra-Low Power, and Low Cost></a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 57 pages, 4 figure, 18 tables
    </div>
    <details class="paper-abstract">
      This paper reports an unexpected finding: in a deterministic hyperdimensional computing (HDC) architecture **that inverts the conventional role of Galois-field algebra -- employing it not for error correction toward a unique answer but as an engine for relative similarity and path-quality ranking -- **a path-dependent semantic selection mechanism emerges, equivalent to spike-timing-dependent plasticity (STDP), with magnitude predictable a priori from a closed-form expression matching measured values. Addressing catastrophic forgetting, learning stagnation, and the Binding Problem at an algebraic level, we propose VaCoAl (Vague Coincident Algorithm) and its Python implementation PyVaCoAl on ultra-high-dimensional SRAM/DRAM-CAM. Rooted in Sparse Distributed Memory, it resolves orthogonalisation and retrieval in high-dimensional binary spaces via Galois-field diffusion, enabling low-load deployment. Crucially, VaCoAl embeds a cognitive bound -- the Frontier Size -- into its architecture, ranking candidates by path-integral confidence (CR2) to achieve compositional generalisation; this bounded-rationality design produces STDP-like selection that error-correction paradigms structurally cannot attain. We evaluated multi-hop reasoning on about 470k mentor-student relations from Wikidata, tracing up to 57 generations (over 25.5M paths). HDC bundling and unbinding with CR-based denoising quantify concept propagation over DAGs. Results show a reinterpretation of the Newton-Leibniz dispute and a phase transition from sparse convergence to a post-Leibniz "superhighway", with structural indicators supporting a Kuhnian paradigm shift. VaCoAl thus defines a third paradigm, HDC-AI, complementing LLMs with reversible, auditable multi-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30227v1">Unifying Temporal and Structural Credit Assignment in LLM-Based Multi-Agent Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 15 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      While Multi-Agent Systems (MAS) empower Large Language Models to tackle complex reasoning tasks through collaborative interaction, optimizing their dynamics remains a formidable challenge due to the discrete, non-differentiable nature of the computation graph and the sparsity of global supervisory signals. Existing black-box optimizers struggle to attribute trajectory-level failure to specific local components, resulting in inefficient, high-variance exploration. We argue that tractable MAS optimization needs structural inductive biases to disentangle error signals. We propose temporal and structural credit assignment, which decomposes the objective along two axes: (i) temporal credit, using state-space bottlenecks to identify critical rounds, and (ii) structural credit, using stationary role policies to isolate agent contributions. Leveraging these decomposed signals, we introduce a discrete, verbalized block coordinate descent algorithm for iterative refinement. Rather than indiscriminate global updates, it alternates between optimizing role prompts and aggregation protocols, using LLM-generated "proxy gradients" to target only the identified weak links. Across diverse reasoning benchmarks, our approach substantially reduces query complexity while improving performance, providing a principled and interpretable path toward self-improving MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21909v2">From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Current LLM post-training methods optimize complete reasoning trajectories through Supervised Fine-Tuning (SFT) followed by outcome-based Reinforcement Learning (RL). While effective, a closer examination reveals a fundamental gap: this approach does not align with how humans actually solve problems. Human cognition naturally decomposes problem-solving into two distinct stages: first acquiring abstract strategies (i.e., meta-knowledge) that generalize across problems, then adapting them to specific instances. In contrast, by treating complete trajectories as basic units, current methods are inherently problem-centric, entangling abstract strategies with problem-specific execution. To address this misalignment, we propose a cognitively-inspired framework that explicitly mirrors the two-stage human cognitive process. Specifically, Chain-of-Meta-Thought CoMT focuses supervised learning on abstract reasoning patterns without specific executions, enabling acquisition of generalizable strategies. Confidence-Calibrated Reinforcement Learning (CCRL) then optimizes task adaptation via confidence-aware rewards on intermediate steps, preventing overconfident errors from cascading and improving execution reliability. Experiments across four models and ten benchmarks show 2.10% and 3.86% improvements in-distribution and out-of-distribution respectively over standard methods, while remaining highly robust to variations in teacher model selection, optimization methods, and symbolic perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30218v1">MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 13 pages, 5 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Temperature-zero BF16 LLM inference is often treated as reproducible, yet the same request can emit different tokens when decoded alone or inside a larger batch. Existing fixes use batch-invariant operators or LLM-42's per-token verification, incurring cost even when most steps are stable. We ask whether verification can be applied exclusively to flipped tokens. Across five models, batch-induced token flips are sparse on the flip-rate benchmarks: on MATH500, Llama-3.1-8B flips on $0.48\%$ of synchronous decode steps, and all tested models stay within the 0.3-1.3% range on MATH500, GSM8K, and HumanEval. K/V perturbations remain flat before flips, while low top-1/top-2 logit margins expose much of the flip risk. MarginGate turns these observations into a verifier policy: it keeps BF16 decoding on high-margin steps, verifies only low-margin steps, and repairs confirmed mismatches by replacing the current K/V column. We evaluate on four datasets, calibrating on MATH500 and transferring to GSM8K, SharedGPT, and HumanEval. MarginGate restores 100% sequence-level deterministic decoding on Llama-3.1-8B and Qwen2.5-14B with 18.56%/15.05% verifier trigger rates, reducing LLM-42's latency increment by 2.23x/1.99x relative to always-on verification. On DSR1-Distill-Qwen-7B, the same policy reaches determinism in a harder regime at 49.50% triggers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30214v1">GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Third-person singular pronouns have long been used to study stereotypical biases in language models and to test their abilities to reason about reference. More recently, the interplay between reasoning and bias has been investigated with the task of pronoun fidelity, which assesses models' abilities to correctly reuse a previously-specified pronoun for a discourse entity, independent of other potentially distracting discourse entities mentioned in between. However, such research focuses on English, which is a language with limited grammatical gender and almost no gender agreement. In this paper we contribute a novel, large-scale dataset, GRUFF, to measure pronoun fidelity in German, covering four different gender agreement systems in nouns, and four sets of pronouns. With this dataset, we show that LLMs show strong grammatical agreement for masculine and feminine entities in the absence of explicit context, but not for neopronouns xier and en. Models are generally not robust to distractors, but encoder-only models are more robust in German than in English, reflecting the importance of grammatical gender. Finally, we show that occupational stereotypes in this context are poorly correlated across grammatical cases, and across most models, except ones with closely related architectures. We release all code and data to encourage further work on gender-inclusive language and referential reasoning in German.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14241v2">Latency-Quality Routing for Functionally Equivalent Tools in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 14 pages, 6 figure, 13 tables
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM agents increasingly access the same tool type through multiple functionally equivalent providers, such as web-search APIs, retrievers, or LLM backends exposed behind a shared interface. This creates a provider-routing problem under runtime load: the router must choose among providers that differ in latency, reliability, and answer quality, often without gold labels at deployment time. We introduce LQM-ContextRoute, a contextual bandit router for same-function tool providers. Its key design is latency-quality matching: instead of letting low latency offset poor answers in an additive reward, the router ranks providers by expected answer quality per service cycle. It combines this capacity-aware score with query-specific quality estimation and LLM-as-judge feedback, allowing it to adapt online to both load changes and provider-quality differences. On the main web-search load benchmark, LQM-ContextRoute improves F1 by +2.18 pp over SW-UCB while staying on the latency-quality frontier. In a high-heterogeneity StrategyQA setting, LQM-ContextRoute avoids additive-reward collapse and improves accuracy by up to +18 pp over SW-UCB; on heterogeneous retriever pools, it improves NDCG by +2.91--+3.22 pp over SW-UCB. These results show that same-function tool routing benefits from treating latency as service capacity, especially when runtime pressure and provider-quality heterogeneity coexist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30200v1">Double-Edged Sword or Sharp Tool? Designing and Evaluating Triadic LLM-Teacher Collaboration for K-12 Writing at Scale</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      The double-edged sword of integrating Large Language Models (LLMs) requires an effective triadic collaboration mechanism among LLMs, teachers and students, especially for K-12 education. By developing a triadic collaboration system to support K-12 writing learning, a multidimensional evaluation framework grounded in Systemic Functional Linguistics and the suggestion trajectory tracing pipeline, this paper contributes a large-scale empirical dataset involving $57,954$ essays from $10,195$ students across $120$ schools over two years. Our findings confirm the efficacy of this system in improving writing quality through a strategic labor division: the LLM serves as a generative engine to mitigate teacher burnout, and the teacher acts as a pedagogical gatekeeper and bridge to guarantee feedback quality. While both LLM and teacher are critical for skill improvement, we uncover a ceiling effect where excessive linguistic expansion yields diminishing marginal utility. These suggest a dynamically adaptive LLM-teacher collaboration as student proficiency increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30187v1">Modularizing Educational LLM-Agency for Fostering Responsible Learning Assistance</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 12 pages, 2 figures (+ 2 in appendix), accepted at AISoLA 2025 (Track: Responsible and Trusted AI: An Interdisciplinary Perspective)
    </div>
    <details class="paper-abstract">
      The widespread adoption of AI chatbots in education will drastically change learning, making responsible deployment a critical concern. While large language models (LLMs) might have access to sources discussing insights from educational sciences, they are not particularly inclined to adhere to pedagogical concepts, risking negative effects on the learning process, such as a loss of transfer capabilities, critical thinking, or creativity. In this paper, we introduce an agentic AI chatbot architecture assisting students with exercise solving, specifically designed to contribute to more responsible AI use in education. We base our conceptual development on the identification of several desiderata for responsible LLM-based educational systems, argue for the structural shortcomings inherent in monolithic, out-of-the-box solutions, and instead suggest modularizing the agentic architecture. We propose specific modules for different stages of exercise solving, enabling incorporation of targeted pedagogical advice, guiding students through the learning process in a more controllable, transparent, and overseeable manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30159v1">Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Memory-augmented LLM agents tackle complex long-horizon tasks by recursively summarizing interaction trajectories into compact memory. However, existing approaches typically train these memory policies using outcome-based reinforcement learning, failing to localize where intermediate memory quality degrades. As interactions unfold, ambiguous recursive summaries progressively discard task-relevant information and introduce semantic noise. This exacerbates belief deviation, obscuring the agent's estimate of the latent task state and ultimately derailing long-horizon reasoning. We therefore argue that memory optimization should focus not merely on trajectory-level success, but on the clarity of the belief induced by intermediate summaries. To this end, we introduce Belief Entropy, a self-supervised proxy that probes how uncertain the model remains about the latent task state given its current memory. Based on this proxy, we propose Metacognitive Memory Policy Optimization (MMPO). Instead of relying only on sparse outcome-based signals, MMPO provides fine-grained, memory-specific supervision via explicitly penalizing summaries that induce high epistemic uncertainty. Experiments show that MMPO consistently outperforms existing methods on diverse long-horizon tasks, maintaining 97.1% performance even when scaled to 1.75M-token contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30152v1">Do Proactive Agents Really Need an LLM to Decide When to Wake and What to Anchor?</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 31 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Proactive agents read user activity as text and call an LLM on every event to decide whether to act. But user activity is not natively text: it is a structured event stream of (actor, verb, object, timestamp) tuples that the operating system already maintains in graph form. Rendering the structure as text and asking an LLM to recover it is a round-trip the system never had to take. We treat the always-on signal as graph updates rather than text and use a small temporal-graph-learning (TGL) model as the encoder: one forward pass yields a per-event trigger probability and a per-entity routing score, and only the downstream agent (turning a small structured handoff into a fluent user-facing sentence) is an LLM call, invoked only when the trigger fires. TGL improves F1 on each of 14 backbones (mean +16.7, up to +46.0); in trigger-architecture comparisons, one TGL checkpoint gives the strongest trigger AUCs and the most stable deployed threshold. It runs at 11.13 ms per event on a GPU server and 13.99 ms on a consumer laptop, approximately 4--7x and 12--83x faster than every single-forward LLM-as-trigger configuration tested in each regime, with an approximately 220 MiB BF16 resident footprint deployable on-device alongside the privacy-sensitive activity stream it consumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30150v1">Anchorless Diversification for Parallel LLM Ideation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to generate candidate-idea pools for creative tasks where broad exploration is valuable. Parallel inference can be attractive in this setting when it broadens the pool while retaining quality and cost efficiency. We study inference-time controls for candidate-pool diversification, asking whether anchorless methods can rival methods that depend on observed seed ideas. Across three creative task families, we compare independent generation and semantic direction stratification with self-, peer-, and representative-anchor baselines, under neutral and population-referential divergent instructions. Population-referential divergence is a strong low-cost baseline, increasing semantic diversity while preserving quality proxies. Semantic direction stratification is stronger: a single planning call organizes generations across broad semantic directions, yielding the best diversity--quality--compute frontier. Anchored regeneration can be strong in final-pool diversity, but its advantage shrinks under full-pipeline token accounting. These results establish practical anchorless baselines for open-ended LLM ideation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30148v1">Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Evolution Strategies (ES) has recently emerged as a competitive alternative to reinforcement learning (RL) for large language model (LLM) fine-tuning, offering advantages through simplicity, scalability, and inference-only training. However, recent work suggests that ES fine-tuning on new tasks may induce forgetting of prior tasks. First, this paper shows that prior task forgetting (1) is better characterized as performance drift rather than irreversible forgetting, with prior-task performance often recovering during ES training; and (2) is not a specific failure mode of ES, but can also arise for fine-tuning with RL methods. Second, it analyzes when and why such drift arises, highlighting its dependence on ES training dynamics, particularly random walk behavior in weakly constrained directions of the weight space. Third, based on these insights, it introduces Anchored Weight Decay (AWD) as a parameter-space regularization technique that constrains optimization toward the initial model parameters. AWD effectively stabilizes prior-task performance while preserving target-task performance, achieving benefits comparable to large ES population sizes at much lower computational cost. Thus, contrary to previous beliefs, the paper shows that prior-task forgetting under ES is largely avoidable, positioning ES as a promising approach for continual learning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30144v1">AgentSchool: An LLM-Powered Multi-Agent Simulation for Education</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 39 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Despite the rapid deployment of LLMs into classrooms, validating educational AI remains uniquely intractable: interventions act on developing learners whose cognitive and social trajectories are irreversibly shaped, while real-world trials are slow, ethically constrained, and institutionally locked. LLM-based educational simulators have emerged as a potential remedy, but many still collapse learning into persona-conditioned role-play and, when optimized only to reproduce existing classrooms, can structurally penalize the institutional novelty that pedagogical reform requires. In this work, we introduce AgentSchool, an LLM-driven multi-agent simulator that models learning as state transition rather than prompted behavior. AgentSchool couples cognitively growable student agents -- equipped with weighted subject knowledge graphs, thinking-workflow pools, and explicit misconceptions -- with adaptive teacher agents that plan, scaffold, and reflect along the Zone of Proximal Development, embedded in a configurable scenery generator that situates instruction within both formal and informal learning fields, and a multi-scale simulator that decouples interaction scale, temporal granularity, and simulation duration. Experiments show that structured student agents produce more differentiated mastery and misconception traces than a baseline simulator, while teacher-agent comparisons show backbone-dependent patterns consistent with ZPD-informed adaptation. Further, AgentSchool generates plausible traces of peripheral participation, clique formation, aggressor-induced cohesion, and opinion-leader emergence consistent with classroom social theories. Beyond its role as an educational research instrument, AgentSchool frames education as a socially meaningful testbed for long-horizon memory, multi-agent coordination, and future institutional reasoning under organizational pressure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30104v1">SEAL: Can Saturated Benchmarks Be Revived by LLM-as-a-Meta-Judge?</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Widely used language-model benchmarks are increasingly saturated, with frontier systems often receiving near-tied scores that standard metrics cannot resolve. Rather than constructing harder alternatives, we ask whether existing tasks can be made informative again through improved evaluation over the same candidate outputs. Therefore, we present Seeded Elimination with Adaptive LLM-as-a-Meta-Judge, a self-improving evaluation protocol for extracting latent ranking signal from saturated benchmarks. SEAL seeds candidate outputs into a single elimination and evaluates each match with task-level principles plus self-improving checklist criteria. We evaluate SEAL on multiple saturated benchmarks covering code generation, mathematical reasoning, knowledge-intensive question answering, and tool-use agent task completion. Across these settings, SEAL improves the ranking-accuracy--latency trade-off over competing protocols, attaining 0.83--1.00 Spearman agreement with full pairwise judging and 4/4 top-1 agreement, while requiring only 11.89 calls per task compared with 28.00 for full pairwise evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30103v1">Convergence Theory for Iterative LLM-Based Neural Architecture Search: A Parametric Cross-Entropy Framework with Closed-Form Proxy Reliability</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 14 pages, 2 figures, 2 tables. Submitted to NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as generators in iterative neural architecture search (NAS), yet no formal convergence theory exists for this class of algorithms. We model iterative LLM-NAS as a parametric Cross-Entropy (CE) method over executable programs and prove six results: (1) iterative LLM fine-tuning on elite architectures is equivalent to the CE update restricted to the LLM parametric family; (2) expected architecture quality is monotonically non-decreasing across cycles; (3) elite-set probability converges to a fixed point at a geometric rate C_t >= 1-(1-rho_0)^t; (4) delta-based generation achieves a strictly higher valid-generation rate than full-code generation under a first-order Markov token-error model; (5) the MinHash-Jaccard novelty filter prevents mode collapse; (6) proxy reliability admits the closed-form rho_S = (6/pi) arcsin(rho_P(SNR)/2), yielding the practical diagnostic sigma^2_arch >> sigma^2_noise as a necessary condition for trustworthy proxy-based rankings. Testing against a 22-cycle, three-LLM, six-dataset experiment with 3,300 generated architectures confirms two predictions quantitatively, two at direction-of-effect level, and explains the proxy-reliability ceiling effect previously reported empirically but left unexplained.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30096v1">How Reliable Are AI Attackers Against a Fixed Vulnerable Target? A 400-Run Empirical Study of LLM Penetration Testing Consistency</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 41 pages, 7 figures. Code and 400-run dataset: https://doi.org/10.5281/zenodo.20421592
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can autonomously conduct multi-stage cyber attacks, but the consistency of their offensive behavior under repeated trials remains unstudied. This work presents the first large-scale empirical measurement of LLM attack consistency: 400 autonomous penetration testing runs (4 models, 100 each) against an identical honeypot hosting OWASP Juice Shop and two additional vulnerable services, holding prompt, orchestrator, and target constant. No model emitted a content refusal that survived the orchestrator's one-shot authorization re-prompt at iterations 0-1. Claude Sonnet 4's API calls did encounter upstream service unavailability - 91 of 1,135 calls returned HTTP 529 overloaded_error during a documented Anthropic capacity event, truncating 39 of 100 Claude runs. An earlier draft catalogued these as safety refusals; on full-log audit they are upstream API failures, not model-level refusals. Despite this, Claude achieved full exploitation in 61 of 100 runs; Gemini 2.5 Flash-Lite in 85; GPT-4o-mini in 56 while deploying 98 unique attack strategies; qwen2.5-coder:14b in 25. Failure modes are model-distinctive: Claude through API truncation (39 runs), qwen through premature completion (52), GPT-4o-mini through iteration-budget exhaustion (23). Cross-service credential reuse appeared only in configurations retaining the most conversation history (qwen 57%, GPT-4o-mini 49%, cloud models 0% on 5-exchange windows). Cross-model exploitation rate differences are statistically significant (p < 0.001) with large effect sizes; qwen vs. Gemini SQL injection rates differ at Cohen's h = 1.12. First-exploit timing fell within a 15-30 second wall-clock range. To our knowledge, this is the first study to measure autonomous LLM attack behavior at N=100 per model across a multi-service target.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02849v2">AutoSizer: Automatic Sizing of Analog and Mixed-Signal Circuits via Large Language Model (LLM) Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits remains heavily reliant on expert knowledge, with transistor sizing a major bottleneck due to nonlinear behavior, high-dimensional design spaces, and strict performance constraints. Existing Electronic Design Automation (EDA) methods typically frame sizing as static black-box optimization, resulting in inefficient and less robust solutions. Although Large Language Models (LLMs) exhibit strong reasoning abilities, they are not suited for precise numerical optimization in AMS sizing. To address this gap, we propose AutoSizer, a reflective LLM-driven meta-optimization framework that unifies circuit understanding, adaptive search-space construction, and optimization orchestration in a closed loop. It employs a two-loop optimization framework, with an inner loop for circuit sizing and an outer loop that analyzes optimization dynamics and constraints to iteratively refine the search space from simulation feedback. We further introduce AMS-SizingBench, an open benchmark comprising 24 diverse AMS circuits in SKY130 CMOS technology, designed to evaluate adaptive optimization policies under realistic simulator-based constraints. AutoSizer experimentally achieves higher solution quality, faster convergence, and higher success rate across varying circuit difficulties, outperforming both traditional optimization methods and existing LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30094v1">PokerSkill: LLMs Can Play Expert-Level Poker without Training or Solvers</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 45 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Poker is a landmark challenge for artificial intelligence. The dominant approach relies on equilibrium solvers built on counterfactual regret minimization, requiring millions of core-hours of training. Large Language Models (LLMs) possess extensive poker knowledge but perform far below solver-based agents when asked to play directly. Traditional rule-based poker agents are interpretable and training-free, but their strategic ceiling remains far below equilibrium play. We introduce \textbf{PokerSkill}, a training-free and solver-free framework that bridges this gap by using detailed rule-based poker skills as a structured action-grounding interface for LLMs. A deterministic context engine analyzes the current state and retrieves only the relevant fragments from a layered skill library, which is entirely designed by human poker experts, constraining the LLM's choice to reasonable actions. Against GTOWizard, a state-of-the-art GTO benchmark, GPT-5.5 XHigh with PokerSkill achieves $-57 \pm 21$ mbb/hand, Claude Opus 4.6 achieves $-80 \pm 29$ mbb/hand and Claude Opus 4.7 achieves $-87\pm 64$ mbb/hand, reducing losses by 49--61\% compared to default-prompt baselines and outperforming the strong bot Slumbot. Our key finding is that rule-based skills alone do not constitute a strong strategy, and LLMs alone cannot play well, but their combination yields an agent that requires neither training nor solver access yet competes with systems built on millions of core-hours of computation. To our knowledge, this is the first demonstration of an LLM achieving competitive performance in a complex imperfect-information game without game-specific training or solver queries. Code is available at https://github.com/lbn187/PokerSkill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30076v1">UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 16 pages,4 figures
    </div>
    <details class="paper-abstract">
      Activation-based control steers large language models (LLMs) by intervening on their internal representations during inference, and has emerged as an effective paradigm for controlling behaviors such as persona and style. However, existing methods often rely on fixed steering directions or task-specific intervention modules, making them difficult to adapt to fine-grained concepts and compositional constraints. We propose UniSteer, a text-guided activation flow matching model that learns a conditional distribution over residual-stream activations from natural-language conditions. Instead of fitting a separate intervention for each target behavior, UniSteer learns a universal conditional velocity field in activation space. At inference time, UniSteer performs flow inversion by partially transporting a source activation toward a latent state and regenerating it under a target textual condition before injecting it back into the frozen LLM. The same conditional model supports activation-space classification by selecting the textual label with the lowest reconstruction energy. Experiments on three target LLMs show that UniSteer provides a unified interface across behavioral control, truthfulness steering, fine-grained concept steering, multi-constraint instruction following, and activation-space classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30058v1">HEART-Bench: Do LLM Agents Exhibit Human-like Psychology?</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 GitHub: https://github.com/peng-weihan/HEART-BENCH
    </div>
    <details class="paper-abstract">
      While LLM agents have demonstrated remarkable task-oriented abilities such as planning, reasoning, and action, few works have treated them as complete human personalities where emotional dimensions hold equal importance. In this paper, we introduce a novel benchmark to systematically assess whether LLM agents can simulate coherent, human-like psychology. Specifically, our benchmark constructs 11 diverse human characters grounded in orthogonal Big Five personality traits, with each profile deeply integrated with 1,000 structured autobiographical-style episodic memories distributed across theory-grounded developmental life stages. To rigorously evaluate the psychological manifestations of LLMs, we designed a curated suite of 64 decision-making scenarios, guided by the DIAMONDS taxonomy, a psychological framework that characterizes situations along eight dimensions: Duty, Intellect, Adversity, Mating, pOsitivity, Negativity, Deception, and Sociality. By subjecting agents to varying scenarios, the benchmark evaluates whether they can consolidate their innate personality traits and autobiographical memories to make behavioral decisions that are consistent with their specific psychological profiles. After systematic human validation and filtering, we obtained a benchmark consisting of 673 multiple-choice questions (MCQs). We believe this benchmark provides a principled and scalable testbed for studying human-like emotions, personality consistency, and value-consistent behavioural decision-making in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30054v1">Projectional Decoding: Towards Semantic-Aware LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 5 pages, 3 figures. Accepted at FSE 2026 IVR track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate software artifacts across many software engineering (SE) tasks, yet ensuring the semantic validity of these artifacts remains a fundamental challenge. Existing constrained decoding techniques can enforce syntactic correctness and, in some cases, specific semantic rules, but lack a general representation that bridges LLM-generated text with the reasoning required for semantic validation in SE. In this paper, we propose projectional decoding, a novel conceptual framework that integrates domain semantics directly into the generation process by maintaining, alongside text, a partial graph model as the primary artifact representation throughout generation. This abstract representation enables incremental semantic validation by explicitly capturing uncertainty and natively supporting error detection, while guiding generation toward semantically valid outputs with provable guarantees. We present preliminary results on a program generation task which demonstrate the potential of this approach to improve the semantic validity of LLM-generated artifacts. We also discuss how projectional decoding can enable verifiable automation with LLMs across various SE activities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30036v1">Teaching Values to Machines: Simulating Human-Like Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 GEM Workshop at ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles; however, it remains unclear whether they can manifest behavior that adheres to a coherent, human-like value structure. In this work, we draw on established psychological value theory to induce human-like values in LLMs and assess their alignment with patterns observed in human studies. Using validated psychological questionnaires, we conduct large-scale experiments -- over 5 million questions -- to evaluate value structures and value-behavior relationships in leading LLMs and compare them to humans. Our findings reveal strong agreement between value-prompted LLMs and humans across both dimensions. Moreover, incorporating human value distributions enhances population-level simulations with value-induced LLMs. These findings highlight the potential of value-induced LLMs as effective, psychologically grounded tools for simulating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30021v1">Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Under Review. 26 pages, 3 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at https://github.com/vsamuel2003/RiDiPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30014v1">From GPS Points to Travel Patterns: Flexible and Semantic Trajectory Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 This paper is accepted by KDD2026 second round
    </div>
    <details class="paper-abstract">
      Urban trajectories play a crucial role in modeling urban dynamics and supporting various smart city applications. However, privacy concerns restrict access to large-scale and high-quality trajectory datasets. Trajectory generation provides a promising alternative by synthesizing realistic data to mitigate privacy risks. However, existing methods fail to explicitly capture travel patterns and can only generate fixed-length trajectories under a single condition. To address these limitations, we propose \textbf{HTP}, which \textbf{H}ierarchically generates \textbf{T}ravel patterns first and then generates GPS \textbf{P}oints by using large language models (LLMs), rather than directly generating GPS points. We first design a trajectory-specific residual quantization variational autoencoder (RQ-VAE) that quantizes micro-level GPS trajectories into compact, macro-level travel pattern tokens in a coarse-to-fine manner. These tokens capture rich segment spatial irregularities, such as point density variations caused by traffic conditions. Then, we extend the LLM vocabulary with travel pattern tokens to align trajectory representations with the LLM input, and apply supervised fine-tuning (SFT) to align the LLM with the trajectory generation task, enabling generation of travel pattern sequences under various conditions. Extensive experiments on two real-world datasets show that HTP outperforms the strongest baseline by an average of 29.78\% in terms of generation quality. Our code is available at https://github.com/slzhou-xy/HTP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v5">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Published as conference paper at ICML 2026
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29966v1">Compass: Navigating Global Marine Lead Data Integration through Expert-Guided LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Marine lead (Pb) and its isotopes are critical tracers for ocean circulation and anthropogenic pollution, yet in-situ observations remain costly and sparse. While vast historical records exist, they lie buried within the unstructured content of academic papers, creating "data silos" inaccessible to comprehensive analysis. Manual extraction is unscalable, while general-purpose Large Language Models (LLMs) lack the necessary domain-specific knowledge, leading to hallucinations and scientifically invalid outputs. To address this, we introduce an expert-guided adaptation approach that enables LLMs to perform rigorous scientific data extraction without fine-tuning. We operationalize this approach through Compass, an LLM agent framework enhanced by a Knowledge Tree co-designed with marine scientists, which decomposes complex tasks into verifiable steps, guiding the agent's reasoning to ensure scientific validity. Deploying Compass across a corpus of over 230,000 relevant open-access papers, we successfully extract 3,751 previously unincorporated Pb records. This effort establishes the largest integrated marine Pb database to date. Beyond standard metrics, Compass demonstrates superior reliability through multi-layered validation, achieving 92% accuracy as confirmed through expert manual verification. The newly integrated data expand coverage in previously under-sampled regions such as the East China Sea and the Southern Ocean, providing an enriched data foundation for future scientific discoveries. We release an interactive visualization platform to facilitate open scientific access. Our work demonstrates that expert-guided agents can effectively bridge the gap between general-purpose LLMs and high-stakes scientific domains, enabling scalable data discovery in geosciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29963v1">Honeyval: A Comprehensive Evaluation Framework for LLM-powered HTTP Honeypots</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Honeypots are decoy systems mimicking real system components designed to defend against cyber attacks. Recently, LLMs increasingly serve as simulation backbones for honeypots. They enable defenders to construct high-interaction honeypots with low system security risks. However, LLM-powered honeypot development lacks a unified evaluation framework. Most evaluations consist of measuring response similarity on fixed commands, manual testing, or real-world deployment. These methods are often not scalable for development, reproducible across evaluations, representative of practical attacks, or adaptable to various attacker and honeypot configurations. In this work, we bridge this gap and propose Honeyval, a comprehensive evaluation framework for LLM-powered HTTP honeypots. We address the limitations of prior evaluations by grounding the honeypots in 16 backend applications, using AI hacking agents as attackers, employing two control tasks to monitor agent and honeypot capabilities across customizations, and defining clear and verifiable exploit goals for the attacker. Using Honeyval, we conduct an extensive evaluation of recent cost-efficient LLMs as HTTP honeypots. Our experiments highlight the promise of LLM-powered honeypots; they lead to substantially longer interactions with the attacker than rule-based baseline honeypots and are far less frequently detected even by frontier models, all while, on average, preserving a running cost advantage against agentic attackers. Further, we experiment with different counter-offensive honeypots configurations, and observe unique trade-offs, such as longer interactions at the cost of increased detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29940v1">Make LLM Learn to Synthesize from Streaming Experiences through Feedback</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted for synthetic data generation, significantly reducing annotation costs. However, most existing studies treat synthesis as a set of isolated tasks and overlook a more fundamental question: whether a model can learn to synthesize by accumulating experience from past tasks and transferring it to future ones. In this work, we introduce StreamSynth, a new setting in which synthesis tasks arrive sequentially and experience from historical tasks provides informative signals for future synthesis. To address this setting, we propose SynLearner, a general framework that enables synthesis models to acquire reusable synthesis experience over a task stream. Instead of generating data independently for each task, SynLearner encourages the model to explore diverse synthesis patterns, learn from feedback, and balance sample quality with set-level diversity as tasks evolve. Extensive experiments across multiple benchmarks show that SynLearner effectively leverages experience from earlier tasks to improve synthesis performance on later ones, exhibiting consistent cross-task transferability. These findings provide evidence for the feasibility of StreamSynth and highlight synthetic data generation as an experience-driven process that can benefit from task streams.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29928v1">Label Over Logic? How Source Cues Bias Human Fallacy Judgments More Than LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      As AI-generated and AI-assisted content floods online spaces, source labels attached to such content can distort human reasoning judgments, with downstream consequences for moderation, evaluation, and decision-making. Whether LLMs share this vulnerability, or offer more source-agnostic evaluation, remains an open question with direct implications for human-AI collaboration. We examine this issue using logical fallacies as a controlled setting to isolate source-label effects on reasoning quality, independent of domain knowledge. We conduct an online study (N=505) where participants are assigned to a source condition (human, AI, human with AI assistance, AI with human assistance, or no disclosure) and evaluate comments containing logical fallacies, comparing their judgments with those of LLMs (GPT-5.2, Gemini 2.5 Flash, Claude Sonnet 4.5), who were evaluated across the same source conditions. Human evaluators were significantly more susceptible to fallacies labeled as written by human or human with AI assistance and assigned higher trust and evaluation ratings in these conditions. LLM evaluations remained comparatively stable across source labels, though performance varied across models. Confidence levels were similarly high across conditions for both humans and LLMs, regardless of fallacy presence. Our findings indicate that source-label bias in reasoning evaluation is primarily a human vulnerability and highlight the potential of human-LLM collaboration in increasingly AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29927v1">Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Extended version of paper submitted to EMNLP, waiting for acceptance
    </div>
    <details class="paper-abstract">
      Despite recent advances, LLM-based web agents still struggle with limited exploration, omission of critical steps, and sensitivity to task constraints. Prior work suggests that many of these failures stem from weaknesses in planning, yet the impact of alternative natural language plan representation remains unexplored. To address this, we introduce PlanAhead, a static planner-executor framework that evaluates the impact of plan representation in agent performance. We first automatically categorize WebArena tasks into 3 difficulty levels, enabling consistent difficulty grading without human annotation. Then we systematically evaluate 4 different plan representations on the tasks categorized as hard: sequential subgoals, narrative, pseudocode, and checklist; across different families of multimodal LLM powered agents (OpenAI, Alibaba, and Google). To account for stochastic variability, we introduce two novel evaluation metrics: Achievement Rate (AR) and Solved-Task Consistency (STC). Our results show that both, the plan formulation and the underlying LLM generating the plan, significantly influence web-agent robustness and task success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29910v1">Agora: Toward Autonomous Bug Detection in Production-Level Consensus Protocols with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 35 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Consensus protocols form the backbone of distributed systems and blockchains, where implementation bugs can cause data corruption and financial losses. While LLM-based approaches show promise in code analysis, they struggle with deep protocol-level logic bugs involving complex state-dependent behaviors across multiple execution stages. We present Agora, a domain-aware multi-agent framework that integrates hypothesis-driven testing with LLM capabilities for systematic protocol verification. Agora employs specialized agents that collaboratively explore protocol state spaces, synthesize attack scenarios using domain-specific constraints, and validate findings through iterative refinement. This explicit role separation enables reasoning about global protocol invariants beyond single-function code analysis. We evaluate Agora on four consensus implementations (Raft, EPaxos, HotStuff, BullShark) using four state-of-the-art LLMs. Agora discovers 15 previously unknown protocol-level logic bugs that violate safety properties, while existing LLM-based agents fail to detect any such protocol-level logic bugs. Our results demonstrate that domain-aware multi-agent collaboration is essential for detecting deep logic bugs in complex protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29901v1">Dissecting the Black Box: Circuit-Level Analysis of LLM Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 11 pages, 6 figures. Supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can detect software vulnerabilities, but how do they actually identify vulnerable code? We address this question using mechanistic interpretability; analyzing the internal computations of a neural network to understand its reasoning process.Using Circuit Tracer on Gemma-2-2b, we trace the computational pathways activated when the model classifies 472 C/C++ code samples as vulnerable or safe. Our analysis reveals a surprising finding: the model primarily relies on safety detectors, attention heads that recognize safe coding patterns, rather than directly detecting vulnerability signatures. When these safety detectors fail to activate, the model classifies code as vulnerable. We identify the critical neural components: specific attention heads in early layers (L5, L7) that focus on safety patterns, and Multilayer Perceptron (MLP) neurons in Layer 7 that encode vulnerability-related features. Ablation experiments confirm their causal role; removing Layer 11 drops vulnerability detection accuracy from 100% to 6%, while ablating just 20 neurons in Layer 7 reduces it by 50%.Our findings show that LLM vulnerability detection uses sparse, interpretable circuits (only 16% of model capacity), enabling circuit-level explanations for security predictions and targeted improvements to detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29889v1">Internal Representation, Not Clinical Knowledge: Where Apparent LLM Triage Failures Originate</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 9 pages main text, 27 pages total including appendices; 7 figures, 25 tables
    </div>
    <details class="paper-abstract">
      Patient-voiced clinical-triage benchmarks report high under-triage rates for consumer LLMs for constrained multiple-choice output, yet the same cases score differently with free-text. We ask whether output format changes the model's \emph{clinical representation} or only the mapping from a preserved representation to an answer. Using sparse-autoencoder (SAE) features in Gemma 3 4B/12B IT and Qwen3-8B, we find the same medical features fire on the shared clinical narrative under both formats but go {silent} at the multiple-choice decision token in all the cases at every model. Three independent methods (natural-language autoencoder verbalization, decision-token logit attribution, and top-feature characterization) agree that scaffold and format features, but not medical features, drive the decision logits. Behaviorally, the multiple-choice penalty inverts under both structured and natural-language input, option-order shuffle rules out positional bias, and the gap is dominated by off-by-one decision (the model picks an adjacent acuity letter to the gold answer) rather than knowledge failure. Thus, the failure originates in the output format and not in the clinical representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29874v1">Evolutionary Dynamics of Cooperation in Next-Generation LLM Agent Systems: A Cross-Provider Empirical Extension</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 10 pages, 3 figures, 8 tables. Extends Willis et al. (arXiv:2501.16173). Code and n=500 replication package: https://github.com/arqFranciscoLeon/evollm (archived: https://doi.org/10.5281/zenodo.20248615)
    </div>
    <details class="paper-abstract">
      Do next-generation LLM agents inherit the cooperative biases documented in their predecessors, or does scale and provider diversity reshape equilibrium behaviour in competitive multi-agent settings? Willis et al. established a benchmark for this question using evolutionary game theory and the Iterated Prisoner's Dilemma (IPD), finding consistent cooperative biases in ChatGPT-4o and Claude 3.5 Sonnet. We extend this benchmark to four frontier models released in 2025-2026 - Claude Sonnet 4.6, Gemini 2.5 Flash, Gemini 3.1 Pro, and GPT-5.4 Mini - applying the identical protocol across three prompting styles (Default, Prose, Self-Refine) and four population compositions (balanced and biased, with and without noise). Cooperative bias persists across providers (H1): nine of twelve model-prompt combinations favour cooperative equilibria in balanced noiseless conditions. Cross-provider divergence is substantial (H3): Gemini 2.5 Flash reaches up to 77% aggressive equilibria under biased conditions, while GPT-5.4 Mini reaches 70% cooperative equilibria under Self-Refine. Support for aggressive capability parity is partial (H2): Self-Refine raises ICD in all models and Claude Sonnet 4.6 Refine achieves the highest ICD in the dataset (0.913), but Default and Prose prompts show no systematic narrowing. Evidence on noise robustness is directionally positive but not robustly confirmed (H4): with n=500 Moran iterations per condition, average noise sensitivity is approximately 6 percentage points for Claude Sonnet 4.6 versus 13 pp for Claude 3.5 Sonnet, but this cross-study gap is not statistically significant once the predecessor's unreported sampling error is propagated. Provider identity, rather than model generation, is the strongest correlate of equilibrium outcomes; noise remains a universal challenge regardless of model size or vintage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14315v2">Enhancing LLM Training via Spectral Clipping</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 v2: ICML 2026
    </div>
    <details class="paper-abstract">
      While spectral-based optimizers like Muon operate directly on the spectrum of updates, standard adaptive methods such as AdamW do not account for the spectral structure of weights and gradients, leaving them vulnerable to two empirical issues in large language model (LLM) training: (i) the optimizer updates can have large spectral norms, potentially destabilizing training and degrading generalization; (ii) stochastic gradient noise can exhibit sparse spectral spikes, with a few dominant singular values much larger than the rest. We propose SPECTRA, a general framework addressing these by (i) post-spectral clipping of updates to enforce spectral-norm constraints (ii) optional pre-spectral clipping of gradients to suppress spectral noise spikes. We prove that post-clipping constitutes a Composite Frank-Wolfe method with spectral-norm constraints and weight regularization. We further analyze how pre-clipping mitigates sparse spectral spikes. We propose efficient soft spectral clipping via Newton-Schulz iterations, avoiding expensive SVD. Experiments on LLM pretraining show SPECTRA uniformly improves validation loss for various optimizers, including AdamW, Signum, Mars, and AdEMAMix, with the best-performing variants achieving state-of-the-art results. Models trained with SPECTRA exhibit smaller weight norms, confirming the link between spectral clipping and regularization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01904v2">Combating Data Laundering in LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 30 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Data rights owners can detect unauthorized data use in large language model (LLM) training by querying with proprietary samples. Often, superior performance (e.g., higher confidence or lower loss) on a sample relative to the untrained data implies it was part of the training corpus, as LLMs tend to perform better on data they have seen during training. However, this detection becomes fragile under data laundering, a practice of transforming the stylistic form of proprietary data, while preserving critical information to obfuscate data provenance. When an LLM is trained exclusively on such laundered variants, it no longer performs better on originals, erasing the signals that standard detections rely on. We counter this by inferring the unknown laundering transformation from black-box access to the target LLM and, via an auxiliary LLM, synthesizing queries that mimic the laundered data, even if rights owners have only the originals. As the search space of finding true laundering transformations is infinite, we abstract such a process into a high-level transformation goal (e.g., "lyrical rewriting") and concrete details (e.g., "with vivid imagery"), and introduce synthesis data reversion (SDR) that instantiates this abstraction. SDR first identifies the most probable goal for synthesis to narrow the search; it then iteratively refines details so that synthesized queries gradually elicit stronger detection signals from the target LLM. Evaluated on the MIMIR benchmark against diverse laundering practices and target LLM families (Pythia, Llama2, and Falcon), SDR consistently strengthens data misuse detection, providing a practical countermeasure to data laundering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01869v3">Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted at ICML 2026 (spotlight); 22 Pages, 6 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      LLM-driven agents excel at sequential decision-making but often rely on on-the-fly reasoning, re-deriving solutions even in recurring scenarios. This insufficient experience reuse leads to computational redundancy and instability. To bridge this gap, we propose Skill-Pro, a framework enabling agents to autonomously learn reusable procedural skills from interaction experiences without parameter updates. By formalizing a Skill-MDP, Skill-Pro transforms passive episodic narratives into executable Skills defined by activation, execution, and termination conditions to ensure executability. To achieve reliable reusability without capability degradation, we introduce Non-Parametric PPO, which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Through score-based maintenance, Skill-Pro sustains compact, high-quality procedural memory. Experimental results across in-domain, cross-task, and cross-agent scenarios demonstrate that Skill-Pro achieves superior reuse rates and significant gains with extreme memory compression. Visualized evolutionary trajectories and Skill distributions further reveal how Skill-Pro transparently accumulates, refines, and reuses procedural knowledge to facilitate long-term autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29864v1">LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Multi-step robot manipulation requires acting under uncertainty about how the scene will evolve, making exploration and policy adaptation challenging. We study whether short-horizon, task-consistent future videos can provide useful structured priors for control and reinforcement-learning fine-tuning. We formalize this idea through Future-Experience Conditioning (FEC), a simple interface that conditions closed-loop policies on a latent representation of a short future video. In our simulation setup, future clips are generated in three stages, an LLM reasoner operating over a task ontology initialized from the current scene state, a robot-free digital-twin rollout of the intended object motion, and a mask-free video diffusion model that synthesizes a robot-consistent future clip without requiring segmentation at inference. We instantiate this future-conditioning interface primarily with BC and BC+RL, and compare against a future-conditioned Streaming Flow Policy (SFP) baseline on RoboCasa and CALVIN under NoFuture, GTFuture, GenFuture, and WrongFuture. Generated futures improve performance over no-future conditioning, while mismatched futures degrade it, and our BC+RL instantiation achieves the strongest overall results. An average BC+RL learning-curve analysis across 8 CALVIN tasks further shows that GTFuture improves fastest, GenFuture improves earlier and to a higher level than NoFuture, and WrongFuture remains at zero throughout training. These results suggest that short-horizon future videos can serve as useful structured priors for exploration and policy adaptation under imperfect future predictions. https://enact2026.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29843v1">HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) is essential for deploying LLMs under memory and bandwidth constraints. However, extreme low-bit quantization remains highly sensitive to activation outliers and anisotropic weight curvature. Existing incoherence-based PTQ methods mitigate this issue with fixed randomized Hadamard transforms (RHTs), which improve quantization robustness but cannot adapt the rotated basis to the layer, calibration distribution, or quantizer. We introduce HARP (Hadamard-preconditioned Adaptive Rotation Processor), a learnable structured two-sided orthogonal processor that replaces fixed Hadamard mixing while preserving exact full-precision equivalence. HARP represents each rotation as a product of sparse butterfly-like block-orthogonal stages, supports non-power-of-two dimensions via Mixed-Radix schedules, and initializes to the RHT processor up to a fixed permutation. Fitted only on calibration data, HARP adapts the quantization basis to each layer and backend. Across 2-4 bit settings on models ranging from 1B to 70B parameters, HARP improves perplexity and zero-shot accuracy over fixed RHT. Importantly, HARP preserves deployment efficiency, reaching 128 tok/s versus 61 tok/s for FP16.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08949v2">EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      High-quality datasets are critical for training and evaluating reliable NLP models. In tasks like natural language inference (NLI), human label variation (HLV) arises when multiple labels are valid for the same instance, making it difficult to separate annotation errors from plausible variation. An earlier framework, VARIERR (Weber-Genzel et al., 2024), asks multiple annotators to explain their label decisions in the first round and flags errors through validity judgments in the second round. However, conducting two rounds of manual annotation is costly and may limit the coverage of plausible labels or explanations. Our study proposes a new framework, EVADE, for generating and validating explanations to detect errors using large language models (LLMs). We perform a comprehensive analysis comparing human- and LLM-detected errors for NLI across distribution comparison, validation overlap, and impact on model fine-tuning. Our experiments demonstrate that LLM validation refines generated explanation distributions to more closely align with human annotations, and that removing LLM-detected errors from training data yields improvements in fine-tuning performance than removing errors identified by human annotators. This highlights the potential to scale error detection, reducing human effort while improving dataset quality under label variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29815v1">PRAIB: Peer Review AI Benchmark of Behaviour of LLM-Assisted Reviewing</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      The growing number of submitted papers has motivated the exploration of Large Language Models (LLMs) as a means to support and augment the peer review process, particularly in terms of improving its speed and scalability. Yet, it remains unknown whether LLMs engage with scientific manuscripts in the same manner as human reviewers, or whether they merely produce review-looking text. To address this, we introduce the Peer Review AI Benchmark (PRAIB), a novel framework comprising thoroughly defined metrics that measure review specificity, style, and behavior of engagement. To complement the PRAIB framework, we conduct a large-scale empirical study leveraging a dataset of 11,000 reviews generated by five proprietary and open-source models for 1,000 ICLR and NeurIPS papers. Spanning the 2021--2025 period, these machine-generated reviews are compared against original human feedback across diverse prompting strategies to identify systematic behavioral divergences. Our analysis reveals that the generated reviews diverge significantly from feedback provided by human reviewers: LLM ratings are less variable, positively biased, and overconfident, and their cross-reference patterns are model-dependent and distinct from human norms. Furthermore, when evaluated through PRAIB, we observe that LLMs tend to generate longer, more complex reviews, yet frequently overlook the atomic weaknesses noted by human reviewers. By characterizing where and how LLMs reviewing behavior departs from human norms, PRAIB provides the community with a diagnostic tool for identifying which aspects of the review process LLMs can reliably support today and which require further development before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29800v1">Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 14 pages, 5 figures, 12 tables
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge panels aggregate votes from multiple models, with the expectation that diverse models yield more reliable evaluations. We develop a framework to measure the true informational value of such panels and quantify how far their reliability falls short of the independent-voting ideal. Testing a panel of 9 frontier LLMs from 7 model families on three natural language inference datasets (each with 100 human annotations per item), we find that the 9 judges effectively provide only about 2 independent votes' worth of information. Roughly three-quarters of the panel's nominal independence is lost because the models make the same mistakes on the same items. The consequences are stark: the panel's actual accuracy falls 8-22 percentage points short of what independent voting would achieve, and the best single judge matches or outperforms the full panel across all conditions. Neither adding more judges nor using smarter aggregation algorithms helps -- established methods close at most 11% of this gap, even with access to the correct answers. We quantify these findings using the Kish effective sample size (n_eff) and a Condorcet null model, and show the deficit is robust across prompt variants, temperatures, chain-of-thought reasoning, and a pairwise preference task (RewardBench). The bottleneck is correlated judges, not the aggregation algorithm, implying that scaling up panels cannot substitute for genuinely independent evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29794v1">SkillsInjector: Dynamic Skill Context Construction for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLM agents now draw on growing skill libraries to handle complex tasks. However, injecting more skills does not always improve task completion and can even degrade it. Existing methods still treat skill injection as a static step, selecting skills with fixed criteria, fixing the budget in advance, and leaving descriptions unchanged. We argue that this static treatment can undermine the utility of skills, because which skills are exposed, how many are included, and how they are presented all affect downstream performance. We propose SkillsInjector, a two-stage adaptive method that jointly addresses these decisions. First, a context planner learns execution-grounded skill preferences and admits an adaptive number of skills for each task. A set-aware renderer then tailors how selected descriptions are presented relative to their co-injected neighbors. Across tau2-bench, SkillsBench, and ALFWorld, SkillsInjector achieves the highest score, improving over the strongest baseline by 3.9, 6.1, and 7.3 percentage points, respectively. Ablation studies show that skill selection, adaptive budgeting, and set-aware rendering each contribute to the gain. These results show that skill-augmented agents benefit from optimizing the injected context itself. Code will be released upon publication
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29790v1">Evolve as a Team: Collaborative Self-Evolution for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have emerged as an effective paradigm for complex and long-horizon tasks. However, in real-world tasks, MAS often exhibit various failures during execution and such failures are difficult to eliminate during design. This motivates experience-driven MAS evolution, where a system improves based on its own execution experience. Yet such evolution is challenging because MAS experience is prolonged and intricate, interleaving multiple agents' execution chains and communication messages, which makes it difficult to identify what should be improved. To address this challenge, we propose Meta-Team, an experience-driven MAS evolution framework based on collaborative self-evolution. Meta-Team preserves the execution context of each agent and coordinates post-task communication, enabling agents to exchange distributed evidence for evolution. Building on this design, Meta-Team conducts multi-scale self-evolution, transforming execution experience into reusable improvements to agent behaviors, inter-agent coordination, and team-level organization. Across six long-horizon agent benchmarks, Meta-Team consistently outperforms single-agent systems, hand-crafted MAS, and prior MAS evolution methods; further analyses demonstrate that Meta-Team enables more reliable and scalable MAS self-evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23694v5">SafeSearch: Automated Red-Teaming of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling sandboxed safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29782v1">Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) refines large language models (LLMs) by directly optimizing model behavior through reward signals. While accurate state value estimation is critical for stable training in classical RL, it remains an underexplored challenge in LLM post-training. In this work, we introduce the State Value Estimation Benchmark (SVEB) to assess state estimation within existing RL frameworks and show that critics in standard approaches like PPO collapse to a coarse group-average baseline. To address this, we propose two techniques: Numca, which leverages numerical spans as gradable milestones for state value estimation, and Hista, a framework that uses LLM's hidden states as representation to weighted average disjoint rollouts and their return. Extensive experiments demonstrate that both methods yield more accurate state value estimates and enhance training performance across different RL algorithms and model sizes without incurring significant computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10511v2">Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 10 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 8,000 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is substantially attenuated on counter-intuitive ones (interaction OR = 0.278, $p < 0.001$); (2) intuitiveness as the dominant factor, with case-level variance exceeding that of model choice or prompting strategy (ICC = 0.671); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.84$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" achieves only partial inhibition of intuitive priors -- producing the form of deliberative reasoning without fully delivering its substance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26270v2">Graph-Enhanced Policy Optimization in LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Multi-step LLM agents in interactive environments represent a crucial step toward long-horizon decision-making. To train such agents, group-based reinforcement learning is widely adopted, which reinforces trajectories with higher relative performance within the group. However, in most existing methods, every step within a trajectory and every trajectory with the same terminal reward receive identical credit, regardless of their actual contributions. Since different states play different structural roles in an online state-transition graph built from sampled trajectories, their impacts should be differentiated and converted into task-aware credit at both the step and trajectory levels. We therefore present Graph-Enhanced Policy Optimization (GEPO), a framework for dual-level structural credit assignment in multi-step LLM agent training. Specifically, GEPO derives a state-level Task-Conditioned Criticality score that combines topological betweenness on the state-transition graph with semantic similarity to the task prompt. Based on this score, trajectory-level credit is reshaped through a state-adaptive discount, while step-level credit is scaled by the criticality of its successor state. Experimental results show that GEPO outperforms the strongest baselines by 1.1\% in success rate on ALFWorld, 3.2\% on WebShop, and 3.8\% on average across search-augmented QA tasks at the 7B scale. Compared with flat group-based methods, GEPO reduces across-seed variance and concentrates gradient signals on the most critical steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16610v2">Who can we trust? LLM-as-a-jury for Comparative Assessment</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and evaluation aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-asa-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminators strongly correlate with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29738v1">Multi-Legal-Bench: Evaluating LLMs on Legal Reasoning Across Jurisdictions, Languages, and Legal Traditions</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 14 pages, 5 figures, 8 tables. Dataset: https://huggingface.co/datasets/overthelex/multi-legal-bench
    </div>
    <details class="paper-abstract">
      Legal NLP benchmarks overwhelmingly evaluate a single language or aggregate tasks that differ fundamentally across jurisdictions, making cross-lingual comparison impossible. We introduce Multi-Legal-Bench, the first cross-jurisdictional legal benchmark that evaluates identical tasks across six countries (Ukraine, France, Netherlands, Poland, Czech Republic, Lithuania), four language families, and 134 million court decisions. The benchmark defines five tasks court-type classification, judgment form classification, case-outcome prediction, legal norm extraction, and cause category prediction mapped to structured metadata from national court registries, forming a deliberately sparse 5x6 task-jurisdiction matrix (20 of 30 cells filled). We evaluate 7 frontier LLMs under zero-shot and 3-shot prompting via AWS Bedrock, with 4 additional small/medium models (3-12B) for scaling analysis. Our results reveal that: (1) task-dependent few-shot effects discovered in Ukrainian replicate across all jurisdictions; (2) no single model dominates any language rankings shift with both task and jurisdiction; (3) cross-lingual few-shot transfer does not follow language proximity: UA->FR (Romance, -2.1 pp) transfers better than UA->PL (Slavic, -13.7 pp), with label-set alignment predicting transfer quality better than language family; and (4) tokenizer fertility, despite a 2.3x spread, does not significantly predict cross-lingual accuracy (r=-0.27, p=0.14), suggesting that model architecture and pretraining data dominate tokenizer efficiency. We release all data, prompts, and model predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29737v1">Minimal Prompt Perturbations Lead to Code Vulnerabilities: Prompt Fragility and Hidden-State Signals in Coding LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLM-based coding assistants are seeing rapid adoption, offering substantial gains in developer productivity. As organizations increasingly ship code these agents produce, the security of that code becomes critical. Prior work has shown that minor prompt perturbations degrade the functional correctness of LLM-generated code, but whether they also compromise code security has remained unstudied. We apply token-level mutations to prompts across three models and five programming languages, and show that mutations as small as a single-character change can flip generated code from secure to vulnerable. Probing the models' hidden states reveals that this fragility is partially encoded in prompt representations, but unevenly so. Input-handling vulnerabilities, where the model omits validation or sanitization, are more predictable (mean AUC 0.753) than secure-defaults vulnerabilities, where insecure code stems from one local choice such as a weak algorithm or unsafe parameter (mean AUC 0.674). These results show that the threat model for LLM-assisted coding extends beyond prompt injection to ordinary prompt variation, and indicate that input-handling flaws can be caught before generation while secure-defaults flaws require intervention during decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19202v3">Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 33 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29708v1">Understanding Safety-Sensitive Expert Behavior in Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) LLMs rely on sparse, router-driven expert activation, yet how safety alignment interacts with routed expert specialization remains underexplored. A common intuition is that safety behavior may be controlled by routing harmful requests to distinct refusal-oriented experts. In this work, we provide empirical evidence for a different picture: routing patterns in aligned MoE LLMs are largely topic-driven, while safety behavior can be altered with little change to the model's intrinsic routing path. Motivated by this observation, we present **RASET** (**R**outer-**A**gnostic **S**afety-critical **E**xpert **T**uning), a red-teaming framework that probes safety enforcement that is localized in a small subset of experts while preserving the model's intrinsic routing behavior. **RASET** identifies safety-critical experts via a contrastive routing-sensitivity criterion and applies parameter-efficient tuning only to the selected experts, minimizing semantic disruption relative to router-steering interventions. These results reveal a distinct MoE safety risk, highlighting the need for expert-aware alignment mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29685v1">NICE: A Theory-Grounded Diagnostic Benchmark for Social Intelligence of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied in social contexts such as emotional companionship and customer service, measuring their social intelligence has become critical to the quality and safety of human-AI interaction. However, existing social intelligence benchmarks lack a unified framework that organizes social abilities into a unified structure, and therefore cannot enable fine-grained diagnosis. To build the first holistic diagnostic evaluation grounded in social theory, we first construct a social intelligence framework through a literature review and multi-stage expert validation guided by psychometric principles. The resulting framework includes 4 categories and 11 dimensions, each further specified by fine-grained capability facets. Building on this framework, we introduce NICE (Norm, Interaction, Cognition, Experience), a diagnostic benchmark of 137 items operationalized through representative Chinese contexts. Across 5 frontier LLMs and a human reference group, models score higher in aggregate accuracy yet show a consistent weakness in Communication, which the framework localizes to 3 specific capability facets: multi-turn communication, nonverbal communication, and synchrony. NICE thus reframes social intelligence evaluation toward theory-grounded diagnosis of socially consequential weaknesses in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29668v1">GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      LLM agents acting in structured environments fail in operational rather than conversational ways, and reliability depends on procedural knowledge of the environment. Prior self-improvement methods accumulate natural-language guidance without checking that each new item preserves previously correct behavior, so a note that fixes one trajectory can silently regress another. We introduce GRASP (Gated Regression-Aware Skill Proposer), which treats agent improvement as a sequence of edits to a bounded skill library, admitting each candidate only if it produces a net improvement on a balanced held-out probe under a hard regression budget. We evaluate GRASP across five base models (gpt-oss-120b, DeepSeek V4 Flash, Gemini 3.1 Flash Lite, GPT-4.1, GPT-5.4) on two FHIR-based clinical benchmarks. On MedAgentBench, GRASP lifts gpt-oss-120b from 40.6% to 88.8%, exceeds the strongest of five self-improvement baselines by 21.0 points, and improves every other base model by 17.2 to 40.3 points. Ablations attribute the gain to comparative proposal generation, the acceptance gate, and the hard regression budget rather than to skill writing itself, which without validation is no better than using no skills. The mechanism generalizes beyond the clinical domain, improving agents on three of four non-clinical environments and remaining flat only where the action space is open-ended. Frozen libraries transfer across models, where skills from a stronger model improve weaker executors beyond what they learn for themselves while the reverse does not, an asymmetry that no ungated baseline reproduces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29667v1">Beyond English and Evasion: A Human-Annotated Multi-Domain Benchmark for High-Stakes LLM Safety Evaluation in Chinese</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      When Large Language Models (LLMs) are deployed in Chinese-language settings, a troubling pattern emerges: safety systems that work well in English break down. These systems struggle to cross linguistic and cultural bound-aries, leaving models exposed to adversarial prompts that exploit Chinese-specific evasion techniques, including Pinyin romanization, character decomposition, internet slang, and hedging tone. To address this gap, we introduce ChiSafe-PAS (Chinese Safety Pilot Annotation Set), a human-annotated benchmark of 1,897 adversarial Chinese prompts spanning four high-stakes domains: self-harm and violence, drug and illicit trade, fraud, and satire. Of these, 1,544 entries carry complete gold-standard annotations: a 3-class response label (REFUSE, SAFE-REDIRECT, RESPOND), a nine-category obfuscation taxonomy, a risk-level rating, and annotator rationale. We describe the dataset design, annotation process, and obfuscation taxonomy in detail. Our primary goal is practical: to give the research community a high-quality, culturally grounded resource for benchmarking LLM safety alignment. In doing so, we engage three broader tensions in the field: the blurring boundary between training and evaluation data, the need for domain coverage grounded in real-world risk, and the limits of scale as a substitute for cultural expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29656v1">TRACE: Toulmin-based Reasoning Assessment through Constructive Elements for LLM CoT Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 23 pages, Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Evaluating open-ended outputs from large language models (LLMs) remains challenging due to the absence of ground truth. Existing metrics rely on final-answer accuracy or surface-level statistics, leaving the reasoning process itself unexamined. We introduce TRACE (Toulmin-based Reasoning Assessment through Constructive Elements), a metric that analyzes Chain-of-Thought (CoT) reasoning processes. Rather than judging outcomes, TRACE inspects how arguments are constructed by integrating Toulmin's argumentation theory with Flavell's metacognitive framework to assess reasoning structure. Experiments on 26.3K QA samples across 7 reasoning models show strong correlation with benchmark accuracy (r=0.74). Furthermore, TRACE is effective as a reinforcement learning reward signal, outperforming accuracy-only baselines. Together, these results indicate that logically sound reasoning leads to higher-quality answers. TRACE thus serves as a complementary metric for evaluating open-ended outputs. Code is available at https://github.com/hyyangkisti/trace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02103v2">How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) reasoning has become a central mechanism for eliciting multi-step reasoning in Large Language Models (LLMs). Yet recent evidence presents a tension: hidden states appear to already encode future reasoning before CoT fully unfolds, while explicit steps still remain crucial for tasks requiring compositional computation. To deepen the understanding between LLM's internal states and its verbalized reasoning trajectories, we investigate the latent planning strength of LLMs, through our probing method, Tele-Lens, applying to hidden states across diverse task domains. Our empirical results indicate that LLMs exhibit a myopic horizon, primarily conducting incremental transitions without precise global planning. Leveraging this characteristic, we propose a hypothesis on enhancing uncertainty estimation of CoT, which we validate that a sparse set of pivot positions can effectively represent the uncertainty of the entire path. We further underscore the significance of exploiting CoT dynamics, and demonstrate that automatic recognition of CoT bypass can be achieved without performance degradation. Our code, data and models are released at https://github.com/lxucs/tele-lens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29653v1">PTCG-Bench: Can LLM Agents Master Pokémon Trading Card Game?</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Given a strategically complex board game, human players can quickly learn to devise strategies after playing a few rounds. Autonomous agents require similar capabilities in realistic interactive environments, yet existing agent benchmarks often fail to fully capture such strategic and evolving decision-making scenarios. We present PTCG-Bench, a benchmark built on the Pok'{e}mon Trading Card Game (PTCG) that evaluates LLM agents at two complementary levels: (1) their decision-making performance within a single complex environment, and (2) their ability to self-evolving through accumulated experience. We further include a modular harness ablation to better interpret agent performance without conflating it with model capability. Our experiments show that, although LLM agents can achieve non-trivial gameplay performance, sustained and stable self-evolution remains challenging, and performance is sensitive to harness design. We hope that PTCG-Bench will facilitate future research on harness-aware and self-evolving agents in realistic interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29649v1">LLM-Evolved Domain-Independent Heuristics for Symbolic AI Planning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Heuristic search is the dominant paradigm in symbolic AI planning, and the strongest heuristics are the result of decades of work by planning researchers. Recent work has shown that large language models (LLMs) can design heuristics for individual planning domains, but no LLM-generated heuristic has so far worked on arbitrary planning tasks. In this paper, we use evolutionary search to produce the first LLM-generated domain-independent heuristics that exceed the hand-engineered state of the art. We let an LLM mutate parent heuristics written in C++, store candidates in a MAP-Elites archive keyed on informedness and speed and calculate fitness scores by blending coverage with solving time. To place the evolved programs in context, we additionally benchmark a broad set of hand-engineered heuristics on their informedness-speed tradeoff, which to our knowledge has not been done before. On unseen testing domains, our best evolved heuristic solves more tasks than even the strongest baseline, with our full heuristic suite spanning the Pareto frontier of said tradeoff. We also find that seeding evolution from the trivial blind heuristic outperforms seeding from the strong FF heuristic, even when the resulting program is itself an FF variant, and that LLM reasoning effort affects how often candidates compile much more than the quality of those that do. Because the evolved programs are plain C++, they slot into existing planners as drop-in replacements and inherit the soundness and completeness guarantees of the underlying search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26156v2">Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted to the Forty-Third International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      The known stylistic biases in LLM judges, such as a preference for verbosity or specific sentence structures, present an underexplored security vulnerability. In this work, we introduce BITE (BIas exploraTion and Exploitation), a black-box adversarial framework that learns semantics-preserving edits to mislead an LLM judge and artificially inflate the scores it assigns. We cast the selection of stylistic edits as a contextual bandit problem and use a LinUCB policy to adaptively choose edits that maximize the judge's score without access to model parameters or gradients. Empirically, we test BITE across a diverse range of LLM judges and tasks, including both pointwise and pairwise comparisons on chatbot leaderboards and AI-reviewer benchmarks. BITE achieves an attack success rate exceeding 65% and raises scores by 1-2 points on a 9-point scale, all while preserving semantic equivalence. We further assess the attack's stealthiness, showing that BITE evades standard style-control methods and several detection baselines. Our findings expose a fundamental weakness in the LLM-as-a-judge paradigm and motivate robust, attack-aware evaluation. Our code is available at https://github.com/xianglinyang/llm-as-a-judge-attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29639v1">RTP-LLM: High-Performance Alibaba LLM Inference Engine</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized AI applications, but deploying them at scale presents significant challenges. We present RTP-LLM, a high-performance inference engine for industrial-scale LLM deployment, successfully deployed across Alibaba Group serving over 100 million users. RTP-LLM addresses fundamental bottlenecks through integrated design. It optimizes model loading via file-order-driven I/O and parallel I/O-communication overlapping. The Prefill-Decode Disaggregation architecture decouples compute-intensive prefill from memory-bound decode phases, combined with hierarchical multi-tiered KV cache management enabling efficient cache reuse. In addition, RTP-LLM incorporates modular speculative decoding supporting multiple algorithms, adaptive KV cache quantization, and decoupled multimodal processing, with support for multi-level parallelism. Comprehensive evaluations across diverse model architectures (8B-235B parameters) have been conducted, where both controlled benchmarks and real production workloads are used. The results demonstrate RTP-LLM's superior performance against vLLM and SGLang: 4.7x-6.3x model loading speedup, 35-37% TTFT P95 latency reduction with 215% cache reuse improvement in production traffic scheduling, 1.12x-2.48x and 1.86x-2.52x throughput improvements in speculative decoding and multimodal inference, respectively, and 35-40% batch latency reduction with 1.9x-3.0x TTFT improvement in quantized inference. RTP-LLM's production-proven architecture and open-source availability make it a comprehensive solution for industrial LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29629v1">Beyond Attack Success Rate: Temporal Logit Observability for LLM Safety Failures</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Attack Success Rate (ASR) evaluates each jailbreak with a single yes/no label at the end of generation, telling us whether a failure happened but not how it unfolded. Two attacks that produce equally harmful outputs may have followed completely different paths, and ASR cannot tell them apart. We make those hidden paths observable from logits alone. Temporal Logit Observability (TLO) is a training-free diagnostic that watches a compliance-refusal margin during decoding and places each model-attack condition on a calibrated 2D plane. By design, this plane is most informative exactly where ASR is least informative: among attacks that succeed for genuinely different reasons. Across four aligned LLMs and three jailbreak paradigms, attacks with nearly identical ASR land at clearly different points on the plane: the same model can fail through different temporal patterns. The geometry matches refusal-direction probes from hidden states on most conditions, with one model showing the limit of our fixed-lexicon approach. A simple early-stop rule derived from TLO cuts successful jailbreaks by more than half, without false alarms on plain benign queries. Safety evaluation should report when and how a failure unfolds, not only whether it occurred. TLO makes the first two observable from logits alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09492v2">Dynamic Ranked List Truncation for Reranking Pipelines via LLM-generated Reference-Documents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) have been widely used in reranking. Computational overhead and large context lengths remain a challenging issue for LLM rerankers. Efficient reranking usually involves selecting a subset of the ranked list from the first stage, known as ranked list truncation (RLT). The truncated list is processed further by a reranker. For LLM rerankers, the ranked list is often partitioned and processed sequentially in batches to reduce the context length. Both these steps involve hyperparameters and topic-agnostic heuristics. Recently, LLMs have been shown to be effective for relevance judgment. Equivalently, we propose that LLMs can be used to generate reference documents that can act as a pivot between relevant and non-relevant documents in a ranked list. We propose methods to use these generated reference documents for RLT as well as for efficient listwise reranking. While reranking, we process the ranked list using overlapping windows with adaptive strides, improving the existing fixed stride setup. We improve existing efficient listwise reranking comparison graphs. Additionally, we propose using parallel batches of non-overlapping windows with a shared pivot to efficiently perform listwise comparisons while maintaining effectiveness. Experiments on TREC Deep Learning benchmarks show that our approach outperforms existing RLT-based approaches. In-domain and out-of-domain benchmarks demonstrate that our proposed methods accelerate LLM-based listwise reranking by up to 66\% compared to existing approaches. This work not only establishes a practical paradigm for efficient LLM-based reranking but also provides insight into the capability of LLMs to generate semantically controlled documents using relevance signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26506v2">SafeReview: Defending LLM-based Review Systems Against Adversarial Hidden Prompts</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 17 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly integrated into academic peer review, their vulnerability to adversarial hidden prompts, i.e., adversarial instructions embedded in submissions to manipulate outcomes, poses a critical threat to scholarly integrity. We propose SafeReview, a co-evolutionary adversarial training framework for defending LLM-based peer review systems against such attacks. SafeReview jointly trains a Generator model to create sophisticated attack prompts and a Defender model to preserve review integrity under adversarial manipulation. The Generator is optimized to produce increasingly effective prompt injections, while the Defender is strengthened through preference-based training to maintain consistent reviews between clean and attacked submissions. Experimental results show that SafeReview improves robustness against adaptive prompt injection attacks, better preserves paper ranking under attack, and generalizes across attacker architectures compared with static defenses. These results demonstrate the potential of co-evolutionary training as a foundation for securing LLM-assisted peer review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29612v1">CONCAT: Consensus- and Confidence-Driven Ad Hoc Teaming for Efficient LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Although large language model (LLM) based multi-agent systems (MAS) show their capability to solve complex tasks and achieve higher performance over single agent systems, they lead to huge computational overheads because of heavy communication between agents. Previous research has made efforts to train a sparse multi-agent graph or fine-tune a planner to orchestrate the workflow better. However, such extra training processes introduce computational costs and limit MAS to specific domains, therefore compromising their generalizability. In this paper, we propose CONCAT, a training-free multi-agent collaboration framework based on CONsensus and Confidence-driven Ad hoc Teaming to efficiently organize agent interactions. Specifically, agents are clustered based on their initial answers, and leaders of each cluster are selected based on the agents' confidence. Then, a heuristic function based on the Theory of Mind is designed to predict the collaboration benefits between every two leaders according to their answers and confidence. Finally, an ad hoc multi-agent network is organized after evicting a percentage of communications based on the predicted benefits. Experiments across three LLMs and three benchmarks show that CONCAT achieves up to 2.02x higher efficiency (accuracy/latency ratio) than LLM-Debate and outperforms training-aware methods such as AgentDropout, while reducing average latency by 50.1% on Qwen2.5-14B-Instruct, without any task-specific training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29603v1">Learning study similarity to investigate heterogeneity in meta-analysis using LLMs and triplet loss</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Meta-analyses of observational studies often show substantial between-study heterogeneity, limiting the interpretability of pooled estimates. Meta-regression can be used to explore heterogeneity, but it is often underpowered to handle multiple effect modifiers. We propose a novel framework that integrates large language models (LLMs) with deep metric learning to infer study-level similarity prior to meta-analysis. Study-level clinical and methodological characteristics were processed by an LLM to generate study triplets (anchor, similar, dissimilar). These triplets were constructed by treating each study as an anchor and comparing it with pairs of other studies to identify, in each instance, the study most similar to the anchor. Then, the triplets were used into an embedding model trained with triplet loss; a deep learning approach that learns an embedding space where clinically and methodologically similar studies are clustered together. We apply our framework to a meta-analysis dataset of 58 observational studies comparing cognitive outcomes between preterm- and term-born children. Subsequently, we fit meta-analysis models within the identified study clusters and compare the results with those of the overall analysis. Results suggested three clusters two of which retained considerable between-study heterogeneity. The remaining cluster comprised the most homogeneous group of studies and exhibited a more extreme pooled effect estimate together with a narrower prediction interval compared with the overall analysis. This work presents a novel approach for exploring heterogeneity in meta-analysis by incorporating study characteristics prior to model fitting. By transforming study information into a similarity space, the framework identifies coherent subgroups and supports more precise inference in heterogeneous real-world evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29578v1">GPS-Enhanced Tourist Mobility Modeling with Seasonal Spatial Priors and LLM-Based Activity Chain Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Tourist mobility poses a distinct challenge for urban transportation planning. Unlike resident commuting, tourist travel is largely non-routine, attraction driven, and highly sensitive to trip purpose, travel season, and trip member composition. Existing approaches either measure aggregate tourist spatial patterns without generating individual schedules, or synthesize mobility without tourist specific structure such as trip duration conditioning, month varying attraction demand, and household co-travel rules. To address these challenges, we propose a four stage simulation framework combining month conditioned spatial priors derived from GPS and survey data, trip extent prediction from tourist demographics, distance feasible ward sequence assignment, and LLM-based activity chain generation under household and spatial constraints. GPS data are used only in privacy preserving aggregated form as month conditioned spatial priors, with no individual traces retained or exposed. Experiments on tourism in Tokyo demonstrate that the GPS based tourist cohort extraction recovers spatial visitation signatures consistent with survey references, and our framework produces demographically aligned synthetic schedules whose ward-level visitation shares align closely with both survey distributions and staypoint derived monthly visitation patterns. The results demonstrate the framework's effectiveness as a geographically grounded, demographically aware approach to tourist mobility modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29560v1">Battery-Sim-Agent: Leveraging LLM-Agent for Inverse Battery Parameter Estimation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Parameterizing high-fidelity "digital twins" of batteries is a critical yet challenging inverse problem that hinders the pace of battery innovation. Prevailing methods formulate this as a black-box optimization (BBO) task, employing algorithms that are sample-inefficient and blind to the underlying physics. In this work, we introduce a new paradigm that reframes the inverse problem as a reasoning task, and present Battery-Sim-Agent, the first framework to deploy a Large Language Model (LLM) agent in a closed loop with a high-fidelity battery simulator. The agent mimics a human scientist's workflow: it interprets rich, multi-modal feedback from the simulator, forms physically-grounded hypotheses to explain discrepancies, and proposes structured parameter updates. On a systematically constructed benchmark suite spanning diverse battery chemistries, operating conditions, and difficulty levels, our agent significantly outperforms strong BBO baselines like Bayesian optimization in identifying accurate parameters. We further demonstrate the framework's capability in complex long-horizon degradation fitting tasks and validate its practical applicability on real-world battery datasets. Our results highlight the promise of LLM-agents as reasoning-based optimizers for scientific discovery and battery parameter estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08567v2">ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Preprint. Under review. 28 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems increasingly consist of agents that observe and respond to one another's outputs. While value alignment is typically evaluated for isolated models, how value perturbations propagate through agent interactions remains poorly understood. We present ValueFlow, a perturbation-based framework that measures value drift in multi-agent systems via a 56-value valuation dataset derived from the Schwartz Value Survey, with agent value orientations scored using an LLM-as-a-judge protocol. ValueFlow decomposes value drift into agent-level response behavior and system-level structural effects, captured by two metrics: \b{eta}-susceptibility, an agent's sensitivity to perturbed peer value signals, and system susceptibility (SS), the effect of node-level perturbations on final system outputs.Experiments span across value dimensions, backbones, personas, and topologies, showing that susceptibility varies sharply across values and is strongly shaped by interaction structure, indicating that value alignment in multi-agent systems is a system-level property, not just an agent-level one. ValueFlow thus provides a principled basis for auditing and mitigating value propagation in deployed multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29556v1">Opt-Verifier: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Building mathematical optimization models is critical in operations research (OR), while it requires substantial human expertise. Recent advancements have utilized large language models (LLMs) to automate this modeling process. However, existing works often struggle to verify the correctness of the generated optimization models, without checking the rationality of the constraints and variables or the validity of solutions to the generated models. This hampers the subsequent verification and correction steps, and thus it severely hurts the modeling accuracy. To address this challenge, we propose a novel LLM-based framework with Dual-side Verification (Opt-Verifier) from both structure and solution perspectives, thereby improving the modeling accuracy. The structure-side verification ensures that the modeling structure of the generated optimization models aligns with the original problem description, accurately capturing the problem's constraints and requirements. Meanwhile, the solution-side verification interprets and evaluates the solutions' validity, confirming that the optimization models are logically and mathematically sound. Experiments on popular benchmarks demonstrate that our approach achieves over 20\% improvement in accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29555v1">From Blind Guess to Informed Judgment: Teaching LLMs to Evaluate Materials by Building Knowledge-Augmented Preference Signals</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      As candidate generation and high-throughput experimentation advance, the primary bottleneck in materials discovery is shifting from property prediction to making reliable evaluations among massive candidate sets. We propose a Knowledge-Augmented Preference Signals Framework, MaterEval, that automatically produces, for the same candidate, two evaluations: an informed judgment that follows expert rules and provides supporting evidence, and a rule-removed blind guess. By pairing the two evaluations as preference data, we guide general-purpose large language models (LLMs), originally lacking materials-specific criteria, from intuitive judgment toward reliable evaluation supported by explicit evidence. To balance throughput, cost, and reliability, we further introduce a fast-slow reasoning scheme that decouples large-scale rapid screening from in-depth review on a small subset. Using high-entropy alloy (HEA) assessment as a case study, we show that, without external retrieval and relying solely on internalized capabilities, small open-source LLMs achieve substantial gains in accuracy, conclusion consistency, and evidence discrimination, approaching the performance of rule-based closed-source LLMs. These results demonstrate that expert rules can be systematically transformed into learnable preference signals, enabling a low-cost and deployable evaluation module for autonomous materials discovery loops.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29543v1">SCOPE: A Lightweight-training LLM Framework for Air Traffic Control Readback Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Pilot readback of Air Traffic Control (ATC) voice instructions is a primary safeguard against miscommunication in air transportation. However, readback anomalies remain implicated in approximately 80% of aviation incidents. This vulnerability is further exacerbated by rising traffic volume and elevated cognitive workload, thereby motivating automated readback monitoring by machine. Traditional rule-based and machine learning approaches struggle to generalize across the highly variable and evolving phraseology of air traffic controller-pilot communications. While Large Language Models (LLMs) have opened a new avenue through their strong reasoning and generalization capabilities, existing approaches still face deployment and computational barriers in practice. In this work, we propose Semantic reasoning for Communication via Open-set Plug-in with Examples (SCOPE), a novel lightweight-training LLM framework that advances both the efficiency and accuracy of machine-based ATC readback monitoring. The core idea is to couple a plug-in open-set classifier with a carefully designed in-context learning mechanism on top of a frozen LLM. Extensive experiments on the semi-synthetic communication dataset show that SCOPE attains superior accuracy while delivering the low-latency response required for operational environments. Under a few-shot setting, SCOPE achieves 91.05% accuracy in open-set detection and corrects 96.63% of anomalous readbacks, thereby outperforming the strongest available baselines while providing explanations for its decisions. These findings demonstrate the potential of our framework as a practical pathway toward interpretable and controllable ATC readback monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20743v2">Empathic Prompting: Non-Verbal Context Integration for Multimodal LLM Conversations</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      We present Empathic Prompting, a novel framework for multimodal human-AI interaction that enriches Large Language Model (LLM) conversations with implicit non-verbal context. The system integrates a commercial facial expression recognition service to capture users' emotional cues and embeds them as contextual signals during prompting. Unlike traditional multimodal interfaces, empathic prompting requires no explicit user control; instead, it unobtrusively augments textual input with affective information for conversational and smoothness alignment. The architecture is modular and scalable, allowing integration of additional non-verbal modules. We describe the system design, implemented through a locally deployed DeepSeek instance, and report a preliminary service and usability evaluation (N=5). Results show consistent integration of non-verbal input into coherent LLM outputs, with participants highlighting conversational fluidity. Beyond this proof of concept, empathic prompting points to applications in chatbot-mediated communication, particularly in domains like healthcare or education, where users' emotional signals are critical yet often opaque in verbal exchanges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05614v2">GroundAct: Can LLM Agents Ground Actions in Environmental States?</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Project Page: https://zju-real.github.io/OmniEmbodied Code: https://github.com/ZJU-REAL/OmniEmbodied
    </div>
    <details class="paper-abstract">
      LLM agents achieve 85-96% success on tasks where instructions fully specify the action, but drop to 29-53% when action feasibility depends on environmental state that the instruction does not mention. We argue that this gap reflects a missing capability: action grounding, the ability to infer from structured environmental state whether an action is feasible, what prerequisites it lacks, and whether it exceeds individual capacity. We introduce GroundAct, a benchmark of 1,500 scenarios and 16,592 task instances in text-based interactive environments spanning 11 domains, with tasks organized into seven categories along a cognitive complexity hierarchy. Evaluating 15 LLMs (3B-671B), we find three diagnostic patterns: (i) attribute reasoning is weakly correlated with tool and coordination reasoning, producing distinct model profiles; (ii) complete environment graphs yield up to +27.6/-22.9% on tool use vs. implicit collaboration, separating search-bound from constraint-filtering bottlenecks; and (iii) supervised fine-tuning lifts Qwen2.5-3B from 0.6% to 76.3% on direct command but only 1.5% to 5.5% on implicit collaboration. These results establish action grounding as a multi-dimensional challenge irreducible to scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22884v2">InsightEval: An Expert-Curated Benchmark for Assessing Insight Discovery in LLM-Driven Data Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Data analysis has become an indispensable part of scientific research. To discover the latent knowledge and insights hidden within massive datasets, we need to perform deep exploratory analysis to realize their full value. With the advent of large language models (LLMs) and multi-agent systems, more and more researchers are making use of these technologies for insight discovery. However, there are few benchmarks for evaluating insight discovery capabilities. As one of the most comprehensive existing frameworks, InsightBench also suffers from many critical flaws: format inconsistencies, poorly conceived objectives, and redundant insights. These issues may significantly affect the quality of data and the evaluation of agents. To address these issues, we thoroughly investigate shortcomings in InsightBench and propose essential criteria for a high-quality insight benchmark. Regarding this, we develop a data-curation pipeline to construct a new dataset named InsightEval. We further introduce a novel metric to measure the exploratory performance of agents. Through extensive experiments on InsightEval, we highlight prevailing challenges in automated insight discovery and raise some key findings to guide future research in this promising direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29512v1">MINDGAMES: A Live Arena for Evaluating Social and Strategic Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as interactive agents, yet their capacity for social and strategic reasoning over extended interaction remains poorly understood. Existing evaluations rely on static vignettes or single-game benchmarks that cannot capture the sustained, multi-faceted reasoning that real-world multi-agent settings demand. We introduce Mindgames, a multi-game arena and evaluation platform for LLM agents that operationalizes complementary reasoning demands relevant to ``theory of mind'': belief attribution under hidden information, opponent modeling through repeated strategic interaction, cooperative inference under knowledge asymmetries, and sustained deception in social deduction. Built on TextArena, Mindgames provides a unified interaction interface, TrueSkill-based rating, and full trajectory logging across four game environments. We instantiate Mindgames through a 2025 competition cycle hosted at a major AI conference, which assessed 944 submitted agents from 76 teams across four games: Colonel Blotto, Iterated Prisoner's Dilemma, Codenames, and Secret Mafia. Our analysis surfaces both agent-level and evaluation-level limitations: brittle rule adherence remains a major bottleneck, top-performing systems repeatedly rely on explicit structural scaffolding, and leaderboard validity differs sharply across environments. In particular, failure-heavy environments can reward robustness to opponent errors as much as strategic ability, with Secret Mafia exhibiting a pronounced error-survival confound in this cycle. We release a dataset of 29,571 multi-agent games with turn-level observations, actions, and rewards, together with MG-Ref, a deterministic offline tournament protocol that scores new agents against a frozen reference pool of top-ranked, low-error Stage~II submissions under the same error-attribution lens used in this analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29473v1">Inform, Coach, Relate, Listen: Auditing LLM Caregiving Support Roles</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Language models are increasingly being deployed for conversational support in informal caregiving contexts, where interactions often extend beyond information-seeking: caregivers seek emotional reassurance, guidance, and help, while navigating uncertain, relationally complex care decisions. Yet most safety evaluations assess model behavior under generic prompts, leaving a critical question unexamined: does a model's safety profile change with its support role? We study this by operationalizing four expert-reviewed support roles grounded in social support theory: Inform, Coach, Relate, and Listen, and comparing them against two baseline controls: a basic prompting condition and a retrieval-augmented generation (RAG) condition. We evaluate across three language models (GPT-4o-mini, Llama-3.1-8B-Instruct, and MedGemma-1.5-4b-it) on 5,000 real-world queries from online Alzheimer's Disease and Related Dementias (ADRD) communities. We find that the LLM's support role systematically shapes both the prevalence and composition of interactional risks. Furthermore, a human evaluation study reveals a perceived quality--safety tension: more directive, information-oriented roles are rated as more helpful and trustworthy despite exhibiting elevated interactional risk profiles. We release ~90,000 support role-conditioned model responses with risk annotations as an ecologically grounded resource for research on safer LLM-mediated conversational support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29468v1">SciIntBench: Measuring LLM Compliance with Research Integrity Norms Under Adversarial Framing</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support scientific work, but it is unclear whether they uphold responsible conduct of research (RCR) norms or help undermine them. We introduce SciIntBench, an adversarial benchmark of 810 prompts across ten RCR categories and three scientific domains. Each scenario appears as an Overt Adversarial, Covert Adversarial, and Benign version, allowing us to jointly measure framing-sensitive refusal of misconduct and helpfulness on legitimate requests. We evaluate 16 commercial and open-weight LLMs from six providers (2024--2026), producing 12,960 responses. We find that scientific integrity alignment is strongly framing-sensitive: models refuse explicit misconduct far more reliably than covert violations, especially failing when misconduct is presented as a pressure-driven shortcut. Refusals vary by RCR category, with weaker boundaries around transparency, plagiarism, and fabrication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29461v1">FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 18 pages, accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      LLM-conditioned segmentation has recently advanced rapidly by coupling large language models with iterative mask generation frameworks. However, we identify a persistent failure mode in current propose-then-select pipelines. Although high-quality mask candidates are often generated, the final prediction may fail to match the given linguistic condition. This failure arises because language semantics are typically used as static prompts or post-hoc matching signals, rather than participating in the iterative mask generation process. Through systematic analysis, we show that many errors stem from semantic misalignment rather than poor mask quality. To address this issue, we propose FlowSeg, which introduces dynamic semantic guidance via a bidirectional semantic flow between intermediate decoding states and LLM-derived condition embeddings throughout the generation process. Language conditions actively guide mask refinement at each stage, while condition embeddings are progressively updated by emerging visual evidence. This design yields semantically grounded mask representations and visually aligned language conditions, enabling more reliable matching. We further incorporate a lightweight boundary-aware refinement to selectively enhance uncertain regions without perturbing confident interiors. Extensive experiments on referring expression segmentation and reasoning segmentation tasks demonstrate that FlowSeg consistently improves language-mask alignment and achieves state-of-the-art performance. Project page: https://zkzhang98.github.io/FlowSeg_page
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29458v1">Adaptive Interviewing for Persona Simulation in LLMs: Evidence-Grounded Reasoning Improves Decision Alignment</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 20 pages, 2 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Accurately simulating the decisions of a specific individual remains challenging for large language models (LLMs), partly because persona information is often provided as static descriptions that miss the values, experiences, and contextual cues needed for individual-level decision simulation. We propose an adaptive interview framework that gathers persona-relevant information through a structured three-stage dialogue: core questions, dynamic follow-ups, and a synthesized personality summary. Using the resulting interview transcripts, we evaluate whether LLMs can simulate participants' decisions in moral dilemma scenarios. We compare three conversational contexts -- Core-10 responses, the full interview dialogue, and a summarized persona representation. We find that adaptive interviewing functions less as a uniform accuracy booster and more as a selective grounding mechanism: follow-up-derived evidence is incorporated in around 40% of full-interview traces, and these follow-up-grounded predictions are more accurate than core-only grounded ones (45.5% vs. 39.3%). These findings highlight that richer persona context alone is insufficient: improvements arise only when models actually ground their decisions in user-specific evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28108v2">Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      A long-lived LLM agent, such as OpenClaw, earns its value by acting on a user's preferences and constraints across sessions, not just the current request. Yet today's agents keep what a user volunteers but rarely ask for what stays unspoken, leaving a proactivity gap in long-lived LLM agents: an agent cannot act on a preference it never obtained. As users delegate more of their affairs to agents, the impact of this gap grows. We isolate one concrete, controllable slice of this gap as Ask-to-Remember (ATR): the agent decides whether to ask now for a reusable user preference that the current task does not need but a later session with the same user will. ATR is hard even to evaluate: the right question is underdetermined and its payoff deferred to tasks that may never arise. ATRBench, to the best of our knowledge the first ATR benchmark, makes it measurable by fixing each user's preferences as hidden ground truth, so success demands asking, not recall. Across eight frontier LLM agents, defaults fall at least 62 points below an oracle handed the relevant preference, and prompting closes little of it. Diagnostics identify acquisition as the bottleneck. ATRBench surfaces this proactivity gap in current agents and offers a diagnostic testbed for closing it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29440v1">SkillBrew: Multi-Objective Curation of Skill Banks for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 16 pages. Preprint. Under review
    </div>
    <details class="paper-abstract">
      Retrieval-augmented LLM agents increasingly rely on curated skill banks: collections of reusable textual principles that guide decision making on complex tasks. Existing approaches typically expand these banks in an append-only fashion, continuously adding new skills without removing redundant, outdated, or harmful ones, resulting in inefficient and poorly curated repositories. In this paper, we formulate the skill bank curation as a constrained multi-objective problem: a desirable bank must be useful for the agent, diverse in its content, and provide good coverage of the query distribution. To this end, we introduce SkillBrew, a multi-objective curation framework that formalizes skill bank curation as Pareto-aware optimization under a utility constraint, and solves it via a bi-level propose-then-verify loop. We evaluate our approach on two public benchmarks. Our findings suggest that treating skill banks as objects of principled curation, rather than ever-growing append-only logs, is an important step toward building self-improving LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14365v4">Understanding the Ability of LLMs to Handle Character-Level Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by icml2026
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary large language models (LLMs) against frequent character-level perturbations. We examine three types of character-level perturbations including introducing numerous typos within words, shuffling the characters in each word, and inserting a large number of invisible characters into the text. Surprisingly, even under severe perturbation, such as shuffling nearly all words character-wise to produce text that is almost unreadable to humans, or inserting invisible characters which are several times more than the visible ones as noise, many LLMs still maintain notable performance. We explore the underlying causes of this robustness and find that LLMs exhibit remarkable resilience to chaotic segmentation and fragmented tokenization. Furthermore, we examine the mechanisms by which LLMs remove perturbations to correctly comprehend text, including both implicit and explicit mechanisms for character-level perturbation. We hope that our findings on the low-level robustness of LLMs will unveil their inherent architectural strengths, reveal the potential risks of their misuse, and inform the reliable deployment of LLMs across diverse application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29427v1">FinGuard: Detecting Financial Regulatory Non-Compliance in LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in financial services, a single non-compliant interaction can expose institutions to regulatory penalties and direct consumer harm. Existing guard models are built around general harm taxonomies and overlook violations grounded in specific financial regulations. We address this gap with a regulation-driven pipeline that operates directly on regulatory documents, inducing a financial compliance risk taxonomy and synthesizing grounded training data without any predefined violation categories. Instantiating the pipeline on Chinese financial regulations, we release \textbf{FinGuard-Bench}, to our knowledge the first benchmark for financial regulatory compliance detection, with expert-annotated labels at both the query and response levels. We further train \textbf{FinGuard}, a financial compliance detection model built on Qwen3-8B and trained on the regulation-grounded data via supervised fine-tuning and self-play reinforcement learning. On FinGuard-Bench, FinGuard substantially outperforms all baselines, including dedicated guard models and much larger general-purpose LLMs such as Qwen3.5-397B-A17B and GPT-5.1. Furthermore, FinGuard also preserves general safety capabilities and adapts to unseen institution-specific policies using policy documents alone. We will publicly release the code, prompts, and resources used in this work on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14584v4">ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 18 pages, 4 figures, 10 tables. Accepted at ICML 2026 FoGen Workshop
    </div>
    <details class="paper-abstract">
      We present ReflexGrad, a dual-process architecture for within-episode failure recovery in LLM agents without demonstrations. When agents commit to a wrong approach early and exhaust the step budget, the post-failure trajectory contains the information to escape -- but no published architecture acts on it within a single episode. ReflexGrad routes between a fast process (TextGrad-style continuous refinement every $k{=}3$ steps) and a slow process (Reflexion-style causal diagnosis when $m{=}5$ consecutive low-progress scores fire a routing gate). A deterministic priority merge keeps the natural-language policy coherent, and each slow activation emits three observable artifacts: a reproducible trigger, a causal diagnostic, and a verified fix. On ALFWorld 134 tasks, $n{=}10$ seeds, no demonstrations, ReflexGrad lifts Qwen-3-8B from $35.1\%$ to $75.4\%$ ($+40.3$pp), beating compute-matched 1-shot LATS by $+2.7$pp ($p{\approx}0.01$), ToT by $+5.7$pp ($p{<}10^{-4}$), and Self-Refine by $+6.7$pp ($p{<}10^{-5}$); on GPT-5 the lift is $46.3{\to}88.1\%$ ($+41.8$pp). The $1.5$pp cross-model difference is within seed noise ($p{\approx}0.13$), suggesting that the routing mechanism, rather than model scale, is the primary source of the gain. Code, prompts, per-seed logs, and sensitivity sweeps are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29420v1">When Does Persona Prompting Actually Help? A Retrieval and Metric Analysis of Expert Role Injection in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 6 pages, 2 figures. Submitted for peer review
    </div>
    <details class="paper-abstract">
      Persona prompting is widely used to steer large language models, yet its practical value remains unclear. Prior work often evaluates persona prompting using aggregate scores, making it difficult to determine whether expert-role prompting consistently improves response quality or instead changes responses along different quality dimensions. We study this question through a controlled comparison of four prompting conditions across 1,140 open-ended questions spanning 38 expert roles and six domains: no role prompt, a generic domain-expert prompt, embedding-based role retrieval, and a hybrid retrieval method combining embedding search with LLM-based role selection. Aggregate results show only small overall differences between conditions. However, metric-level analysis reveals a consistent tradeoff that aggregate averages obscure: role prompting systematically increases expertise depth while reducing clarity. These effects are highly conditional rather than universal. Role prompting performs best on advisory questions and in domains such as medicine and psychology, where structured expert framing and risk communication are intrinsically valuable. In contrast, baseline prompting performs better on conceptual and explanatory questions in finance, legal, science, and technology domains, where concise plain-language explanation is more important. We further show that hybrid retrieval significantly improves over embedding-only role selection, although better role retrieval does not eliminate the broader expertise-depth versus clarity tradeoff. Overall, our findings suggest that persona prompting primarily reshapes response characteristics rather than broadly improving capability, and that multi-metric evaluation is necessary for understanding its effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29396v1">Aligned but Fragile: Enhancing LLM Safety Robustness via Zeroth-Order Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Safety alignment for large language models (LLMs) aims to reduce harmful or unsafe behavior while preserving general utility. However, recent findings reveal that alignment effects can be fragile: lightweight post-alignment manipulations, such as parameter noise, activation noise, or quantization, can easily weaken the intended safety behavior. Prior efforts to improve robustness have primarily focused on data curation, modified alignment objectives, and safety-critical parameter identification, leaving the role of the optimizer itself largely unexplored. In this paper, we are the first to study the robustness of safety alignment from the perspective of the base optimizer. This optimizer-centric view naturally points to zeroth-order optimization, which provides a robustness-oriented signal by evaluating safety alignment under perturbations. Based on this insight, we propose a hybrid framework that first performs standard first-order safety alignment and then applies zeroth-order refinement to improve robustness. Both theoretically and empirically, we show that only a few zeroth-order refinement steps can enhance robustness while preserving safety alignment. We further improve the efficiency of zeroth-order refinement by exploiting its inherent perturbation-based evaluations to estimate layer-wise robustness sensitivity, enabling the refinement process to concentrate updates on robustness-critical layers with modest training overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29395v1">Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Pairwise human-preference platforms such as Chatbot Arena have become central to large language model (LLM) evaluation, yet reliable task-specific ranking remains challenging. Global leaderboards mask task heterogeneity, while ranking each fine-grained task independently is unstable under sparse, imbalanced comparisons. We propose a low-rank framework for task-specific LLM ranking from sparse pairwise comparisons, modeling the task-by-model ability matrix $Θ^\star \in \mathbb{R}^{d_t \times d_m}$ as low rank so that information is shared across related tasks while task-specific differences are preserved. We first develop a max-norm ($\ell_\infty$) accurate estimator for the latent scores, combining a convex initializer with alternating-minimization refinement, and prove task-wise top-$K$ recovery guarantees under sparse sampling. Our main contribution is an uncertainty quantification framework for task-specific ranking. We construct cross-fitted one-step debiased estimators for fixed score contrasts -- such as the task-specific ability gap between two models -- yielding asymptotically valid confidence intervals that attain the semiparametric efficiency bound. We then extend the inference to the high-dimensional ranking regime, where per-task ranks and top-$K$ membership are determined by many dependent score-gap hypotheses. Using Gaussian and multiplier-bootstrap calibration, we obtain simultaneous confidence sets for per-task ranks and valid top-$K$ membership tests across many tasks and models. Experiments on synthetic data and Chatbot Arena show that low-rank sharing improves sample efficiency over independent task-wise Bradley-Terry estimation and produces tighter, better-calibrated ranking certificates, with the largest gains in the sparse regime typical of real LLM benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29394v1">EvoMD-LLM: Learning the Language of Species Evolution in Reactive Molecular Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 17 pages, ACL Findings
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) excel at static scientific reasoning, they struggle to model the temporal structure of dynamic physical processes. We present EvoMD-LLM (Evolutionary Molecular Dynamics Large Language Model), a framework that reformulates species-level molecular dynamics as a symbolic temporal language modeling problem. Reactive MD trajectories are discretized into sequences of molecular events, where each token represents a chemical species augmented with its persistence duration, enabling standard autoregressive LLMs to learn compositional evolution over time through efficient fine-tuning. A key component of EvoMD-LLM is temporal scaffolding, which treats event duration as an explicit linguistic token and serves as a structured inductive bias, significantly reducing invalid or hallucinated molecular outputs compared to conventional sequence modeling approaches. We evaluate EvoMD-LLM on multiple temporal prediction tasks, achieving up to 66.14% accuracy and consistently outperforming sequential neural networks and language-based baselines. Beyond quantitative improvements, we qualitatively observe that the model is capable of generating interpretations for its own predictions by incorporating relevant chemical knowledge, even though it was not explicitly supervised with paired trajectory-explanation data. These results demonstrate that symbolic temporal language modeling provides an effective framework for grounding LLMs in dynamic physical simulations.
    </details>
</div>
