# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13397v1">Mod-Guide: An LLM-based Content Moderation Feedback System to Address Insensitive Speech toward Indigenous Ethnic and Religious Minority Communities</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Language operates as a mechanism of both marginalization and resistance, especially for minority communities navigating insensitive and harmful speech online. As content moderation increasingly depends on large language models (LLMs), concerns arise about whether these systems can recognize culturally insensitive speech-language that disregards or marginalizes the cultural and religious perspectives of historically underrepresented communities, often through implicit erasure, misrepresentation, or normative framing, rather than overt hostility. Focusing on Bangladesh's Hindu and Chakma communities -- the country's largest religious and Indigenous ethnic minorities, respectively -- this paper investigates the epistemic limits of LLM-based moderation systems and explores methods for incorporating minority perspectives. We co-created a culturally grounded corpus of insensitive speech with community members and integrated their narratives into moderation pipelines using retrieval augmented generation (RAG). Our tool, Mod-Guide, improves LLM sensitivity to minority viewpoints by leveraging contextual cues derived from lived experience. Through mixed-method evaluations involving both minority and majority participants, we demonstrate that RAG-enhanced moderation responses are more contextually accurate and perceived differently across ethnic lines. This work advances research in human-computer interaction, AI ethics, and social computing by foregrounding restorative justice and hermeneutical inclusion in the design of content moderation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13380v1">An LLM System for Autonomous Variational Quantum Circuit Design</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 63 pages, 19 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The design of high performing quantum circuits remains largely dependent on human expertise. We introduce an autonomous agentic framework that employs large language models (LLMs) to conduct iterative quantum circuit designs under explicit design constraints. Our system integrates seven components: Exploration, Generation, Discussion, Validation, Storage, Evaluation, and Review. These components form a closed-loop workflow that combines web-based knowledge acquisition, literature-grounded critique, executable code generation, and experimental feedback. We evaluate the framework on two tasks: quantum feature map construction for quantum machine learning and ansatz generation for variational quantum eigensolver applications in quantum chemistry. In image classification benchmarks, the best generated feature map outperforms representative quantum feature maps and, when scaled to larger qubit counts, surpasses the classical radial basis function kernel. In molecular ground state estimation across seven molecules, the generated ansatz attains competitive accuracy with widely used chemically inspired and hardware-efficient constructions while satisfying the imposed scaling constraints. These results establish LLM driven agentic system as a viable paradigm for automated quantum circuit design and illustrate how AI systems can participate in iterative scientific optimization workflows across scientific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12160v2">A Controlled Study of Decoding-Time Truthfulness Methods on Instruction-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Decoding-time truthfulness methods -- layer-contrast decoding, inference-time intervention, and learned logit adapters -- have demonstrated 10-30 point gains on TruthfulQA when applied to base language models. However, modern instruction-tuned LLMs already achieve substantially higher baselines (61-76%), raising the question of whether these methods remain effective in practice. We design a six-control evaluation framework -- out-of-distribution training, multi-judge validation, simple decoding baselines, confound controls, bootstrap confidence intervals, and seed variance -- and apply it across 5 models (1B-70B), 3 benchmarks, and 15 methods. We find that previously reported gains shrink substantially under strict controls: on the full TruthfulQA benchmark (N=817), no token-level method achieves statistically significant improvement, and the best learned adapter scores -2.0 points below greedy (p=.23). We identify five evaluation sensitivities -- contamination, judge choice, missing baselines, confounds, and statistical noise -- that individually or jointly account for these discrepancies. Cross-benchmark validation on HaluEval QA and TriviaQA confirms that these patterns extend beyond TruthfulQA. Deliberative prompting methods (chain-of-thought, self-critique) appear more robust in the evaluated regime, with CoT achieving +5.6-19pp across benchmarks as a training-free, single-pass method. We release a seven-point evaluation checklist and discuss implications for future truthfulness research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13317v1">SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Skill self-evolution methods for LLM agents aim to turn execution trajectories into reusable skill documents, but current pipelines typically learn from one trajectory per task, merge candidate skill patches before checking them, and load the full skill corpus before inference. We propose SkillCAT, a training-free framework that separates this process into three stages. Contrastive Causal Extraction (CCE) samples multiple trajectories for each task and compares same-task success/failure pairs to identify evidence that explains outcome differences. Assessment-Augmented Evolution (AAE) replays each candidate patch on source-task clones and keeps only patches that improve or preserve task outcomes before hierarchical skill patch merging. Topology-Aware Task Execution (TTE) compiles the evolved skills into a routable sub-skill topology, so inference loads only the capability nodes relevant to the task. We evaluate SkillCAT on common agent benchmarks, including SpreadsheetBench, WikiTableQuestions, and DocVQA, and further test cross-model and out-of-distribution generalization. Across these settings, SkillCAT raises the average score over baselines by up to 40.40%, demonstrating reliable skill evolution without model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13316v1">ReSum: Synergizing LLM Reasoning and Summarization with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 24 pages, including 13 pages of main text and 11 pages of appendix
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is a central technique for improving long-horizon reasoning in Large Language Models (LLMs). However, existing RLVR methods often encourage unnecessarily long reasoning rollouts, which can degrade reasoning coherence and exhaust the available context budget. Existing approaches to long-context organization often depend on external mechanisms to organize rollouts, rather than enabling the model to manage its own reasoning trajectory. To address this limitation, we propose ReSum, a novel RLVR framework that enables LLMs to compress and organize their reasoning trajectories through self-summarization. Our pilot studies show that self-summarization stabilizes generation by lowering token-level entropy, and that introducing a ``summarization'' phrase can substantially mitigate errors propagated from an incorrect rollout prefix. Motivated by these findings, ReSum adopts a summarization-aware adaptive rollout mechanism that contrastively evaluates whether self-summarization benefits the ongoing reasoning process. Specifically, when the model spontaneously triggers self-summarization, ReSum masks the summarization phrase to create a contrastive branch; for non-summarization positions, it instead randomly injects the phrase to create a matched branch. We further design a summarization-aware advantage to enable finer-grained comparison between contrastive rollout trajectories. Extensive experiments show that ReSum improves performance at an average of 4\% while reducing rollout length by 18.6\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13254v1">Evaluating Pluralism in LLMs through Latent Perspectives</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Pluralistic Alignment Workshop @ ICML 2026
    </div>
    <details class="paper-abstract">
      The growing need to represent diverse perspectives has increased interest in pluralistic LLM generation. Although difficult to operationalize, identifying perspectives expressed in text would provide clear guidance on pluralistic alignment and more clearly articulate the pluralistic gap in LLM generation. While models have been shown to reduce the diversity of training data and generate homogeneously, this has been demonstrated primarily on multiple-choice questionnaires or using high-level characteristics of free-form text. In this paper, we introduce and implement a domain-agnostic multi-layered framework for unsupervised extraction of perspectives suitable for identifying the pluralistic gap in LLM-generated text. We evaluate our framework on book reviews, a highly opinionated dataset representing diverse perspectives, and compare various prompts and models. Our results show that while some models and prompting techniques come close to covering a broad spectrum of perspectives, rarer perspectives remain disproportionately underrepresented, resulting in distributions that diverge from human text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09500v3">Deterministic Integrity Gates for LLM-Assisted Clinical Manuscript Preparation: An Auditable Biomedical Informatics Architecture</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 28 pages, 3 figures, 4 tables; includes supplementary material (deterministic-detector inventory, per-class defect breakdown, worked example). Software (MIT): https://github.com/Aperivue/medsci-skills . Archived on Zenodo: concept DOI https://doi.org/10.5281/zenodo.20155321 and version DOI (v3.8.0) https://doi.org/10.5281/zenodo.20582972
    </div>
    <details class="paper-abstract">
      As autonomous research agents and AI co-scientist systems push large language models (LLMs) from drafting toward end-to-end manuscript production, the bottleneck shifts from generation to verification. Fluent LLM output can hide fabricated citations, numbers that drift from source tables, and unmet reporting-guideline items; existing tools generate without verifying, and self-critique inherits the blind spots that produce confident fabrication. We describe an architecture pairing generation with verification, resting on three principles: decompose the workflow into self-contained skills, gate every stage transition with halt-on-failure, and resolve each integrity question with the cheapest sufficient mechanism, a deterministic, re-executable check where one suffices and a prose-level probe only where interpretation is unavoidable. This determinism-where-possible split, organized as an integrity-gate taxonomy, is the core contribution. It is realized as MedSci Skills, an open-source toolkit of 43 skills with a 21-detector deterministic tier, evaluated on three public-dataset pipelines (STARD, PRISMA, STROBE) and a seeded-defect ablation. Across the three pipelines every content-hash manifest verified clean and the gates surfaced real defects; on 27 identical injected defects the deterministic gates detected all 27 with no false positives on the matched clean fixtures, whereas a single-prompt LLM reviewer detected 11, its misses in code, bibliography, and style defects the prose hides. Determinism-where-possible verification yields an auditable, re-executable trail that exposes the evidence a human needs to check an LLM-assisted manuscript: feasibility and reproducibility evidence, not a claim of human-competitive quality, which a separate blinded study addresses. MedSci Skills is MIT-licensed and archived (v3.8.0).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17062v2">The Range Shrinks, the Threat Remains: Re-evaluating LLM Package Hallucinations on the 2026 Frontier-Model Cohort</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 13 pages, 3 figures, 4 tables. v2: incorporates coordinated-disclosure feedback from PyPI Security and Socket.dev; registrable attack surface refined to 53 names (41 PyPI, 12 npm). Headline rates unchanged. Replication of Spracklen et al. (USENIX Security 2025). Data and code: https://github.com/churik5/slopsquatting-replication-2026 and https://doi.org/10.5281/zenodo.19859120
    </div>
    <details class="paper-abstract">
      Spracklen et al. (USENIX Security '25) showed that code-generating large language models hallucinate package names that do not exist on PyPI or npm at rates ranging from 5.2% on commercial models to 21.7% on open-source models, creating an attack surface for slopsquatting -- the registration of malicious packages under hallucinated names. We replicate their methodology on five frontier code-capable LLMs released between October 2025 and March 2026: Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.4-mini, Gemini 2.5 Pro, and DeepSeek V3.2. Across 199,845 paired Python and JavaScript prompts validated against PyPI and npm master lists, we measure overall hallucination rates between 4.62% (Claude Haiku 4.5) and 6.10% (GPT-5.4-mini) -- an order-of-magnitude compression of the inter-model spread observed by Spracklen, but not a retirement of the threat. Beyond replication, we identify a set of 127 package names (109 on PyPI, 18 on npm) that all five evaluated models invent identically; following coordinated disclosure with PyPI Security and Socket.dev, 53 of these (41 on PyPI, 12 on npm) remain registrable by an attacker after each registry's existing defenses, constituting a model-agnostic supply-chain attack surface that no single-model study can reveal. We further document a Python-over-JavaScript hallucination asymmetry that inverts Spracklen's 2024 finding, identify a Haiku-below-Sonnet inversion within the Anthropic family, and observe a Jaccard-similarity peak between DeepSeek V3.2 and GPT-5.4-mini (J = 0.343) suggestive of shared training-data origins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13221v1">From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Evaluating new large language models typically requires costly human annotation campaigns at scale. LLM-as-a-judge offers a cheaper alternative, but judge scores carry systematic errors - such as position bias, self-preference, or intransitivity - that can strongly miscalibrate the resulting rankings. We quantify the resulting judge-human disagreement at two complementary levels. At the local level, we estimate per-battle uncertainty from the judge's own score differences by propagating calibrated win probabilities rather than hard labels into the Bradley-Terry procedure. This alone provides a drastic improvement to Elo estimation accuracy, bringing LLM-derived ratings within 17.9 Elo MAE of human-derived ones when averaged over 55 held-out models on LMArena. At the global level, we apply split conformal prediction to the residual gap between LLM-derived and human-derived Elo ratings across held-out models, producing prediction intervals with distribution-free marginal coverage guarantees that account for irreducible LLM-human disagreement. Together, these two layers yield a low-cost evaluation tool that provides developers with calibrated Elo estimates and honest uncertainty bounds, without access to large-scale human annotations.To facilitate reproducibility, we release our code at https://github.com/kargibora/SoftElo .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13220v1">LLM-as-an-Investigator: Evidence-First Reasoning for Robust Interactive Problem Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as interactive assistants for technical problem solving. However, when users provide incomplete descriptions or plausible but unverified explanations, LLMs may prematurely align with these assumptions and propose solutions before collecting sufficient evidence. We refer to this behavior as user-driven sycophancy: the tendency of an LLM to reinforce a user-provided hypothesis instead of testing alternative explanations. This paper introduces LLM-as-an-Investigator, an evidence-first agentic AI methodology for robust problem diagnosis. The approach is implemented through a Solution Investigator Agent, which estimates the ambiguity of an initial problem description, generates candidate hypotheses, asks targeted clarification questions, and updates hypothesis probabilities after each answer. Rather than producing an immediate response, the agent continues the investigation until the evidence makes one candidate explanation stronger than the alternatives. To evaluate the approach, we build a benchmark from solved technical forum threads in mechanical, electrical, and hydraulic domains. We use a three-agent evaluation pipeline in which a Problem-Solution Extractor Agent converts solved threads into structured cases, a Ground-Truth Evaluator Agent simulates the user while hiding the known solution, and the tested assistant attempts to recover the solution through dialogue. The experiments compare standard assistants, reasoning-oriented LLMs, and the proposed investigator-based model across LLM backbones. In addition to diagnostic accuracy, we analyze how standard assistants follow misleading user hypotheses in diagnostic cases. The results show that the proposed approach identifies the problem more accurately than direct prompting and reasoning-only baselines, while its evidence-first protocol helps reduce user-induced conversational bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13218v1">When Similar Means Different: Evaluating LLMs on Arabic--Hebrew Cognates</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Arabic and Hebrew, as closely related Semitic languages, share a substantial lexicon of true cognates, misleading false friends, and modern loanwords. This overlap poses a challenge for cross-lingual semantic understanding in large language models (LLMs). To evaluate this capability, we introduce SemCog Bench, a curated benchmark of 1,858 Arabic--Hebrew word pairs with sentence-level annotations for cognate identification and semantic disambiguation. We evaluate open-source and commercial LLMs across multiple input representations (raw, diacritized, Romanized, and phonetic) and reveal a critical gap in cross-lingual reasoning. While models achieve high accuracy on true cognates, performance drops sharply on false friends and loanwords, reflecting a strong reliance on surface-form similarity. Furthermore, sentence-level context yields only modest improvements, suggesting that contextual cues alone are insufficient to overcome misleading form-based signals. These findings reveal a fundamental limitation of current LLMs in resolving cross-lingual form--meaning conflicts and establish SemCog Bench as a rigorous benchmark for multilingual semantic reasoning. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08098v2">When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Preprint. 16 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Majority voting over sampled answers is the dominant unsupervised aggregator for multi-sample LLM inference. In this paper, we show a delegation-based aggregator (Propagational Proxy Voting, PPV; Sakai et al., 2025) yields an unsupervised consensus rule that beats majority on MMLU-Pro by +1.5 pp overall and +2.24 pp on the non-trivial subset (paired McNemar p ~ 1.0e-14, n = 8,099). Majority discards two signals that every sample carries: within-group letter entropy and between-group reasoning geometry. PPV exposes per-voter levers that consume exactly these two signals: When (how much weight a voter keeps on its own pick) and Whom (how it splits the remainder across peers). We drive When with letter entropy and Whom with per-question-centered embedding cosine. Our method needs no gold labels and no auxiliary training: per-question, we partition 128 sampled generations into 16 groups, compute each group's letter-level semantic entropy and reasoning embedding centroid, and feed both into a stochastic delegation matrix whose stationary distribution selects the consensus answer. We walk through an example in which PPV overturns a clear 10-6 majority for the wrong letter: the 10-voter majority cluster is geometrically incoherent (mean within-cluster cosine -0.02) while the 6-voter minority is tight (+0.26), so propagated delegation mass concentrates on the minority's answer even though entropy alone would keep the majority ahead. We further report delegation strategies with negative results that constrain the design space for unsupervised LLM aggregation. No within-question ensemble of confidence modes closes the oracle gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13189v1">SICI: A Semantic-Pragmatic Complexity Index Reveals Regime Shifts in LLM Stance Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Prompt-based LLMs are increasingly used for stance detection, but harder examples are not always repaired by clearer instructions, reasoning prompts, retrieval, or debate. We introduce SICI (Stance Inference Complexity Index), a seven-dimensional diagnostic measure of the semantic-pragmatic burden imposed by a target--text pair. Across SemEval-2016 and VAST, SICI predicts LLM accuracy better than surface proxies and shows substantial cross-scorer reliability ($α=0.771$). More importantly, LLM errors change regime as SICI increases: low-complexity examples invite over-attribution, especially Against predictions; intermediate examples form an unstable boundary; and high-complexity examples rapidly concentrate on None. This phase-transition-like structure persists across GPT-3.5, GPT-4o-mini, DeepSeek-V3, and GPT-4o, although stronger models move the boundaries. A 15-method intervention study further shows that prompting, retrieval, and debate often shift models along the attribution--abstention axis rather than removing the high-complexity bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13177v1">MemRefine: LLM-Guided Compression for Long-Term Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly expected to operate over long-term interactions, where information from past dialogues must be preserved and recalled to support future tasks. However, as interactions accumulate, the memory store grows without bound and fills with redundant entries that inflate storage cost and degrade retrieval by crowding out the most useful evidence. Furthermore, this is especially limiting on resource-constrained platforms with hard memory budgets, motivating us to formulate storage-budgeted memory management, the task of keeping an already constructed memory store within a fixed budget while preserving information useful for future interactions. To this end, we then propose MemRefine, an LLM-guided framework that, since surface similarity poorly reflects factual value, uses similarity only to propose candidate pairs and defers delete, merge, and preserve decisions to an LLM judge based on factual content, iterating until the budget is met. Across multiple memory frameworks and long-term conversation benchmarks, MemRefine consistently meets target budgets while preserving downstream performance and outperforming rule-based baselines under tight budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13171v1">NTS-CoT: Mitigating Hallucinations in LLM-based News Timeline Summarization with Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      The rapid updates of online news make tracking event developments challenging, highlighting the need for timeline summarization (TLS). Hallucinations, where LLM-generated content deviates from source news, still remain a critical issue in LLM-based TLS and are not well studied in existing works. To bridge this gap, we identify two primary types of hallucinations: unfaithful content during news summarization and information omission in date-event summarization. Then, we propose NTS-CoT, a novel framework that leverages Chain-of-Thought (CoT) reasoning to mitigate hallucinations in TLS. The framework consists of three key modules: i) Element-CoT to capture essential news elements for faithful summarization, ii) Date Selection to combine temporal saliency and event prominence for timestamp selection, and iii) Causal-CoT to infer causal relationships and reduce omissions in date-event summarization. Extensive experiments, including quantitative analysis on three TLS benchmarks and human evaluation, demonstrate that NTS-CoT outperforms state-of-the-art baselines, effectively mitigating hallucinations and improving LLM-based TLS performance. Our source code is available at https://anonymous.4open.science/r/NTS-CoT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13140v1">MIDSim: Simulating Multi-Channel Information Diffusion in Social Media with LLM-Powered Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Information diffusion in social media shapes public opinion and collective behavior, making its modeling and simulation an important research problem. Existing studies have investigated information diffusion through epidemic-based, cascade-based, and point process models. However, they predominantly focus on diffusion through social links, overlooking other diffusion channels enabled by platform algorithms (e.g., recommender systems) and failing to capture user behavioral complexity. To address these limitations, we propose an LLM-powered multi-agent system for simulating multi-channel information diffusion, where large language models instantiate personalized user agents and the diffusion process jointly models social and algorithmic exposure streams. We further construct three real-world diffusion dataset spanning Sina Weibo, RedNote, and Twitter, containing diffusion records, user profiles, historical posts, and social relationships. Experimental results on real diffusion events show that our proposed framework realistically simulate macro diffusion phenomenon and generate diverse comment content, significantly outperforming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13111v1">MÖVE: A Holistic LLM Benchmark for the German Public Sector</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      We present MÖVE (Modelle für die Öffentliche Verwaltung Evaluieren), a holistic benchmark for evaluating large language models (LLMs) in the context of the German public sector. While LLMs are increasingly adopted in public administration, model selection remains largely ad hoc, and existing benchmarks offer limited guidance: they are predominantly English-centric, US-centric in content, and focus exclusively on task performance. MÖVE addresses these gaps by evaluating 39 models across two complementary dimensions. Performance criteria cover summarization, question answering, and topic extraction. Governance criteria assess hallucination tendencies, energy consumption, provider transparency, and alignment with German constitutional values and knowledge about positions by German political parties. In total, we utilize ten German-language datasets, including gold- and silverstandard datasets that we constructed to reflect public-administration domains. We employ a multi-metric evaluation strategy combining classical NLP metrics, embedding-based methods, and LLM-as-a-judge approaches. Our results show that no single model dominates across all criteria: top performers differ between tasks, and model size alone is a poor predictor of quality. We further evaluate the benchmark itself, analyzing its statistical precision, LLM judge reliability, the impact of our private datasets on model rankings, the sensitivity of our results to prompt formulation, and the validity of our energy consumption estimates. MÖVE is designed as a living benchmark under active development; results are publicly available at https://moeve.bundesdruckerei.de/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13095v1">Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted in Interspeech 2026
    </div>
    <details class="paper-abstract">
      Multi-talker speech recognition is often addressed by combining automatic speech recognition (ASR) and speaker diarization in a pipeline system. Recently, LLM-based approaches have shown promise by jointly modeling semantic and speaker information, but they typically require large-scale multi-talker corpora that are costly to annotate. In this paper, we investigate how to efficiently train an LLM-based system with limited real-recorded data while maintaining high accuracy in speaker attribution. We propose several strategies: (1) a dual-encoder architecture to extract semantic and speaker features, (2) a feature interleaving format to merge these features as the inputs to the LLM, (3) a length-aware speaker ID loss to enhance diarization capability, and (4) an adaptive threshold strategy for ASR loss computation to mitigate hallucinations caused by speech overlaps. These strategies balance training between ASR and diarization tasks. Our system outperforms open-source baseline approaches, achieving relative improvements of 18% on the AliMeeting corpus and 24% on the Aishell4 corpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13082v1">sebis at CRF Filling 2026: A Two-Stage Local LLM Pipeline for Medical CRF Filling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Published in Proceedings of the Third Workshop on Patient-Oriented Language Processing (CL4Health), LREC 2026
    </div>
    <details class="paper-abstract">
      The extraction of structured clinical information from unstructured EHR notes is a persistent bottleneck in healthcare informatics. While large language models (LLMs) offer high performance, their deployment in clinical settings is hindered by privacy risks, inference costs, and the tendency to hallucinate beyond textual evidence. We address these challenges for the CL4Health 2026 Case Report Form (CRF) filling task by proposing a fully local, domain-adapted pipeline using the MedGemma-27B model. Our two-stage architecture, which separates binary presence classification from value extraction, enforces strict adherence to textual evidence and ensures deterministic outputs for negated, uncertain, or unknown states. By leveraging item-specific, few-shot in-context learning without external API calls or fine-tuning, our approach achieves a macro-F1 score of 0.55 on the official English test track. This result secures second place among all locally-hosted, open-source submissions. Our work demonstrates that privacy-preserving, on-premise LLM pipelines can achieve near-competitive performance with proprietary frontier models, providing a practical, data-sovereign framework for clinical NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04885v2">CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) serve a global audience, alignment must transition from enforcing universal consensus to respecting cultural pluralism. We demonstrate that dense models, when forced to fit conflicting value distributions, suffer from \textbf{Mean Collapse}, converging to a generic average that fails to represent diverse groups. We attribute this to \textbf{Cultural Sparsity}, where gradient interference prevents dense parameters from spanning distinct cultural modes. To resolve this, we propose \textbf{\textsc{CuMA}} (\textbf{Cu}ltural \textbf{M}ixture of \textbf{A}dapters), a framework that frames alignment as a \textbf{conditional capacity separation} problem. By incorporating demographic-aware routing, \textsc{CuMA} internalizes a \textit{Latent Cultural Topology} to explicitly disentangle conflicting gradients into specialized expert subspaces. Extensive evaluations on WorldValuesBench, Community Alignment, and PRISM demonstrate that \textsc{CuMA} achieves state-of-the-art performance, significantly outperforming both dense baselines and semantic-only MoEs. Crucially, our analysis confirms that \textsc{CuMA} effectively mitigates mean collapse, preserving cultural diversity. Our code is available at https://github.com/Throll/CuMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13054v1">TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit exceptional general language processing capabilities, but their memory and compute costs hinder deployment. Ternarization has emerged as a promising compression technique, offering significant reductions in model size and inference complexity. However, existing methods struggle with heavy-tailed activation distributions and therefore keep activations in high precision, fundamentally limiting end-to-end inference acceleration. To overcome this limitation, we propose TWLA, a post-training quantization (PTQ) framework that achieves 1.58-bit weight compression and 4-bit activation quantization while maintaining high accuracy. TWLA comprises three components: (1) Euclidean-to-Manifold Asymmetric Ternary Quantizer (E2M-ATQ) minimizes layer-output error under weight ternarization via a two-stage optimization from Euclidean initialization to manifold relocation; (2) Kronecker Orthogonal Tri-Modal Shaping (KOTMS) applies a Kronecker-structured orthogonal rotation to reshape weights into ternary-friendly tri-modal distributions, while the shared rotation statistically suppresses activation outliers; and (3) Inter-Layer Aware Activation Mixed Precision (ILA-AMP) explicitly introduces adjacent-layer second-order interaction costs in bit allocation and jointly optimizes for the layer-wise disparity of activation quantization gains induced by the shared orthogonal transform, preventing cascades triggered by a few weak layers. Extensive experiments demonstrate that TWLA maintains high accuracy under W1.58A4, while delivering significant inference acceleration. The code is available at <https://github.com/Kishon-zzx/TWLA>.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14568v2">Given, When, Then, Again: Mining Subscenario Refactoring Candidates in Behaviour-Driven Test Suites with ML Classifiers and LLM-Judge Baselines</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 31 pages, 10 figures, 6 tables, 56 references. v2: retitled; reference list fully corrected and verified; decision-threshold sensitivity analysis and imbalance-robust baseline metrics added; figures restyled. Reproduction package at https://github.com/amughalbscs16/cukereuse_subscenarios_release (Apache-2.0). Upstream cukereuse corpus at https://doi.org/10.5281/zenodo.19754359
    </div>
    <details class="paper-abstract">
      Context. Behaviour-Driven Development (BDD) test suites accumulate duplicated step subsequences. Three published refactoring patterns are available (within-file Background, within-repo reusable-scenario invocation, cross-organisational shared higher-level step), but no prior work automates which recurring subsequences are worth extracting or which mechanism applies. Objective. Rank recurring step subsequences ("slices") by refactoring suitability (extraction-worthy), pre-map each to one of the three patterns, and quantify prevalence across the public BDD ecosystem. Method. Every contiguous L-step window (L in [2, 18]) in a 339-repository / 276-upstream-owner Gherkin corpus is keyed by paraphrase-robust cluster identifiers and counted under three scopes. SBERT / UMAP / HDBSCAN clustering recovers paraphrase-equivalent slices. Three authors label a stratified 200-slice pool against a written rubric. An XGBoost extraction-worthy classifier trained under 5-fold cross-validation is compared with a tuned rule baseline and two open-weight Large Language Model (LLM) judges. Results. The miner produces 5,382,249 slices collapsing to 692,020 recurring patterns. Three-author Fleiss' kappa = 0.56 (extraction-worthy) and 0.79 (mechanism). The classifier reaches out-of-fold F1 = 0.891 (95% CI [0.852, 0.927]), outperforming both the rule baseline (F1 = 0.836, p = 0.017) and the better LLM judge (F1 = 0.728, p = 1.5e-4). 75.0%, 59.5%, and 11.7% of scenarios carry a within-file Background, within-repo reusable-scenario, and cross-organisational shared-step candidate, respectively; the figures are stable under a sweep of the classifier decision threshold. Conclusion. Paraphrase-robust subscenario discovery yields a corpus-wide census of BDD refactoring candidates; pipeline, classifier predictions, labelled pool, and rubric are released under Apache-2.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13020v1">SciR: A Controllable Benchmark for Scientific Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Three paradigmatic forms of inference recur across scientific reasoning: deduction, induction, and causal abduction. Reliably evaluating LLMs on these in scientific settings is currently out of reach: scientific benchmarks built on human annotations are costly and lack mechanistic ground truth, while synthetic logical-reasoning benchmarks do not resemble real scientific documents. We introduce SciR, a benchmark that combines multi-paradigm reasoning with controllable scientific rendering, anchored on three paradigmatic scientific problems. Tasks are generated from formal objects (deduction tree, inductive rule hypothesis, causal graph) to guarantee verifiable answers, then rendered into multi-document scientific discourse via per-track domain-tuned genres. The construction lets us independently vary two difficulty axes: how hard it is to extract the key information needed for inference, and how hard the principled inference itself is. We test six models. Both axes hurt every model, and their effects compound. The rendering even hurts neurosymbolic pipelines, which hand inference to a verified solver. The two axes yield a per-model extraction-vs-inference profile: for instance, reasoning models like deepseek-r1 mostly surpass non-reasoning instruct models on the inference axis. To our knowledge, SciR is the first multi-paradigm scientific-reasoning benchmark with parametric control on both extraction and inference difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13007v1">scLLM-DSC: LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering for Single-Cell RNA Sequencing</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Clustering is fundamental to scRNA-seq analysis, serving as a cornerstone for identifying cell populations and resolving tissue heterogeneity. However, existing methods focus on mining numerical statistical patterns, suffering from semantic agnosticism by neglecting the intrinsic biological functions encoded by genes. While Large Language Models (LLMs) offer promising semantic capabilities, their direct adaptation to cell clustering is hindered by the structural mismatch between generative pre-training objectives and discriminative downstream tasks. To bridge this gap, we propose scLLM-DSC, a novel LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering framework. Diverging from data-driven paradigms, scLLM-DSC establishes a semantically-grounded representation by synergizing two views: a Knowledge-Driven Semantic View derived from NCBI gene priors and contextualized Cell2Sentence embeddings, and a Structure-Aware Topological View extracted via a graph-guided encoder. Crucially, we introduce a cross-modal contrastive alignment mechanism to enforce consistency between biological semantics and transcriptomic features within a unified latent space. Extensive benchmarks demonstrate that scLLM-DSC significantly outperforms eleven state-of-the-art baselines in clustering accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13006v1">Emo-LiPO: Listwise Preference Optimization for Fine-Grained Emotion Intensity Control in LLM-based Text-to-Speech</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted by IJCAI 2026. Emotional TTS, Preference Optimization, Emotion Intensity Control
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based text-to-speech (TTS) systems enable prompt-conditioned emotional control but struggle with fine-grained emotion intensity due to the semantic -- acoustic gap between text and speech. To address this challenge, we formulate emotion intensity control in LLM-based TTS as a learning-to-rank problem and propose Emo-LiPO, a listwise preference optimization framework that aligns prompt-conditioned speech generation with relative emotion intensity expressed in text. Emo-LiPO explicitly models global intensity ordering within each emotion under fixed transcripts, enabling more faithful and continuous emotional expression. We further construct ESD-plus, a multi-speaker dataset with explicit emotion intensity variations, to support fine-grained emotion modeling and evaluation. Experiments on ESD-plus demonstrate that Emo-LiPO significantly improves emotion accuracy and intensity controllability over both supervised- and DPO-based LLM TTS baselines, with particularly pronounced gains at high intensity levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01572v2">LLM-based Embeddings: Attention Values Encode Sentence Semantics Better Than Hidden States</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Sentence representations are foundational to many Natural Language Processing (NLP) applications. While recent methods leverage Large Language Models (LLMs) to derive sentence representations, most rely on final-layer hidden states, which are optimized for next-token prediction and thus often fail to capture global, sentence-level semantics. This paper introduces a novel perspective, demonstrating that attention value vectors capture sentence semantics more effectively than hidden states. We propose Value Aggregation (VA), a simple method that pools token values across multiple layers and token indices. In a training-free setting, VA outperforms other LLM-based embeddings, even matches or surpasses the ensemble-based MetaEOL. Furthermore, we demonstrate that when paired with suitable prompts, the layer attention outputs can be interpreted as aligned weighted value vectors. Specifically, the attention scores of the last token function as the weights, while the output projection matrix ($W_O$) aligns these weighted value vectors with the common space of the LLM residual stream. This refined method, termed Aligned Weighted VA (AlignedWVA), achieves state-of-the-art performance among training-free LLM-based embeddings, outperforming the high-cost MetaEOL by a substantial margin. Finally, we highlight the potential of obtaining strong LLM embedding models through fine-tuning Value Aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27960v2">LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have achieved impressive reasoning milestones but continue to struggle with high computational costs, logical inconsistencies, and sharp performance degradation on high-complexity problems. While neuro-symbolic methods attempt to mitigate these issues by coupling LLMs with symbolic reasoners, existing approaches typically rely on monotonic logics (e.g., SMT) that cannot represent defeasible reasoning -- essential components of human cognition. We present "LLM+ASP," a framework that translates natural language into Answer Set Programming (ASP), a nonmonotonic formalism based on stable model semantics. Unlike prior "LLM+ASP" approaches that require manually authored knowledge modules, domain-specific prompts, or evaluation restricted to single problem classes, our framework operates without any per-task engineering and applies uniformly across diverse reasoning tasks. Our system utilizes an automated self-correction loop where structured feedback from the ASP solver enables iterative refinement. Evaluating across six diverse benchmarks, we demonstrate that: (1) stable model semantics allow LLMs to naturally express default rules and exceptions, outperforming SMT-based alternatives by significant margins on nonmonotonic tasks; (2) iterative self-correction is the primary driver of performance, effectively replacing the need for handcrafted domain knowledge; (3) compact in-context reference guides substantially outperform verbose documentation, revealing a "context rot" phenomenon where excessive context hinders constraint adherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12983v1">Structured Testbench Generation for LLM-Driven HDL Design and Verification-Oriented Data Curation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 9 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Automated testbench generation has become a critical bottleneck in large language model (LLM)-driven Register Transfer Level (RTL) workflows, where large numbers of candidate designs must be verified rapidly and reliably. Existing prompt-based approaches treat testbench generation as unconstrained code synthesis, yielding stochastic outputs with high token cost, low reproducibility, and insufficient coverage. To address this gap, we present STG, a Structured Testbench Generation framework that exploits the inherent structure of hardware designs to generate deterministic testbenches. As a direct verification tool, STG runs 720x faster than an iterative LLM-based testbench generation flow and higher rate of successful compilation, achieves higher coverage, and reduces false-pass verdicts on incorrect DUTs. STG also helps identify errors in RTL generation benchmarks by exposing faulty benchmark testbenches. As a data curation engine, it is 11x faster than LLM-based filtering on a single CPU core with 127x less energy, and the resulting distilled models provide state-of-the-art performance in our multi-benchmark evaluation. As a test-time scaling oracle, it reduces node count by 14-47\%. Our models are available at https://huggingface.co/collections/AS-SiliconMind/siliconmind-v12.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12950v1">Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted to the 46th IEEE International Conference on Distributed Computing Systems (ICDCS 2026). 11 pages
    </div>
    <details class="paper-abstract">
      Large Language Model based Multi-Agent Systems (LLM-MAS) have emerged as a powerful paradigm for tackling complex tasks by breaking them into collaborative workflows of specialized LLM-powered agents. However, deploying such multi-agent workloads at scale poses significant system challenges. Each user query spawns an iterative pipeline of LLM calls, greatly amplifying resource consumption compared to single-turn queries. In resource-constrained cloud settings, these workflows face non-deterministic and input-dependent costs at decode stage, heavy-tailed multi-model requirements with memory fragmentation and over-provisioning, and cross-cluster scheduling trade-offs. We present Maestro, a workload-aware scheduling system designed for LLM-MAS serving under strict GPU budgets. Maestro explicitly leverages agent semantics and roles: it predicts the output length and memory usage of each stage and uses this prediction to drive a hierarchical scheduler. At the node level, Maestro enables dynamic multi-model co-location via hierarchical weight caching and elastic memory provisioning. At the cluster level, it performs latency-aware routing to avoid cold-start delays and memory overloads. At the global level, it enforces workflow-aware prioritization to minimize head-of-line blocking for interactive tasks. Across prototype experiments and trace-driven simulations, Maestro reduces KV-reservation HBM by 67.2% and improves high-contention SLO attainment over EDF by 23.6 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16430v3">A Theory of Training Profit-Optimal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Minor edits for preprint
    </div>
    <details class="paper-abstract">
      Scaling LLMs requires tremendous computational resources, and recent advances in AI have gone hand in hand with massive amounts of capital expenditure. While it is established that scaling up LLMs reliably increases model quality (quantified in terms of loss or downstream evaluations), it is unclear how these quality improvements translate to potential revenue, and whether revenue increases would offset costs of larger-scale training and inference. In this work, we develop an economic model for characterizing the rational behavior of an LLM training firm by combining scaling laws with microeconomic theory. Under our model of firm behavior, LLM quality can be increased with more parameters and training tokens, leading to more potential adoption by consumers, who each have a quality threshold for using the LLM. On the other hand, additional parameters and training tokens both incur additional costs. We analyze the profit maximization problem for this model under compute-bound and data-bound regimes. In the compute-bound regime, optimal model size and token budget track hardware efficiency $E$ (FLOPs/\$) at a near-linear rate; total training cost then scales sub-quadratically in $E$. Data efficiency improvements incentivize larger models and training expenditure. When we are limited to $D$ data, profit-optimal training expenditure scales as $D^2/E$, i.e, increase with data and decreases with hardware efficiency (as well as data efficiency). Finally, we analyze practical trends in training expenditure: current trends are consistent with our most permissive model variants in the compute-bound regime, but are not profit-optimal in the data-bound regime or assuming hardware advances will stall. Overall, our results provide a theory of profit-optimal LLM training, providing a foundation for engaging critically with industry statements and supporting long-term economic decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12935v1">MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Parallel test-time scaling samples many reasoning traces and majority-votes their answers, improving LLM accuracy but requiring traces to run to completion, incurring substantial computational overhead. We observe that probing partial traces at intermediate checkpoints can extract current answers without disrupting generation, revealing an evolving aggregate vote. Based on this observation, we introduce MARS, a margin-adversarial stopping rule that estimates which active traces are likely to change their answers and stops once the leader remains safe under a conservative bound on future vote movement. The rule separates two sources of uncertainty. It learns the trace-level switch probabilities that determine how much of the current margin is likely to be retained, while handling the harder question of where switching traces land through an adversarial bound calibrated from warmup traces. With true switch probabilities, MARS guarantees with high probability that the early-stopped answer matches the full-budget vote. In practice, a five-feature logistic model closely matches oracle switching behavior. Across three reasoning models and three competition-math benchmarks, MARS saves 25-47% of self-consistency tokens and 14-29% on top of DeepConf Online, a strong confidence-weighted baseline that already filters and truncates weak traces, while matching the accuracy of the corresponding full-budget baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22695v2">LLM-ODDR: A Large Language Model Framework for Joint Order Dispatching and Driver Repositioning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Published in IEEE Transactions on Intelligent Transportation Systems (TITS)
    </div>
    <details class="paper-abstract">
      Ride-hailing platforms face significant challenges in optimizing order dispatching and driver repositioning operations in dynamic urban environments. Traditional approaches based on combinatorial optimization, rule-based heuristics, and reinforcement learning often overlook driver income fairness, interpretability, and adaptability to real-world dynamics. To address these gaps, we propose LLM-ODDR, a novel framework leveraging Large Language Models (LLMs) for joint Order Dispatching and Driver Repositioning (ODDR) in ride-hailing services. LLM-ODDR framework comprises three key components: (1) Multi-objective-guided Order Value Refinement, which evaluates orders by considering multiple objectives to determine their overall value; (2) Fairness-aware Order Dispatching, which balances platform revenue with driver income fairness; and (3) Spatiotemporal Demand-Aware Driver Repositioning, which optimizes idle vehicle placement based on historical patterns and projected supply. We also develop JointDR-GPT, a fine-tuned model optimized for ODDR tasks with domain knowledge. Extensive experiments on real-world datasets from Manhattan taxi operations demonstrate that our framework significantly outperforms traditional methods in terms of effectiveness, adaptability to anomalous conditions, and decision interpretability. To our knowledge, this is the first exploration of LLMs as decision-making agents in ride-hailing ODDR tasks, establishing foundational insights for integrating advanced language models within intelligent transportation systems. While the current framework incurs higher computational costs than traditional methods, we show that parallel decomposition and model distillation can reduce latency to production-viable levels for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.20208v2">From Benchmarks to Skills: Low-Rank Factors for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Current evaluations of large language models (LLMs) rely heavily on a growing collection of benchmarks and on aggregate benchmark scores, yet it remains unclear what this comparison actually captures, and what these scores reveal about models' underlying capabilities. Here, we propose a new paradigm for LLM evaluation, by asking whether benchmark performance reflects many independent abilities, or rather relies on a small number of shared dimensions. To answer this, we apply Factor Analysis (FA) to a massive performance matrix of LLMs versus benchmarks \((60\times44)\) revealing an \emph{intrinsically low-rank} structure of that matrix. That is, a small number of latent factors captures most of the structure in the full task space. This low-rank geometry reveals substantial redundancy across existing tasks and explains why many benchmarks appear to be measuring overlapping abilities. We further show that these latent factors correspond to coherent, skill-like, dimensions of LLM behavior. Leveraging this latent skill-space, we deliver three practical tools for LLM evaluation and downstream users: (i)~identifying redundant tasks, (ii)~profiling new models using a small subset of tasks, and (iii)~selecting models aligned with desired skill profiles. Our method provides a solid alternative to the de-facto standard of a single aggregate score, and establishes an interpretable and practical framework for understanding and benchmarking LLM core capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12922v1">Polar: A Benchmark for Evaluating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Submitted to ARR 2026 May cycle
    </div>
    <details class="paper-abstract">
      Political bias in large language models (LLMs) is increasingly significant, but difficult to measure reproducibly across political and linguistic contexts. We introduce Polar, a 4,026-instance multiple-choice benchmark that measures political bias through option-level likelihoods rather than prompt-based generation. Polar covers two ideological axes and eight issue categories derived from the Manifesto Project, and evaluates models in parallel across U.S. and South Korean political contexts. Across 38 LLMs, measured bias varies systematically with political context, issue category, model group, and presentation language. All models lean left-progressive on U.S. political content, but show more centered and mixed patterns on South Korean content. Translation experiments further show that presentation language alone can shift measured bias. These findings highlight the need for multilingual and cross-contextual evaluation of political bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12900v1">Zero-source LLM Hallucination Detection with Human-like Criteria Probing</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often hallucinate by generating factually incorrect or unfaithful content, posing significant risks to their safe use. Detecting such hallucinations is particularly challenging under the zero-source constraint, where no model internals or external references are available, and detection must rely solely on the textual query-answer pair. In this paper, we propose Human-like Criteria Probing for Hallucination Detection (HCPD), a paradigm that emulates the multi-faceted reasoning of human evaluators. Its core is a Human-like Criteria Probing (HCP) mechanism, in which a LLM agent adaptively decomposes its judgment into a weighted set of interpretable criteria and aggregates criterion-specific scores into a final truthfulness measure. To achieve this adaptive capability, we introduce a reward-based alignment scheme using only weak supervision from semantic consistency. At inference, we employ a multi-sampling aggregation strategy to ensure robust decisions while preserving full interpretability. We further provide theoretical analysis supporting the reliability of our approach. Extensive experiments show that HCPD consistently outperforms state-of-the-art baselines, offering an effective and explainable solution for zero-source hallucination detection. Code is available at https://github.com/TRISKEL10N/HCPD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09379v3">LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at https://github.com/Lingxi-mental-health/LingxiDiagBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08794v3">One Token to Fool LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trusted as automated judges, assisting evaluation and providing reward signals for training other models, particularly in reference-based settings like Reinforcement Learning with Verifiable Rewards (RLVR). However, we uncover a critical vulnerability even in this reference-based paradigm: generative reward models are systematically susceptible to reward hacking. We find that superficial inputs, which we term ''master keys'' such as non-word symbols (e.g., '':'' or ''.'') or generic reasoning openers (e.g., ''Thought process:'' or ''Let's solve this problem step by step.''), can consistently elicit false positive rewards without any substantive reasoning. Our systematic evaluation demonstrates this is a widespread failure affecting a diverse range of models, including leading proprietary systems such as GPT-o1 and Claude-4. These results challenge the assumed robustness of LLM judges and pose a significant threat to their reliability. To address this, we propose a simple yet effective data augmentation strategy using truncated model outputs as adversarial negative examples. The resulting Master Reward Models (Master-RMs) demonstrate state-of-the-art robustness against these ''master key'' attacks while maintaining high performance in standard evaluation settings. We supplement these findings with a comprehensive analysis of the vulnerability across model scales, prompt variations, and common inference-time strategies, offering insights to guide future research on robust LLM evaluation. We release our robust, general-domain reward models and the synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12876v1">Multi-Bitwidth Quantization for LLMs Using Additive Codebooks</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 37 pages, 12 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed across heterogeneous hardware with varying resource constraints, the ability to adaptively manage the trade-off between performance and efficiency without retraining is critical. We propose Drop-by-Drop, a novel multi-bitwidth post-training quantization framework that enables inference-time precision control over LLM weights from a single trained model. Our method is theoretically grounded in information theory and successive refinement. We establish that LLM weights, which commonly follow a Gaussian distribution, can be optimally reconstructed with increasing fidelity as additional bits are incorporated, under a weighted mean squared error distortion motivated by LLM loss functions. To realize this in practice, Drop-by-Drop incorporates Matryoshka-style supervision into the loss function, exploiting the structure of additive codebooks. Drop-by-Drop produces a single model where ordered subsets of codebooks yield accurate partial reconstructions at each precision level. This approach significantly reduces storage and memory overhead by allowing a single checkpoint to serve multiple bitwidths, while maintaining competitive perplexity and accuracy across major architectures, such as Qwen, LLaMA, Gemma, and Mistral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16548v2">A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      The emergence of writable, cross-session persistent memory in LLM agents introduces a qualitatively different threat landscape from conventional input-centric security concerns, characterized by three properties: persistence, statefulness, and propagation. To systematically characterize this landscape, we propose a Memory Lifecycle Framework that organizes attacks, defenses, and their cross-phase dependencies along two axes: six lifecycle phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) and four security objectives (Integrity, Confidentiality, Availability, Governance). This analysis in turn exposes the need for formal security guarantees at the system level, motivating Verifiable Memory Governance(VMG), a framework of five architectural primitives that specifies what verifiable mechanisms a long-term-memory system must provide to maintain auditable, recoverable control over its memory state. Our analysis indicates that robust Long-Term Memory (LTM) security cannot be retrofitted at retrieval or execution time alone, but must be anchored in storage-time provenance, versioning, and policy-aware retention from the outset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12854v1">Small LLMs for Biomedical Claim Verification: Cost-Effective Fine-Tuning, Structural Dataset Shortcuts, and Cross-Domain Generalization</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 8 pages, 2 figures, 12 tables. To appear at BioNLP Workshop, ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models such as GPT-4o and GPT-5 achieve strong zero-shot performance on biomedical claim verification, but cost and opacity limit scalable use. We fine-tune three small LLMs: Phi-3-mini (3.8B), Qwen2.5-3B, and Mistral-7B, via QLoRA on SciFact and HealthVer, providing the first study of QLoRA models against GPT-4o and fine-tuned BioLinkBERT encoders. Mistral-7B QLoRA surpasses both GPT-4o and GPT-5 (up to 12% F1 gain) at a fractional cost using just 1,008 training examples. We conduct extensive in-domain and cross-domain evaluation: models trained on SciFact tested on HealthVer and vice versa, at matched sizes to isolate dataset structure from data quantity. We identify a previously unreported structural artifact in SciFact that inflates in-domain scores, and show through bidirectional out-of-domain evaluation that training on structurally sound data enables robust cross-domain transfer. We plan to release all code and adapter checkpoints.
    </details>
</div>
