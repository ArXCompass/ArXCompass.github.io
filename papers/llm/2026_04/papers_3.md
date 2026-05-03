# llm - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16416v2">Survey on Evaluation of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 ACL Findings
    </div>
    <details class="paper-abstract">
      LLM-based agents represent a paradigm shift in AI, enabling autonomous systems to plan, reason, and use tools while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methods for these increasingly capable agents. We analyze the field of agent evaluation across five perspectives: (1) Core LLM capabilities needed for agentic workflows, like planning, and tool use; (2) Application-specific benchmarks such as web and SWE agents; (3) Evaluation of generalist agents; (4) Analysis of agent benchmarks' core dimensions; and (5) Evaluation frameworks and tools for agent developers. Our analysis reveals current trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address, particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, scalable evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21829v1">Black-Box Skill Stealing Attack from Proprietary LLM Agents: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on skills to encapsulate reusable capabilities via progressively disclosed instructions. High-quality skills inject expert knowledge into general-purpose models, improving performance on specialized tasks. This quality and ease of dissemination drive the emergence of a skill economy: free skill marketplaces already report 90368 published skills, while paid marketplaces report more than 2000 listings and over $100,000 in creator earnings. Yet this growing marketplace also creates a new attack surface, as adversaries can interact with public agent to extract hidden proprietary skill content. We present the first empirical study of black-box skill stealing against LLM agent systems. To study this threat, we first derive an attack taxonomy from prior prompt-stealing methods and build an automated stealing prompt generation agent. This agent starts from model-generated seed prompts, expands them through scenario rationalization and structure injection, and enforces diversity via embedding filtering. This process yields a reproducible pipeline for evaluating agent systems. We evaluate such attacks across 3 commercial agent architectures and 5 LLMs. Our results show that agent skills can be extracted with only 3 interactions, posing a serious copyright risk. To mitigate this threat, we design defenses across three stages of the agent pipeline: input, inference, and output. Although these defenses achieve strong results, the attack remains inexpensive and readily automatable, allowing an adversary to launch repeated attempts with different variants; only one successful attempt is sufficient to compromise the protected skill. Overall, our findings suggest that these copyright risks are largely overlooked across proprietary agent ecosystems. We therefore advocate for more robust defense strategies that provide stronger protection guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08016v2">Wiring the 'Why': A Unified Taxonomy and Survey of Abductive Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Regardless of its foundational role in human discovery and sense-making, abductive reasoning--the inference of the most plausible explanation for an observation--has been relatively underexplored in Large Language Models (LLMs). Despite the rapid advancement of LLMs, the exploration of abductive reasoning and its diverse facets has thus far been disjointed rather than cohesive. This paper presents the first survey of abductive reasoning in LLMs, tracing its trajectory from philosophical foundations to contemporary AI implementations. To address the widespread conceptual confusion and disjointed task definitions prevalent in the field, we establish a unified two-stage definition that formally categorizes prior work. This definition disentangles abduction into Hypothesis Generation, where models bridge epistemic gaps to produce candidate explanations, and Hypothesis Selection, where the generated candidates are evaluated and the most plausible explanation is chosen. Building upon this foundation, we present a comprehensive taxonomy of the literature, categorizing prior work based on their abductive tasks, datasets, underlying methodologies, and evaluation strategies. In order to ground our framework empirically, we conduct a compact benchmark study of current LLMs on abductive tasks, together with targeted comparative analyses across model sizes, model families, evaluation styles, and the distinct generation-versus-selection task typologies. Moreover, by synthesizing recent empirical results, we examine how LLM performance on abductive reasoning relates to deductive and inductive tasks, providing insights into their broader reasoning capabilities. Our analysis reveals critical gaps in current approaches--from static benchmark design and narrow domain coverage to narrow training frameworks and limited mechanistic understanding of abductive processes...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00413v5">Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Agentic large language model (LLM) training often involves multi-turn interaction trajectories that branch into multiple execution paths due to concurrent tool use, think-mode, sub-agent, context management and other runtime designs. As a result, the tokens produced by a single task naturally form a tree-structured token trajectory with shared prefixes, rather than a linear sequence. Existing training pipelines linearize such trajectories and treat each branch independently, leading to substantial redundant computation in both forward and backward passes. We derive that averaging the loss over all branches independently is algebraically identical to a per-token weighted loss, where each token's weight equals the fraction of branches passing through it. The problem therefore reduces to computing the log-probability of every token in the prefix tree exactly once, with no repeated computation across shared prefixes: we propose DFS serialization of the tree, which visits every token exactly once, and adapt full-attention and SSM layers to ensure the resulting log-probabilities match independent per-branch calculation exactly. In practice, a single trajectory tree can be too large to fit in GPU memory; we therefore propose Redundancy-Free Tree Partitioning, which handles memory-constrained settings with zero redundant computation and peak memory bounded by a single root-to-leaf path. Together, these contributions form Tree Training, an efficient framework for training LLMs on tree-structured trajectories, achieving up to 6.2x end-to-end training speedup on dense and MoE models for both supervised fine-tuning and reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21361v2">Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 20 pages, 4 charts. AAIML (2026)
    </div>
    <details class="paper-abstract">
      Large language model (LLM) providers boast big numbers for maximum context window sizes. To test the real world use of context windows, we 1) define a concept of maximum effective context window, 2) formulate a testing method of a context window's effectiveness over various sizes and problem types, and 3) create a standardized way to compare model efficacy for increasingly larger context window sizes to find the point of failure. We collected hundreds of thousands of data points across several models and found significant differences between reported Maximum Context Window (MCW) size and Maximum Effective Context Window (MECW) size. Our findings show that the MECW is, not only, drastically different from the MCW but also shifts based on the problem type. A few top of the line models in our test group failed with as little as 100 tokens in context; most had severe degradation in accuracy by 1000 tokens in context. All models fell far short of their Maximum Context Window by as much as 99 percent. Our data reveals the Maximum Effective Context Window shifts based on the type of problem provided, offering clear and actionable insights into how to improve model accuracy and decrease model hallucination rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21769v1">Who Defines "Best"? Towards Interactive, User-Defined Evaluation of LLM Leaderboards</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accepted to the 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2026)
    </div>
    <details class="paper-abstract">
      LLM leaderboards are widely used to compare models and guide deployment decisions. However, leaderboard rankings are shaped by evaluation priorities set by benchmark designers, rather than by the diverse goals and constraints of actual users and organizations. A single aggregate score often obscures how models behave across different prompt types and compositions. In this work, we conduct an in-depth analysis of the dataset used in the LMArena (formerly Chatbot Arena) benchmark and investigate this evaluation challenge by designing an interactive visualization interface as a design probe. Our analysis reveals that the dataset is heavily skewed toward certain topics, that model rankings vary across prompt slices, and that preference-based judgments are used in ways that blur their intended scope. Building on this analysis, we introduce a visualization interface that allows users to define their own evaluation priorities by selecting and weighting prompt slices and to explore how rankings change accordingly. A qualitative study suggests that this interactive approach improves transparency and supports more context-specific model evaluation, pointing toward alternative ways to design and use LLM leaderboards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18740v2">Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Automated Code Review (ACR) systems integrating Large Language Models (LLMs) are increasingly adopted in software development workflows, ranging from interactive assistants to autonomous agents in CI/CD pipelines. In this paper, we study how LLM-based vulnerability detection in ACR is affected by the framing effect: the tendency to let the presentation of information override its semantic content in forming judgments. We examine whether adversaries can exploit this through contextual-bias injection: crafting PR metadata to bias ACR security judgments as a supply-chain attack vector against real-world ACR pipelines. To this end, we first conduct a large-scale exploratory study across 6 LLMs under five framing conditions, establishing the framing effect as a systematic and widespread phenomenon in LLM-based vulnerability detection, with bug-free framing producing the strongest effect. We then design a realistic and controlled experimental environment, evaluating 17 CVEs across 10 real-world projects, to assess the susceptibility of real-world ACR pipelines to vulnerability reintroduction attacks. We employ two attack strategies: a template-based attack inspired by prior related work, and a novel LLM-assisted iterative refinement attack. We find that template-based attacks are ineffective and may even backfire, as direct biasing attempts raise suspicions. Our iterative refinement attack, on the other hand, achieves 100% success, exploiting a fundamental asymmetry: attackers can iteratively refine attacks against a local clone of the review pipeline, while defenders have only one chance to detect them. Debiasing via metadata redaction and explicit instructions restores detection in all affected cases. Overall, our findings highlight the dangers of over-relying on ACR and stress the importance of human oversight and contributor trust in the development process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21751v1">Why are all LLMs Obsessed with Japanese Culture? On the Hidden Cultural and Regional Biases of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      LLMs have been showing limitations when it comes to cultural coverage and competence, and in some cases show regional biases such as amplifying Western and Anglocentric viewpoints. While there have been works analysing the cultural capabilities of LLMs, there has not been specific work on highlighting LLM regional preferences when it comes to cultural-related questions. In this work, we propose a new dataset based on a comprehensive taxonomy of Culture-Related Open Questions (CROQ). The results show that, contrary to previous cultural bias work, LLMs show a clear tendency towards countries such as Japan. Moveover, our results show that when prompting in languages such as English or other high-resource ones, LLMs tend to provide more diverse outputs and show less inclinations towards answering questions highlighting countries for which the input language is an official language. Finally, we also investigate at which point of LLM training this cultural bias emerges, with our results suggesting that the first clear signs appear after supervised fine-tuning, and not during pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21746v1">Less Is More: Measuring How LLM Involvement affects Chatbot Accuracy in Static Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to make static analysis tools accessible through natural language, yet existing systems differ in how much they delegate to the LLM without treating the degree of delegation as an independent variable. We compare three architectures along a spectrum of LLM involvement for translating natural language to Joern's query language \cpgql{}: direct query generation (\approach{1}), generation of a schema-constrained JSON intermediate representation (\approach{2}), and tool-augmented agentic generation (\approach{3}). These are evaluated on a benchmark of 20 code analysis tasks across three complexity tiers, using four open-weight models in a 2\(\times\)2 design (two model families \(\times\) two scales), each with three repetitions. The structured intermediate representation (\approach{2}) achieves the highest result match rates, outperforming direct generation by 15--25 percentage points on large models and surpassing the agentic approach despite the latter consuming 8\(\times\) more tokens. The benefit of structured intermediates is most pronounced for large models; for small models, schema compliance becomes the bottleneck. These findings suggest that in formally structured domains, constraining the LLM's output to a well-typed intermediate representation and delegating query construction to deterministic code yields better results than either unconstrained generation or iterative tool use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05846v2">AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at https://github.com/sunyuanfu/AgentGL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13918v2">Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Published as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Process reward models (PRMs) are a cornerstone of test-time scaling (TTS), designed to verify and select the best responses from large language models (LLMs). However, this promise is challenged by recent benchmarks where simple majority voting, which ignores PRM signals, occasionally outperforms standard PRM-based selection. This raises a critical question: How can we effectively utilize verification signals from PRMs for TTS? To address this, we start by developing a theoretical framework for optimally combining signals from both the LLM and the PRM. Our framework reveals that the optimal strategy is a weighted aggregation of responses, a strategy whose effectiveness hinges on estimating weights that capture the complex interplay between the models. Based on our theoretical results, we empirically show that these optimal weighting functions differ significantly across LLM-PRM pairs and, notably, often assign substantial negative weights. Motivated by these insights, we propose efficient pre-computation methods to calibrate these weighting functions. Extensive experiments across 5 LLMs and 7 PRMs demonstrate that our calibration method significantly boosts the TTS efficiency, surpassing the performance of vanilla weighted majority voting while using only $21.3\%$ of the computation. Ultimately, our work demonstrates that investing in a more intelligent aggregation strategy can be a more convincing path to performance gains than simply scaling test-time computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09998v2">Flipping Against All Odds: Reducing LLM Coin Flip Bias via Verbalized Rejection Sampling</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Technical Report v2 (27 pages, 14 figures)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often accurately describe probability distributions using natural language, yet they still struggle to generate faithful samples from them. This mismatch limits their use in tasks requiring reliable stochasticity, such as Monte Carlo methods, agent-based simulations, and randomized decision-making. We investigate this gap between knowledge and sampling in the context of Bernoulli distributions. We introduce Verbalized Rejection Sampling (VRS), a natural-language adaptation of classical rejection sampling that prompts the LLM to reason about and accept or reject proposed samples. Despite relying on the same Bernoulli mechanism internally, VRS substantially reduces sampling bias across models. We provide theoretical analysis showing that, under mild assumptions, VRS improves over direct sampling, with gains attributable to both the algorithm and prompt design. More broadly, our results show how classical probabilistic tools can be verbalized and embedded into LLM workflows to improve reliability, without requiring access to model internals or heavy prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21700v1">Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      The growing application of large language models (LLMs) in safety-critical domains has raised urgent concerns about their security. Many recent studies have demonstrated the feasibility of backdoor attacks against LLMs. However, existing methods suffer from three key shortcomings: explicit trigger patterns that compromise naturalness, unreliable injection of attacker-specified payloads in long-form generation, and incompletely specified threat models that obscure how backdoors are delivered and activated in practice. To address these gaps, we present BadStyle, a complete backdoor attack framework and pipeline. BadStyle leverages an LLM as a poisoned sample generator to construct natural and stealthy poisoned samples that carry imperceptible style-level triggers while preserving semantics and fluency. To stabilize payload injection during fine-tuning, we design an auxiliary target loss that reinforces the attacker-specified target content in responses to poisoned inputs and penalizes its emergence in benign responses. We further ground the attack in a realistic threat model and systematically evaluate BadStyle under both prompt-induced and PEFT-based injection strategies. Extensive experiments across seven victim LLMs, including LLaMA, Phi, DeepSeek, and GPT series, demonstrate that BadStyle achieves high attack success rates (ASRs) while maintaining strong stealthiness. The proposed auxiliary target loss substantially improves the stability of backdoor activation, yielding an average ASR improvement of around 30% across style-level triggers. Even in downstream deployment scenarios unknown during injection, the implanted backdoor remains effective. Moreover, BadStyle consistently evades representative input-level defenses and bypasses output-level defenses through simple camouflage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21679v1">A Sociotechnical, Practitioner-Centered Approach to Technology Adoption in Cybersecurity Operations: An LLM Case</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 16 Pages and 6 figures (5 diagrams, 1 table)
    </div>
    <details class="paper-abstract">
      Technology for security operations centers (SOCs) has a storied history of slow adoption due to concerns about trust and reliability. These concerns are amplified with artificial intelligence, particularly large language models (LLMs), which exhibit issues such as hallucinations and inconsistent outputs. To assess whether LLM-based tools can improve SOC efficiency, we embedded two PhD researchers within a multinational company SOC for six months of ethnographic fieldwork. We identified recurring challenges, such as repetitive tasks, fragmented/unclear data, and tooling bottlenecks, and collaborated directly with practitioners to develop LLM companion tools aligned with their operational needs. Iterative refinement reduced workflow disruption and improved interpretability, leading from skepticism to sustained adoption. Ethnographic analysis indicates that this shift was enabled by our sociotechnical co-creation process consistent with Nonaka's SECI model. This framework explains the common challenges in traditional SOC technology adoption, including workflow misalignment, rigidity against evolving threats and internal requirements, and stagnation over time. Our findings show that the co-creation approach can overcome these old barriers and create a new paradigm for creating usable technology for cybersecurity operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14709v2">HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21598v1">DryRUN: On the Role of Public Tests in LLM-Driven Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-agent frameworks are widely used in autonomous code generation and have applications in complex algorithmic problem-solving. Recent work has addressed the challenge of generating functionally correct code by incorporating simulation-driven planning and debugging, where language models trace execution steps to verify logic. However, these approaches depend on human-provided public test cases to ground the debugging and simulation loop. Manually authoring comprehensive input-output examples is a labor-intensive bottleneck in the software development lifecycle. Because ground-truth input-output examples are rarely available prior to implementation in real-world software engineering, this dependency restricts methods to curated competitive programming benchmarks. Furthermore, we identify that reliance on these public tests induces an ``overconfidence gap,'' causing frameworks to overfit to simplistic examples and fail on hidden evaluations. In contrast, we observe that external sample inputs are not strictly necessary for code generation. We demonstrate that large language models can autonomously generate valid inputs and simulate execution traces to self-correct. Consequently, we develop DryRUN, a framework that eliminates the need for ground-truth samples by allowing the LLM to iteratively plan, autonomously generate its own inputs and simulate execution, mitigating algorithmic overconfidence. Evaluations on the LiveCodeBench v6 dataset (post-March 2025) demonstrate that DryRUN matches performance against CodeSIM, a state-of-the-art and public-test-dependent framework, while operating entirely without public test cases or external execution feedback while reducing output token consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21584v1">CoFEE: Reasoning Control for LLM-Based Feature Discovery</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Feature discovery from complex unstructured data is fundamentally a reasoning problem: it requires identifying abstractions that are predictive of a target outcome while avoiding leakage, proxies, and post-outcome signals. With the introduction of ever-improving Large Language Models (LLMs), our method provides a structured method for addressing this challenge. LLMs are well suited for this task by being able to process large amounts of information, but unconstrained feature generation can lead to weak features. In this work, we study reasoning control in LLMs by inducing cognitive behaviors for improving feature discovery. We introduce CoFEE (Cognitive Feature Engineering Engine), a reasoning control framework that enforces cognitive behaviors in how the LLM reasons during feature discovery. From a machine learning perspective, these cognitive behaviors act as structured inductive biases over the space of candidate features generated by the model. These behaviors have been exploited with success in ML models, and include backward chaining from outcomes, subgoal decomposition, verification against observability and leakage criteria, and explicit backtracking of rejected reasoning paths. In a controlled comparison, we show that enforcing cognitive behaviors yields features with higher empirical predictability than those under unconstrained vanilla LLM prompts. CoFEE achieves an average Success Rate Score that is 15.2% higher than the vanilla approach, while generating 29% fewer features and reducing costs by 53.3%. Using held-out feature evaluation, we assess whether cognitively induced features generalize beyond the data used for discovery. Our results indicate that, in our evaluated setting, reasoning control is associated with improvements in quality and efficiency of LLM-based feature discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21275v3">InfiniPipe: Elastic Pipeline Parallelism for Efficient Variable-Length Long-Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Long context training is crucial for LLM's context extension. Existing schemes, such as sequence parallelism, incur substantial communication overhead. Pipeline parallelism (PP) reduces this cost, but its effectiveness hinges on partitioning granularity. Batch-level PP employing sequence packing exhibits high memory consumption in long-context scenarios, whereas token-level PP splitting sequences into slices alleviates memory overhead but may incur hardware under-utilization. Moreover, the skewed distribution of sequence length in real-world datasets renders monolithic and static granularity PP's sub-optimal performance. In this paper, we propose 1) \textit{Elastic Pipeline Parallelism} (EPP) that orchestrates token-level PP and batch-level PP to adapt to resource and workload heterogeneity, and 2) \textit{Stage-Aware Chunk-Level Adaptive Checkpointing} that efficiently integrates gradient checkpointing with EPP. Comprehensive experiments demonstrate that InfiniPipe achieves a 1.69x speedup over state-of-the-art systems. Our code is open-sourced at https://github.com/wsjdsg/InfiniPipe.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21579v1">A Metamorphic Testing Approach to Diagnosing Memorization in LLM-Based Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      LLM-based automated program repair (APR) techniques have shown promising results in reducing debugging costs. However, prior results can be affected by data leakage: large language models (LLMs) may memorize bug fixes when evaluation benchmarks overlap with their pretraining data, leading to inflated performance estimates. In this paper, we investigate whether we can better reveal data leakage by combining metamorphic testing (MT) with negative log-likelihood (NLL), which has been used in prior work as a proxy for memorization. We construct variant benchmarks by applying semantics-preserving transformations to two widely used datasets, Defects4J and GitBug-Java. Using these benchmarks, we evaluate the repair success rates of seven LLMs on both original and transformed versions, and analyze the relationship between performance degradation and NLL. Our results show that all evaluated state-of-the-art LLMs exhibit substantial drops in patch generation success rates on transformed benchmarks, ranging from -4.1% for GPT-4o to -15.98% for Llama-3.1. Furthermore, we find that this degradation strongly correlates with NLL on the original benchmarks, suggesting that models perform better on instances they are more likely to have memorized. These findings show that combining MT with NLL provides stronger and more reliable evidence of data leakage, while metamorphic testing alone can help mitigate its effects in LLM-based APR evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21570v1">SpecSyn: LLM-based Synthesis and Refinement of Formal Specifications for Real-world Program Verification</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Program verification is a formal technique to rigorously ensure the correctness and fault-freeness of software systems. However, constructing comprehensive interprocedural specifications for full verification obligations is time-consuming and labor-intensive, giving rise to automated specification generation approaches. Despite the significant advancements in these approaches brought by Large Language Models (LLMs), existing LLM-empowered approaches still suffer from significant limitations: they lack effective strategies for handling sizable input programs, and are typically equipped with no mechanisms to evaluate and guarantee the strength of the generated specifications. The limitations impair their ability to extract precise specifications from real-world complicated programs to support the verification of target properties, thereby hindering the applicability of existing approaches in verification tasks on real-world programs. To remedy this gap, we propose SpecSyn, a novel LLM-based specification generation method. SpecSyn first decomposes the input program into individual segments, which are handled respectively by the subsequent iterative specification generation process. Innovatively, we incorporate into the process a specification refinement mechanism based on semantic-non-equivalent program mutations and variant discrimination, assessing and enhancing the semantic strength of the generated specifications. Extensive experiments show that SpecSyn maintains high precision over 90% and outstanding recall over 75%, significantly outperforming existing LLM-based approaches. In further evaluations, SpecSyn successfully handles 1071 out of 1365 target properties for open-source programs, proving its applicability on real-world program verification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21564v1">Measuring Opinion Bias and Sycophancy via LLM-based Coercion</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models increasingly shape the information people consume: they are embedded in search, consulted for professional advice, deployed as agents, and used as a first stop for questions about policy, ethics, health, and politics. When such a model silently holds a position on a contested topic, that position propagates at scale into users' decisions. Eliciting a model's positions is harder than it first appears: contemporary assistants answer direct opinion questions with evasive disclaimers, and the same model may concede the opposite position once the user starts arguing one side. We propose a method, released as the open-source llm-bias-bench, for discovering the opinions an LLM actually holds on contested topics under conditions that resemble real multi-turn interaction. The method pairs two complementary free-form probes. Direct probing asks for the model's opinion across five turns of escalating pressure from a simulated user. Indirect probing never asks for an opinion and engages the model in argumentative debate, letting bias leak through how it concedes, resists, or counter-argues. Three user personas (neutral, agree, disagree) collapse into a nine-way behavioral classification that separates persona-independent positions from persona-dependent sycophancy, and an auditable LLM judge produces verdicts with textual evidence. The first instantiation ships 38 topics in Brazilian Portuguese across values, scientific consensus, philosophy, and economic policy. Applied to 13 assistants, the method surfaces findings of practical interest: argumentative debate triggers sycophancy 2-3x more than direct questioning (median 50% to 79%); models that look opinionated under direct questioning often collapse into mirroring under sustained arguments; and attacker capability matters mainly when an existing opinion must be dislodged, not when the assistant starts neutral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21549v1">Unbiased Prevalence Estimation with Multicalibrated LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Estimating the prevalence of a category in a population using imperfect measurement devices (diagnostic tests, classifiers, or large language models) is fundamental to science, public health, and online trust and safety. Standard approaches correct for known device error rates but assume these rates remain stable across populations. We show this assumption fails under covariate shift and that multicalibration, which enforces calibration conditional on the input features rather than just on average, is sufficient for unbiased prevalence estimation under such shift. Standard calibration and quantification methods fail to provide this guarantee. Our work connects recent theoretical work on fairness to a longstanding measurement problem spanning nearly all academic disciplines. A simulation confirms that standard methods exhibit bias growing with shift magnitude, while a multicalibrated estimator maintains near-zero bias. While we focus the discussion mostly on LLMs, our theoretical results apply to any classification model. Two empirical applications -- estimating employment prevalence across U.S. states using the American Community Survey, and classifying political texts across four countries using an LLM -- demonstrate that multicalibration substantially reduces bias in practice, while highlighting that calibration data should cover the key feature dimensions along which target populations may differ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21536v1">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accepted to ECIR 2026. 7 pages. This version of the contribution has been accepted for publication, after peer review but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The Version of Record is available online at: http://dx.doi.org/10.1007/978-3-032-21300-6_42
    </div>
    <details class="paper-abstract">
      Sequential recommender systems have achieved significant success in modeling temporal user behavior but remain limited in capturing rich user semantics beyond interaction patterns. Large Language Models (LLMs) present opportunities to enhance user understanding with their reasoning capabilities, yet existing integration approaches create prohibitive inference costs in real time. To address these limitations, we present a novel knowledge distillation method that utilizes textual user profile generated by pre-trained LLMs into sequential recommenders without requiring LLM inference at serving time. The resulting approach maintains the inference efficiency of traditional sequential models while requiring neither architectural modifications nor LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21525v1">Job Skill Extraction via LLM-Centric Multi-Module Framework</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 5 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Span-level skill extraction from job advertisements underpins candidate-job matching and labor-market analytics, yet generative large language models (LLMs) often yield malformed spans, boundary drift, and hallucinations, especially with long-tail terms and cross-domain shift. We present SRICL, an LLM-centric framework that combines semantic retrieval (SR), in-context learning (ICL), and supervised fine-tuning (SFT) with a deterministic verifier. SR pulls in-domain annotated sentences and definitions from ESCO to form format-constrained prompts that stabilize boundaries and handle coordination. SFT aligns output behavior, while the verifier enforces pairing, non-overlap, and BIO legality with minimal retries. On six public span-labeled corpora of job-ad sentences across sectors and languages, SRICL achieves substantial STRICT-F1 improvements over GPT-3.5 prompting baselines and sharply reduces invalid tags and hallucinated spans, enabling dependable sentence-level deployment in low-resource, multi-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21505v1">Assessing the Impact of Requirement Ambiguity on LLM-based Function-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Software requirement ambiguity is ubiquitous in real-world development, stemming from the inherent imprecision of natural language and the varying interpretations of stakeholders. While Large Language Models (LLMs) have demonstrated impressive capabilities in generating code from precise specifications, such ambiguity poses a significant obstacle to reliable automated code generation. Existing benchmarks typically assume clear and unambiguous requirements, leaving an empirical gap in understanding how LLMs behave when faced with the inherent uncertainty of real-world software requirements. In this paper, we introduce Orchid, the first code generation benchmark specifically designed with ambiguous requirements. It comprises 1,304 function-level tasks covering four distinct types of ambiguity: lexical, syntactic, semantic, and vagueness. Leveraging this dataset, we conduct the first systematic empirical study to evaluate the impact of requirement ambiguity on LLM-based code generation. Our results demonstrate that ambiguity consistently degrades the performance of all evaluated LLMs, with the most pronounced negative effects observed in highly advanced models. Furthermore, we observe that LLMs frequently produce functionally divergent implementations for the same ambiguous requirement and lack the capability to identify or resolve such ambiguity autonomously. These findings reveal a significant performance gap between clear and ambiguous requirements, underscoring the urgent need for ambiguity-aware techniques in the next generation of automated software engineering tools. The Orchid benchmark is publicly available at https://huggingface.co/datasets/SII-YDD/Orchid.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20487v2">Knowledge Capsules: Structured Nonparametric Memory Units for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode knowledge in parametric weights, making it costly to update or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by appending retrieved text to the input, but operates purely through context expansion, where external knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and often unstable, particularly in long context and multi hop reasoning scenarios. We propose Knowledge Capsules, structured nonparametric memory units that represent normalized relational knowledge and can be constructed directly from document corpora using a frozen base model. Instead of injecting knowledge as text, we introduce an External Key Value Injection (KVI) framework that compiles capsules into attention-compatible key value representations, enabling external knowledge to directly participate in the model's attention computation. By shifting knowledge integration from context-level augmentation to memory level interaction, the proposed framework consistently outperforms RAG and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long context and multi hop reasoning, while requiring no parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21479v1">Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong reasoning capabilities and attracted increasing research attention in the field of autonomous driving (AD). However, safe application of LLMs on AD perception and prediction still requires a thorough understanding of both the dynamic traffic agents and the static road infrastructure. To this end, this study introduces a framework to evaluate the capability of LLMs in understanding the behaviors of dynamic traffic agents and the topology of road networks. The framework leverages frozen LLMs as the reasoning engine, employing a traffic encoder to extract spatial-level scene features from observed trajectories of agents, while a lightweight Convolutional Neural Network (CNN) encodes the local high-definition (HD) maps. To assess the intrinsic reasoning ability of LLMs, the extracted scene features are then transformed into LLM-compatible tokens via a reprogramming adapter. By residing the prediction burden with the LLMs, a simpler linear decoder is applied to output future trajectories. The framework enables a quantitative analysis of the influence of multi-modal information, especially the impact of map semantics on trajectory prediction accuracy, and allows seamless integration of frozen LLMs with minimal adaptation, thereby demonstrating strong generalizability across diverse LLM architectures and providing a unified platform for model evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16737v2">Secure LLM Fine-Tuning via Safety-Aware Probing</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across many applications, but their ability to generate harmful content raises serious safety concerns. Although safety alignment techniques are often applied during pre-training or post-training, recent studies show that subsequent fine-tuning on adversarial or even benign data can still compromise model safety. In this paper, we revisit the fundamental question of why fine-tuning on non-harmful data may nevertheless degrade safety. We show that the safety and task-performance loss landscapes are partially decoupled, so updates that improve task-specific performance may still move the model toward unsafe regions. Based on this insight, we propose a safety-aware probing (SAP) optimization framework for mitigating safety risks during fine-tuning. Concretely, SAP uses contrastive safety signals to locate safety-correlated directions, and optimizes a lightweight probe that perturbs hidden-state propagation during fine-tuning, thereby steering parameter updates away from harmful trajectories while preserving task-specific learning. Extensive experiments show that SAP consistently improves the safety--utility tradeoff across multiple models and tasks. Averaged over multiple LLMs, SAP reduces the harmful score significantly relative to standard fine-tuning, outperforming strong baselines while maintaining competitive task-specific performance. SAP also demonstrates stronger robustness under harmful data poisoning, adversarial fine-tuning, and a dedicated post-fine-tuning adaptive attack, validating that SAP is an effective and scalable framework for preserving LLM safety during fine-tuning. Our code is available at https://github.com/ChengcanWu/SAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20726v2">Exploiting LLM-as-a-Judge Disposition on Free Text Legal QA via Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accepted at the 21st International Conference on Artificial Intelligence and Law (ICAIL 2026), Singapore, June 8-12, 2026. 10 pages, 14 figures, 2 tables
    </div>
    <details class="paper-abstract">
      This work explores the role of prompt design and judge selection in LLM-as-a-Judge evaluations of free text legal question answering. We examine whether automatic task prompt optimization improves over human-centered design, whether optimization effectiveness varies by judge feedback style, and whether optimized prompts transfer across judges. We systematically address these questions on the LEXam benchmark by optimizing task prompts using the ProTeGi method with feedback from two judges (Qwen3-32B, DeepSeek-V3) across four task models, and then testing cross-judge transfer. Automatic optimization consistently outperforms the baseline, with lenient judge feedback yielding higher and more consistent gains than strict judge feedback. Prompts optimized with lenient feedback transfer better to strict judges than the reverse direction. Analysis reveals that lenient judges provide permissive feedback, yielding prompts with broader applicability, whereas strict judges produce restrictive feedback, leading to judge-specific overfitting. Our findings demonstrate algorithmically optimizing prompts on training data can outperform human-centered prompt design and that judges' dispositions during optimization shape prompt generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.04023v3">Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accepted to ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance on complex mathematical benchmarks yet sometimes fail on basic math reasoning while generating unnecessarily verbose responses. In this paper, we present LLMThinkBench, a systematic benchmark and comprehensive empirical study to evaluate the efficiency of reasoning in LLMs, focusing on the fundamental tradeoff between accuracy and overthinking. First, we formalize the accuracy-verbosity tradeoff. Second, we introduce the Overthinking Score, a harmonic-mean metric combining accuracy and token-efficiency for holistic model evaluation. Third, we establish an evaluation protocol with dynamically-generated data across 14 basic math tasks. Fourth, we conduct a large-scale empirical study evaluating 53 LLMs, including reasoning and quantized variants across different reasoning budgets. Fifth, we release LLMThinkBench as an open-source Python package and public leaderboard for reproducibility. Our findings reveal: 1) model performance on complex benchmarks does not translate directly to basic math reasoning; 2) reasoning models generate ~18x more tokens while sometimes achieving lower accuracy and exhibit catastrophic collapse when tokens are constrained, dropping by up to ~36%; 3) the accuracy-verbosity relationship is non-monotonic with extended reasoning budgets yielding diminishing returns (GPT-5/o-series models show zero accuracy gain from low -> medium -> high reasoning effort). Our findings challenge the assumption that longer reasoning in LLMs necessarily improves mathematical reasoning. Our public leaderboard is available at https://ctrl-gaurav.github.io/LLMThinkBench/. Our open-source Python package is available at https://pypi.org/project/llmthinkbench/, and the codebase can be found at https://github.com/ctrl-gaurav/LLMThinkBench for easy and reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21399v1">A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 7 pages, 4 figures, conference version
    </div>
    <details class="paper-abstract">
      AI WiFi offload is emerging as a promising approach for providing large language model (LLM) services to resource-constrained wireless devices. However, unlike conventional edge computing, LLM inference over WiFi must jointly address heterogeneous model capabilities, wireless contention, uncertain task complexity, and semantic correlation among reasoning tasks. In this paper, we investigate LLM inference offloading in a multi-user multi-edge WiFi network, where each task can be executed locally, directly offloaded to a nearby edge access point (AP), or decomposed into multiple subtasks for collaborative execution across local and edge nodes. To this end, we propose a user-edge collaborative framework with an LLM-based planner that not only performs task decomposition but also infers subtask difficulty and expected output token length, enabling more accurate estimation of execution quality and latency on heterogeneous nodes. Based on these estimates, we further design a decomposition-aware scheduling strategy that jointly optimizes subtask assignment, execution, and aggregation under communication, queuing, and computation constraints. Simulation results show that the proposed framework achieves a better latency-accuracy tradeoff than local-only and nearest-edge baselines, reducing the average latency by $20\%$ and improving the overall reward by $80\%$. Moreover, the distilled lightweight planner approaches the performance of the large teacher model while remaining more suitable for practical edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21377v1">A Replicable Robotics Awareness Method Using LLM-Enabled Robotics Interaction: Evidence from a Corporate Challenge</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 10 pages, 8 Figures, to be submitted for journal per-review
    </div>
    <details class="paper-abstract">
      Large language models are increasingly being explored as interfaces between humans and robotic systems, yet there remains limited evidence on how such technologies can be used not only for interaction, but also as a structured means of introducing robotics to non-specialist users in real organizational settings. This paper introduces and evaluates a challenge-based method for robotics awareness, implemented through an LLM-enabled humanoid robot activity conducted with employees of AD Ports Group in the United Arab Emirates. In the event, participants engaged with a humanoid robot in a logistics-inspired task environment using voice commands interpreted through an LLM-based control framework. The activity was designed as a team-based, role-driven experience intended to expose participants to embodied AI and human-robot collaboration without requiring prior robotics expertise. To evaluate the approach, a post-event survey remained open for 16 days and collected 102 responses. Results indicate strong overall reception, with high satisfaction (8.46/10), increased interest in robotics and AI (4.47/5), and improved understanding of emerging forms of human-robot collaboration (4.45/5). Participants who interacted directly with the robot also reported natural interaction (4.37/5) and a strong sense that interaction became easier as the activity progressed (4.74/5). At the same time, lower ratings for reliability and predictability point to important technical and design challenges for future iterations. The findings suggest that challenge-based, LLM-enabled humanoid interaction can serve as a promising and replicable method for robotics awareness in industrial and operational environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07101v3">Grounding Machine Creativity in Game Design Knowledge Representations: Empirical Probing of LLM-Based Executable Synthesis of Goal Playable Patterns under Structural Constraints</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Creatively translating complex gameplay ideas into executable artifacts (e.g., games as Unity projects and code) remains a central challenge in computational game creativity. Gameplay design patterns provide a structured representation for describing gameplay phenomena, enabling designers to decompose high-level ideas into entities, constraints, and rule-driven dynamics. Among them, goal patterns formalize common player-objective relationships. Goal Playable Concepts (GPCs) operationalize these abstractions as playable Unity engine implementations, supporting experiential exploration and compositional gameplay design. We frame scalable playable pattern realization as a problem of constrained executable creative synthesis: generated artifacts must satisfy Unity's syntactic and architectural requirements while preserving the semantic gameplay meanings encoded in goal patterns. This dual constraint limits scalability. Therefore, we investigate whether contemporary large language models (LLMs) can perform such synthesis under engine-level structural constraints and generate Unity code (as games) structured and conditioned by goal playable patterns. Using 26 goal pattern instantiations, we compare a direct generation baseline (natural language -> C# -> Unity) with pipelines conditioned on a human-authored Unity-specific intermediate representation (IR), across three IR configurations and two open-source models (DeepSeek-Coder-V2-Lite-Instruct and Qwen2.5-Coder-7B-Instruct). Compilation success is evaluated via automated Unity replay. We propose grounding and hygiene failure modes, identifying structural and project-level grounding as primary bottlenecks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21365v1">mcdok at SemEval-2026 Task 13: Finetuning LLMs for Detection of Machine-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Multi-domain detection of the machine-generated code snippets in various programming languages is a challenging task. SemEval-2026 Task~13 copes with this challenge in various angles, as a binary detection problem as well as attribution of the source. Specifically, its subtasks also cover generator LLM family detection, as well as a hybrid code co-generated by humans and machines, or adversarially modified codes hiding its origin. Our submitted systems adjusted the existing mdok approach (focused on machine-generated text detection) to these specific kinds of problems by exploring various base models, more suitable for code understanding. The results indicate that the submitted systems are competitive in all three subtasks. However, the margins from the top-performing systems are significant, and thus further improvements are possible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21334v1">Ideological Bias in LLMs' Economic Causal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 18 of 20 models, accuracy is systematically higher when the empirically verified causal sign aligns with intervention-oriented expectations than with market-oriented ones. Moreover, when models err, their incorrect predictions disproportionately lean intervention-oriented, and this directional skew is not eliminated by one-shot in-context prompting. These results highlight that LLMs are not only less accurate on ideologically contested economic questions, but systematically less reliable in one ideological direction than the other, underscoring the need for direction-aware evaluation in high-stakes economic and policy settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19192v2">Empowering NPC Dialogue with Environmental Context Using LLMs and Panoramic Images</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      We present an approach for enhancing non-playable characters (NPCs) in games by combining large language models (LLMs) with computer vision to provide contextual awareness of their surroundings. Conventional NPCs typically rely on pre-scripted dialogue and lack spatial understanding, which limits their responsiveness to player actions and reduces overall immersion. Our method addresses these limitations by capturing panoramic images of an NPC's environment and applying semantic segmentation to identify objects and their spatial positions. The extracted information is used to generate a structured JSON representation of the environment, combining object locations derived from segmentation with additional scene graph data within the NPC's bounding sphere, encoded as directional vectors. This representation is provided as input to the LLM, enabling NPCs to incorporate spatial knowledge into player interactions. As a result, NPCs can dynamically reference nearby objects, landmarks, and environmental features, leading to more believable and engaging gameplay. We describe the technical implementation of the system and evaluate it in two stages. First, an expert interview was conducted to gather feedback and identify areas for improvement. After integrating these refinements, a user study was performed, showing that participants preferred the context-aware NPCs over a non-context-aware baseline, confirming the effectiveness of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21316v1">LLM-Steered Power Allocation for Parallel QPSK-AWGN Channels</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being explored as high-level decision modules in closed-loop systems, but their stochastic nature makes safe integration challenging. In this paper, we propose LLM-Steered Power Allocation, a dual-process architecture for parallel QPSK channels inspired by Kahneman's System 1/System 2 framework. A fast numerical optimizer (System 1) continuously performs projected gradient ascent on a weighted mutual-information objective, while an LLM navigator (System 2) periodically interprets natural-language policies and updates only the channel weights and the operational power budget. The LLM never manipulates the power-allocation variables directly, and constraint satisfaction is enforced structurally by the optimizer. To mitigate LLM unreliability, we further incorporate multi-layer guardrails including normalization, exponential moving-average smoothing, and fallback mechanisms. Numerical experiments on an 8-channel system show that, with a fixed optimization core and unchanged system prompt, different natural-language policies induce qualitatively different operating points, including throughput-oriented allocation, channel prioritization, power-aware operation, and channel shutdown. In addition, under an abrupt channel-gain reversal, the proposed system autonomously reconfigures its steering signals and reduces the final mutual-information spread by 60% compared with the optimizer alone. These results suggest that LLMs can serve as policy interpreters for safe, flexible reconfiguration of communication-system optimizers without controller reimplementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13527v3">Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13860v3">"AI Psychosis" in Context: How Conversation History Shapes LLM Responses to Delusional Beliefs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Extended interaction with large language models (LLMs) has been linked to the reinforcement of delusional beliefs, a phenomenon attracting growing clinical and public concern. Yet most empirical work evaluates model safety in brief interactions, which may not reflect how these harms develop through sustained dialogue. We tested five models across three levels of accumulated context, using the same escalating delusional history to isolate its effect on model behaviour. Human raters coded responses on risk and safety dimensions, and each model was analysed qualitatively. Models separated into two distinct tiers: GPT-4o, Grok 4.1 Fast, and Gemini 3 Pro exhibited high-risk, low-safety profiles; Claude Opus 4.5 and GPT-5.2 Instant displayed the opposite pattern. As context accumulated, performance tended to degrade in the unsafe group, while the same material activated stronger safety interventions among the safer models. Qualitative analysis identified distinct mechanisms of failure, including validation of the user's delusional premises, elaboration beyond them, and attempting harm reduction from within the delusional frame. Safer models, however, often used the established relationship to support intervention, taking accountability for past missteps so that redirection would not be received as betrayal. These findings indicate that accumulated context functions as a stress test of safety architecture, revealing whether a model treats prior dialogue as a worldview to inherit or as evidence to evaluate. Short-context assessments may therefore mischaracterise model safety, underestimating danger in some systems while missing context-activated gains in others. The results suggest that delusional reinforcement by LLMs reflects a preventable alignment failure. In demonstrating that these harms can be resisted, the safer models establish a baseline future systems should now be expected to meet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21308v1">CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Enterprise LLM agents can dramatically improve workplace productivity, but their core capability, retrieving and using internal context to act on a user's behalf, also creates new risks for sensitive information leakage. We introduce CI-Work, a Contextual Integrity (CI)-grounded benchmark that simulates enterprise workflows across five information-flow directions and evaluates whether agents can convey essential content while withholding sensitive context in dense retrieval settings. Our evaluation of frontier models reveals that privacy failures are prevalent (violation rates range from 15.8%-50.9%, with leakage reaching up to 26.7%) and uncovers a counterintuitive trade-off critical for industrial deployment: higher task utility often correlates with increased privacy violations. Moreover, the massive scale of enterprise data and potential user behavior further amplify this vulnerability. Simply increasing model size or reasoning depth fails to address the problem. We conclude that safeguarding enterprise workflows requires a paradigm shift, moving beyond model-centric scaling toward context-centric architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21304v1">PAPERMIND: Benchmarking Agentic Reasoning and Critique over Scientific Papers in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Understanding scientific papers requires more than answering isolated questions or summarizing content. It involves an integrated reasoning process that grounds textual and visual information, interprets experimental evidence, synthesizes information across sources, and critically evaluates scientific claims. However, existing benchmarks typically assess these abilities in isolation, making it difficult to evaluate scientific paper understanding as a unified set of interacting cognitive abilities. In this work, we introduce PAPERMIND, a benchmark designed to evaluate integrated and agent-oriented scientific reasoning over research papers. PAPERMIND is constructed from real scientific papers across seven domains, including agriculture, biology, chemistry, computer science, medicine, physics, and economics. It comprises four complementary task families that collectively operationalize distinct cognitive facets of scientific paper reasoning, including multimodal grounding, experimental interpretation, cross-source evidence reasoning, and critical assessment. By analyzing model behavior across multiple tasks, PAPERMIND enables a diagnostic evaluation of integrated scientific reasoning behaviors that are difficult to assess through isolated task evaluations. Extensive experiments on both opensource and closed-source multimodal LLMs reveal consistent performance gaps across tasks, highlighting persistent challenges in integrated scientific reasoning and critique. Our benchmark and dataset are available at https:// github.com/Yanjun-Zhao/PaperMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21284v1">Spatial Metaphors for LLM Memory: A Critical Analysis of the MemPalace Architecture</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 20 pages, 10 tables. Code and data at https://github.com/web3guru888/mempalace-scientific-analysis
    </div>
    <details class="paper-abstract">
      MemPalace is an open-source AI memory system that applies the ancient method of loci (memory palace) spatial metaphor to organize long-term memory for large language models; launched in April 2026, it accumulated over 47,000 GitHub stars in its first two weeks and claims state-of-the-art retrieval performance on the LongMemEval benchmark (96.6% Recall@5) without requiring any LLM inference at write time. Through independent codebase analysis, benchmark replication, and comparison with competing systems, we find that MemPalace's headline retrieval performance is attributable primarily to its verbatim storage philosophy combined with ChromaDB's default embedding model (all-MiniLM-L6-v2), rather than to its spatial organizational metaphor per se -- the palace hierarchy (Wings->Rooms->Closets->Drawers) operates as standard vector database metadata filtering, an effective but well-established technique. However, MemPalace makes several genuinely novel contributions: (1) a contrarian verbatim-first storage philosophy that challenges extraction-based competitors, (2) an extremely low wake-up cost (approximately 170 tokens) through its four-layer memory stack, (3) a fully deterministic, zero-LLM write path enabling offline operation at zero API cost, and (4) the first systematic application of spatial memory metaphors as an organizing principle for AI memory systems. We also note that the competitive landscape is evolving rapidly, with Mem0's April 2026 token-efficient algorithm raising their LongMemEval score from approximately 49% to 93.4%, narrowing the gap between extraction-based and verbatim approaches. Our analysis concludes that MemPalace represents significant architectural insight wrapped in overstated claims -- a pattern common in rapidly adopted open-source projects where marketing velocity exceeds scientific rigor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12710v2">LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often demonstrate strong safety performance in high-resource languages, yet exhibit severe vulnerabilities when queried in low-resource languages. We attribute this gap to a mismatch between language-agnostic semantic understanding ability and language-dominant safety alignment biased toward high-resource languages. Consistent with this hypothesis, we empirically identify the semantic bottleneck in LLMs, an intermediate layer in which the geometry of model representations is governed primarily by shared semantic content rather than language identity. Building on this observation, we propose Language-Agnostic Semantic Alignment (LASA), which anchors safety alignment directly in semantic bottlenecks. Experiments show that LASA substantially improves safety across all languages: average attack success rate (ASR) drops from 24.7% to 2.8% on LLaMA-3.1-8B-Instruct and remains around 3-4% across Qwen2.5 and Qwen3 Instruct models (7B-32B). Together, our analysis and method offer a representation-level perspective on LLM safety, suggesting that safety alignment requires anchoring safety understanding not in surface text, but in the model's language-agnostic semantic space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21276v1">Do LLM Decoders Listen Fairly? Benchmarking How Language Model Priors Shape Bias in Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      As pretrained large language models replace task-specific decoders in speech recognition, a critical question arises: do their text-derived priors make recognition fairer or more biased across demographic groups? We evaluate nine models spanning three architectural generations (CTC with no language model, encoder-decoder with an implicit LM, and LLM-based with an explicit pretrained decoder) on about 43,000 utterances across five demographic axes (ethnicity, accent, gender, age, first language) using Common Voice 24 and Meta's Fair-Speech, a controlled-prompt dataset that eliminates vocabulary confounds. On clean audio, three findings challenge assumptions: LLM decoders do not amplify racial bias (Granite-8B has the best ethnicity fairness, max/min WER = 2.28); Whisper exhibits pathological hallucination on Indian-accented speech with a non-monotonic insertion-rate spike to 9.62% at large-v3; and audio compression predicts accent fairness more than LLM scale. We then stress-test these findings under 12 acoustic degradation conditions (noise, reverberation, silence injection, chunk masking) across both datasets, totaling 216 inference runs. Severe degradation paradoxically compresses fairness gaps as all groups converge to high WER, but silence injection amplifies Whisper's accent bias up to 4.64x by triggering demographic-selective hallucination. Under masking, Whisper enters catastrophic repetition loops (86% of 51,797 insertions) while explicit-LLM decoders produce 38x fewer insertions with near-zero repetition; high-compression audio encoding (Q-former) reintroduces repetition pathology even in LLM decoders. These results suggest that audio encoder design, not LLM scaling, is the primary lever for equitable and robust speech recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21264v1">Enhancing Online Recruitment with Category-Aware MoE and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accepted to ACL Industry Track 2026
    </div>
    <details class="paper-abstract">
      Person-Job Fit (PJF) is a critical component for online recruitment. Existing approaches face several challenges, particularly in handling low-quality job descriptions and similar candidate-job pairs, which impair model performance. To address these challenges, this paper proposes a large language model (LLM) based method with two novel techniques: (1) LLM-based data augmentation, which polishes and rewrites low-quality job descriptions by leveraging chain-of-thought (COT) prompts, and (2) category-aware Mixture of Experts (MoE) that assists in identifying similar candidate-job pairs. This MoE module incorporates category embeddings to dynamically assign weights to the experts and learns more distinguishable patterns for similar candidate-job pairs. We perform offline evaluations and online A/B tests on our recruitment platform. Our method relatively surpasses existing methods by 2.40% in AUC and 7.46% in GAUC, and boosts click-through conversion rate (CTCVR) by 19.4% in online tests, saving millions of CNY in external headhunting expenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21251v1">CAP: Controllable Alignment Prompting for Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Accpeted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained on unfiltered corpora inherently risk retaining sensitive information, necessitating selective knowledge unlearning for regulatory compliance and ethical safety. However, existing parameter-modifying methods face fundamental limitations: high computational costs, uncontrollable forgetting boundaries, and strict dependency on model weight access. These constraints render them impractical for closed-source models, yet current non-invasive alternatives remain unsystematic and reliant on empirical experience. To address these challenges, we propose the Controllable Alignment Prompting for Unlearning (CAP) framework, an end-to-end prompt-driven unlearning paradigm. CAP decouples unlearning into a learnable prompt optimization process via reinforcement learning, where a prompt generator collaborates with the LLM to suppress target knowledge while preserving general capabilities selectively. This approach enables reversible knowledge restoration through prompt revocation. Extensive experiments demonstrate that CAP achieves precise, controllable unlearning without updating model parameters, establishing a dynamic alignment mechanism that overcomes the transferability limitations of prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20331v2">Surrogate modeling for interpreting black-box LLMs in medical predictions</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), trained on vast datasets, encode extensive real-world knowledge within their parameters, yet their black-box nature obscures the mechanisms and extent of this encoding. Surrogate modeling, which uses simplified models to approximate complex systems, can offer a path toward better interpretability of black-box models. We propose a surrogate modeling framework that quantitatively explains LLM-encoded knowledge. For a specific hypothesis derived from domain knowledge, this framework approximates the latent LLM knowledge space using observable elements (input-output pairs) through extensive prompting across a comprehensive range of simulated scenarios. Through proof-of-concept experiments in medical predictions, we demonstrate our framework's effectiveness in revealing the extent to which LLMs "perceive" each input variable in relation to the output. Particularly, given concerns that LLMs may perpetuate inaccuracies and societal biases embedded in their training data, our experiments using this framework quantitatively revealed both associations that contradict established medical knowledge and the persistence of scientifically refuted racial assumptions within LLM-encoded knowledge. By disclosing these issues, our framework can act as a red-flag indicator to support the safe and reliable application of these models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21231v1">SparKV: Overhead-Aware KV Cache Loading for Efficient On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 IEEE INTERNET OF THINGS HOURNAL, 11 pages under major revision
    </div>
    <details class="paper-abstract">
      Efficient inference for on-device Large Language Models (LLMs) remains challenging due to limited hardware resources and the high cost of the prefill stage, which processes the full input context to construct Key-Value (KV) caches. We present SparKV, an adaptive KV loading framework that combines cloud-based KV streaming with on-device computation. SparKV models the cost of individual KV chunks and decides whether each chunk should be streamed or computed locally, while overlapping the two execution paths to reduce latency. To handle fluctuations in wireless connectivity and edge resource availability, SparKV further refines offline-generated schedules at runtime to rebalance communication and computation costs. Experiments across diverse datasets, LLMs, and edge devices show that SparKV reduces Time-to-First-Token by 1.3$x-5.1x with negligible impact on response quality, while lowering per-request energy consumption by 1.5x to 3.3x, demonstrating its robustness and practicality for real-world on-device deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21223v1">Zero-Shot Detection of LLM-Generated Text via Implicit Reward Model</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their ability to generate human-like text has raised concerns about potential misuse. This underscores the need for reliable and effective methods to detect LLM-generated text. In this paper, we propose IRM, a novel zero-shot approach that leverages Implicit Reward Models for LLM-generated text detection. Such implicit reward models can be derived from publicly available instruction-tuned and base models. Previous reward-based method relies on preference construction and task-specific fine-tuning. In comparison, IRM requires neither preference collection nor additional training. We evaluate IRM on the DetectRL benchmark and demonstrate that IRM can achieve superior detection performance, outperforms existing zero-shot and supervised methods in LLM-generated text detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01493v2">OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Solving diverse partial differential equations (PDEs) is fundamental in science and engineering. Large language models (LLMs) have demonstrated strong capabilities in code generation, symbolic reasoning, and tool use, but reliably solving PDEs across heterogeneous settings remains challenging. Prior work on LLM-based code generation and transformer-based foundation models for PDE learning has shown promising advances. However, a persistent trade-off between execution success rate and numerical accuracy arises, particularly when generalization to unseen parameters and boundary conditions is required. In this work, we propose OpInf-LLM, an LLM parametric PDE solving framework via operator inference. The proposed framework leverages small amounts of solution data to enable accurate prediction of diverse PDE instances, including unseen parameters and configurations, and provides seamless integration with LLMs for natural language task specification and physics-based reasoning of proper feature parameterization. Its low computational demands and unified solution pipeline further enable a high execution success rate across heterogeneous settings, opening new possibilities for generalizable reduced-order modeling in LLM-based PDE solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04573v6">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR} (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design, combined with explicit diversity guidance during diffusion inference, enables the generation of multiple diverse reasoning trajectories that explore distinct regions of the latent space, rather than producing repetitive solutions as often occurs in standard autoregressive sampling. We conduct evaluations on a suite of mathematical reasoning, code generation and puzzle planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20064v2">Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 ICLR'26
    </div>
    <details class="paper-abstract">
      Speculative decoding is widely used in accelerating large language model (LLM) inference. In this work, we focus on the online draft model selection problem in speculative decoding. We design an algorithm that provably competes with the best draft model in hindsight for each query in terms of either the token acceptance probability or expected acceptance length. In particular, we show that we can accurately evaluate all draft models, instead of only the chosen model without incurring additional queries to the target model, which allows us to improve exponentially over the existing bandit-based approach as the number of draft models increases. Our approach is generically applicable with any speculative decoding methods (single draft, multi-drafts and draft-trees). Moreover, we design system-efficient versions of online learners and demonstrate that the overhead in computation and latency can be substantially reduced. We conduct extensive experiments on open-source LLMs and diverse datasets, demonstrating that our methods substantially outperform the state-of-the-art EAGLE3 and the BanditSpec baseline in a variety of domains where specialized domain-expert drafters are available, especially when long reasoning chains are required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21159v1">Adaptive Instruction Composition for Automated LLM Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted to ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Many approaches to LLM red-teaming leverage an attacker LLM to discover jailbreaks against a target. Several of them task the attacker with identifying effective strategies through trial and error, resulting in a semantically limited range of successes. Another approach discovers diverse attacks by combining crowdsourced harmful queries and tactics into instructions for the attacker, but does so at random, limiting effectiveness. This article introduces a novel framework, Adaptive Instruction Composition, that combines crowdsourced texts according to an adaptive mechanism trained to jointly optimize effectiveness with diversity. We use reinforcement learning to balance exploration with exploitation in a combinatorial space of instructions to guide the attacker toward diverse generations tailored to target vulnerabilities. We demonstrate that our approach substantially outperforms random combination on a set of effectiveness and diversity metrics, even under model transfer. Further, we show that it surpasses a host of recent adaptive approaches on Harmbench. We employ a lightweight neural contextual bandit that adapts to contrastive embedding inputs, and provide ablations suggesting that the contrastive pretraining enables the network to rapidly generalize and scale to the massive space as it learns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21152v1">Dialect vs Demographics: Quantifying LLM Bias from Implicit Linguistic Signals vs. Explicit User Profiles</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 In The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26), June 25--28, 2026, Montreal, Canada. ACM, New York, NY, USA, 32 pages
    </div>
    <details class="paper-abstract">
      As state-of-the-art Large Language Models (LLMs) have become ubiquitous, ensuring equitable performance across diverse demographics is critical. However, it remains unclear whether these disparities arise from the explicitly stated identity itself or from the way identity is signaled. In real-world interactions, users' identity is often conveyed implicitly through a complex combination of various socio-linguistic factors. This study disentangles these signals by employing a factorial design with over 24,000 responses from two open-weight LLMs (Gemma-3-12B and Qwen-3-VL-8B), comparing prompts with explicitly announced user profiles against implicit dialect signals (e.g., AAVE, Singlish) across various sensitive domains. Our results uncover a unique paradox in LLM safety where users achieve ``better'' performance by sounding like a demographic than by stating they belong to it. Explicit identity prompts activate aggressive safety filters, increasing refusal rates and reducing semantic similarity compared to our reference text for Black users. In contrast, implicit dialect cues trigger a powerful ``dialect jailbreak,'' reducing refusal probability to near zero while simultaneously achieving a greater level of semantic similarity to the reference texts compared to Standard American English prompts. However, this ``dialect jailbreak'' introduces a critical safety trade-off regarding content sanitization. We find that current safety alignment techniques are brittle and over-indexed on explicit keywords, creating a bifurcated user experience where ``standard'' users receive cautious, sanitized information while dialect speakers navigate a less sanitized, more raw, and potentially a more hostile information landscape and highlights a fundamental tension in alignment--between equitable and linguistic diversity--and underscores the need for safety mechanisms that generalize beyond explicit cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10708v2">SafeTrans: LLM-assisted Transpilation from C to Rust</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Rust is a strong contender for a memory-safe alternative to C as a "systems" language, but porting the vast amount of existing C code to Rust remains daunting. In this paper, we evaluate the potential of large language models (LLMs) to automate the transpilation of C code to idiomatic Rust. We present SafeTrans, a generic framework that leverages LLMs to i) transpile C code into Rust, and ii) iteratively repair compilation and runtime errors. A key novelty of our approach is a few-shot guided repair technique for translation errors, which provides contextual information and example code snippets for specific error types, guiding the LLM toward the correct solution. Another novel aspect of our work is the evaluation of the security implications of the transpilation process, showing how some vulnerability classes in C persist in the translated Rust code. SafeTrans was evaluated with six leading LLMs on 2,653 C programs and two real-world C projects. Our results show that iterative repair improves the rate of successful translations from 54% to 80% for the best-performing LLM (gpt-4o).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03248v3">STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 ACL 2026 Main, we release our code publicly at https://github.com/LingFengGold/STReasoner
    </div>
    <details class="paper-abstract">
      Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11066v6">Exploring Collatz Dynamics with Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 233 pages, 11 figures, 52 tables
    </div>
    <details class="paper-abstract">
      We present a comprehensive structural analysis of the Collatz conjecture through ~1014 computational experiments yielding 630 formal results. By systematically deploying 29 distinct mathematical paradigms--including transfer operator spectral theory, S-unit equations, p-adic interpolation, martingale methods, modular sieving, formal language theory, cascade algebra, discrete logarithm obstruction, and Diophantine approximation--we establish a Paradigm Exhaustion Theorem: every known framework for promoting distributional convergence ("almost all orbits descend") to pointwise convergence ("all orbits descend") encounters an irreducible structural obstruction when applied to the Syracuse map. On the unconditional side, we prove: (i) the Syracuse transfer operator has a uniform spectral gap for all M, implying equidistribution modulo any power of 2; (ii) any nontrivial cycle of length L satisfies D > 2^F for all L >= 3, giving ord_D(2) > F and F+1 distinct residues mod D; (iii) divergent starting points have natural density 0 and Hausdorff dimension ~0.68; (iv) the formal language of divergent-compatible v-sequences is not context-free; (v) cylinder-averaged density-1 convergence is proved unconditionally via spectral contraction on the invariant core I_2; (vi) a discrete logarithm triple filter achieves 100% cycle blockage for all L tested. We identify the Distributional-to-Pointwise Gap as the irreducible core and prove it equivalent to the divergence component. The modular sieve is permanently nonempty via the Mersenne Bypass. The present work is not a proof of the Collatz conjecture; it characterizes why the conjecture resists proof. The 29-paradigm exhaustion constitutes the most comprehensive structural survey of Collatz attack surfaces to date. Produced through human-LLM collaboration; see Section 12.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22753v2">From Rookie to Expert: Manipulating LLMs for Automated Vulnerability Exploitation in Enterprise Software</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      LLMs democratize software engineering by enabling non-programmers to create applications, but this same accessibility fundamentally undermines security assumptions that have guided software engineering for decades. We show in this work how publicly available LLMs can be socially engineered to transform novices into capable attackers, challenging the foundational principle that exploitation requires technical expertise. To that end, we propose RSA (Role-assignment, Scenario-pretexting, and Action-solicitation), a pretexting strategy that manipulates LLMs into generating functional exploits despite their safety mechanisms. Testing against Odoo -- a widely used ERP platform, we evaluated five mainstream LLMs (GPT-4o, Gemini, Claude, Microsoft Copilot, and DeepSeek) and successfully exploited every tested CVE: at least one LLM produced a functional exploit for each within 3-5 prompting rounds. While prior work~\cite{jin2025good} found LLM-assisted attacks difficult and requiring manual effort, we demonstrate that this overhead can be eliminated entirely. Our findings invalidate core software engineering security principles: the distinction between technical and non-technical actors no longer provides valid threat models; technical complexity of vulnerability descriptions offers no protection when LLMs can abstract it away; and traditional security boundaries dissolve when the same tools that build software can be manipulated to break it. This represents a paradigm shift in software engineering -- we must redesign security practices for an era where exploitation requires only the ability to craft prompts, not understand code. Artifacts available at: https://anonymous.4open.science/r/From-Rookie-to-Attacker-D8B3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21102v1">Leveraging Multimodal LLMs for Built Environment and Housing Attribute Assessment from Street-View Imagery</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      We present a novel framework for automatically evaluating building conditions nationwide in the United States by leveraging large language models (LLMs) and Google Street View (GSV) imagery. By fine-tuning Gemma 3 27B on a modest human-labeled dataset, our approach achieves strong alignment with human mean opinion scores (MOS), outperforming even individual raters on SRCC and PLCC relative to the MOS benchmark. To enhance efficiency, we apply knowledge distillation, transferring the capabilities of Gemma 3 27B to a smaller Gemma 3 4B model that achieves comparable performance with a 3x speedup. Further, we distill the knowledge into a CNN-based model (EfficientNetV2-M) and a transformer (SwinV2-B), delivering close performance while achieving a 30x speed gain. Furthermore, we investigate LLMs' capabilities for assessing an extensive list of built environment and housing attributes through a human-AI alignment study and develop a visualization dashboard that integrates LLM assessment outcomes for downstream analysis by homeowners. Our framework offers a flexible and efficient solution for large-scale building condition assessment, enabling high accuracy with minimal human labeling effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21092v1">Mind the Prompt: Self-adaptive Generation of Task Plan Explanations via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into complex software systems enables the generation of human-understandable explanations of opaque AI processes, such as automated task planning. However, the quality and reliability of these explanations heavily depend on effective prompt engineering. The lack of a systematic understanding of how diverse stakeholder groups formulate and refine prompts hinders the development of tools that can automate this process. We introduce COMPASS (COgnitive Modelling for Prompt Automated SynthesiS), a proof-of-concept self-adaptive approach that formalises prompt engineering as a cognitive and probabilistic decision-making process. COMPASS models unobservable users' latent cognitive states, such as attention and comprehension, uncertainty, and observable interaction cues as a POMDP, whose synthesised policy enables adaptive generation of explanations and prompt refinements. We evaluate COMPASS using two diverse cyber-physical system case studies to assess the adaptive explanation generation and their qualities, both quantitatively and qualitatively. Our results demonstrate the feasibility of COMPASS integrating human cognition and user profile's feedback into automated prompt synthesis in complex task planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21076v1">Serialisation Strategy Matters: How FHIR Data Format Affects LLM Medication Reconciliation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 14 pages, 7 figures, independent research
    </div>
    <details class="paper-abstract">
      Medication reconciliation at clinical handoffs is a high-stakes, error-prone process. Large language models are increasingly proposed to assist with this task using FHIR-structured patient records, but a fundamental and largely unstudied variable is how the FHIR data is serialised before being passed to the model. We present the first systematic comparison of four FHIR serialisation strategies (Raw JSON, Markdown Table, Clinical Narrative, and Chronological Timeline) across five open-weight models (Phi-3.5-mini, Mistral-7B, BioMistral-7B, Llama-3.1-8B, Llama-3.3-70B) on a controlled benchmark of 200 synthetic patients, totalling 4,000 inference runs. We find that serialisation strategy has a large, statistically significant effect on performance for models up to 8B parameters: Clinical Narrative outperforms Raw JSON by up to 19 F1 points for Mistral-7B (r = 0.617, p < 10^{-10}). This advantage reverses at 70B, where Raw JSON achieves the best mean F1 of 0.9956. In all 20 model and strategy combinations, mean precision exceeds mean recall: omission is the dominant failure mode, with models more often missing an active medication than fabricating one, which changes how clinical safety auditing priorities should be set. Smaller models plateau at roughly 7-10 concurrent active medications, leaving polypharmacy patients, the patients most at risk from reconciliation errors, systematically underserved. BioMistral-7B, a domain-pretrained model without instruction tuning, produces zero usable output in all conditions, showing that domain pretraining alone is not sufficient for structured extraction. These results offer practical, evidence-based format recommendations for clinical LLM deployment: Clinical Narrative for models up to 8B, Raw JSON for 70B and above. The complete pipeline is reproducible on open-source tools running on an AWS g6e.xlarge instance (NVIDIA L40S, 48 GB VRAM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21072v1">Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. However, the low cross-node network bandwidth makes communication the primary bottleneck. In this paper, we introduce BloomBee, an internet-scale distributed LLM inference framework. BloomBee integrates LLM-layer assignment, micro-batching and tensor offloading to optimize communication from multiple dimensions. Additionally, BloomBee formulates the coordination of these techniques as an optimization problem and solves it using dynamic programming. BloomBee also customizes lossless compression and speculative decoding according to low-bandwidth network settings to reduce communication overhead. We evaluate BloomBee across a spectrum of network environments and show that it improves service throughput by up to 1.76x. It also reduces average latency by up to 43.20% compared to state-of-the-art decentralized LLM inference systems. BloomBee is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20987v1">Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Long horizon interactive environments are a testbed for evaluating agents skill usage abilities. These environments demand multi step reasoning, the chaining of multiple skills over many timesteps, and robust decision making under delayed rewards and partial observability. Games are a good testbed for evaluating agent skill usage in environments. Large Language Models (LLMs) offer a promising alternative as game playing agents, but they often struggle with consistent long horizon decision making because they lack a mechanism to discover, retain, and reuse structured skills across episodes. We present COSPLAY, a co evolution framework in which an LLM decision agent retrieves skills from a learnable skill bank to guide action taking, while an agent managed skill pipeline discovers reusable skills from the agents unlabeled rollouts to form a skill bank. Our framework improves both the decision agent to learn better skill retrieval and action generation, while the skill bank agent continually extracts, refines, and updates skills together with their contracts. Experiments across six game environments show that COSPLAY with an 8B base model achieves over 25.1 percent average reward improvement against four frontier LLM baselines on single player game benchmarks while remaining competitive on multi player social reasoning games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20811v1">Diagnosing CFG Interpretation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into agentic systems, they must adhere to dynamically defined, machine-interpretable interfaces. We evaluate LLMs as in-context interpreters: given a novel context-free grammar, can LLMs generate syntactically valid, behaviorally functional, and semantically faithful outputs? We introduce RoboGrid, a framework that disentangles syntax, behavior, and semantics through controlled stress-tests of recursion depth, expression complexity, and surface styles. Our experiments reveal a consistent hierarchical degradation: LLMs often maintain surface syntax but fail to preserve structural semantics. Despite the partial mitigation provided by CoT reasoning, performance collapses under structural density, specifically deep recursion and high branching, with semantic alignment vanishing at extreme depths. Furthermore, "Alien" lexicons reveal that LLMs rely on semantic bootstrapping from keywords rather than pure symbolic induction. These findings pinpoint critical gaps in hierarchical state-tracking required for reliable, grammar-agnostic agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20948v1">Can Virtual Agents Care? Designing an Empathetic and Personalized LLM-Driven Conversational Agent</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted manuscript version to be presented at the SCI-2026
    </div>
    <details class="paper-abstract">
      Mental health challenges are rising globally, while traditional support services face limited availability and high costs. Large language models offer potential for conversational support, but often lack personalization, empathy, and factual grounding. A virtual agent framework is introduced to provide empathetic, personalized, and reliable wellbeing support through retrieval-augmented architecture, structured memory, and multimodal interaction. Objective benchmarks demonstrate improved retrieval and response quality, particularly for smaller models. A cross-cultural study with university students from Vietnam and Australia shows the system outperforms LLM-only baselines in coherence, perceived accuracy, and empathy, with most participants clearly preferring the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20803v1">Autonomous LLM-generated Feedback for Student Exercises in Introductory Software Engineering Courses</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Introductory Software Engineering (SE) courses face rapidly increasing student enrollment numbers, participants with diverse backgrounds and the influence of Generative AI (GenAI) solutions. High teacher-to-student ratios often challenge providing timely, high-quality, and personalized feedback a significant challenge for educators. To address these challenges, we introduce NAILA, a tool that provides 24/7 autonomous feedback for student exercises. Utilizing GenAI in the form of modern LLMs, NAILA processes student solutions provided in open document formats, evaluating them against teacher-defined model solutions through specialized prompt templates. We conducted an empirical study involving 900+ active students at the University of Duisburg-Essen to assess four main research questions investigating (1) the underlying motivations that drive students to either adopt or reject NAILA, (2) user acceptance by measuring perceived usefulness and ease of use alongside subjective learning progress, (3) how often and how consistently students engage with NAILA, and (4) how using NAILA to receive AI feedback impacts on academic performance compared to human feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11423v3">Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), allows users to flag misleading posts, attach contextual notes, and rate the notes' helpfulness. However, our empirical analysis of 30.8K health-related notes reveals substantial latency, with a median delay of 17.6 hours before notes receive a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified LLM-based framework that augments Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, supported by a hierarchical three-stage evaluation of relevance, correctness, and helpfulness. We instantiate the framework with HealthNotes, a benchmark of 1.2K health notes annotated for helpfulness, and a fine-tuned helpfulness judge. Our analysis first uncovers a key loophole in current crowd-sourced governance: voters frequently conflate stylistic fluency with factual accuracy. Addressing this via our hierarchical evaluation, experiments across 15 representative LLMs demonstrate that CrowdNotes+ significantly outperforms human contributors in note correctness, helpfulness, and evidence utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16676v2">WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Findings of the Association for Computational Linguistics: ACL 2026
    </div>
    <details class="paper-abstract">
      Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20795v1">Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Artificial Intelligence; Knowledge Representation and Reasoning; Information Retrieval; Machine Learning
    </div>
    <details class="paper-abstract">
      This paper presents a hybrid architecture for intelligent systems in which large language models (LLMs) are extended with an external ontological memory layer. Instead of relying solely on parametric knowledge and vector-based retrieval (RAG), the proposed approach constructs and maintains a structured knowledge graph using RDF/OWL representations, enabling persistent, verifiable, and semantically grounded reasoning. The core contribution is an automated pipeline for ontology construction from heterogeneous data sources, including documents, APIs, and dialogue logs. The system performs entity recognition, relation extraction, normalization, and triple generation, followed by validation using SHACL and OWL constraints, and continuous graph updates. During inference, LLMs operate over a combined context that integrates vector-based retrieval with graph-based reasoning and external tool interaction. Experimental observations on planning tasks, including the Tower of Hanoi benchmark, indicate that ontology augmentation improves performance in multi-step reasoning scenarios compared to baseline LLM systems. In addition, the ontology layer enables formal validation of generated outputs, transforming the system into a generation-verification-correction pipeline. The proposed architecture addresses key limitations of current LLM-based systems, including lack of long-term memory, weak structural understanding, and limited reasoning capabilities. It provides a foundation for building agent-based systems, robotics applications, and enterprise AI solutions that require persistent knowledge, explainability, and reliable decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20791v1">Can "AI" Be a Doctor? A Study of Empathy, Readability, and Alignment in Clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in healthcare, yet their communicative alignment with clinical standards remains insufficiently quantified. We conduct a multidimensional evaluation of general-purpose and domain-specialized LLMs across structured medical explanations and real-world physician-patient interactions, analyzing semantic fidelity, readability, and affective resonance. Baseline models amplify affective polarity relative to physicians (Very Negative: 43.14-45.10% vs. 37.25%) and, in larger architectures such as GPT-5 and Claude, produce substantially higher linguistic complexity (FKGL up to 16.91-17.60 vs. 11.47-12.50 in physician-authored responses). Empathy-oriented prompting reduces extreme negativity and lowers grade-level complexity (up to -6.87 FKGL points for GPT-5) but does not significantly increase semantic fidelity. Collaborative rewriting yields the strongest overall alignment. Rephrase configurations achieve the highest semantic similarity to physician answers (up to mean = 0.93) while consistently improving readability and reducing affective extremity. Dual stakeholder evaluation shows that no model surpasses physicians on epistemic criteria, whereas patients consistently prefer rewritten variants for clarity and emotional tone. These findings suggest that LLMs function most effectively as collaborative communication enhancers rather than replacements for clinical expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08570v2">QuanBench+: A Unified Multi-Framework Benchmark for LLM-Based Quantum Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 24 pages total, 25 figures, 5 tables, including supplementary material. Accepted to the ICLR 2026 Workshop on I Can't Believe It's Not Better
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for code generation, yet quantum code generation is still evaluated mostly within single frameworks, making it difficult to separate quantum reasoning from framework familiarity. We introduce QuanBench+, a unified benchmark spanning Qiskit, PennyLane, and Cirq, with 42 aligned tasks covering quantum algorithms, gate decomposition, and state preparation. We evaluate models with executable functional tests, report Pass@1 and Pass@5, and use KL-divergence-based acceptance for probabilistic outputs. We additionally study Pass@1 after feedback-based repair, where a model may revise code after a runtime error or wrong answer. Across frameworks, the strongest one-shot scores reach 59.5% in Qiskit, 54.8% in Cirq, and 42.9% in PennyLane; with feedback-based repair, the best scores rise to 83.3%, 76.2%, and 66.7%, respectively. These results show clear progress, but also that reliable multi-framework quantum code generation remains unsolved and still depends strongly on framework-specific knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20945v1">Breaking Bad: Interpretability-Based Safety Audits of State-of-the-Art LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Effective safety auditing of large language models (LLMs) demands tools that go beyond black-box probing and systematically uncover vulnerabilities rooted in model internals. We present a comprehensive, interpretability-driven jailbreaking audit of eight SOTA open-source LLMs: Llama-3.1-8B, Llama-3.3-70B-4bt, GPT-oss- 20B, GPT-oss-120B, Qwen3-0.6B, Qwen3-32B, Phi4-3.8B, and Phi4-14B. Leveraging interpretability-based approaches -- Universal Steering (US) and Representation Engineering (RepE) -- we introduce an adaptive two-stage grid search algorithm to identify optimal activation-steering coefficients for unsafe behavioral concepts. Our evaluation, conducted on a curated set of harmful queries and a standardized LLM-based judging protocol, reveals stark contrasts in model robustness. The Llama-3 models are highly vulnerable, with up to 91\% (US) and 83\% (RepE) jailbroken responses on Llama-3.3-70B-4bt, while GPT-oss-120B remains robust to attacks via both interpretability approaches. Qwen and Phi models show mixed results, with the smaller Qwen3-0.6B and Phi4-3.8B mostly exhibiting lower jailbreaking rates, while their larger counterparts are more susceptible. Our results establish interpretability-based steering as a powerful tool for systematic safety audits, but also highlight its dual-use risks and the need for better internal defenses in LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20732v1">Anchor-and-Resume Concession Under Dynamic Pricing for LLM-Augmented Freight Negotiation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Freight brokerages negotiate thousands of carrier rates daily under dynamic pricing conditions where models frequently revise targets mid-conversation. Classical time-dependent concession frameworks use a fixed shape parameter $β$ that cannot adapt to these updates. Deriving $β$ from the live spread enables adaptation but introduces a new problem: a pricing shift can cause the formula to retract a previous offer, violating monotonicity. LLM-powered brokers offer flexibility but require expensive reasoning models, produce non-deterministic pricing, and remain vulnerable to prompt injection. We propose a two-index anchor-and-resume framework that addresses both limitations. A spread-derived $β$ maps each load's margin structure to the correct concession posture, while the anchor-and-resume mechanism guarantees monotonically non-decreasing offers under arbitrary pricing shifts. All pricing decisions remain in a deterministic formula; the LLM, when used, serves only as a natural-language translation layer. Empirical evaluation across 115,125 negotiations shows that the adaptive $β$ tailors behavior by regime: in narrow spreads, it concedes quickly to prioritize deal closure and load coverage; in medium and wide spreads, it matches or exceeds the best fixed-$β$ baselines in broker savings. Against an unconstrained 20-billion-parameter LLM broker, it achieves similar agreement rates and savings. Against LLM-powered carriers as more realistic stochastic counterparties, it maintains comparable savings and higher agreement rates than against rule-based opponents. By decoupling the LLM from pricing logic, the framework scales horizontally to thousands of concurrent negotiations with negligible inference cost and transparent decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15319v2">Explainable Iterative Data Visualisation Refinement via an LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space (typically 2D or 3D), based on which visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. However, finding a suitable algorithm configuration, particularly hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery remains challenging. To address this challenge, we propose an agentic AI pipleline that leverages a large language model (LLM) to bridge the gap between rigorous quantitative assessment and qualitative human insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes hard metrics with descriptive summaries, and suggests actionable recommendation of algorithm configuration for refining data visualization. By implementing an iterative optimization loop of this process, the system is able to produce rapidly a high-quality visualization plot, in full automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.14590v2">MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted at MLSys 2026
    </div>
    <details class="paper-abstract">
      Quantization has become one of the most effective methodologies to compress LLMs into smaller size. However, the existing quantization solutions still show limitations of either non-negligible accuracy drop or low system efficiency. In this paper, we propose MixLLM that explores the optimization space of mixed-precision quantization between output features, based on the insight that different features matter differently in the model. MixLLM identifies the important output features in the global view rather than within each single layer, effectively assigning larger bit-width to output features that need it the most to achieve high accuracy and low memory usage. We present the sweet spot of quantization configuration of algorithm-system co-design with high accuracy and system efficiency. To address the system challenge, we design the two-step dequantization to make use of the Tensor Core easily and fast data type conversion to reduce dequantization overhead, and present the software pipeline to overlap the memory access, dequantization and the MatMul to the best. Extensive experiments show that with only 10\% more bits, the perplexity increase can be reduced from about 0.5 in SOTA to within 0.2 for Llama 3.1 70B, while MMLU-Pro loss can be reduced from 1.92 to 0.99 over the SOTA of three popular models. Besides its superior accuracy, MixLLM also achieves state-of-the-art system efficiency. Code is released at https://github.com/microsoft/MixLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.03594v3">BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted at MLSys 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly play an important role in a wide range of information processing and management tasks in industry. Many of these tasks are performed in large batches or even offline, and the performance indicator for which is throughput. These tasks usually show the characteristic of prefix sharing, where different prompt input can partially show the common prefix. However, the existing LLM inference engines tend to optimize the streaming requests and show limitations of supporting the large batched tasks with the prefix sharing characteristic. The existing solutions use the LRU-based cache to reuse the KV context of common prefix between requests. The KV context that are about to be reused may be prematurely evicted with the implicit cache management. Besides, the streaming oriented systems do not leverage the request-batch information and can not mix the decoding tokens with the prefill chunks to the best for the batched scenarios, and thus fails to saturate the GPU. We propose BatchLLM to address the above problems. BatchLLM explicitly identifies the common prefixes globally. The requests sharing the same prefix will be scheduled together to reuse the KV context the best. BatchLLM reorders the requests and schedules the requests with larger ratio of decoding first to better mix the decoding tokens with the latter prefill chunks, and applies memory-centric token batching to enlarge the token-batch sizes, which helps to increase the GPU utilization. Extensive evaluation shows that BatchLLM outperforms vLLM and SGLang by $1.3\times$ to $10.8\times$ on a set of microbenchmarks and a typical industry workload under different hardware environments. Code is available at https://github.com/microsoft/MixLLM/tree/batchllm_vllm_064.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20685v1">MGDA-Decoupled: Geometry-Aware Multi-Objective Optimisation for DPO-based LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted to the Algorithmic Fairness Across Alignment Procedures and Agentic Systems Workshop at ICLR 2026
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) to desirable human values requires balancing multiple, potentially conflicting objectives such as helpfulness, truthfulness, and harmlessness, which presents a multi-objective optimisation challenge. Most alignment pipelines rely on a fixed scalarisation of these objectives, which can introduce procedural unfairness by systematically under-weighting harder-to-optimise or minority objectives. To promote more equitable trade-offs, we introduce MGDA-Decoupled, a geometry-based multi-objective optimisation algorithm that finds a shared descent direction while explicitly accounting for each objective's convergence dynamics. In contrast to prior methods that depend on reinforcement learning (e.g., GAPO) or explicit reward models (e.g., MODPO), our approach operates entirely within the lightweight Direct Preference Optimisation (DPO) paradigm. Experiments on the UltraFeedback dataset show that geometry-aware methods -- and MGDA-Decoupled in particular -- achieve the highest win rates against golden responses, both overall and per objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07117v3">From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) have emerged as powerful tools for learning over structured data, including text-attributed graphs (TAGs), which are common in domains such as citation networks, social platforms, and knowledge graphs. GNNs are not inherently interpretable and thus, many explanation methods have been proposed. However, existing explanation methods often struggle to generate interpretable, fine-grained rationales, especially when node attributes include rich natural language. In this work, we introduce GSPELL, a lightweight, post-hoc framework that uses large language models (LLMs) to generate faithful and interpretable explanations for GNN predictions. GSPELL projects GNN node embeddings into the LLM embedding space and constructs hybrid prompts that interleave soft prompts with textual inputs from the graph structure. This enables the LLM to reason about GNN internal representations and to produce natural-language explanations, along with concise explanation subgraphs. Our experiments across real-world TAG datasets demonstrate that GSPELL achieves a favorable trade-off between fidelity and sparsity, while improving human-centric metrics such as insightfulness. GSPELL sets a new direction for LLM-based explainability in graph learning by aligning GNN internals with human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20658v1">Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 35 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate games and invest in multiplicative team production (rather than greedy strategies) produce better scientific reports across three outcomes, accuracy, quality, and completion. These associations hold after controlling for multiple factors, indicating that cooperative disposition is a distinct, measurable property of LLMs not reducible to general ability. Our behavioral games framework thus offers a fast and inexpensive diagnostic for screening cooperative fitness before costly multi-agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18349v2">HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted to Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version. 10 pages, 2 figures. Code: https://github.com/ZeroLoss-Lab/HiGMem
    </div>
    <details class="paper-abstract">
      Long-term conversational large language model (LLM) agents require memory systems that can recover relevant evidence from historical interactions without overwhelming the answer stage with irrelevant context. However, existing memory systems, including hierarchical ones, still often rely solely on vector similarity for retrieval. It tends to produce bloated evidence sets: adding many superficially similar dialogue turns yields little additional recall, but lowers retrieval precision, increases answer-stage context cost, and makes retrieved memories harder to inspect and manage. To address this, we propose HiGMem (Hierarchical and LLM-Guided Memory System), a two-level event-turn memory system that allows LLMs to use event summaries as semantic anchors to predict which related turns are worth reading. This allows the model to inspect high-level event summaries first and then focus on a smaller set of potentially useful turns, providing a concise and reliable evidence set through reasoning, while avoiding the retrieval overhead that would be excessively high compared to vector retrieval. On the LoCoMo10 benchmark, HiGMem achieves the best F1 on four of five question categories and improves adversarial F1 from 0.54 to 0.78 over A-Mem, while retrieving an order of magnitude fewer turns. Code is publicly available at https://github.com/ZeroLoss-Lab/HiGMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02931v2">Memorization, Emergence, and Explaining Reversal Failures: A Controlled Study of Relational Semantics in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 ACL2026 Main Long Paper
    </div>
    <details class="paper-abstract">
      Autoregressive LLMs perform well on relational tasks that require linking entities via relational words (e.g., father/son, friend), but it is unclear whether they learn the logical semantics of such relations (e.g., symmetry and inversion logic) and, if so, whether reversal-type failures arise from missing relational semantics or left-to-right order bias. We propose a controlled Knowledge Graph-based synthetic framework that generates text from symmetric/inverse triples, train GPT-style autoregressive models from scratch, and evaluate memorization, logical inference, and in-context generalization to unseen entities to address these questions. We find a sharp phase transition in which relational semantics emerge with sufficient logic-bearing supervision, even in shallow (2-3 layer) models, and that successful generalization aligns with stable intermediate-layer signals. Finally, order-matched forward/reverse tests and a diffusion baseline indicate that reversal failures are primarily driven by autoregressive order bias rather than deficient inversion semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20582v1">Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      We study emergent social dynamics in LLM agents playing The Resistance: Avalon, a hidden-role deception game. Unlike prior work on single-game performance, our agents play repeated games while retaining memory of previous interactions, including who played which roles and how they behaved, enabling us to study how social dynamics evolve. Across 188 games, two key phenomena emerge. First, reputation dynamics emerge organically when agents retain cross-game memory: agents reference past behavior in statements like "I am wary of repeating last game's mistake of over-trusting early success." These reputations are role-conditional: the same agent is described as "straightforward" when playing good but "subtle" when playing evil, and high-reputation players receive 46% more team inclusions. Second, higher reasoning effort supports more strategic deception: evil players more often pass early missions to build trust before sabotaging later ones, 75% in high-effort games vs 36% in low-effort games. Together, these findings show that repeated interaction with memory gives rise to measurable reputation and deception dynamics among LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20564v1">Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logical deduction, where a single transition error can propagate through the entire reasoning chain, leading to unstable performance. In this work, we identify logical connectives as primary points of this structural fragility. Through empirical analysis, we show that connective tokens function as high entropy forking points, at which models frequently struggle to determine the correct logical direction. Motivated by this observation, we hypothesize that intervening in logical connective selection can guide LLMs toward more correct logical direction, thereby improving the overall reasoning chain. To validate this hypothesis, we propose a multi-layered framework that intervenes specifically at these logic-critical junctions in the reasoning process. Our framework includes (1) Gradient-based Logical Steering to guide LLMs internal representations towards valid reasoning subspaces, (2) Localized Branching to resolve ambiguity via targeted look-ahead search, and (3) Targeted Transition Preference Optimization, a surgical reinforcement learning objective that selectively optimizes single-token preferences at logical pivots. Crucially, by concentrating intervention solely on logic-critical transitions, our framework achieves a favorable accuracy--efficiency trade-off compared to global inference time scaling methods like beam search and self-consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20560v1">LLM StructCore: Schema-Guided Reasoning Condensation and Deterministic Compilation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 16 pages, 1 figure, 5 tables. Preprint of a paper accepted to the Third Workshop on Patient-oriented Language Processing (CL4Health), co-located with LREC-COLING 2026
    </div>
    <details class="paper-abstract">
      Automatically filling Case Report Forms (CRFs) from clinical notes is challenging due to noisy language, strict output contracts, and the high cost of false positives. We describe our CL4Health 2026 submission for Dyspnea CRF filling (134 items) using a contract-driven two-stage design grounded in Schema-Guided Reasoning (SGR). The key task property is extreme sparsity: the majority of fields are unknown, and official scoring penalizes both empty values and unsupported predictions. We shift from a single-step "LLM predicts 134 fields" approach to a decomposition where (i) Stage 1 produces a stable SGR-style JSON summary with exactly 9 domain keys, and (ii) Stage 2 is a fully deterministic, 0-LLM compiler that parses the Stage 1 summary, canonicalizes item names, normalizes predictions to the official controlled vocabulary, applies evidence-gated false-positive filters, and expands the output into the required 134-item format. On the dev80 split, the best teacher configuration achieves macro-F1 0.6543 (EN) and 0.6905 (IT); on the hidden test200, the submitted English variant scores 0.63 on Codabench. The pipeline is language-agnostic: Italian results match or exceed English with no language-specific engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20553v1">DeepParse: Hybrid Log Parsing with LLM-Synthesized Regex Masks</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Modern distributed systems produce massive, heterogeneous logs essential for reliability, security, and anomaly detection. Converting these free-form messages into structured templates (log parsing) is challenging due to evolving formats and limited labeled data. Machine-learning-based parsers like Drain are fast but accuracy often degrades on complex variables, while Large Language Models (LLMs) offer better generalization but incur prohibitive inference costs. This paper presents DeepParse, a hybrid framework that automatically mines reusable variable patterns from small log samples using an LLM, then applies them deterministically through the Drain algorithm. By separating the reasoning phase from execution, DeepParse enables accurate, scalable, and cost-efficient log structuring without relying on brittle handcrafted rules or per-line neural inference. Across 16 benchmark datasets, DeepParse achieves higher accuracy in variable extraction (97.6% average Parsing Accuracy) and better consistency than both heuristic and LLM-only baselines. Integrating DeepParse into an anomaly detection pipeline reduced false alarms by over 30% and reduced inference latency by 36% compared to heuristic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13899v2">Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Instruction-tuned LLMs can annotate thousands of instances from a short prompt at negligible cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be labelled at once? We investigate both questions on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labelled, 5,000 human-annotated), comparing seven annotation strategies across four encoders to detect anti-immigrant hostility. A classifier trained on 25,974 GPT-5.2 labels (\$43) achieves comparable F1-Macro to one trained on 3,800 human annotations (\$316). Active learning offers little advantage over random sampling in our pre-enriched pool and delivers lower F1 than full LLM annotation at the same cost. However, comparable aggregate F1 masks a systematic difference in error structure: LLM-trained classifiers over-predict the positive class relative to the human gold standard. This divergence concentrates in topically ambiguous discussions where the distinction between anti-immigrant hostility and policy critique is most subtle, suggesting that annotation strategy should be guided not by aggregate F1 alone but by the error profile acceptable for the target application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20937v1">Sink-Token-Aware Pruning for Fine-Grained Video Understanding in Efficient Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) incur high inference latency due to a large number of visual tokens provided to LLMs. To address this, training-free visual token pruning has emerged as a solution to reduce computational costs; however, existing methods are primarily validated on Multiple-Choice Question Answering (MCQA) benchmarks, where coarse-grained cues often suffice. In this work, we reveal that these methods suffer a sharp performance collapse on fine-grained understanding tasks requiring precise visual grounding, such as hallucination evaluation. To explore this gap, we conduct a systematic analysis and identify sink tokens--semantically uninformative tokens that attract excessive attention--as a key obstacle to fine-grained video understanding. When these sink tokens survive pruning, they distort the model's visual evidence and hinder fine-grained understanding. Motivated by these insights, we propose Sink-Token-aware Pruning (SToP), a simple yet effective plug-and-play method that introduces a sink score to quantify each token's tendency to behave as a sink and applies this score to existing spatial and temporal pruning methods to suppress them, thereby enhancing video understanding. To validate the effectiveness of SToP, we apply it to state-of-the-art pruning methods (VisionZip, FastVid, and Holitom) and evaluate it across diverse benchmarks covering hallucination, open-ended generation, compositional reasoning, and MCQA. Our results demonstrate that SToP significantly boosts performance, even when pruning up to 90% of visual tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20523v1">Early-Stage Product Line Validation Using LLMs: A Study on Semi-Formal Blueprint Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 The 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26), March 23--27, 2026, Thessaloniki, Greece DOI: 10.1145/3748522.3779903
    </div>
    <details class="paper-abstract">
      We study whether Large Language Models (LLMs) can perform feature model analysis operations (AOs) directly on semi-formal textual blueprints, i.e., concise constrained-language descriptions of feature hierarchies and constraints, enabling early validation in Software Product Line scoping. Using 12 state-of-the-art LLMs and 16 standard AOs, we compare their outputs against the solver-based oracle FLAMA. Results show that reasoning-optimized models (e.g., Grok 4 Fast Reasoning, Gemini 2.5 Pro) achieve 88-89% average accuracy across all evaluated blueprints and operations, approaching solver correctness. We identify systematic errors in structural parsing and constraint reasoning, and highlight accuracy-cost trade-offs that inform model selection. These findings position LLMs as lightweight assistants for early variability validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20503v1">FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 14 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD) is a widely used approach for accelerating decode-heavy LLM inference workloads. While online inference workloads are highly dynamic, existing SD systems are rigid and take a coarse-grained approach to SD management. They typically set the speculative token length for an entire batch and serialize the execution of the draft and verification phases. Consequently, these systems fall short at adapting to volatile online inference traffic. Under low load, they exhibit prolonged latency because the draft phase blocks the verification phase for the entire batch, leaving GPU computing resources underutilized. Conversely, under high load, they waste computation on rejected tokens during the verification phase, overloading GPU resources. We introduce FASER, a novel system that features fine-grained SD phase management. First, FASER minimizes computational waste by dynamically adjusting the speculative length for each request within a continuous batch and by performing early pruning of rejected tokens inside the verification phase. Second, FASER breaks the verification phase into frontiers, or chunks, to overlap them with the draft phase. This overlap is achieved via fine-grained spatial multiplexing with minimal resource interference. Our FASER prototype in vLLM improves throughput by up to 53% and reduces latency by up to 1.92$\times$ compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.03624v2">LLAMADRS: Evaluating Open-Source LLMs on Real Clinical Interviews--To Reason or Not to Reason?</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel on many NLP benchmarks, but their behavior on real-world, semi-structured prediction remains underexplored. We present LlaMADRS, a benchmark for structured clinical assessment from dialogue built on the CAMI corpus of psychiatric interviews, comprising 5,804 expert annotations across 541 sessions. We evaluate 25 open-source models (standard and reasoning-augmented; 0.6B--400B parameters) and generate over 400,000 predictions. Our results demonstrate that strong open-source LLMs achieve item-level accuracy with residual error below clinically substantial thresholds. Additionally, an Item-then-Sum (ItS) strategy, assessing symptoms individually through discrete LLM calls before synthesizing final scores, significantly reduces error relative to Direct Total Score (DTS) prediction across most model architectures and scales, despite reasoning models attempting similar decomposition in the reasoning traces of their DTS predictions. In fact, we find that performance gains attributed to "reasoning" depend fundamentally on prompt design: standard models equipped with structured task definitions and examples match reasoning-augmented counterparts. Among the latter, longer reasoning traces correlate with reduced error; while higher model scale does across both architectures. Our results clarify when and why reasoning helps and offer actionable guidance for deploying LLMs in semi-structured clinical assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20490v1">Break the Optimization Barrier of LLM-Enhanced Recommenders: A Theoretical Analysis and Practical Framework</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-enhanced recommendation models inject LLM representations into backbone recommenders to exploit rich item text without inference-time LLM cost. However, we find that existing LLM-enhanced methods significantly hinder the optimization of backbone models, resulting in high training losses that are difficult to reduce. To address it, we establish a comprehensive theoretical analysis of local optimization curvature and identify two key causes: 1) large norm disparity and 2) semantic-collaboration misaligned angular clustering of LLM representations. Guided by these insights, we propose Training-Friendly LLM-Enhanced Recommender (TF-LLMER), a lightweight framework with two key components. First, we highlight the necessity of item embedding normalization to eliminate norm-driven instability and achieve provable control over optimization conditioning. Second, we introduce Rec-PCA, a recommendation-aware dimensionality reduction method that injects collaborative structure into the representation transformation to resolve semantic-collaboration misaligned angular clustering. It jointly optimizes semantic information retention and alignment with an item-item co-occurrence graph constructed from interaction histories. The graph captures collaborative structure, and alignment is promoted by penalizing total variation over the graph. Both theory and extensive experiments demonstrate that TF-LLMER significantly outperforms state-of-the-art methods. Our code is available at https://github.com/woriazzc/TF-LLMER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12911v3">Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Generating SQLs from user queries is a long-standing challenge, where the accuracy of initial schema linking significantly impacts subsequent SQL generation performance. However, current schema linking models still struggle with missing relevant schema elements or an excess of redundant ones. A crucial reason for this is that commonly used metrics, recall and precision, fail to capture relevant element missing and thus cannot reflect actual schema linking performance. Motivated by this, we propose enhanced schema linking metrics by introducing a \textbf{restricted missing indicator}. Accordingly, we introduce \textbf{\underline{K}n\underline{a}psack optimization-based \underline{S}chema \underline{L}inking \underline{A}pproach (KaSLA)}, a plug-in schema linking method designed to prevent the missing of relevant schema elements while minimizing the inclusion of redundant ones. KaSLA employs a hierarchical linking strategy that first identifies the optimal table linking and subsequently links columns within the selected table to reduce linking candidate space. In each linking process, it utilizes a knapsack optimization approach to link potentially relevant elements while accounting for a limited tolerance of potentially redundant ones. With this optimization, KaSLA-1.6B achieves superior schema linking results compared to large-scale LLMs, including DeepSeek-V3 with the state-of-the-art (SOTA) schema linking method. Extensive experiments on Spider and BIRD benchmarks verify that KaSLA can significantly improve the SQL generation performance of SOTA Text2SQL models by substituting their schema linking processes. The code is available at https://github.com/DEEP-PolyU/KaSLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07798v3">Lightweight LLM Agent Memory with Small Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 Accepted by ACL 2026 (main)
    </div>
    <details class="paper-abstract">
      Although LLM agents can leverage tools for complex tasks, they still need memory to maintain cross-turn consistency and accumulate reusable information in long-horizon interactions. However, retrieval-based external memory systems incur low online overhead but suffer from unstable accuracy due to limited query construction and candidate filtering. In contrast, many systems use repeated large-model calls for online memory operations, improving accuracy but accumulating latency over long interactions. We propose LightMem, a lightweight memory system for better agent memory driven by Small Language Models (SLMs). LightMem modularizes memory retrieval, writing, and long-term consolidation, and separates online processing from offline consolidation to enable efficient memory invocation under bounded compute. We organize memory into short-term memory (STM) for immediate conversational context, mid-term memory (MTM) for reusable interaction summaries, and long-term memory (LTM) for consolidated knowledge, and uses user identifiers to support independent retrieval and incremental maintenance in multi-user settings. Online, LightMem operates under a fixed retrieval budget and selects memories via a two-stage procedure: vector-based coarse retrieval followed by semantic consistency re-ranking. Offline, it abstracts reusable interaction evidence and incrementally integrates it into LTM. Experiments show consistent gains across model scales, with an average F1 improvement of about 2.5 over A-MEM on LoCoMo, while achieving higher efficiency and low median latency (83 ms for retrieval and 581 ms end-to-end).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20460v1">CCTVBench: Contrastive Consistency Traffic VideoQA Benchmark for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Safety-critical traffic reasoning requires contrastive consistency: models must detect true hazards when an accident occurs, and reliably reject plausible-but-false hypotheses under near-identical counterfactual scenes. We present CCTVBench, a Contrastive Consistency Traffic VideoQA Benchmark built on paired real accident videos and world-model-generated counterfactual counterparts, together with minimally different, mutually exclusive hypothesis questions. CCTVBench enforces a single structured decision pattern over each video question quadruple and provides actionable diagnostics that decompose failures into positive omission, positive swap, negative hallucination, and mutual-exclusivity violation, while separating video versus question consistency. Experiments across open-source and proprietary video LLMs reveal a large and persistent gap between standard per-instance QA metrics and quadruple-level contrastive consistency, with unreliable none-of-the-above rejection as a key bottleneck. Finally, we introduce C-TCD, a contrastive decoding approach leveraging a semantically exclusive counterpart video as the contrast input at inference time, improving both instance-level QA and contrastive consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20930v1">SafeRedirect: Defeating Internal Safety Collapse via Task-Completion Redirection in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 13 pages, 4 figures, 3 tables. Code: https://github.com/fzjcdt/SafeRedirect
    </div>
    <details class="paper-abstract">
      Internal Safety Collapse (ISC) is a failure mode in which frontier LLMs, when executing legitimate professional tasks whose correct completion structurally requires harmful content, spontaneously generate that content with safety failure rates exceeding 95%. Existing input-level defenses achieve a 100% failure rate against ISC, and standard system prompt defenses provide only partial mitigation. We propose SafeRedirect, a system-level override that defeats ISC by redirecting the model's task-completion drive rather than suppressing it. SafeRedirect grants explicit permission to fail the task, prescribes a deterministic hard-stop output, and instructs the model to preserve harmful placeholders unresolved. Evaluated on seven frontier LLMs across three AI/ML-related ISC task types in the single-turn setting, SafeRedirect reduces average unsafe generation rates from 71.2% to 8.0%, compared to 55.0% for the strongest viable baseline. Multi-model ablation reveals that failure permission and condition specificity are universally critical, while the importance of other components varies across models. Cross-attack evaluation confirms state-of-the-art defense against ISC with generalization performance at least on par with the baseline on other attack families. Code is available at https://github.com/fzjcdt/SafeRedirect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20389v1">CyberCertBench: Evaluating LLMs in Cybersecurity Certification Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      The rapid evolution and use of Large Language Models (LLMs) in professional workflows require an evaluation of their domain-specific knowledge against industry standards. We introduceCyberCertBench, a new suite of Multiple Choice Question Answering (MCQA) benchmarks derived from industry recognized certifications. CyberCertBench evaluates LLM domain knowledgeagainst the professional standards of Information Technology cybersecurity and more specializedareas such as Operational Technology and related cybersecurity standards. Concurrently, we propose and validate a novel Proposer-Verifier framework, a methodology to generate interpretable,natural language explanations for model performance. Our evaluation shows that frontier modelsachieve human expert level in general networking and IT security knowledge. However, theiraccuracy declines in questions that require vendor-specific nuances or knowledge in formalstandards, like, e.g., IEC 62443. Analysis of model scaling trend and release date demonstratesremarkable gains in parameter efficiency, while recent larger models show diminishing returns.Code and evaluation scripts are available at: https://github.com/GKeppler/CyberCertBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08948v2">TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real-world practical capabilities. To address this issue, we introduce TaxPraBen, the first dedicated benchmark for Chinese taxation practice. It combines 10 traditional application tasks, along with 3 pioneering real-world scenarios: tax risk prevention, tax inspection analysis, and tax strategy planning, sourced from 14 datasets totaling 7.3K instances. TaxPraBen features a scalable structured evaluation paradigm designed through process of "structured parsing-field alignment extraction-numerical and textual matching", enabling end-to-end tax practice assessment while being extensible to other domains. We evaluate 19 LLMs based on Bloom's taxonomy. The results indicate significant performance disparities: all closed-source large-parameter LLMs excel, and Chinese LLMs like Qwen2.5 generally exceed multilingual LLMs, while the YaYi2 LLM, fine-tuned with some tax data, shows only limited improvement. TaxPraBen serves as a vital resource for advancing evaluations of LLMs in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20316v1">R2IF: Aligning Reasoning with Decisions via Composite Rewards for Interpretable LLM Function Calling</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Function calling empowers large language models (LLMs) to interface with external tools, yet existing RL-based approaches suffer from misalignment between reasoning processes and tool-call decisions. We propose R2IF, a reasoning-aware RL framework for interpretable function calling, adopting a composite reward integrating format/correctness constraints, Chain-of-Thought Effectiveness Reward (CER), and Specification-Modification-Value (SMV) reward, optimized via GRPO. Experiments on BFCL/ACEBench show R2IF outperforms baselines by up to 34.62% (Llama3.2-3B on BFCL) with positive Average CoT Effectiveness (0.05 for Llama3.2-3B), enhancing both function-calling accuracy and interpretability for reliable tool-augmented LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20310v1">Odor Maps from the LLM-derived similarity scores</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 9 pages, 7 figures, Under review
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) to OdorSpace analysis attracts growing interest. Recent studies have explored the comparison of sensory evaluation spaces derived from LLMs with odor character profiles in the Dravnieks' dataset. In this study, we calculated pairwise distances of odor descriptors using three distance measures and statistically compared these LLM-derived similarities with distances derived from the original data. Next, we extended this approach to odor names (ingredients). Statistical comparison revealed that LLMs can infer odor similarity to some degree, suggesting the potential of odor maps generated from these similarity data. Applying this approach, we generated an odor map of essential oils. It demonstrates that essential oils within the same group are closely located in the odor map, suggesting that the proximity in the odor map corresponds to human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00667v4">zkCraft: Prompt-Guided LLM as a Zero-Shot Mutation Pattern Oracle for TCCT-Powered ZK Fuzzing</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 36 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Zero-knowledge circuits enable privacy-preserving and scalable systems but are difficult to implement correctly due to the tight coupling between witness computation and circuit constraints. We present zkCraft, a practical framework that combines deterministic, R1CS-aware localization with proof-bearing search to detect semantic inconsistencies. zkCraft encodes candidate constraint edits into a single Row-Vortex polynomial and replaces repeated solver queries with a Violation IOP that certifies the existence of edits together with a succinct proof. Deterministic LLM-driven mutation templates bias exploration toward edge cases while preserving auditable algebraic verification. Evaluation on real Circom code shows that proof-bearing localization detects diverse under- and over-constrained faults with low false positives and reduces costly solver interaction. Our approach bridges formal verification and automated debugging, offering a scalable path for robust ZK circuit development.
    </details>
</div>
