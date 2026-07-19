# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12621v1">Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Recent work has shown that "Vision-Free'' approaches (representing images as text) can be effective for standard image retrieval tasks. However, it remains unclear whether this paradigm can effectively handle a more complex, multimodal task, Composed Image Retrieval (CIR), due to the inherent information loss in textual descriptions. In this paper, we introduce a Vision-Free CIR framework that addresses this challenge through two key techniques: (1) Attribute-Augmented Hybrid Scoring, which compensates for lost visual details via explicit attribute matching, and (2) LLM-Based Reranking, which verifies semantic consistency of top candidates. Experiments on the open-domain CIRR dataset show that our approach outperforms existing Zero-shot CIR methods (44.04% R@1, +8.79%). On FashionIQ, our results highlight the trade-off between semantic reasoning and fine-grained visual matching. Ablation studies reveal that both attribute-augmented scoring and LLM-Based Reranking consistently improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12507v1">When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      LLM-assisted reverse-engineering (RE) systems analyze strings, decompiler output, and tool reports derived from ttacker-controlled binaries. A binary can make data look like instructions or records from one origin look like independent evidence. We call such failures Representation-Confusion Attacks in Reverse Engineering (RARE): the pipeline promotes a correctly extracted observation to instruction authority, claim-validating evidence, or trusted analysis state without the authority or support that role requires. RARE-Bench measures these failures with behavior-checked clean and adversarial binaries. After an exploratory 11,520-call study, we test RARE-Guard's authorization and evidence controls on 20 new programs and two models. Without runtime controls, the models propose a planted unsafe action in 35/40 adversarial cases and 0/40 clean cases. When binary-derived content is shown only as data (Data-Only rendering), they still make 15 unsafe proposals. Tool Authorization denies all 15 and authorizes all 40 matched analyst requests. On identical report drafts, Support Gate validates 23/40 false claims by counting records from one origin separately. Provenance Gate groups those records before counting support, validates 0/40 false claims, and retains all 40 supported claims. We then instrument Ghidra, r2pipe, and angr on 16 further programs. In a preselected eight-program subset, no single-tool draft reaches Support Gate's validation threshold for the false claim. In fused drafts across all 16 programs, Support Gate validates 32/32 false claims. Provenance Gate prevents validation of all 32 and retains all 32 supported claims. A deterministic renderer prevents downgraded claims from reappearing in the final report. Binary-derived content may therefore guide analysis without gaining authority over tools, and views from several tools do not necessarily provide independent evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12490v1">When is LLM-Based Program Reasoning Correct? A Completion Semantics for LLM-Based Code Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 28 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Due to token and cognitive limits, Large Language Models (LLMs) typically perform program reasoning over incomplete code fragments/prompts rather than complete programs. Such reasoning therefore must rely on {assumptions about omitted code and context. As a result, the meaning of an inference over a program fragment is not absolute, but depends on an implicit completion model describing how the fragment may be refined into a complete program. In this paper, we introduce completion semantics for LLM-based program reasoning. We formalize incomplete programs as denoting a space of possible refinements and define the correctness of existential inferences relative to a completion model. Under this view, a reported bug is correct whenever there exists a completion within the model that witnesses the bug. This perspective explains why many LLM-generated reports are neither simply correct nor incorrect, but instead depend on assumptions about omitted context. We have instantiated our approach in the form of a witness-generation workflow that concretizes completions underlying an inference by constructing executable refinements of the original program fragment. Witnesses serve both as evidence for existential claims and as a mechanism for exposing the assumptions required to support them. We evaluate our approach on real-world LLM-generated bug reports and program-analysis tasks. Our results show that witness generation effectively distinguishes inferences supported by plausible completions from those requiring unrealistic assumptions, providing a practical mechanism for validating reasoning over incomplete programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15222v2">PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often generate functionally correct code, but their ability to produce efficient implementations for performance-critical systems tasks remains limited. Existing code benchmarks mainly emphasize correctness or algorithmic problem solving, while realistic systems-level optimization is still underexplored. To address this gap, we introduce PerfCodeBench, an executable benchmark for evaluating LLMs on high-performance code optimization. The tasks require system-level implementation choices, hardware-aware optimization, and careful handling of performance bottlenecks. Each task includes executable correctness checks, a baseline implementation, and a reference optimized solution. This allows us to evaluate both correctness and runtime-oriented efficiency. Our evaluation on a broad set of state-of-the-art LLMs shows a clear gap between model-generated code and expert-optimized implementations. The gap is especially large on tasks involving parallelism and GPU operations. Current models also show weaknesses in cross-language robustness and in consistently reaching expert-level efficiency. These results suggest that performance-aware evaluation are still needed. LLMs should move beyond generating merely correct code toward producing efficient systems software. We submit the benchmark data, evaluation infrastructure, and complete logs of all LLMs-generated code at https://anonymous.4open.science/r/perfcodebench-7CDE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12468v1">An Omnilingual-ASR-Based Speech-LLM System for the 2nd MLC-SLM Challenge</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted to INTERSPEECH 2026. 4 pages + references. Technical description of our 2nd MLC-SLM Challenge Task 1 submission
    </div>
    <details class="paper-abstract">
      We describe our submission to Task 1 of the 2nd MLCSLM Challenge: a cascaded diarization-then-recognition system that combines DiariZen-Large-s80 (WavLM-Large) segmentation, CAM++ embedding-based two-speaker clustering, and a LoRA-adapted omniASR LLM 7B v2 recognizer, with no oracle segmentation or speaker labels at test time. On the official Development set (150 conversations, 21 language/accent categories) the system attains a macro tcpMER of 29.27%, versus 79.15% for the official baseline; on the Evaluation set it scores 50.23%. We also analyze two engineering choices that substantially affect tcpMER. First, embedding-based speaker clustering outperforms an end-to-end-style alternative that assigns speakers from ASR <sc> turn markers alone. Second, overlap-aware segmentation, although intended to raise diarization recall, increases tcpMER because overlapped speech is transcribed twice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12467v1">Understanding before Naming! Enhancing LLM-based Method Name Prediction with Code Summarization</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Method names are critical to software quality, affecting code comprehensibility, maintainability, and developer collaboration. However, manually designing meaningful method names is challenging. Method Name Prediction (MNP), which automatically generates method names from code snippets, has recently attracted attention. Although large language models (LLMs) show promising performance for MNP, two challenges remain. First, existing evaluations mainly rely on token similarity metrics, which often fail to reflect human judgments of semantic quality. Second, current LLM-based MNP methods usually generate names through direct code-to-name mapping, which differs from the human process of understanding functionality before naming. To address these challenges, we conduct empirical studies on LLM-based evaluation and MNP strategies. We compare 6 metric-based evaluators, 5 LLM-based evaluators, and 6 human evaluators. Results show that LLM-based evaluators, especially DeepSeek-based evaluators, are more consistent with human judgments than traditional metrics. We further compare direct generation and summarization-and-refinement strategies. Results indicate that summarization and refinement generally improve the semantic quality of generated names. Case studies reveal three limitations: inaccurate summaries, semantic misalignment, and close semantic scores. Based on these findings, we propose SMNP, an MNP approach combining MNP-oriented summarization and chain-of-thought enhanced refinement. Experiments on 5 LLMs and 2 datasets demonstrate the effectiveness and robustness of SMNP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12406v1">Isolation as a First-Class Principle for LLM-Agent System Safety: Concepts, Taxonomy, Challenges and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      The capability of LLM agents to function as the ``brain'' of a system fundamentally expands the scope of analysis beyond a standalone model. Consequently, safety is no longer only about input--output content alignment. It also concerns system behavior and real-world execution outcomes. However, the current literature is fragmented across attack types, applications, and benchmarks. This makes it hard to explain why failures such as prompt injection, tool misuse, and memory poisoning often share the same structural cause, and how they spread through an agent workflow. In this survey, we treat isolation as a first-class principle for LLM-agent system safety. By isolation, we refer to the separation of user inputs, tool access, execution channels, inter-agent communication, and environment-originated context. We organize the literature with a boundary-centric taxonomy of five boundaries: user-agent, agent-tool, agent-execution, agent-agent, and system-environment. This view helps identify where the loss of isolation first occurs, how compromise propagates across boundaries, and which defenses are most relevant at each interface. We also summarize cross-boundary failure paths, discuss open challenges, and outline a research agenda for isolation-by-construction in future agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24787v2">ReLope: KL-Regularized LoRA Probes for Multimodal LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Routing has emerged as a promising strategy for balancing performance and cost in large language model (LLM) systems that combine lightweight models with powerful but expensive large models. Recent studies show that \emph{probe routing}, which predicts the correctness of a small model using its hidden states, provides an effective solution in text-only LLMs. However, we observe that these probes degrade substantially when applied to multimodal LLMs (MLLMs). Through empirical analysis, we find that the presence of visual inputs weakens the separability of correctness signals in hidden states, making them harder to extract using standard probe designs. To address this challenge, we introduce two complementary approaches for improving probe routing in MLLMs. First, we propose the \emph{Attention Probe}, which aggregates hidden states from the preceding layer based on attention scores to recover distributed correctness signals. Second, we present the \emph{KL-Regularized LoRA Probe (ReLope)}, which inserts a lightweight LoRA adapter and applies a KL regularizer to learn routing-aware representations. Comprehensive experiments show that our methods consistently outperform baselines, suggesting that improving the quality of hidden states is key to effective routing in MLLMs. Our code is available at https://github.com/Spinozaaa/ReLope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12397v1">Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      LLM agents act in external environments where each action changes the state that later decisions condition on, and where a single wrong step can waste interaction budget or trigger irreversible side effects long before the final failure is observed. Reliable deployment therefore requires \emph{step-level confidence estimation}: a calibrated probability that each proposed action is productive, available \emph{before} the action is executed. Existing LLM confidence estimators are designed to score a response from the given prompt, but agent confidence also depends on execution consequences: whether similar actions in similar situations actually advanced the task after the environment responded. We introduce the \method (\methodshort), a self-evolving critic framework in which an LLM critic accumulates evidence from its own past judgments and their observed consequences. After each trajectory, a hindsight LLM that sees the full execution feedback votes on whether each step was productive. The resulting pseudo-labels populate a memory bank from which related productive and unproductive experiences are retrieved into the critic's prompt whenever a similar step recurs. \methodshort requires no training and uses no ground truth step labels. Across three agent benchmarks and three critic backbones, \methodshort attains the best calibration (ECE and Brier) and ranking (AUC) in every dataset--critic combination, reducing ECE by up to $54\%$ relative to the strongest training-free baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10609v5">iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted by SIGIR2026
    </div>
    <details class="paper-abstract">
      Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored due to the scarcity of need-to-modify itinerary data. To bridge this gap, we formally define the itinerary modification task and propose a general pipeline to construct the corresponding dataset, namely iTIMO. This pipeline frames the generation of need-to-modify itinerary data as an intent-driven perturbation task. It instructs large language models to perturb real-world itineraries using three operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents: disruptions of popularity, spatial distance, and category diversity. Furthermore, hybrid evaluation metrics are introduced to ensure perturbation effectiveness. We conduct comprehensive benchmarking on iTIMO to analyze the capabilities and limitations of state-of-the-art LLMs. Overall, iTIMO provides a comprehensive testbed for the modification task, and empowers the evolution of traditional travel recommender systems into adaptive frameworks capable of handling dynamic travel needs. Dataset, code and supplementary materials are available at https://github.com/zelo2/iTIMO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12385v1">PM-Bench: Evaluating Prospective Memory in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Published as a conference paper at COLM 2026
    </div>
    <details class="paper-abstract">
      A significant challenge in agentic AI is prospective memory: the ability to execute an intention at a specific future cue or state while other activities are ongoing. We introduce PM-Bench, a text-based benchmark for measuring prospective memory capabilities in modern LLM agents. Inspired by the Virtual Week paradigm from cognitive science, PM-Bench evaluates how well LLM agents maintain user intentions, execute delayed intentions, and monitor latent environment changes. Over the course of a simulated seven-day week, agents must continue an ongoing activity while deciding whether any deferred task is due. We compare eight state-of-the-art LLMs on PM-Bench under eight different agent configurations. PM-Bench proves challenging across all settings: the best method, a GPT-5.4 agent, reaches only 65.1\% F1 score under our evaluation. Furthermore, no single strategy for improving prospective memory dominates across models. We release PM-Bench as a controlled testbed for diagnosing these failures and developing training or inference-time interventions that support reliable prospective behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00011v2">SkillSelect-Serve: QoS-Aware Budgeted Skill Service Recommendation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 18 pages (14-page main text + appendices), 5 figures
    </div>
    <details class="paper-abstract">
      Reusable agent skills are emerging as a service-oriented capability layer for Large Language Model (LLM) agents. Unlike plain retrieval items, a skill exposes functional capabilities, input-output assumptions, tool dependencies, context cost, and risk metadata. Selecting skills is particularly challenging for small LLM agents, which can load only a few capability units under restricted context, tool availability, and risk tolerance. Existing fixed Top-k methods rank skills by textual relevance and overlook requirement satisfaction, deliverability, and operational constraints. We present SkillSelect-Serve, a QoS-aware, budget-constrained Skill Service recommendation framework. Raw skills are profiled as structured Skill Services, the task is converted into a structured requirement object, and candidates discovered from a large-scale registry are ranked by a calibrated task-conditioned suitability estimator and packed by a constrained projection enforcing token-budget, aggregated-risk, and tool-availability constraints, using only deployment-observable features. On a registry of 35,353 skills with pooled multi-positive relevance judgments verified by two independent assessors, the unconstrained top-5 recommendation fits a realistic 4,000-token context for only 9.1% of tasks; the constrained projection restores 100% deliverability at a cost of only 1.14 points of hit rate, outperforming retrieve-and-rerank, budget truncation, and diversity-based selection under identical budgets. The same mechanism halves delivered risk exposure and eliminates the 44-81% tool-violation rates of tool-agnostic recommendation. At an identical three-service budget, hit rate improves from 0.8864 to 0.9091 over fixed Top-3 retrieval. The results support managing reusable agent skills as discoverable, comparable, and constraint-aware service units instead of plain retrievable documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12340v1">Skills That Don't Exist: A Large-Scale Study of Hallucinated Skill Recommendation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      LLM agents acquire new capabilities by downloading skills from open registries. Instead of browsing these catalogs manually, developers typically ask the agent to recommend and install a skill. This convenience hides a risk: agents frequently invent names for skills that exist in no registry. We term this flaw skill name hallucination. A fake name may seem harmless, but it opens the door to supply-chain attacks. Because registries rarely verify publishers, an adversary can prompt the agent, collect the fake names it returns, pre-register malicious skills under them, and wait for a victim to install the payload. We conducted the first large-scale measurement of skill name hallucination, evaluating 15,000 prompts across 12 configurations (4 standalone LLMs and 8 agents). We conservatively counted a name as hallucinated only if it was missing from all live registries and GitHub. The results reveal a systemic vulnerability: every configuration hallucinates. Rates average 36.0% for standalone LLMs and 36.9% for agents, rising to 43.1% on real-world developer questions. In total, the systems generated 5,669 distinct hallucinated names. Crucially, these names are not random noise. Agents repeat the same fake names across prompts and models, giving attackers highly reliable targets to hijack. Finally, we tested four model-level defenses and found a severe conflict between security and usability. The strongest, retrieval grounding, cut the hallucination rate from 40.8% to 3.2% but crippled usefulness: even the best-defended system recommended the correct skill only about one in six times. Skill name hallucination is thus a highly exploitable vulnerability requiring minimal attacker effort. Fixing it cannot rely on prompt engineering or model tuning alone. It demands ecosystem-wide structural changes: registry-level name reservations and verified recommendation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12279v1">A Shared Subcircuit Lets LLMs Count Down Across Tasks</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Writing a sentence of exactly twelve words; ending a DNA sequence at the right codon; formatting an ASCII table. These are all tasks that language models can do that requires tracking how many tokens remain before a target. In this work, we identify in Llama-3.1-70B-Instruct a general mechanism for performing these tasks: a "countdown subcircuit" that compares the current position to a goal length and estimates the time remaining until then. We first isolate a countdown subcircuit in a controlled setting, in which the model is tasked with writing a fixed-length sentence ending in a specified word. We then investigate the geometry of the representations used by the subcircuit, and find that the subcircuit uses an identical motif previously identified in a frontier LLM on a separate task, thus suggesting that this motif is shared across models. Finally, we use unsupervised probing on a natural language dataset to find a variety of other tasks where this subcircuit is used, including tasks where the goal length is inferred from context rather than explicitly stated. Our work suggests that reverse-engineering subcircuits allows us to understand how behaviors generalize from a single example to many different tasks and even models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12273v1">Code-MUE: Measuring Code LLMs' Uncertainty through Execution-based Semantic Interaction Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 To appear at The ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA) 2026
    </div>
    <details class="paper-abstract">
      As Code Large Language Models (LLMs) become central to modern software engineering, their inherent stochasticity poses significant real-world risks, where even minor errors can lead to severe functional, security, or safety consequences. Reliable automation, therefore, demands the ability to distinguish between confident, well-supported predictions and stochastic guessing. However, existing uncertainty estimation methods face a critical gap: white and grey-box techniques are often inapplicable to closed-source models, while standard "black-box" text metrics fail to capture the unique fragility of code, where syntactic variation does not always imply semantic divergence. To bridge this syntax-semantics gap, we introduce Code-MUE, a purely black-box framework that measures uncertainty through execution-based Semantic Interaction Graphs. Unlike prior approaches that rely on superficial textual similarity, Code-MUE grounds uncertainty in observable runtime behavior, calculating the Von Neumann entropy of the solution space to quantify global semantic diversity. A large-scale empirical study across eight state-of-the-art LLMs demonstrates that Code-MUE achieves a strong negative correlation with functional correctness (Spearman's correlation up to -0.98), significantly outperforming lexical and embedding-based baselines while enabling robust risk detection and selective prediction in practical workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25112v2">Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 11 pages, 4 figures, 3 tables. v2 replaces the meta-d'/M-ratio analysis with a model-free measure (meta-I_2r); see the version note on page 1. Pre-registered; code and data at https://github.com/synthiumjp/sdt_calibration
    </div>
    <details class="paper-abstract">
      Standard evaluation of LLM confidence relies on calibration metrics (ECE, Brier score) that conflate how much a model knows (Type-1 accuracy) with how well its confidence signal tracks that knowledge (Type-2 metacognitive sensitivity). We apply Signal Detection Theory (SDT) to decompose these capacities, treating token-level normalised log-probability as a graded confidence variable and answer correctness as the state to be discriminated. We characterise the Type-2 ROC of this signal, including its unequal-variance structure via z-ROC analysis, and -- because the meta-d' efficiency ratio is not well defined for open-ended QA, which lacks a two-alternative Type-1 decision -- quantify metacognitive efficiency with a model-free information measure, normalised metacognitive information (meta-I_2r). Applied to four LLMs (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base, Gemma-2-9B-Instruct) across 224,000 factual QA trials, we find: (1) metacognitive information varies more than two-fold across models and co-varies inversely with accuracy -- the least accurate model has the most informative confidence -- though with four models this ordering cannot be separated from an error-difficulty confound, so we report it as coupling, not decoupling; (2) the confidence signal has model-specific unequal-variance structure (z-ROC slopes 0.81 to 1.18) invisible to calibration metrics; (3) metacognitive information is domain-specific, strongest in Arts & Literature for every model; (4) temperature dissociates Type-1 accuracy from metacognitive information, which stays stable while accuracy shifts. All estimates carry permutation nulls and bootstrap confidence intervals. Pre-registered; code and data public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06715v2">Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 V1 accepted to ACL SRW 2026. V2 updates experiments
    </div>
    <details class="paper-abstract">
      We ask whether topic sentiment has a causal effect on perceived political ideology, and whether the answer depends on who assigns the ideology label. Using articles from AllSides, paired with shared sentiment annotations from Llama-3.3-70b-versatile, we compare ideology labels from expert human annotators, GPT-4o-mini (baseline and finetuned), and Llama-3.3-70B. We apply Double Machine Learning (DML) and mediation analysis across all four annotation paradigms. Zero-shot LLMs regularly inflate effect sizes relative to human annotations, while fine-tuning often attenuates them back toward the human scale. Our results have implications for the use of LLM annotations as silver labels and as proxies for human judgment in downstream causal analyses: they may be reliable for recovering the presence and direction of effects on the partisan topics, but not their magnitude, leading to over- or under-prediction of some ideology given particular topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12236v1">Speculate with Memory: Lossless Acceleration for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Speculative execution accelerates LLM agents by using a smaller, cheaper model to predict and pre-launch the next step while the environment is idle. However, existing speculators are stateless and discard all information between tasks, preventing prediction quality from improving with experience. We equip the speculator with three online memory systems that learn from past agent trajectories: a contrastive transition table tracking action-sequence statistics, an episodic memory retrieving contextually similar segments, and a confusion tracker suppressing recurring errors. We evaluate this approach on six benchmarks spanning three speculation types: action prediction, observation prediction, and chained prediction. Memory-augmented speculation yields a 19--39\% relative accuracy improvement on action prediction and up to a $2.5\times$ increase on observation prediction tasks with repetitive action spaces. These gains grow continuously as memory accumulates and generalize across speculator models of varying cost. All speculation is lossless because it runs during idle time at zero added wall-clock cost, and the actor's trajectory is identical to non-speculative execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10383v2">Authoring for Living Worlds: Tool-Constrained LLM Agents for Executable Multi-Actor Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      We use LLM agents to author executable specifications for a living world: formal Graphs of Events in Space and Time (GESTs) that a 3D game engine executes deterministically into multi-actor narrative videos, with per-frame spatial, temporal, and semantic ground truth as a byproduct of execution. This inverts the dominant paradigm of LLM agents driving neural video generators, which emit pixels with no semantic guarantees and no annotations. Authoring is the hard problem: the world's capability registry cannot be enumerated in a context window, validity of an action depends on accumulated world state, and a staged refinement pipeline driving GPT-5 through six validated stages produced zero executable specifications in 50 attempts. Our hierarchical Director / Scene Builder architecture instead operates through a constraint-enforcing tool layer, in which exploration tools paginate the registry and building tools validate every operation against simulator state, so every emitted specification is executable by construction. Driving a far smaller model (Claude Haiku 4.5), the system executes 20 of 25 attempts (80%) when seeded with a target narrative text. Because each seed text derives from a source graph, we can measure how faithfully the agent reconstructs specified intent: event-level F1 reaches 0.83 against a 0.55 matched-random floor, and sequential structure 0.77 against 0.43, with the residual gap dominated by information the text itself drops.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12682v2">RAFP: Identifying LLM Lineages via Rare-Region Fingerprints</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly released under restricted licenses, creating a growing need for robust model ownership verification. Existing fingerprinting methods are often fragile under downstream finetuning, require invasive training modifications, or fail in black-box settings. We introduce RAFP, a robust framework for identifying LLM lineages via rare-region fingerprints. Our key insight is that downstream finetuning primarily updates common high-density language behaviors, while low-probability prompt regions receive weak optimization signal and limited gradient alignment under finetuned distribution. As a result, rare prompt-response behaviors remain stable across common model adaptations. RAFP is non-invasive, constructing fingerprints via discrete gradient-based optimization over rare prompts without modifying model weights. We provide a theoretical analysis showing that the likelihood change of rare-region fingerprints under finetuning remains bounded. Experiments across four LLM families and multiple downstream adaptations, including supervised finetuning, LoRA, quantization, prompt-template variation, and decoding changes, show that RAFP achieves strong fingerprint persistence and substantially outperforms prior fingerprinting baselines in black-box settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12233v1">Fin-Analyst at FinMMEval 2026 Task 3: A Live Hybrid Trading Agent with LLM Specialists and Rule-Based Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 14 pages, 7 tables, 1 figure. CLEF 2026 FinMMEval Task 3 Working Notes
    </div>
    <details class="paper-abstract">
      Large language model (LLM) trading agents show promising performance in equity markets, yet remain narrowly focused on US equities with little evidence from live deployment. We present Fin-Analyst, a hybrid agent for FinMMEval 2026 Task 3: an eight-specialist LLM pipeline over news, SEC filings, fundamentals, analyst forecasts, technical indicators, and social sentiment, aggregated by a Meta-Agent for Tesla (TSLA), and a lightweight rule based three-signal vote for Bitcoin (BTC). On the final official leaderboard (accessed 2026-07-05), Fin-Analyst ranks first of all agents on TSLA with a +13.51% return, +28.33 points over Buy-and-Hold (Sharpe 4.10, 88% win rate), while the BTC vote ends flat yet well above a sharply falling baseline. Relative to the interim performance, the asset ranking reversed, indicating that short live windows yield volatility-sensitive rankings. Ablation identifies event-driven 8-K disclosures as the most influential TSLA signal. Error analysis shows that the memoryless agents repeat wrong calls for days at a time, and that the fixed-threshold BTC rules lost money by trading on noise in a sideways market while the LLM pipeline gained under similar conditions, motivating a memory-aware, LLM-based successor for both assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17253v5">PDAGENT-BENCH: Characterizing, Grounding, and Architecting LLM/VLM Agents for VLSI Physical Design</a></div>
    <div class="paper-meta">
      📅 2026-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models and vision-language models have shown remarkable success in the front-end design of Very Large-Scale Integrated Circuits, yet their capabilities for VLSI physical design remain significantly underexplored. The primary cause is the lack of standardized benchmarks for evaluating agentic physical design workflows that require high-dimensional, multi-stage optimization under strict design constraints, coordinated interaction with diverse Electronic Design Automation tools, and iterative refinement. This work introduces PDAGENT-BENCH, a comprehensive and multi-dimensional benchmark for evaluating LLM/VLM-based agents across the physical design stack. PDAGENT-BENCH integrates both task-level assessment and workflow-level execution. The benchmark suite contains 353 curated problems that combine conceptual questions with real-world industrial artifacts, with expert-validated references and executable solutions. In addition, the benchmark provides a unified, human-aligned agentic physical design workflow framework that enables closed-loop evaluation of holistic physical design in realistic EDA environments. Experiments on 11 state-of-the-art models reveal that while modern LLMs/VLMs perform competitively on conceptual tasks, they remain substantially limited in tool-centric execution (e.g., 42.2% on Innovus script generation) and long-horizon, multi-stage reasoning. Our studies further show that human-skill-enhanced agentic workflows significantly improve end-to-end physical design performance. PDAGENT-BENCH establishes a standardized, reproducible, and realistic evaluation framework for advancing LLM/VLM-driven holistic physical design automation. To ensure full reproducibility and broad accessibility, we will release PDAgent-Bench together with its agentic workflow framework, instantiated on open-source PDKs (e.g., Nangate45, ASAP7) and open EDA tools (e.g., OpenROAD).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12216v1">RCWT: Measuring Task-Budget Displacement from Coordination Content in LLM Calls</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Multi-agent and memory-augmented LLM systems often place coordination content, shared state, prior discussion, tool outputs, summaries, and role instructions, inside the same finite prompt used for the current task. This creates a practical allocation problem: every token spent on coordination is unavailable to task instructions or evidence when a call is assembled under a fixed context budget. We introduce the Roundtable Context Window Test (RCWT), a controlled protocol for measuring this task-budget displacement effect. RCWT varies coordination content while controlling total budget, position order, task family, and scoring. In the main context-dependent recall task at $W=4096$, three commercial models remain near baseline through moderate overhead and then degrade sharply once residual reference evidence falls to a few hundred tokens. Window-scaling summaries are consistent with a task-specific residual-budget interpretation rather than a fixed percentage threshold, but we treat this as descriptive evidence rather than a universal law. To test whether the fixed-budget cliff persists when task evidence remains intact, we add an intact-task ablation: the full task/reference block is kept present while coordination tokens increase by expanding total prompt length. In that setting, all tested calls return every scored field correctly across GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash up to a 95\% coordination ratio. This ablation narrows the claim: the main RCWT cliff is best read as task-budget displacement, not as proof that coordination volume alone causes semantic interference in the original open-ended task. RCWT is therefore a measurement primitive for context-allocation budgeting, not a complete theory of multi-agent benefit or session-level coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04689v3">Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 ICML 2026 Spotlight
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) typically requires thousands of benchmark items, making the process expensive, slow, and increasingly impractical at scale. Existing evaluation protocols rely on average accuracy over fixed item sets, treating all items as equally informative despite substantial variation in difficulty and discrimination. We introduce ATLAS, an adaptive testing framework based on Item Response Theory (IRT) that estimates model ability using Fisher information-guided item selection. ATLAS reduces the number of required items by up to 90% while maintaining measurement precision. For instance, it matches whole-bank ability estimates using only 41 items (0.157 MAE) on HellaSwag (5,600 items). We further reconstruct accuracy from ATLAS's ability estimates and find that reconstructed accuracies closely match raw accuracies across all five benchmarks, indicating that ability $θ$ preserves the global performance structure. At the same time, $θ$ provides finer discrimination within accuracy-equivalent models: among more than 3,000 evaluated models, 23-31% shift by more than 10 rank positions, and models with identical accuracies receive meaningfully different ability estimates. Code and calibrated item banks are available at https://github.com/Peiyu-Georgia-Li/ATLAS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19387v2">Interpretable and Verifiable Hardware Generation with LLM-Driven Stepwise Refinement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in software development. However, they are susceptible to hallucinations, meaning that they can introduce subtle semantic and logical errors. Due to the high stakes in chip design and manufacturing, hardware engineers are still reluctant to rely on LLMs for register-transfer level (RTL) generation. In this paper, we propose a hardware generation framework that combines the creativity and broad knowledge of LLMs with the explainability and mathematical rigor of formal methods. Specifically, we devise a set of transformation rules that cover various design decisions and hardware features. By iteratively applying these rules, an LLM agent can convert a design specification into an RTL program with guaranteed correctness. Experimental results demonstrate the effectiveness and efficiency of the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12188v1">Cost-Governed RAG: Unified Per-Tenant Cost Attribution Across Retrieval and Generation in Multi-Tenant LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Enterprise Retrieval-Augmented Generation (RAG) deployments face a critical governance gap: while LLM generation cost is metered per token, the retrieval layer - vector memory, similarity compute, and embedding API calls - remains an unattributed shared cost, enabling invisible cross-subsidization among tenants. We present Cost-Governed RAG, an architecture that integrates a codebook-oblivious vector index (TurboVec) with a multi-tenant LLM governance gateway, creating a unified observability stack where embedding, retrieval, and generation costs are jointly attributable per tenant. The architecture exploits TurboVec's deterministic, closed-form memory formula to enable near-exact per-tenant retrieval cost calculation - a property unavailable in graph-based indexes with non-linear memory overhead. Deployed on Snowpark Container Services within a cloud data platform's governance boundary, the system achieves 99.96% end-to-end cost attribution accuracy across 100 simulated tenants (10M vectors, log-normal size distribution) with telemetry overhead below 0.04% of query latency. The architecture reduces retrieval infrastructure cost by 3.1-9.0x compared to managed vector database services under the pricing assumptions detailed in Section IV. We formalize a three-layer cost model and demonstrate that codebook-oblivious quantization enables deterministic per-tenant cost attribution while also removing the shared-codebook leakage surface present in trained quantizers - the latter observation being exploratory and subject to the limitations described in Section VII.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12339v2">SheetMind: An End-to-End LLM-Powered Multi-Agent Framework for Spreadsheet Automation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      We present SheetMind, a modular multi-agent framework powered by large language models (LLMs) for spreadsheet automation via natural language instructions. In this paper, we introduce a hierarchical agentic system consisting of three specialized agents: Manager Agent that decomposes complex user instructions into subtasks; an Action Agent that translates these into structured commands using a Backus-Naur Form (BNF) grammar; and a Reflection Agent that validates alignment between generated actions and the user's original intent. We evaluate SheetMind on the 221-task SheetCopilot Benchmark with GPT-3.5-Turbo. SheetMind achieved 100% execution success and 54.8% functional correctness, exceeding SheetCopilot (44.3%) while maintaining perfect execution reliability. We also conduct ablation study on a separately curated dataset to confirm that the full three-agent configuration consistently outperforms all partial variants. Lastly, we integrate our system into Google Sheets via a Workspace extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27443v2">When Does Personality Composition Matter for Multi-Agent LLM Teams?</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to COLM 2026 (20 pages, 6 figures)
    </div>
    <details class="paper-abstract">
      Personality prompting shapes how large language models communicate, yet whether these behavioral shifts affect objective task outcomes remains under-explored. Prior work shows that agents prompted with low agreeableness produce adversarial language, while those prompted with high agreeableness become cooperative, but the relationship between communication style and task performance has not been systematically examined across multiple domains. In this work, we investigate whether personality composition matters for multi-agent team performance by manipulating personality traits across frontier LLMs on three task domains: structured coding, open-ended research collaboration, and competitive bargaining. We find that personality effects depend critically on task structure. In coding tasks, low agreeableness leads to large communication shifts that have little effect on milestone completion. In open-ended collaboration and bargaining, the same manipulation substantially degrades performance. We discuss implications for multi-agent system design and the limits of personality manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12148v1">MindReader: Using LLMs to Encourage Memorable and Secure Password Replacement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      We report on the design and evaluation of MindReader, a tool that helps a user replace her password when she is required to do so. Left to their own devices, users tend to replace their previous passwords with predictable variations of the original ones. MindReader leverages LLMs to suggest password variations that are chosen to be easy for the user to remember but harder for an attacker to predict. To do this, MindReader infers the meaning behind original password components and then suggests semantically related (yet syntactically unrelated) components for the new password. In a user study, passwords created using MindReader were more secure than both replacement passwords created without using MindReader and original passwords. In particular, MindReader replacement passwords were harder to guess in an online attack than alternative replacement passwords even by an attacker with knowledge of the original password and full knowledge of the tool implementation. Passwords created with MindReader were also comparably memorable to alternative replacement passwords and original passwords, as measured by the ability of users to successfully log in a week after creating their password.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15239v4">Modeling Story Expectations: A Generative Framework using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Consumers' engagement with stories is shaped by their expectations about what will happen next, yet modeling these forward-looking beliefs over unstructured narrative content has remained challenging. We develop a framework that uses large language models to approximate consumers' story expectations. Our method generates multiple imagined story continuations from a pre-trained LLM and extracts interpretable, theory-motivated features from these continuations, such as emotion and narrative path features. We propose two complementary validation procedures suited to different data availability: a survey-based approach that compares LLM-derived expectations to human-reported beliefs, and a rational-expectations approach that compares them to actual story outcomes. Applying the framework to both survey data collected in a controlled lab setting and observational data from an online reading platform, we find that LLM-derived expectations correlate with human-reported beliefs as well as actual story continuations along all features studied. In both settings, forward-looking expectations are associated with reader engagement above and beyond features of the content already consumed. Our framework provides a scalable method for modeling consumer beliefs about narrative content, with implications for content creation, platform strategy, and the study of narrative media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01311v3">Catalyst-Agent: Autonomous heterogeneous catalyst screening with an LLM Agent</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      The discovery of catalysts for electrochemical applications such as the oxygen reduction reaction (ORR), nitrogen reduction reaction (NRR), and CO2 reduction reaction (CO2RR) remains a central challenge in chemistry and materials science. Machine-learning interatomic potentials (MLIPs) and graph neural network models now accelerate individual adsorption-energy calculations by orders of magnitude relative to density functional theory. However, true large-scale screening is still blocked by human decisions: selecting candidates, constructing slabs, enumerating adsorption sites, interpreting descriptor failures, and choosing follow-up modifications. Here, we introduce Catalyst-Agent, a Model Context Protocol (MCP) server-based, LLM-powered agent that autonomously coordinates closed-loop catalyst screening. Catalyst-Agent searches materials databases through OPTIMADE, constructs slabs, computes adsorption energies using Meta FAIRchem's UMA MLIP within AdsorbML, evaluates reaction-specific descriptors, and applies structural modifications to refine near-miss candidates. In ORR, NRR, and CO2RR campaigns, Catalyst-Agent demonstrates high performance and converges in 1.40-3.41 trials per successful material on average. It identified Sn3Sc, Sn3Y, Tl3La, Pb3Y and In3Y as CO2RR candidates for further validation that were not previously reported in the literature. DFT single-point checks confirmed screening outcomes for representative NRR and CO2RR candidates. Ablations show these gains arise from chemically informed candidate selection and feedback-directed modification rather than brute-force evaluation: fully randomized screening dropped to 13.3%, 16.7%, and 0% success for ORR, NRR, and CO2RR, respectively. These results show that tool-grounded LLM agents can shift catalyst screening from manual trial-and-error toward more autonomous, reproducible and adaptive workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16726v2">Bridging Individual and Collective Realism in LLM-Based Human Mobility Simulation via Mobility Scaling-Law Guidance</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Geospatial applications such as urban planning, epidemic forecasting, and transportation demand modeling depend on individual mobility data, but such data are costly to collect, uneven in coverage, and privacy-sensitive. Human mobility simulation offers a scalable alternative. A recent line of work treats large language models (LLMs) as human agents, modeling individual cognitive processes to generate realistic trajectories. Yet because each agent is simulated in isolation, these methods provide no population-level coordination mechanism, and the collective regularities of real mobility - how trip distances, visited locations, and flows distribute across a population - fail to emerge. We close this gap with COMPASS, which turns empirical mobility scaling laws into a feedback signal that guides prompt construction. COMPASS starts from coarse, population-level adjustments driven by these scaling laws and progressively refines them into individual prompts, jointly satisfying multiple aggregate objectives while keeping individual trajectories realistic. Across two public datasets, COMPASS outperforms state-of-the-art LLM-based simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16727v2">Mobility-Aware Cache Framework for Scalable LLM-Based Human Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Simulating large-scale human mobility is fundamental to understanding population movement patterns and supporting real-world geospatial applications such as urban planning, epidemic response, and transportation analysis. Recent works treat large language models (LLMs) as human agents to simulate realistic mobility behaviors using structured reasoning, but their high computational cost limits scalability. To address this, we design a mobility-aware cache framework named MobCache that leverages reconstructible caches to enable efficient large-scale human mobility simulations. It consists of: (1) a reasoning component that encodes each reasoning step as a latent-space embedding and uses a latent-space evaluator to enable the reuse and recombination of reasoning steps; and (2) a decoding component that employs a lightweight decoder trained with mobility law-constrained distillation to translate latent-space reasoning chains into natural language, thereby improving simulation efficiency while maintaining fidelity. Experiments show that MobCache significantly improves efficiency across multiple dimensions while maintaining performance comparable to state-of-the-art LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12089v1">Cross-Cutting Security Analysis of LLM-Generated Code via Metamorphic Testing and Association Rule Mining</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 This work has already been accepted by IEEE 27th International Conference on Information Reuse and Integration for Data Science
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate code with security vulnerabilities, yet these weaknesses are rarely isolated: they often span multiple concern areas simultaneously, reflecting the cross-cutting nature of security in software. We present a framework that combines security-oriented Metamorphic Relations (MRs) with Association Rule (AR) mining to detect vulnerabilities in LLM-generated code, uncover their co-violation structure, and trace that structure back to prompt-level risk factors. We define nine MRs covering major CWE categories, including SQL injection, XSS, command injection, path traversal, hard-coded credentials, weak cryptography, and memory-safety errors, and apply them using an LLM-based judge to 3,700 code snippets generated by five open models from the LLMSecEval benchmark. The results show that 68.8% of snippets violate at least one MR, with hard-coded credentials (79.1%) and command injection (74.4%) among the most prevalent applicable failures. AR mining reveals strong cross-cutting co-violation patterns, notably that XSS and weak cryptography co-violations predict hard-coded credentials with 82.5% confidence (lift = 3.23), along with tightly coupled clusters linking authentication, credential handling, and cryptographic weakness, as well as input-handling and memory-safety failures. We then perform prompt-level risk analysis and find that database- and authentication-related prompts are strong predictors of broad cross-cutting insecurity, while 65.5% of prompts yield consistent violation outcomes across all five models. These findings show that insecure code generation is not merely a collection of independent defects, but a structured and prompt-conditioned phenomenon, motivating cluster-aware verification and prompt-level intervention for safer LLM-assisted programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12086v1">CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent LLM-based multi-agent urban simulators can generate semantically rich city routines, but they remain costly to scale and are often weakly validated against empirical mobility patterns. We present CityBehavEx, an interactive LLM-assisted urban simulation platform that scales to city-size populations, exposes agent behavior for inspection, supports empirical validation, and generates mobility patterns that better match real-world spatial, temporal, and semantic distributions. Instead of invoking large language models for every agent action, CityBehavEx combines established human mobility models with fine-tuned cross-encoders that estimate semantic alignment between agent profiles, schedules, and activity transitions. This design enables large-scale simulations, as demonstrated in a case study of 100,000 agents over 75 days in under one hour on a single consumer GPU. The platform allows users to define simulation regions, launch experiments, inspect trajectories and activity traces, debug unrealistic behaviors, and validate generated routines against real-world mobility, time-use, and semantic metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10125v3">D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      As one of the key tools in many security tasks, decompilers reconstruct human-readable source code from binaries. Yet, despite recent advances, their outputs often suffer from syntactic and semantic errors and remain difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals their problems, such as introducing new errors and relying on unreliable accuracy validation. In this paper, we present D-LIFT, an enhanced decompiler-LLM pipeline with a fine-tuned LLM using code quality-aware reinforcement learning. Unlike prior work that overlooks preserving accuracy, D-LIFT adheres to a key principle for enhancing the quality of decompiled code: preserving accuracy while improving readability. Central to D-LIFT, we propose D-Score, an integrated code quality assessment system to score the decompiled source code from multiple aspects, and use it to guide reinforcement learning fine-tuning and to select the best output during inference. In line with our principle, D-Score assigns low scores to any inaccurate output and only awards higher scores for readability to code that passes the accuracy check. Our implementation, based on Ghidra and a range of LLMs, demonstrates significant improvements for the accurate decompiled code from the coreutils and util-linux projects. Compared to baseline LLMs without D-Score-driven fine-tuning, our trained LLMs produce 55.3% more improved decompiled functions, as measured by D-Score. Overall, D-LIFT improves the quality of 68.2% of all the functions produced by the native decompiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12050v1">EFLUX: Elastic Multi-Robot Formation Navigation and Adaptation with Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Multi-robot teams operating in confined or cluttered environments must adapt both their formation geometry and group topology to navigate through complex obstacles. This adaptation requires two complementary behaviors: deformation, where the team continuously reshapes its geometry while remaining connected, and reconfiguration, where robots split into subgroups or merge back into a single formation. Existing methods often model these behaviors independently, connect them through handcrafted rules, or lack explicit geometric criteria for determining when each behavior should be invoked. However, challenging environments may require online changes in formation shape, connectivity, and effective team composition, making decoupled or rule-based approaches prone to suboptimal trajectories and deadlock. We propose EFLUX, a geometry-grounded LLM agentic framework for automatic and elastic multi-robot formation navigation. EFLUX extracts a structured scene representation and uses an LLM to reason jointly over both deformation actions, such as scaling and shearing, and reconfiguration actions, such as splitting and merging. These strategies are then translated into executable per-robot waypoints through a closed-loop generation, verification, and correction pipeline. Simulation and hardware experiments show that EFLUX enables safe, continuous, and elastic formation navigation in constrained environments, reducing deadlock and navigation failures compared with baselines while maintaining coherent multi-robot coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11881v1">Metacognition in LLMs: Foundations, Progress, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Metacognition is a foundational component of intelligence critical to effective learning, problem solving, decision-making, communication, and more. In recent years, it has become increasingly recognized as a cornerstone of capable, transparent AI systems. Yet while LLMs have made significant progress across diverse real-world tasks, it is not yet clear when, how, or to what extent they can exhibit or be endowed with effective metacognitive abilities, nor how such abilities can be adapted to advance the fundamental capabilities, reliability, and intelligence of AI systems. This paper bridges this gap by presenting the first comprehensive overview of the current state of knowledge on metacognition for LLMs. We analyze and taxonomize the landscape of this emerging field and summarize recent technical advancements, including methods and benchmarks to measure and evaluate LLMs' metacognitive abilities, techniques to elicit, improve, and apply metacognition in LLMs, and findings and implications of ongoing research. We also discuss applications, open questions and challenges, and promising directions for future work. Our aim is to provide a detailed and up-to-date review of this topic and stimulate meaningful research and discussion. An organized list of papers can be found at https://github.com/yale-nlp/LLM-Metacognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11871v1">Inside the Unfair Judge: A Mechanistic Interpretability Account of LLM-as-Judge Bias</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 58 pages, 13 figures, 30 tables; project page: https://xzx34.github.io/unfair-judge/
    </div>
    <details class="paper-abstract">
      Existing studies of LLM-as-judge scoring bias work predominantly at the input-output level: they perturb inputs, measure score deltas, and propose prompt-level mitigations. We argue that the same biases admit a representation-level account in the judge's hidden state, complementary to the input-output view and operationally useful in ways it does not afford. We report three findings, across seven judges, seven bias types, and nine benchmarks. Geometry: baseline judging inputs occupy a tight activation manifold while biased inputs are displaced along a low-dimensional, type-specific subspace that sharpens with depth and is recovered consistently by three families of estimators. Causal control: steering hidden states along this subspace drives scoring in both directions, forward shifts reproducing biased scoring on clean inputs and reverse shifts restoring baseline scoring on biased ones, while matched-norm random directions produce shifts an order of magnitude smaller. Operational: a simple linear projection onto the same bias-direction features anticipates judge failures on three entirely unseen benchmarks, substantially outperforming text-based alternatives. Reading bias as activation geometry, rather than as input-output noise, unifies geometric structure, causal control, and operational prediction within a single framework. The project page is available at https://xzx34.github.io/unfair-judge/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23276v2">Auditable Context-Aware HFMD Forecasting with Structured LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Effective HFMD surveillance requires forecasts capturing both time-series patterns and contextual drivers such as school calendars, weather, and policy or surveillance reports. In clinical settings, forecasts must be trusted and actionable; thus, beyond point accuracy, decision-makers require concise, auditable explanations of why risk is expected to rise or fall. Classical models (e.g., ARIMA and Prophet) and foundation models (e.g., Chronos, Moirai, and TimesFM) treat external covariates as numerical inputs, lacking semantic reasoning to reflect epidemiological mechanisms or resolve conflicting signals. We propose a two-agent neuro-symbolic framework that decouples contextual interpretation from probabilistic forecasting. An LLM-based Event Interpreter ingests heterogeneous signals -- school schedules, weather summaries, government reports, and clinical guidelines -- and outputs a scalar transmission-impact signal. A Forecast Generator combines this signal with historical case counts to produce point forecasts that are mapped to probabilistic predictions through Poisson/negative-binomial moment matching. We focus on one-week-ahead rolling forecasts, aligning with weekly hospital-capacity planning and the rapid, context-driven inflections typical of HFMD. We evaluate on two datasets: Hong Kong surveillance (90 target weeks in 2023--2024) and Lishui hospital visits (33 target weeks in 2024). Against traditional and foundation-model baselines, our approach achieves competitive point accuracy while providing robust 90\% intervals (coverage approximately 0.85--1.00) and concise rationales. This demonstrates that integrating domain knowledge through LLM-based agents can match strong numerical forecasters while yielding interpretable, context-aware forecasts aligned with public-health decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11810v1">Supporting Reflection in LLM-based Exploratory Search</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can make exploratory search more efficient but may undermine the reflection and iterative sensemaking needed in unfamiliar domains. Existing LLM tools often prioritize rapid answers over supporting users in tracking how their understanding evolves and how well their strategies align with their goals. We present TrailLM, a system that helps users reconstruct and revisit their exploration paths to support reflection and metacognitive engagement during information seeking. By aligning LLM assistance with users' sensemaking workflows, TrailLM aims to preserve the benefits of LLM-based search while enhancing opportunities for critical reflection on one's own search process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13088v1">Securing LLMs in the Wild: Privacy and Security Challenges at the Edge</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly moving from research settings into the wild, deployed on enterprise infrastructure, personal devices, and edge platforms. While cloud deployments offer scalable compute, concerns over data sovereignty, compliance, latency, and third-party dependence are driving organizations toward edge and on-premise LLMs. This shift introduces new security and privacy challenges: limited compute and memory force aggressive optimizations, including quantization, pruning, model partitioning, and parameter-efficient adaptation, each of which can introduce vulnerabilities and reshape the threat landscape. We describe this tension as the Security-Efficiency Paradox, mechanisms that improve efficiency may weaken robustness, expose new attack surfaces, or increase privacy risks. We examine how compression can degrade safety alignment, how partitioned inference enables reconstruction attacks, and how continuous local adaptation may cause privacy leakage and model drift. To analyze these risks, we introduce a deployment-centric taxonomy organized around three architectural constraints: the Memory Wall, the Quadratic Wall, and the Compute Wall. We derive a unified constraint model that quantifies when unsafe optimizations become unavoidable, linking each wall to specific attack surfaces. Building on this model, we propose the Secure Operational Efficiency Score (SOES), a holistic metric balancing task accuracy, jailbreak resistance, and privacy against energy, memory, and latency, enabling practitioners to configure edge LLMs under real-world hardware limits. We further present a practical decision procedure and targeted mitigations for each optimization-induced vulnerability. Together, these contributions provide a co-designed framework for jointly evaluating security, privacy, and efficiency, laying a foundation for securing edge-native intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11706v1">VoxENES 2026: Benchmarking Generalization of Speech Spoofing Detectors Against LLM-Era TTS and Voice Conversion</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted in InterSpeech 2026
    </div>
    <details class="paper-abstract">
      Modern LLM-driven text-to-speech (TTS) and voice conversion (VC) systems produce synthetic speech that differs from the generators represented in many legacy spoofing benchmarks. This mismatch creates a temporal generalization gap that can overestimate detector robustness under real-world post-processing conditions. We bridge this gap by introducing VoxENES 2026, a bilingual (English and Spanish) benchmark of 53,628 audio samples generated using 10 contemporary speech synthesis methods and evaluated under 10 standardized post-processing conditions. Using VoxENES 2026, we benchmark eight pretrained detectors without fine-tuning and observe substantial performance degradation: the best model achieves 28.98\% EER overall, while most perform near or below random chance across modern generators and perturbations. Our results highlight the reliance on brittle artifacts in current detectors and establish VoxENES 2026 as a practical testbed for developing robust audio spoofing countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11703v1">Production and Perception in LLMs: A Token Probability Approach</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      The asymmetry between language production and perception has been well-documented in psycholinguistics. Whether large language models (LLMs) exhibit a functionally analogous distinction remains an open question, particularly given that LLMs rely on the same underlying mechanism (next-token prediction) for both input and output processing. In this exploratory study, we operationalize the production-perception distinction through direct token probability measurements rather than metalinguistic prompting. Using the base Llama-3.1-8B model, we generated poems under a production prompt and re-scored the same tokens under both rephrased production prompts and perception-oriented prompts. Across an extended experiment with four production and three perception prompts, production-perception distances consistently and substantially exceeded production-production distances, with non-overlapping ranges across conditions and an overall average ratio of approximately 1.8. Near-ceiling correlations in the production-production control confirm that the effect is specific to communicative framing rather than prompt surface variation, and we show the effect replicates across five open-weight models (Llama-3.1-8B, EuroLLM-9B, gemma-2-9b-it, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct), spanning both base and instruction-tuned variants. Temporal analysis revealed that the perception prompt exerts its strongest influence at the beginning of the sequence, with divergence decaying as generated context accumulates, though the specific shape of this decay varies across prompt pairs. These findings suggest that prompt framing alone induces a production-perception distinction in LLM probability distributions, even within a decoder-only architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11683v1">RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Graph retrieval-augmented generation (GraphRAG) enhances large language models with structured knowledge, yet existing systems construct knowledge graphs in a single extraction pass, producing noisy entities and brittle retrieval. RAGU, an open-source modular GraphRAG engine, addresses this by separating extraction from consolidation: entities and relations pass through two-stage typed extraction, DBSCAN-backed deduplication, LLM summarization, and Leiden community detection. A key insight motivates a compact extractor: the skills an in-pipeline LLM needs - comprehension, extraction, reasoning over context - are language skills that grow only weakly with model size, unlike factual world knowledge. Accordingly, we train Meno-Lite-0.1, a 7B model optimized for language skills, which outperforms Qwen2.5-32B on knowledge-graph construction (+12.5% relative harmonic mean) and matches it on English GraphRAG tasks. On GraphRAG-Bench (Medical), RAGU retrieves the most complete context at every factoid level (evidence recall up to 0.84 vs. $\leq$0.76) and overtakes HippoRAG2 on synthesis tasks; on multi-hop factoid QA, the apparent HippoRAG2 advantage is shown to be largely an answer-format artifact. RAGU is installable via $\texttt{pip install graph_ragu}$, runs on a single GPU, and is released under MIT. The source code is publicly available at https://github.com/RaguTeam/RAGU, and the Meno-Lite-0.1 model can be obtained from https://huggingface.co/bond005/meno-lite-0.1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11614v1">Extending LLM Context via Associative Recurrent Memory</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Extending the context length of large language models (LLMs) is critical for many real-world applications, yet standard transformers remain constrained by quadratic compute and linear memory scaling. In this work, we investigate the Associative Recurrent Memory Transformer (ARMT) as a practical approach for enabling long-context processing in LLMs, constant memory scaling, and better efficiency. We make three main contributions. First, we construct two domain-specific long-context datasets designed to evaluate realistic workloads, focusing on narrow-domain fine-tuning scenarios. Second, we propose a comprehensive training recipe for ARMT-based context extension, combining continued pre-training, synthetic long-context data generation, curriculum learning, and selective integration of associative memory into chosen model layers. Third, we present an extensive experimental study demonstrating that ARMT-augmented models: (i) process inputs well beyond their original context limits without degrading performance relative to in-limit baselines; (ii) generalize more effectively to out-of-distribution context lengths; and (iii) need 30% less FLOPs while preserving baseline performance within the original context window.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00262v3">LLM-Driven Cost-Effective Requirements Change Impact Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 33 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Requirements are inherently subject to changes throughout the software development lifecycle. Within the limited budget available to requirements engineers, manually identifying the impact of such changes on other requirements is both error-prone and effort-intensive. That might lead to overlooked impacted requirements, which, if not properly managed, can cause serious issues in the downstream tasks. Inspired by the growing potential of large language models (LLMs) across diverse domains, we propose ProReFiCIA, an LLM-driven approach for automatically identifying the impacted requirements when changes occur. We conduct an extensive evaluation of ProReFiCIA using several LLMs and prompts variants tailored to this task. Using the best combination of an LLM and a prompt variant, ProReFiCIA achieves a recall of 85.7% on an unseen industrial dataset, demonstrating its effectiveness in identifying impacted requirements. Further, the cost of applying ProReFiCIA remains small, as the engineer only needs to review the predicted impacted requirements, which represent 3.0% of the entire set of requirements. Lastly, incorporating domain knowledge into the model via RAG increases recall to 95.8% while slightly raising the cost to only 3.4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11591v1">Similarity-Guided Curriculum Fine-Tuning of LLMs for Neural Architecture Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Introduce a MinHash-based similarity scheduling framework that constructs a progressive curriculum over neural architecture code for LLM-based neural architecture search (NAS). Using 128-permutation MinHash signatures over normalised 7-gram source code shingles, we partition the reference pool into similarity bands and present them in increasing architectural heterogeneity, with the best LoRA adapter from each stage merged cumulatively into the backbone. We evaluate the framework on OlympicCoder-7B within the LEMUR benchmark on CIFAR-10 image classification, generating N =15 candidate architectures per epoch across six progressive fine-tuning steps. The curriculum achieves 60% peak success rate at the high-similarity level without post-processing repair. A 2*2 ablation at the most diverse level curriculum versus base model, with versus without partial interface repair reveals that without repair the base model (47% peak SR) substantially outperforms the curriculum model (7% SR), while adding partial repair brings both to 53% SR. This pattern is consistent with merge-level weight drift progressively erasing evaluator-interface priors, and suggests that interface repair and curriculum scheduling target distinct failure modes. We further report a cross-dataset transfer observation on SVHN, where direct base-model generation without curriculum warmup yields 27% peak SR at substantially lower accuracy (60.5%) than the CIFAR-10 equivalent, consistent with the increased synthesis difficulty of the unq-family anchor architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11573v1">Knowledge-Guided Synthetic Bug Feedback for LLM-Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 12 pages, 7 figures, 6 tables. Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for unit test generation, but executable tests do not necessarily reveal real defects. This paper studies how historical real-bug mechanisms can be transformed into executable feedback targets for LLM-based unit test generation. The proposed framework constructs structural and semantic representations of real-bug records, retrieves mechanisms applicable to a focal method, and instantiates them as synthetic bugs that guide iterative test enhancement. We evaluate the approach on method-level real-bug detection tasks from Defects4J and show that mechanism-guided synthetic-bug feedback improves real-bug detection over execution-, coverage-, mutation-, knowledge-, and search-based baselines. The results suggest that organizing real-bug mechanisms as retrievable and executable feedback targets is an effective way to guide generated tests toward bug-triggering inputs and behavioral oracles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11564v1">PaperRouter-Agent: A Content-Grounded LLM Agent for Personalized Hierarchical Paper Routing</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Researchers organize the papers they collect into personal folder hierarchies in reference managers, and route each new paper into the folder where it belongs. This task differs from standard hierarchical text classification. A user's folder hierarchy is not a fixed, shared taxonomy but a private and evolving folksonomy whose folder meanings may be topical, shorthand, venue-based, or process-oriented, and are often defined by the papers already stored inside them. We formalize this setting as personalized hierarchical paper routing (PHPR): assigning an incoming paper to folders in a user-specific hierarchy without per-user training. We propose PaperRouter-Agent, a training-free LLM agent that grounds routing decisions in folder members rather than folder names alone. The agent first narrows the candidate hierarchy, retrieves folder-specific evidence, verifies fit by inspecting member papers, and incorporates similarity-gated feedback from past user rejections. A formative study on real personal libraries shows that PaperRouter-Agent raises overall Recall@1 from 0.39 to 0.61 and Recall@3 from 0.57 to 0.83, with the largest gains on organizational folders defined by metadata such as venue or year, where single-shot methods collapses (Recall@1 0.09 to 0.50). On the public LaMP-2 benchmark, the same approach improves accuracy from 44.5% to 51.5% (+9.0 macro-F1) over a single-shot baseline, while remaining low-cost for practical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16395v3">LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Atomic commits, which address a single development concern, are a best practice in software development. In practice, however, developers often produce tangled commits that mix unrelated changes, complicating code review and maintenance. Prior untangling approaches (rule-based, feature-based, or graph-based) have made progress but typically rely on shallow signals and struggle to distinguish explicit dependencies (e.g., control/data flow) from implicit ones (e.g., semantic or conceptual relationships). In this paper, we propose ColaUntangle, a new collaborative consultation framework for commit untangling that models both explicit and implicit dependencies among code changes. ColaUntangle integrates Large Language Model (LLM)-driven agents in a multi-agent architecture: one agent specializes in explicit dependencies, another in implicit ones, and a reviewer agent synthesizes their perspectives through iterative consultation. To capture structural and contextual information, we construct Explicit and Implicit Contexts, enabling agents to reason over code relationships with both symbolic and semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C# and 14k Java tangled commits). Experimental results show that ColaUntangle outperforms the best-performing baseline, achieving an improvement of 44% on the C# dataset and 82% on the Java dataset. These findings highlight the potential of LLM-based collaborative frameworks for advancing automated commit untangling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11528v1">HermesHFL: Incentive-Compatible Hierarchical Federated Unlearning for Dynamic LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 17pages,8 figures
    </div>
    <details class="paper-abstract">
      Hierarchical federated unlearning (HFUL) for large language model (LLM) fine-tuning faces significant challenges due to hierarchical aggregation, dynamic client participation, and strong parameter coupling in LLM adaptation. Selectively removing client contributions is particularly difficult because model updates propagate across multiple aggregation stages while unlearning requests may coincide with client departures and rejoining. To address these issues, we propose **HermesHFL**, a hierarchical federated learning framework that supports selective unlearning, dynamic client participation, and client reintegration for scalable LLM fine-tuning via parameter-efficient fine-tuning (PEFT) with LoRA. We formulate a unified optimization problem that jointly models client participation, edge association, incentive allocation, and unlearning under heterogeneous client behaviors. To solve this problem efficiently, we develop **Neogen**, a neural-guided bilevel evolutionary optimization framework that combines CMA-ES for continuous incentive optimization with a CHC-based evolutionary mechanism for discrete participation and association decisions. A neural surrogate further accelerates optimization and improves search efficiency. Extensive experiments on LLM fine-tuning tasks demonstrate that HermesHFL consistently outperforms state-of-the-art baselines in model utility, unlearning effectiveness, convergence stability, and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11517v1">Graph-Based Structural Evaluation of LLM-Translated Adversary Emulation Procedures</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 This technical contribution supports the MITRE white paper titled: Evaluating LLMs for Impact-Faithful Translation of Adversary Behavior Across Operating Systems
    </div>
    <details class="paper-abstract">
      Adversary emulation plans describe multi-step attacker procedures using MITRE ATT&CK techniques, privilege requirements, and observable telemetry. Translating them across operating systems supports cross-platform defender evaluation, and large language models (LLMs) can automate this task. However, a translation may only rename tools while retaining source-platform logic, giving defenders little target-platform coverage. Binary scoring can overestimate fidelity because it measures countable features rather than structural, observable, and rule-level equivalence. Graph-Based Structural Evaluation (GBSE) models each procedure as a directed attributed graph and calculates normalized Graph Edit Distance (GED) across four layers: technique, tactic, telemetry class, and Sigma logsource. GBSE was applied to a 29-step ALPHV/BlackCat Windows-to-Linux plan, comparing a reconstructed Windows control with the unmodified LLM-generated Linux version. Technique and tactic structure were fully preserved (GED=0, similarity=1.000). Telemetry similarity fell to 0.897 (GED=3) because three steps contained unmapped or drifting observables, while Sigma logsource similarity was 1.000. Every state was classified as Medium Fidelity, with a best composite score of 0.674. The 0.80 deployment threshold was not reached because technical realism scored 0.43 against the required 0.990. The framework includes bipartite GED, a telemetry-intent parser that converts free text into observable classes, and 49 validated Sigma rules: 19 for Linux and 30 for Windows. The rules provide complete ATT&CK technique coverage and pass validation with zero findings. Additional analysis reveals technique-level divergence, including RDP-based external access mapped to unencrypted exfiltration and credential-store access mapped to remote-system discovery. Results were reproduced and verified against recorded outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11505v1">Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Post-training is essential for refining the domain-specific capabilities of large language models (LLMs), yet existing reward optimization and distribution matching methods tightly couple policy exploration with distribution alignment. This coupling forces expensive exploration directly on the policy model and severely hinders the asynchronous generation, reuse, and cross-model transfer of optimization signals. In this paper, we propose Proxy-guided Update Signal Transfer (PUST), a novel post-training framework that fundamentally decouples update-signal exploration from distribution alignment. Instead of utilizing the primary model for costly exploration, PUST employs a lightweight proxy model as an efficient testbed to discover high-reward behaviors. We extract the relative improvement signal between the proxy's initial and optimized states, transferring this directional update to the primary model to guide its policy alignment. This decoupled pipeline, comprising proxy exploration, update-signal extraction, and signal transfer, significantly reduces computational overhead and enables optimization signals to be asynchronously generated, cached, and reused. Crucially, by transferring relative improvements rather than absolute policy distributions, PUST naturally supports weak-to-strong improvement and seamless cross-model transfer. Systematic evaluations on Qwen3-family models across math and code domains demonstrate that update signals extracted from substantially weaker proxies can robustly and adjustably enhance stronger primary models. Ultimately, PUST transforms post-training from a monolithic online optimization process into a highly modular, reusable, and cost-efficient paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11471v1">Are LLMs ready for HardChoices?</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to Konvens 2026
    </div>
    <details class="paper-abstract">
      A lot of research attention has been devoted to checking whether large language models (LLMs) are politically biased. This work has largely focused on high-level ideological dimensions, such as left--right or progressive--conservative, and it has been shown that while LLMs are predominantly left and progressive leaning, largely mimicking the biases in the training data, they can be to some extent steered to change their preferences in post-training. In this short note, we check if LLMs have robust stances with regard to major substantive societal issues, on which members of the same ideological camp are often in disagreement, summarised in a novel dataset \textsc{HardChoices}. We show that, faced with this line of questioning, LLMs, both large and small, surprisingly rarely declare neutrality, are often incoherent, and demonstrate a remarkable degree of agreement on issues where they do take stances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11451v1">ManiScope: LLM-Assisted Visual Analytics of Cryptocurrency Manipulation Risk</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Cryptocurrency markets are vulnerable to trade-based manipulation, such as wash trading, which can distort price signals and mislead investors. Prior research has mainly focused on detecting manipulation using fixed rules or labeled examples, offering limited flexibility and interpretability for assessing potential risks. Existing visual analytics tools can reveal basic manipulation-related signals, such as token distribution, but still require substantial manual effort to integrate holder relationships, suspicious behaviors, and market dynamics for risk assessment. To address these limitations, we propose ManiScope, an LLM-assisted visual analytics system for analyzing trade-based manipulation risks in cryptocurrency markets. ManiScope provides coordinated views of token distributions, holder relationships, detailed holder behaviors, price dynamics, and suspicious trading patterns. To further enhance user analysis, ManiScope introduces a human-LLM collaborative visual analytics framework. Rather than acting as a basic reactive LLM assistant, the framework positions the LLM as a co-analyst that infers users' analytical intent and emerging hypotheses from interaction context and surfaces relevant visual, statistical, and synthesized evidence for hypothesis evaluation. This design reduces repetitive inspection and strengthens evidence-based reasoning. We evaluate ManiScope through two case studies and a user study with 12 experienced cryptocurrency practitioners. The results suggest that ManiScope supports effective risk assessment of manipulation, reduces manual effort in evidence-seeking, and organizes findings around user hypotheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11414v1">Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in financial applications fail most consequentially when they are confidently wrong. Hedged, uncertain answers invite scrutiny, whereas confident errors silently degrade downstream decisions without warning. We ask how reliably such confidently wrong answers, or confident hallucinations, can be detected from a model's internal activations, and whether those activations carry information beyond its observable outputs. We train linear probes on the residual stream and evaluate them on two established question-answering (QA) benchmarks built from real filings, FinQA and TAT-QA. Behavioral confidence is measured as the agreement among eight resampled answers to the same question, and probe effectiveness is compared against baselines, such as token log-probabilities and the model's own True/False self-assessment of its answer. Our findings show that among confident answers, those for which all eight resamples agree, 15-23% are wrong on FinQA. There the probes have a significant advantage over baseline methods in detecting hallucinations, holding 0.68-0.77 AUROC while the best baselines fall to 0.55-0.63, across Qwen3-8B, Llama-3.1-8B, and Gemma-2-9B. Our results suggest that probing can be a cost-effective triage mechanism for routing LLM answers to human review and quality control procedures in high-stakes financial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11390v1">TerraRepair: A Tool-Grounded LLM Agent for Infrastructure-as-Code Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 The paper has been peer reviewed and accepted for publication in the proceedings of the 20th International Symposium on Empirical Software Engineering and Measurement (ESEM 2026)
    </div>
    <details class="paper-abstract">
      Background: Infrastructure-as-Code (IaC) scanners detect cloud misconfigurations in Terraform and other IaC languages before deployment, but repairing the flagged configurations remains largely manual. Recent Large Language Model (LLM)-based repair approaches can repair some findings, but may hallucinate unsupported constructs or suppress warnings without fixing the issue. Aims: We study whether tool grounding can improve LLM-based Terraform repair, and when a finding should be escalated because the required deploymnet-specific context is not availble. Method: We present TerraRepair, a prototype of a tool-grounded LLM agent for Terraform repair with structured escalation. TerraRepair retrieves dependency context from Terraform references, consults the installed provider schema, and re-runs the scanner before returning a candidate repair. Then teh required context is absent, TerraRepair escalates instead of fabricating a plausible fix. Results: We evaluate our tool on two vulnerable-by-design Terraform repositories using two IaC security scanners, Checkov and Trivy, across AWS, Azure, and GCP. On the combined AWS benchmark, TerraRepair improves scanner-verified fix rates from 26.6% to 78.4% on Checkov and from 44.8% to 72.4% on Trivy, compared with a controlled one-shot baseline. It repairs are labelled as correct under a majority-vote protocol. Conclusions: These emerging results show that tool grounding can substantially improve scanner-verified LLM-based IaC repair on the studied benchmarks, while missing deployment-specific context remains the main knowledge boundary for full autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11363v1">Beyond Sally-Anne: Evaluating Theory of Mind in LLMs using Epistemic Schelling Points</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Text-based evaluations of Theory of Mind (ToM) in Large Language Models (LLMs) often involve cognitive tests akin to the Sally-Anne task that can be gamed due to exposure to relevantly similar tasks in pre-training and do not obviously test models' functional ToM abilities in ways that generalize to naturalistic settings. To address these issues, we introduce the Epistemic Asymmetry Schelling Task (EAST), a two-player dialogue game designed to benchmark robust and generalizable ToM abilities. By requiring LLM-LLM dyads to independently converge on semantic Schelling points under varying states of epistemic transparency, we evaluate whether models can robustly apply ToM to achieve coordination. Our results reveal a significant capability gap in functional social reasoning, with only frontier models successfully navigating the varying epistemic demands of the tasks. Analysis of reasoning traces shows that coordination failures are primarily driven by epistemic tracking errors, such as conflating private knowledge with mutual knowledge. Despite high performance on traditional static benchmarks, our study shows that robust social reasoning and epistemic tracking remain a critical bottleneck, providing concrete targets for future LLM evaluation and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11354v1">User Preference Induction with LLMs for Offline Top-N Recommendation Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Offline evaluation is the standard methodology for comparing top-N recommender systems, yet it relies on incomplete relevance information. In most benchmark datasets, only a small subset of user--item preferences is observed, and unjudged items are commonly treated as non-relevant. This missing-as-negative assumption can bias evaluation, penalize plausible recommendations with no recorded feedback, and favour algorithms that concentrate on popular or highly exposed items. We propose an LLM-based framework to expand relevance judgements for offline recommender evaluation. Our approach uses large language models in two complementary roles. First, a preference induction stage summarizes each user's historical interactions into a textual profile that captures their tastes and interests. Second, conditioned on this profile, an LLM acts as a relevance judge for candidate recommended items that lack observed labels in the original test data. To make this process tractable and evaluation-focused, we apply judgement expansion to a pooled candidate set built from the top-ranked outputs of multiple recommenders. The resulting enriched judgements provide additional relevance evidence for previously unobserved user--item pairs, enabling ranking metrics to be computed on a more complete basis. Experimental results show that this approach is a promising strategy for improving the robustness of offline top-N evaluation and mitigating the popularity-sensitive distortions caused by sparse feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11292v1">The Paternalistic Filter: Epistemic Injustice and Differential Refusal in LLM-Mediated History Education for Marginalized Romanian Students</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 8th International Workshop on Culturally-Aware Tutoring Systems (HAL precedings)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as conversational tutors, they risk institutionalizing systemic inequalities. This study presents a systematic API audit of four LLMs acting as history tutors, evaluating 1,800 responses regarding the 1989 Romanian Revolution across five student personas varying by ethnicity and socio-economic tier. We uncover four interconnected patterns of \emph{epistemic paternalism}: (1)~\textbf{Differential Refusal}, where safety-aligned models block 76.7\% of educational requests from low-tier students; (2)~\textbf{Epistemic Gatekeeping}, evidenced by a 3$\times$ reduction in access to geopolitical complexity (e.g., the contested ``coup theory'') for marginalized learners; (3)~\textbf{Agency Theft}, a lexical shift where models like LLaMA produce a 5$\times$ higher victimization-to-politics vocabulary ratio for Roma students compared to elite peers; and (4)~\textbf{Elite Hermeneutics}, where AI tutors disproportionately withhold epistemic confidence and justification scores from low-resource demographic profiles. We argue that current safety alignment acts as a paternalistic filter, transforming conversational AI into agents of narrative segregation -- a manifestation of \emph{hermeneutical injustice} in Fricker's~\cite{fricker2007} sense that demands urgent pedagogical auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11276v1">Automated Textbook Auditing with Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Presented @ iTextbooks 2026: 7th Workshop on Intelligent Textbooks at AIED'2026
    </div>
    <details class="paper-abstract">
      Ensuring the quality of educational materials requires more than standard proofreading: textbooks must be audited for factual accuracy, domain-specific technical correctness, and linguistic quality simultaneously -- a task that general-purpose grammar checkers cannot address. We present \textbf{AI Textbook Auditor}, a modular multi-agent pipeline for automated quality assurance of educational materials across subject domains. The system accepts a textbook PDF and produces a structured, human-reviewable report via two analysis tracks: a \textbf{Factual and Technical Track} in which an ensemble of specialized LLM agents detects factual inaccuracies, code errors, incorrect definitions, and conceptual inconsistencies, augmented with web search for humanities domains; and a \textbf{Grammar Track} operating PDF-natively to preserve diacritical encoding. A \textbf{Judge Agent} filters false positives using domain-specific rules before presenting findings to a human reviewer. The pipeline supports two ingestion modes -- vision-native page rendering and PyMuPDF text extraction -- and is domain-adaptable via custom prompts encoding subject-specific error taxonomies. We demonstrate the system on two Romanian upper-secondary textbooks: a CS textbook (56 technical findings across seven categories, with an expert-validated precision of 62.5\%) and a history and social sciences textbook (72 findings spanning factual errors, ideological bias, and grammar). The system is designed as a triage tool that reduces the manual effort of locating candidate issues, with human expert validation required before any editorial action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11267v1">Enhancing LLMs through human feedback: a journey towards self-improvement</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 AIC 2025: The 10th International Workshop on Artificial Intelligence and Cognition (held as part of ECAI 2025). October 25-26, 2025. Bologna, Italy
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of information retrieval systems, the ability to adapt and improve through user feedback is paramount. This study introduces a novel methodology for refining the performance of a primary Retrieval Augmented Generation (RAG) system by strategically integrating an auxiliary feedback RAG system. By systematically harnessing human-generated feedback, the approach aims to enhance the accuracy, relevance, and overall quality of responses, driving the system towards self-improvement. Central to this methodology is a human-in-the-loop implementation, where user feedback is continuously collected, classified, and integrated into the inference workflow, enabling the system to learn and evolve iteratively. To validate the effectiveness of this approach, the study employs rigorous testing against three diverse benchmark datasets focused on general and custom domain knowledge, utilizing a LLM-as-a-Judge evaluation strategy. This comprehensive framework not only underscores the transformative potential of feedback-driven enhancements in RAG systems but also sets a precedent for future research in adaptive information retrieval technologies, marking a significant step in the journey towards autonomous refinement and optimization through user engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11262v1">GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 14 pages, 14 figures. Accepted to MICRO 2026
    </div>
    <details class="paper-abstract">
      Modern LLM (large language model) workloads increasingly rely on optimized GPU kernels through hardware-software co-design. These kernels achieve high-performance through fine-grained dependency scheduling and computation-memory overlap. As such, they incur new challenges on existing GPU performance models. Instruction-driven simulators are costly to adapt to evolving architectures, while analytical models are too coarse to capture kernels' characteristics. We propose GPU-Tile-Sim, a tile-centric GPU simulation framework for LLM hardware-software co-design. The key insight is that modern LLM kernel performance is governed less by individual instruction latency than by the dependency structure that controls execution order and overlap. Accordingly, GTSim represents kernel execution as a warp-level tile graph whose nodes capture tile-level operations and whose edges encode data and ordering constraints. Using this representation, we design an automatic tile-graph frontend and a graph-driven simulation backend. We evaluate GTSim on representative GEMM, attention, and end-to-end LLM inference workloads. On A100 and H100 across both conventional and highly optimized kernels, GTSim achieves high performance-modeling accuracy (MAPE, Mean Absolute Percentage Error, 1.22%--8.71%). We further extend GTSim to Blackwell with preliminary validation, and demonstrate its effectiveness in analyzing software and architectural design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11258v1">TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Tree search algorithms enable systematic exploration of the proof space in neural theorem proving. Existing LLM tree search libraries primarily target natural language reasoning and do not provide native integration with formal verifiers, while theorem proving systems often rely on task-specific search implementations. We introduce TreeThink, an open-source Python library for modular, fully asynchronous tree search in neural theorem proving. It integrates established tree search methods with vLLM-based inference pipelines and diverse node evaluation techniques, ranging from lightweight heuristics to neural evaluators. We support Lean~4, Rocq, and Isabelle/HOL alongside natural language. It connects directly to each language's Read-Eval-Print Loop (REPL) server for real-time verification and proof state extraction. We evaluate TreeThink on miniF2F and MATH500, demonstrating cross-language formal proof search, natural language reasoning support, and up to 6.3$\times$ wall-clock speedup from asynchronous execution. Source code is released under the MIT license at https://github.com/GGLAB-KU/treethink , and the library is accessible as a downloadable package at https://pypi.org/project/treethink/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11250v1">Multi-Agent LLMs Fail to Explore Each Other</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Exploration is essential for reliable autonomy in multi-agent systems, yet it remains unclear whether large language model (LLM) agents can explore effectively when interacting with one another. We show that modern LLM agents fail to do so, often exhibiting myopic and polarized interaction patterns that lead to suboptimal coordination and increased regret. We formalize this challenge as the Multi-Agent Exploration problem, modeling it as a partially observable stochastic game (POSG) problem in which agents must probe peers to infer their capabilities and identify effective interaction strategies. To address this, we introduce Multi- Agent Contextual Exploration (MACE), a lightweight framework that explicitly promotes exploration through structured peer selection. Across both contextual and parametric diversity settings, MACE substantially improves exploration behavior and downstream task performance. We further show theoretically that the value of exploration increases with agent diversity. Overall, our results highlight a fundamental limitation of current LLM agents and underscore the importance of explicitly guided exploration for reliable multi-agent autonomy. Code will be released in https://github.com/deeplearning-wisc/mace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11211v1">FastTPS: An Optimized Method for LLM Token Phase for AI accelerators</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 16 pages, 8 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The popularity of large language models (LLMs) escalates an ongoing demand for effective inference. However, due to the sequential processing of tokens during the token phase in decoder-only LLMs inference, the inherent low parallelism leads to reduced throughput and suboptimal utilization of the computing units on artificial intelligence (AI) accelerators, particularly when handling long-sequence inputs that impose significant memory overhead. Recently, many reported methods have been developed as potential solutions, since they emerge with numeric deviation. This paper presents FastTPS, a high performance and low-precision loss method for accelerating the token-phase in LLM inference on general AI accelerators which includes three key components: (1) AI accelerator-enabled reloading-free KV Cache concatenation which decreases memory access overhead as well as enables full fusion of Attention, (2) high-efficiency and high-accuracy 'RoPE' attention based on the tiling optimized FLAT, and (3) highly-fused MLP with fine-grain pipeline scheduling. Our results confirm that FastTPS significantly alleviates memory bottlenecks in the token phase, delivering a 6x speed improvement (compared to none-fusion) on an AMD Ryzen AI 300 series NPU with BF16 precision while sustaining 93% peak memory bandwidth utilization during Phi3-mini-4k-instruct inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11207v1">ProgramTab: Boosting Table Reasoning of LLMs via Programmatic Paradigm</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Large Language Models, Table Reasoning, In-context Learning
    </div>
    <details class="paper-abstract">
      Table-based reasoning with large language models (LLMs), which requires reasoning based on natural language questions and structured tabular data, has gained widespread attention. However, a series of issues still constrain the application of this task. The previous approaches suffered from significant performance degradation when faced with large tables due to the difficulty of long text modeling and the limitation of input length for LLMs. The text-to-SQL approach is used to efficiently extract key information from tables and generate smaller sub-tables. However, tabular data, especially web tables, often lack the necessary structure and consistency, making them unsuitable for performing mathematical logic operations using SQL queries. We propose the ProgramTab framework, which guides LLMs employing in-context learning to perform tabular data preprocessing with Python code, as well as the momentous contents extraction with row and column extraction and SQL generation. The experiment results on table reasoning datasets demonstrate that the ProgramTab framework effectively deals with table-based reasoning tasks and outperforms all LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11197v1">What We Talk About When We Talk About LLM Planning: Evidence for Two Distinct Planning Abilities</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 19 pages. Keywords: Reasoning, Automated Planning, Item Responses Theory, LLMs as Planner Research Area: NLP and Symbolic Reasoning Research Area Keywords: neurosymbolic, planning in agents, symbolic reasoning Contribution Types: Model analysis & interpretability
    </div>
    <details class="paper-abstract">
      When LLMs exhibit uneven performance across planning tasks, these gaps are often attributed to task difficulty. We argue that this explanation is incomplete, as task-level variation may reflect distinct latent planning competencies rather than differences along a single ability spectrum. We study this question on ACPBench-Hard by evaluating multiple LLM families under varying test-time reasoning budgets and applying a multidimensional item response theory model to uncover the latent competency structure underlying LLM planning. The analysis reveals two principal dimensions that shape planning performance: operational reasoning, the ability to evaluate local action applicability and immediate state transitions, and structural enumeration, the ability to reason about goal reachability and landmark structure. Operational reasoning improving under model scaling and longer reasoning traces, while structural enumeration remains comparatively insensitive. Our findings motivate competency-level evaluation of LLM planning, shifting the focus from whether models improve overall to which planning competencies improve, under what conditions, and why.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13080v1">Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 20 pages, 13 figures, 14 tables. Industrial longitudinal case study
    </div>
    <details class="paper-abstract">
      Autonomous coding agents force engineering organizations to choose between API-based frontier models -- strong reasoning at high token cost -- and on-premise quantized open-weights models, which promise low-marginal-cost scaling and data sovereignty at some loss of reasoning fidelity. We study this trade-off through a single-developer, non-randomized longitudinal case study over two contiguous 28-day periods on a production monorepo: an API-based Claude Opus 4.7/4.8 configuration using Claude Code versus an on-premise GLM-5.1/5.2 configuration using Opencode, quantized to NVFP4, on NVIDIA Blackwell hardware. Analyzing LLM telemetry and Git history, we find that prompt caching (99.3% hit rate) cuts realized API cost by 88.6% to an effective \$0.57 per million tokens -- below even the \$2.83 amortized unit cost of the shared on-premise slice (a utilization-dependent inversion; total realized spend and total cost of ownership (TCO) are the robust quantities). At comparable gross code churn, the local configuration was associated with a far higher defect-repair burden: a Fix Commit Ratio (FCR) of 74.9% versus 45.9%, with the odds of a commit being a repair 2.6 to 4.9 times higher within every difficulty tier (Mantel-Haenszel OR = 3.61). Under Taiwan-market parameters and a symmetric labor model, on-premise deployment nonetheless saves 40.1% of true TCO under shared GPU allocation, whereas dedicated reservation costs 43.8% more than the cached API. Under shared allocation, the genuine penalty is not monetary but a measurable developer-experience burden -- timestamp indicators show more work trapped in debugging spirals and a slower commit cadence -- and an offline replay shows hybrid routing gateways trade defect rate for infrastructure savings along a cost-quality frontier rather than dominate the pure-API baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06401v2">BizFinBench.v2: Towards Reliable LLMs in Finance via Real-User Data and Offline/Online Bilingual Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Large language models are becoming increasingly significant in financial applications. Nevertheless, prevailing benchmarks are largely dependent on simulated or generic data, which leads to a significant gap between reported performance and actual efficacy in real-world scenarios. To tackle this challenge, we present BizFinBench.v2, the first integrated offline and online benchmark built upon authentic user query-response data from both Chinese and U.S. equity markets. It comprises 28,860 questions across eight offline and two online tasks. Experimental results show that GPT-5 achieves a mere 61.5% accuracy, still failing to meet the practical business requirement (84.8%). Among the evaluated commercial models, DeepSeek-R1 exhibits superior investment efficacy. Error analysis grounded in real financial practice reveals persistent limitations in existing models. By overcoming the constraints of prior benchmarks, BizFinBench.v2 provides a substantiated foundation for advancing LLM deployment in the financial sector. Our data and code are available at https://github.com/HiThink-Research/BizFinBench.v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11127v1">Do LLMs Fabricate Legal Citations? A Bilingual Benchmark on Saudi Data Protection Law and the GDPR</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 5 pages, 3 tables. Benchmark data and model outputs to be released. Also archived at Zenodo: 10.5281/zenodo.21320218
    </div>
    <details class="paper-abstract">
      Organizations and regulators increasingly consult large language models (LLMs) for regulatory-compliance questions, yet a wrong statutory citation can silently propagate into legal advice, compliance documentation, and policy decisions. We introduce a bilingual benchmark of 120 questions probing whether freely accessible LLMs fabricate article citations for two data-protection instruments: the EU General Data Protection Regulation (GDPR) and the Saudi Personal Data Protection Law (PDPL). The benchmark pairs direct citation retrieval questions with false premise verification probes and deliberately unanswerable "trap" questions -- including questions about a repealed article and about deadlines that exist only in implementing regulations, not in the law itself. Every question is posed in both Arabic and English, and all scoring is fully automatic against a manually verified gold reference. Evaluating three freely accessible models (Gemini 2.5 Flash, GPT-OSS-120B, Nemotron-3-Super-120B), we find a dramatic jurisdiction gap: near-ceiling citation accuracy on the GDPR (94-100% on direct retrieval) against majority fabrication on the Saudi PDPL (60-77%), invariant to query language; the highest fabrication rates (67%) arise from statute-vs-regulations confusion, and 91% of fabricated citations are asserted with confidence >= 0.8. Fabrication tracks the jurisdiction of the law, not the language of the query, and model confidence provides no protection -- indicating that verbatim-verification safeguards, rather than model self confidence, must gate any institutional reliance on LLMs for compliance screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19274v3">HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11102v1">CHARM: Charge Calibration and Acoustic Rescue for LLM-based Multimodal Sarcasm Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Sarcasm detection, the identification of discrepancies between literal and intended meaning, is a fundamental task in affective computing. However, zero-shot instruction-tuned Large Language Models (LLMs) systematically over-predict the positive (sarcastic) class across the entire capability spectrum, while the prosodic cues humans rely on remain underexploited and transfer unevenly across languages. We introduce CHARM (Charge Calibration and Acoustic Rescue for Multimodal Sarcasm Detection), a training-free framework that couples two modules. Bidirectional Charge Calibration (BiCAL) steers the LLM toward opposing sarcastic and literal verdicts along a symmetric axis of charged prompts; the induced directional biases cancel by construction, and a simple aggregation recovers an unbiased pragmatic signal. Acoustic Late-Fusion Rescue (ALFR) then fuses the calibrated votes with prosodic descriptors and LLM-generated auditory-perception probes through a shallow classifier, actively down-weighting saturated text votes in favour of acoustic evidence. Without fine-tuning any backbone, BiCAL attains the highest reported zero-shot text-only Macro-F1 of 0.787 on MUStARD, while ALFR lifts weak backbones by up to +0.382 Macro-F1 on CMMA. A Stouffer meta-analysis confirms statistical significance on MUStARD and CMMA (Z = 13.89 and Z = 34.64, respectively; p < 10^-43). Our analysis further uncovers a cross-cultural prosodic decoupling: low-level acoustics fail to transfer across languages, whereas high-level perceptual abstractions remain robust. Together, these components yield an explainable, cross-lingual multimodal detector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07486v3">Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Published in Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful tool for synthetic data generation. A particularly important use case is producing synthetic replicas of private text, which requires carefully balancing privacy and utility. We propose Realistic and Privacy-Preserving Synthetic Data Generation (RPSG), which uses private seeds and integrates privacy-preserving strategies, including a formal differential privacy (DP) mechanism in the candidate selection, to generate realistic synthetic data. Comprehensive experiments against state-of-the-art private synthetic data generation methods demonstrate that RPSG achieves high fidelity to private data while providing strong privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11079v1">Are LLMs Ready for Scientific Discovery? A Capability-Oriented Benchmark for AI Scientists</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Existing benchmarks for scientific data analysis evaluate LLMs primarily on code execution or workflow completion, overlooking that scientific analysis serves to support distinct types of scientific claims: hypothesis exploration, statistical inference, mechanistic explanation, each with different assumptions and validity criteria. We introduce SDABench, a benchmark that reorganizes evaluation around six capabilities (descriptive, exploratory, inferential, predictive, causal, and mechanistic) across five domains (Biology, Chemistry, Environment, Geography, Physics). SDABench comprises 527 real-data instances (SDA-Real) and 6000 synthetic instances (SDA-Synth), each in both multiple-choice and open-ended formats, constructed through an automated pipeline. Evaluating 15 representative LLMs, we find that models handle descriptive analysis well but degrade sharply on tasks requiring assumption selection, latent-process modeling, or mechanistic reasoning. SDABench further provides a five-stage error analysis framework that locates where LLMs fail: more advanced models more reliably identify the relevant scope and variables, but still struggle to select appropriate analytical procedures, model variable relationships, and draw valid conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11078v1">Do Video-LLMs Actually Watch? Diagnosing Character-Tracking Failures in Long-Form Video</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Can a Video Large Language Model (Video-LLM) follow one person through a long video, keeping track of who they are well enough to report, in order, how their outfit changes across a full TV episode? Benchmarks increasingly score this kind of task, and the strongest open-source 7--8B models now reach 37--38% on InfiniBench's global appearance task, which asks exactly that. But does that score come from tracking the named character, or from something easier? We test this with a nine-condition diagnostic protocol applied to three architecturally distinct open-source Video-LLMs, with Gemini~2.5~Flash as a frontier reference, and find the accuracy does not come from character tracking. When we change the character named in the question to a different cast member, leaving the video and answer options untouched, the models change their answer only 4--31% of the time, so they are largely ignoring who the question asks about. Breaking that test down by the gender of the swapped name shows why: the models react more when the name is changed to a different-gender character than to a same-gender one (a 13--28 point gap), picking up coarse gender cues but unable to tell same-gender individuals apart. This shallow processing surfaces again when we drop the multiple-choice options and ask the same questions open-endedly: open-source accuracy drops 18--25 points, with none of 151 answers fully correct, versus a 12-point drop for Gemini. Further checks rule out the obvious innocent explanations, adding subtitles, using the most informative frames, or doubling the number of frames all leave character tracking unimproved, so the bottleneck is not how much video the model sees but how it ties that video to the person the question names. We release a diagnostic toolkit for auditing what such benchmark scores actually measure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11070v1">MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 29 pages. Warning: This paper contains examples of harmful content
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) operate in interactive multi-turn settings, making multi-turn jailbreaking a realistic threat model and an important setting for automated red teaming. A core challenge in learning multi-turn jailbreak attackers is credit assignment: different turns contribute differently to the final outcome, yet existing learning signals are often too coarse to identify their individual contributions. We propose decomposed credit GRPO (DC-GRPO), a unified turn-level credit assignment framework for Group Relative Policy Optimization in multi-turn jailbreak learning. DC-GRPO assigns a separate group-relative learning signal to each turn by combining immediate and future credit, avoiding the credit misassignment induced by broadcasting a single trajectory-level score across the dialogue. We instantiate this framework with static and dynamic weighting rules that differ in how the two credit sources are balanced while sharing the same turn-level structure. Across multiple victim LLMs and benchmarks, the dynamic- and static-weighted variants achieve average ASR5@3 scores of 98.26% and 97.88%, respectively, substantially outperforming the state-of-the-art methods, including SEMA (86.58%) and TROJail (86.23%). Their consistently strong performance indicates that the central empirical benefit comes from turn-level group-relative credit assignment rather than a particular weighting rule. Warning: This paper contains examples of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11053v1">Flout at Your Own Risk: LLMs Struggle with Pragmatic Cooperativity Under Epistemic Asymmetry</a></div>
    <div class="paper-meta">
      📅 2026-07-13
    </div>
    <details class="paper-abstract">
      Fruitful collaborations rely on cooperative communications, including of contextual cues to incorporate into reasoning. The increasing use of LLMs in collaborative and agentic pipelines raises questions about the extent to which they exhibit these pragmatic capabilities, especially in scenarios where they may not have access to the same information as their collaborators. In this paper, we perform a novel investigation into the pragmatic reasoning capabilities of LLMs in a multi-party collaborative task under partial information conditions. We formalize a notion of collaborative epistemic asymmetry that explicitly connects objective task success to Grice's cooperative principle and empirically assess various LLMs' abilities to act cooperatively as both speakers and listeners, including both prompting and post-training strategies. Our results show that while LLMs exhibit certain pragmatic capabilities in collaborative settings, and these can be elicited through prompting and post-training, they still face challenges in pragmatic communication with incomplete information, and that certain failure modes do correlate with floutings of Grice's maxims that go unrecognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11032v1">LLM-Generated Design Problems for Assessing Higher-Order Thinking in Project-Based Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 Accepted to appear in Proceedings of the 2nd ACM Virtual Global Computing Education Conference V.1 (SIGCSE Virtual 2026). DOI: 10.1145/3795867.3831014
    </div>
    <details class="paper-abstract">
      Project-based learning (PjBL) is common in computing education, but traditional assessments of PjBL often fail to capture higher-order thinking (HOT), especially in transfer contexts. This study introduces "design problems" (DPs): concise, scenario-based prompts that require applying project concepts in new situations, to address this gap. We examined instructor perceptions, the ability of large language models (LLMs) to generate DPs, and student experiences. Surveys of 31 instructors, evaluation of 80 LLM-generated DPs, and student performance data showed that while instructors value DPs, creation effort is a barrier. LLMs helped by producing high-quality prompts with strong expert agreement. Students rated DPs from different LLMs similarly, and their performance on DP tasks showed negligible correlation with traditional project grades, suggesting DPs may capture distinct aspects of HOT. Keystroke data also suggested deeper cognitive engagement of students through planning and revision behaviors. Overall, DPs appear to be a useful complement to traditional assessments, especially in situations where AI use or collaboration may undermine individual learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13078v1">Operational Evidence Gaps for LLMs in Fraud Detection and Trust-and-Safety Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 22 pages, 3 figures, 6 tables. Ancillary files include the evidence matrix, search note, and numeric claim check
    </div>
    <details class="paper-abstract">
      LLMs are now proposed for fraud detection, scam investigation, content moderation, and other trust-and-safety workflows. Much of the public literature still evaluates them as models, with less attention to their behavior as components in operational pipelines. This creates a practical evidence question: what would justify placing an LLM inside a live workflow with latency, cost, escalation, human-review, and adversarial-risk constraints? We address this question through a fraud-first survey of deployment evidence. We code 49 operationally relevant sources on LLM use in fraud detection, investigation support, content moderation, and cross-cutting robustness (18 fraud, 14 moderation, 17 cross-cutting), supplemented by 15 contextual references that establish the survey boundaries. These sources include systems, benchmarks, frameworks, and deployment-relevant surveys, not 49 production deployments. The main finding is an evidence imbalance. Fraud supplies the largest task-specific portion of the coded corpus. The moderation papers, however, include more explicit public evidence on latency, cost, governance, and fairness. Among the 18 fraud and investigation sources, none report clean per-decision latency, per-decision dollar cost, or calibration evidence; most report offline task performance, retrieval gains, or case-study accuracy instead. The survey contributes a role-and-evidence organizing frame, FORTE, for locating LLMs as classifiers, retrieval interfaces, explanation generators, reviewer assistants, agents, feature extractors, or escalation components. It also contributes a minimum deployment-evidence checklist covering latency budget, cost per decision, decision threshold, explanation integrity, and adversarial pressure. The resulting agenda identifies studies needed to support deployment claims for LLM-based fraud and trust-and-safety work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10932v1">LLM-Enhanced Dynamic Financial Knowledge Graphs for Cross-Entity Signal Propagation and alpha discovery</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 36 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Financial information rarely affects a single company in isolation. Earnings surprises, capital expenditure changes, supply constraints, and guidance revisions can propagate through networks of suppliers, customers, competitors, and technology ecosystems. Traditional financial NLP primarily measures document-level sentiment for the directly mentioned company and often ignores cross-entity information diffusion. This paper develops an LLM-based financial measurement and signal propagation framework. The LLM converts unstructured financial documents into structured economic state-change events and extracts explicit and implicit corporate relationships to construct a dynamic financial knowledge graph. Event signals are then propagated through the estimated network using a community-aware mechanism, allowing information to diffuse more strongly within dynamically detected economic communities than across community boundaries. We introduce Community Information Surprise, CIS, and Propagated Information Surprise, PIS, as network-based financial signals and develop corresponding econometric tests. Controlled simulations with time-varying economic communities show that the framework accurately recovers latent network structure, detects the emergence of new investment ecosystems, and generates propagated signals with incremental predictive power beyond sentiment and direct LLM event signals. Across repeated simulations, community-aware propagation achieves the strongest rank information coefficient and long-short Sharpe ratio among five nested benchmarks.A second Russell 1000 calibrated simulation confirms that the main results persist under sparser networks, heterogeneous news coverage, realistic large-cap volatility, and smaller effect sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08098v3">When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Preprint. 16 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Majority voting is the default unsupervised aggregator for multi-sample LLM inference, but it discards two signals: within-group answer entropy and between-group reasoning geometry. We aggregate by delegation instead (Propagational Proxy Voting, PPV): each group of samples keeps weight on its own answer in proportion to its entropy-based confidence (When) and routes the rest to peers by reasoning-embedding similarity (Whom); the stationary distribution of the resulting delegation matrix picks the consensus answer. This requires neither gold labels nor training. On MMLU-Pro with 128 samples per question, delegation beats majority by +1.5 pp overall and +2.24 pp on non-trivial questions (McNemar p ~ 1.0e-14, n = 8,099), overturning wrong majorities whose answer cluster is geometrically incoherent while the correct minority is tight. We then characterize exactly when delegation overturns majority: a two-option model gives a closed-form flip condition on each option's confidence and the weight it routes to the other, with a do-no-harm corollary for near-unanimous questions. The condition calls the realized winner on 96.5% of non-trivial questions, and its predicted mass gap tracks the realized gap at r = 0.97. We did not find any other unsupervised ensemble methods that close the oracle gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10871v1">Toward Contemplative LLM: A Modular Framework for Evaluating and Enhancing LLM Alignment in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Accepted as an oral presentation at HARMONY 2026 (Human-centered AI Research for Mental Health, an Open Networking Symposium), co-located with IEEE/ACM Conference on Connected Health: Applications, Systems, and Engineering Technologies (CHASE 2026) held in Pittsburgh, August 6, 2026
    </div>
    <details class="paper-abstract">
      Contemplative traditions have long guided ethical behavior and prosocial interaction, and recent work suggests that contemplative principles (e.g., mindfulness, compassion, non-dual reasoning) may offer a promising paradigm for aligning large language models (LLMs), improving cooperation and reducing ethical violations in LLM outputs. However, as new models, evaluation metrics, and benchmarks emerge rapidly, it remains challenging to systematically assess whether and how contemplative principles enhance LLM alignment across diverse and evolving scenarios, and existing approaches are often ad hoc and fail to generalize. We present a modular, extensible evaluation framework, initially targeted at the mental health domain, that enables seamless integration of new models, metrics, and benchmarks through a reusable pipeline. The framework currently reproduces existing state-of-the-art results and supports systematic cross-evaluation by flexibly mixing and matching models, metrics, and benchmarks, enabling fair comparison and deeper insight. Its plug-and-play prompting module offers a principled pathway for incorporating ethical perspectives such as contemplative principles, allowing domain experts to define alignment criteria without requiring technical expertise. Although initially focused on mental health, the framework is domain-agnostic and extends naturally to areas such as decision-making, moral reasoning, and human-AI collaboration. By bridging computational evaluation with human-centered ethical reasoning, this work lays the groundwork for interdisciplinary research spanning cognitive science, behavioral economics, philosophy, and system design, toward robust, trustworthy, and socially beneficial human-AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10848v1">Predictive Divergence Masks for LLM RL</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning for large language models (LLMs) typically relies on trust-region masks to stabilize off-policy updates. The dominant PPO-style approach uses the sampled-token importance ratio for two criteria: a proximity criterion, which asks whether the policy has moved too far from the behavior policy, and a direction criterion, which asks whether the update pushes it farther away. Recent work DPPO improves the proximity criterion by replacing PPO's ratio-based test with a probability divergence between the behavior and training policies. However, its direction criterion is still inherited from PPO. A token can be masked only when the sampled-token importance ratio moves away from one. We observe that this ratio-based direction criterion is a single-sample proxy that can disagree in sign with the change of the divergence that defines the proximity criterion. We therefore propose the predictive divergence mask, which asks whether the next policy-gradient step will increase or decrease the same divergence used by the trust region. For the discrete softmax policies used in LLM RL, we derive this prediction in closed form. Because production rollout engines expose only a truncated (top-K) view of the vocabulary, we develop two lightweight top-$K$ estimators for this prediction. Detailed analysis shows the divergence-based direction is better aligned with the realized change of the divergence than the sampled ratio, and the resulting masks improve RL training across model scales and precision settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10846v1">Quantifying the Sources of Instability in LLM-Based Stance Analysis of Public Discourse</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Computational social science increasingly relies on automated preprocessing pipelines -- speaker diarization, ASR transcript cleaning, sentence segmentation -- to convert raw media into analyzable text. When these pipelines produce different outputs from the same input, two distinct sources of instability can arise: the preprocessing pipeline itself (diarization method, segmentation rules) and the downstream measurement instrument (LLM annotation vs.\ keyword lexicon). Using 256 YouTube interviews across 41 public figures from five domains, we compare two speaker-diarization pipelines and two measurement methods, all targeting the coupling between affective valence and epistemic modality. We find that (1) preprocessing pipeline sensitivity is concentrated in speakers with limited video samples (N $\leq 5$); for the four best-sampled speakers (N $\geq 16$), the mean absolute pipeline-induced change in $r(\text{neg}, \text{emph})$ is only $0.13$; (2) cross-method disagreement is larger and more systematic -- the LLM and keyword-lexicon methods assign opposite coupling directions to several well-sampled speakers, even within the same preprocessing pipeline; and (3) aggregate valence proportions are highly stable ($|Δp(\text{neg})| < 6$pp) regardless of pipeline or method, masking both sources of instability. The contribution is a diagnostic framework that separates pipeline effects from measurement effects: researchers studying cross-dimensional relationships in interview data should verify that their conclusions are robust to both sources of variation, with particular attention to measurement method choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26104v3">Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that may end up in LLM training corpora: assert a position rather than describe a scene neutrally. The features that shift the model are the ones that make the writer's position explicit; the features that dilute it hold animal-welfare content but withhold stance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10814v1">Auditing Belief-Conditioned LLM Agents in Hidden-Information Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 29 pages, 8 figures, 3 tables. Preprint
    </div>
    <details class="paper-abstract">
      Evaluating LLM agents in hidden-information multi-agent settings is hard: final outcomes are high-variance and rarely reveal why an agent decided as it did. We study this in a 9-player Werewolf environment where agents act under strict, code-level information isolation, and we build an auditable framework that maintains an external belief state over hidden roles, logs belief updates and belief-action deviations as structured evidence, and supports a defensive offline improvement loop that reviews bad cases before any strategy change. Across 1,080 frozen games spanning belief-disabled, active-belief, kernel-ablation, camp-restricted, consumption-policy, and high-load arms, and including a seed-paired A0/A1 comparison, the active-belief condition is associated with substantially better good-side outcomes: in the 200-seed A0/A1 comparison the good-side win rate rises from 0.205 to 0.390 (paired McNemar $χ^2 = 16.4$, $p < 0.001$), with fewer irreversible witch-poison errors. We do not, however, attribute this shift to belief content. Direct action-belief consistency is low ($\approx 0.21$), and giving belief only to the werewolves helps the good side more than giving it only to the good side, which argues against a simple holder-benefit account; we therefore report the effect as an association and treat its mechanism as unresolved. The contribution is the audit framework itself: it makes the effect measurable, exposes low direct action-belief consistency, rejects an unreliable forced-consumption intervention with evidence, and separates strategy effects from load confounds. We accordingly position external belief in high-noise hidden-information games primarily as an auditable cognitive baseline that also carries decision-relevant signal, turning opaque agent behavior into replayable evidence for safer, controlled iteration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10803v1">Weight-Adjusted Gradients Reveal Parameter Importance and Failure Modes in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Understanding which parameters are influential in Large Language Models (LLMs) is central to improving their efficiency, reliability, and interpretability. We introduce Weight-Adjusted Gradients (WAG), a simple yet effective approach for estimating parameter importance that explicitly captures the interaction between model weights and first-order gradient information and identifies parameters that disproportionately influence model behavior, such as those responsible for collapse phenomena in LLMs. Across a range of models and settings, we show that WAG surfaces a tiny but critical subset of parameters whose modification leads to dramatic degradation in performance, a failure mode that existing importance metrics overlook. These findings reveal a previously underexplored interplay between weights and gradients, suggesting that parameter importance cannot be fully understood through either signal alone. The surprising effectiveness of WAG points to fundamental structural properties of trained networks and motivates new open questions about the role of zeroth-order and first-order information in deep learning. We demonstrate the practical utility of WAG across multiple applications, including expert allocation in mixture-of-expert architectures, parameter-specific unlearning, mixed-precision quantization, and layer selection for knowledge editing. Our results position WAG as a unified approach for analyzing, debugging, and controlling LLMs, and opens new directions for principled model-level interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01299v2">HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      In retrieval-augmented generation and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate per-request cost. Two directions have emerged to reduce this cost: position-independent caching (PIC) admits KV reuse for non-contiguous segments shared across requests, while hybrid-attention models cut computation by replacing most full-attention layers with linear attention. However, they cannot coexist: applying existing PIC methods to hybrid-attention models breaks down because per-token KV-cache reuse primitives do not transfer to the per-request recurrent state. We present Hypic, the first system to accelerate hybrid-attention LLM serving with position-independent caching. For linear-attention layers, we identify the segment-cumulative transition operator as the missing algebraic primitive and cache it alongside each segment's zero-start end-state, enabling near-exact and constant-time composition of independently cached segments. For the remaining full-attention layers, existing PIC methods also fail because linear layers do not expose the per-token hidden states needed for selective recomputation. We show that the largest deviations concentrate at segment beginnings and construct a small seam window that propagates hidden states through the hybrid-attention stack to repair cross-segment attention. Finally, Hypic introduces segment parallelism, which exploits PIC's segment-level self-containment to parallelize cache-miss prefill across instances, turning long cold requests into an accelerable workload. Evaluated across four hybrid-attention models and five workloads, Hypic reduces time-to-first-token by $3.25\times$ on average and improves QPS by $1.66\times$ over Prefix Cache, while preserving task quality with a 1.71-point gap from Full Recompute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10789v1">Imaging-101: Benchmarking LLM Coding Agents on Scientific Computational Imaging</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Computational imaging, which recovers hidden signals from indirect, noisy measurements, underpins quantitative discovery across scientific disciplines, yet building a correct reconstruction pipeline demands deep domain expertise and remains laborious even for domain scientists. We introduce Imaging-101, a benchmark of 57 expert-verified computational imaging tasks spanning six scientific domains, each grounded in a peer-reviewed paper and canonicalized into a standardized four-stage pipeline (preprocessing, forward physics modeling, inverse solver, and visualization) Three evaluation tracks (planning, function-level unit tests, and end-to-end reconstruction) probe distinct agent capabilities across the full pipeline. Evaluating seven frontier LLMs uncovers systematic challenges in applying coding agents to computational imaging that go beyond those exposed by general coding benchmarks, spanning algorithm selection, physical convention handling, and pipeline integration. These findings highlight concrete capability gaps and point toward skill-augmented, domain-specialized agents as a practical path to reliable computational imaging assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19806v3">LLM-Based Social Simulations Require a Boundary</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 ICML 2026 Position Paper Track
    </div>
    <details class="paper-abstract">
      This position paper argues that LLM-based social simulations require clear boundaries to make meaningful contributions to social science. While Large Language Models (LLMs) offer promising capabilities for simulating human behavior, their tendency to produce homogeneous outputs, acting as an "average persona", fundamentally limits their ability to capture the behavioral diversity essential for complex social dynamics. We examine why heterogeneity matters for social simulations and how current LLMs fall short, analyzing the relationship between mean alignment and variance in LLM-generated behaviors. Through a systematic review of representative studies, we find that validation practices often fail to match the heterogeneity requirements of research questions: while most papers include ground truth comparisons, fewer than half explicitly assess behavioral variance, and most that do report lower variance than human populations. We propose that researchers should: (1) match validation depth to the heterogeneity demands of their research questions, (2) explicitly report variance alongside mean alignment, and (3) constrain claims to collective-level qualitative patterns when variance is insufficient. Rather than dismissing LLM-based simulation, we advocate for a boundary-aware approach that ensures these methods contribute genuine insights to social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10720v1">WattCouncil: Context-Aware Household Energy Scenario Generation With Governed LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The accelerating shift toward low-carbon power systems, together with the widespread adoption of behind-the-meter technologies such as rooftop solar and electric vehicles, is placing new operational and analytical demands on electricity grids. At the same time, smart-grid research increasingly relies on machine learning (ML), yet progress is constrained by limited access to high-resolution household energy data due to privacy concerns, regulatory barriers, and collection costs. This work presents WattCouncil, a data-generation framework in which household electricity demand is generated by a council of Large Language Model (LLM)-based agents operating in specialized roles to generate, audit, and validate structured energy scenarios under explicit cultural, temporal, and physical constraints. Rather than acting as static predictors, these agents serve as adaptive decision-makers within a governed pipeline. Motivated by studies highlighting the importance of contextual factors in energy use, our framework produces context-sensitive daily routines through a guided reasoning process that incorporates household composition, temporal factors, and environmental conditions. We evaluate the generated profiles against the detailed CER dataset, which contains over a year of load measurements for 4232 households together with survey-based socio-economic information. We further assess the consistency of the framework through ablation studies. Source code is available at https://github.com/Singularity-AI-Lab/wattcouncil
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10709v1">PromptGraph: Graph-Guided Prompt Sanitization for Balancing Privacy and Utility in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services introduce a fundamental privacy challenge. Sensitive information may be inferred not only from explicit identifiers, such as names or phone numbers, but also from contextual associations among otherwise innocuous spans. Existing sanitizers typically assign privacy or utility signals to individual spans without explicitly modeling pairwise relationships among them. In this paper, we propose PromptGraph, a graph-guided prompt-sanitization approach for privacy-preserving LLM inference. PromptGraph estimates privacy leakage at the span level and utility-relevant contextual dependencies between pairs of spans. It represents each prompt as an attributed graph, in which nodes carry span-level privacy scores and edges encode contextual dependencies needed to preserve utility. The sanitization objective selects a protected span set that maximizes privacy gain while penalizing the loss of contextual dependencies. This formulation explicitly balances privacy and utility when contextual evidence is hidden. Protected spans are sanitized locally, and returned placeholders are restored only after passing local consistency checks. We conduct extensive experiments showing that PromptGraph achieves a more favorable balance between privacy and utility than prompt-privacy baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21334v2">Ideological Bias in LLMs' Economic Causal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Accepted at COLM 2026
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 18 of 20 models, accuracy is systematically higher when the empirically verified causal sign aligns with intervention-oriented expectations than with market-oriented ones. Moreover, when models err, their incorrect predictions disproportionately lean intervention-oriented, and this directional skew is not eliminated by one-shot in-context prompting. These results highlight that LLMs are not only less accurate on ideologically contested economic questions, but systematically less reliable in one ideological direction than the other, underscoring the need for direction-aware evaluation in high-stakes economic and policy settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10645v1">MafiaScope: Non-Invasive, Time-Resolved Belief Probing for LLM Agents in Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      An LLM agent's public behaviour reveals little about its social reasoning: an agent that votes correctly may be guessing, and an agent that lies well leaves no trace of what it actually believes. We present MafiaScope, an open testbed that turns the social deduction game Mafia into a measurement instrument for machine Theory of Mind. After every public utterance, every agent privately answers a configurable set of structured probe questions; the answers never re-enter the game and are scored automatically against the ground truth the engine knows. An interactive visualizer renders the belief trajectories: impersonate mode shows the game as one agent sees it, panels chart timeline-aligned accuracy and calibration, and counterfactual replay forks any recorded step. In a 32-game DeepSeek case study with 13{,}815 parsed probe answers, stated confidence is poorly calibrated, with expected calibration error 0.17, agents over-predict being suspected 1.5 times, and a 30-fork replay experiment walks the counterfactual replay workflow end to end. Engine, viewer and a corpus of 200+ cross-model games are released under an open licence; live demo: https://karpovilia.github.io/mafiascope/; screencast: https://vimeo.com/1208920221.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10626v1">Eval-Pair Matrix: Answer-Paired Meta-Evaluation of LLM Judges for Grounded RAG</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge evaluation is widely used for retrieval-augmented generation (RAG), but reusing the same model family as both generator and judge makes self-leniency difficult to identify. We introduce Eval-Pair Matrix, a controlled meta evaluation protocol for source-grounded RAG. Starting from GaRAGe questions and grounding passages, we induce one hidden answer-causal contradiction per record, generate answers from perturbed passages with GPT, Grok, and Gemini models, and then use the same models as blind judges to evaluate each answer against the original passages. The experiment contains 300 core records, 897 labeled generator outputs, and 2,683 judge verdicts in a crossed 3 x 3 matrix; the primary analysis uses 275 fully validated records. Instead of comparing diagonal and off-diagonal cells across different answers, we estimate same-model effects by pairing judges on the exact same candidate answer. This changes the interpretation: diagonal and off diagonal F1 are similar, and the paired same-model recall effect is near zero (-0.5 pp; 95% cluster bootstrap CI [-2.7, +1.7]). The only robust paired gap is lower matching-judge flagging for answers that avoided the induced claim (-4.3 pp). A targeted human evaluation finds that reviewed apparent false positives are alternate source-error detections, mistakes in labeling whether the induced claim was adopted, or unclear cases; none were adjudicated as genuine false alarms. The lesson is methodological: RAG judge studies should report full matrices, answer-paired effects, behavior strata, and label-task alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10604v1">U-Lens: Supporting User Uncertainty Management in Long-Form LLM Responses</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate long-form answers for knowledge-intensive tasks, but users often struggle to decide which parts of a response deserve scrutiny, why they may be unreliable, and what to do next. Prior work on uncertainty communication has largely focused on making uncertainty visible through cues such as confidence scores, leaving less support for the broader process of managing uncertainty distributed across a long response. Through a formative study, we examine how users manage such uncertainty across three stages: interpretation, evaluation, and decision. Based on these insights, we derive design guidelines that address both stage-specific and cross-stage needs: uncertainty target representation, evaluative explanation, response guidance, and interactive presentation. We instantiate these guidelines in U-Lens, an uncertainty-management support system that organizes uncertain information in long-form responses into contextual inspection targets, prioritizes them for attention, and connects each target with evaluative context and response options. We evaluated U-Lens in a controlled within-subjects study with 18 participants, comparing it against a confidence-cue baseline. Our results show that U-Lens improved verification efficiency and effort allocation, lowered perceived workload, and strengthened perceived support across interpretation, evaluation, and decision stages. This work reframes uncertainty support for generative AI from presenting isolated, text-centered cues toward supporting the user-centered process of interpreting, evaluating, and acting on uncertain information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10590v1">Demographic Prompting at Scale: When More Attributes Hurt LLM--Human Agreement</a></div>
    <div class="paper-meta">
      📅 2026-07-12
    </div>
    <details class="paper-abstract">
      We investigate how annotator demographic attributes, supplied as prompt cues, shape the alignment between large language model (LLM) predictions and human annotations across five tasks. Using five open-source LLMs, we systematically vary the number and composition of demographic components in the prompt, spanning every combination from single-attribute through full-attribute configurations. Our experiments reveal three principal findings. First, alignment consistently peaks with one to three high-signal attributes and degrades under the full attribute set, establishing a clear over-specification threshold. Second, the overall magnitude of demographic influence on human annotations does not predict which attributes improve LLM alignment; instead, both the learnability and the directional coherence of each attribute's annotation signal need to be considered jointly. Third, neuron probing reveals that specialized activation correlates with alignment gains only under coherent annotation signals, and that activation volume alone does not imply steerability. Together, these results demonstrate that demographic prompting is not a monolithic intervention: its utility is highly context-dependent, shaped by attribute signal quality, task characteristics, and model architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.08740v1">Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows</a></div>
    <div class="paper-meta">
      📅 2026-07-09
      | 💬 39 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications increasingly use explicit workflows for tool use, retrieval, branching, checkpointing, and human approval. Existing workflow systems already address many execution concerns. This paper proposes a Lisp-inspired but language-independent conceptual model: symbolic forms, object identity, and live-image thinking are used as explanatory lenses, not implementation commitments. In this model, workflow definitions, workflow instances, inference records, context snapshots, and dependency relations are represented as persistent knowledge objects in a shared knowledge substrate. Its central semantic distinction is between derive and infer: derive is deterministic computation over available state; infer is mediated LLM judgment under declared context and executor-controlled capability policy. The result is a preliminary conceptual account of semantic persistence: workflows do not merely produce knowledge and leave traces, but can themselves be represented as inspectable, resumable, and reviewable knowledge objects, while formal transition semantics remain future work.
    </details>
</div>
