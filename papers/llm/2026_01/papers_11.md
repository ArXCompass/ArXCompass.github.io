# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04653v1">Vibe Coding an LLM-powered Theorem Prover</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      We present Isabellm, an LLM-powered theorem prover for Isabelle/HOL that performs fully automatic proof synthesis. Isabellm works with any local LLM on Ollama and APIs such as Gemini CLI, and it is designed to run on consumer grade computers. The system combines a stepwise prover, which uses large language models to propose proof commands validated by Isabelle in a bounded search loop, with a higher-level proof planner that generates structured Isar outlines and attempts to fill and repair remaining gaps. The framework includes beam search for tactics, tactics reranker ML and RL models, premise selection with small transformer models, micro-RAG for Isar proofs built from AFP, and counter-example guided proof repair. All the code is implemented by GPT 4.1 - 5.2, Gemini 3 Pro, and Claude 4.5. Empirically, Isabellm can prove certain lemmas that defeat Isabelle's standard automation, including Sledgehammer, demonstrating the practical value of LLM-guided proof search. At the same time, we find that even state-of-the-art LLMs, such as GPT 5.2 Extended Thinking and Gemini 3 Pro struggle to reliably implement the intended fill-and-repair mechanisms with complex algorithmic designs, highlighting fundamental challenges in LLM code generation and reasoning. The code of Isabellm is available at https://github.com/zhehou/llm-isabelle
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04620v1">AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Recent progress in large language model (LLM) agents has largely focused on embedding self-improvement mechanisms inside the agent or searching over many concurrent variants. While these approaches can raise aggregate scores, they often yield unstable and hard-to-audit improvement trajectories, making it difficult to guarantee non-regression or to reason about failures across versions. We reframe agent improvement as \textbf{release engineering}: agents are treated as shippable artifacts, and improvement is externalized into a regression-aware release pipeline. We introduce \textbf{AgentDevel}, a release engineering pipeline that iteratively runs the current agent, produces implementation-blind, symptom-level quality signals from execution traces, synthesizes a single release candidate (RC) via executable diagnosis, and promotes it under flip-centered gating. AgentDevel features three core designs: (i) an implementation-blind LLM critic that characterizes failure appearances without accessing agent internals, (ii) script-based executable diagnosis that aggregates dominant symptom patterns and produces auditable engineering specifications, and (iii) flip-centered gating that prioritizes pass to fail regressions and fail to pass fixes as first-class evidence. Unlike population-based search or in-agent self-refinement, AgentDevel maintains a single canonical version line and emphasizes non-regression as a primary objective. Experiments on execution-heavy benchmarks demonstrate that AgentDevel yields stable improvements with significantly fewer regressions while producing reproducible, auditable artifacts. Overall, AgentDevel provides a practical development discipline for building, debugging, and releasing LLM agents as software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14904v2">Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 15 pages,3 figures,5 tables
    </div>
    <details class="paper-abstract">
      Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04597v1">THaLLE-ThaiLLM: Domain-Specialized Small LLMs for Finance and Thai -- Technical Report</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant potential across various domains, particularly in banking and finance, where they can automate complex tasks and enhance decision-making at scale. Due to privacy, security, and regulatory concerns, organizations often prefer on-premise deployment of LLMs. The ThaiLLM initiative aims to enhance Thai language capabilities in open-LLMs, enabling Thai industry to leverage advanced language models. However, organizations often face a trade-off between deploying multiple specialized models versus the prohibitive expense of training a single multi-capability model. To address this, we explore model merging as a resource-efficient alternative for developing high-performance, multi-capability LLMs. We present results from two key experiments: first, merging Qwen-8B with ThaiLLM-8B demonstrates how ThaiLLM-8B enhances Thai general capabilities, showing an uplift of M3 and M6 O-NET exams over the general instruction-following Qwen-8B. Second, we merge Qwen-8B with both ThaiLLM-8B and THaLLE-CFA-8B. This combination results in further improvements in performance across both general and financial domains, by demonstrating an uplift in both M3 and M6 O-NET, Flare-CFA, and Thai-IC benchmarks. The report showcases the viability of model merging for efficiently creating multi-capability LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06346v2">LPFQA: A Long-Tail Professional Forum-based Benchmark for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perform well on standard reasoning and question-answering benchmarks, yet such evaluations often fail to capture their ability to handle long-tail, expertise-intensive knowledge in real-world professional scenarios. We introduce LPFQA, a long-tail knowledge benchmark derived from authentic professional forum discussions, covering 7 academic and industrial domains with 430 curated tasks grounded in practical expertise. LPFQA evaluates specialized reasoning, domain-specific terminology understanding, and contextual interpretation, and adopts a hierarchical difficulty structure to ensure semantic clarity and uniquely identifiable answers. Experiments on over multiple mainstream LLMs reveal substantial performance gaps, particularly on tasks requiring deep domain reasoning, exposing limitations overlooked by existing benchmarks. Overall, LPFQA provides an authentic and discriminative evaluation framework that complements prior benchmarks and informs future LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07107v2">MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR first performs structured self-assessment through simulated critical thinking, such as perspective-taking and consequential reasoning to uncover latent model misalignments. These reflections are formalized into dynamic rule-based knowledge graphs that evolve with emerging risk patterns. To enforce these rules at inference time, we introduce activation steering, a method that directly modulates the model's internal representations to ensure compliance. Experiments demonstrate that MENTOR substantially reduces attack success rates across all tested domains and achieves risk analysis performance comparable to human experts. Our work offers a scalable and adaptive pathway toward robust domain-specific alignment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02393v4">AP2O-Coder: Adaptively Progressive Preference Optimization for Reducing Compilation and Runtime Errors in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      LLMs' code generation capabilities have yielded substantial improvements in the effectiveness of programming tasks. However, LLM-generated code still suffers from compilation and runtime errors. Existing offline preference optimization methods primarily focus on enhancing LLMs' coding abilities using pass/fail signals in the preference data, overlooking the deep-level error types in the failed codes. To address this, we propose Adaptively Progressive Preference Optimization (AP2O) for coding (i.e., AP2O-Coder), a method that guides LLMs adaptively and methodically to reduce code errors for code generation. Specifically, we construct an error notebook from failed codes and progressively optimize the LLM to correct errors type by type. Furthermore, we adaptively replay error types to tailor to the LLM's changing weaknesses throughout the training process. Through extensive experiments on both code and general LLMs (Llama, Qwen, and DeepSeek series) with parameters ranging from 0.5B to 34B, our AP2O-Coder improves code generation performance by up to 3% in pass@k while using less preference data. Code: https://github.com/TsingZ0/AP2O
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04574v1">FeedEval: Pedagogically Aligned Evaluation of LLM-Generated Essay Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Going beyond the prediction of numerical scores, recent research in automated essay scoring has increasingly emphasized the generation of high-quality feedback that provides justification and actionable guidance. To mitigate the high cost of expert annotation, prior work has commonly relied on LLM-generated feedback to train essay assessment models. However, such feedback is often incorporated without explicit quality validation, resulting in the propagation of noise in downstream applications. To address this limitation, we propose FeedEval, an LLM-based framework for evaluating LLM-generated essay feedback along three pedagogically grounded dimensions: specificity, helpfulness, and validity. FeedEval employs dimension-specialized LLM evaluators trained on datasets curated in this study to assess multiple feedback candidates and select high-quality feedback for downstream use. Experiments on the ASAP++ benchmark show that FeedEval closely aligns with human expert judgments and that essay scoring models trained with FeedEval-filtered high-quality feedback achieve superior scoring performance. Furthermore, revision experiments using small LLMs show that the high-quality feedback identified by FeedEval leads to more effective essay revisions. We will release our code and curated datasets upon accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19361v3">AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic method for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04566v1">BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04562v1">Reasoning Over Space: Enabling Geographic Reasoning for LLM-Based Generative Next POI Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Generative recommendation with large language models (LLMs) reframes prediction as sequence generation, yet existing LLM-based recommenders remain limited in leveraging geographic signals that are crucial in mobility and local-services scenarios. Here, we present Reasoning Over Space (ROS), a framework that utilizes geography as a vital decision variable within the reasoning process. ROS introduces a Hierarchical Spatial Semantic ID (SID) that discretizes coarse-to-fine locality and POI semantics into compositional tokens, and endows LLM with a three-stage Mobility Chain-of-Thought (CoT) paradigm that models user personality, constructs an intent-aligned candidate space, and performs locality informed pruning. We further align the model with real world geography via spatial-guided Reinforcement Learning (RL). Experiments on three widely used location-based social network (LBSN) datasets show that ROS achieves over 10% relative gains in hit rate over strongest LLM-based baselines and improves cross-city transfer, despite using a smaller backbone model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04556v1">4D-ARE: Bridging the Attribution Gap in LLM Agent Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 39 pages, 11 tables
    </div>
    <details class="paper-abstract">
      We deployed an LLM agent with ReAct reasoning and full data access. It executed flawlessly, yet when asked "Why is completion rate 80%?", it returned metrics instead of causal explanation. The agent knew how to reason but we had not specified what to reason about. This reflects a gap: runtime reasoning frameworks (ReAct, Chain-of-Thought) have transformed LLM agents, but design-time specification--determining what domain knowledge agents need--remains under-explored. We propose 4D-ARE (4-Dimensional Attribution-Driven Agent Requirements Engineering), a preliminary methodology for specifying attribution-driven agents. The core insight: decision-makers seek attribution, not answers. Attribution concerns organize into four dimensions (Results -> Process -> Support -> Long-term), motivated by Pearl's causal hierarchy. The framework operationalizes through five layers producing artifacts that compile directly to system prompts. We demonstrate the methodology through an industrial pilot deployment in financial services. 4D-ARE addresses what agents should reason about, complementing runtime frameworks that address how. We hypothesize systematic specification amplifies the power of these foundational advances. This paper presents a methodological proposal with preliminary industrial validation; rigorous empirical evaluation is planned for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04548v1">Identifying Good and Bad Neurons for Task-Level Controllable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated remarkable capabilities on multiple-choice question answering benchmarks, but the complex mechanisms underlying their large-scale neurons remain opaque, posing significant challenges for understanding and steering LLMs. While recent studies made progress on identifying responsible neurons for certain abilities, these ability-specific methods are infeasible for task-focused scenarios requiring coordinated use of multiple abilities. Moreover, these approaches focus only on supportive neurons that correlate positively with task completion, while neglecting neurons with other roles-such as inhibitive roles-and misled neuron attribution due to fortuitous behaviors in LLMs (i.e., correctly answer the questions by chance rather than genuine understanding). To address these challenges, we propose NeuronLLM, a novel task-level LLM understanding framework that adopts the biological principle of functional antagonism for LLM neuron identification. The key insight is that task performance is jointly determined by neurons with two opposing roles: good neurons that facilitate task completion and bad neurons that inhibit it. NeuronLLM achieves a holistic modeling of neurons via contrastive learning of good and bad neurons, while leveraging augmented question sets to mitigate the fortuitous behaviors in LLMs. Comprehensive experiments on LLMs of different sizes and families show the superiority of NeuronLLM over existing methods in four NLP tasks, providing new insights into LLM functional organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v4">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03508v3">One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Evaluating LLMs' instruction-following ability in multi-topic dialogues is essential yet challenging. Existing benchmarks are limited to a fixed number of turns, susceptible to saturation and failing to account for users' interactive experience. In this work, we propose a novel framework featuring a three-layer tracking mechanism and a query synthesis agent to mimic sequential user behaviors. Grounded in Flow Theory, we introduce process-centric metrics and terminate a conversational evaluation only upon exhausting user patience. Leveraging this framework, we present EvolIF, an evolving benchmark covering 12 constraint groups. Our analysis reveals deficiencies in failure recovery and fine-grained instruction following, with performance stratification becoming evident as conversational depth increases. GPT-5 demonstrates the most sustained resilience, maintaining a 66.40% robustness score, outperforming Gemini-3-Pro by 5.59%, while other models lag behind. Data and code will be released at https://github.com/JiaQiSJTU/EvolIF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24307v2">Exploring Similarity between Neural and LLM Trajectories in Language Processing</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Understanding the similarity between large language models (LLMs) and human brain activity is crucial for advancing both AI and cognitive neuroscience. In this study, we provide a multilinguistic, large-scale assessment of this similarity by systematically comparing 16 publicly available pretrained LLMs with human brain responses during natural language processing tasks in both English and Chinese. Specifically, we use ridge regression to assess the representational similarity between LLM embeddings and electroencephalography (EEG) signals, and analyze the similarity between the "neural trajectory" and the "LLM latent trajectory." This method captures key dynamic patterns, such as magnitude, angle, uncertainty, and confidence. Our findings highlight both similarities and crucial differences in processing strategies: (1) We show that middle-to-high layers of LLMs are central to semantic integration and correspond to the N400 component observed in EEG; (2) The brain exhibits continuous and iterative processing during reading, whereas LLMs often show discrete, stage-end bursts of activity, which suggests a stark contrast in their real-time semantic processing dynamics. This study could offer new insights into LLMs and neural processing, and also establish a critical framework for future investigations into the alignment between artificial intelligence and biological intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.03135v2">Beyond Retrieval: Improving Evidence Quality for LLM-based Multimodal Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      The increasing multimodal disinformation, where deceptive claims are reinforced through coordinated text and visual content, poses significant challenges to automated fact-checking. Recent efforts leverage Large Language Models (LLMs) for this task, capitalizing on their strong reasoning and multimodal understanding capabilities. Emerging retrieval-augmented frameworks further equip LLMs with access to open-domain external information, enabling evidence-based verification beyond their internal knowledge. Despite their promising gains, our empirical study reveals notable shortcomings in the external search coverage and evidence quality evaluation. To mitigate those limitations, we propose Aletheia, an end-to-end framework for automated multimodal fact-checking. It introduces a novel evidence retrieval strategy that improves evidence coverage and filters useless information from open-domain sources, enabling the extraction of high-quality evidence for verification. Extensive experiments demonstrate that Aletheia achieves an accuracy of 88.3% on two public multimodal disinformation datasets and 90.2% on newly emerging claims. Compared with existing evidence retrieval strategies, our approach improves verification accuracy by up to 30.8%, highlighting the critical role of evidence quality in LLM-based disinformation verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04537v1">Not All Steps are Informative: On the Linearity of LLMs' RLVR Training</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on all four benchmarks by extrapolating beyond the step range where RL training remains stable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11423v2">Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), allows users to flag misleading posts, attach contextual notes, and rate the notes' helpfulness. However, our empirical analysis of 30.8K health-related notes reveals substantial latency, with a median delay of 17.6 hours before notes receive a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified LLM-based framework that augments Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, supported by a hierarchical three-stage evaluation of relevance, correctness, and helpfulness. We instantiate the framework with HealthNotes, a benchmark of 1.2K health notes annotated for helpfulness, and a fine-tuned helpfulness judge. Our analysis first uncovers a key loophole in current crowd-sourced governance: voters frequently conflate stylistic fluency with factual accuracy. Addressing this via our hierarchical evaluation, experiments across 15 representative LLMs demonstrate that CrowdNotes+ significantly outperforms human contributors in note correctness, helpfulness, and evidence utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04505v1">CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Under review, 13 pages, 11 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronics design, as large language models (LLMs) frequently hallucinate in granular details, violate electrical constraints, and produce non-machine-readable outputs. We present CircuitLM, a novel multi-agent LLM-aided circuit design pipeline that translates user prompts into structured, visually interpretable CircuitJSON schematics through five sequential stages: (i) LLM-based component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning by an electronics expert agent, (iv) JSON schematic synthesis, and (v) force-directed SVG visualization. Anchored by a curated, embedding-powered component knowledge base. While LLMs often violate electrical constraints, CircuitLM bridges this gap by grounding generation in a verified and dynamically extensible component database, initially comprising 50 components. To ensure safety, we incorporate a hybrid evaluation framework, namely Dual-Metric Circuit Validation (DMCV), validated against human-expert assessments, which achieves high fidelity in microcontroller-centric designs. We evaluate the system on 100 diverse embedded-systems prompts across six LLMs and introduce DMCV to assess both structural and electrical validity. This work bridges natural language input to deployable hardware designs, enabling reliable circuit prototyping by non-experts. Our code and data will be made public upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04491v1">A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 6 pages, 6 figures, 6 tables, Conference: Robotics, Automation, and Artificial Intelligence 2025
    </div>
    <details class="paper-abstract">
      Personalized nutrition management aims to tailor dietary guidance to an individual's intake and phenotype, but most existing systems handle food logging, nutrient analysis and recommendation separately. We present a next-generation mobile nutrition assistant that combines image based meal logging with an LLM driven multi agent controller to provide meal level closed loop support. The system coordinates vision, dialogue and state management agents to estimate nutrients from photos and update a daily intake budget. It then adapts the next meal plan to user preferences and dietary constraints. Experiments with SNAPMe meal images and simulated users show competitive nutrient estimation, personalized menus and efficient task plans. These findings demonstrate the feasibility of multi agent LLM control for personalized nutrition and reveal open challenges in micronutrient estimation from images and in large scale real world studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06216v1">LLM Agents in Law: Taxonomy, Applications, and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have precipitated a dramatic improvement in the legal domain, yet the deployment of standalone models faces significant limitations regarding hallucination, outdated information, and verifiability. Recently, LLM agents have attracted significant attention as a solution to these challenges, utilizing advanced capabilities such as planning, memory, and tool usage to meet the rigorous standards of legal practice. In this paper, we present a comprehensive survey of LLM agents for legal tasks, analyzing how these architectures bridge the gap between technical capabilities and domain-specific needs. Our major contributions include: (1) systematically analyzing the technical transition from standard legal LLMs to legal agents; (2) presenting a structured taxonomy of current agent applications across distinct legal practice areas; (3) discussing evaluation methodologies specifically for agentic performance in law; and (4) identifying open challenges and outlining future directions for developing robust and autonomous legal assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03627v2">Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 EACL 2026 Industry
    </div>
    <details class="paper-abstract">
      We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on https://github.com/seemdog/EPAG, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04463v1">Beyond Static Summarization: Proactive Memory Extraction for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Memory management is vital for LLM agents to handle long-term interaction and personalization. Most research focuses on how to organize and use memory summary, but often overlooks the initial memory extraction stage. In this paper, we argue that existing summary-based methods have two major limitations based on the recurrent processing theory. First, summarization is "ahead-of-time", acting as a blind "feed-forward" process that misses important details because it doesn't know future tasks. Second, extraction is usually "one-off", lacking a feedback loop to verify facts, which leads to the accumulation of information loss. To address these issues, we propose proactive memory extraction (namely ProMem). Unlike static summarization, ProMem treats extraction as an iterative cognitive process. We introduce a recurrent feedback loop where the agent uses self-questioning to actively probe the dialogue history. This mechanism allows the agent to recover missing information and correct errors. Our ProMem significantly improves the completeness of the extracted memory and QA accuracy. It also achieves a superior trade-off between extraction quality and token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.03505v3">Dynamic and Adaptive Feature Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      The representation of feature space is a crucial environment where data points get vectorized and embedded for subsequent modeling. Thus the efficacy of machine learning (ML) algorithms is closely related to the quality of feature engineering. As one of the most important techniques, feature generation transforms raw data into an optimized feature space conducive to model training and further refines the space. Despite the advancements in automated feature engineering and feature generation, current methodologies often suffer from three fundamental issues: lack of explainability, limited applicability, and inflexible strategy. These shortcomings frequently hinder and limit the deployment of ML models across varied scenarios. Our research introduces a novel approach adopting large language models (LLMs) and feature-generating prompts to address these challenges. We propose a dynamic and adaptive feature generation method that enhances the interpretability of the feature generation process. Our approach broadens the applicability across various data types and tasks and offers advantages over strategic flexibility. A broad range of experiments showcases that our approach is significantly superior to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05420v1">Efficient Inference for Noisy LLM-as-a-Judge Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators of generative AI outputs, a paradigm often referred to as "LLM-as-a-judge." In practice, LLM judges are imperfect predictions for the underlying truth and can exhibit systematic, non-random errors. Two main approaches have recently been proposed to address this issue: (i) direct measurementerror correction based on misclassification models such as Rogan-Gladen-style estimators, and (ii) surrogate-outcome approaches such as prediction-powered inference (PPI), which correct bias by calibrating prediction residuals on a small set of gold-standard human labels. In this paper, we systematically study the performance of these two approaches for estimating mean parameters (e.g., average benchmark scores or pairwise win rates). Leveraging tools from semiparametric efficiency theory, we unify the two classes of estimators by deriving explicit forms of efficient influence function (EIF)-based efficient estimators and characterize conditions under which PPI-style estimators attain strictly smaller asymptotic variance than measurement-error corrections. We verify our theoretical results in simulations and demonstrate the methods on real-data examples. We provide an implementation of the benchmarked methods and comparison utilities at https://github.com/yiqunchen/debias-llm-as-a-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05414v1">Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines across domains like educational assessment and synthetic data construction, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces N=1000 samples within one response, and Independent Requests, comprising $N=1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 13% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the requested sampling horizon N increases. Finally, we demonstrate the propagation of these failures into downstream tasks: models fail to enforce uniform answer-position constraints in MCQ generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating the use of external tools for applications requiring statistical guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05385v1">DafnyPro: LLM-Assisted Automated Verification for Dafny Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      We present DafnyPro, an inference-time framework that enhances LLMs for generating verification annotations in Dafny. DafnyPro comprises three key components: a diff-checker that prevents modifications to base program logic, a pruner that removes unnecessary invariants, and a hint-augmentation system that retrieves and applies predefined, problem-independent proof strategies. We evaluate DafnyPro using Claude Sonnet 3.5 and 3.7 on four benchmarks: Clover, MBPP-Dafny, HumanEval-Dafny, and DafnyBench, achieving consistent performance gains in all cases. Notably, on DafnyBench, the most challenging benchmark, Claude Sonnet 3.5 enhanced with DafnyPro achieves 86% correct proofs, a 16 pp improvement over the base model. We also fine-tune two Qwen models on training data derived from verification attempts by larger models enhanced with DafnyPro. Our 7B and 14B models achieve 68% and 70% correct proofs on DafnyBench, respectively, demonstrating that smaller models can maintain high verification accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18839v4">Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05307v1">The LLM Mirage: Economic Interests and the Subversion of Weaponization Controls</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      U.S. AI security policy is increasingly shaped by an $\textit{LLM Mirage}$, the belief that national security risks scale in proportion to the compute used to train frontier language models. That premise fails in two ways. It miscalibrates strategy because adversaries can obtain weaponizable capabilities with task-specific systems that use specialized data, algorithmic efficiency, and widely available hardware, while compute controls harden only a high-end perimeter. It also destabilizes regulation because, absent a settled definition of "AI weaponization," compute thresholds are easily renegotiated as domestic priorities shift, turning security policy into a proxy contest over industrial competitiveness. We analyze how the LLM Mirage took hold, propose an intent-and-capability definition of AI weaponization grounded in effects and international humanitarian law, and outline measurement infrastructure based on live benchmarks across the full AI Triad (data, algorithms, compute) for weaponization-relevant capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05192v1">LELA: an LLM-based Entity Linking Approach with Zero-Shot Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Entity linking (mapping ambiguous mentions in text to entities in a knowledge base) is a foundational step in tasks such as knowledge graph construction, question-answering, and information extraction. Our method, LELA, is a modular coarse-to-fine approach that leverages the capabilities of large language models (LLMs), and works with different target domains, knowledge bases and LLMs, without any fine-tuning phase. Our experiments across various entity linking settings show that LELA is highly competitive with fine-tuned approaches, and substantially outperforms the non-fine-tuned ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05187v1">SimuAgent: An LLM-Based Simulink Modeling Assistant Enhanced with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized text-based code automation, but their potential in graph-oriented engineering workflows remains under-explored. We introduce SimuAgent, an LLM-powered modeling and simulation agent tailored for Simulink. SimuAgent replaces verbose XML with a concise, dictionary-style Python representation, dramatically cutting token counts, improving interpretability, and enabling fast, in-process simulation. A lightweight plan-execute architecture, trained in two stages, equips the agent with both low-level tool skills and high-level design reasoning. To tackle sparse rewards in long-horizon tasks, we propose Reflection-GRPO (ReGRPO), which augments Group Relative Policy Optimization (GRPO) with self-reflection traces that supply rich intermediate feedback, accelerating convergence and boosting robustness. Experiments on SimuBench, our newly released benchmark comprising 5300 multi-domain modeling tasks, show that a Qwen2.5-7B model fine-tuned with SimuAgent converges faster and achieves higher modeling accuracy than standard RL baselines, and even surpasses GPT-4o when evaluated with few-shot prompting on the same benchmark. Ablations confirm that the two-stage curriculum and abstract-reconstruct data augmentation further enhance generalization. SimuAgent trains and runs entirely on-premise with modest hardware, delivering a privacy-preserving, cost-effective solution for industrial model-driven engineering. SimuAgent bridges the gap between LLMs and graphical modeling environments, offering a practical solution for AI-assisted engineering design in industrial settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.19850v3">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Finding information in hour-long videos is a challenging task even for top-performing Vision Language Models (VLMs), as encoding visual content quickly exceeds available context windows. To tackle this challenge, we present FALCONEye, a novel video agent based on a training-free, model-agnostic meta-architecture composed of a VLM and a Large Language Model (LLM). FALCONEye answers open-ended questions using an exploration-based search algorithm guided by calibrated confidence from the VLM's answers. We also introduce the FALCON-Bench benchmark, extending Question Answering problem to Video Answer Search-requiring models to return both the answer and its supporting temporal window for open-ended questions in hour-long videos. With just a 7B VLM and a lightweight LLM, FALCONEye outscores all open-source 7B VLMs and comparable agents in FALCON-Bench. It further demonstrates its generalization capability in MLVU benchmark with shorter videos and different tasks, surpassing GPT-4o on single-detail tasks while slashing inference cost by roughly an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05114v1">Evaluative Fingerprints: Stable and Systematic Differences in LLM Evaluator Behavior</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 23 pages, 6 figures, code and artifacts at : https://github.com/wajid-nasser/evaluative-fingerprints
    </div>
    <details class="paper-abstract">
      LLM-as-judge systems promise scalable, consistent evaluation. We find the opposite: judges are consistent, but not with each other; they are consistent with themselves. Across 3,240 evaluations (9 judges x 120 unique video x pack items x 3 independent runs), inter-judge agreement is near-zero (Krippendorff's α = 0.042). On two dimensions, judges disagree more than random noise would predict (α < 0). Yet this disagreement isn't chaos; it's structured. A classifier identifies which judge produced an evaluation with 77.1% accuracy from rubric scores alone, rising to 89.9% with disposition features. Within model families, the signal is even stronger: GPT-4.1 and GPT-5.2 are distinguishable with 99.6% accuracy. We call this the reliability paradox: judges cannot agree on what constitutes quality, yet their disagreement patterns are so stable they function as fingerprints. Each judge implements a distinct, stable theory of quality: an "evaluative disposition" that shapes how it interprets any rubric. We characterize these dispositions along multiple axes: harshness/leniency, dimension emphasis, within-judge stability (ICC), and evidence behavior (receipt validity, semantic linkage via NLI, and shotgun index). The implication is stark: LLM judges are not interchangeable instruments measuring a shared construct. They are distinct measurement devices, each encoding its own implicit theory of quality. Averaging their scores produces a synthetic verdict that corresponds to no judge's actual values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05106v1">Token-Level LLM Collaboration via FusionRoute</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21007v3">Can Confidence Estimates Decide When Chain-of-Thought Is Necessary for LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) prompting is a common technique for improving the reasoning abilities of large language models (LLMs). However, extended reasoning is often unnecessary and substantially increases token usage. As such, a key question becomes how to optimally allocate compute to when reasoning is actually needed. We study this through confidence-gated CoT, where a model produces a direct answer and a confidence estimate to decide whether to invoke CoT. We present an evaluation framework together with the first systematic study of confidence signals for this decision. We evaluate four representative confidence measures and compare them with random gating and an oracle upper bound. Experiments across two model families and diverse reasoning tasks show that existing training-free confidence measures can reduce redundant reasoning. However, we also find that the utility of individual confidence measures is inconsistent across settings. Through our evaluation framework and analysis, our study provides practical guidance toward developing and evaluating models that selectively use CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17189v4">Talking with Tables for Better LLM Factual Data Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 20 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with requests related to information retrieval and data manipulation that frequently arise in real-world scenarios under multiple conditions. In this paper, we demonstrate that leveraging tabular structures in LLM interactions, is more effective than utilizing other structures for handling prevalent requests that operate over factual data. Through comprehensive evaluations across various scenarios and request types, we show that providing tabular structures yields a 40.29\% average performance gain along with better robustness and token efficiency. Through attention-value analysis, we discover that tables help LLMs better locate relevant information, explaining these improvements. Beyond tables and text, we evaluate whether (1) blending structuredness within text, such as providing templates or fixing the order of attributes, and (2) other representative structures, such as knowledge graphs and JSON are helpful. We observe that utilizing tables offers the best balance between efficiency and effectiveness. The method remains robust to task complexity and adapts to unstructured sources through text-to-table conversion. Overall, we highlight the untapped potential of tabular representations for future LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13691v3">Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using five strategically selected datasets: EPFL PhD manuscripts, a private collection of Venetian historical records, two sets of Wikipedia articles on related topics, and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05022v1">Knowledge-to-Data: LLM-Driven Synthesis of Structured Network Traffic for Testbed-Free IDS Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Realistic, large-scale, and well-labeled cybersecurity datasets are essential for training and evaluating Intrusion Detection Systems (IDS). However, they remain difficult to obtain due to privacy constraints, data sensitivity, and the cost of building controlled collection environments such as testbeds and cyber ranges. This paper investigates whether Large Language Models (LLMs) can operate as controlled knowledge-to-data engines for generating structured synthetic network traffic datasets suitable for IDS research. We propose a methodology that combines protocol documentation, attack semantics, and explicit statistical rules to condition LLMs without fine-tuning or access to raw samples. Using the AWID3 IEEE~802.11 benchmark as a demanding case study, we generate labeled datasets with four state-of-the-art LLMs and assess fidelity through a multi-level validation framework including global similarity metrics, per-feature distribution testing, structural comparison, and cross-domain classification. Results show that, under explicit constraints, LLM-generated datasets can closely approximate the statistical and structural characteristics of real network traffic, enabling gradient-boosting classifiers to achieve F1-scores up to 0.956 when evaluated on real samples. Overall, the findings suggest that constrained LLM-driven generation can facilitate on-demand IDS experimentation, providing a testbed-free, privacy-preserving alternative that overcomes the traditional bottlenecks of physical traffic collection and manual labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14195v2">N-GLARE: An Non-Generative Latent Representation-Efficient LLM Safety Evaluator</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Evaluating the safety robustness of LLMs is critical for their deployment. However, mainstream Red Teaming methods rely on online generation and black-box output analysis. These approaches are not only costly but also suffer from feedback latency, making them unsuitable for agile diagnostics after training a new model. To address this, we propose N-GLARE (A Non-Generative, Latent Representation-Efficient LLM Safety Evaluator). N-GLARE operates entirely on the model's latent representations, bypassing the need for full text generation. It characterizes hidden layer dynamics by analyzing the APT (Angular-Probabilistic Trajectory) of latent representations and introducing the JSS (Jensen-Shannon Separability) metric. Experiments on over 40 models and 20 red teaming strategies demonstrate that the JSS metric exhibits high consistency with the safety rankings derived from Red Teaming. N-GLARE reproduces the discriminative trends of large-scale red-teaming tests at less than 1\% of the token cost and the runtime cost, providing an efficient output-free evaluation proxy for real-time diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04940v1">CurricuLLM: Designing Personalized and Workforce-Aligned Cybersecurity Curricula Using Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      The cybersecurity landscape is constantly evolving, driven by increased digitalization and new cybersecurity threats. Cybersecurity programs often fail to equip graduates with skills demanded by the workforce, particularly concerning recent developments in cybersecurity, as curriculum design is costly and labor-intensive. To address this misalignment, we present a novel Large Language Model (LLM)-based framework for automated design and analysis of cybersecurity curricula, called CurricuLLM. Our approach provides three key contributions: (1) automation of personalized curriculum design, (2) a data-driven pipeline aligned with industry demands, and (3) a comprehensive methodology for leveraging fine-tuned LLMs in curriculum development. CurricuLLM utilizes a two-tier approach consisting of PreprocessLM, which standardizes input data, and ClassifyLM, which assigns course content to nine Knowledge Areas in cybersecurity. We systematically evaluated multiple Natural Language Processing (NLP) architectures and fine-tuning strategies, ultimately selecting the Bidirectional Encoder Representations from Transformers (BERT) model as ClassifyLM, fine-tuned on foundational cybersecurity concepts and workforce competencies. We are the first to validate our method with human experts who analyzed real-world cybersecurity curricula and frameworks, motivating that CurricuLLM is an efficient solution to replace labor-intensive curriculum analysis. Moreover, once course content has been classified, it can be integrated with established cybersecurity role-based weights, enabling alignment of the educational program with specific job roles, workforce categories, or general market needs. This lays the foundation for personalized, workforce-aligned cybersecurity curricula that prepare students for the evolving demands in cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15949v2">ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language models show promise for financial decision-making, yet deploying them as autonomous trading agents raises fundamental challenges: how to adapt instructions when rewards arrive late and obscured by market noise, how to synthesize heterogeneous information streams into coherent decisions, and how to bridge the gap between model outputs and executable market actions. We present ATLAS (Adaptive Trading with LLM AgentS), a unified multi-agent framework that integrates structured information from markets, news, and corporate fundamentals to support robust trading decisions. Within ATLAS, the central trading agent operates in an order-aware action space, ensuring that outputs correspond to executable market orders rather than abstract signals. The agent can incorporate feedback while trading using Adaptive-OPRO, a novel prompt-optimization technique that dynamically adapts the prompt by incorporating real-time, stochastic feedback, leading to increasing performance over time. Across regime-specific equity studies and multiple LLM families, Adaptive-OPRO consistently outperforms fixed prompts, while reflection-based feedback fails to provide systematic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04885v1">CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) serve a global audience, alignment must transition from enforcing universal consensus to respecting cultural pluralism. We demonstrate that dense models, when forced to fit conflicting value distributions, suffer from \textbf{Mean Collapse}, converging to a generic average that fails to represent diverse groups. We attribute this to \textbf{Cultural Sparsity}, where gradient interference prevents dense parameters from spanning distinct cultural modes. To resolve this, we propose \textbf{\textsc{CuMA}} (\textbf{Cu}ltural \textbf{M}ixture of \textbf{A}dapters), a framework that frames alignment as a \textbf{conditional capacity separation} problem. By incorporating demographic-aware routing, \textsc{CuMA} internalizes a \textit{Latent Cultural Topology} to explicitly disentangle conflicting gradients into specialized expert subspaces. Extensive evaluations on WorldValuesBench, Community Alignment, and PRISM demonstrate that \textsc{CuMA} achieves state-of-the-art performance, significantly outperforming both dense baselines and semantic-only MoEs. Crucially, our analysis confirms that \textsc{CuMA} effectively mitigates mean collapse, preserving cultural diversity. Our code is available at https://github.com/Throll/CuMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12611v3">An LLM + ASP Workflow for Joint Entity-Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 In Proceedings ICLP 2025, arXiv:2601.00047
    </div>
    <details class="paper-abstract">
      Joint entity-relation extraction (JERE) identifies both entities and their relationships simultaneously. Traditional machine-learning based approaches to performing this task require a large corpus of annotated data and lack the ability to easily incorporate domain specific information in the construction of the model. Therefore, creating a model for JERE is often labor intensive, time consuming, and elaboration intolerant. In this paper, we propose harnessing the capabilities of generative pre-trained large language models (LLMs) and the knowledge representation and reasoning capabilities of Answer Set Programming (ASP) to perform JERE. We present a generic workflow for JERE using LLMs and ASP. The workflow is generic in the sense that it can be applied for JERE in any domain. It takes advantage of LLM's capability in natural language understanding in that it works directly with unannotated text. It exploits the elaboration tolerant feature of ASP in that no modification of its core program is required when additional domain specific knowledge, in the form of type specifications, is found and needs to be used. We demonstrate the usefulness of the proposed workflow through experiments with limited training data on three well-known benchmarks for JERE. The results of our experiments show that the LLM + ASP workflow is better than state-of-the-art JERE systems in several categories with only 10% of training data. It is able to achieve a 2.5 times (35% over 15%) improvement in the Relation Extraction task for the SciERC corpus, one of the most difficult benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04801v1">MPM-LLM4DSE: Reaching the Pareto Frontier in HLS with Multimodal Learning and LLM-Driven Exploration</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      High-Level Synthesis (HLS) design space exploration (DSE) seeks Pareto-optimal designs within expansive pragma configuration spaces. To accelerate HLS DSE, graph neural networks (GNNs) are commonly employed as surrogates for HLS tools to predict quality of results (QoR) metrics, while multi-objective optimization algorithms expedite the exploration. However, GNN-based prediction methods may not fully capture the rich semantic features inherent in behavioral descriptions, and conventional multi-objective optimization algorithms often do not explicitly account for the domain-specific knowledge regarding how pragma directives influence QoR. To address these limitations, this paper proposes the MPM-LLM4DSE framework, which incorporates a multimodal prediction model (MPM) that simultaneously fuses features from behavioral descriptions and control and data flow graphs. Furthermore, the framework employs a large language model (LLM) as an optimizer, accompanied by a tailored prompt engineering methodology. This methodology incorporates pragma impact analysis on QoR to guide the LLM in generating high-quality configurations (LLM4DSE). Experimental results demonstrate that our multimodal predictive model significantly outperforms state-of-the-art work ProgSG by up to 10.25$\times$. Furthermore, in DSE tasks, the proposed LLM4DSE achieves an average performance gain of 39.90\% over prior methods, validating the effectiveness of our prompting methodology. Code and models are available at https://github.com/wslcccc/MPM-LLM4DSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05547v2">Automated Invoice Data Extraction: Using LLM and OCR</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low-quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01014v2">IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Instruction-following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction-following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic for fine-grained, efficient, and reliable instruction-following evaluation. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments show that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including o4-mini and Gemini-3-Pro. With the reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v1">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04740v1">RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12225v3">Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted by KDD 2026 (Research Track). Project page: https://aster2024.github.io/swift-website/
    </div>
    <details class="paper-abstract">
      Best-of-N sampling is a powerful method for improving Large Language Model (LLM) performance, but it is often limited by its dependence on massive, text-based reward models. These models are not only computationally expensive but also data-hungry, requiring extensive labeled datasets for training. This creates a significant data challenge, as they overlook a rich, readily available data source: the LLM's own internal hidden states. To address this data and efficiency gap, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel and lightweight method that learns a reward function directly from the rich information embedded in LLM hidden states. Operating at the token embedding level, SWIFT employs simple linear layers to effectively distinguish between preferred and dispreferred generations, eliminating the need for computationally intensive text-based modeling. Extensive experiments on standard benchmarks show that SWIFT outperforms existing baselines (12.7% higher accuracy than EurusRM-7B on MATH dataset) while using less than 0.005% of their parameters. Its robust scalability, compatibility with certain closed-source models via logit access, and ability to combine with traditional reward models for additional performance highlight SWIFT's practical value and contribution to more efficient data-driven LLM post-training. Our code is available at https://github.com/aster2024/SWIFT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21830v4">GAPO: Robust Advantage Estimation for Real-World Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods, such as GRPO, are popular due to their critic-free and normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable noise, leading to distorted advantage computation and increased rollout outliers. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an interval with the highest SNR (Signal to Noise Ratio) per prompt and uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation to reduce noise further. This adaptive Q robustly handles rollout noise while remaining plug-and-play and efficient. We evaluate GAPO on nine instruction-tuned LLMs (3B-14B) using a collected large dataset of 51,844 real-world, history-aware code-editing tasks spanning 10 programming languages. GAPO yields up to 4.35 in-domain (ID) and 5.30 out-of-domain (OOD) exact-match improvements over GRPO and its variant DAPO, while achieving lower clipping ratios and higher GPU throughput. Code: https://github.com/TsingZ0/verl-GAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04711v1">DSC2025 -- ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      The reliability of large language models (LLMs) in production environments remains significantly constrained by their propensity to generate hallucinations -- fluent, plausible-sounding outputs that contradict or fabricate information. While hallucination detection has recently emerged as a priority in English-centric benchmarks, low-to-medium resource languages such as Vietnamese remain inadequately covered by standardized evaluation frameworks. This paper introduces the DSC2025 ViHallu Challenge, the first large-scale shared task for detecting hallucinations in Vietnamese LLMs. We present the ViHallu dataset, comprising 10,000 annotated triplets of (context, prompt, response) samples systematically partitioned into three hallucination categories: no hallucination, intrinsic, and extrinsic hallucinations. The dataset incorporates three prompt types -- factual, noisy, and adversarial -- to stress-test model robustness. A total of 111 teams participated, with the best-performing system achieving a macro-F1 score of 84.80\%, compared to a baseline encoder-only score of 32.83\%, demonstrating that instruction-tuned LLMs with structured prompting and ensemble strategies substantially outperform generic architectures. However, the gap to perfect performance indicates that hallucination detection remains a challenging problem, particularly for intrinsic (contradiction-based) hallucinations. This work establishes a rigorous benchmark and explores a diverse range of detection methodologies, providing a foundation for future research into the trustworthiness and reliability of Vietnamese language AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04710v1">Prior-Informed Zeroth-Order Optimization with Adaptive Direction Alignment for Memory-Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 12pages, 6figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) has achieved remarkable success across various NLP tasks, but the substantial memory overhead during backpropagation remains a critical bottleneck, especially as model scales grow. Zeroth-order (ZO) optimization alleviates this issue by estimating gradients through forward passes and Gaussian sampling, avoiding the need for backpropagation. However, conventional ZO methods suffer from high variance in gradient estimation due to their reliance on random perturbations, leading to slow convergence and suboptimal performance. We propose a simple plug-and-play method that incorporates prior-informed perturbations to refine gradient estimation. Our method dynamically computes a guiding vector from Gaussian samples, which directs perturbations toward more informative directions, significantly accelerating convergence compared to standard ZO approaches. We further investigate a greedy perturbation strategy to explore the impact of prior knowledge on gradient estimation. Theoretically, we prove that our gradient estimator achieves stronger alignment with the true gradient direction, enhancing optimization efficiency. Extensive experiments across LLMs of varying scales and architectures demonstrate that our proposed method could seamlessly integrate into existing optimization methods, delivering faster convergence and superior performance. Notably, on the OPT-13B model, our method outperforms traditional ZO optimization across all 11 benchmark tasks and surpasses gradient-based baselines on 9 out of 11 tasks, establishing a robust balance between efficiency and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10029v2">Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable progress, they remain vulnerable to jailbreak attacks. Existing methods, primarily relying on discrete input optimization (e.g., GCG), often suffer from high computational costs and generate high-perplexity prompts that are easily blocked by simple filters. To overcome these limitations, we propose Latent Fusion Jailbreak (LFJ), a stealthy white-box attack that operates in the continuous latent space. Unlike previous approaches, LFJ constructs adversarial representations by mathematically fusing the hidden states of a harmful query with a thematically similar benign query, effectively masking malicious intent while retaining semantic drive. We further introduce a gradient-guided optimization strategy to balance attack success and computational efficiency. Extensive evaluations on Vicuna-7B, LLaMA-2-7B-Chat, Guanaco-7B, LLaMA-3-70B, and Mistral-7B-Instruct show that LFJ achieves an average Attack Success Rate (ASR) of 94.01%, significantly outperforming state-of-the-art baselines like GCG and AutoDAN while avoiding detectable input artifacts. Furthermore, we identify that thematic similarity in the latent space is a critical vulnerability in current safety alignments. Finally, we propose a latent adversarial training defense that reduces LFJ's ASR by over 80% without compromising model utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04700v1">PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Current techniques for post-training Large Language Models (LLMs) rely either on costly human supervision or on external verifiers to boost performance on tasks such as mathematical reasoning and code generation. However, as LLMs improve their problem-solving, any further improvement will potentially require high-quality solutions to difficult problems that are not available to humans. As a result, learning from unlabeled data is becoming increasingly attractive in the research community. Existing methods extract learning signal from a model's consistency, either by majority voting or by converting the model's internal confidence into reward. Although internal consistency metric such as entropy or self-certainty require no human intervention, as we show in this work, these are unreliable signals for large-scale and long-term training. To address the unreliability, we propose PRISM, a unified training framework that uses a Process Reward Model (PRM) to guide learning alongside model's internal confidence in the absence of ground-truth labels. We show that effectively combining PRM with self-certainty can lead to both stable training and better test-time performance, and also keep the model's internal confidence in check.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04694v1">ResMAS: Resilience Optimization in LLM-based Multi-agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-based MAS), where multiple LLM agents collaborate to solve complex tasks, have shown impressive performance in many areas. However, MAS are typically distributed across different devices or environments, making them vulnerable to perturbations such as agent failures. While existing works have studied the adversarial attacks and corresponding defense strategies, they mainly focus on reactively detecting and mitigating attacks after they occur rather than proactively designing inherently resilient systems. In this work, we study the resilience of LLM-based MAS under perturbations and find that both the communication topology and prompt design significantly influence system resilience. Motivated by these findings, we propose ResMAS: a two-stage framework for enhancing MAS resilience. First, we train a reward model to predict the MAS's resilience, based on which we train a topology generator to automatically design resilient topology for specific tasks through reinforcement learning. Second, we introduce a topology-aware prompt optimization method that refines each agent's prompt based on its connections and interactions with other agents. Extensive experiments across a range of tasks show that our approach substantially improves MAS resilience under various constraints. Moreover, our framework demonstrates strong generalization ability to new tasks and models, highlighting its potential for building resilient MASs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04690v1">Do LLMs Benefit from User and Item Embeddings in Recommendation Tasks?</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Presented in Multimodal Algorithmic Reasoning Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising recommendation systems, offering novel ways to model user preferences through generative approaches. However, many existing methods often rely solely on text semantics or incorporate collaborative signals in a limited manner, typically using only user or item embeddings. These methods struggle to handle multiple item embeddings representing user history, reverting to textual semantics and neglecting richer collaborative information. In this work, we propose a simple yet effective solution that projects user and item embeddings, learned from collaborative filtering, into the LLM token space via separate lightweight projector modules. A finetuned LLM then conditions on these projected embeddings alongside textual tokens to generate recommendations. Preliminary results show that this design effectively leverages structured user-item interaction data, improves recommendation performance over text-only LLM baselines, and offers a practical path for bridging traditional recommendation systems with modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04688v1">ToolGate: Contract-Grounded and Verified Tool Execution for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 First version of ToolGate
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) augmented with external tools have demonstrated remarkable capabilities in complex reasoning tasks. However, existing frameworks rely heavily on natural language reasoning to determine when tools can be invoked and whether their results should be committed, lacking formal guarantees for logical safety and verifiability. We present \textbf{ToolGate}, a forward execution framework that provides logical safety guarantees and verifiable state evolution for LLM tool calling. ToolGate maintains an explicit symbolic state space as a typed key-value mapping representing trusted world information throughout the reasoning process. Each tool is formalized as a Hoare-style contract consisting of a precondition and a postcondition, where the precondition gates tool invocation by checking whether the current state satisfies the required conditions, and the postcondition determines whether the tool's result can be committed to update the state through runtime verification. Our approach guarantees that the symbolic state evolves only through verified tool executions, preventing invalid or hallucinated results from corrupting the world representation. Experimental validation demonstrates that ToolGate significantly improves the reliability and verifiability of tool-augmented LLM systems while maintaining competitive performance on complex multi-step reasoning tasks. This work establishes a foundation for building more trustworthy and debuggable AI systems that integrate language models with external tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04680v1">Leveraging LLMs for Efficient and Personalized Smart Home Automation</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      The proliferation of smart home devices has increased the complexity of controlling and managing them, leading to user fatigue. In this context, large language models (LLMs) offer a promising solution by enabling natural-language interfaces for Internet of Things (IoT) control. However, existing LLM-based approaches suffer from unreliable and inefficient device control due to the non-deterministic nature of LLMs, high inference latency and cost, and limited personalization. To address these challenges, we present IoTGPT, an LLM-based smart home agent designed to execute IoT commands in a reliable, efficient, and personalized manner. Inspired by how humans manage complex tasks, IoTGPT decomposes user instructions into subtasks and memorizes them. By reusing learned subtasks, subsequent instructions can be processed more efficiently with fewer LLM calls, improving reliability and reducing both latency and cost. IoTGPT also supports fine-grained personalization by adapting individual subtasks to user preferences. Our evaluation demonstrates that IoTGPT outperforms baselines in accuracy, latency/cost, and personalization, while reducing user workload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04675v1">LLM-Guided Quantified SMT Solving over Uninterpreted Functions</a></div>
    <div class="paper-meta">
      📅 2026-01-08
    </div>
    <details class="paper-abstract">
      Quantified formulas with Uninterpreted Functions (UFs) over non-linear real arithmetic pose fundamental challenges for Satisfiability Modulo Theories (SMT) solving. Traditional quantifier instantiation methods struggle because they lack semantic understanding of UF constraints, forcing them to search through unbounded solution spaces with limited guidance. We present AquaForte, a framework that leverages Large Language Models to provide semantic guidance for UF instantiation by generating instantiated candidates for function definitions that satisfy the constraints, thereby significantly reducing the search space and complexity for solvers. Our approach preprocesses formulas through constraint separation, uses structured prompts to extract mathematical reasoning from LLMs, and integrates the results with traditional SMT algorithms through adaptive instantiation. AquaForte maintains soundness through systematic validation: LLM-guided instantiations yielding SAT solve the original problem, while UNSAT results generate exclusion clauses for iterative refinement. Completeness is preserved by fallback to traditional solvers augmented with learned constraints. Experimental evaluation on SMT-COMP benchmarks demonstrates that AquaForte solves numerous instances where state-of-the-art solvers like Z3 and CVC5 timeout, with particular effectiveness on satisfiable formulas. Our work shows that LLMs can provide valuable mathematical intuition for symbolic reasoning, establishing a new paradigm for SMT constraint solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04093v1">SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 We find that the key to jailbreak the LLM is objectifying its safety responsibility, thus we delegate the open-web to inject harmful semantics and get the huge gain from unmoderated web resources
    </div>
    <details class="paper-abstract">
      Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04086v1">KDCM: Reducing Hallucination in LLMs through Explicit Reasoning Structures</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      To mitigate hallucinations in large language models (LLMs), we propose a framework that focuses on errors induced by prompts. Our method extends a chain-style knowledge distillation approach by incorporating a programmable module that guides knowledge graph exploration. This module is embedded as executable code within the reasoning prompt, allowing the model to leverage external structured knowledge during inference. Based on this design, we develop an enhanced distillation-based reasoning framework that explicitly regulates intermediate reasoning steps, resulting in more reliable predictions. We evaluate the proposed approach on multiple public benchmarks using GPT-4 and LLaMA-3.3. Experimental results show that code-guided reasoning significantly improves contextual modeling and reduces prompt-induced hallucinations. Specifically, HIT@1, HIT@3, and HIT@5 increase by 15.64%, 13.38%, and 13.28%, respectively, with scores exceeding 95% across several evaluation settings. These findings indicate that the proposed method effectively constrains erroneous reasoning while improving both accuracy and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20721v2">User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. In contrast, five proxy LLMs reached high agreement, yet each proxy LLM had low correlation with users' evaluations. These results indicate that proxy LLMs cannot accurately estimate users' wide range of perceptions of utility and privacy in privacy-sensitive scenarios. We discuss the need for more user-centered studies to measure LLMs' ability to help users while preserving privacy, and for improving alignment between LLMs and users in estimating perceived privacy and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00224v2">Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 WITS 2025 (Workshop on Information Technologies and Systems 2025)
    </div>
    <details class="paper-abstract">
      As large language model (LLM) assistants become increasingly integrated into enterprise workflows, their ability to generate accurate, semantically aligned, and executable outputs is critical. However, current conversational business analytics (CBA) systems often lack built-in verification mechanisms, leaving users to manually validate potentially flawed results. This paper introduces two complementary verification techniques: Q*, which performs reverse translation and semantic matching between code and user intent, and Feedback+, which incorporates execution feedback to guide code refinement. Embedded within a generator-discriminator framework, these mechanisms shift validation responsibilities from users to the system. Evaluations on three benchmark datasets, Spider, Bird, and GSM8K, demonstrate that both Q* and Feedback+ reduce error rates and task completion time. The study also identifies reverse translation as a key bottleneck, highlighting opportunities for future improvement. Overall, this work contributes a design-oriented framework for building more reliable, enterprise-grade GenAI systems capable of trustworthy decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03986v1">Benchmark^2: Systematic Evaluation of LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      The rapid proliferation of benchmarks for evaluating large language models (LLMs) has created an urgent need for systematic methods to assess benchmark quality itself. We propose Benchmark^2, a comprehensive framework comprising three complementary metrics: (1) Cross-Benchmark Ranking Consistency, measuring whether a benchmark produces model rankings aligned with peer benchmarks; (2) Discriminability Score, quantifying a benchmark's ability to differentiate between models; and (3) Capability Alignment Deviation, identifying problematic instances where stronger models fail but weaker models succeed within the same model family. We conduct extensive experiments across 15 benchmarks spanning mathematics, reasoning, and knowledge domains, evaluating 11 LLMs across four model families. Our analysis reveals significant quality variations among existing benchmarks and demonstrates that selective benchmark construction based on our metrics can achieve comparable evaluation performance with substantially reduced test sets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11830v3">VISTA: Mitigating Semantic Inertia in Video-LLMs via Training-Free Dynamic Chain-of-Thought Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models have successfully transitioned towards System 2 reasoning, yet applying these paradigms to video understanding remains challenging. While prevailing research attributes failures in Video-LLMs to perceptual limitations, our empirical analysis reveals a cognitive misalignment termed Semantic Inertia, where models suppress valid visual evidence in favor of dominant language priors. To rectify this, we propose VISTA, a training-free framework designed to align perception with logical deduction. By dynamically routing inference paths and materializing implicit visual features into explicit textual anchors, our approach effectively counterbalances the influence of parametric knowledge. Furthermore, we incorporate a Latent Reasoning Consensus mechanism to mitigate stochastic hallucinations. VISTA showed outstanding results on a wide range of benchmarks, and outperforms its base model by 9.3% on Egochema and 5.6% on VideoEspresso, rivalling or even surpassing larger and proprietary models. Our codebase will be publicly available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03878v1">Understanding Specification-Driven Code Generation with LLMs: An Empirical Study Design</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 This paper is a Stage 1 Registered Report. The study protocol and analysis plan were peer reviewed and accepted at SANER 2026 with a Continuity Acceptance (CA) score for Stage 2
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software development workflows, yet their behavior in structured, specification-driven processes remains poorly understood. This paper presents an empirical study design using CURRANTE, a Visual Studio Code extension that enables a human-in-the-loop workflow for LLM-assisted code generation. The tool guides developers through three sequential stages--Specification, Tests, and Function--allowing them to define requirements, generate and refine test suites, and produce functions that satisfy those tests. Participants will solve medium-difficulty problems from the LiveCodeBench dataset, while the tool records fine-grained interaction logs, effectiveness metrics (e.g., pass rate, all-pass completion), efficiency indicators (e.g., time-to-pass), and iteration behaviors. The study aims to analyze how human intervention in specification and test refinement influences the quality and dynamics of LLM-generated code. The results will provide empirical insights into the design of next-generation development environments that align human reasoning with model-driven code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02626v2">Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Prior works have shown that fine-tuning on new knowledge can induce factual hallucinations in large language models (LLMs), leading to incorrect outputs when evaluated on previously known information. However, the specific manifestations of such hallucination and its underlying mechanisms remain insufficiently understood. Our work addresses this gap by designing a controlled dataset \textit{Biography-Reasoning}, and conducting a fine-grained analysis across multiple knowledge types and two task types, including knowledge question answering (QA) and knowledge reasoning tasks. We find that hallucinations not only severely affect tasks involving newly introduced knowledge, but also propagate to other evaluation tasks. Moreover, when fine-tuning on a dataset in which a specific knowledge type consists entirely of new knowledge, LLMs exhibit elevated hallucination tendencies. This suggests that the degree of unfamiliarity within a particular knowledge type, rather than the overall proportion of new knowledge, is a stronger driver of hallucinations. Through interpretability analysis, we show that learning new knowledge weakens the model's attention to key entities in the input question, leading to an over-reliance on surrounding context and a higher risk of hallucination. Conversely, reintroducing a small amount of known knowledge during the later stages of training restores attention to key entities and substantially mitigates hallucination behavior. Finally, we demonstrate that disrupted attention patterns can propagate across lexically similar contexts, facilitating the spread of hallucinations beyond the original task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10390v3">Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Existing black-box jailbreak attacks achieve certain success on non-reasoning models but degrade significantly on recent SOTA reasoning models. To improve attack ability, inspired by adversarial aggregation strategies, we integrate multiple jailbreak tricks into a single developer template. Especially, we apply Adversarial Context Alignment to purge semantic inconsistencies and use NTP (a type of harmful prompt) -based few-shot examples to guide malicious outputs, lastly forming DH-CoT attack with a fake chain of thought. In experiments, we further observe that existing red-teaming datasets include samples unsuitable for evaluating attack gains, such as BPs, NHPs, and NTPs. Such data hinders accurate evaluation of true attack effect lifts. To address this, we introduce MDH, a Malicious content Detection framework integrating LLM-based annotation with Human assistance, with which we clean data and build RTA dataset suite. Experiments show that MDH reliably filters low-quality samples and that DH-CoT effectively jailbreaks models including GPT-5 and Claude-4, notably outperforming SOTA methods like H-CoT and TAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03858v1">What Does Loss Optimization Actually Teach, If Anything? Knowledge Dynamics in Continual Pre-training of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Continual Pre-Training (CPT) is widely used for acquiring and updating factual knowledge in LLMs. This practice treats loss as a proxy for knowledge learning, while offering no grounding into how it changes during training. We study CPT as a knowledge learning process rather than a solely optimization problem. We construct a controlled, distribution-matched benchmark of factual documents and interleave diagnostic probes directly into the CPT loop, enabling epoch-level measurement of knowledge acquisition dynamics and changes in Out-Of-Domain (OOD) general skills (e.g., math). We further analyze how CPT reshapes knowledge circuits during training. Across three instruction-tuned LLMs and multiple CPT strategies, optimization and learning systematically diverge as loss decreases monotonically while factual learning is unstable and non-monotonic. Acquired facts are rarely consolidated, learning is strongly conditioned on prior exposure, and OOD performance degrades from early epochs. Circuit analysis reveals rapid reconfiguration of knowledge pathways across epochs, providing an explanation for narrow acquisition windows and systematic forgetting. These results show that loss optimization is misaligned with learning progress in CPT and motivate evaluation of stopping criteria based on task-level learning dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03857v1">Once Upon a Team: Investigating Bias in LLM-Driven Software Team Composition and Task Allocation</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to boost productivity and support software engineering tasks. However, when applied to socially sensitive decisions such as team composition and task allocation, they raise concerns of fairness. Prior studies have revealed that LLMs may reproduce stereotypes; however, these analyses remain exploratory and examine sensitive attributes in isolation. This study investigates whether LLMs exhibit bias in team composition and task assignment by analyzing the combined effects of candidates' country and pronouns. Using three LLMs and 3,000 simulated decisions, we find systematic disparities: demographic attributes significantly shaped both selection likelihood and task allocation, even when accounting for expertise-related factors. Task distributions further reflected stereotypes, with technical and leadership roles unevenly assigned across groups. Our findings indicate that LLMs exacerbate demographic inequities in software engineering contexts, underscoring the need for fairness-aware assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03846v1">When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      LLMs-based agents increasingly operate in multi-agent environments where strategic interaction and coordination are required. While existing work has largely focused on individual agents or on interacting agents sharing explicit communication, less is known about how interacting agents coordinate implicitly. In particular, agents may engage in covert communication, relying on indirect or non-linguistic signals embedded in their actions rather than on explicit messages. This paper presents a game-theoretic study of covert communication in LLM-driven multi-agent systems. We analyse interactions across four canonical game-theoretic settings under different communication regimes, including explicit, restricted, and absent communication. Considering heterogeneous agent personalities and both one-shot and repeated games, we characterise when covert signals emerge and how they shape coordination and strategic outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14540v3">VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 In Proceedings ICLP 2025, arXiv:2601.00047
    </div>
    <details class="paper-abstract">
      A recent approach to neurosymbolic reasoning is to explicitly combine the strengths of large language models (LLMs) and symbolic solvers to tackle complex reasoning tasks. However, current approaches face significant limitations, including poor generalizability due to task-specific prompts, inefficiencies caused by the lack of separation between knowledge and queries, and restricted inferential capabilities. These shortcomings hinder their scalability and applicability across diverse domains. In this paper, we introduce VERUS-LM, a novel framework designed to address these challenges. VERUS-LM employs a generic prompting mechanism, clearly separates domain knowledge from queries, and supports a wide range of different logical reasoning tasks. This framework enhances adaptability, reduces computational cost, and allows for richer forms of reasoning, such as optimization and constraint satisfaction. We show that our approach succeeds in diverse reasoning on a novel dataset, markedly outperforming LLMs. Additionally, our system achieves competitive results on common reasoning benchmarks when compared to similar state-of-the-art approaches, and significantly surpasses them on the difficult AR-LSAT dataset. By pushing the boundaries of hybrid reasoning, VERUS-LM represents a significant step towards more versatile neurosymbolic AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01768v2">Can LLMs Track Their Output Length? A Dynamic Feedback Mechanism for Precise Length Regulation</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Precisely controlling the length of generated text is a common requirement in real-world applications. However, despite significant advancements in following human instructions, Large Language Models (LLMs) still struggle with this task. In this work, we demonstrate that LLMs often fail to accurately measure their response lengths, leading to poor adherence to length constraints. To address this issue, we propose a novel length regulation approach that incorporates dynamic length feedback during generation, enabling adaptive adjustments to meet target lengths. Experiments on summarization and biography tasks show our training-free approach significantly improves precision in achieving target token, word, or sentence counts without compromising quality. Additionally, we demonstrate that further supervised fine-tuning allows our method to generalize effectively to broader text-generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03808v1">From Brute Force to Semantic Insight: Performance-Guided Data Transformation Design with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved notable performance in code synthesis; however, data-aware augmentation remains a limiting factor, handled via heuristic design or brute-force approaches. We introduce a performance-aware, closed-loop solution in the NNGPT ecosystem of projects that enables LLMs to autonomously engineer optimal transformations by internalizing empirical performance cues. We fine-tune LLMs with Low-Rank Adaptation on a novel repository of more than 6,000 empirically evaluated PyTorch augmentation functions, each annotated solely by downstream model accuracy. Training uses pairwise performance ordering (better-worse transformations), enabling alignment through empirical feedback without reinforcement learning, reward models, or symbolic objectives. This reduces the need for exhaustive search, achieving up to 600x times fewer evaluated candidates than brute-force discovery while maintaining competitive peak accuracy and shifting generation from random synthesis to task-aligned design. Ablation studies show that structured Chain-of-Thought prompting introduces syntactic noise and degrades performance, whereas direct prompting ensures stable optimization in performance-critical code tasks. Qualitative and quantitative analyses demonstrate that the model internalizes semantic performance cues rather than memorizing syntax. These results show that LLMs can exhibit task-level reasoning through non-textual feedback loops, bypassing explicit symbolic rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03791v1">Do LLMs Really Memorize Personally Identifiable Information? Revisiting PII Leakage with a Cue-Controlled Memorization Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been reported to "leak" Personally Identifiable Information (PII), with successful PII reconstruction often interpreted as evidence of memorization. We propose a principled revision of memorization evaluation for LLMs, arguing that PII leakage should be evaluated under low lexical cue conditions, where target PII cannot be reconstructed through prompt-induced generalization or pattern completion. We formalize Cue-Resistant Memorization (CRM) as a cue-controlled evaluation framework and a necessary condition for valid memorization evaluation, explicitly conditioning on prompt-target overlap cues. Using CRM, we conduct a large-scale multilingual re-evaluation of PII leakage across 32 languages and multiple memorization paradigms. Revisiting reconstruction-based settings, including verbatim prefix-suffix completion and associative reconstruction, we find that their apparent effectiveness is driven primarily by direct surface-form cues rather than by true memorization. When such cues are controlled for, reconstruction success diminishes substantially. We further examine cue-free generation and membership inference, both of which exhibit extremely low true positive rates. Overall, our results suggest that previously reported PII leakage is better explained by cue-driven behavior than by genuine memorization, highlighting the importance of cue-controlled evaluation for reliably quantifying privacy-relevant memorization in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03785v1">Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22156v3">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 18 pages,5 figures
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03779v1">Tracing the complexity profiles of different linguistic phenomena through the intrinsic dimension of LLM representations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      We explore the intrinsic dimension (ID) of LLM representations as a marker of linguistic complexity, asking if different ID profiles across LLM layers differentially characterize formal and functional complexity. We find the formal contrast between sentences with multiple coordinated or subordinated clauses to be reflected in ID differences whose onset aligns with a phase of more abstract linguistic processing independently identified in earlier work. The functional contrasts between sentences characterized by right branching vs. center embedding or unambiguous vs. ambiguous relative clause attachment are also picked up by ID, but in a less marked way, and they do not correlate with the same processing phase. Further experiments using representational similarity and layer ablation confirm the same trends. We conclude that ID is a useful marker of linguistic complexity in LLMs, that it allows to differentiate between different types of complexity, and that it points to similar stages of linguistic processing across disparate LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03775v1">Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce verbalized self-explanations, yet prior studies suggest that such rationales may not reliably reflect the model's true decision process. We ask whether these explanations nevertheless help users predict model behavior, operationalized as counterfactual simulatability. Using StrategyQA, we evaluate how well humans and LLM judges can predict a model's answers to counterfactual follow-up questions, with and without access to the model's chain-of-thought or post-hoc explanations. We compare LLM-generated counterfactuals with pragmatics-based perturbations as alternative ways to construct test cases for assessing the potential usefulness of explanations. Our results show that self-explanations consistently improve simulation accuracy for both LLM judges and humans, but the degree and stability of gains depend strongly on the perturbation strategy and judge strength. We also conduct a qualitative analysis of free-text justifications written by human users when predicting the model's behavior, which provides evidence that access to explanations helps humans form more accurate predictions on the perturbed questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03746v1">Whose Facts Win? LLM Source Preferences under Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Data and code: https://github.com/JaSchuste/llm-source-preference
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18795v3">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08473v3">AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance but introduces critical safety vulnerabilities: even minimal harmful data can severely compromise safety measures. We observe that perturbations orthogonal to the alignment direction - defined by weight differences between aligned (safe) and unaligned models - rapidly compromise model safety. In contrast, updates along the alignment direction largely preserve it, revealing the parameter space as a "narrow safety basin". To address this, we propose AsFT (Anchoring Safety in Fine-Tuning) to maintain safety by explicitly constraining update directions during fine-tuning. By penalizing updates orthogonal to the alignment direction, AsFT effectively constrains the model within the "narrow safety basin," thus preserving its inherent safety. Extensive experiments on multiple datasets and models show that AsFT reduces harmful behaviors by up to 7.60%, improves task performance by 3.44%, and consistently outperforms existing methods across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23163v3">Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14803v4">OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      In online learning environments, students often lack personalized peer interactions, which are crucial for cognitive development and learning engagement. Although previous studies have employed large language models (LLMs) to simulate interactive learning environments, these interactions are limited to conversational exchanges, failing to adapt to learners' individualized cognitive and psychological states. As a result, students' engagement is low and they struggle to gain inspiration. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs integrated with Theory of Mind (ToM). OnlineMate simulates peer-like roles, infers learners' psychological states such as misunderstandings and confusion during collaborative discussions, and dynamically adjusts interaction strategies to support higher-order thinking. Comprehensive evaluations, including simulation-based experiments, human assessments, and real classroom trials, demonstrate that OnlineMate significantly promotes deep learning and cognitive engagement by elevating students' average cognitive level while substantially improving emotional engagement scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01211v2">Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      With the proliferation of LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern. Once MAS is induced to trust a malicious link, attackers can use it as a springboard to expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack manipulating unique structures of web links to deceive MAS. We design 12 representative attack variants that encompass various methods, such as homoglyph deception, sub-directory nesting, and parameter obfuscation. Through extensive experiments on these attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input design, lowering the threshold for attacks significantly. These results underscore the importance of addressing Web fraud attacks, providing new insights into MAS safety. Our code is available at https://github.com/JiangYingEr/Web-Fraud-Attack-in-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03687v1">Personalized Medication Planning via Direct Domain Modeling and LLM-Generated Heuristics</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Personalized medication planning involves selecting medications and determining a dosing schedule to achieve medical goals specific to each individual patient. Previous work successfully demonstrated that automated planners, using general domain-independent heuristics, are able to generate personalized treatments, when the domain and problems are modeled using a general domain description language (\pddlp). Unfortunately, this process was limited in practice to consider no more than seven medications. In clinical terms, this is a non-starter. In this paper, we explore the use of automatically-generated domain- and problem-specific heuristics to be used with general search, as a method of scaling up medication planning to levels allowing closer work with clinicians. Specifically, we specify the domain programmatically (specifying an initial state and a successor generation procedure), and use an LLM to generate a problem specific heuristic that can be used by a fixed search algorithm (GBFS). The results indicate dramatic improvements in coverage and planning time, scaling up the number of medications to at least 28, and bringing medication planning one step closer to practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03682v1">From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Recent studies reveal that large language models (LLMs) exhibit limited logical reasoning abilities in mathematical problem-solving, instead often relying on pattern-matching and memorization. We systematically analyze this limitation, focusing on logical relationship understanding, which is a core capability underlying genuine logical reasoning, and reveal that errors related to this capability account for over 90\% of incorrect predictions, with Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) failing to substantially reduce these errors. To address this bottleneck, we propose First-Step Logical Reasoning (FSLR), a lightweight training framework targeting logical relationship understanding. Our key insight is that the first planning step-identifying which variables to use and which operation to apply-encourages the model to derive logical relationships directly from the problem statement. By training models on this isolated step, FSLR provides explicit supervision for logical relationship understanding, unlike CoT-SFT which implicitly embeds such relationships within complete solution trajectories. Extensive experiments across multiple models and datasets demonstrate that FSLR consistently outperforms CoT-SFT under both in-distribution and out-of-distribution settings, with average improvements of 3.2\% and 4.6\%, respectively. Moreover, FSLR achieves 4-6x faster training and reduces training token consumption by over 80\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18851v3">Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Identifying LLM-generated code through watermarking poses a challenge in preserving functional correctness. Previous methods rely on the assumption that watermarking high-entropy tokens effectively maintains output quality. Our analysis reveals a fundamental limitation of this assumption: syntax-critical tokens such as keywords often exhibit the highest entropy, making existing approaches vulnerable to logic corruption. We present STONE, a syntax-aware watermarking method that embeds watermarks only in non-syntactic tokens and preserves code integrity. For its rigorous assessment, we also introduce STEM, a comprehensive framework that balances three critical dimensions: correctness, detectability, and imperceptibility. Across Python, C++, and Java, STONE preserves correctness, sustains strong detectability, and achieves balanced performance with minimal overhead. Our implementation is available at https://anonymous.4open.science/r/STONE-watermarking-AB4B/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03676v1">Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 The code and data for our methods and experiments are available at https://github.com/weiyifan1023/STEPS
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and agent-based systems often struggle with compositional generalization due to a data bottleneck in which complex skill combinations follow a long-tailed, power-law distribution, limiting both instruction-following performance and generalization in agent-centric tasks. To address this challenge, we propose STEPS, a Skill Taxonomy guided Entropy-based Post-training data Synthesis framework for generating compositionally challenging data. STEPS explicitly targets compositional generalization by uncovering latent relationships among skills and organizing them into an interpretable, hierarchical skill taxonomy using structural information theory. Building on this taxonomy, we formulate data synthesis as a constrained information maximization problem, selecting skill combinations that maximize marginal structural information within the hierarchy while preserving semantic coherence. Experiments on challenging instruction-following benchmarks show that STEPS outperforms existing data synthesis baselines, while also yielding improved compositional generalization in downstream agent-based evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.02930v5">Video LLMs for Temporal Reasoning in Long Videos</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      We introduce TemporalVLM, a video large language model (video LLM) for temporal reasoning and fine-grained understanding in long videos. Our approach includes a visual encoder for mapping a long-term video into features which are time-aware and contain both local and global cues. It first divides an input video into short-term clips, which are jointly encoded with timestamps and fused across overlapping temporal windows into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, consisting of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments show that TemporalVLM outperforms previous methods across temporal reasoning and fine-grained understanding tasks, i.e., dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To our best knowledge, our work is the first to incorporate LSTMs into video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08726v3">Improved LLM Agents for Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 13 pages, 6 figures. More analysis is added to Appendix C
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02110v2">Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface, where adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. The proposed attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even against prompt-level defenses, auditor-based detection, and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface. Notably, AMA is orthogonal to injection attacks and can be combined with them to achieve stronger attack efficacy, highlighting the need for execution-level defenses beyond prompt-level and auditor-based mechanisms. Code is available at https://github.com/SEAIC-M/AMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05738v3">LLM-assisted Mutation for Whitebox API Testing</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Cloud applications heavily rely on APIs to communicate with each other and exchange data. To ensure the reliability of cloud applications, cloud providers widely adopt API testing techniques. Unfortunately, existing API testing approaches are insufficient to reach strict conditions, a problem known as fitness plateaus, due to the lack of gradient provided by coverage metrics. To address this issue, we propose MioHint, a novel white-box API testing approach that leverages the code comprehension capabilities of Large Language Model (LLM) to boost API testing. The key challenge of LLM-based API testing lies in system-level testing, which emphasizes the dependencies between requests and targets across functions and files, thereby making the entire codebase the object of analysis. However, feeding the entire codebase to an LLM is impractical due to its limited context length and short memory. MioHint addresses this challenge by synergizing static analysis with LLMs. We retrieve relevant code with data-dependency analysis at the statement level, including def-use analysis for variables used in the target and function expansion for subfunctions called by the target. To evaluate the effectiveness of our method, we conducted experiments across 16 real-world REST API services. The findings reveal that MioHint achieves an average increase of 4.95% absolute in line coverage compared to the baseline, EvoMaster, alongside a remarkable factor of 67x improvement in mutation accuracy. Furthermore, our method successfully covers over 57% of hard-to-cover targets while in baseline the coverage is less than 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.23314v2">SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled dynamic reasoning in automated data analytics, yet recent multi-agent systems remain limited by rigid, single-path workflows that restrict strategic exploration and often lead to suboptimal outcomes. To overcome these limitations, we propose SPIO (Sequential Plan Integration and Optimization), a framework that replaces rigid workflows with adaptive, multi-path planning across four core modules: data preprocessing, feature engineering, model selection, and hyperparameter tuning. In each module, specialized agents generate diverse candidate strategies, which are cascaded and refined by an optimization agent. SPIO offers two operating modes: SPIO-S for selecting a single optimal pipeline, and SPIO-E for ensembling top-k pipelines to maximize robustness. Extensive evaluations on Kaggle and OpenML benchmarks show that SPIO consistently outperforms state-of-the-art baselines, achieving an average performance gain of 5.6%. By explicitly exploring and integrating multiple solution paths, SPIO delivers a more flexible, accurate, and reliable foundation for automated data science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03645v1">LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03640v1">Verbatim Data Transcription Failures in LLM Code Generation: A State-Tracking Stress Test</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Many real-world software tasks require exact transcription of provided data into code, such as cryptographic constants, protocol test vectors, allowlists, and calibration tables. These tasks are operationally sensitive because small omissions or alterations can remain silent while producing syntactically valid programs. This paper introduces a deliberately minimal transcription-to-code benchmark to isolate this reliability concern in LLM-based code generation. Given a list of high-precision decimal constants, a model must generate Python code that embeds the constants verbatim and performs a simple aggregate computation. We describe the prompting variants, evaluation protocol based on exact-string inclusion, and analysis framework used to characterize state-tracking and long-horizon generation failures. The benchmark is intended as a compact stress test that complements existing code-generation evaluations by focusing on data integrity rather than algorithmic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02813v2">HAL: Inducing Human-likeness in LLMs with Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Conversational human-likeness plays a central role in human-AI interaction, yet it has remained difficult to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale human evaluations, models aligned with HAL are more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03630v1">Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents the first systematic comparison investigating whether Large Reasoning Models (LRMs) are superior judge to non-reasoning LLMs. Our empirical analysis yields four key findings: 1) LRMs outperform non-reasoning LLMs in terms of judgment accuracy, particularly on reasoning-intensive tasks; 2) LRMs demonstrate superior instruction-following capabilities in evaluation contexts; 3) LRMs exhibit enhanced robustness against adversarial attacks targeting judgment tasks; 4) However, LRMs still exhibit strong biases in superficial quality. To improve the robustness against biases, we propose PlanJudge, an evaluation strategy that prompts the model to generate an explicit evaluation plan before execution. Despite its simplicity, our experiments demonstrate that PlanJudge significantly mitigates biases in both LRMs and standard LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03627v1">Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 EACL 2026 Industry
    </div>
    <details class="paper-abstract">
      We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on https://github.com/seemdog/EPAG, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings.
    </details>
</div>
