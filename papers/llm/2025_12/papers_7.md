# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08242v1">Chopper: A Multi-Level GPU Characterization Tool & Derived Insights Into LLM Training Inefficiency</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) efficiently requires a deep understanding of how modern GPU systems behave under real-world distributed training workloads. While prior work has focused primarily on kernel-level performance or single-GPU microbenchmarks, the complex interaction between communication, computation, memory behavior, and power management in multi-GPU LLM training remains poorly characterized. In this work, we introduce Chopper, a profiling and analysis framework that collects, aligns, and visualizes GPU kernel traces and hardware performance counters across multiple granularities (i.e., from individual kernels to operations, layers, phases, iterations, and GPUs). Using Chopper, we perform a comprehensive end-to-end characterization of Llama 3 8B training under fully sharded data parallelism (FSDP) on an eight-GPU AMD InstinctTM MI300X node. Our analysis reveals several previously underexplored bottlenecks and behaviors, such as memory determinism enabling higher, more stable GPU and memory frequencies. We identify several sources of inefficiencies, with frequency overhead (DVFS effects) being the single largest contributor to the gap between theoretical and observed performance, exceeding the impact of MFMA utilization loss, communication/computation overlap, and kernel launch overheads. Overall, Chopper provides the first holistic, multi-granularity characterization of LLM training on AMD InstinctTM MI300X GPUs, yielding actionable insights for optimizing training frameworks, improving power-management strategies, and guiding future GPU architecture and system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08213v1">Secure or Suspect? Investigating Package Hallucinations of Shell Command in Original and Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models for code (LLMs4Code) are increasingly used to generate software artifacts, including library and package recommendations in languages such as Go. However, recent evidence shows that LLMs frequently hallucinate package names or generate dependencies containing known security vulnerabilities, posing significant risks to developers and downstream software supply chains. At the same time, quantization has become a widely adopted technique to reduce inference cost and enable deployment of LLMs on resource-constrained environments. Despite its popularity, little is known about how quantization affects the correctness and security of LLM-generated software dependencies while generating shell commands for package installation. In this work, we conduct the first systematic empirical study of the impact of quantization on package hallucination and vulnerability risks in LLM-generated Go packages. We evaluate five Qwen model sizes under full-precision, 8-bit, and 4-bit quantization across three datasets (SO, MBPP, and paraphrase). Our results show that quantization substantially increases the package hallucination rate (PHR), with 4-bit models exhibiting the most severe degradation. We further find that even among the correctly generated packages, the vulnerability presence rate (VPR) rises as precision decreases, indicating elevated security risk in lower-precision models. Finally, our analysis of hallucinated outputs reveals that most fabricated packages resemble realistic URL-based Go module paths, such as most commonly malformed or non-existent GitHub and golang.org repositories, highlighting a systematic pattern in how LLMs hallucinate dependencies. Overall, our findings provide actionable insights into the reliability and security implications of deploying quantized LLMs for code generation and dependency recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08211v1">MobileFineTuner: A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 15 pages, 9 figures, submitted to Mobisys 2026
    </div>
    <details class="paper-abstract">
      Mobile phones are the most ubiquitous end devices, generating vast amounts of human-authored data and serving as the primary platform for end-side applications. As high-quality public data for large language models (LLMs) approaches exhaustion, on-device fine-tuning provides an opportunity to leverage private user data while preserving privacy. However, existing approaches are predominantly simulation-based or rely on IoT devices and PCs, leaving commodity mobile phones largely unexplored. A key gap is the absence of an open-source framework that enables practical LLM fine-tuning on mobile phones. We present MobileFineTuner, a unified open-source framework that enables end-to-end LLM fine-tuning directly on commodity mobile phones. MobileFineTuner is designed for efficiency, scalability, and usability, supporting full-parameters fine-tuning (Full-FT) and parameter-efficient fine-tuning (PEFT). To address the memory and energy limitations inherent to mobile phones, we introduce system-level optimizations including parameter sharding, gradient accumulation, and energy-aware computation scheduling. We demonstrate the practicality of MobileFineTuner by fine-tuning GPT-2, Gemma 3, and Qwen 2.5 on real mobile phones. Extensive experiments and ablation studies validate the effectiveness of the proposed optimizations and establish MobileFineTuner as a viable foundation for future research on on-device LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09199v1">LLMs for Analog Circuit Design Continuum (ACDC)</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and transformer architectures have shown impressive reasoning and generation capabilities across diverse natural language tasks. However, their reliability and robustness in real-world engineering domains remain largely unexplored, limiting their practical utility in human-centric workflows. In this work, we investigate the applicability and consistency of LLMs for analog circuit design -- a task requiring domain-specific reasoning, adherence to physical constraints, and structured representations -- focusing on AI-assisted design where humans remain in the loop. We study how different data representations influence model behavior and compare smaller models (e.g., T5, GPT-2) with larger foundation models (e.g., Mistral-7B, GPT-oss-20B) under varying training conditions. Our results highlight key reliability challenges, including sensitivity to data format, instability in generated designs, and limited generalization to unseen circuit configurations. These findings provide early evidence on the limits and potential of LLMs as tools to enhance human capabilities in complex engineering tasks, offering insights into designing reliable, deployable foundation models for structured, real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09187v1">WOLF: Werewolf-based Observations for LLM Deception and Falsehoods</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Spotlight Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Deception is a fundamental challenge for multi-agent reasoning: effective systems must strategically conceal information while detecting misleading behavior in others. Yet most evaluations reduce deception to static classification, ignoring the interactive, adversarial, and longitudinal nature of real deceptive dynamics. Large language models (LLMs) can deceive convincingly but remain weak at detecting deception in peers. We present WOLF, a multi-agent social deduction benchmark based on Werewolf that enables separable measurement of deception production and detection. WOLF embeds role-grounded agents (Villager, Werewolf, Seer, Doctor) in a programmable LangGraph state machine with strict night-day cycles, debate turns, and majority voting. Every statement is a distinct analysis unit, with self-assessed honesty from speakers and peer-rated deceptiveness from others. Deception is categorized via a standardized taxonomy (omission, distortion, fabrication, misdirection), while suspicion scores are longitudinally smoothed to capture both immediate judgments and evolving trust dynamics. Structured logs preserve prompts, outputs, and state transitions for full reproducibility. Across 7,320 statements and 100 runs, Werewolves produce deceptive statements in 31% of turns, while peer detection achieves 71-73% precision with ~52% overall accuracy. Precision is higher for identifying Werewolves, though false positives occur against Villagers. Suspicion toward Werewolves rises from ~52% to over 60% across rounds, while suspicion toward Villagers and the Doctor stabilizes near 44-46%. This divergence shows that extended interaction improves recall against liars without compounding errors against truthful roles. WOLF moves deception evaluation beyond static datasets, offering a dynamic, controlled testbed for measuring deceptive and detective capacity in adversarial multi-agent interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13557v6">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 new version
    </div>
    <details class="paper-abstract">
      Coarse-Grained Reconfigurable Arrays (CGRAs) offer high performance and energy efficiency across domains, yet design remains difficult due to a vast, interdependent space and costly manual iteration. We present MACO, an open-source multi-agent LLM framework for CGRA HW/SW co-design. MACO generates and refines architectures through four stages: HW/SW Co-design, Design Error Correction, Best-Design Selection, and Evaluation & Feedback, iteratively improves power, performance, and area (PPA) via agent reasoning and closed-loop feedback. To traverse the space efficiently, we introduce exponentially decaying exploration; to accelerate convergence, we incorporate an LLM self-learning mechanism that adaptively selects promising candidate CGRAs. In addition, we propose a rule-based mechanism to correct CGRA design errors. Evaluated against other state-of-the-art methods, MACO achieves superior PPA while substantially reducing human effort, highlighting the promise of LLM-driven automation for practical CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09117v1">A Categorical Analysis of Large Language Models and Why LLMs Circumvent the Symbol Grounding Problem</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      This paper presents a formal, categorical framework for analysing how humans and large language models (LLMs) transform content into truth-evaluated propositions about a state space of possible worlds W , in order to argue that LLMs do not solve but circumvent the symbol grounding problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09108v1">Evolving Excellence: Automated Optimization of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies. We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications. We evaluate ARTEMIS on four representative agent systems: the \emph{ALE Agent} for competitive programming on AtCoder Heuristic Contest, achieving a \textbf{$13.6\%$ improvement} in acceptance rate; the \emph{Mini-SWE Agent} for code optimization on SWE-Perf, with a statistically significant \textbf{10.1\% performance gain}; and the \emph{CrewAI Agent} for cost and mathematical reasoning on Math Odyssey, achieving a statistically significant \textbf{$36.9\%$ reduction} in the number of tokens required for evaluation. We also evaluate the \emph{MathTales-Teacher Agent} powered by a smaller open-source model (Qwen2.5-7B) on GSM8K primary-level mathematics problems, achieving a \textbf{22\% accuracy improvement} and demonstrating that ARTEMIS can optimize agents based on both commercial and local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09088v1">Calibrated Trust in Dealing with LLM Hallucinations: A Qualitative Study</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Hallucinations are outputs by Large Language Models (LLMs) that are factually incorrect yet appear plausible [1]. This paper investigates how such hallucinations influence users' trust in LLMs and users' interaction with LLMs. To explore this in everyday use, we conducted a qualitative study with 192 participants. Our findings show that hallucinations do not result in blanket mistrust but instead lead to context-sensitive trust calibration. Building on the calibrated trust model by Lee & See [2] and Afroogh et al.'s trust-related factors [3], we confirm expectancy [3], [4], prior experience [3], [4], [5], and user expertise & domain knowledge [3], [4] as userrelated (human) trust factors, and identify intuition as an additional factor relevant for hallucination detection. Additionally, we found that trust dynamics are further influenced by contextual factors, particularly perceived risk [3] and decision stakes [6]. Consequently, we validate the recursive trust calibration process proposed by Blöbaum [7] and extend it by including intuition as a user-related trust factor. Based on these insights, we propose practical recommendations for responsible and reflective LLM use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08998v1">DermETAS-SNA LLM: A Dermatology Focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM Assistant</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Our work introduces the DermETAS-SNA LLM Assistant that integrates Dermatology-focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM. The assistant dynamically learns skin-disease classifiers and provides medically informed descriptions to facilitate clinician-patient interpretation. Contributions include: (1) Developed an ETAS framework on the SKINCON dataset to optimize a Vision Transformer (ViT) tailored for dermatological feature representation and then fine-tuned binary classifiers for each of the 23 skin disease categories in the DermNet dataset to enhance classification performance; (2) Designed a StackNet architecture that integrates multiple fine-tuned binary ViT classifiers to enhance predictive robustness and mitigate class imbalance issues; (3) Implemented a RAG pipeline, termed Diagnostic Explanation and Retrieval Model for Dermatology, which harnesses the capabilities of the Google Gemini 2.5 Pro LLM architecture to generate personalized, contextually informed diagnostic descriptions and explanations for patients, leveraging a repository of verified dermatological materials; (4) Performed extensive experimental evaluations on 23 skin disease categories to demonstrate performance increase, achieving an overall F1-score of 56.30% that surpasses SkinGPT-4 (48.51%) by a considerable margin, representing a performance increase of 16.06%; (5) Conducted a domain-expert evaluation, with eight licensed medical doctors, of the clinical responses generated by our AI assistant for seven dermatological conditions. Our results show a 92% agreement rate with the assessments provided by our AI assistant (6) Created a proof-of-concept prototype that fully integrates our DermETAS-SNA LLM into our AI assistant to demonstrate its practical feasibility for real-world clinical and educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v7">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08875v1">When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08870v1">Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      LLM agents are widely deployed in complex interactive tasks, yet privacy constraints often preclude centralized optimization and co-evolution across dynamic environments. While Federated Learning (FL) has proven effective on static datasets, its extension to the open-ended self-evolution of agents remains underexplored. Directly applying standard FL is challenging: heterogeneous tasks and sparse, trajectory-level rewards introduce severe gradient conflicts, destabilizing the global optimization process. To bridge this gap, we propose Fed-SE, a Federated Self-Evolution framework for LLM agents. Fed-SE establishes a local evolution-global aggregation paradigm. Locally, agents employ parameter-efficient fine-tuning on filtered, high-return trajectories to achieve stable gradient updates. Globally, Fed-SE aggregates updates within a low-rank subspace that disentangles environment-specific dynamics, effectively reducing negative transfer across clients. Experiments across five heterogeneous environments demonstrate that Fed-SE improves average task success rates by approximately 18% over federated baselines, validating its effectiveness in robust cross-environment knowledge transfer in privacy-constrained deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08814v1">Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Understanding human personality is crucial for web applications such as personalized recommendation and mental health assessment. Existing studies on personality detection predominantly adopt a "posts -> user vector -> labels" modeling paradigm, which encodes social media posts into user representations for predicting personality labels (e.g., MBTI labels). While recent advances in large language models (LLMs) have improved text encoding capacities, these approaches remain constrained by limited supervision signals due to label scarcity, and under-specified semantic mappings between user language and abstract psychological constructs. We address these challenges by proposing ROME, a novel framework that explicitly injects psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME leverages LLMs' role-play capability to simulate user responses to validated psychometric questionnaires. These generated question-level answers transform free-form user posts into interpretable, questionnaire-grounded evidence linking linguistic cues to personality labels, thereby providing rich intermediate supervision to mitigate label scarcity while offering a semantic reasoning chain that guides and simplifies the text-to-personality mapping learning. A question-conditioned Mixture-of-Experts module then jointly routes over post and question representations, learning to answer questionnaire items under explicit supervision. The predicted answers are summarized into an interpretable answer vector and fused with the user representation for final prediction within a multi-task learning framework, where question answering serves as a powerful auxiliary task for personality detection. Extensive experiments on two real-world datasets demonstrate that ROME consistently outperforms state-of-the-art baselines, achieving improvements (15.41% on Kaggle dataset).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v3">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08810v1">Multicalibration for LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted at AI-SQE 2026 (The 1st International Workshop on AI for Software Quality Evaluation: Judgment, Metrics, Benchmarks, and Beyond)
    </div>
    <details class="paper-abstract">
      As AI-based code generation becomes widespread, researchers are investigating the calibration of code LLMs - ensuring their confidence scores faithfully represent the true likelihood of code correctness. To do so, we investigate multicalibration, which can capture additional factors about a coding problem, such as complexity, code length, or programming language used. We study four multicalibration approaches on three function synthesis benchmarks, using latest-generation code LLMs (Qwen3 Coder, GPT-OSS, DeepSeek-R1-Distill). Our results demonstrate that multicalibration can yield distinct improvements over both uncalibrated token likelihoods (+1.03 in skill score) and baseline calibrations (+0.37 in skill score). We study the influence of the aforementioned factors in ablations, and make our dataset (consisting of code generations, likelihoods, and correctness labels) available for future research on code LLM calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08786v1">A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08764v1">Financial News Summarization: Can extractive methods still offer a true alternative to LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 8 pages, 5 tables
    </div>
    <details class="paper-abstract">
      Financial markets change rapidly due to news, economic shifts, and geopolitical events. Quick reactions are vital for investors to avoid losses or capture short-term gains. As a result, concise financial news summaries are critical for decision-making. With over 50,000 financial articles published daily, automation in summarization is necessary. This study evaluates a range of summarization methods, from simple extractive techniques to advanced large language models (LLMs), using the FinLLMs Challenge dataset. LLMs generated more coherent and informative summaries, but they are resource-intensive and prone to hallucinations, which can introduce significant errors into financial summaries. In contrast, extractive methods perform well on short, well-structured texts and offer a more efficient alternative for this type of article. The best ROUGE results come from fine-tuned LLM model like FT-Mistral-7B, although our data corpus has limited reliability, which calls for cautious interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08706v1">RESTifAI: LLM-Based Workflow for Reusable REST API Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted for ICSE 2026 Demonstration track
    </div>
    <details class="paper-abstract">
      With this paper, we introduce RESTifAI, an LLM-driven approach for generating reusable, CI/CD ready REST API tests, following the happy-path approach. Unlike existing tools that often focus primarily on internal server errors, RESTifAI systematically constructs valid test scenarios (happy paths) and derives negative cases to verify both intended functionality (2xx responses) and robustness against invalid inputs or business-rule violations (4xx responses). The results indicate that RESTifAI performs on par with the latest LLM tools, i.e., AutoRestTest and LogiAgent, while addressing limitations related to reusability, oracle complexity, and integration. To support this, we provide common comparative results and demonstrate the tool's applicability in industrial services. For tool demonstration, please refer to https://www.youtube.com/watch?v=2vtQo0T0Lo4. RESTifAI is publicly available at https://github.com/casablancahotelsoftware/RESTifAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02943v6">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is labor-intensive, especially for strongly typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in LLMs have enabled oracle generation from natural language descriptions, aligning better with user requirements. However, existing LLM-based methods often require fine-tuning or rely on external tools such as EvoSuite for test prefix generation, making them costly or cumbersome to apply in practice. In this work, we propose CANDOR, a novel prompt engineering-based LLM framework for automated unit test generation in Java. CANDOR orchestrates multiple specialized LLM agents to collaboratively generate complete tests. To mitigate the notorious hallucinations in LLMs and improve oracle correctness, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generates accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments show that CANDOR is comparable with EvoSuite in generating tests with high code coverage and clearly superior in terms of mutation score. Moreover, our prompt engineering-based approach CANDOR significantly outperforms the SOTA fine-tuning-based oracle generator TOGLL by at least 21.1 percentage points in oracle correctness on both correct and faulty source code. Further ablation studies confirm the critical contributions of key agents in generating high-quality tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11599v2">SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v23">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.15378v6">Improving LLM Reliability with RAG in Religious Question-Answering: MufassirQAS</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Religious teachings can sometimes be complex and challenging to grasp, but chatbots can serve as effective assistants in this domain. Large Language Model (LLM) based chatbots, powered by Natural Language Processing (NLP), can connect related topics and provide well-supported responses to intricate questions, making them valuable tools for religious education. However, LLMs are prone to hallucinations as they can generate inaccurate or irrelevant information, and these can include sensitive content that could be offensive, inappropriate, or controversial. Addressing such topics without inadvertently promoting hate speech or disrespecting certain beliefs remains a significant challenge. As a solution to these issues, we introduce MufassirQAS, a system that enhances LLM accuracy and transparency using a vector database-driven Retrieval-Augmented Generation (RAG) approach. We built a dataset comprising fundamental books containing Turkish translations and interpretations of Islamic texts. This database is leveraged to answer religious inquiries while ensuring that responses remain reliable and contextually grounded. Our system also presents the relevant dataset sections alongside the LLM-generated answers, reinforcing transparency. We carefully designed system prompts to prevent harmful, offensive, or disrespectful outputs, ensuring that responses align with ethical and respectful discourse. Moreover, MufassirQAS provides supplementary details, such as source page numbers and referenced articles, to enhance credibility. To evaluate its effectiveness, we tested MufassirQAS against ChatGPT with sensitive questions, and our system demonstrated superior performance in maintaining accuracy and reliability. Future work will focus on improving accuracy and refining prompt engineering techniques to further minimize biases and ensure even more reliable responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06749v2">DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. Within the Magnetic-One agent framework, on the datasets derived from GAIA and AssistantBench, DoVer flips 18-28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. DoVer also performs effectively on a different dataset (GSMPlus) and agent framework (AG2), where it recovers 49% of failed trials. These results highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems. Project website and code will be available at https://aka.ms/DoVer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03291v2">UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance across diverse tasks but face prohibitive computational and memory costs. Pruning offers a promising path by inducing sparsity while preserving architectural flexibility. However, existing methods struggle to balance efficiency and robustness: local metric approaches prune layer by layer but often collapse under high sparsity, whereas global feedback methods enforce consistency at the cost of expensive weight updates or restrictive semi-structured formats. We present UniPruning, a unified post-training pruning framework that combines the speed of local saliency metrics with the stability of global coordination, enabled by a mirror descent based optimization, all without updating model weights. UniPruning leverages fast layer-wise scoring and a lightweight global controller to allocate a single sparsity budget, supporting both unstructured and semi-structured N :M pruning within one framework. After a brief calibration, it can generate pruning masks for arbitrary sparsity levels in one shot, and adapts seamlessly to hardware-aware constraints. Extensive experiments on multiple pretrained LLM families and standard benchmarks show that UniPruning consistently delivers competitive or superior perplexity and zero-shot accuracy. Ablation studies further highlight the importance of mirror descent and local saliency anchoring. Overall, UniPruning provides an efficient, principled, and scalable solution for sparsifying large-scale LLMs. Our code is available at: https://github.com/RainbowQTT/UniPruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08536v1">Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Ethical awareness is critical for robots operating in human environments, yet existing automated planning tools provide little support. Manually specifying ethical rules is labour-intensive and highly context-specific. We present Principles2Plan, an interactive research prototype demonstrating how a human and a Large Language Model (LLM) can collaborate to produce context-sensitive ethical rules and guide automated planning. A domain expert provides the planning domain, problem details, and relevant high-level principles such as beneficence and privacy. The system generates operationalisable ethical rules consistent with these principles, which the user can review, prioritise, and supply to a planner to produce ethically-informed plans. To our knowledge, no prior system supports users in generating principle-grounded rules for classical planning contexts. Principles2Plan showcases the potential of human-LLM collaboration for making ethical automated planning more practical and feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16809v2">When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted to ICSE 2026 (RECODE workshop)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08493v1">LLM-based Vulnerable Code Augmentation: Generate or Refactor?</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 6 pages, Submitted to ESAAN 2026, Under pier review
    </div>
    <details class="paper-abstract">
      Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented CWEs. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a vulnerability classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24319v2">Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08476v1">A Multi-Agent LLM Framework for Design Space Exploration in Autonomous Driving Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Designing autonomous driving systems requires efficient exploration of large hardware/software configuration spaces under diverse environmental conditions, e.g., with varying traffic, weather, and road layouts. Traditional design space exploration (DSE) approaches struggle with multi-modal execution outputs and complex performance trade-offs, and often require human involvement to assess correctness based on execution outputs. This paper presents a multi-agent, large language model (LLM)-based DSE framework, which integrates multi-modal reasoning with 3D simulation and profiling tools to automate the interpretation of execution outputs and guide the exploration of system designs. Specialized LLM agents are leveraged to handle user input interpretation, design point generation, execution orchestration, and analysis of both visual and textual execution outputs, which enables identification of potential bottlenecks without human intervention. A prototype implementation is developed and evaluated on a robotaxi case study (an SAE Level 4 autonomous driving application). Compared with a genetic algorithm baseline, the proposed framework identifies more Pareto-optimal, cost-efficient solutions with reduced navigation time under the same exploration budget. Experimental results also demonstrate the efficiency of the adoption of the LLM-based approach for DSE. We believe that this framework paves the way to the design automation of autonomous driving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20781v3">Using LLMs in Generating Design Rationale for Software Architecture Decisions</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Preprint accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM), 2025
    </div>
    <details class="paper-abstract">
      Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. To further understand the trustworthiness and applicability of LLM-generated DR in practice, we conducted semi-structured interviews with six practitioners. Based on the experimental and interview results, we discussed the pros and cons of the three prompting strategies, the strengths and limitations of LLM-generated DR, and the implications for the practical use of LLM-generated DR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02350v2">LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)
    </div>
    <details class="paper-abstract">
      Converting natural language questions into SQL queries enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early text-to-SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the large language model era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models, including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek-R1, and others. Notably, DeepSeek-R1 achieves 88.40% accuracy in a zero-shot setting, and models under 10B parameters surpass 90% accuracy after fine-tuning. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark. Unlike the original WikiSQL, which was tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural-language-to-SQL models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08417v1">Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08403v1">DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Audio deepfake detection has recently garnered public concern due to its implications for security and reliability. Traditional deep learning methods have been widely applied to this task but often lack generalisability when confronted with newly emerging spoofing techniques and more tasks such as spoof attribution recognition rather than simple binary classification. In principle, Large Language Models (LLMs) are considered to possess the needed generalisation capabilities. However, previous research on Audio LLMs (ALLMs) indicates a generalization bottleneck in audio deepfake detection performance, even when sufficient data is available. Consequently, this study investigates the model architecture and examines the effects of the primary components of ALLMs, namely the audio encoder and the text-based LLM. Our experiments demonstrate that the careful selection and combination of audio encoders and text-based LLMs are crucial for unlocking the deepfake detection potential of ALLMs. We further propose an ALLM structure capable of generalizing deepfake detection abilities to out-of-domain spoofing tests and other deepfake tasks, such as spoof positioning and spoof attribution recognition. Our proposed model architecture achieves state-of-the-art (SOTA) performance across multiple datasets, including ASVSpoof2019, InTheWild, and Demopage, with accuracy reaching up to 95.76% on average, and exhibits competitive capabilities in other deepfake detection tasks such as attribution, and localisation compared to SOTA audio understanding models. Data and codes are provided in supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04964v6">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07172v2">NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 71 pages, 21 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using counterfactual law shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07497v2">How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 48 pages, 3 tables, 2 listings
    </div>
    <details class="paper-abstract">
      We investigate how large language models (LLMs) fail when operating as autonomous agents with tool-use capabilities. Using the Kamiwaza Agentic Merit Index (KAMI) v0.1 benchmark, we analyze 900 execution traces from three representative models - Granite 4 Small, Llama 4 Maverick, and DeepSeek V3.1 - across filesystem, text extraction, CSV analysis, and SQL scenarios. Rather than focusing on aggregate scores, we perform fine-grained, per-trial behavioral analysis to surface the strategies that enable successful multi-step tool execution and the recurrent failure modes that undermine reliability. Our findings show that model scale alone does not predict agentic robustness: Llama 4 Maverick (400B) performs only marginally better than Granite 4 Small (32B) in some uncertainty-driven tasks, while DeepSeek V3.1's superior reliability derives primarily from post-training reinforcement learning rather than architecture or size. Across models, we identify four recurring failure archetypes: premature action without grounding, over-helpfulness that substitutes missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. These patterns highlight the need for agentic evaluation methods that emphasize interactive grounding, recovery behavior, and environment-aware adaptation, suggesting that reliable enterprise deployment requires not just stronger models but deliberate training and design choices that reinforce verification, constraint discovery, and adherence to source-of-truth data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18321v3">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into multi-agent systems (MAS), where peer interactions shape individual decisions. While prior work has mainly examined conformity bias, we broaden the view to include how LLMs build rapport from prior interactions, discern and integrate high-quality peer information, and resist misleading inputs-abilities essential for achieving collective intelligence under complex social dynamics. We introduce KAIROS, a benchmark that simulates quiz-style collaboration with peer agents whose rapport levels and behaviours can be precisely controlled in both historical interactions and the current round. This unified setup enables systematic analysis of how rapport, peer actions, and the model's self-confidence jointly influence decision-making. Using KAIROS, we evaluate prompting, supervised fine-tuning, and reinforcement learning via Group Relative Policy Optimisation (GRPO). Results show that model scale is a primary factor moderating susceptibility to social influence: larger models are more resilient and benefit from prompting-based mitigation, whereas smaller models remain vulnerable. Only carefully configured GRPO training yields consistent robustness and performance gains for small models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08366v1">Reflecting with Two Voices: A Co-Adaptive Dual-Strategy Framework for LLM-Based Agent Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents often rely on external demonstrations or retrieval-augmented planning, leading to brittleness, poor generalization, and high computational overhead. Inspired by human problem-solving, we propose DuSAR (Dual-Strategy Agent with Reflecting) - a demonstration-free framework that enables a single frozen LLM to perform co-adaptive reasoning via two complementary strategies: a high-level holistic plan and a context-grounded local policy. These strategies interact through a lightweight reflection mechanism, where the agent continuously assesses progress via a Strategy Fitness Score and dynamically revises its global plan when stuck or refines it upon meaningful advancement, mimicking human metacognitive behavior. On ALFWorld and Mind2Web, DuSAR achieves state-of-the-art performance with open-source LLMs (7B-70B), reaching 37.1% success on ALFWorld (Llama3.1-70B) - more than doubling the best prior result (13.0%) - and 4.02% on Mind2Web, also more than doubling the strongest baseline. Remarkably, it reduces per-step token consumption by 3-9X while maintaining strong performance. Ablation studies confirm the necessity of dual-strategy coordination. Moreover, optional integration of expert demonstrations further boosts results, highlighting DuSAR's flexibility and compatibility with external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20616v2">Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Accepted by IEEE Control Systems Letters (L-CSS)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in knowledge acquisition, reasoning, and tool use, making them promising candidates for autonomous agent applications. However, training LLM agents for complex multi-turn task planning faces significant challenges, including sparse episode-wise rewards, credit assignment across long horizons, and the computational overhead of reinforcement learning in multi-turn interaction settings. To this end, this paper introduces a novel approach that transforms multi-turn task planning into single-turn task reasoning problems, enabling efficient policy optimization through Group Relative Policy Optimization (GRPO) with dense and verifiable reward from expert trajectories. Our theoretical analysis shows that GRPO improvement on single-turn task reasoning results in a lower bound of the multi-turn success probability under the minimal turns, as well as the generalization to subtasks with shorter horizons. Experimental evaluation on the complex task planning benchmark demonstrates that our 1.5B parameter model trained with single-turn GRPO achieves superior performance compared to larger baseline models up to 14B parameters, with success rates of 70% for long-horizon planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07797v1">LLM Use for Mental Health: Crowdsourcing Users' Sentiment-based Perspectives and Values from Social Discussions</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) chatbots like ChatGPT are increasingly used for mental health support. They offer accessible, therapeutic support but also raise concerns about misinformation, over-reliance, and risks in high-stakes contexts of mental health. We crowdsource large-scale users' posts from six major social media platforms to examine how people discuss their interactions with LLM chatbots across different mental health conditions. Through an LLM-assisted pipeline grounded in Value-Sensitive Design (VSD), we mapped the relationships across user-reported sentiments, mental health conditions, perspectives, and values. Our results reveal that the use of LLM chatbots is condition-specific. Users with neurodivergent conditions (e.g., ADHD, ASD) report strong positive sentiments and instrumental or appraisal support, whereas higher-risk disorders (e.g., schizophrenia, bipolar disorder) show more negative sentiments. We further uncover how user perspectives co-occur with underlying values, such as identity, autonomy, and privacy. Finally, we discuss shifting from "one-size-fits-all" chatbot design toward condition-specific, value-sensitive LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07795v1">ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 11 pages, 3 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in settings where reasoning, such as multi-step problem solving and chain-of-thought, is essential. Yet, current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding. This omission creates a blind spot because practitioners cannot reliably assess whether a method's reported performance is stable, reproducible, or cost-consistent. We introduce ReasonBENCH, the first benchmark designed to quantify the underlying instability in LLM reasoning. ReasonBENCH provides (i) a modular evaluation library that standardizes reasoning frameworks, models, and tasks, (ii) a multi-run protocol that reports statistically reliable metrics for both quality and cost, and (iii) a public leaderboard to encourage variance-aware reporting. Across tasks from different domains, we find that the vast majority of reasoning strategies and models exhibit high instability. Notably, even strategies with similar average performance can display confidence intervals up to four times wider, and the top-performing methods often incur higher and less stable costs. Such instability compromises reproducibility across runs and, consequently, the reliability of reported performance. To better understand these dynamics, we further analyze the impact of prompts, model families, and scale on the trade-off between solve rate and stability. Our results highlight reproducibility as a critical dimension for reliable LLM reasoning and provide a foundation for future reasoning methods and uncertainty quantification techniques. ReasonBENCH is publicly available at https://github.com/au-clan/ReasonBench .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07777v1">Mary, the Cheeseburger-Eating Vegetarian: Do LLMs Recognize Incoherence in Narratives?</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Leveraging a dataset of paired narratives, we investigate the extent to which large language models (LLMs) can reliably separate incoherent and coherent stories. A probing study finds that LLMs' internal representations can reliably identify incoherent narratives. However, LLMs generate responses to rating questions that fail to satisfactorily separate the coherent and incoherent narratives across several prompt variations, hinting at a gap in LLM's understanding of storytelling. The reasoning LLMs tested do not eliminate these deficits, indicating that thought strings may not be able to fully address the discrepancy between model internal state and behavior. Additionally, we find that LLMs appear to be more sensitive to incoherence resulting from an event that violates the setting (e.g., a rainy day in the desert) than to incoherence arising from a character violating an established trait (e.g., Mary, a vegetarian, later orders a cheeseburger), suggesting that LLMs may rely more on prototypical world knowledge than building meaning-based narrative coherence. The consistent asymmetry found in our results suggests that LLMs do not have a complete grasp on narrative coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06405v2">TOOL4POI: A Tool-Augmented LLM Framework for Next POI Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 A critical technical error was discovered during our internal review, leading to unreliable experimental results. The issue cannot be resolved, and the paper has also been formally withdrawn from AAAI 2026. We therefore request withdrawal of the arXiv version to maintain scientific accuracy
    </div>
    <details class="paper-abstract">
      Next Point-of-Interest (POI) recommendation is a fundamental task in location-based services. While recent advances leverage Large Language Model (LLM) for sequential modeling, existing LLM-based approaches face two key limitations: (i) strong reliance on the contextual completeness of user histories, resulting in poor performance on out-of-history (OOH) scenarios; (ii) limited scalability, due to the restricted context window of LLMs, which limits their ability to access and process a large number of candidate POIs. To address these challenges, we propose Tool4POI, a novel tool-augmented framework that enables LLMs to perform open-set POI recommendation through external retrieval and reasoning. Tool4POI consists of three key modules: preference extraction module, multi-turn candidate retrieval module, and reranking module, which together summarize long-term user interests, interact with external tools to retrieve relevant POIs, and refine final recommendations based on recent behaviors. Unlike existing methods, Tool4POI requires no task-specific fine-tuning and is compatible with off-the-shelf LLMs in a plug-and-play manner. Extensive experiments on three real-world datasets show that Tool4POI substantially outperforms state-of-the-art baselines, achieving up to 40% accuracy on challenging OOH scenarios where existing methods fail, and delivering average improvements of 20% and 30% on Acc@5 and Acc@10, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05156v2">Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 23 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07611v1">Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      This study presents a systematic comparison of three Reinforcement Learning (RL) algorithms (PPO, GRPO, and DAPO) for improving complex reasoning in large language models (LLMs). Our main contribution is a controlled transfer-learning evaluation: models are first fine-tuned on the specialized Countdown Game and then assessed on a suite of general-purpose reasoning benchmarks. Across all tasks, RL-trained models outperform their corresponding base models, although the degree of improvement differs by benchmark. Our parametric analysis offers practical guidance for RL-based LLM training. Increasing the group size in GRPO and DAPO leads to more stable training dynamics and higher accuracy, while the impact of the KL-penalty coefficient is non-monotonic. Additionally, we find that the Dynamic Sampling (DS) component in DAPO does not improve performance; in fact, the best overall results are achieved with DAPO when DS is disabled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06504v2">When Code Crosses Borders: A Security-Centric Study of LLM-based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Code translation is crucial for cross-language codebase migration, and large language models (LLMs) have emerged as a promising technique to automate this process. However, the security implications of using LLMs for code translation remain largely unexplored, as existing evaluations primarily focus on syntactic and functional correctness. To bridge this gap, we conduct a security-centric empirical study to investigate the risks of vulnerabilities being introduced or preserved during LLM-based translation. Our study involves a rigorous evaluation of five state-of-the-art LLMs on a curated dataset of 720 security-related code samples across four programming languages (Java, PHP, C, C++) and nine Common Weakness Enumeration (CWE) categories. The results reveal significant security degradation, with 28.6\% to 45\% of translations introducing new vulnerabilities. Web-related flaws, particularly in input validation, proved most challenging for LLMs. Furthermore, we identify and categorize the root causes of these vulnerable translations into a taxonomy of five major error types. Based on our findings, we develop and evaluate a Retrieval-Augmented Generation (RAG)-based mitigation strategy, which successfully reduces the vulnerability introduction rate by 32.8\%. Our study provides the first large-scale evidence of serious security risks in LLM-based code translation and demonstrates the potential of knowledge-enhanced prompting to mitigate them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16653v2">H2EAL: Hybrid-Bonding Architecture with Hybrid Sparse Attention for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 International Conference on Computer-Aided Design (ICCAD) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in a wide range of natural language processing applications. However, the high energy and latency overhead induced by the KV cache limits the edge deployment, especially for long contexts. Emerging hybrid bonding (HB) technology has been proposed as a promising alternative to conventional near-memory processing (NMP) architectures, offering improved bandwidth efficiency and lower power consumption while exhibiting characteristics of distributed memory. In this paper, we propose H2EAL, a hybrid bonding-based accelerator with sparse attention algorithm-hardware co-design for efficient LLM inference at the edge. At the algorithm level, we propose a hybrid sparse attention scheme with static and dynamic sparsity for different heads to fully leverage the sparsity with high accuracy. At the hardware level, we co-design the hardware to support hybrid sparse attention and propose memory-compute co-placement to address the distributed memory bottleneck. Since different attention heads exhibit different sparse patterns and the attention structure often mismatches the HB architecture, we further develop a load-balancing scheduler with parallel tiled attention to address workload imbalance and optimize the mapping strategy. Extensive experiments demonstrate H2EAL achieves 5.20~48.21x speedup and 6.22~73.48x energy efficiency improvement over baseline HB implementation, with a negligible average accuracy drop of 0.87% on multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15830v4">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-based approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07533v1">VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      We propose VulnLLM-R, the~\emph{first specialized reasoning LLM} for vulnerability detection. Our key insight is that LLMs can reason about program states and analyze the potential vulnerabilities, rather than simple pattern matching. This can improve the model's generalizability and prevent learning shortcuts. However, SOTA reasoning LLMs are typically ultra-large, closed-source, or have limited performance in vulnerability detection. To address this, we propose a novel training recipe with specialized data selection, reasoning data generation, reasoning data filtering and correction, and testing-phase optimization. Using our proposed methodology, we train a reasoning model with seven billion parameters. Through extensive experiments on SOTA datasets across Python, C/C++, and Java, we show that VulnLLM-R has superior effectiveness and efficiency than SOTA static analysis tools and both open-source and commercial large reasoning models. We further conduct a detailed ablation study to validate the key designs in our training recipe. Finally, we construct an agent scaffold around our model and show that it outperforms CodeQL and AFL++ in real-world projects. Our agent further discovers a set of zero-day vulnerabilities in actively maintained repositories. This work represents a pioneering effort to enable real-world, project-level vulnerability detection using AI agents powered by specialized reasoning models. The code is available at~\href{https://github.com/ucsb-mlsec/VulnLLM-R}{github}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07525v1">Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 20 pages, 6 figures, under review
    </div>
    <details class="paper-abstract">
      Rotary Position Embeddings (RoPE) have become a standard for encoding sequence order in Large Language Models (LLMs) by applying rotations to query and key vectors in the complex plane. Standard implementations, however, utilize only the real component of the complex-valued dot product for attention score calculation. This simplification discards the imaginary component, which contains valuable phase information, leading to a potential loss of relational details crucial for modeling long-context dependencies. In this paper, we propose an extension that re-incorporates this discarded imaginary component. Our method leverages the full complex-valued representation to create a dual-component attention score. We theoretically and empirically demonstrate that this approach enhances the modeling of long-context dependencies by preserving more positional information. Furthermore, evaluations on a suite of long-context language modeling benchmarks show that our method consistently improves performance over the standard RoPE, with the benefits becoming more significant as context length increases. The code is available at https://github.com/OpenMOSS/rope_pp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07501v1">AutoICE: Automatically Synthesizing Verifiable C Code via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Automatically synthesizing verifiable code from natural language requirements ensures software correctness and reliability while significantly lowering the barrier to adopting the techniques of formal methods. With the rise of large language models (LLMs), long-standing efforts at autoformalization have gained new momentum. However, existing approaches suffer from severe syntactic and semantic errors due to the scarcity of domain-specific pre-training corpora and often fail to formalize implicit knowledge effectively. In this paper, we propose AutoICE, an LLM-driven evolutionary search for synthesizing verifiable C code. It introduces the diverse individual initialization and the collaborative crossover to enable diverse iterative updates, thereby mitigating error propagation inherent in single-agent iterations. Besides, it employs the self-reflective mutation to facilitate the discovery of implicit knowledge. Evaluation results demonstrate the effectiveness of AutoICE: it successfully verifies $90.36$\% of code, outperforming the state-of-the-art (SOTA) approach. Besides, on a developer-friendly dataset variant, AutoICE achieves a $88.33$\% verification success rate, significantly surpassing the $65$\% success rate of the SOTA approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07497v1">How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 48 pages, 3 tables, 2 listings
    </div>
    <details class="paper-abstract">
      We investigate how large language models (LLMs) fail when operating as autonomous agents with tool-use capabilities. Using the Kamiwaza Agentic Merit Index (KAMI) v0.1 benchmark, we analyze 900 execution traces from three representative models - Granite 4 Small, Llama 4 Maverick, and DeepSeek V3.1 - across filesystem, text extraction, CSV analysis, and SQL scenarios. Rather than focusing on aggregate scores, we perform fine-grained, per-trial behavioral analysis to surface the strategies that enable successful multi-step tool execution and the recurrent failure modes that undermine reliability. Our findings show that model scale alone does not predict agentic robustness: Llama 4 Maverick (400B) performs only marginally better than Granite 4 Small (32B) in some uncertainty-driven tasks, while DeepSeek V3.1's superior reliability derives primarily from post-training reinforcement learning rather than architecture or size. Across models, we identify four recurring failure archetypes: premature action without grounding, over-helpfulness that substitutes missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. These patterns highlight the need for agentic evaluation methods that emphasize interactive grounding, recovery behavior, and environment-aware adaptation, suggesting that reliable enterprise deployment requires not just stronger models but deliberate training and design choices that reinforce verification, constraint discovery, and adherence to source-of-truth data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.07639v2">PowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Efficiently solving Optimal Power Flow (OPF) problems in power systems is crucial for operational planning and grid management. There is a growing need for scalable algorithms capable of handling the increasing variability, constraints, and uncertainties in modern power networks while providing accurate and fast solutions. To address this, machine learning techniques, particularly Graph Neural Networks (GNNs) have emerged as promising approaches. This letter introduces PowerGraph-LLM, the first framework explicitly designed for solving OPF problems using Large Language Models (LLMs). The proposed approach combines graph and tabular representations of power grids to effectively query LLMs, capturing the complex relationships and constraints in power systems. A new implementation of in-context learning and fine-tuning protocols for LLMs is introduced, tailored specifically for the OPF problem. PowerGraph-LLM demonstrates reliable performances using off-the-shelf LLM. Our study reveals the impact of LLM architecture, size, and fine-tuning and demonstrates our framework's ability to handle realistic grid components and constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07462v1">Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07454v1">Persian-Phi: Efficient Cross-Lingual Adaptation of Compact LLMs via Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      The democratization of AI is currently hindered by the immense computational costs required to train Large Language Models (LLMs) for low-resource languages. This paper presents Persian-Phi, a 3.8B parameter model that challenges the assumption that robust multilingual capabilities require massive model sizes or multilingual baselines. We demonstrate how Microsoft Phi-3 Mini -- originally a monolingual English model -- can be effectively adapted to Persian through a novel, resource-efficient curriculum learning pipeline. Our approach employs a unique "warm-up" stage using bilingual narratives (Tiny Stories) to align embeddings prior to heavy training, followed by continual pretraining and instruction tuning via Parameter-Efficient Fine-Tuning (PEFT). Despite its compact size, Persian-Phi achieves competitive results on Open Persian LLM Leaderboard in HuggingFace. Our findings provide a validated, scalable framework for extending the reach of state-of-the-art LLMs to underrepresented languages with minimal hardware resources. The Persian-Phi model is publicly available at https://huggingface.co/amirakhlaghiqqq/PersianPhi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v1">Do LLMs Trust the Code They Write?</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07375v1">LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess vast knowledge acquired from extensive training corpora, but they often cannot remove specific pieces of information when needed, which makes it hard to handle privacy, bias mitigation, and knowledge correction. Traditional model unlearning approaches require computationally expensive fine-tuning or direct weight editing, making them impractical for real-world deployment. In this work, we introduce LoRA-based Unlearning with Negative Examples (LUNE), a lightweight framework that performs negative-only unlearning by updating only low-rank adapters while freezing the backbone, thereby localizing edits and avoiding disruptive global changes. Leveraging Low-Rank Adaptation (LoRA), LUNE targets intermediate representations to suppress (or replace) requested knowledge with an order-of-magnitude lower compute and memory than full fine-tuning or direct weight editing. Extensive experiments on multiple factual unlearning tasks show that LUNE: (I) achieves effectiveness comparable to full fine-tuning and memory-editing methods, and (II) reduces computational cost by about an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07374v1">Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Unlearning in large foundation models (e.g., LLMs) is essential for enabling dynamic knowledge updates, enforcing data deletion rights, and correcting model behavior. However, existing unlearning methods often require full-model fine-tuning or access to the original training data, which limits their scalability and practicality. In this work, we introduce Recover-to-Forget (R2F), a novel framework for efficient unlearning in LLMs based on reconstructing full-model gradient directions from low-rank LoRA adapter updates. Rather than performing backpropagation through the full model, we compute gradients with respect to LoRA parameters using multiple paraphrased prompts and train a gradient decoder to approximate the corresponding full-model gradients. To ensure applicability to larger or black-box models, the decoder is trained on a proxy model and transferred to target models. We provide a theoretical analysis of cross-model generalization and demonstrate that our method achieves effective unlearning while preserving general model performance. Experimental results demonstrate that R2F offers a scalable and lightweight alternative for unlearning in pretrained LLMs without requiring full retraining or access to internal parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07312v1">DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) is pushing AI accelerators toward increasingly powerful and specialized designs. Instead of further complicating software development with deeply hierarchical scratchpad memories (SPMs) and their asynchronous management, we investigate the opposite point of the design spectrum: a multi-core AI accelerator equipped with a shared system-level cache and application-aware management policies, which keeps the programming effort modest. Our approach exploits dataflow information available in the software stack to guide cache replacement (including dead-block prediction), in concert with bypass decisions and mechanisms that alleviate cache thrashing. We assess the proposal using a cycle-accurate simulator and observe substantial performance gains (up to 1.80x speedup) compared with conventional cache architectures. In addition, we build and validate an analytical model that takes into account the actual overlapping behaviors to extend the measurement results of our policies to real-world larger-scale workloads. Experiment results show that when functioning together, our bypassing and thrashing mitigation strategies can handle scenarios both with and without inter-core data sharing and achieve remarkable speedups. Finally, we implement the design in RTL and the area of our design is $\mathbf{0.064mm^2}$ with 15nm process, which can run at 2 GHz clock frequency. Our findings explore the potential of the shared cache design to assist the development of future AI accelerator systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24282v2">SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol, the global industry standard for smart home communication, SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 16 agents under a unified ReAct framework reveals distinct capabilities and limitations across models. Models under 7B parameters exhibited negligible performance across all query types. Even GPT-4.1, the best-performing standard model, struggled with implicit intent inference, state verification, and particularly temporal scheduling. While reasoning models such as GPT-5.1 consistently outperformed standard models on every query type, they required over three times the average inference time, which can be prohibitive for real-time smart home applications. This highlights a critical trade-off between task performance and real-world practicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06041v4">DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      How can we effectively handle queries for on-device large language models (LLMs) with varying runtime constraints, such as latency and accuracy? Multi-scale quantization addresses this challenge by enabling memory-efficient runtime model adaptation of LLMs through the overlaying of multiple model variants quantized to different bitwidths. Meanwhile, an important question still remains open-ended: how can models be properly configured to match a target precision or latency? While mixed-precision offers a promising solution, we take this further by leveraging the key observation that the sensitivity of each layer dynamically changes across decoding steps. Building on this insight, we introduce DP-LLM, a novel mechanism that dynamically assigns precision to each layer based on input values. Experimental results across multiple models and benchmarks demonstrate that DP-LLM achieves a superior performance-latency trade-off, outperforming prior approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07246v1">Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability embedded in large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline (1) relies on the black box, implicit decision process, thus failing to provide explainability for the detection results, and (2) is highly sensitive to prompts, yielding inconsistent outputs due to inherent model stochasticity, therefore lacking robustness. To address these limitations, we propose an LLM-as-an-inducer framework that adopts LLM to induce the decision tree for ED (termed TreeED) and further ensembles multiple such trees for consensus detection (termed ForestED), thereby improving explainability and robustness. Specifically, based on prompts derived from data context, decision tree specifications and output requirements, TreeED queries the LLM to induce the decision tree skeleton, whose root-to-leaf decision paths specify the stepwise procedure for evaluating a given sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks (e.g., format or range), (2) Graph Neural Network (GNN) nodes that capture complex patterns (e.g., functional dependencies), and (3) leaf nodes that output the final decision types (error or clean). Furthermore, ForestED employs uncertainty-based sampling to obtain multiple row subsets, constructing a decision tree for each subset using TreeED. It then leverages an Expectation-Maximization-based algorithm that jointly estimates tree reliability and optimizes the consensus ED prediction. Extensive xperiments demonstrate that our methods are accurate, explainable and robust, achieving an average F1-score improvement of 16.1% over the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05485v2">TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00417v3">CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04668v2">Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20993v2">Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17432v4">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 13 pages, 8 figures, Accepted by ACM MM2025, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLMś language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07122v1">RisConFix: LLM-based Automated Repair of Risk-Prone Drone Configurations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Flight control software is typically designed with numerous configurable parameters governing multiple functionalities, enabling flexible adaptation to mission diversity and environmental uncertainty. Although developers and manufacturers usually provide recommendations for these parameters to ensure safe and stable operations, certain combinations of parameters with recommended values may still lead to unstable flight behaviors, thereby degrading the drone's robustness. To this end, we propose a Large Language Model (LLM) based approach for real-time repair of risk-prone configurations (named RisConFix) that degrade drone robustness. RisConFix continuously monitors the drone's operational state and automatically triggers a repair mechanism once abnormal flight behaviors are detected. The repair mechanism leverages an LLM to analyze relationships between configuration parameters and flight states, and then generates corrective parameter updates to restore flight stability. To ensure the validity of the updated configuration, RisConFix operates as an iterative process; it continuously monitors the drone's flight state and, if an anomaly persists after applying an update, automatically triggers the next repair cycle. We evaluated RisConFix through a case study of ArduPilot (with 1,421 groups of misconfigurations). Experimental results show that RisConFix achieved a best repair success rate of 97% and an optimal average number of repairs of 1.17, demonstrating its capability to effectively and efficiently repair risk-prone configurations in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11227v2">Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Accepted by NeurIPS 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07112v1">FOAM: Blocked State Folding for Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance due to their large parameter counts and extensive training data. However, their scale leads to significant memory bottlenecks during training, especially when using memory-intensive optimizers like Adam. Existing memory-efficient approaches often rely on techniques such as singular value decomposition (SVD), projections, or weight freezing, which can introduce substantial computational overhead, require additional memory for projections, or degrade model performance. In this paper, we propose Folded Optimizer with Approximate Moment (FOAM), a method that compresses optimizer states by computing block-wise gradient means and incorporates a residual correction to recover lost information. Theoretically, FOAM achieves convergence rates equivalent to vanilla Adam under standard non-convex optimization settings. Empirically, FOAM reduces total training memory by approximately 50\%, eliminates up to 90\% of optimizer state memory overhead, and accelerates convergence. Furthermore, FOAM is compatible with other memory-efficient optimizers, delivering performance and throughput that match or surpass both full-rank and existing memory-efficient baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00450v2">Dual Collaborative LLMs via Continual Fine-Tuning for Serendipitous Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Traditional recommendation systems tend to trap users in strong feedback loops by excessively pushing content aligned with their historical preferences, thereby limiting exploration opportunities and causing content fatigue. Although large language models (LLMs) demonstrate potential with their diverse content generation capabilities, existing LLM-enhanced dual-model frameworks face two major limitations: first, they overlook long-term preferences driven by group identity, leading to biased interest modeling; second, they suffer from static optimization flaws, as a one-time alignment process fails to leverage incremental user data for closed-loop optimization. To address these challenges, we propose the Co-Evolutionary Alignment (CoEA) method. For interest modeling bias, we introduce Dual-Stable Interest Exploration (DSIE) module, jointly modeling long-term group identity and short-term individual interests through parallel processing of behavioral sequences. For static optimization limitations, we design a Periodic Collaborative Optimization (PCO) mechanism. This mechanism regularly conducts preference verification on incremental data using the Relevance LLM, then guides the Novelty LLM to perform fine-tuning based on the verification results, and subsequently feeds back the output of the continually fine-tuned Novelty LLM to the Relevance LLM for re-evaluation, thereby achieving a dynamic closed-loop optimization. Extensive online and offline experiments verify the effectiveness of the CoEA model in serendipitous recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09442v3">Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07090v1">Leveraging KV Similarity for Online Structured Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Pruning has emerged as a promising direction for accelerating large language model (LLM) inference, yet existing approaches often suffer from instability because they rely on offline calibration data that may not generalize across inputs. In this work, we introduce Token Filtering, a lightweight online structured pruning technique that makes pruning decisions directly during inference without any calibration data. The key idea is to measure token redundancy via joint key-value similarity and skip redundant attention computations, thereby reducing inference cost while preserving critical information. To further enhance stability, we design a variance-aware fusion strategy that adaptively weights key and value similarity across heads, ensuring that informative tokens are retained even under high pruning ratios. This design introduces no additional memory overhead and provides a more reliable criterion for token importance. Extensive experiments on LLaMA-2 (7B/13B), LLaMA-3 (8B), and Mistral (7B) demonstrate that Token Filtering consistently outperforms prior structured pruning methods, preserving accuracy on commonsense reasoning benchmarks and maintaining strong performance on challenging tasks such as MMLU, even with 50% pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07086v1">ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 This version includes the final camera-ready manuscript accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07081v1">ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 10 pages, 2 figures. Submitted to AMIA 2026 Informatics Summit Student Paper Track
    </div>
    <details class="paper-abstract">
      Heart failure (HF) is one of the leading causes of rehospitalization among older adults in the United States. Although clinical notes contain rich, detailed patient information and make up a large portion of electronic health records (EHRs), they remain underutilized for HF readmission risk analysis. Traditional computational models for HF readmission often rely on expert-crafted rules, medical thesauri, and ontologies to interpret clinical notes, which are typically written under time pressure and may contain misspellings, abbreviations, and domain-specific jargon. We present ClinNoteAgents, an LLM-based multi-agent framework that transforms free-text clinical notes into (1) structured representations of clinical and social risk factors for association analysis and (2) clinician-style abstractions for HF 30-day readmission prediction. We evaluate ClinNoteAgents on 3,544 notes from 2,065 patients (readmission rate=35.16%), demonstrating strong performance in extracting risk factors from free-text, identifying key contributing factors, and predicting readmission risk. By reducing reliance on structured fields and minimizing manual annotation and model training, ClinNoteAgents provides a scalable and interpretable approach to note-based HF readmission risk modeling in data-limited healthcare systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08121v1">Balanced Accuracy: The Right Metric for Evaluating LLM Judges -- Explained through Youden's J statistic</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08093v1">Training LLMs for Honesty via Confessions</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be dishonest when reporting on their actions and beliefs -- for example, they may overstate their confidence in factual claims or cover up evidence of covert actions. Such dishonesty may arise due to the effects of reinforcement learning (RL), where challenges with reward shaping can result in a training process that inadvertently incentivizes the model to lie or misrepresent its actions. In this work we propose a method for eliciting an honest expression of an LLM's shortcomings via a self-reported *confession*. A confession is an output, provided upon request after a model's original answer, that is meant to serve as a full account of the model's compliance with the letter and spirit of its policies and instructions. The reward assigned to a confession during training is solely based on its honesty, and does not impact positively or negatively the main answer's reward. As long as the "path of least resistance" for maximizing confession reward is to surface misbehavior rather than covering it up, this incentivizes models to be honest in their confessions. Our findings provide some justification this empirical assumption, especially in the case of egregious model misbehavior. To demonstrate the viability of our approach, we train GPT-5-Thinking to produce confessions, and we evaluate its honesty in out-of-distribution scenarios measuring hallucination, instruction following, scheming, and reward hacking. We find that when the model lies or omits shortcomings in its "main" answer, it often confesses to these behaviors honestly, and this confession honesty modestly improves with training. Confessions can enable a number of inference-time interventions including monitoring, rejection sampling, and surfacing issues to the user.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08088v1">Adaptation of Embedding Models to Financial Filings via LLM Distillation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 In proceedings of LLM-Finance 2025 : The 2nd IEEE International Workshop on Large Language Models for Finance
    </div>
    <details class="paper-abstract">
      Despite advances in generative large language models (LLMs), practical application of specialized conversational AI agents remains constrained by computation costs, latency requirements, and the need for precise domain-specific relevance measures. While existing embedding models address the first two constraints, they underperform on information retrieval in specialized domains like finance. This paper introduces a scalable pipeline that trains specialized models from an unlabeled corpus using a general purpose retrieval embedding model as foundation. Our method yields an average of 27.7% improvement in MRR$\texttt{@}$5, 44.6% improvement in mean DCG$\texttt{@}$5 across 14 financial filing types measured over 21,800 query-document pairs, and improved NDCG on 3 of 4 document classes in FinanceBench. We adapt retrieval embeddings (bi-encoder) for RAG, not LLM generators, using LLM-judged relevance to distill domain knowledge into a compact retriever. There are prior works which pair synthetically generated queries with real passages to directly fine-tune the retrieval model. Our pipeline differs from these by introducing interaction between student and teacher models that interleaves retrieval-based mining of hard positive/negative examples from the unlabeled corpus with iterative retraining of the student model's weights using these examples. Each retrieval iteration uses the refined student model to mine the corpus for progressively harder training examples for the subsequent training iteration. The methodology provides a cost-effective solution to bridging the gap between general-purpose models and specialized domains without requiring labor-intensive human annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08083v1">Exploiting the Randomness of Large Language Models (LLM) in Text Classification Tasks: Locating Privileged Documents in Legal Matters</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      In legal matters, text classification models are most often used to filter through large datasets in search of documents that meet certain pre-selected criteria like relevance to a certain subject matter, such as legally privileged communications and attorney-directed documents. In this context, large language models have demonstrated strong performance. This paper presents an empirical study investigating the role of randomness in LLM-based classification for attorney-client privileged document detection, focusing on four key dimensions: (1) the effectiveness of LLMs in identifying legally privileged documents, (2) the influence of randomness control parameters on classification outputs, (3) their impact on overall classification performance, and (4) a methodology for leveraging randomness to enhance accuracy. Experimental results showed that LLMs can identify privileged documents effectively, randomness control parameters have minimal impact on classification performance, and importantly, our developed methodology for leveraging randomness can have a significant impact on improving accuracy. Notably, this methodology that leverages randomness could also enhance a corporation's confidence in an LLM's output when incorporated into its sanctions-compliance processes. As organizations increasingly rely on LLMs to augment compliance workflows, reducing output variability helps build internal and regulatory confidence in LLM-derived sanctions-screening decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03503v2">Understanding LLM Reasoning for Abstractive Summarization</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 26 pages,15 figures
    </div>
    <details class="paper-abstract">
      While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03817v2">Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Multi-agent systems of large language models (LLMs) show promise for complex reasoning, but their effectiveness is often limited by fixed collaboration protocols. These frameworks typically focus on macro-level orchestration while overlooking agents' internal deliberative capabilities. This critical meta-cognitive blindspot treats agents as passive executors unable to adapt their strategy based on internal cognitive states like uncertainty or confidence. We introduce the Meta-Policy Deliberation Framework (MPDF), where agents learn a decentralized policy over a set of high-level meta-cognitive actions: Persist, Refine, and Concede. To overcome the instability of traditional policy gradients in this setting, we develop SoftRankPO, a novel reinforcement learning algorithm. SoftRankPO stabilizes training by shaping advantages based on the rank of rewards mapped through smooth normal quantiles, making the learning process robust to reward variance. Experiments show that MPDF with SoftRankPO achieves a a 4-5% absolute gain in average accuracy across five mathematical and general reasoning benchmarks compared to six state-of-the-art heuristic and learning-based multi-agent reasoning algorithms. Our work presents a paradigm for learning adaptive, meta-cognitive policies for multi-agent LLM systems, shifting the focus from designing fixed protocols to learning dynamic, deliberative strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05154v3">Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07785v1">Automating High Energy Physics Data Analysis with LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 16 pages, 6 figures, 2 tables, the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) - Machine Learning and the Physical Sciences (ML4PS) workshop (poster)
    </div>
    <details class="paper-abstract">
      We present a proof-of-principle study demonstrating the use of large language model (LLM) agents to automate a representative high energy physics (HEP) analysis. Using the Higgs boson diphoton cross-section measurement as a case study with ATLAS Open Data, we design a hybrid system that combines an LLM-based supervisor-coder agent with the Snakemake workflow manager. In this architecture, the workflow manager enforces reproducibility and determinism, while the agent autonomously generates, executes, and iteratively corrects analysis code in response to user instructions. We define quantitative evaluation metrics including success rate, error distribution, costs per specific task, and average number of API calls, to assess agent performance across multi-stage workflows. To characterize variability across architectures, we benchmark a representative selection of state-of-the-art LLMs spanning the Gemini and GPT-5 series, the Claude family, and leading open-weight models. While the workflow manager ensures deterministic execution of all analysis steps, the final outputs still show stochastic variation. Although we set the temperature to zero, other sampling parameters (e.g., top-p, top-k) remained at their defaults, and some reasoning-oriented models internally adjust these settings. Consequently, the models do not produce fully deterministic results. This study establishes the first LLM-agent-driven automated data-analysis framework in HEP, enabling systematic benchmarking of model capabilities, stability, and limitations in real-world scientific computing environments. The baseline code used in this work is available at https://huggingface.co/HWresearch/LLM4HEP. This work was accepted as a poster at the Machine Learning and the Physical Sciences (ML4PS) workshop at NeurIPS 2025. The initial submission was made on August 30, 2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20833v2">LLMs are Biased Evaluators But Not Biased for Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 15 pages, 14 tables, 5 figures Accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated that large language models (LLMs) exhibit significant biases in evaluation tasks, particularly in preferentially rating and favoring self-generated content. However, the extent to which this bias manifests in fact-oriented tasks, especially within retrieval-augmented generation (RAG) frameworks, where keyword extraction and factual accuracy take precedence over stylistic elements, remains unclear. Our study addresses this knowledge gap by simulating two critical phases of the RAG framework. In the first phase, LLMs evaluated human-authored and model-generated passages, emulating the \textit{pointwise reranking phase}. The second phase involves conducting pairwise reading comprehension tests to simulate the \textit{generation phase}. Contrary to previous findings indicating a self-preference in rating tasks, our results reveal no significant self-preference effect in RAG frameworks. Instead, we observe that factual accuracy significantly influences LLMs' output, even in the absence of prior knowledge. These findings are consistent among three common QA datasets (NQ, MARCO, TriviaQA Datasets) and 5 widely adopted language models (GPT-3.5, GPT-4o-mini, Gemini, LLaMA3, and Mistral). Our research contributes to the ongoing discourse on LLM biases and their implications for RAG-based system, offering insights that may inform the development of more robust and unbiased LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06776v1">From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at generation but dominant autoregressive (AR) decoding is inherently sequential, creating a throughput bottleneck. Diffusion Language Models (DLMs)--especially block-wise variants--enable parallel generation and intra-block bidirectional reasoning, yet training large DLMs from scratch is costly and wastes the knowledge in mature AR checkpoints. Prior "adaptation" attempts either modify logits or randomly grow attention masks to full-sequence diffusion, or simply transplant AR weights into a block-diffusion recipe, leaving a fundamental mismatch between AR causality and block-wise bidirectionality unaddressed. We reframe adaptation as a intra-paradigm path from AR to Block-Diffusion by viewing AR as Block-Diffusion with blocksize=1. Concretely, we design the pathway of adaptation as follows: we use a context-causal attention mask (causal in context, bidirectional only within the active block), an efficient parallel adaptation procedure, an auxiliary AR loss to maximize data utilization and retain pretrained knowledge, and gradual increment of the generation block size. The recipe integrates cleanly with masked block-diffusion and maintains train-inference consistency. Built on these components, NBDiff-7B (Base and Instruct) could inherit the long-context modeling and reasoning capabilities, and achieve state-of-the-art performance among the 7B-class DLMs, delivering strong gains on general-knowledge, math, and code benchmarks over strong baselines. These results demonstrate that principled AR-to-block-diffusion adaptation is an effective and compute-efficient alternative to training DLMs from scratch. Codes: https://github.com/YuchuanTian/NBDiff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06749v1">DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. Within the Magnetic-One agent framework, on the datasets derived from GAIA and AssistantBench, DoVer flips 18-28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. DoVer also performs effectively on a different dataset (GSMPlus) and agent framework (AG2), where it recovers 49% of failed trials. These results highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems. Project website and code will be available at https://aka.ms/DoVer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06747v1">PrivLLMSwarm: Privacy-Preserving LLM-Driven UAV Swarms for Secure IoT Surveillance</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as powerful enablers for autonomous reasoning and natural-language coordination in unmanned aerial vehicle (UAV) swarms operating within Internet of Things (IoT) environments. However, existing LLM-driven UAV systems process sensitive operational data in plaintext, exposing them to privacy and security risks. This work introduces PrivLLMSwarm, a privacy-preserving framework that performs secure LLM inference for UAV swarm coordination through Secure Multi-Party Computation (MPC). The framework incorporates MPC-optimized transformer components with efficient approximations of nonlinear activations, enabling practical encrypted inference on resource-constrained aerial platforms. A fine-tuned GPT-based command generator, enhanced through reinforcement learning in simulation, provides reliable instructions while maintaining confidentiality. Experimental evaluation in urban-scale simulations demonstrates that PrivLLMSwarm achieves high semantic accuracy, low encrypted inference latency, and robust formation control under privacy constraints. Comparative analysis shows PrivLLMSwarm offers a superior privacy-utility balance compared to differential privacy, federated learning, and plaintext baselines. To support reproducibility, the full implementation including source code, MPC components, and a synthetic dataset is publicly available. PrivLLMSwarm establishes a practical foundation for secure, LLM-enabled UAV swarms in privacy-sensitive IoT applications including smart-city monitoring and emergency response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09946v1">ELANA: A Simple Energy and Latency Analyzer for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      The latency and power consumption of large language models (LLMs) are major constraints when serving them across a wide spectrum of hardware platforms, from mobile edge devices to cloud GPU clusters. Benchmarking is crucial for optimizing efficiency in both model deployment and next-generation model development. To address this need, we open-source a simple profiling tool, \textbf{ELANA}, for evaluating LLMs. ELANA is designed as a lightweight, academic-friendly profiler for analyzing model size, key-value (KV) cache size, prefilling latency (Time-to-first-token, TTFT), generation latency (Time-per-output-token, TPOT), and end-to-end latency (Time-to-last-token, TTLT) of LLMs on both multi-GPU and edge GPU platforms. It supports all publicly available models on Hugging Face and offers a simple command-line interface, along with optional energy consumption logging. Moreover, ELANA is fully compatible with popular Hugging Face APIs and can be easily customized or adapted to compressed or low bit-width models, making it ideal for research on efficient LLMs or for small-scale proof-of-concept studies. We release the ELANA profiling tool at: https://github.com/enyac-group/Elana.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20964v2">OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) on the task of generating complete formal specifications for verifying the functional correctness of operating system kernels. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each of which is a long-context task of about 20k-30k tokens. The benchmark formulates the specification generation task as a program synthesis problem confined to a domain for specifying states and transitions. This formulation is provided to LLMs through a programming model. The LLMs must be able to understand the programming model and verification assumptions before delineating the correct search space for syntax and semantics and generating formal specifications. Guided by the operating system's high-level functional description, the LLMs are asked to generate a specification that fully describes all correct states and transitions for a potentially buggy code implementation of the operating system. Experimental results with 12 state-of-the-art LLMs indicate limited performance of existing LLMs on the specification generation task for operating system verification. Significant disparities in their performance highlight differences in their ability to handle long-context code generation tasks. The code are available at https://github.com/lishangyu-hkust/OSVBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06721v1">ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are emerging to transform daily life. However, existing LLM agents primarily follow a reactive paradigm, relying on explicit user instructions to initiate services, which increases both physical and cognitive workload. In this paper, we propose ProAgent, the first end-to-end proactive agent system that harnesses massive sensory contexts and LLM reasoning to deliver proactive assistance. ProAgent first employs a proactive-oriented context extraction approach with on-demand tiered perception to continuously sense the environment and derive hierarchical contexts that incorporate both sensory and persona cues. ProAgent then adopts a context-aware proactive reasoner to map these contexts to user needs and tool calls, providing proactive assistance. We implement ProAgent on Augmented Reality (AR) glasses with an edge server and extensively evaluate it on a real-world testbed, a public dataset, and through a user study. Results show that ProAgent achieves up to 33.4% higher proactive prediction accuracy, 16.8% higher tool-calling F1 score, and notable improvements in user satisfaction over state-of-the-art baselines, marking a significant step toward proactive assistants. A video demonstration of ProAgent is available at https://youtu.be/pRXZuzvrcVs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.11192v5">I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 The paper has been accepted to EMNLP 2024 (Main Conference) there is a follow up paper: Efficiently Selecting Response Generation Strategies for Synthetic Data Construction by Self-Aligned Perplexity Note: This is a revised version of arXiv:2402.11192 (v1, submitted 17 Feb 2024)
    </div>
    <details class="paper-abstract">
      This paper explores an intriguing observation: fine-tuning a large language model (LLM) with responses generated by a LLM often yields better results than using responses generated by humans, particularly in reasoning tasks. We conduct an in-depth investigation to understand why this occurs. Contrary to the common belief that these instances is due to the more detailed nature of LLM-generated content, our study identifies another contributing factor: an LLM is inherently more "familiar" with LLM generated responses. This familiarity is evidenced by lower perplexity before fine-tuning. We design a series of experiments to understand the impact of the "familiarity" and our conclusion reveals that this "familiarity" significantly impacts learning performance. Training with LLM-generated responses not only enhances performance but also helps maintain the model's capabilities in other reasoning tasks after fine-tuning on a specific task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06659v1">The Evolution of Agentic AI in Cybersecurity: From Single LLM Reasoners to Multi-Agent Systems and Autonomous Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted at ICAIC 2026
    </div>
    <details class="paper-abstract">
      Cybersecurity has become one of the earliest adopters of agentic AI, as security operations centers increasingly rely on multi-step reasoning, tool-driven analysis, and rapid decision-making under pressure. While individual large language models can summarize alerts or interpret unstructured reports, they fall short in real SOC environments that require grounded data access, reproducibility, and accountable workflows. In response, the field has seen a rapid architectural evolution from single-model helpers toward tool-augmented agents, distributed multi-agent systems, schema-bound tool ecosystems, and early explorations of semi-autonomous investigative pipelines. This survey presents a five-generation taxonomy of agentic AI in cybersecurity. It traces how capabilities and risks change as systems advance from text-only LLM reasoners to multi-agent collaboration frameworks and constrained-autonomy pipelines. We compare these generations across core dimensions - reasoning depth, tool use, memory, reproducibility, and safety. In addition, we also synthesize emerging benchmarks used to evaluate cyber-oriented agents. Finally, we outline the unresolved challenges that accompany this evolution, such as response validation, tool-use correctness, multi-agent coordination, long-horizon reasoning, and safeguards for high-impact actions. Collectively, this work provides a structured perspective on how agentic AI is taking shape within cybersecurity and what is required to ensure its safe and reliable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06655v1">GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20561v2">Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as rethinking and error correction, as a form of in-context exploration. However, the Markovian policy obtained from conventional RL training does not give rise to reflective exploration behaviors since the policy depends on the history only through the state and therefore has no incentive to enrich identical states with additional context. Instead, RL exploration is only useful during training to learn the optimal policy in a trial-and-error manner. Therefore, it remains unclear whether reflective reasoning will emerge during RL, or why it is beneficial. To remedy this, we recast reflective exploration within a Bayesian RL framework, which optimizes the expected return under a posterior distribution over Markov decision processes induced by the training data. This Bayesian formulation admits uncertainty-adaptive policies that, through belief updates, naturally incentivize information-gathering actions and induce self-reflection behaviors. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms conventional RL approaches, achieving superior test-time performance and token efficiency. Our code is available at https://github.com/shenao-zhang/BARL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06274v2">Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06607v1">A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06982v1">LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning
    </div>
    <details class="paper-abstract">
      Designing state encoders for reinforcement learning (RL) with multiple information sources -- such as sensor measurements, time-series signals, image observations, and textual instructions -- remains underexplored and often requires manual design. We formalize this challenge as a problem of composite neural architecture search (NAS), where multiple source-specific modules and a fusion module are jointly optimized. Existing NAS methods overlook useful side information from the intermediate outputs of these modules -- such as their representation quality -- limiting sample efficiency in multi-source RL settings. To address this, we propose an LLM-driven NAS pipeline that leverages language-model priors and intermediate-output signals to guide sample-efficient search for high-performing composite state encoders. On a mixed-autonomy traffic control task, our approach discovers higher-performing architectures with fewer candidate evaluations than traditional NAS baselines and the LLM-based GENIUS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03994v2">Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted to the AAAI 2026 Deployable AI (DAI) Workshop
    </div>
    <details class="paper-abstract">
      Aligning proprietary large language models (LLMs) with internal organizational policies has become an urgent priority as organizations increasingly deploy LLMs in sensitive domains such as legal support, finance, and medical services. Beyond generic safety filters, enterprises require reliable mechanisms to detect policy violations within their regulatory and operational frameworks, where breaches can trigger legal and reputational risks. Existing content moderation frameworks, such as guardrails, remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and lack interpretability. To address these limitations, we propose a training-free and efficient method that treats policy violation detection as an out-of-distribution (OOD) detection problem. Inspired by whitening techniques, we apply a linear transformation to decorrelate the model's hidden activations and standardize them to zero mean and unit variance, yielding near-identity covariance matrix. In this transformed space, we use the Euclidean norm as a compliance score to detect policy violations. The method requires only the policy text and a small number of illustrative samples, which makes it light-weight and easily deployable. On a challenging policy benchmark, our approach achieves state-of-the-art results, surpassing both existing guardrails and fine-tuned reasoning models. This work provides organizations with a practical and statistically grounded framework for policy-aware oversight of LLMs, advancing the broader goal of deployable AI governance. Code is available at: https://tinyurl.com/policy-violation-detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16978v2">Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit (ΔScore = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity (ΔScore = 3.4, d_z = 1.86), ranked-choice voting (ΔScore = 2.4, d_z = 1.20), and token penalties (ΔScore = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.
    </details>
</div>
