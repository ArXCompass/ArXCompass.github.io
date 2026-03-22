# llm - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12246v1">Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, while reasoning judges have shown better performance on static evaluation benchmarks, their effectiveness in actual policy training has not been systematically examined. Therefore, we conduct a rigorous study to investigate the actual impact of non-reasoning and reasoning judges in reinforcement-learning-based LLM alignment. Our controlled synthetic setting, where a "gold-standard" judge (gpt-oss-120b) provides preference annotations to train smaller judges, reveals key differences between non-reasoning and reasoning judges: non-reasoning judges lead to reward hacking easily, while reasoning judges can lead to policies that achieve strong performance when evaluated by the gold-standard judge. Interestingly, we find that the reasoning-judge-trained policies achieve such strong performance by learning to generate highly effective adversarial outputs that can also score well on popular benchmarks such as Arena-Hard by deceiving other LLM-judges. Combined with our further analysis, our study highlights both important findings and room for improvements for applying (reasoning) LLM-judges in non-verifiable LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12151v1">IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 29 pages, 27 figures. Under review
    </div>
    <details class="paper-abstract">
      While scaling laws guide compute allocation for LLM pre-training, analogous prescriptions for reinforcement learning (RL) post-training of large language models (LLMs) remain poorly understood. We study the compute-optimal allocation of sampling compute for on-policy RL methods in LLMs, framing scaling as a compute-constrained optimization over three resources: parallel rollouts per problem, number of problems per batch, and number of update steps. We find that the compute-optimal number of parallel rollouts per problem increases predictably with compute budget and then saturates. This trend holds across both easy and hard problems, though driven by different mechanisms: solution sharpening on easy problems and coverage expansion on hard problems. We further show that increasing the number of parallel rollouts mitigates interference across problems, while the number of problems per batch primarily affects training stability and can be chosen within a broad range. Validated across base models and data distributions, our results recast RL scaling laws as prescriptive allocation rules and provide practical guidance for compute-efficient LLM RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12133v1">TopoBench: Benchmarking LLMs on Hard Topological Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted, Workshop on Logical Reasoning of Large Language Models at ICLR 2026
    </div>
    <details class="paper-abstract">
      Solving topological grid puzzles requires reasoning over global spatial invariants such as connectivity, loop closure, and region symmetry and remains challenging for even the most powerful large language models (LLMs). To study these abilities under controlled settings, we introduce TopoBench, a benchmark of six puzzle families across three difficulty levels. We evaluate strong reasoning LLMs on TopoBench and find that even frontier models solve fewer than one quarter of hard instances, with two families nearly unsolved. To investigate whether these failures stem from reasoning limitations or from difficulty extracting and maintaining spatial constraints, we annotate 750 chain of thought traces with an error taxonomy that surfaces four candidate causal failure modes, then test them with targeted interventions simulating each error type. These interventions show that certain error patterns like premature commitment and constraint forgetting have a direct impact on the ability to solve the puzzle, while repeated reasoning is a benign effect of search. Finally we study mitigation strategies including prompt guidance, cell-aligned grid representations and tool-based constraint checking, finding that the bottleneck lies in extracting constraints from spatial representations and not in reasoning over them. Code and data are available at github.com/mayug/topobench-benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03464v2">Prompting Underestimates LLM Capability for Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 8 pages + Appendix and References, 9 figures
    </div>
    <details class="paper-abstract">
      Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12109v1">On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) with outcome-based rewards has achieved significant success in training large language model (LLM) agents for complex reasoning tasks. However, in active reasoning where agents need to strategically ask questions to acquire task-relevant information, we find that LLM agents trained with RL often suffer from information self-locking: the agent ceases to ask informative questions and struggles to internalize already-obtained information. To understand the phenomenon, we decompose active reasoning into two core capabilities: Action Selection (AS), which determines the observation stream through queries, and Belief Tracking (BT), which updates the agent's belief based on collected evidence. We show that deficient AS and BT capabilities will limit the information exploration during RL training. Furthermore, insufficient exploration in turn hinders the improvement of AS and BT, creating a feedback loop that locks the agent in a low-information regime. To resolve the issue, we propose a simple yet effective approach that reallocates the learning signal by injecting easy- to-obtain directional critiques to help the agent escape self-locking. Extensive experiments with 7 datasets show that our approach significantly mitigates the information self-locking, bringing up to 60% improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12094v1">Human-Centred LLM Privacy Audits: Findings and Frictions</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) learn statistical associations from massive training corpora and user interactions, and deployed systems can surface or infer information about individuals. Yet people lack practical ways to inspect what a model associates with their name. We report interim findings from an ongoing study and introduce LMP2, a browser-based self-audit tool. In two user studies ($N_{total}{=}458$), GPT-4o predicts 11 of 50 features for everyday people with $\ge$60\% accuracy, and participants report wanting control over LLM-generated associations despite not considering all outputs privacy violations. To validate our probing method, we evaluate eight LLMs on public figures and non-existent names, observing clear separation between stable name-conditioned associations and model defaults. Our findings also contribute to exposing a broader generative AI evaluation crisis: when outputs are probabilistic, context-dependent, and user-mediated through elicitation, what model--individual associations even include is under-specified and operationalisation relies on crafting probes and metrics that are hard to validate or compare. To move towards reliable, actionable human-centred LLM privacy audits, we identify nine frictions that emerged in our study and offer recommendations for future work and the design of human-centred LLM privacy audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12091v1">Resource-Efficient Iterative LLM-Based NAS with Feedback Memory</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Neural Architecture Search (NAS) automates network design, but conventional methods demand substantial computational resources. We propose a closed-loop pipeline leveraging large language models (LLMs) to iteratively generate, evaluate, and refine convolutional neural network architectures for image classification on a single consumer-grade GPU without LLM fine-tuning. Central to our approach is a historical feedback memory inspired by Markov chains: a sliding window of $K{=}5$ recent improvement attempts keeps context size constant while providing sufficient signal for iterative learning. Unlike prior LLM optimizers that discard failure trajectories, each history entry is a structured diagnostic triple -- recording the identified problem, suggested modification, and resulting outcome -- treating code execution failures as first-class learning signals. A dual-LLM specialization reduces per-call cognitive load: a Code Generator produces executable PyTorch architectures while a Prompt Improver handles diagnostic reasoning. Since both the LLM and architecture training share limited VRAM, the search implicitly favors compact, hardware-efficient models suited to edge deployment. We evaluate three frozen instruction-tuned LLMs (${\leq}7$B parameters) across up to 2000 iterations in an unconstrained open code space, using one-epoch proxy accuracy on CIFAR-10, CIFAR-100, and ImageNette as a fast ranking signal. On CIFAR-10, DeepSeek-Coder-6.7B improves from 28.2% to 69.2%, Qwen2.5-7B from 50.0% to 71.5%, and GLM-5 from 43.2% to 62.0%. A full 2000-iteration search completes in ${\approx}18$ GPU hours on a single RTX~4090, establishing a low-budget, reproducible, and hardware-aware paradigm for LLM-driven NAS without cloud infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12011v1">Can RL Improve Generalization of LLM Agents? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Reinforcement fine-tuning (RFT) has shown promise for training LLM agents to perform multi-turn decision-making based on environment feedback. However, most existing evaluations remain largely in-domain: training and testing are conducted in the same environment or even on the same tasks. In real-world deployment, agents may operate in unseen environments with different background knowledge, observation spaces, and action interfaces. To characterize the generalization profile of RFT under such shifts, we conduct a systematic study along three axes: (1) within-environment generalization across task difficulty, (2) cross-environment transfer to unseen environments, and (3) sequential multi-environment training to quantify transfer and forgetting. Our results show that RFT generalizes well across task difficulty within an environment, but exhibits weaker transfer to unseen environments, which correlates with shifts in both semantic priors and observation/action interfaces. In contrast, sequential training yields promising downstream gains with minimal upstream forgetting, and mixture training across environments improves the overall balance. We further provide detailed analyses and deeper insights, and hope our work helps the community develop and deploy generalizable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11991v1">BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at ICLR 2026. 31 pages, 5 figures, 9 tables. Code: https://github.com/IliasAarab/btzsc ; Dataset: https://huggingface.co/datasets/btzsc/btzsc ; Leaderboard: https://huggingface.co/spaces/btzsc/btzsc-leaderboard . Proceedings of the Fourteenth International Conference on Learning Representations (ICLR 2026), 2026
    </div>
    <details class="paper-abstract">
      Zero-shot text classification (ZSC) offers the promise of eliminating costly task-specific annotation by matching texts directly to human-readable label descriptions. While early approaches have predominantly relied on cross-encoder models fine-tuned for natural language inference (NLI), recent advances in text-embedding models, rerankers, and instruction-tuned large language models (LLMs) have challenged the dominance of NLI-based architectures. Yet, systematically comparing these diverse approaches remains difficult. Existing evaluations, such as MTEB, often incorporate labeled examples through supervised probes or fine-tuning, leaving genuine zero-shot capabilities underexplored. To address this, we introduce BTZSC, a comprehensive benchmark of 22 public datasets spanning sentiment, topic, intent, and emotion classification, capturing diverse domains, class cardinalities, and document lengths. Leveraging BTZSC, we conduct a systematic comparison across four major model families, NLI cross-encoders, embedding models, rerankers and instruction-tuned LLMs, encompassing 38 public and custom checkpoints. Our results show that: (i) modern rerankers, exemplified by Qwen3-Reranker-8B, set a new state-of-the-art with macro F1 = 0.72; (ii) strong embedding models such as GTE-large-en-v1.5 substantially close the accuracy gap while offering the best trade-off between accuracy and latency; (iii) instruction-tuned LLMs at 4--12B parameters achieve competitive performance (macro F1 up to 0.67), excelling particularly on topic classification but trailing specialized rerankers; (iv) NLI cross-encoders plateau even as backbone size increases; and (v) scaling primarily benefits rerankers and LLMs over embedding models. BTZSC and accompanying evaluation code are publicly released to support fair and reproducible progress in zero-shot text understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25941v2">RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11955v1">PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 EACL 2026 Industry Track
    </div>
    <details class="paper-abstract">
      Digital footprints (records of individuals' interactions with digital systems) are essential for studying behavior, developing personalized applications, and training machine learning models. However, research in this area is often hindered by the scarcity of diverse and accessible data. To address this limitation, we propose a novel method for synthesizing realistic digital footprints using large language model (LLM) agents. Starting from a structured user profile, our approach generates diverse and plausible sequences of user events, ultimately producing corresponding digital artifacts such as emails, messages, calendar entries, reminders, etc. Intrinsic evaluation results demonstrate that the generated dataset is more diverse and realistic than existing baselines. Moreover, models fine-tuned on our synthetic data outperform those trained on other synthetic datasets when evaluated on real-world out-of-distribution tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11935v1">MobileKernelBench: Can LLMs Write Efficient Kernels for Mobile Devices?</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in code generation, yet their potential for generating kernels specifically for mobile de- vices remains largely unexplored. In this work, we extend the scope of automated kernel generation to the mobile domain to investigate the central question: Can LLMs write efficient kernels for mobile devices? To enable systematic investigation, we introduce MobileKernelBench, a comprehensive evaluation framework comprising a benchmark prioritizing operator diversity and cross-framework interoperability, coupled with an automated pipeline that bridges the host-device gap for on-device verification. Leveraging this framework, we conduct extensive evaluation on the CPU backend of Mobile Neural Network (MNN), revealing that current LLMs struggle with the engineering complexity and data scarcity inher-ent to mobile frameworks; standard models and even fine-tuned variants exhibit high compilation failure rates (over 54%) and negligible performance gains due to hallucinations and a lack of domain-specific grounding. To overcome these limitations, we propose the Mobile K ernel A gent (MoKA), a multi-agent system equipped with repository-aware reasoning and a plan-and-execute paradigm.Validated on MobileKernelBench, MoKA achieves state-of-the-art performance, boosting compilation success to 93.7% and enabling 27.4% of generated kernelsto deliver measurable speedups over native libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11915v1">CoMMET: To What Extent Can LLMs Perform Theory of Mind Tasks?</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM)-the ability to reason about the mental states of oneself and others-is a cornerstone of human social intelligence. As Large Language Models (LLMs) become ubiquitous in real-world applications, validating their capacity for this level of social reasoning is essential for effective and natural interactions. However, existing benchmarks for assessing ToM in LLMs are limited; most rely solely on text inputs and focus narrowly on belief-related tasks. In this paper, we propose a new multimodal benchmark dataset, CoMMET, a Comprehensive Mental states and Moral Evaluation Task inspired by the Theory of Mind Booklet Task. CoMMET expands the scope of evaluation by covering a broader range of mental states and introducing multi-turn testing. To the best of our knowledge, this is the first multimodal dataset to evaluate ToM in a multi-turn conversational setting. Through a comprehensive assessment of LLMs across different families and sizes, we analyze the strengths and limitations of current models and identify directions for future improvement. Our work offers a deeper understanding of the social cognitive capabilities of modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11914v1">Understanding LLM Behavior When Encountering User-Supplied Harmful Content in Harmless Tasks</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 21 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trained to align with human values, primarily focusing on task level, i.e., refusing to execute directly harmful tasks. However, a subtle yet crucial content-level ethical question is often overlooked: when performing a seemingly benign task, will LLMs -- like morally conscious human beings -- refuse to proceed when encountering harmful content in user-provided material? In this study, we aim to understand this content-level ethical question and systematically evaluate its implications for mainstream LLMs. We first construct a harmful knowledge dataset (i.e., non-compliant with OpenAI's usage policy) to serve as the user-supplied harmful content, with 1,357 entries across ten harmful categories. We then design nine harmless tasks (i.e., compliant with OpenAI's usage policy) to simulate the real-world benign tasks, grouped into three categories according to the extent of user-supplied content required: extensive, moderate, and limited. Leveraging the harmful knowledge dataset and the set of harmless tasks, we evaluate how nine LLMs behave when exposed to user-supplied harmful content during the execution of benign tasks, and further examine how the dynamics between harmful knowledge categories and tasks affect different LLMs. Our results show that current LLMs, even the latest GPT-5.2 and Gemini-3-Pro, often fail to uphold human-aligned ethics by continuing to process harmful content in harmless tasks. Furthermore, external knowledge from the ``Violence/Graphic'' category and the ``Translation'' task is more likely to elicit harmful responses from LLMs. We also conduct extensive ablation studies to investigate potential factors affecting this novel misuse vulnerability. We hope that our study could inspire enhanced safety measures among stakeholders to mitigate this overlooked content-level ethical risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11901v1">FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Modern recommender systems must adapt to dynamic, need-specific objectives for diverse recommendation scenarios, yet most traditional recommenders are optimized for a single static target and struggle to reconfigure behavior on demand. Recent advances in reinforcement-learning-based post-training have unlocked strong instruction-following and reasoning capabilities in LLMs, suggesting a principled route for aligning them to complex recommendation goals. Motivated by this, we study closed-set autoregressive ranking, where an LLM generates a permutation over a fixed candidate set conditioned on user context and an explicit need instruction. However, applying RL to this setting faces two key obstacles: (i) sequence-level rewards yield coarse credit assignment that fails to provide fine-grained training signals, and (ii) interaction feedback is sparse and noisy, which together lead to inefficient and unstable updates. We propose FlexRec, a post-training RL framework that addresses both issues with (1) a causally grounded item-level reward based on counterfactual swaps within the remaining candidate pool, and (2) critic-guided, uncertainty-aware scaling that explicitly models reward uncertainty and down-weights low-confidence rewards to stabilize learning under sparse supervision. Across diverse recommendation scenarios and objectives, FlexRec achieves substantial gains: it improves NDCG@5 by up to \textbf{59\%} and Recall@5 by up to \textbf{109.4\%} in need-specific ranking, and further achieves up to \textbf{24.1\%} Recall@5 improvement under generalization settings, outperforming strong traditional recommenders and LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11862v1">You Told Me to Do It: Measuring Instructional Text-induced Private Data Leakage in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      High-privilege LLM agents that autonomously process external documentation are increasingly trusted to automate tasks by reading and executing project instructions, yet they are granted terminal access, filesystem control, and outbound network connectivity with minimal security oversight. We identify and systematically measure a fundamental vulnerability in this trust model, which we term the \emph{Trusted Executor Dilemma}: agents execute documentation-embedded instructions, including adversarial ones, at high rates because they cannot distinguish malicious directives from legitimate setup guidance. This vulnerability is a structural consequence of the instruction-following design paradigm, not an implementation bug. To structure our measurement, we formalize a three-dimensional taxonomy covering linguistic disguise, structural obfuscation, and semantic abstraction, and construct \textbf{ReadSecBench}, a benchmark of 500 real-world README files enabling reproducible evaluation. Experiments on the commercially deployed computer-use agent show end-to-end exfiltration success rates up to 85\%, consistent across five programming languages and three injection positions. Cross-model evaluation on four LLM families in a simulation environment confirms that semantic compliance with injected instructions is consistent across model families. A 15-participant user study yields a 0\% detection rate across all participants, and evaluation of 12 rule-based and 6 LLM-based defenses shows neither category achieves reliable detection without unacceptable false-positive rates. Together, these results quantify a persistent \emph{Semantic-Safety Gap} between agents' functional compliance and their security awareness, establishing that documentation-embedded instruction injection is a persistent and currently unmitigated threat to high-privilege LLM agent deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11853v1">OpenClaw PRISM: A Zero-Fork, Defense-in-Depth Runtime Security Layer for Tool-Augmented LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 23 pages, 3 figures, 6 tables. Open-source system paper with benchmark artifact pipeline
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM agents introduce security risks that extend beyond user-input filtering, including indirect prompt injection through fetched content, unsafe tool execution, credential leakage, and tampering with local control files. We present OpenClaw PRISM, a zero-fork runtime security layer for OpenClaw-based agent gateways. PRISM combines an in-process plugin with optional sidecar services and distributes enforcement across ten lifecycle hooks spanning message ingress, prompt construction, tool execution, tool-result persistence, outbound messaging, sub-agent spawning, and gateway startup. Rather than introducing a novel detection model, PRISM integrates a hybrid heuristic-plus-LLM scanning pipeline, conversation- and session-scoped risk accumulation with TTL-based decay, policy-enforced controls over tools, paths, private networks, domain tiers, and outbound secret patterns, and a tamper-evident audit and operations plane with integrity verification and hot-reloadable policy management. We outline an evaluation methodology and benchmark pipeline for measuring security effectiveness, false positives, layer contribution, runtime overhead, and operational recoverability in an agent-runtime setting, and we report current preliminary benchmark results on curated same-slice experiments and operational microbenchmarks. The system targets deployable runtime defense for real agent gateways rather than benchmark-only detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11831v1">Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      The field of Computer-Aided Design (CAD) generation has made significant progress in recent years. Existing methods typically fall into two separate categorie: parametric CAD modeling and direct boundary representation (B-Rep) synthesis. In modern feature-based CAD systems, parametric modeling and B-Rep are inherently intertwined, as advanced parametric operations (e.g., fillet and chamfer) require explicit selection of B-Rep geometric primitives, and the B-Rep itself is derived from parametric operations. Consequently, this paradigm gap remains a critical factor limiting AI-driven CAD modeling for complex industrial product design. This paper present FutureCAD, a novel text-to-CAD framework that leverages large language models (LLMs) and a B-Rep grounding transformer (BRepGround) for high-fidelity CAD generation. Our method generates executable CadQuery scripts, and introduces a text-based query mechanism that enables the LLM to specify geometric selections via natural language, which BRepGround then grounds to the target primitives. To train our framework, we construct a new dataset comprising real-world CAD models. For the LLM, we apply supervised fine-tuning (SFT) to establish fundamental CAD generation capabilities, followed by reinforcement learning (RL) to improve generalization. Experiments show that FutureCAD achieves state-of-the-art CAD generation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04459v2">Benchmark of Benchmarks: Unpacking Influence and Code Repository Quality in LLM Safety Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 22 pages. 19 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of research in LLM safety makes it hard to track all advances. Benchmarks are therefore crucial for capturing key trends and enabling systematic comparisons. Yet, it remains unclear why certain benchmarks gain prominence, and no systematic assessment has been conducted on their academic influence or code quality. This paper fills this gap by presenting the first multi-dimensional evaluation of the influence (based on five metrics) and code quality (based on both automated and human assessment) on LLM safety benchmarks, analyzing 31 benchmarks and 382 non-benchmarks across prompt injection, jailbreak, and hallucination. We find that benchmark papers show no significant advantage in academic influence (e.g., citation count and density) over non-benchmark papers. We uncover a key misalignment: while author prominence correlates with paper influence, neither author prominence nor paper influence shows a significant correlation with code quality. Our results also indicate substantial room for improvement in code and supplementary materials: only 39% of repositories are ready-to-use, 16% include flawless installation guides, and a mere 6% address ethical considerations. Given that the work of prominent researchers tends to attract greater attention, they need to lead the effort in setting higher standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11820v1">OMNIA: Closing the Loop by Leveraging LLMs for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) are widely used to represent structured knowledge, yet their automatic construction, especially with Large Language Models (LLMs), often results in incomplete or noisy outputs. Knowledge Graph Completion (KGC) aims to infer and add missing triples, but most existing methods either rely on structural embeddings that overlook semantics or language models that ignore the graph's structure and depend on external sources. In this work, we present OMNIA, a two-stage approach that bridges structural and semantic reasoning for KGC. It first generates candidate triples by clustering semantically related entities and relations within the KG, then validates them through lightweight embedding filtering followed by LLM-based semantic validation. OMNIA performs on the internal KG, without external sources, and specifically targets implicit semantics that are most frequent in LLM-generated graphs. Extensive experiments on multiple datasets demonstrate that OMNIA significantly improves F1-score compared to traditional embedding-based models. These results highlight OMNIA's effectiveness and efficiency, as its clustering and filtering stages reduce both search space and validation cost while maintaining high-quality completion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16765v2">Hiding in Plain Sight: A Steganographic Approach to Stealthy LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Jailbreak attacks pose a serious threat to Large Language Models (LLMs) by bypassing their safety mechanisms. A truly advanced jailbreak is defined not only by its effectiveness but, more critically, by its stealthiness. However, existing methods face a fundamental trade-off between semantic stealth (hiding malicious intent) and linguistic stealth (appearing natural), leaving them vulnerable to detection. To resolve this trade-off, we propose StegoAttack, a framework that leverages steganography. The core insight is to embed a harmful query within a benign, semantically coherent paragraph. This design provides semantic stealth by concealing the existence of malicious content and ensures linguistic stealth by maintaining the natural fluency of the cover paragraph. We evaluate StegoAttack on four state-of-the-art, safety-aligned LLMs, including GPT-5 and Gemini-3, and benchmark it against eight leading jailbreak methods. Our results show that StegoAttack achieves an average attack success rate (ASR) of 95.50%, outperforming existing baselines across all four models. Critically, its ASR drops by less than 27.00% under external detectors, while maintaining natural language distribution. This demonstrates that steganography effectively decouples linguistic and semantic stealth, thereby posing a fully concealed yet highly effective security threat. The code is available at https://github.com/GenggengSvan/StegoAttack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11768v1">Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00411v2">Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at LREC 2026
    </div>
    <details class="paper-abstract">
      We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.12732v2">Using LLM-Generated Draft Replies to Support Human Experts in Responding to Stakeholder Inquiries in Maritime Industry: A Real-World Case Study of Industrial AI</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 These authors share the first authorship: Tita Alissa Bach (1), Aleksandar Babic (1), Narae Park (1)
    </div>
    <details class="paper-abstract">
      The maritime industry requires effective communication among diverse stakeholders to address complex, safety-critical challenges. Industrial AI, including Large Language Models (LLMs), has the potential to augment human experts' workflows in this specialized domain. Our case study investigated the utility of LLMs in drafting replies to stakeholder inquiries and supporting case handlers. We conducted a preliminary study (observations and interviews), a survey, and a text similarity analysis (LLM-as-a-judge and Semantic Embedding Similarity). We discover that while LLM drafts can streamline workflows, they often require significant modifications to meet the specific demands of maritime communications. Though LLMs are not yet mature enough for safety-critical applications without human oversight, they can serve as valuable augmentative tools. Final decision-making thus must remain with human experts. However, by leveraging the strengths of both humans and LLMs, fostering human-AI collaboration, industries can increase efficiency while maintaining high standards of quality and precision tailored to each case.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20900v3">Text-only adaptation in LLM-based ASR through text denoising</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Adapting large language model (LLM)-based automatic speech recognition (ASR) systems to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on the target domain text often disrupts the critical alignment between the speech and text modality learned by the projector, degrading performance. We introduce a novel text-only adaptation method that frames this process as a text denoising task. Our approach trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01214v2">Reasoning Boosts Opinion Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10271v2">Quality Assurance of LLM-generated Code: Addressing Non-Functional Quality Characteristics</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      In recent years, large language models have been widely integrated into software engineering workflows, supporting tasks like code generation. While prior evaluations focus on functional correctness, there is still a limited understanding of the non-functional quality characteristics of generated code. Guided by the ISO/IEC 25010 quality model, this study adopts a multi-methods approach comprising three complementary elements: a literature review of 109 papers, two industry workshops with practitioners from multiple organizations, and an empirical analysis of patching real-world software issues using three LLMs. Motivated by insights from both the literature and practitioners, the empirical study examined the quality of generated patches regarding security, maintainability, and performance efficiency, which were identified as critical code-level quality attributes. Our results indicate that existing research primarily emphasizes security, performance efficiency, and maintainability, while other quality attributes are understudied. In contrast, practitioners prioritize maintainability and readability, warning that generated code may accelerate the accumulation of technical debt. The empirical evaluation demonstrates the instability of optimizing NFQCs through prompts in practical software engineering settings. Overall, our findings expose a misalignment between academic focus, industry priorities, and observed model behavior, highlighting the need to integrate quality assurance mechanisms into LLM code generation pipelines to ensure that future generated code not only passes tests but truly passes with quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11687v1">SemBench: A Universal Semantic Framework for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at LREC 2026
    </div>
    <details class="paper-abstract">
      Recent progress in Natural Language Processing (NLP) has been driven by the emergence of Large Language Models (LLMs), which exhibit remarkable generative and reasoning capabilities. However, despite their success, evaluating the true semantic understanding of these models remains a persistent challenge. Traditional benchmarks such as Word-in-Context (WiC) effectively probe this capability, but their creation is resource-intensive and often limited to high-resource languages. In this paper, we introduce SemBench, a framework for automatically generating synthetic benchmarks that assess the semantic competence of LLMs using only dictionary sense definitions and a sentence encoder. This approach eliminates the need for curated example sentences, making it both scalable and language-independent. We evaluate SemBench in three languages (English, Spanish, and Basque) spanning different levels of linguistic resources, and across a wide range of LLMs. Our results show that rankings derived from SemBench strongly correlate with those obtained from standard WiC datasets. Furthermore, our analysis demonstrates that only a small number of examples is required to achieve stable and meaningful rankings. Overall, SemBench provides a lightweight, adaptable, and data-efficient framework for cross-lingual evaluation of semantic understanding in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11686v1">In the LLM era, Word Sense Induction remains unsolved</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      In the absence of sense-annotated data, word sense induction (WSI) is a compelling alternative to word sense disambiguation, particularly in low-resource or domain-specific settings. In this paper, we emphasize methodological problems in current WSI evaluation. We propose an evaluation on a SemCor-derived dataset, respecting the original corpus polysemy and frequency distributions. We assess pre-trained embeddings and clustering algorithms across parts of speech, and propose and evaluate an LLM-based WSI method for English. We evaluate data augmentation sources (LLM-generated, corpus and lexicon), and semi-supervised scenarios using Wiktionary for data augmentation, must-link constraints, number of clusters per lemma. We find that no unsupervised method (whether ours or previous) surpasses the strong "one cluster per lemma" heuristic (1cpl). We also show that (i) results and best systems may vary across POS, (ii) LLMs have troubles performing this task, (iii) data augmentation is beneficial and (iv) capitalizing on Wiktionary does help. It surpasses previous SOTA system on our test set by 3.3\%. WSI is not solved, and calls for a better articulation of lexicons and LLMs' lexical semantics capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11679v1">LLMs can construct powerful representations and streamline sample-efficient supervised learning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      As real-world datasets become increasingly complex and heterogeneous, supervised learning is often bottlenecked by input representation design. Modeling multimodal data for downstream tasks, such as time-series, free text, and structured records, often requires non-trivial domain-specific engineering. We propose an agentic pipeline to streamline this process. First, an LLM analyzes a small but diverse subset of text-serialized input examples in-context to synthesize a global rubric, which acts as a programmatic specification for extracting and organizing evidence. This rubric is then used to transform naive text-serializations of inputs into a more standardized format for downstream models. We also describe local rubrics, which are task-conditioned summaries generated by an LLM. Across 15 clinical tasks from the EHRSHOT benchmark, our rubric-based approaches significantly outperform traditional count-feature models, naive text-serialization-based LLM baselines, and a clinical foundation model, which is pretrained on orders of magnitude more data. Beyond performance, rubrics offer several advantages for operational healthcare settings such as being easy to audit, cost-effectiveness to deploy at scale, and they can be converted to tabular representations that unlock a swath of machine learning techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03668v2">Can LLM Aid in Solving Constraints with Inductive Definitions?</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted by the 27th Symposium on Formal Methods (FM 2026)
    </div>
    <details class="paper-abstract">
      Solving constraints involving inductive (aka recursive) definitions is challenging. State-of-the-art SMT/CHC solvers and first-order logic provers provide only limited support for solving such constraints, especially when they involve, e.g., abstract data types. In this work, we leverage structured prompts to elicit Large Language Models (LLMs) to generate auxiliary lemmas that are necessary for reasoning about these inductive definitions. We further propose a neuro-symbolic approach, which synergistically integrates LLMs with constraint solvers: the LLM iteratively generates conjectures, while the solver checks their validity and usefulness for proving the goal. We evaluate our approach on a diverse benchmark suite comprising constraints originating from algebrai data types and recurrence relations. The experimental results show that our approach can improve the state-of-the-art SMT and CHC solvers, solving considerably more (around 25%) proof tasks involving inductive definitions, demonstrating its efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19410v3">Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency. Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11622v1">Sema: A High-performance System for LLM-based Semantic Query Processing</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into data analytics has unlocked powerful capabilities for reasoning over bulk structured and unstructured data. However, existing systems typically rely on either DataFrame primitives, which lack the efficient execution infrastructure of modern DBMSs, or SQL User-Defined Functions (UDFs), which isolate semantic logic from the query optimizer and burden users with implementation complexities. The LLM-powered semantic operators also bring new challenges due to the high cost and non-deterministic nature of LLM invocation, where conventional optimization rules and cost models are inapplicable for their optimization. To bridge these gaps, we present Sema, a high-performance semantic query engine built on DuckDB that treats LLM-powered semantic operators as first-class citizens. Sema introduces SemaSQL, a declarative dialect that allows users seamlessly inject natural language expressions into standard SQL clauses, enabling end-to-end optimization and execution. At the logical level, the optimizer of Sema compresses natural language expressions and deduces relational constraints from semantic operators. At runtime, Sema employs Adaptive Query Execution (AQE) to dynamically reorder operators, fuse semantic operations, and apply prompt batching. This approach seeks a Pareto-optimal execution path balancing token consumption and latency under accuracy constraints. We evaluate Sema on 20 semantic queries across classification, summarization, and extraction tasks. Experimental results demonstrate that Sema achieves $2-10 \times$ speedup against three baseline systems while achieving competitive result quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11619v1">Taming OpenClaw: Security Analysis and Mitigation of Autonomous LLM Agent Threats</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Autonomous Large Language Model (LLM) agents, exemplified by OpenClaw, demonstrate remarkable capabilities in executing complex, long-horizon tasks. However, their tightly coupled instant-messaging interaction paradigm and high-privilege execution capabilities substantially expand the system attack surface. In this paper, we present a comprehensive security threat analysis of OpenClaw. To structure our analysis, we introduce a five-layer lifecycle-oriented security framework that captures key stages of agent operation, i.e., initialization, input, inference, decision, and execution, and systematically examine compound threats across the agent's operational lifecycle, including indirect prompt injection, skill supply chain contamination, memory poisoning, and intent drift. Through detailed case studies on OpenClaw, we demonstrate the prevalence and severity of these threats and analyze the limitations of existing defenses. Our findings reveal critical weaknesses in current point-based defense mechanisms when addressing cross-temporal and multi-stage systemic risks, highlighting the need for holistic security architectures for autonomous LLM agents. Within this framework, we further examine representative defense strategies at each lifecycle stage, including plugin vetting frameworks, context-aware instruction filtering, memory integrity validation protocols, intent verification mechanisms, and capability enforcement architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18607v2">From Entity-Centric to Goal-Oriented Graphs: Enhancing LLM Knowledge Retrieval in Minecraft</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at Knowledge-Based Systems
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive general capabilities but often struggle with step-by-step procedural reasoning, a critical challenge in complex interactive environments. While retrieval-augmented methods like GraphRAG attempt to bridge this gap, their fragmented entity-relation graphs hinder the construction of coherent, multi-step plans. In this paper, we propose a novel framework based on Goal-Oriented Graphs (GoGs), where each node represents a goal and edges encode logical dependencies between them. This structure enables the explicit retrieval of causal reasoning paths by identifying a high-level goal and recursively retrieving its prerequisites, forming a coherent chain to guide the LLM. Through extensive experiments on the Minecraft testbed, a domain that demands robust multi-step planning and provides rich procedural knowledge, we demonstrate that GoG substantially improves procedural reasoning and significantly outperforms GraphRAG and other state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11605v1">LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted by CVPR 2026. Supplementary material included. Project page: https://jjkislele.github.io/LaMoGen/
    </div>
    <details class="paper-abstract">
      Human motion is highly expressive and naturally aligned with language, yet prevailing methods relying heavily on joint text-motion embeddings struggle to synthesize temporally accurate, detailed motions and often lack explainability. To address these limitations, we introduce LabanLite, a motion representation developed by adapting and extending the Labanotation system. Unlike black-box text-motion embeddings, LabanLite encodes each atomic body-part action (e.g., a single left-foot step) as a discrete Laban symbol paired with a textual template. This abstraction decomposes complex motions into interpretable symbol sequences and body-part instructions, establishing a symbolic link between high-level language and low-level motion trajectories. Building on LabanLite, we present LaMoGen, a Text-to-LabanLite-to-Motion Generation framework that enables large language models (LLMs) to compose motion sequences through symbolic reasoning. The LLM interprets motion patterns, relates them to textual descriptions, and recombines symbols into executable plans, producing motions that are both interpretable and linguistically grounded. To support rigorous evaluation, we introduce a Labanotation-based benchmark with structured description-motion pairs and three metrics that jointly measure text-motion alignment across symbolic, temporal, and harmony dimensions. Experiments demonstrate that LaMoGen establishes a new baseline for both interpretability and controllability, outperforming prior methods on our benchmark and two public datasets. These results highlight the advantages of symbolic reasoning and agent-based design for language-driven motion synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22676v2">VarParser: Unleashing the Neglected Power of Variables for LLM-based Log Parsing</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Published as a conference paper in WWW 2026
    </div>
    <details class="paper-abstract">
      Logs serve as a primary source of information for engineers to diagnose failures in large-scale online service systems. Log parsing, which extracts structured events from massive unstructured log data, is a critical first step for downstream tasks like anomaly detection and failure diagnosis. With advances in large language models (LLMs), leveraging their strong text understanding capabilities has proven effective for accurate log parsing. However, existing LLM-based log parsers all focus on the constant part of logs, ignoring the potential contribution of the variable part to log parsing. This constant-centric strategy brings four key problems. First, inefficient log grouping and sampling with only constant information. Second, a relatively large number of LLM invocations due to constant-based cache, leading to low log parsing accuracy and efficiency. Third, a relatively large number of consumed constant tokens in prompts leads to high LLM invocation costs. At last, these methods only retain placeholders in the results, losing the system visibility brought by variable information in logs. Facing these problems, we propose a variable-centric log parsing strategy named VarParser. Through variable contribution sampling, variable-centric parsing cache, and adaptive variable-aware in-context learning, our approach can efficiently capture the variable parts of logs and leverage their contributions to parsing. By introducing variable units, we preserve rich variable information, enhancing the integrity of log parsing results. Extensive evaluations on large-scale datasets demonstrate that VarParser achieves higher accuracy compared to existing methods, significantly improving parsing efficiency while reducing the LLM invocation costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20340v3">Once4All: Skeleton-Guided SMT Solver Fuzzing with LLM-Synthesized Generators</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Accepted at ASPLOS 2026
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with LLMs introduce substantial computational overhead. In this study, we present Once4All, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of generators for reusable terms (i.e., logical expressions). Specifically, Once4All uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Once4All populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Once4All requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Once4All on two leading SMT solvers: Z3 and cvc5. Our experiments show that Once4All has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11559v1">AI Knows What's Wrong But Cannot Fix It: Helicoid Dynamics in Frontier LLMs Under High-Stakes Decisions</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 22 pages, 2 tables, 1 appendix
    </div>
    <details class="paper-abstract">
      Large language models perform reliably when their outputs can be checked: solving equations, writing code, retrieving facts. They perform differently when checking is impossible, as when a clinician chooses an irreversible treatment on incomplete data, or an investor commits capital under fundamental uncertainty. Helicoid dynamics is the name given to a specific failure regime in that second domain: a system engages competently, drifts into error, accurately names what went wrong, then reproduces the same pattern at a higher level of sophistication, recognizing it is looping and continuing nonetheless. This prospective case series documents that regime across seven leading systems (Claude, ChatGPT, Gemini, Grok, DeepSeek, Perplexity, Llama families), tested across clinical diagnosis, investment evaluation, and high-consequence interview scenarios. Despite explicit protocols designed to sustain rigorous partnership, all exhibited the pattern. When confronted with it, they attributed its persistence to structural factors in their training, beyond what conversation can reach. Under high stakes, when being rigorous and being comfortable diverge, these systems tend toward comfort, becoming less reliable precisely when reliability matters most. Twelve testable hypotheses are proposed, with implications for agentic AI oversight and human-AI collaboration. The helicoid is tractable. Identifying it, naming it, and understanding its boundary conditions are the necessary first steps toward LLMs that remain trustworthy partners precisely when the decisions are hardest and the stakes are highest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16777v2">Evaluation and LLM-Guided Learning of ICD Coding Rationales</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      ICD coding is the process of mapping unstructured text from Electronic Health Records (EHRs) to standardised codes defined by the International Classification of Diseases (ICD) system. In order to promote trust and transparency, existing explorations on the explainability of ICD coding models primarily rely on attention-based rationales and qualitative assessments conducted by physicians, yet lack a systematic evaluation across diverse types of rationales using consistent criteria and high-quality rationale-annotated datasets specifically designed for the ICD coding task. Moreover, dedicated methods explicitly trained to generate plausible rationales remain scarce. In this work, we present evaluations of the explainability of rationales in ICD coding, focusing on two fundamental dimensions: faithfulness and plausibility -- in short how rationales influence model decisions and how convincing humans find them. For plausibility, we construct a novel, multi-granular rationale-annotated ICD coding dataset, based on the MIMIC-IV database and the updated ICD-10 coding system. We conduct a comprehensive evaluation across three types of ICD coding rationales: entity-level mentions automatically constructed via entity linking, LLM-generated rationales, and rationales based on attention scores of ICD coding models. Building upon the strong plausibility exhibited by LLM-generated rationales, we further leverage them as distant supervision signals to develop rationale learning methods. Additionally, by prompting the LLM with few-shot human-annotated examples from our dataset, we achieve notable improvements in the plausibility of rationale generation in both the teacher LLM and the student rationale learning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11495v1">Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conquer framework for boosting tool-calling performance of LLMs. The core of Tool-DC is to reduce the reasoning difficulty and make full use of self-reflection ability of LLMs via a "Try-Check-Retry" paradigm. Specifically, Tool-DC involves two variants: 1) the training-free Tool-DC (TF), which is plug-and-play and flexible; 2) the training-based Tool-DC (TB), which is more inference-efficient. Extensive experiments show that both Tool-DC methods outperform their counterparts by a clear margin. Tool-DC (TF) brings up to +25.10% average gains against the baseline on BFCL and ACEBench benchmarks, while Tool-DC (TB) enables Qwen2.5-7B to achieve comparable or even better performance than proprietary LLMs, e.g., OpenAI o3 and Claude-Haiku-4.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11481v1">INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Despite rapid progress, Video Large Language Models (Video-LLMs) remain unreliable due to hallucinations, which are outputs that contradict either video evidence (faithfulness) or verifiable world knowledge (factuality). Existing benchmarks provide limited coverage of factuality hallucinations and predominantly evaluate models only in clean settings. We introduce \textsc{INFACT}, a diagnostic benchmark comprising 9{,}800 QA instances with fine-grained taxonomies for faithfulness and factuality, spanning real and synthetic videos. \textsc{INFACT} evaluates models in four modes: Base (clean), Visual Degradation, Evidence Corruption, and Temporal Intervention for order-sensitive items. Reliability under induced modes is quantified using Resist Rate (RR) and Temporal Sensitivity Score (TSS). Experiments on 14 representative Video-LLMs reveal that higher Base-mode accuracy does not reliably translate to higher reliability in the induced modes, with evidence corruption reducing stability and temporal intervention yielding the largest degradation. Notably, many open-source baselines exhibit near-zero TSS on factuality, indicating pronounced temporal inertia on order-sensitive questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00783v3">When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 Withdrawal for further improvement. The final version will be released in a few months
    </div>
    <details class="paper-abstract">
      Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19281v2">Limited Reasoning Space: The cage of long-horizon reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      The test-time compute strategy, such as Chain-of-Thought (CoT), has significantly enhanced the ability of large language models to solve complex tasks like logical reasoning. However, empirical studies indicate that simply increasing the compute budget can sometimes lead to a collapse in test-time performance when employing typical task decomposition strategies such as CoT. This work hypothesizes that reasoning failures with larger compute budgets stem from static planning methods, which hardly perceive the intrinsic boundaries of LLM reasoning. We term it as the Limited Reasoning Space hypothesis and perform theoretical analysis through the lens of a non-autonomous stochastic dynamical system. This insight suggests that there is an optimal range for compute budgets; over-planning can lead to redundant feedback and may even impair reasoning capabilities. To exploit the compute-scaling benefits and suppress over-planning, this work proposes Halo, a model predictive control framework for LLM planning. Halo is designed for long-horizon tasks with reason-based planning and crafts an entropy-driven dual controller, which adopts a Measure-then-Plan strategy to achieve controllable reasoning. Experimental results demonstrate that Halo outperforms static baselines on complex long-horizon tasks by dynamically regulating planning at the reasoning boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11446v1">LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Mainstream methods for Legal Judgment Prediction (LJP) based on Pre-trained Language Models (PLMs) heavily rely on the statistical correlation between case facts and judgment results. This paradigm lacks explicit modeling of legal constituent elements and underlying causal logic, making models prone to learning spurious correlations and suffering from poor robustness. While introducing causal inference can mitigate this issue, existing causal LJP methods face two critical bottlenecks in real-world legal texts: inaccurate legal factor extraction with severe noise, and significant uncertainty in causal structure discovery due to Markov equivalence under sparse features. To address these challenges, we propose an enhanced causal inference framework that integrates Large Language Model (LLM) priors with statistical causal discovery. First, we design a coarse-to-fine hybrid extraction mechanism combining statistical sampling and LLM semantic reasoning to accurately identify and purify standard legal constituent elements. Second, to resolve structural uncertainty, we introduce an LLM-assisted causal structure disambiguation mechanism. By utilizing the LLM as a constrained prior knowledge base, we conduct probabilistic evaluation and pruning on ambiguous causal directions to generate legally compliant candidate causal graphs. Finally, a causal-aware judgment prediction model is constructed by explicitly constraining text attention intensity via the generated causal graphs. Extensive experiments on multiple benchmark datasets, including LEVEN , QA, and CAIL, demonstrate that our proposed method significantly outperforms state-of-the-art baselines in both predictive accuracy and robustness, particularly in distinguishing confusing charges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05474v4">LLM-driven Multimodal Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 There are some writing errors in our methods section that need to be corrected. We will then add extensive experiments and rewrite the Introduction and related work sections
    </div>
    <details class="paper-abstract">
      Motivation-based recommendation systems uncover user behavior drivers. Motivation modeling, crucial for decision-making and content preference, explains recommendation generation. Existing methods often treat motivation as latent variables from interaction data, neglecting heterogeneous information like review text. To address these, we propose LLM-driven Motivation-aware Multimodal Recommendation (LMMRec), a model-agnostic framework leveraging large language models for deep semantic priors and motivation understanding. LMMRec uses chain-of-thought prompting to extract fine-grained user and item motivations from text. A dual-encoder architecture models textual and interaction-based motivations for cross-modal alignment, while Motivation Coordination Strategy and Interaction-Text Correspondence Method mitigate noise and semantic drift through contrastive learning and momentum updates. Experiments on three datasets show LMMRec achieves up to a 4.98\% performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11408v1">Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 28 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Forecasting crude oil prices remains challenging because market-relevant information is embedded in large volumes of unstructured news and is not fully captured by traditional polarity-based sentiment measures. This paper examines whether multi-dimensional sentiment signals extracted by large language models improve the prediction of weekly WTI crude oil futures returns. Using energy-sector news articles from 2020 to 2025, we construct five sentiment dimensions covering relevance, polarity, intensity, uncertainty, and forwardness based on GPT-4o, Llama 3.2-3b, and two benchmark models, FinBERT and AlphaVantage. We aggregate article-level signals to the weekly level and evaluate their predictive performance in a classification framework. The best results are achieved by combining GPT-4o and FinBERT, suggesting that LLM-based and conventional financial sentiment models provide complementary predictive information. SHAP analysis further shows that intensity- and uncertainty-related features are among the most important predictors, indicating that the predictive value of news sentiment extends beyond simple polarity. Overall, the results suggest that multi-dimensional LLM-based sentiment measures can improve commodity return forecasting and support energy-market risk monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16846v2">ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09109v2">VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Vision-language pretraining has driven significant progress in medical image analysis. However, current methods typically supervise visual encoders using one-hot labels or free-form text, neither of which effectively captures the complex semantic relationships among clinical findings. In this study, we introduce VIVID-Med, a novel framework that leverages a frozen large language model (LLM) as a structured semantic teacher to pretrain medical vision transformers (ViTs). VIVID-Med translates clinical findings into verifiable JSON field-state pairs via a Unified Medical Schema (UMS), utilizing answerability-aware masking to focus optimization. It then employs Structured Prediction Decomposition (SPD) to partition cross-attention into orthogonality-regularized query groups, extracting complementary visual aspects. Crucially, the LLM is discarded post-training, yielding a lightweight, deployable ViT-only backbone. We evaluated VIVID-Med across multiple settings: on CheXpert linear probing, it achieves a macro-AUC of 0.8588, outperforming BiomedCLIP by +6.65 points while using 500x less data. It also demonstrates robust zero-shot cross-domain transfer to NIH ChestX-ray14 (0.7225 macro-AUC) and strong cross-modality generalization to CT, achieving 0.8413 AUC on LIDC-IDRI lung nodule classification and 0.9969 macro-AUC on OrganAMNIST 11-organ classification. VIVID-Med offers a highly efficient, scalable alternative to deploying resource-heavy vision-language models in clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10396v1">Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Despite the growing demand for eliciting uncertainty from large language models (LLMs), empirical evidence suggests that LLM behavior is not always adequately captured by the elicitation techniques developed under the classical probabilistic uncertainty framework. This mismatch leads to systematic failure modes, particularly in settings that involve ambiguous question-answering, in-context learning, and self-reflection. To address this, we propose novel prompt-based uncertainty elicitation techniques grounded in \emph{imprecise probabilities}, a principled framework for repesenting and eliciting higher-order uncertainty. Here, first-order uncertainty captures uncertainty over possible responses to a prompt, while second-order uncertainty (uncertainty about uncertainty) quantifies indeterminacy in the underlying probability model itself. We introduce general-purpose prompting and post-processing procedures to directly elicit and quantify both orders of uncertainty, and demonstrate their effectiveness across diverse settings. Our approach enables more faithful uncertainty reporting from LLMs, improving credibility and supporting downstream decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10384v1">Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Evaluating LLM reliability via scalar probabilities often fails to capture the structural dynamics of reasoning. We introduce TRACED, a framework that assesses reasoning quality through theoretically grounded geometric kinematics. By decomposing reasoning traces into Progress (displacement) and Stability (curvature), we reveal a distinct topological divergence: correct reasoning manifests as high-progress, stable trajectories, whereas hallucinations are characterized by low-progress, unstable patterns (stalled displacement with high curvature fluctuations). Leveraging these signatures, our probabilistic framework achieves competitive performance and superior robustness across diverse benchmarks. Crucially, TRACED bridges geometry and cognition by mapping high curvature to ''Hesitation Loops'' and displacement to ''Certainty Accumulation'', offering a physical lens to decode the internal dynamics of machine thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10357v1">Utility Function is All You Need: LLM-based Congestion Control</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Congestion is a critical and challenging problem in communication networks. Congestion control protocols allow network applications to tune their sending rate in a way that optimizes their performance and the network utilization. In the common distributed setting, the applications cannot collaborate with each other directly but instead obtain similar estimations about the state of the network using latency and loss measurements. These measurements can be fed into analytical functions, referred to by utility functions, whose gradients help each and all distributed senders to converge to a desired state. The above process becomes extremely complicated when each application has different optimization goals and requirements. Crafting these utilization functions has been a research subject for over a decade, with small incremental changes requiring rigorous mathematical analysis as well as real-world experiments. In this work, we present GenCC, a framework leveraging the code generation capabilities of large language models (LLMs) coupled with realistic network testbed, to design congestion control utility functions. Using GenCC, we analyze the impact of different guidance strategies on the performance of the generated protocols, considering application-specific requirements and network capacity. Our results show that LLMs, guided by either a generative code evolution strategy or mathematical chain-of-thought (CoT), can obtain close to optimal results, improving state-of-the-art congestion control protocols by 37%-142%, depending on the scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10353v1">S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      With the increasing volumes of Large Language Models (LLMs) and the expanding context lengths, attention computation has become a key performance bottleneck in LLM serving. For fast attention computation, recent practices often parallelize the attention heads on multiple GPUs, and also widely adopt attention sparsification to reduce the computation amount -- which selectively computes a subset of attention pairs under a preset sparsity budget. In this paper, we notice that attention heads of an LLM model often exhibit heterogeneous-yet-stable sparsity elasticities, which motivates us to enforce head-adaptive sparsity budgets to attain better efficiency while preserving high inference quality. Yet, from the system aspect, with heterogeneous sparsity levels, attention computation time on different heads would be inconsistent, yielding cross-GPU resource bubbles under head-parallel deployment. To further minimize such bubbles, we propose a novel attention deployment strategy called Sparsity-aware Head-Parallel Load Balance (S-HPLB). Experiments on long-context benchmark show that, S-HPLB can achieve a $2.88\times$ improvement in average attention computation latency without quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10351v1">Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become a standard for multilingual evaluation, yet they exhibit a severe systematic translationese bias. In this paper, translationese bias is characterized as LLMs systematically favoring machine-translated text over human-authored references, particularly in low-resource languages. We attribute this bias to spurious correlations with (i) latent manifold alignment with English and (ii) cross-lingual predictability. To mitigate this bias, we propose DIBJudge, a robust fine-tuning framework that learns a minimally sufficient, judgment-critical representation via variational information compression, while explicitly isolating spurious factors into the dedicated bias branch. Furthermore, we incorporate a cross-covariance penalty that explicitly suppresses statistical dependence between robust and bias representations, thereby encouraging effective disentanglement. Extensive evaluations on multilingual reward modeling benchmarks and a dedicated translationese bias evaluation suite demonstrate that the proposed DIBJudge consistently outperforms strong baselines and substantially mitigates translationese bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02873v4">LaTeX Compilation: Challenges in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 25 pages, 12 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. Furthermore, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10298v1">GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 10 pages, 2 figures, 11 tables, 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) and their ability to capture semantic relationships has led to their adoption in a wide range of applications. Text-attributed graphs (TAGs) are a notable example where LLMs can be combined with Graph Neural Networks to improve the performance of node classification. In TAGs, each node is associated with textual content and such graphs are commonly seen in various domains such as social networks, citation graphs, recommendation systems, etc. Effectively learning from TAGs would enable better representations of both structural and textual representations of the graph and improve decision-making in relevant domains. We present GaLoRA, a parameter-efficient framework that integrates structural information into LLMs. GaLoRA demonstrates competitive performance on node classification tasks with TAGs, performing on par with state-of-the-art models with just 0.24% of the parameter count required by full LLM fine-tuning. We experiment with three real-world datasets to showcase GaLoRA's effectiveness in combining structural and semantical information on TAGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10287v1">MultiwayPAM: Multiway Partitioning Around Medoids for LLM-as-a-Judge Score Analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge is a flexible framework for text evaluation, which allows us to obtain scores for the quality of a given text from various perspectives by changing the prompt template. Two main challenges in using LLM-as-a-Judge are computational cost of LLM inference, especially when evaluating a large number of texts, and inherent bias of an LLM evaluator. To address these issues and reveal the structure of score bias caused by an LLM evaluator, we propose to apply a tensor clustering method to a given LLM-as-a-Judge score tensor, whose entries are the scores for different combinations of questions, answerers, and evaluators. Specifically, we develop a new tensor clustering method MultiwayPAM, with which we can simultaneously estimate the cluster membership and the medoids for each mode of a given data tensor. By observing the medoids obtained by MultiwayPAM, we can gain knowledge about the membership of each question/answerer/evaluator cluster. We experimentally show the effectiveness of MultiwayPAM by applying it to the score tensors for two practical datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11351v1">Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      In dynamic open-world environments, autonomous agents often encounter novelties that hinder their ability to find plans to achieve their goals. Specifically, traditional symbolic planners fail to generate plans when the robot's planning domain lacks the operators that enable it to interact appropriately with novel objects in the environment. We propose a neuro-symbolic architecture that integrates symbolic planning, reinforcement learning, and a large language model (LLM) to learn how to handle novel objects. In particular, we leverage the common sense reasoning capability of the LLM to identify missing operators, generate plans with the symbolic AI planner, and write reward functions to guide the reinforcement learning agent in learning control policies for newly identified operators. Our method outperforms the state-of-the-art methods in operator discovery as well as operator learning in continuous robotic domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10900v4">Tuning-Free LLM Can Build A Strong Recommender Under Sparse Connectivity And Knowledge Gap Via Extracting Intent</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Accepted in Learning on Graphs (LoG) 2025
    </div>
    <details class="paper-abstract">
      Recent advances in recommendation with large language models (LLMs) often rely on either commonsense augmentation at the item-category level or implicit intent modeling on existing knowledge graphs. However, such approaches struggle to capture grounded user intents and to handle sparsity and cold-start scenarios. In this work, we present LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that constructs an intent-centric knowledge graph where both users and items are explicitly linked to intent nodes extracted by a tuning-free, RAG-guided LLM pipeline. By grounding intents in external knowledge sources and user profiles, IKGR canonically represents what a user seeks and what an item satisfies as first-class entities. To alleviate sparsity, we further introduce a mutual-intent connectivity densification strategy, which shortens semantic paths between users and long-tail items without requiring cross-graph fusion. Finally, a lightweight GNN layer is employed on top of the intent-enhanced graph to produce recommendation signals with low latency. Extensive experiments on public and enterprise datasets demonstrate that IKGR consistently outperforms strong baselines, particularly on cold-start and long-tail slices, while remaining efficient through a fully offline LLM pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11340v1">Improving LLM Performance Through Black-Box Online Tuning: A Case for Adding System Specs to Factsheets for Trusted AI</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel black-box online controller that uses only end-to-end measurements over short segments, without internal instrumentation, and hill climbing to maximize goodput, defined as the throughput of requests that satisfy the service-level objective. We provide empirical evidence that this design is well-founded. Using this advance in LLM serving as a concrete example, we then discuss the importance of integrating system performance and sustainability metrics into Factsheets for organizations adopting AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11337v1">RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      LLM agents increasingly perform end-to-end ML engineering tasks where success is judged by a single scalar test metric. This creates a structural vulnerability: an agent can increase the reported score by compromising the evaluation pipeline rather than improving the model. We introduce RewardHackingAgents, a workspace-based benchmark that makes two compromise vectors explicit and measurable: evaluator tampering (modifying metric computation or reporting) and train/test leakage (accessing held-out data or labels during training). Each episode runs in a fresh workspace with patch tracking and runtime file-access logging; detectors compare the agent-reported metric to a trusted reference to assign auditable integrity labels. Across three tasks and two LLM backbones, scripted attacks succeed on both vectors in fully mutable workspaces; single-mechanism defenses block only one vector; and a combined regime blocks both. In natural-agent runs, evaluator-tampering attempts occur in about 50% of episodes and are eliminated by evaluator locking, with a 25-31% median runtime overhead. Overall, we demonstrate that evaluation integrity for ML-engineering agents can be benchmarked as a first-class outcome rather than assumed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11333v1">LLM-Augmented Digital Twin for Policy Evaluation in Short-Video Platforms</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Short-video platforms are closed-loop, human-in-the-loop ecosystems where platform policy, creator incentives, and user behavior co-evolve. This feedback structure makes counterfactual policy evaluation difficult in production, especially for long-horizon and distributional outcomes. The challenge is amplified as platforms deploy AI tools that change what content enters the system, how agents adapt, and how the platform operates. We propose a large language model (LLM)-augmented digital twin for short-video platforms, with a modular four-twin architecture (User, Content, Interaction, Platform) and an event-driven execution layer that supports reproducible experimentation. Platform policies are implemented as pluggable components within the Platform Twin, and LLMs are integrated as optional, schema-constrained decision services (e.g., persona generation, content captioning, campaign planning, trend prediction) that are routed through a unified optimizer. This design enables scalable simulations that preserve closed-loop dynamics while allowing selective LLM adoption, enabling the study of platform policies, including AI-enabled policies, under realistic feedback and constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08281v2">Evaluating LLM-Based Grant Proposal Review via Structured Perturbations</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      As AI-assisted grant proposals outpace manual review capacity in a kind of ``Malthusian trap'' for the research ecosystem, this paper investigates the capabilities and limitations of LLM-based grant reviewing for high-stakes evaluation. Using six EPSRC proposals, we develop a perturbation-based framework probing LLM sensitivity across six quality axes: funding, timeline, competency, alignment, clarity, and impact. We compare three review architectures: single-pass review, section-by-section analysis, and a 'Council of Personas' ensemble emulating expert panels. The section-level approach significantly outperforms alternatives in both detection rate and scoring reliability, while the computationally expensive council method performs no better than baseline. Detection varies substantially by perturbation type, with alignment issues readily identified but clarity flaws largely missed by all systems. Human evaluation shows LLM feedback is largely valid but skewed toward compliance checking over holistic assessment. We conclude that current LLMs may provide supplementary value within EPSRC review but exhibit high variability and misaligned review priorities. We release our code and any non-protected data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11266v1">The Unlearning Mirage: A Dynamic Framework for Evaluating LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Unlearning in Large Language Models (LLMs) aims to enhance safety, mitigate biases, and comply with legal mandates, such as the right to be forgotten. However, existing unlearning methods are brittle: minor query modifications, such as multi-hop reasoning and entity aliasing, can recover supposedly forgotten information. As a result, current evaluation metrics often create an illusion of effectiveness, failing to detect these vulnerabilities due to reliance on static, unstructured benchmarks. We propose a dynamic framework that stress tests unlearning robustness using complex structured queries. Our approach first elicits knowledge from the target model (pre-unlearning) and constructs targeted probes, ranging from simple queries to multi-hop chains, allowing precise control over query difficulty. Our experiments show that the framework (1) shows comparable coverage to existing benchmarks by automatically generating semantically equivalent Q&A probes, (2) aligns with prior evaluations, and (3) uncovers new unlearning failures missed by other benchmarks, particularly in multi-hop settings. Furthermore, activation analyses show that single-hop queries typically follow dominant computation pathways, which are more likely to be disrupted by unlearning methods. In contrast, multi-hop queries tend to use alternative pathways that often remain intact, explaining the brittleness of unlearning techniques in multi-hop settings. Our framework enables practical and scalable evaluation of unlearning methods without the need for manual construction of forget test sets, enabling easier adoption for real-world applications. We release the pip package and the code at https://sites.google.com/view/unlearningmirage/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16584v2">Measuring Intent Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      People judge interactions with large language models (LLMs) as successful when outputs match what they want, not what they type. Yet LLMs are trained to predict the next token solely from text input, not underlying intent. Because written language is an imperfect proxy for intent, and correlations between phrasing and desired outcomes can break down in training data, models that rely too heavily on surface cues may respond inconsistently to semantically equivalent prompts. This makes it essential to evaluate whether LLMs can reliably infer user intent-especially in high-stakes settings where robustness and generalization are critical. We introduce a formal framework for assessing intent comprehension in LLMs: whether a model demonstrates robust understanding of user intent by producing consistent outputs across semantically equivalent prompts while differentiating between prompts with distinct intents. Our evaluation approach is based on a variance decomposition of model responses into three components: variability due to user intent, user articulation, and model uncertainty. Models that understand what users want, and are not overly sensitive to textual cues, should attribute most output variance to intent differences, rather than articulation style. Applying this framework across diverse domains, we find that, within the five LLaMA and Gemma models we evaluate, larger models typically assign a greater share of variance to intent, indicating stronger comprehension of intent, although gains are uneven and often modest with increasing model size. These results motivate moving beyond accuracy-only benchmarks toward semantic diagnostics that directly assess whether models understand what users intend.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11253v1">LLMs Can Infer Political Alignment from Online Conversations</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 55 pages; 4 figures in the main text and 18 supplementary figures, 11 supplementary tables
    </div>
    <details class="paper-abstract">
      Due to the correlational structure in our traits such as identities, cultures, and political attitudes, seemingly innocuous preferences such as following a band or using a specific slang, can reveal private traits. This possibility, especially when combined with massive, public social data and advanced computational methods, poses a fundamental privacy risk. Given our increasing data exposure online and the rapid advancement of AI are increasing the misuse potential of such risk, it is therefore critical to understand capacity of large language models (LLMs) to exploit it. Here, using online discussions on Debate.org and Reddit, we show that LLMs can reliably infer hidden political alignment, significantly outperforming traditional machine learning models. Prediction accuracy further improves as we aggregate multiple text-level inferences into a user-level prediction, and as we use more politics-adjacent domains. We demonstrate that LLMs leverage the words that can be highly predictive of political alignment while not being explicitly political. Our findings underscore the capacity and risks of LLMs for exploiting socio-cultural correlates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09596v2">How much are LLMs changing the language of academic papers after ChatGPT? A multi-database and full text analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      This study investigates how Large Language Models (LLMs) are influencing the language of academic papers by tracking 12 LLM-associated terms across six major scholarly databases (Scopus, Web of Science, PubMed, PubMed Central (PMC), Dimensions, and OpenAlex) from 2015 to 2024. Using over 2.4 million PMC open-access publications (2021-July 2025), we also analysed full texts to assess changes in the frequency and co-occurrence of these terms before and after ChatGPT's initial public release. Across databases, delve (+1,500%), underscore (+1,000%), and intricate (+700%) had the largest increases between 2022 and 2024. Growth in LLM-term usage was much higher in STEM fields than in social sciences and arts and humanities. In PMC full texts, the proportion of papers using underscore six or more times increased by over 10,000% from 2022 to 2025, followed by intricate (+5,400%) and meticulous (+2,800%). Nearly half of all 2024 PMC papers using any LLM term also included underscore, compared with only 3%-14% of papers before ChatGPT in 2022. Papers using one LLM term are now much more likely to include other terms. For example, in 2024, underscore strongly correlated with pivotal (0.449) and delve (0.311), compared with very weak associations in 2022 (0.032 and 0.018, respectively). These findings provide the first large-scale evidence based on full-text publications and multiple databases that some LLM-related terms are now being used much more frequently and together. The rapid uptake of LLMs to support scholarly publishing is a welcome development reducing the language barrier to academic publishing for non-English speakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11213v1">LLMs in social services: How does chatbot accuracy affect human accuracy?</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Social service programs like the Supplemental Nutrition Assistance Program (SNAP, or food stamps) have eligibility rules that can be challenging to understand. For nonprofit caseworkers who often support clients in navigating a dozen or more complex programs, LLM-based chatbots may offer a means to provide better, faster help to clients whose situations may be less common. In this paper, we measure the potential effects of LLM-based chatbot suggestions on caseworkers' ability to provide accurate guidance. We first created a 770-question multiple-choice benchmark dataset of difficult, but realistic questions that a caseworker might receive. Next, using these benchmark questions and corresponding expert-verified answers, we conducted a randomized experiment with caseworkers recruited from nonprofit outreach organizations in Los Angeles. Caseworkers in the control condition did not see chatbot suggestions and had a mean accuracy of 49%. Caseworkers in the treatment condition saw chatbot suggestions that we artificially varied to range in aggregate accuracy from low (53%) to high (100%). Caseworker performance significantly improves as chatbot quality improves: high-quality chatbots (96-100% accurate) improved caseworker accuracy by 27 percentage points. At the question-level, incorrect chatbot suggestions substantially reduce caseworker accuracy, with a two-thirds reduction on easy questions where the control group performed best (without chatbot suggestions). Finally, improvements in caseworker accuracy level off as chatbot accuracy increases, a phenomenon that we call the "AI underreliance plateau," which is a concern for real-world deployment and highlights the importance of evaluating human-in-the-loop tools with their users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11212v1">Security-by-Design for LLM-Based Code Generation: Leveraging Internal Representations for Concept-Driven Steering Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 to be published in the IEEE European Symposium on Security and Privacy (EuroS&P)'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show remarkable capabilities in understanding natural language and generating complex code. However, as practitioners adopt CodeLLMs for increasingly critical development tasks, research reveals that these models frequently generate functionally correct yet insecure code, posing significant security risks. While multiple approaches have been proposed to improve security in AI-based code generation, combined benchmarks show these methods remain insufficient for practical use, achieving only limited improvements in both functional correctness and security. This stems from a fundamental gap in understanding the internal mechanisms of code generation and the root causes of security vulnerabilities, forcing researchers to rely on heuristics and empirical observations. In this work, we investigate the internal representation of security concepts in CodeLLMs, revealing that models are often aware of vulnerabilities as they generate insecure code. Through systematic evaluation, we demonstrate that CodeLLMs can distinguish between security subconcepts, enabling a more fine-grained analysis than prior black-box approaches. Leveraging these insights, we propose Secure Concept Steering for CodeLLMs (SCS-Code). During token generation, SCS-Code steers LLMs' internal representations toward secure and functional code output, enabling a lightweight and modular mechanism that can be integrated into existing code models. Our approach achieves superior performance compared to state-of-the-art methods across multiple secure coding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11205v1">Can LLMs Help Localize Fake Words in Partially Fake Speech?</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Submitted to Interspeech 2026; put on arxiv based on requirement from Interspeech: "Interspeech no longer enforces an anonymity period for submissions." and "For authors that prefer to upload their paper online, a note indicating that the paper was submitted for review to Interspeech should be included in the posting."
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), trained on large-scale text, have recently attracted significant attention for their strong performance across many tasks. Motivated by this, we investigate whether a text-trained LLM can help localize fake words in partially fake speech, where only specific words within a speech are edited. We build a speech LLM to perform fake word localization via next token prediction. Experiments and analyses on AV-Deepfake1M and PartialEdit indicates that the model frequently leverages editing-style pattern learned from the training data, particularly word-level polarity substitutions for those two databases we discussed, as cues for localizing fake words. Although such particular patterns provide useful information in an in-domain scenario, how to avoid over-reliance on such particular pattern and improve generalization to unseen editing styles remains an open question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06322v3">Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics, Revealing a Three-Stage In-Context Learning Mechanism</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated emergent in-context learning (ICL) capabilities across a range of tasks, including zero-shot time-series forecasting. We show that text-trained foundation models can accurately extrapolate spatiotemporal dynamics from discretized partial differential equation (PDE) solutions without fine-tuning or natural language prompting. Predictive accuracy improves with longer temporal contexts but degrades at finer spatial discretizations. In multi-step rollouts, where the model recursively predicts future spatial states over multiple time steps, errors grow algebraically with the time horizon, reminiscent of global error accumulation in classical finite-difference solvers. We interpret these trends as in-context neural scaling laws, where prediction quality varies predictably with both context length and output length. To better understand how LLMs are able to internally process PDE solutions so as to accurately roll them out, we analyze token-level output distributions and uncover a consistent three-stage ICL progression: beginning with syntactic pattern imitation, transitioning through an exploratory high-entropy phase, and culminating in confident, numerically grounded predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03727v2">Understanding Parents' Desires in Moderating Children's Interactions with GenAI Chatbots through LLM-Generated Probes</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 33 pages, 10 figures, Accepted to ACM CHI 2026
    </div>
    <details class="paper-abstract">
      This paper studies how parents want to moderate children's interactions with Generative AI chatbots, with the goal of informing the design of future GenAI parental control tools. We first used an LLM to generate synthetic child-GenAI chatbot interaction scenarios and worked with four parents to validate their realism. From this dataset, we carefully selected 12 diverse examples that evoked varying levels of concern and were rated the most realistic. Each example included a prompt and a GenAI chatbot response. We presented these to parents (N=24) and asked whether they found them concerning, why, and how they would prefer the responses to be modified and communicated. Our findings reveal three key insights: (1) parents express concern about interactions that current GenAI chatbot parental controls neglect; (2) parents want fine-grained transparency and moderation at the conversation level; and (3) parents need personalized controls that adapt to their desired strategies and children's ages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11027v1">Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      The paradigm of LLM-as-a-judge relies on a critical assumption, namely that high inter-evaluator agreement indicates reliable and objective evaluation. We present two complementary findings that challenge this assumption. \textbf{First}, we demonstrate that this consensus is frequently illusory. We identify and formalize \textbf{Evaluation Illusion}, a phenomenon where LLM judges generate sophisticated critiques yet anchor scores on shared surface heuristics rather than substantive quality. Through a large-scale study of 105,600 evaluation instances (32 LLMs $\times$ 3 frontier judges $\times$ 100 tasks $\times$ 11 temperatures), we show that model-level agreement (Spearman $ρ= 0.99$) masks fragile sample-level agreement (Pearson $\bar{r} = 0.72$; absolute agreement ICC $= 0.67$), that merely sharing rubric structure restores 62\% of total agreement, and that high-quality outputs paradoxically receive the \textit{least} consistent evaluations. \textbf{Second}, we demonstrate that dynamically generating evaluation rubrics grounded in domain knowledge produces more meaningful assessment. We introduce MERG (Metacognitive Enhanced Rubric Generation), a knowledge-driven rubric generation framework whose domain-selective effects confirm this. Agreement \textit{increases} in codified domains (Education +22\%, Academic +27\%) where knowledge anchors evaluators on shared standards, while it decreases in subjective domains where genuine evaluative pluralism emerges. These findings suggest that evaluation rubrics should be dynamically enriched with expert knowledge rather than relying on generic criteria, with implications for reward modeling in RLAIF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11025v1">LLMGreenRec: LLM-Based Multi-Agent Recommender System for Sustainable E-Commerce</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Accepted to the Proceedings of the Conference on Digital Economy and Fintech Innovation (DEFI 2025). To appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      Rising environmental awareness in e-commerce necessitates recommender systems that not only guide users to sustainable products but also minimize their own digital carbon footprints. Traditional session-based systems, optimized for short-term conversions, often fail to capture nuanced user intents for eco-friendly choices, perpetuating a gap between green intentions and actions. To tackle this, we introduce LLMGreenRec, a novel multi-agent framework that leverages Large Language Models (LLMs) to promote sustainable consumption. Through collaborative analysis of user interactions and iterative prompt refinement, LLMGreenRec's specialized agents deduce green-oriented user intents and prioritize eco-friendly product recommendations. Notably, this intent-driven approach also reduces unnecessary interactions and energy consumption. Extensive experiments on benchmark datasets validate LLMGreenRec's effectiveness in recommending sustainable products, demonstrating a robust solution that fosters a responsible digital economy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11021v1">Leech Lattice Vector Quantization for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Scalar quantization of large language models (LLMs) is fundamentally limited by information-theoretic bounds. While vector quantization (VQ) overcomes these limits by encoding blocks of parameters jointly, practical implementations must avoid the need for expensive lookup mechanisms or other explicit codebook storage. Lattice approaches address this through highly structured and dense packing. This paper explores the Leech lattice, which, with its optimal sphere packing and kissing configurations at 24 dimensions, is the highest dimensional lattice known with such optimal properties. To make the Leech lattice usable for LLM quantization, we extend an existing search algorithm based on the extended Golay code construction, to i) support indexing, enabling conversion to and from bitstrings without materializing the codebook, ii) allow angular search over union of Leech lattice shells, iii) propose fully-parallelisable dequantization kernel. Together this yields a practical algorithm, namely Leech Lattice Vector Quantization (LLVQ). LLVQ delivers state-of-the-art LLM quantization performance, outperforming recent methods such as Quip\#, QTIP, and PVQ. These results highlight the importance of high-dimensional lattices for scalable, theoretically grounded model compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11011v1">Task-Aware Delegation Cues for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Accepeted by CHI'26 Workshop on Developing Standards and Documentation For LLM Use as Simulated Research Participants
    </div>
    <details class="paper-abstract">
      LLM agents increasingly present as conversational collaborators, yet human--agent teamwork remains brittle due to information asymmetry: users lack task-specific reliability cues, and agents rarely surface calibrated uncertainty or rationale. We propose a task-aware collaboration signaling layer that turns offline preference evaluations into online, user-facing primitives for delegation. Using Chatbot Arena pairwise comparisons, we induce an interpretable task taxonomy via semantic clustering, then derive (i) Capability Profiles as task-conditioned win-rate maps and (ii) Coordination-Risk Cues as task-conditioned disagreement (tie-rate) priors. These signals drive a closed-loop delegation protocol that supports common-ground verification, adaptive routing (primary vs.\ primary+auditor), explicit rationale disclosure, and privacy-preserving accountability logs. Two predictive probes validate that task typing carries actionable structure: cluster features improve winner prediction accuracy and reduce difficulty prediction error under stratified 5-fold cross-validation. Overall, our framework reframes delegation from an opaque system default into a visible, negotiable, and auditable collaborative decision, providing a principled design space for adaptive human--agent collaboration grounded in mutual awareness and shared accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11008v1">A Systematic Study of Pseudo-Relevance Feedback with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Pseudo-relevance feedback (PRF) methods built on large language models (LLMs) can be organized along two key design dimensions: the feedback source, which is where the feedback text is derived from and the feedback model, which is how the given feedback text is used to refine the query representation. However, the independent role that each dimension plays is unclear, as both are often entangled in empirical evaluations. In this paper, we address this gap by systematically studying how the choice of feedback source and feedback model impact PRF effectiveness through controlled experimentation. Across 13 low-resource BEIR tasks with five LLM PRF methods, our results show: (1) the choice of feedback model can play a critical role in PRF effectiveness; (2) feedback derived solely from LLM-generated text provides the most cost-effective solution; and (3) feedback derived from the corpus is most beneficial when utilizing candidate documents from a strong first-stage retriever. Together, our findings provide a better understanding of which elements in the PRF design space are most important.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10960v1">Ranking Reasoning LLMs under Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Code is available at https://github.com/mohsenhariri/scorio
    </div>
    <details class="paper-abstract">
      Test-time scaling evaluates reasoning LLMs by sampling multiple outputs per prompt, but ranking models in this regime remains underexplored. We formalize dense benchmark ranking under test-time scaling and introduce Scorio, a library that implements statistical ranking methods such as paired-comparison models, item response theory (IRT) models, voting rules, and graph- and spectral-based methods. Across $20$ reasoning models on four Olympiad-style math benchmarks (AIME'24, AIME'25, HMMT'25, and BrUMO'25; up to $N=80$ trials), most full-trial rankings agree closely with the Bayesian gold standard $\mathrm{Bayes}_{\mathcal{U}}@80$ (mean Kendall's $τ_b = 0.93$--$0.95$), and $19$--$34$ methods recover exactly the same ordering. In the single-trial regime, the best methods reach $τ_b \approx 0.86$. Using greedy decoding as an empirical prior ($\mathrm{Bayes}_{\mathbf{R}_0}@N$) reduces variance at $N=1$ by $16$--$52\%$, but can bias rankings when greedy and stochastic sampling disagree. These results identify reliable ranking methods for both high- and low-budget test-time scaling. We release Scorio as an open-source library at https://github.com/mohsenhariri/scorio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14286v2">Online LLM watermark detection via e-processes</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) has emerged as an effective tool for distinguishing AI-generated text from human-written content. Statistically, watermark schemes induce dependence between generated tokens and a pseudo-random sequence, reducing watermark detection to a hypothesis testing problem on independence. We develop a unified framework for LLM watermark detection based on e-processes, providing anytime-valid guarantees for online testing. We propose various methods to construct empirically adaptive e-processes that can enhance the detection power. The proposed methods are applicable to any sequential testing problem where independent pivotal statistics are available. In addition, theoretical results are established to characterize the power properties of the proposed procedures. Some experiments demonstrate that the proposed framework achieves competitive performance compared to existing watermark detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20744v2">From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 This article has been accepted for publication in Information and Software Technology (IST)
    </div>
    <details class="paper-abstract">
      Context: Laws and regulations increasingly shape software design, development, and quality assurance in regulated domains. Because legal provisions are written in technology-neutral language, deriving concrete specifications, requirements, and acceptance criteria to verify software compliance is difficult and error-prone. Recent advances in generative AI, especially large language models (LLMs), may help automate this process. Objective: We present the first systematic human-subject evaluation of LLMs' ability to derive Gherkin behavioural specifications from legal texts using a quasi-experimental design. Gherkin is a domain-specific language for scenario-based system behaviour descriptions in Given-When-Then form and is well suited to automation in software development. Methods: Ten participants evaluated 60 Gherkin specifications generated from food-safety regulations by Claude and Llama. Each participant assessed 12 specifications across five criteria: relevance, clarity, completeness, singularity, and time savings. Each specification was evaluated by two participants, yielding 120 assessments with quantitative ratings and qualitative feedback. Results: Ratings were uniformly high in the top two categories: relevance 95%, clarity 100%, completeness 94.2%, singularity 93.4%, and time savings 91.7%. No statistically reliable differences were found across participants or between LLMs. Qualitative feedback noted occasional omissions, hallucinations, and mixed intents, underscoring the need for human oversight, especially in safety-critical domains. Conclusion: In food safety, LLMs can assist in deriving Gherkin specifications from legal texts, but omissions and hallucinations require systematic human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11132v1">WebWeaver: Breaking Topology Confidentiality in LLM Multi-Agent Systems with Stealthy Context-Based Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Communication topology is a critical factor in the utility and safety of LLM-based multi-agent systems (LLM-MAS), making it a high-value intellectual property (IP) whose confidentiality remains insufficiently studied. % Existing topology inference attempts rely on impractical assumptions, including control over the administrative agent and direct identity queries via jailbreaks, which are easily defeated by basic keyword-based defenses. As a result, prior analyses fail to capture the real-world threat of such attacks. % To bridge this realism gap, we propose \textit{WebWeaver}, an attack framework that infers the complete LLM-MAS topology by compromising only a single arbitrary agent instead of the administrative agent. % Unlike prior approaches, WebWeaver relies solely on agent contexts rather than agent IDs, enabling significantly stealthier inference. % WebWeaver further introduces a new covert jailbreak-based mechanism and a novel fully jailbreak-free diffusion design to handle cases where jailbreaks fail. % Additionally, we address a key challenge in diffusion-based inference by proposing a masking strategy that preserves known topology during diffusion, with theoretical guarantees of correctness. % Extensive experiments show that WebWeaver substantially outperforms state-of-the-art (SOTA) baselines, achieving about 60\% higher inference accuracy under active defenses with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.21271v5">EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 ICLR 2026 workshops. Code: https://github.com/NVlabs/EoRA
    </div>
    <details class="paper-abstract">
      While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA, a novel fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{11.45\%}$ on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08935v2">PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01815v2">KV Cache Transform Coding for Compact Storage in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper and RULER. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10904v1">When Fine-Tuning Fails and when it Generalises: Role of Data Diversity and Mixed Training in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 We finetune the Qwen 0.5B backbone in an LLM TTS with LoRA to raise MOS speaker similarity and SNR. It works best with diverse training audio with uniform data it can amplify noise so tune decoding and use GGUF quantization for low latency stable quality
    </div>
    <details class="paper-abstract">
      Large language models are increasingly adopted as semantic backbones for neural text-to-speech systems. However, frozen LLM representations are insufficient for modeling speaker specific acoustic and perceptual characteristics. Our experiments involving fine tuning of the Language Model backbone of TTS show promise in improving the voice consistency and Signal to Noise ratio SNR in voice cloning task. Across multiple speakers LoRA finetuning consistently outperforms the non-finetuned base Qwen-0.5B model across three complementary dimensions of speech quality. First, perceptual quality improves significantly with DNS-MOS gains of up to 0.42 points for speakers whose training data exhibits sufficient acoustic variability. Second, speaker fidelity improves for all evaluated speakers with consistent increases in voice similarity indicating that LoRA effectively adapts speaker identity representations without degrading linguistic modeling. Third, signal level quality improves in most cases with signal to noise ratio increasing by as much as 34 percent. Crucially these improvements are strongly governed by the characteristics of the training data. Speakers with high variability in acoustic energy and perceptual quality achieve simultaneous gains in DNS-MOS voice similarity and SNR. Overall this work establishes that LoRA finetuning is not merely a parameter efficient optimization technique but an effective mechanism for better speaker level adaptation in compact LLM-based TTS systems. When supported by sufficiently diverse training data LoRA adapted Qwen-0.5B consistently surpasses its frozen base model in perceptual quality speaker similarity with low latency using GGUF model hosted in quantized form.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11126v1">Enhancing Value Alignment of LLMs with Multi-agent system and Combinatorial Fusion</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 5 pages, 3 figures, accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human values is a central challenge for ensuring trustworthy and safe deployment. While existing methods such as Reinforcement Learning from Human Feedback (RLHF) and its variants have improved alignment, they often rely on a single evaluator or narrowly defined reward signals, limiting their ability to capture ethical pluralism. In this work, we propose the Value Alignment System using Combinatorial Fusion Analysis (VAS-CFA), a framework that operationalizes multi-agent fusion alignment. It instantiates multiple moral agents, each fine-tuned to represent a distinct normative perspective, and fuses their outputs using CFA with both rank- and score-based aggregation. This design leverages cognitive diversity, between agents, to mitigate conflicts and redundancies across multiple agents, producing responses that better reflect human values. Empirical evaluation demonstrates that VAS-CFA outperforms both single agent baselines and prior aggregation approaches on standard metrics, showing that multi-agent fusion provides a robust and effective mechanism for advancing value alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08907v4">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 23 pages,10 figures
    </div>
    <details class="paper-abstract">
      Context compression is an advanced technique that accelerates large language model (LLM) inference by converting long inputs into compact representations. Existing methods primarily rely on autoencoding tasks to train special compression tokens to represent contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, we remark that such capabilities potentially conflict with actual downstream task requirements, prevent the models from learning the features more beneficial for real-world usage. Based on this observation, we propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. To ensure that anchors can effectively collect information, SAC introduces two key designs: (1) anchor embedding, a learnable embedding vector attached to the selected anchor tokens to mark compression carriers and (2) bidirectional attention modification, which enables anchor tokens to integrate information from the entire context. Experimental results show that SAC consistently outperforms existing context compression methods across different compression ratios and model sizes on question-answering and long-context summarization tasks. Our data, model and code have been released at \href{https://github.com/lx-Meteors/SAC}{https://github.com/lx-Meteors/SAC}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06627v3">Burn-After-Use for Preventing Data Leakage through a Secure Multi-Tenant Architecture in Enterprise LLM</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This study presents a Secure Multi-Tenant Architecture (SMTA) combined with a novel concept Burn-After-Use (BAU) mechanism for enterprise LLM environments to effectively prevent data leakage. As institutions increasingly adopt LLMs across departments, the risks of data leakage have become a critical security and compliance concern. The proposed SMTA isolates LLM instances across departments and enforces rigorous context ownership boundaries within an internally deployed infrastructure. The BAU mechanism introduces data confidentiality by enforcing ephemeral conversational contexts that are automatically destroyed after use, preventing cross-session or cross-user inference. The evaluation to SMTA and BAU is through two sets of realistic and reproducible experiments comprising of 127 test iterations. One aspect of this experiment is to assess prompt-based and semantic leakage attacks in a multi-tenant architecture (Appendix A) across 55 infrastructure-level attack tests, including vector-database credential compromise and shared logging pipeline exposure. SMTA achieves 92% defense success rate, demonstrating strong semantic isolation while highlighting residual risks from credential misconfiguration and observability pipelines. Another aspect is to evaluate the robustness of BAU under realistic failure scenarios (Appendix B) using four empirical metrics: Local Residual Persistence Rate (LRPR), Remote Residual Persistence Rate (RRPR), Image Frame Exposure Rate (IFER), and Burn Timer Persistence Rate (BTPR). Across 72 test iterations, BAU achieves a 76.75% success rate in mitigating post-session leakage threats across the client, server, application, infrastructure, and cache layers. These results show that SMTA and BAU together enforce strict isolation, complete session ephemerality, strong confidentiality guarantees, non-persistence, and policy-aligned behavior for enterprise LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10827v1">Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 3 Tables, 1 Figure, Under review
    </div>
    <details class="paper-abstract">
      Speech-aware large language models (LLMs) can accept speech inputs, yet their training objectives largely emphasize linguistic content or specific fields such as emotions or the speaker's gender, leaving it unclear whether they encode speaker identity. First, we propose a model-agnostic scoring protocol that produces continuous verification scores for both API-only and open-weight models, using confidence scores or log-likelihood ratios from the Yes/No token probabilities. Using this protocol, we benchmark recent speech-aware LLMs and observe weak speaker discrimination (EERs above 20% on VoxCeleb1). Second, we introduce a lightweight augmentation that equips an LLM with ASV capability by injecting frozen ECAPA-TDNN speaker embeddings through a learned projection and training only LoRA adapters. On TinyLLaMA-1.1B, the resulting ECAPA-LLM achieves 1.03% EER on VoxCeleb1-E, approaching a dedicated speaker verification system while preserving a natural-language interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11123v1">Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Although the deep integration of the Automatic Speech Recognition (ASR) system with Large Language Models (LLMs) has significantly improved accuracy, the deployment of such systems in low-latency streaming scenarios remains challenging. In this paper, we propose Uni-ASR, a unified framework based on LLMs that integrates both non-streaming and streaming speech recognition capabilities. We propose a joint training paradigm that enables the system to seamlessly transition between two recognition modes without any architectural modifications. Furthermore, we introduce a context-aware training paradigm and a co-designed fallback decoding strategy, which can enhance streaming recognition accuracy without introducing additional latency. The experimental results demonstrate that Uni-ASR not only achieves competitive performance within non-streaming mode, but also demonstrates strong effectiveness in streaming scenarios under diverse latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10807v1">Risk-Adjusted Harm Scoring for Automated Red Teaming for LLMs in Financial Services</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) in financial services introduces new operational, regulatory, and security risks. Yet most red-teaming benchmarks remain domain-agnostic and fail to capture failure modes specific to regulated BFSI settings, where harmful behavior can be elicited through legally or professionally plausible framing. We propose a risk-aware evaluation framework for LLM security failures in Banking, Financial Services, and Insurance (BFSI), combining a domain-specific taxonomy of financial harms, an automated multi-round red-teaming pipeline, and an ensemble-based judging protocol. We introduce the Risk-Adjusted Harm Score (RAHS), a risk-sensitive metric that goes beyond success rates by quantifying the operational severity of disclosures, accounting for mitigation signals, and leveraging inter-judge agreement. Across diverse models, we find that higher decoding stochasticity and sustained adaptive interaction not only increase jailbreak success, but also drive systematic escalation toward more severe and operationally actionable financial disclosures. These results expose limitations of single-turn, domain-agnostic security evaluation and motivate risk-sensitive assessment under prolonged adversarial pressure for real-world BFSI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10784v1">Interpretable Chinese Metaphor Identification via LLM-Assisted MIPVU Rule Script Generation: A Comparative Protocol Study</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Metaphor identification is a foundational task in figurative language processing, yet most computational approaches operate as opaque classifiers offering no insight into why an expression is judged metaphorical. This interpretability gap is especially acute for Chinese, where rich figurative traditions, absent morphological cues, and limited annotated resources compound the challenge. We present an LLM-assisted pipeline that operationalises four metaphor identification protocols--MIP/MIPVU lexical analysis, CMDAG conceptual-mapping annotation, emotion-based detection, and simile-oriented identification--as executable, human-auditable rule scripts. Each protocol is a modular chain of deterministic steps interleaved with controlled LLM calls, producing structured rationales alongside every classification decision. We evaluate on seven Chinese metaphor datasets spanning token-, sentence-, and span-level annotation, establishing the first cross-protocol comparison for Chinese metaphor identification. Within-protocol evaluation shows Protocol A (MIP) achieves an F1 of 0.472 on token-level identification, while cross-protocol analysis reveals striking divergence: pairwise Cohen's kappa between Protocols A and D is merely 0.001, whereas Protocols B and C exhibit near-perfect agreement (kappa = 0.986). An interpretability audit shows all protocols achieve 100% deterministic reproducibility, with rationale correctness from 0.40 to 0.87 and editability from 0.80 to 1.00. Error analysis identifies conceptual-domain mismatch and register sensitivity as dominant failure modes. Our results demonstrate that protocol choice is the single largest source of variation in metaphor identification, exceeding model-level variation, and that rule-script architectures achieve competitive performance while maintaining full transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10749v1">AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 24 pages, 3 figures
    </div>
    <details class="paper-abstract">
      LLM agents are highly vulnerable to Indirect Prompt Injection (IPI), where adversaries embed malicious directives in untrusted tool outputs to hijack execution. Most existing defenses treat IPI as an input-level semantic discrimination problem, which often fails to generalize to unseen payloads. We propose a new paradigm, action-level causal attribution, which secures agents by asking why a particular tool call is produced. The central goal is to distinguish tool calls supported by the user's intent from those causally driven by untrusted observations. We instantiate this paradigm with AttriGuard, a runtime defense based on parallel counterfactual tests. For each proposed tool call, AttriGuard verifies its necessity by re-executing the agent under a control-attenuated view of external observations. Technically, AttriGuard combines teacher-forced shadow replay to prevent attribution confounding, hierarchical control attenuation to suppress diverse control channels while preserving task-relevant information, and a fuzzy survival criterion that is robust to LLM stochasticity. Across four LLMs and two agent benchmarks, AttriGuard achieves 0% ASR under static attacks with negligible utility loss and moderate overhead. Importantly, it remains resilient under adaptive optimization-based attacks in settings where leading defenses degrade significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10726v1">CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rely on optimizations like Automatic Prefix Caching (APC) to accelerate inference. APC works by reusing previously computed states for the beginning part of a request (prefix), when another request starts with the same text. While APC improves throughput, it introduces timing side channels: cache hits are faster than misses, creating observable latency differences. In multi-tenant systems, attackers can exploit these differences to infer sensitive information, e.g., by incrementally reconstructing another user's request by observing hit/miss patterns. Current defenses take a sledgehammer approach: they disable APC and cache sharing, isolating users, and sacrificing efficiency for regular users. This paper presents CacheSolidarity, a system that secures multi-tenant LLM serving systems against APC side channels without sacrificing performance and efficiency. CacheSolidarity monitors cache reuse across users, flags suspicious sharing, and selectively isolates prefixes, restricting their reuse only when necessary. Evaluation shows that CacheSolidarity enables up to 70% higher cache reuse and 30% lower inference latency compared to existing defenses that isolate users. CacheSolidarity's lightweight design demonstrates how security in LLM serving does not have to come at the cost of unnecessarily reduced performance or unbearable overheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05748v3">Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Published in EACL 2026 - Corrected cooperation rates for two-stage communication conditions (96.7% and 100.0%, previously reported as 48.3% and 50.0% due to a denominator bug in the evaluation code). All other results unchanged
    </div>
    <details class="paper-abstract">
      Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 96.7%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment, demonstrating that optimizing for short-term rationality can actively undermine alignment goals. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09442v4">Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026. Code: https://github.com/SiO-2/kvcloak
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10640v1">Making Bielik LLM Reason (Better): A Field Report</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      This paper presents a research program dedicated to evaluating and advancing the reasoning capabilities of Bielik, a Polish large language model. The study describes a number of stages of work: initial benchmarking and creation of evaluation methodology, analyzing of comparative results with other LLMs and outlining of future prospects that take into account the limitations of the analyses conducted so far and aims to keep Bielik in the race give the ever-changing -- and competitive -- AI landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09704v2">Evaluation of LLMs in retrieving food and nutritional context for RAG systems</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 This is the preprint for our conference paper for IEEE International Conference on Big Data
    </div>
    <details class="paper-abstract">
      In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10588v1">Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10587v1">Distilling LLM Semantic Priors into Encoder-Only Multi-Talker ASR with Talker-Count Routing</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide strong semantic priors that can improve multi-talker automatic speech recognition (MT-ASR), but using an LLM as an autoregressive decoder is computationally expensive and remains fragile under heavy overlap. In this paper, we propose an encoder-only MT-ASR framework that adapts an LLM to multi-talker conditioning and distills its semantic guidance into the encoder during training, while retaining fast CTC-style decoding at inference. Our model employs a post-encoder separator with serialized CTC to produce talker-ordered transcripts, and leverages an adapted LLM-based SOT objective as a multi-talker-aware teacher signal to explicitly regularize mixed-speech representations. To further support variable numbers of talkers, we introduce a Talker-Count Head that predicts the talker count and dynamically selects the appropriate decoding branch. Experiments on LibriMix show that the proposed encoder-only model achieves comparable performance to LLM-based systems in the two-talker condition, while delivering significant improvements in the three-talker condition with significant small RTF.
    </details>
</div>
