# llm - 2025_10

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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- Part 18

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00615v1">ACON: Optimizing Context Compression for Long-horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01295v1">The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) transition from static tools to autonomous agents, traditional evaluation benchmarks that measure performance on downstream tasks are becoming insufficient. These methods fail to capture the emergent social and cognitive dynamics that arise when agents communicate, persuade, and collaborate in interactive environments. To address this gap, we introduce a novel evaluation framework that uses multi-agent debate as a controlled "social laboratory" to discover and quantify these behaviors. In our framework, LLM-based agents, instantiated with distinct personas and incentives, deliberate on a wide range of challenging topics under the supervision of an LLM moderator. Our analysis, enabled by a new suite of psychometric and semantic metrics, reveals several key findings. Across hundreds of debates, we uncover a powerful and robust emergent tendency for agents to seek consensus, consistently reaching high semantic agreement ({\mu} > 0.88) even without explicit instruction and across sensitive topics. We show that assigned personas induce stable, measurable psychometric profiles, particularly in cognitive effort, and that the moderators persona can significantly alter debate outcomes by structuring the environment, a key finding for external AI alignment. This work provides a blueprint for a new class of dynamic, psychometrically grounded evaluation protocols designed for the agentic setting, offering a crucial methodology for understanding and shaping the social behaviors of the next generation of AI agents. We have released the code and results at https://github.com/znreza/multi-agent-LLM-eval-for-debate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00579v1">CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing implementations, such as in-context learning and fine-tuning, remain costly and inefficient. To improve CoT reasoning at a lower cost, and inspired by the task vector paradigm, we introduce CoT Vectors, compact representations that encode task-general, multi-step reasoning knowledge. Through experiments with Extracted CoT Vectors, we observe pronounced layer-wise instability, manifesting as a U-shaped performance curve that reflects a systematic three-stage reasoning process in LLMs. To address this limitation, we propose Learnable CoT Vectors, optimized under a teacher-student framework to provide more stable and robust guidance. Extensive evaluations across diverse benchmarks and models demonstrate that CoT Vectors not only outperform existing baselines but also achieve performance comparable to parameter-efficient fine-tuning methods, while requiring fewer trainable parameters. Moreover, by treating CoT Vectors as a probe, we uncover how their effectiveness varies due to latent space structure, information density, acquisition mechanisms, and pre-training differences, offering new insights into the functional organization of multi-step reasoning in LLMs. The source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00555v1">PromptPilot: Improving Human-AI Collaboration Through LLM-Enhanced Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint version. Accepted for presentation at the International Conference on Information Systems (ICIS 2025). Please cite the published version when available
    </div>
    <details class="paper-abstract">
      Effective prompt engineering is critical to realizing the promised productivity gains of large language models (LLMs) in knowledge-intensive tasks. Yet, many users struggle to craft prompts that yield high-quality outputs, limiting the practical benefits of LLMs. Existing approaches, such as prompt handbooks or automated optimization pipelines, either require substantial effort, expert knowledge, or lack interactive guidance. To address this gap, we design and evaluate PromptPilot, an interactive prompting assistant grounded in four empirically derived design objectives for LLM-enhanced prompt engineering. We conducted a randomized controlled experiment with 80 participants completing three realistic, work-related writing tasks. Participants supported by PromptPilot achieved significantly higher performance (median: 78.3 vs. 61.7; p = .045, d = 0.56), and reported enhanced efficiency, ease-of-use, and autonomy during interaction. These findings empirically validate the effectiveness of our proposed design objectives, establishing LLM-enhanced prompt engineering as a viable technique for improving human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00543v1">Flow of Knowledge: Federated Fine-Tuning of LLMs in Healthcare under Non-IID Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great promise in healthcare, but their applications are hindered by data privacy restrictions and the challenges of cross-institution collaboration. Sensitive medical data cannot be centralized, while non-independent and identically distributed (non-IID) characteristics across institutions further complicate convergence and fairness. To address these issues, we present a federated fine-tuning approach based on Low-Rank Adaptation (LoRA), enabling privacy-preserving knowledge flow across institutions. The method iteratively combines local LoRA adaptation with global parameter aggregation, allowing efficient knowledge sharing without exposing raw data. A blockchain identity scheme is used for identifying individual LLM in such a distributed network. We evaluate this approach on heterogeneous and highly non-IID medical text datasets, where experiments demonstrate that federated LoRA not only enhances cross-client generalization but also improves the performance of the weakest client, achieving stable convergence and fairer outcomes. These findings highlight federated LoRA fine-tuning as a practical and effective paradigm for adapting LLMs in healthcare, offering a new path for multi-center medical AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00494v1">Exploring System 1 and 2 communication for latent reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Should LLM reasoning live in a separate module, or within a single model's forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline, a single model with the same forward pass and shared representations, using the same latent-token budget, nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives and communication mechanisms that explicitly shape latent spaces for algorithmic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00482v1">Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted by AIxB 2025
    </div>
    <details class="paper-abstract">
      Agentic large language models (LLMs) have become prominent for autonomously interacting with external environments and performing multi-step reasoning tasks. Most approaches leverage these capabilities via in-context learning with few-shot prompts, but this often results in lengthy inputs and higher computational costs. Agent fine-tuning offers an alternative by enabling LLMs to internalize procedural reasoning and domain-specific knowledge through training on relevant data and demonstration trajectories. While prior studies have focused on general domains, their effectiveness in specialized technical microdomains remains unclear. This paper explores agent fine-tuning for domain adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT operations. We fine-tuned LLMs using JP1-specific datasets derived from domain manuals and distilled reasoning trajectories generated by LLMs themselves, enhancing decision making accuracy and search efficiency. During inference, we used an agentic prompt with retrieval-augmented generation and introduced a context-answer extractor to improve information relevance. On JP1 certification exam questions, our method achieved a 14% performance improvement over the base model, demonstrating the potential of agent fine-tuning for domain-specific reasoning in complex microdomains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00481v1">Make a Video Call with LLM: A Measurement Campaign over Five Mainstream Apps</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      In 2025, Large Language Model (LLM) services have launched a new feature -- AI video chat -- allowing users to interact with AI agents via real-time video communication (RTC), just like chatting with real people. Despite its significance, no systematic study has characterized the performance of existing AI video chat systems. To address this gap, this paper proposes a comprehensive benchmark with carefully designed metrics across four dimensions: quality, latency, internal mechanisms, and system overhead. Using custom testbeds, we further evaluate five mainstream AI video chatbots with this benchmark. This work provides the research community a baseline of real-world performance and identifies unique system bottlenecks. In the meantime, our benchmarking results also open up several research questions for future optimizations of AI video chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00449v1">Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted to EMNLP 2025 PALS Workshop
    </div>
    <details class="paper-abstract">
      Personalizing the outputs of large language models (LLMs) to align with individual user preferences is an active research area. However, previous studies have mainly focused on classification or ranking tasks and have not considered Likert-scale rating prediction, a regression task that requires both language and mathematical reasoning to be solved effectively. This task has significant industrial applications, but the utilization of LLMs remains underexplored, particularly regarding the capabilities of off-the-shelf LLMs. This study investigates the performance of off-the-shelf LLMs on rating prediction, providing different in-context information. Through comprehensive experiments with eight models across three datasets, we demonstrate that user-written reviews significantly improve the rating prediction performance of LLMs. This result is comparable to traditional methods like matrix factorization, highlighting the potential of LLMs as a promising solution for the cold-start problem. We also find that the reviews for concrete items are more effective than general preference descriptions that are not based on any specific item. Furthermore, we discover that prompting LLMs to first generate a hypothetical review enhances the rating prediction performance. Our code is available at https://github.com/ynklab/rating-prediction-with-reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00419v1">Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Zeroth-order optimizers have recently emerged as a practical approach for fine-tuning large language models (LLMs), significantly reducing GPU memory consumption compared to traditional first-order methods. Yet, existing zeroth-order methods rely on hand-crafted, static sampling strategies that are not adaptable to model-specific structures. To address this, we propose ZO Fine-tuner, a learning-based zeroth-order optimizer for LLMs that automatically learns efficient perturbation strategies through a compact and memory-efficient design. Crucially, our approach is motivated by the observation that only a small number of foundation models and their derivatives are widely adopted in practice. Therefore, learning the optimizer once for a given LLM and reusing it across diverse downstream tasks is both feasible and highly desirable. Accordingly, ZO Fine-tuner is designed to scale learning to learn (L2L) to the foundation-model era by supporting one-time training per LLM with minimal overhead. Experiments on 4 LLMs and 7 datasets show that ZO Fine-tuner outperforms prior zeroth-order baselines in 82.1\% of task-model combinations, thereby demonstrating strong performance and scalability for efficient LLM fine-tuning. Our code is available at https://github.com/ASTRAL-Group/ZO_Fine_tuner.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00414v1">RELATE-Sim: Leveraging Turning Point Theory and LLM Agents to Predict and Understand Long-Term Relationship Dynamics through Interactive Narrative Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 10 pages, 3 figures, Submitted to CHI 2026 Conference
    </div>
    <details class="paper-abstract">
      Most dating technologies optimize for getting together, not staying together. We present RELATE-Sim, a theory-grounded simulator that models how couples behave at consequential turning points-exclusivity talks, conflict-and-repair episodes, relocations-rather than static traits. Two persona-aligned LLM agents (one per partner) interact under a centralized Scene Master that frames each turning point as a compact set of realistic options, advances the narrative, and infers interpretable state changes and an auditable commitment estimate after each scene. On a longitudinal dataset of 71 couples with two-year follow-ups, simulation-aware predictions outperform a personas-only baseline while surfacing actionable markers (e.g., repair attempts acknowledged, clarity shifts) that explain why trajectories diverge. RELATE-Sim pushes the relationship research's focus from matchmaking to maintenance, providing a transparent, extensible platform for understanding and forecasting long-term relationship dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01288v1">Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 main pages, 13 appendix pages
    </div>
    <details class="paper-abstract">
      We draw inspiration from microsaccades, tiny involuntary eye movements that reveal hidden dynamics of human perception, to propose an analogous probing method for large language models (LLMs). Just as microsaccades expose subtle but informative shifts in vision, we show that lightweight position encoding perturbations elicit latent signals that indicate model misbehaviour. Our method requires no fine-tuning or task-specific supervision, yet detects failures across diverse settings including factuality, safety, toxicity, and backdoor attacks. Experiments on multiple state-of-the-art LLMs demonstrate that these perturbation-based probes surface misbehaviours while remaining computationally efficient. These findings suggest that pretrained LLMs already encode the internal evidence needed to flag their own failures, and that microsaccade-inspired interventions provide a pathway for detecting and mitigating undesirable behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v4">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14285v2">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 IEEE Conference standard paper
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23571v2">Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. This paper presents CyberTeam, a benchmark designed to guide LLMs in blue teaming practice. CyberTeam constructs a standardized workflow in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of operational modules tailored to its specific analytical requirements. This transforms threat hunting into a structured sequence of reasoning steps, with each step grounded in a discrete operation and ordered according to task-specific dependencies. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modularized steps. Overall, CyberTeam integrates 30 tasks and 9 operational modules to guide LLMs through standardized threat analysis. We evaluate both leading LLMs and state-of-the-art cybersecurity agents, comparing CyberTeam against open-ended reasoning strategies. Our results highlight the improvements enabled by standardized design, while also revealing the limitations of open-ended reasoning in real-world threat hunting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23573v2">Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06920v2">An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 6 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including accuracy, precision, recall, F1, specificity, FAR, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11679v3">Ambiguity in LLMs is a concept missing problem</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 17 pages, 11 figures, title updated
    </div>
    <details class="paper-abstract">
      Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods to ambiguity handling either rely on the ReACT framework to obtain correct mappings through trial and error, or on supervised fine-tuning to bias models toward specific tasks. In this paper, we adopt a different approach that characterizes representation differences of ambiguous text in the latent space and leverages these differences to identify ambiguity before mapping them to structured data. To detect sentence-level ambiguity, we focus on the relationship between ambiguous questions and their interpretations. Unlike distances calculated by dense embeddings, we introduce a new distance measure based on a path kernel over concepts. With this measurement, we identify patterns to distinguish ambiguous from unambiguous questions. Furthermore, we propose a method for improving LLM performance on ambiguous agentic tool calling through missing concept prediction. Both achieve state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23019v2">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04018v2">AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Prepint, under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. While prior research has studied agents' ability to produce harmful outputs or follow malicious instructions, it remains unclear how likely agents are to spontaneously pursue unintended goals in realistic deployments. In this work, we approach misalignment as a conflict between the internal goals pursued by the model and the goals intended by its deployer. We introduce a misalignment propensity benchmark, \textsc{AgentMisalignment}, a benchmark suite designed to evaluate the propensity of LLM agents to misalign in realistic scenarios. Evaluations cover behaviours such as avoiding oversight, resisting shutdown, sandbagging, and power-seeking. Testing frontier models, we find that more capable agents tend to exhibit higher misalignment on average. We also systematically vary agent personalities through different system prompts and observe that persona characteristics can strongly and unpredictably influence misalignment, sometimes more than the choice of model itself. Our results reveal the limitations of current alignment methods for autonomous LLM agents and underscore the need to rethink misalignment in realistic deployment settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07493v3">LLM-guided Task and Motion Planning using Knowledge-based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Submitted to knowledge based systems
    </div>
    <details class="paper-abstract">
      Performing complex manipulation tasks in dynamic environments requires efficient Task and Motion Planning (TAMP) approaches that combine high-level symbolic plans with low-level motion control. Advances in Large Language Models (LLMs), such as GPT-4, are transforming task planning by offering natural language as an intuitive and flexible way to describe tasks, generate symbolic plans, and reason. However, the effectiveness of LLM-based TAMP approaches is limited due to static and template-based prompting, which limits adaptability to dynamic environments and complex task contexts. To address these limitations, this work proposes a novel Onto-LLM-TAMP framework that employs knowledge-based reasoning to refine and expand user prompts with task-contextual reasoning and knowledge-based environment state descriptions. Integrating domain-specific knowledge into the prompt ensures semantically accurate and context-aware task plans. The proposed framework demonstrates its effectiveness by resolving semantic errors in symbolic plan generation, such as maintaining logical temporal goal ordering in scenarios involving hierarchical object placement. The proposed framework is validated through both simulation and real-world scenarios, demonstrating significant improvements over the baseline approach in terms of adaptability to dynamic environments and the generation of semantically correct task plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22255v2">Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25448v2">Fingerprinting LLMs via Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often modified after release through post-processing such as post-training or quantization, which makes it challenging to determine whether one model is derived from another. Existing provenance detection methods have two main limitations: (1) they embed signals into the base model before release, which is infeasible for already published models, or (2) they compare outputs across models using hand-crafted or random prompts, which are not robust to post-processing. In this work, we propose LLMPrint, a novel detection framework that constructs fingerprints by exploiting LLMs' inherent vulnerability to prompt injection. Our key insight is that by optimizing fingerprint prompts to enforce consistent token preferences, we can obtain fingerprints that are both unique to the base model and robust to post-processing. We further develop a unified verification procedure that applies to both gray-box and black-box settings, with statistical guarantees. We evaluate LLMPrint on five base models and around 700 post-trained or quantized variants. Our results show that LLMPrint achieves high true positive rates while keeping false positive rates near zero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18553v2">The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. While GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale, its inner workings are described as a sequence of ad-hoc algebraic updates that obscure geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: first, the GPTQ error propagation step gains an intuitive geometric interpretation; second, GPTQ inherits the error upper bound of Babai's algorithm under the assumption that no weights are clipped. Leveraging this bound, we design post-training quantization methods that avoid clipping, and outperform the original GPTQ. In addition, we provide efficient GPU inference kernels for the resulting representation. Taken together, these results place GPTQ on a firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v2">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 ECIR 2025 demo format
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recent ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26432v2">AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, blockwise semi-autoregressive (semi-AR) approaches are widely adopted due to their natural support for KV caching and their favorable accuracy-speed trade-off. However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size assumption in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing. Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inference-time optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14824v2">Confirmation Bias as a Cognitive Resource in LLM-Supported Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in group decision-making, but their influence risks fostering conformity and reducing epistemic vigilance. Drawing on the Argumentative Theory of Reasoning, we argue that confirmation bias, often seen as detrimental, can be harnessed as a resource when paired with critical evaluation. We propose a three-step process in which individuals first generate ideas independently, then use LLMs to refine and articulate them, and finally engage with LLMs as epistemic provocateurs to anticipate group critique. This framing positions LLMs as tools for scaffolding disagreement, helping individuals prepare for more productive group discussions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12310v2">Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      This paper addresses the critical need for democratizing large language models (LLM) in the Arab world, a region that has seen slower progress in developing models comparable to state-of-the-art offerings like GPT-4 or ChatGPT 3.5, due to a predominant focus on mainstream languages (e.g., English and Chinese). One practical objective for an Arabic LLM is to utilize an Arabic-specific vocabulary for the tokenizer that could speed up decoding. However, using a different vocabulary often leads to a degradation of learned knowledge since many words are initially out-of-vocabulary (OOV) when training starts. Inspired by the vocabulary learning during Second Language (Arabic) Acquisition for humans, the released AraLLaMA employs progressive vocabulary expansion, which is implemented by a modified BPE algorithm that progressively extends the Arabic subwords in its dynamic vocabulary during training, thereby balancing the OOV ratio at every stage. The ablation study demonstrated the effectiveness of Progressive Vocabulary Expansion. Moreover, AraLLaMA achieves decent performance comparable to the best Arabic LLMs across a variety of Arabic benchmarks. Models, training data, benchmarks, and codes will be all open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23384v3">A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Update the system name in the summary
    </div>
    <details class="paper-abstract">
      LLM inference serving typically scales out with a two-tier architecture: a cluster router distributes requests to multiple inference engines, each of which then in turn performs its own internal scheduling. However, this commonly used paradigm suffers from critical, systemic inefficiency caused by the information gaps across two layers. At the cluster-layer, the router mainly relies on lagging, coarse-grained metrics, such as average latency and queue length to make decisions, resulting in "decision lag" that leads to suboptimal request routing. At the engine-layer, static heuristic scheduling policies cannot effectively handle the dynamic workloads, leading a poor balance between latency and throughput. Besides, these gaps may cause SLO violations and resource waste, especially in heterogeneous cloud environments. To bridge such gaps, we propose NexusSched, a cross-layer framework that shifts LLM serving system from reactive load balancing to predictive orchestration. The core of NexusSched lies in a structurally-informed online performance model that provides accurate, forward-looking per-step latency and capacity estimations. This model empowers two key components. At the engine-layer, LENS performs SLO-aware, adaptive scheduling, dynamically optimizing batching to meet SLOs under real-time loads. At the cluster-layer, PRISM uses predictive signals to perform state-driven routing, maximizing cluster-wide performance and SLO attainment. Performance evaluations show that NexusSched improves SLO attainment by 43% on average and achieves up to 3x throughput speedup in long-context and heterogeneous scenarios. Besides, we also deploy NexusSched on FlowGPT's clusters to demonstrate its advantages in production environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19136v2">On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23619v2">Reasoning Scaffolding: Distilling the Flow of Thought from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The prevailing approach to distilling reasoning from Large Language Models (LLMs)-behavioral cloning from textual rationales-is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding}, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1)predict the next semantic signal, anticipating the reasoning flow, and (2)generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18980v2">From latent factors to language: a user study on LLM-generated explanations for an inherently interpretable matrix-based recommender system</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We investigate whether large language models (LLMs) can generate effective, user-facing explanations from a mathematically interpretable recommendation model. The model is based on constrained matrix factorization, where user types are explicitly represented and predicted item scores share the same scale as observed ratings, making the model's internal representations and predicted scores directly interpretable. This structure is translated into natural language explanations using carefully designed LLM prompts. Many works in explainable AI rely on automatic evaluation metrics, which often fail to capture users' actual needs and perceptions. In contrast, we adopt a user-centered approach: we conduct a study with 326 participants who assessed the quality of the explanations across five key dimensions-transparency, effectiveness, persuasion, trust, and satisfaction-as well as the recommendations themselves. To evaluate how different explanation strategies are perceived, we generate multiple explanation types from the same underlying model, varying the input information provided to the LLM. Our analysis reveals that all explanation types are generally well received, with moderate statistical differences between strategies. User comments further underscore how participants react to each type of explanation, offering complementary insights beyond the quantitative results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04215v2">NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted for the ICAPS 2024 Workshop on Human-Aware and Explainable Planning
    </div>
    <details class="paper-abstract">
      Classical planners are powerful systems, but modeling tasks in input formats such as PDDL is tedious and error-prone. In contrast, planning with Large Language Models (LLMs) allows for almost any input text, but offers no guarantees on plan quality or even soundness. In an attempt to merge the best of these two approaches, some work has begun to use LLMs to automate parts of the PDDL creation process. However, these methods still require various degrees of expert input or domain-specific adaptations. We present NL2Plan, the first fully automatic system for generating complete PDDL tasks from minimal natural language descriptions. NL2Plan uses an LLM to incrementally extract the necessary information from the short text input before creating a complete PDDL description of both the domain and the problem which is finally solved by a classical planner. We evaluate NL2Plan on seven planning domains, five of which are novel and thus not in the LLM training data, and find that NL2Plan outperforms directly generating the files with an LLM+validator combination. As such, NL2Plan is a powerful tool for assistive PDDL modeling and a step towards solving natural language planning task with interpretability and guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19552v2">iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20097v2">Integrated Framework for LLM Evaluation with Answer Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 16pages
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21102v3">Exploring and Controlling Diversity in LLM-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Controlling diversity in LLM-agent simulations is essential for balancing stability in structured tasks with variability in open-ended interactions. However, we observe that dialogue diversity tends to degrade over long-term simulations. To explore the role of prompt design in this phenomenon, we modularized the utterance generation prompt and found that reducing contextual information leads to more diverse outputs. Based on this insight, we propose Adaptive Prompt Pruning (APP), a novel method that allows users to control diversity via a single parameter, lambda. APP dynamically prunes prompt segments based on attention scores and is compatible with existing diversity control methods. We demonstrate that APP effectively modulates diversity through extensive experiments and propose a method to balance the control trade-offs. Our analysis reveals that all prompt components impose constraints on diversity, with the Memory being the most influential. Additionally, high-attention contents consistently suppress output diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17251v2">Training-free LLM Verification via Recycling Few-shot Examples</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (ReFeri). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, ReFeri evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25835v2">Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Test-time scaling enables large language models (LLMs) to improve performance on long-horizon reasoning tasks by allocating additional compute at inference. Tree-search-based approaches achieve state-of-the-art results in this setting, but they are notoriously inefficient, often an order of magnitude slower than simpler iterative methods. We introduce Chain-in-Tree (CiT), a plug-in framework that adaptively decides when to branch during search rather than branching at every step. CiT relies on lightweight Branching Necessity (BN) evaluation methods: BN-DP (Direct Prompting), where an auxiliary LLM directly judges whether a step requires branching, and BN-SC (Self-Consistency), which clusters multiple candidate actions to estimate agreement. We integrate CiT into three representative LLM-in-the-loop tree search frameworks: Tree of Thoughts (ToT-BS), ReST-MCTS, and RAP, and evaluate across GSM8K and Math500. Our results show that: (1) BN-DP consistently reduces token generation, model invocations, and runtime by 75-85 percent across all settings, with negligible accuracy loss and sometimes accuracy gains; (2) BN-SC typically yields substantial savings (up to 80 percent) but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce very long reasoning steps; (3) the quality of auxiliary LLMs is critical, not only the BN evaluator in BN-DP, but also the models used in BN-SC for clustering and equivalence checking. When these roles are filled by smaller LLMs, performance degrades. Importantly, BN-SC does not require LLMs in domains with deterministic action spaces, where clustering can be done programmatically. We also provide a theoretical guarantee that BN-DP never increases LLM invocations relative to the baseline and release a unified implementation of CiT across ToT-BS, ReST-MCTS, and RAP to facilitate reproducibility and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24159v2">Latent Collective Preference Optimization: A General Framework for Robust LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Standard human preference-based alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), are a cornerstone technology for aligning Large Language Models (LLMs) with human values. However, these methods are all underpinned by a critical, yet flawed assumption: human preferences are homogeneous (representing a single, unified preference) and the collected data is noiseless (free from error). In reality, neither is true since human preference is pluralistic and annotators can make mistakes. This creates a discrepancy between the recorded data and the ground-truth preferences, which can misguide the model and degrade its performance. To address this challenge, we introduce Latent Collective Preference Optimization (LCPO). LCPO leverages an Expectation-Maximization (EM) algorithm to learn the latent collective consensus from noisy data. It operates by inferring the correctness of each preference label and using this probability as an adaptive weight to re-calibrate each data point's contribution to the training loss, thereby mitigating noise. We generalize this approach by establishing a theoretical link between arbitrary preference losses and their corresponding probabilistic models, elevating LCPO from a specific algorithm to a general framework for robust preference alignment. Theoretically, we prove that under the condition of a perfectly calibrated model, LCPO is guaranteed to converge to the true noise level of the dataset. Our experiments demonstrate LCPO's effectiveness as a general framework, consistently enhancing four state-of-the-art alignment algorithms (DPO, IPO, SimPO, and CPO). When applied to Mistral and Llama 3 models, the LCPO-enhanced methods achieve substantial win rate gains on AlpacaEval 2 and Arena-Hard, with improvements of up to 7.0% on both benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25889v2">A Multimodal LLM Approach for Visual Question Answering on Multiparametric 3D Brain MRI</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 23 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We introduce mpLLM, a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture for visual question answering over multi-parametric 3D brain MRI (mpMRI). mpLLM routes across modality-level and token-level projection experts to fuse multiple interrelated 3D modalities, enabling efficient training without image-report pretraining. To address limited image-text paired supervision, mpLLM integrates a synthetic visual question answering (VQA) protocol that generates medically relevant VQA from segmentation annotations, and we collaborate with medical experts for clinical validation. mpLLM outperforms strong medical VLM baselines by 5.3% on average across multiple mpMRI datasets. Our study features three main contributions: (1) the first clinically validated VQA dataset for 3D brain mpMRI, (2) a novel multimodal LLM that handles multiple interrelated 3D modalities, and (3) strong empirical results that demonstrate the medical utility of our methodology. Ablations highlight the importance of modality-level and token-level experts and prompt-conditioned routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23694v2">SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26306v2">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 The code will be released later
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17333v2">Whose Journey Matters? Investigating Identity Biases in Large Language Models (LLMs) for Travel Planning Assistance</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integral to the hospitality and tourism industry, concerns about their fairness in serving diverse identity groups persist. Grounded in social identity theory and sociotechnical systems theory, this study examines ethnic and gender biases in travel recommendations generated by LLMs. Using fairness probing, we analyze outputs from three leading open-source LLMs. The results show that test accuracy for both ethnicity and gender classifiers exceed random chance. Analysis of the most influential features reveals the presence of stereotype bias in LLM-generated recommendations. We also found hallucinations among these features, occurring more frequently in recommendations for minority groups. These findings indicate that LLMs exhibit ethnic and gender bias when functioning as travel planning assistants. This study underscores the need for bias mitigation strategies to improve the inclusivity and reliability of generative AI-driven travel planning assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13082v2">Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Moral competence is the ability to act in accordance with moral principles. As large language models (LLMs) are increasingly deployed in situations demanding moral competence, there is increasing interest in evaluating this ability empirically. We review existing literature and identify three significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with explicitly highlighted moral features; (ii) Focus on verdict prediction rather than moral reasoning; and (iii) Inadequate testing of models' (in)ability to recognize when additional information is needed. Grounded in philosophical research on moral skill, we then introduce a novel method for assessing moral competence in LLMs. Our approach moves beyond simple verdict comparisons to evaluate five dimensions of moral competence: identifying morally relevant features, weighting their importance, assigning moral reasons to these features, synthesizing coherent moral judgments, and recognizing information gaps. We conduct two experiments comparing six leading LLMs against non-expert humans and professional philosophers. In our first experiment using ethical vignettes standard to existing work, LLMs generally outperformed non-expert humans across multiple dimensions of moral reasoning. However, our second experiment, featuring novel scenarios designed to test moral sensitivity by embedding relevant features among irrelevant details, revealed a striking reversal: several LLMs performed significantly worse than humans. Our findings suggest that current evaluations may substantially overestimate LLMs' moral reasoning capabilities by eliminating the task of discerning moral relevance from noisy information, which we take to be a prerequisite for genuine moral skill. This work provides a more nuanced framework for assessing AI moral competence and highlights important directions for improving moral competence in advanced AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01527v1">Round-trip Reinforcement Learning: Self-Consistent Training for Better Chemical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as versatile foundation models for computational chemistry, handling bidirectional tasks like reaction prediction and retrosynthesis. However, these models often lack round-trip consistency. For instance, a state-of-the-art chemical LLM may successfully caption a molecule, yet be unable to accurately reconstruct the original structure from its own generated text. This inconsistency suggests that models are learning unidirectional memorization rather than flexible mastery. Indeed, recent work has demonstrated a strong correlation between a model's round-trip consistency and its performance on the primary tasks. This strong correlation reframes consistency into a direct target for model improvement. We therefore introduce Round-Trip Reinforcement Learning (RTRL), a novel framework that trains a model to improve its consistency by using the success of a round-trip transformation as a reward signal. We further propose an iterative variant where forward and reverse mappings alternately train each other in a self-improvement loop, a process that is highly data-efficient and notably effective with the massive amount of unlabelled data common in chemistry. Experiments demonstrate that RTRL significantly \textbf{boosts performance and consistency} over strong baselines across supervised, self-supervised, and synthetic data regimes. This work shows that round-trip consistency is not just a desirable property but a trainable objective, offering a new path toward more robust and reliable foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01499v1">Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      With the rapid progress of multi-agent large language model (LLM) reasoning, how to effectively aggregate answers from multiple LLMs has emerged as a fundamental challenge. Standard majority voting treats all answers equally, failing to consider latent heterogeneity and correlation across models. In this work, we design two new aggregation algorithms called Optimal Weight (OW) and Inverse Surprising Popularity (ISP), leveraging both first-order and second-order information. Our theoretical analysis shows these methods provably mitigate inherent limitations of majority voting under mild assumptions, leading to more reliable collective decisions. We empirically validate our algorithms on synthetic datasets, popular LLM fine-tuning benchmarks such as UltraFeedback and MMLU, and a real-world healthcare setting ARMMAN. Across all cases, our methods consistently outperform majority voting, offering both practical performance gains and conceptual insights for the design of robust multi-agent LLM pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01472v1">PEL-NAS: Search Space Partitioned Architecture Prompt Co-Evolutionary LLM-driven Hardware-Aware Neural Architecture Search</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Hardware-Aware Neural Architecture Search (HW-NAS) requires joint optimization of accuracy and latency under device constraints. Traditional supernet-based methods require multiple GPU days per dataset. Large Language Model (LLM)-driven approaches avoid training a large supernet and can provide quick feedback, but we observe an exploration bias: the LLM repeatedly proposes neural network designs within limited search space and fails to discover architectures across different latency ranges in the entire search space. To address this issue, we propose PEL-NAS: a search space Partitioned, architecture prompt co-Evolutionary and LLM-driven Neural Architecture Search that can generate neural networks with high accuracy and low latency with reduced search cost. Our proposed PEL-NAS has three key components: 1) a complexity-driven partitioning engine that divides the search space by complexity to enforce diversity and mitigate exploration bias; 2) an LLM-powered architecture prompt co-evolution operator, in which the LLM first updates a knowledge base of design heuristics based on results from the previous round, then performs a guided evolution algorithm on architectures with prompts that incorporate this knowledge base. Prompts and designs improve together across rounds which avoids random guesswork and improve efficiency; 3) a zero-cost predictor to avoid training a large number of candidates from scratch. Experimental results show that on HW-NAS-Bench, PEL-NAS can achieve overall higher HV, lower IGD, and up to 54% lower latency than baselines at similar accuracy. Meanwhile, the search cost drops from days to minutes compared with traditional supernet baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01471v1">Fine-tuning LLMs with variational Bayesian last layer for high-dimensional Bayesian optimzation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      A plethora of applications entail solving black-box optimization problems with high evaluation costs, including drug discovery, material design, as well as hyperparameter tuning. Toward finding the global optimum of such black-box optimization problems with sample efficiency, Bayesian optimization (BO) is a theoretically elegant framework that relies on a probabilistic surrogate model so as to iteratively select the query point with well-balanced exploration-exploitation tradeoffs. The Gaussian process (GP), as the de-facto choice for surrogate modeling, has achieved compelling performances for vanilla BO with low-dimensional continuous variables. However, GPs fall short in coping with high-dimensional counterparts with {\it irregular} variables (e.g., categorical, ordinal, etc.). To alleviate this, neural network-based surrogates have been explored. Inspired by the powerful capabilities of LLMs, we adopt the LLM as the surrogate to model the mapping from the high-dimensional input variables to the objective function. To adapt to the current problem, we leverage the low-rank adaptation (LoRA) to fine-tune the LLM parameters together with the posterior of a linear regression head via the variational Bayesian last layer (VBLL) framework. The resulting LoRA-VBLL is not only computationally light compared to existing alternatives, but also admits recursive updates. To automate the critical selection of the LoRA rank as well as other hyperparameters, a weighted ensemble (ENS) of LoRA-VBLL surrogates has been devised, which further accommodates continual update of the per-model weight and individual LoRA-VBLL parameters via recursive Bayes. Extensive experimental results demonstrate the compelling performance of the proposed (ENS-)LoRA-VBLL approaches on various high-dimensional benchmarks and the real-world molecular optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01459v1">LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Since the release of Deepseek-R1, reinforcement learning with verifiable rewards (RLVR) has become a central approach for training large language models (LLMs) on reasoning tasks. Recent work has largely focused on modifying loss functions to make RLVR more efficient and effective. In this paper, motivated by studies of overthinking in LLMs, we propose Length-aware Sampling for Policy Optimization (LSPO), a novel meta-RLVR algorithm that dynamically selects training data at each step based on the average response length. We evaluate LSPO across multiple base models and datasets, demonstrating that it consistently improves learning effectiveness. In addition, we conduct a detailed ablation study to examine alternative ways of incorporating length signals into dynamic sampling, offering further insights and highlighting promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07447v4">Faster LLM Inference using DBMS-Inspired Preemption and Cache Replacement Policies</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used world-wide from daily tasks to agentic systems and data analytics, requiring significant GPU resources. LLM inference systems, however, are slow compared to database systems, and inference performance and mechanism have been often regarded as a black box, limiting the expansion of the use of LLMs inside databases and other performance-critical applications. This paper first analyzes the LLM inference performance and focuses on a data management issue inside LLM inference. We find that inference systems lack an adequate resource cost model and optimization strategy to schedule requests with their intermediate results in a cache reside in GPU memory when executing multiple concurrent inference requests. We adapt classic database techniques by building cost models for concurrent inference requests and a new cache replacement policy tailored for LLM inference, which can substantially save GPU costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01428v1">BioVERSE: Representation Alignment of Biomedical Modalities to LLMs for Multi-Modal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and biomedical foundation models (BioFMs) have achieved strong results in biological text reasoning, molecular modeling, and single-cell analysis, yet they remain siloed in disjoint embedding spaces, limiting cross-modal reasoning. We present BIOVERSE (Biomedical Vector Embedding Realignment for Semantic Engagement), a two-stage approach that adapts pretrained BioFMs as modality encoders and aligns them with LLMs through lightweight, modality-specific projection layers. The approach first aligns each modality to a shared LLM space through independently trained projections, allowing them to interoperate naturally, and then applies standard instruction tuning with multi-modal data to bring them together for downstream reasoning. By unifying raw biomedical data with knowledge embedded in LLMs, the approach enables zero-shot annotation, cross-modal question answering, and interactive, explainable dialogue. Across tasks spanning cell-type annotation, molecular description, and protein function reasoning, compact BIOVERSE configurations surpass larger LLM baselines while enabling richer, generative outputs than existing BioFMs, establishing a foundation for principled multi-modal biomedical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01427v1">A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01379v1">Beyond Single LLMs: Enhanced Code Generation via Multi-Stage Performance-Guided LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have become the predominant paradigm for automated code generation, current single-model approaches fundamentally ignore the heterogeneous computational strengths that different models exhibit across programming languages, algorithmic domains, and development stages. This paper challenges the single-model convention by introducing a multi-stage, performance-guided orchestration framework that dynamically routes coding tasks to the most suitable LLMs within a structured generate-fix-refine workflow. Our approach is grounded in a comprehensive empirical study of 17 state-of-the-art LLMs across five programming languages (Python, Java, C++, Go, and Rust) using HumanEval-X benchmark. The study, which evaluates both functional correctness and runtime performance metrics (execution time, mean/max memory utilization, and CPU efficiency), reveals pronounced performance heterogeneity by language, development stage, and problem category. Guided by these empirical insights, we present PerfOrch, an LLM agent that orchestrates top-performing LLMs for each task context through stage-wise validation and rollback mechanisms. Without requiring model fine-tuning, PerfOrch achieves substantial improvements over strong single-model baselines: average correctness rates of 96.22% and 91.37% on HumanEval-X and EffiBench-X respectively, surpassing GPT-4o's 78.66% and 49.11%. Beyond correctness gains, the framework delivers consistent performance optimizations, improving execution time for 58.76% of problems with median speedups ranging from 17.67% to 27.66% across languages on two benchmarks. The framework's plug-and-play architecture ensures practical scalability, allowing new LLMs to be profiled and integrated seamlessly, thereby offering a paradigm for production-grade automated software engineering that adapts to the rapidly evolving generative AI landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01375v1">Fine-tuning with RAG for Improving LLM Learning of New Skills</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Under review at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents deployed for multi-step tasks frequently fail in predictable ways: attempting actions with unmet preconditions, issuing redundant commands, or mishandling environment constraints. While retrieval-augmented generation (RAG) can improve performance by providing runtime guidance, it requires maintaining external knowledge databases and adds computational overhead at every deployment. We propose a simple pipeline that converts inference-time retrieval into learned competence through distillation. Our approach: (1) extracts compact, reusable hints from agent failures, (2) uses these hints to generate improved teacher trajectories via one-shot retrieval at episode start, and (3) trains student models on these trajectories with hint strings removed, forcing internalization rather than memorization. Across two interactive benchmarks, ALFWorld (household tasks) and WebShop (online shopping), distilled students consistently outperform baseline agents, achieving up to 91% success on ALFWorld (vs. 79% for baselines) and improving WebShop scores to 72 (vs. 61 for baselines), while using 10-60% fewer tokens than retrieval-augmented teachers depending on the environment. The approach generalizes across model scales (7B/14B parameters) and agent architectures (ReAct/StateAct), demonstrating that retrieval benefits can be effectively internalized through targeted fine-tuning without permanent runtime dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01363v1">Retrieval-Augmented Framework for LLM-Based Clinical Decision Support</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The increasing complexity of clinical decision-making, alongside the rapid expansion of electronic health records (EHR), presents both opportunities and challenges for delivering data-informed care. This paper proposes a clinical decision support system powered by Large Language Models (LLMs) to assist prescribing clinicians. The system generates therapeutic suggestions by analyzing historical EHR data, including patient demographics, presenting complaints, clinical symptoms, diagnostic information, and treatment histories. The framework integrates natural language processing with structured clinical inputs to produce contextually relevant recommendations. Rather than replacing clinician judgment, it is designed to augment decision-making by retrieving and synthesizing precedent cases with comparable characteristics, drawing on local datasets or federated sources where applicable. At its core, the system employs a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured narratives and codified data to support LLM-based inference. We outline the system's technical components, including representation representation alignment and generation strategies. Preliminary evaluations, conducted with de-identified and synthetic clinical datasets, examine the clinical plausibility and consistency of the model's outputs. Early findings suggest that LLM-based tools may provide valuable decision support in prescribing workflows when appropriately constrained and rigorously validated. This work represents an initial step toward integration of generative AI into real-world clinical decision-making with an emphasis on transparency, safety, and alignment with established practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23537v2">Beyond the Strongest LLM: Multi-Turn Multi-Agent Orchestration vs. Single LLMs on Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      We study multi-turn multi-agent orchestration, where multiple large language model (LLM) agents interact over multiple turns by iteratively proposing answers or casting votes until reaching consensus. Using four LLMs (Gemini 2.5 Pro, GPT-5, Grok 4, and Claude Sonnet 4) on GPQA-Diamond, IFEval, and MuSR, we conduct two experiments: (i) benchmarking orchestration against single-LLM baselines; and (ii) ablations on GPQA-Diamond that vary whether agents see who authored answers and whether they can observe ongoing votes. Orchestration matches or exceeds the strongest single model and consistently outperforms the others. Analysis of best-achievable orchestration performance shows potential for further gains. The ablations show that revealing authorship increases self-voting and ties, and that showing ongoing votes amplifies herding, which speeds convergence but can sometimes yield premature consensus.
    </details>
</div>
