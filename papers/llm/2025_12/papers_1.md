# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.25065v1">Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 27 pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Resource-management tasks in modern operating and distributed systems continue to rely primarily on hand-designed heuristics for tasks such as scheduling, caching, or active queue management. Designing performant heuristics is an expensive, time-consuming process that we are forced to continuously go through due to the constant flux of hardware, workloads and environments. We propose a new alternative: synthesizing instance-optimal heuristics -- specialized for the exact workloads and hardware where they will be deployed -- using code-generating large language models (LLMs). To make this synthesis tractable, Vulcan separates policy and mechanism through LLM-friendly, task-agnostic interfaces. With these interfaces, users specify the inputs and objectives of their desired policy, while Vulcan searches for performant policies via evolutionary search over LLM-generated code. This interface is expressive enough to capture a wide range of system policies, yet sufficiently constrained to allow even small, inexpensive LLMs to generate correct and executable code. We use Vulcan to synthesize performant heuristics for cache eviction and memory tiering, and find that these heuristics outperform all human-designed state-of-the-art algorithms by upto 69% and 7.9% in performance for each of these tasks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.25055v1">Context-aware LLM-based AI Agents for Human-centered Energy Management Systems in Smart Buildings</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      This study presents a conceptual framework and a prototype assessment for Large Language Model (LLM)-based Building Energy Management System (BEMS) AI agents to facilitate context-aware energy management in smart buildings through natural language interaction. The proposed framework comprises three modules: perception (sensing), central control (brain), and action (actuation and user interaction), forming a closed feedback loop that captures, analyzes, and interprets energy data to respond intelligently to user queries and manage connected appliances. By leveraging the autonomous data analytics capabilities of LLMs, the BEMS AI agent seeks to offer context-aware insights into energy consumption, cost prediction, and device scheduling, thereby addressing limitations in existing energy management systems. The prototype's performance was evaluated using 120 user queries across four distinct real-world residential energy datasets and different evaluation metrics, including latency, functionality, capability, accuracy, and cost-effectiveness. The generalizability of the framework was demonstrated using ANOVA tests. The results revealed promising performance, measured by response accuracy in device control (86%), memory-related tasks (97%), scheduling and automation (74%), and energy analysis (77%), while more complex cost estimation tasks highlighted areas for improvement with an accuracy of 49%. This benchmarking study moves toward formalizing the assessment of LLM-based BEMS AI agents and identifying future research directions, emphasizing the trade-off between response accuracy and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02761v4">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24968v1">The Impact of LLMs on Online News Consumption and Production</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) change how consumers acquire information online; their bots also crawl news publishers' websites for training data and to answer consumer queries; and they provide tools that can lower the cost of content creation. These changes lead to predictions of adverse impact on news publishers in the form of lowered consumer demand, reduced demand for newsroom employees, and an increase in news "slop." Consequently, some publishers strategically responded by blocking LLM access to their websites using the robots.txt file standard. Using high-frequency granular data, we document four effects related to the predicted shifts in news publishing following the introduction of generative AI (GenAI). First, we find a consistent and moderate decline in traffic to news publishers occurring after August 2024. Second, using a difference-in-differences approach, we find that blocking GenAI bots can have adverse effects on large publishers by reducing total website traffic by 23% and real consumer traffic by 14% compared to not blocking. Third, on the hiring side, we do not find evidence that LLMs are replacing editorial or content-production jobs yet. The share of new editorial and content-production job listings increases over time. Fourth, regarding content production, we find no evidence that large publishers increased text volume; instead, they significantly increased rich content and use more advertising and targeting technologies. Together, these findings provide early evidence of some unforeseen impacts of the introduction of LLMs on news production and consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24947v1">CPJ: Explainable Agricultural Pest Diagnosis via Caption-Prompt-Judge with LLM-Judged Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 This paper is 6 pages in length and contains 2 figures. Tao Fang (Corresponding Author), Lina Lu (Co-corresponding Author)
    </div>
    <details class="paper-abstract">
      Accurate and interpretable crop disease diagnosis is essential for agricultural decision-making, yet existing methods often rely on costly supervised fine-tuning and perform poorly under domain shifts. We propose Caption--Prompt--Judge (CPJ), a training-free few-shot framework that enhances Agri-Pest VQA through structured, interpretable image captions. CPJ employs large vision-language models to generate multi-angle captions, refined iteratively via an LLM-as-Judge module, which then inform a dual-answer VQA process for both recognition and management responses. Evaluated on CDDMBench, CPJ significantly improves performance: using GPT-5-mini captions, GPT-5-Nano achieves \textbf{+22.7} pp in disease classification and \textbf{+19.5} points in QA score over no-caption baselines. The framework provides transparent, evidence-based reasoning, advancing robust and explainable agricultural diagnosis without fine-tuning. Our code and data are publicly available at: https://github.com/CPJ-Agricultural/CPJ-Agricultural-Diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24940v1">Iterative Deployment Improves Planning Skills in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      We show that iterative deployment of large language models (LLMs), each fine-tuned on data carefully curated by users from the previous models' deployment, can significantly change the properties of the resultant models. By testing this mechanism on various planning domains, we observe substantial improvements in planning skills, with later models displaying emergent generalization by discovering much longer plans than the initial models. We then provide theoretical analysis showing that iterative deployment effectively implements reinforcement learning (RL) training in the outer-loop (i.e. not as part of intentional model training), with an implicit reward function. The connection to RL has two important implications: first, for the field of AI safety, as the reward function entailed by repeated deployment is not defined explicitly, and could have unexpected implications to the properties of future model deployments. Second, the mechanism highlighted here can be viewed as an alternative training regime to explicit RL, relying on data curation rather than explicit rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24933v1">Adaptive Dependency-aware Prompt Optimization Framework for Multi-Step LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Multi-step LLM pipelines invoke large language models multiple times in a structured sequence and can effectively solve complex tasks, but their performance heavily depends on the prompts used at each step. Jointly optimizing these prompts is difficult due to missing step-level supervision and inter-step dependencies. Existing end-to-end prompt optimization methods struggle under these conditions and often yield suboptimal or unstable updates. We propose ADOPT, an Adaptive Dependency-aware Prompt Optimization framework for multi-step LLM pipelines. ADOPT explicitly models the dependency between each LLM step and the final task outcome, enabling precise text-gradient estimation analogous to computing analytical derivatives. It decouples textual gradient estimation from gradient updates, reducing multi-prompt optimization to flexible single-prompt optimization steps, and employs a Shapley-based mechanism to adaptively allocate optimization resources. Experiments on real-world datasets and diverse pipeline structures show that ADOPT is effective and robust, consistently outperforming state-of-the-art prompt optimization baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24867v1">Encyclo-K: Evaluating LLMs with Dynamically Composed Knowledge Statements</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Benchmarks play a crucial role in tracking the rapid advancement of large language models (LLMs) and identifying their capability boundaries. However, existing benchmarks predominantly curate questions at the question level, suffering from three fundamental limitations: vulnerability to data contamination, restriction to single-knowledge-point assessment, and reliance on costly domain expert annotation. We propose Encyclo-K, a statement-based benchmark that rethinks benchmark construction from the ground up. Our key insight is that knowledge statements, not questions, can serve as the unit of curation, and questions can then be constructed from them. We extract standalone knowledge statements from authoritative textbooks and dynamically compose them into evaluation questions through random sampling at test time. This design directly addresses all three limitations: the combinatorial space is too vast to memorize, and model rankings remain stable across dynamically generated question sets, enabling reliable periodic dataset refresh; each question aggregates 8-10 statements for comprehensive multi-knowledge assessment; annotators only verify formatting compliance without requiring domain expertise, substantially reducing annotation costs. Experiments on over 50 LLMs demonstrate that Encyclo-K poses substantial challenges with strong discriminative power. Even the top-performing OpenAI-GPT-5.1 achieves only 62.07% accuracy, and model performance displays a clear gradient distribution--reasoning models span from 16.04% to 62.07%, while chat models range from 9.71% to 50.40%. These results validate the challenges introduced by dynamic evaluation and multi-statement comprehensive understanding. These findings establish Encyclo-K as a scalable framework for dynamic evaluation of LLMs' comprehensive understanding over multiple fine-grained disciplinary knowledge statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.21740v2">Towards Operational Validation of LLM-Agent Social Simulations: A Replicated Study of a Reddit-like Technology Forum</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable generative social simulations that can capture culturally informed, norm-guided interaction on online social platforms. We build a technology community simulation modeled on Voat, a Reddit-like alt-right news aggregator and discussion platform active from 2014 to 2020. Using the YSocial framework, we seed the simulation with a fixed catalog of technology links sampled from Voat's shared URLs (covering 30+ domains) and calibrate parameters to Voat's v/technology using samples from the MADOC dataset. Agents use a base, uncensored model (Dolphin 3.0, based on Llama 3.1 8B) and concise personas (demographics, political leaning, interests, education, toxicity propensity) to generate posts, replies, and reactions under platform rules for link and text submissions, threaded replies and daily activity cycles. We run a 30-day simulation and evaluate operational validity by comparing distributions and structures with matched Voat data: activity patterns, interaction networks, toxicity, and topic coverage. Results indicate familiar online regularities: similar activity rhythms, heavy-tailed participation, sparse low-clustering interaction networks, core-periphery structure, topical alignment with Voat, and elevated toxicity. Limitations of the current study include the stateless agent design and evaluation based on a single 30-day run, which constrains external validity and variance estimates. The simulation generates realistic discussions, often featuring toxic language, primarily centered on technology topics such as Big Tech and AI. This approach offers a valuable method for examining toxicity dynamics and testing moderation strategies within a controlled environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08899v3">Toward Robust Legal Text Formalization into Defeasible Deontic Logic using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 This version is an extended version with additional results and discussion
    </div>
    <details class="paper-abstract">
      We present a comprehensive approach to the automated formalization of legal texts using large language models (LLMs), targeting their transformation into Defeasible Deontic Logic (DDL). Our method employs a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. We introduce a refined success metric that more precisely captures the completeness of formalizations, and a novel two-stage pipeline with a dedicated refinement step to improve logical consistency and coverage. The evaluation procedure has been strengthened with stricter error assessment, and we provide comparative results across multiple LLM configurations, including newly released models and various prompting and fine-tuning strategies. Experiments on legal norms from the Australian Telecommunications Consumer Protections Code demonstrate that, when guided effectively, LLMs can produce formalizations that align closely with expert-crafted representations, underscoring their potential for scalable legal informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24750v1">Analyzing Communication Predictability in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Effective communication is essential in distributed training, with predictability being one of its most significant characteristics. However, existing studies primarily focus on exploiting predictability through online profiling for runtime optimization, without a systematic understanding of it. In this work, we aim to systematically formulate communication predictability in distributed training, particularly in Large Language Models (LLMs) that utilize hybrid parallelism. Our analysis focuses on both traffic patterns and communication overhead. Specifically, we investigate predictable traffic patterns in typical LLMs and evaluate how various factors influence GPU utilization and effective bandwidth (two critical variables affecting communication overhead). Furthermore, we develop an analytical formulation to estimate communication overhead in LLM training, which is validated with high accuracy against empirical data. Leveraging this formulation, we propose a configuration tuning tool, ConfigTuner, to optimize training performance. Compared to Megatron-LM, the training configurations optimized by ConfigTuner demonstrate up to a 1.36$\times$ increase in throughput. Compared to Alpa, ConfigTuner generates the same configuration suggestion while significantly reducing the search complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24686v1">BatteryAgent: Synergizing Physics-Informed Interpretation with LLM Reasoning for Intelligent Battery Fault Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Fault diagnosis of lithium-ion batteries is critical for system safety. While existing deep learning methods exhibit superior detection accuracy, their "black-box" nature hinders interpretability. Furthermore, restricted by binary classification paradigms, they struggle to provide root cause analysis and maintenance recommendations. To address these limitations, this paper proposes BatteryAgent, a hierarchical framework that integrates physical knowledge features with the reasoning capabilities of Large Language Models (LLMs). The framework comprises three core modules: (1) A Physical Perception Layer that utilizes 10 mechanism-based features derived from electrochemical principles, balancing dimensionality reduction with physical fidelity; (2) A Detection and Attribution Layer that employs Gradient Boosting Decision Trees and SHAP to quantify feature contributions; and (3) A Reasoning and Diagnosis Layer that leverages an LLM as the agent core. This layer constructs a "numerical-semantic" bridge, combining SHAP attributions with a mechanism knowledge base to generate comprehensive reports containing fault types, root cause analysis, and maintenance suggestions. Experimental results demonstrate that BatteryAgent effectively corrects misclassifications on hard boundary samples, achieving an AUROC of 0.986, which significantly outperforms current state-of-the-art methods. Moreover, the framework extends traditional binary detection to multi-type interpretable diagnosis, offering a new paradigm shift from "passive detection" to "intelligent diagnosis" for battery safety management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13940v2">Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Code: https://github.com/EnVision-Research/MTI
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +9.28% average improvement on six benchmarks for DeepSeek-R1-7B and +11.25% on AIME2024 using Ling-mini-2.0-while remaining highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17884v3">When Intelligence Fails: An Empirical Study on Why LLMs Struggle with Password Cracking</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of Large Language Models (LLMs) in natural language understanding and generation have sparked interest in their potential for cybersecurity applications, including password guessing. In this study, we conduct an empirical investigation into the efficacy of pre-trained LLMs for password cracking using synthetic user profiles. Specifically, we evaluate the performance of state-of-the-art open-source LLMs such as TinyLLaMA, Falcon-RW-1B, and Flan-T5 by prompting them to generate plausible passwords based on structured user attributes (e.g., name, birthdate, hobbies). Our results, measured using Hit@1, Hit@5, and Hit@10 metrics under both plaintext and SHA-256 hash comparisons, reveal consistently poor performance, with all models achieving less than 1.5% accuracy at Hit@10. In contrast, traditional rule-based and combinator-based cracking methods demonstrate significantly higher success rates. Through detailed analysis and visualization, we identify key limitations in the generative reasoning of LLMs when applied to the domain-specific task of password guessing. Our findings suggest that, despite their linguistic prowess, current LLMs lack the domain adaptation and memorization capabilities required for effective password inference, especially in the absence of supervised fine-tuning on leaked password datasets. This study provides critical insights into the limitations of LLMs in adversarial contexts and lays the groundwork for future efforts in secure, privacy-preserving, and robust password modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24618v1">Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 57 pages, 26 figures
    </div>
    <details class="paper-abstract">
      We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24609v1">Reinforcement Learning-Augmented LLM Agents for Collaborative Decision Making and Performance Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Accepted by IEEE ICFTIC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perform well in language tasks but often lack collaborative awareness and struggle to optimize global performance in multi-agent settings. We present a reinforcement learning-augmented LLM agent framework that formulates cooperation as a decentralized partially observable Markov decision process (Dec-POMDP) and adopts centralized training with decentralized execution (CTDE). We introduce Group Relative Policy Optimization (GRPO) to jointly optimize agent policies with access to global signals during training, together with a simplified joint reward that balances task quality, speed, and coordination cost. On collaborative writing and coding benchmarks, our framework delivers a 3x increase in task processing speed over single-agent baselines, 98.7% structural/style consistency in writing, and a 74.6% test pass rate in coding. The approach consistently outperforms strong multi-agent LLM baselines and provides a practical path toward reliable collaboration in complex workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13886v2">OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Post-training model pruning is a promising solution, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT offer a practical trade-off in optimality by applying efficient, approximate heuristic weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. Solving these QPs yields the per-row globally optimal update with respect to the reconstruction objective given the estimated Hessian. The shared-Hessian structure makes the problem highly amenable to batching on accelerators. We implement an accelerator-friendly QP solver that accumulates one Hessian per layer and solves many small QPs in parallel, enabling one-shot post-training pruning at scale on a single accelerator without fine-tuning. OPTIMA integrates with existing mask selectors and consistently improves zero-shot performance across multiple LLM families and sparsity regimes, yielding up to 3.97% absolute accuracy improvement. On an NVIDIA H100, OPTIMA prunes a 8B-parameter transformer end-to-end in 40 hours with 60GB peak memory. Together, these results set a new state-of-the-art accuracy-efficiency trade-offs for one-shot post-training pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24570v1">On the Effectiveness of Training Data Optimization for LLM-based Code Generation: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in code generation, largely driven by the availability of high-quality code datasets for effective training. To further improve data quality, numerous training data optimization techniques have been proposed; however, their overall effectiveness has not been systematically evaluated. To bridge this gap, we conduct the first large-scale empirical study, examining five widely-used training data optimization techniques and their pairwise combinations for LLM-based code generation across three benchmarks and four LLMs. Our results show that data synthesis is the most effective technique for improving functional correctness and reducing code smells, although it performs relatively worse on code maintainability compared to data refactoring, cleaning, and selection. Regarding combinations, we find that most combinations do not further improve functional correctness but can effectively enhance code quality (code smells and maintainability). Among all combinations, data synthesis combined with data refactoring achieves the strongest overall performance. Furthermore, our fine-grained analysis reinforces these findings and provides deeper insights into how individual techniques and their combinations influence code generation effectiveness. Overall, this work represents a first step toward a systematic understanding of training data optimization and combination strategies, offering practical guidance for future research and deployment in LLM-based code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24565v1">MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24562v1">HaluNet: Multi-Granular Uncertainty Modeling for Efficient Hallucination Detection in LLM Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at question answering (QA) but often generate hallucinations, including factual errors or fabricated content. Detecting hallucinations from internal uncertainty signals is attractive due to its scalability and independence from external resources. Existing methods often aim to accurately capture a single type of uncertainty while overlooking the complementarity among different sources, particularly between token-level probability uncertainty and the uncertainty conveyed by internal semantic representations, which provide complementary views on model reliability. We present \textbf{HaluNet}, a lightweight and trainable neural framework that integrates multi granular token level uncertainties by combining semantic embeddings with probabilistic confidence and distributional uncertainty. Its multi branch architecture adaptively fuses what the model knows with the uncertainty expressed in its outputs, enabling efficient one pass hallucination detection. Experiments on SQuAD, TriviaQA, and Natural Questions show that HaluNet delivers strong detection performance and favorable computational efficiency, with or without access to context, highlighting its potential for real time hallucination detection in LLM based QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24556v1">Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state of the art models (GPT-5.1, Gemini 3 Pro, and Claude 4.5 Opus) using HausaSafety, a novel adversarial dataset grounded in West African threat scenarios (e.g., Yahoo-Yahoo fraud, Dane gun manufacturing). Employing a 2 x 4 factorial design across 1,440 evaluations, we tested the non-linear interaction between language (English vs. Hausa) and temporal framing. Our results challenge the prevailing multilingual safety gap narrative. Instead of a simple degradation in low-resource settings, we identified a mechanism of Complex Interference where safety is determined by the intersection of variables. While models exhibited a Reverse Linguistic with Claude 4.5 Opus proving significantly safer in Hausa (45.0%) than in English (36.7%) due to uncertainty-driven refusal they suffered catastrophic failures in temporal reasoning. We report a profound Temporal Asymmetry, where past-tense framing bypassed defenses (15.6% safe) while future-tense scenarios triggered hyper-conservative refusals (57.2% safe). The magnitude of this volatility is illustrated by a 9.2x disparity between the safest and most vulnerable configurations, proving that safety is not a fixed property but a context-dependent state. We conclude that current models rely on superficial heuristics rather than robust semantic understanding, creating Safety Pockets that leave Global South users exposed to localized harms. We propose Invariant Alignment as a necessary paradigm shift to ensure safety stability across linguistic and temporal shifts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08990v2">Effective and Efficient Jailbreaks of Black-Box LLMs with Cross-Behavior Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Code is at https://github.com/gohil-vasudev/JCB
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs) and their alignment, they can still be jailbroken, i.e., harmful and toxic content can be elicited from them. While existing red-teaming methods have shown promise in uncovering such vulnerabilities, these methods struggle with limited success and high computational and monetary costs. To address this, we propose a black-box Jailbreak method with Cross-Behavior attacks (JCB), that can automatically and efficiently find successful jailbreak prompts. JCB leverages successes from past behaviors to help jailbreak new behaviors, thereby significantly improving the attack efficiency. Moreover, JCB does not rely on time- and/or cost-intensive calls to auxiliary LLMs to discover/optimize the jailbreak prompts, making it highly efficient and scalable. Comprehensive experimental evaluations show that JCB significantly outperforms related baselines, requiring up to 94% fewer queries while still achieving 12.9% higher average attack success. JCB also achieves a notably high 37% attack success rate on Llama-2-7B, one of the most resilient LLMs, and shows promising zero-shot transferability across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24532v1">From Building Blocks to Planning: Multi-Step Spatial Reasoning in LLMs with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Spatial reasoning in large language models (LLMs) has gained increasing attention due to applications in navigation and planning. Despite strong general language capabilities, LLMs still struggle with spatial transformations and multi-step planning in structured environments. We propose a two-stage approach that decomposes spatial reasoning into atomic building blocks and their composition. First, we apply supervised fine-tuning on elementary spatial transformations, such as rotation, translation, and scaling, to equip the model with basic spatial physics. We then freeze this physics-aware model and train lightweight LoRA adapters within the GRPO framework to learn policies that compose these building blocks for multi-step planning in puzzle-based environments, in a closed-loop manner. To support this pipeline, we synthesize an ASCII-art dataset and construct a corresponding ASCII-based reinforcement learning environment. Our method consistently outperforms baselines, including the generic backbone, physics-aware model, and end-to-end RL models, under both Dynamic environments with explicit state updates and Static environments where the model must rely on its internal state across steps. In addition, the proposed approach converges faster and exhibits more stable training compared to end-to-end reinforcement learning from scratch. Finally, we analyze attention patterns to assess whether fine-tuning induces meaningful improvements in spatial understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04481v2">Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 Camera-ready version of a paper accepted at the AIIDE 2025 Workshop on Experimental AI in Games (EXAG)
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation (PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24050v2">Collaborative Device-Cloud LLM Inference through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 We propose a unified post-training framework that integrates routing optimization, enabling the on-device LLM to improve its problem-solving ability while learning routing strategies
    </div>
    <details class="paper-abstract">
      Device-cloud collaboration has emerged as a promising paradigm for deploying large language models (LLMs), combining the efficiency of lightweight on-device inference with the superior performance of powerful cloud LLMs. An essential problem in this scenario lies in deciding whether a given query is best handled locally or delegated to the cloud. Existing approaches typically rely on external routers, implemented as binary classifiers, which often struggle to determine task difficulty from the prompt's surface pattern. To address these limitations, we propose a framework where the on-device LLM makes routing decisions at the end of its solving process, with this capability instilled through post-training. In particular, we formulate a reward maximization problem with carefully designed rewards that encourage effective problem solving and judicious offloading to the cloud. To solve this problem, we develop a group-adaptive policy gradient algorithm, featuring a group-level policy gradient, designed to yield an unbiased gradient estimator of the reward, and adaptive prompt filtering, developed to enforce the constraint on cloud LLM usage. Extensive experiments across models and benchmarks show that the proposed methodology consistently outperforms existing baselines and significantly narrows the gap to full cloud LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00121v1">Ask, Clarify, Optimize: Human-LLM Agent Collaboration for Smarter Inventory Control</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      Inventory management remains a challenge for many small and medium-sized businesses that lack the expertise to deploy advanced optimization methods. This paper investigates whether Large Language Models (LLMs) can help bridge this gap. We show that employing LLMs as direct, end-to-end solvers incurs a significant "hallucination tax": a performance gap arising from the model's inability to perform grounded stochastic reasoning. To address this, we propose a hybrid agentic framework that strictly decouples semantic reasoning from mathematical calculation. In this architecture, the LLM functions as an intelligent interface, eliciting parameters from natural language and interpreting results while automatically calling rigorous algorithms to build the optimization engine. To evaluate this interactive system against the ambiguity and inconsistency of real-world managerial dialogue, we introduce the Human Imitator, a fine-tuned "digital twin" of a boundedly rational manager that enables scalable, reproducible stress-testing. Our empirical analysis reveals that the hybrid agentic framework reduces total inventory costs by 32.1% relative to an interactive baseline using GPT-4o as an end-to-end solver. Moreover, we find that providing perfect ground-truth information alone is insufficient to improve GPT-4o's performance, confirming that the bottleneck is fundamentally computational rather than informational. Our results position LLMs not as replacements for operations research, but as natural-language interfaces that make rigorous, solver-based policies accessible to non-experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00110v1">CTMap: LLM-Enabled Connectivity-Aware Path Planning in Millimeter-Wave Digital Twin Networks</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      In this paper, we present \textit{CTMAP}, a large language model (LLM) empowered digital twin framework for connectivity-aware route navigation in millimeter-wave (mmWave) wireless networks. Conventional navigation tools optimize only distance, time, or cost, overlooking network connectivity degradation caused by signal blockage in dense urban environments. The proposed framework constructs a digital twin of the physical mmWave network using OpenStreetMap, Blender, and NVIDIA Sionna's ray-tracing engine to simulate realistic received signal strength (RSS) maps. A modified Dijkstra algorithm then generates optimal routes that maximize cumulative RSS, forming the training data for instruction-tuned GPT-4-based reasoning. This integration enables semantic route queries such as ``find the strongest-signal path'' and returns connectivity-optimized paths that are interpretable by users and adaptable to real-time environmental updates. Experimental results demonstrate that CTMAP achieves up to a tenfold improvement in cumulative signal strength compared to shortest-distance baselines, while maintaining high path validity. The synergy of digital twin simulation and LLM reasoning establishes a scalable foundation for intelligent, interpretable, and connectivity-driven navigation, advancing the design of AI-empowered 6G mobility systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00097v1">The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 15 figures
    </div>
    <details class="paper-abstract">
      We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.16893v2">From Transformers to LLMs: A Systematic Survey of Efficiency Considerations in NLP</a></div>
    <div class="paper-meta">
      📅 2025-12-31
      | 💬 63 pages, 5 tables and 22 figures
    </div>
    <details class="paper-abstract">
      The emergence of Transformer-based Large Language Models (LLMs) has substantially augmented the capabilities of Natural Language Processing (NLP), thereby intensifying the demand for computational resources. Therefore, enhancing efficiency based on factors like computational requirements, energy consumption, carbon footprint and financial cost has become a vital area of research. This motivates us to conduct a systematic literature review on Transformer-based LLMs in NLP from the perspective of efficiency. In this survey of 312 articles published between the years 2011 and 2025, efficiency-improvement endeavors have been systematically discussed targeting various aspects such as data curation, model design, model downsizing, and dynamic inferencing. This has been augmented with efficiency considerations in model adaptation strategies like pre-training, fine-tuning, prompt-engineering and Retrieval-Augmented Generation (RAG). Furthermore, a statistical analysis of the articles has been performed followed by an in-depth evaluation of the efficiency and efficacy of more than 30 renowned NLP models has been conducted on 13 evaluation benchmarks. This paper offers valuable insights for researchers, professionals as well as scholars, and explores the trend of research toward sustainable practices in NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00065v1">The Trojan in the Vocabulary: Stealthy Sabotage of LLM Composition</a></div>
    <div class="paper-meta">
      📅 2025-12-31
    </div>
    <details class="paper-abstract">
      The open-weight LLM ecosystem is increasingly defined by model composition techniques (such as weight merging, speculative decoding, and vocabulary expansion) that remix capabilities from diverse sources. A critical prerequisite for applying these methods across different model families is tokenizer transplant, which aligns incompatible vocabularies to a shared embedding space. We demonstrate that this essential interoperability step introduces a supply-chain vulnerability: we engineer a single "breaker token" that is functionally inert in a donor model yet reliably reconstructs into a high-salience malicious feature after transplant into a base model. By exploiting the geometry of coefficient reuse, our attack creates an asymmetric realizability gap that sabotages the base model's generation while leaving the donor's utility statistically indistinguishable from nominal behavior. We formalize this as a dual-objective optimization problem and instantiate the attack using a sparse solver. Empirically, the attack is training-free and achieves spectral mimicry to evade outlier detection, while demonstrating structural persistence against fine-tuning and weight merging, highlighting a hidden risk in the pipeline of modular AI composition. Code is available at https://github.com/xz-liu/tokenforge
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24511v1">Understanding LLM Checkpoint/Restore I/O Strategies and Patterns</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 SCA/HPCAsia 2026 Workshops: Supercomputing Asia and International Conference on High Performance Computing in the Asia Pacific Region Workshops
    </div>
    <details class="paper-abstract">
      As LLMs and foundation models scale, checkpoint/restore has become a critical pattern for training and inference. With 3D parallelism (tensor, pipeline, data), checkpointing involves many processes, each managing numerous tensors of varying shapes and sizes, that must be persisted frequently to stable storage (e.g., parallel file systems). This turns checkpoint/restore into a big-data I/O problem characterized by volume, variety, and velocity. The workflow must traverse the full storage stack -- from GPU memory through host memory and local storage to external repositories -- whose tiers differ by orders of magnitude in performance, creating bottlenecks under concurrency even with asynchronous flush/prefetch. Kernel-accelerated I/O libraries such as \texttt{liburing} may mitigate these issues versus POSIX, but their effectiveness for LLM checkpointing remains underexplored. We develop microbenchmarks to quantify trade-offs when using \texttt{liburing}, evaluating how aggregation, alignment, and I/O coalescing interact under buffered and direct I/O. We find that uncoalesced small-buffer operations halve throughput relative to synthetic workloads, while file system-aware aggregation restores bandwidth and reduces metadata overhead. Compared to state-of-the-art LLM checkpointing engines, our approach achieves up to $3.9\times$ higher write throughput than DataStates-LLM and $7.6\times$ higher than TorchSnapshot. These results highlight the need for aggregation and coalescing strategies that align with modern file systems and I/O backends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24505v1">Evaluating the Reasoning Abilities of LLMs on Underrepresented Mathematics Competition Problems</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 7 pages, submitted to ACM Transactions on Intelligent Systems and Technology
    </div>
    <details class="paper-abstract">
      Understanding the limitations of Large Language Models, or LLMs, in mathematical reasoning has been the focus of several recent studies. However, the majority of these studies use the same datasets for benchmarking, which limits the generalizability of their findings and may not fully capture the diverse challenges present in mathematical tasks. The purpose of the present study is to analyze the performance of LLMs on underrepresented mathematics competition problems. We prompted three leading LLMs, namely GPT-4o-mini, Gemini-2.0-Flash, and DeepSeek-V3, with the Missouri Collegiate Mathematics Competition problems in the areas of Calculus, Analytic Geometry, and Discrete Mathematics. The LLMs responses were then compared to the known correct solutions in order to determine the accuracy of the LLM for each problem domain. We also analyzed the LLMs reasoning to explore patterns in errors across problem types and models. DeepSeek-V3 has the best performance in all three categories of Calculus, Analytic Geometry, and Discrete Mathematics, both in reasoning and correct final answers. All three LLMs exhibited notably weak performance in Geometry. The majority of errors made by DeepSeek-V3 were attributed to computational and logical mistakes, whereas GPT-4o-mini frequently exhibited logical and approach-related errors. Gemini, on the other hand, tended to struggle with incomplete reasoning and drawing rushed conclusions. In conclusion, evaluating LLMs on underrepresented mathematics competition datasets can provide deeper insights into their distinct error patterns and highlight ongoing challenges in structured reasoning, particularly within the domain of Geometry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.15834v2">FEDSTR: Money-In AI-Out | A Decentralized Marketplace for Federated Learning and LLM Training on the NOSTR Protocol</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The NOSTR is a communication protocol for the social web, based on the w3c websockets standard. Although it is still in its infancy, it is well known as a social media protocol, with thousands of trusted users and multiple user interfaces, offering a unique experience and enormous capabilities. To name a few, the NOSTR applications include but are not limited to direct messaging, file sharing, audio/video streaming, collaborative writing, blogging and data processing through distributed AI directories. In this work, we propose an approach that builds upon the existing protocol structure with end goal a decentralized marketplace for federated learning and LLM training. In this proposed design there are two parties: on one side there are customers who provide a dataset that they want to use for training an AI model. On the other side, there are service providers, who receive (parts of) the dataset, train the AI model, and for a payment as an exchange, they return the optimized AI model. To demonstrate viability, we present a proof-of-concept implementation over public NOSTR relays. The decentralized and censorship resistant features of the NOSTR enable the possibility of designing a fair and open marketplace for training AI models and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24449v1">PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have demonstrated remarkable potential across a wide range of practical applications. However, long-context inference remains a significant challenge due to the substantial memory requirements of the key-value (KV) cache, which can scale to several gigabytes as sequence length and batch size increase. In this paper, we present \textbf{PackKV}, a generic and efficient KV cache management framework optimized for long-context generation. %, which synergistically supports both latency-critical and throughput-critical inference scenarios. PackKV introduces novel lossy compression techniques specifically tailored to the characteristics of KV cache data, featuring a careful co-design of compression algorithms and system architecture. Our approach is compatible with the dynamically growing nature of the KV cache while preserving high computational efficiency. Experimental results show that, under the same and minimum accuracy drop as state-of-the-art quantization methods, PackKV achieves, on average, \textbf{153.2}\% higher memory reduction rate for the K cache and \textbf{179.6}\% for the V cache. Furthermore, PackKV delivers extremely high execution throughput, effectively eliminating decompression overhead and accelerating the matrix-vector multiplication operation. Specifically, PackKV achieves an average throughput improvement of \textbf{75.7}\% for K and \textbf{171.7}\% for V across A100 and RTX Pro 6000 GPUs, compared to cuBLAS matrix-vector multiplication kernels, while demanding less GPU memory bandwidth. Code available on https://github.com/BoJiang03/PackKV
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11557v2">Invisible Languages of the LLM Universe</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Large Language Models are trained on massive multilingual corpora, yet this abundance masks a profound crisis: of the world's 7,613 living languages, approximately 2,000 languages with millions of speakers remain effectively invisible in digital ecosystems. We propose a critical framework connecting empirical measurements of language vitality (real world demographic strength) and digitality (online presence) with postcolonial theory and epistemic injustice to explain why linguistic inequality in AI systems is not incidental but structural. Analyzing data across all documented human languages, we identify four categories: Strongholds (33%, high vitality and digitality), Digital Echoes (6%, high digitality despite declining vitality), Fading Voices (36%, low on both dimensions), and critically, Invisible Giants (27%, high vitality but near-zero digitality) - languages spoken by millions yet absent from the LLM universe. We demonstrate that these patterns reflect continuities from colonial-era linguistic hierarchies to contemporary AI development, constituting digital epistemic injustice. Our analysis reveals that English dominance in AI is not a technical necessity but an artifact of power structures that systematically exclude marginalized linguistic knowledge. We conclude with implications for decolonizing language technology and democratizing access to AI benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11007v4">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, together with their large model size, make their deployment more challenging. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design TMO, a local-cloud LLM inference system with Three-M Offloading: Multi-modal, Multi-task, and Multi-dialogue. TMO incorporates (i) a lightweight local LLM that can process simple tasks at high speed and (ii) a large-scale cloud LLM that can handle multi-modal data sources. We develop a resource-constrained reinforcement learning (RCRL) strategy for TMO that optimizes the inference location (i.e., local vs. cloud) and multi-modal data sources to use for each task/dialogue, aiming to maximize the long-term reward (response quality, latency, and usage cost) while adhering to resource constraints. We also contribute M4A1, a new dataset we curated that contains reward and cost metrics across multiple modality, task, dialogue, and LLM configurations, enabling evaluation of offloading decisions. We demonstrate the effectiveness of TMO compared to several exploration-decision and LLM-as-Agent baselines, showing significant improvements in latency, cost, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18536v2">Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 AACL Findings 2025
    </div>
    <details class="paper-abstract">
      This work considers a black-box threat model in which adversaries attempt to propagate arbitrary non-relevant content in search. We show that retrievers, rerankers, and LLM relevance judges are all highly vulnerable to attacks that enable arbitrary content to be promoted to the top of search results and to be assigned perfect relevance scores. We investigate how attackers may achieve this via content injection, injecting arbitrary sentences into relevant passages or query terms into arbitrary passages. Our study analyzes how factors such as model class and size, the balance between relevant and non-relevant content, injection location, toxicity and severity of injected content, and the role of LLM-generated content influence attack success, yielding novel, concerning, and often counterintuitive results. Our results reveal a weakness in embedding models, LLM-based scoring models, and generative LLMs, raising concerns about the general robustness, safety, and trustworthiness of language models regardless of the type of model or the role in which they are employed. We also emphasize the challenges of robust defenses against these attacks. Classifiers and more carefully prompted LLM judges often fail to recognize passages with content injection, especially when considering diverse text topics and styles. Our findings highlight the need for further research into arbitrary content injection attacks. We release our code for further study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24314v1">QianfanHuijin Technical Report: A Novel Multi-Stage Training Paradigm for Finance Industrial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Domain-specific enhancement of Large Language Models (LLMs) within the financial context has long been a focal point of industrial application. While previous models such as BloombergGPT and Baichuan-Finance primarily focused on knowledge enhancement, the deepening complexity of financial services has driven a growing demand for models that possess not only domain knowledge but also robust financial reasoning and agentic capabilities. In this paper, we present QianfanHuijin, a financial domain LLM, and propose a generalizable multi-stage training paradigm for industrial model enhancement. Our approach begins with Continual Pre-training (CPT) on financial corpora to consolidate the knowledge base. This is followed by a fine-grained Post-training pipeline designed with increasing specificity: starting with Financial SFT, progressing to Finance Reasoning RL and Finance Agentic RL, and culminating in General RL aligned with real-world business scenarios. Empirical results demonstrate that QianfanHuijin achieves superior performance across various authoritative financial benchmarks. Furthermore, ablation studies confirm that the targeted Reasoning RL and Agentic RL stages yield significant gains in their respective capabilities. These findings validate our motivation and suggest that this fine-grained, progressive post-training methodology is poised to become a mainstream paradigm for various industrial-enhanced LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23126v2">InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) and its variants have become standard for aligning Large Language Models due to their simplicity and offline stability. However, we identify two fundamental limitations. First, the optimal policy depends on arbitrary modeling choices (scalarization function, reference policy), yielding behavior reflecting parameterization artifacts rather than true preferences. Second, treating response generation in isolation fails to leverage comparative information in pairwise data, leaving the model's capacity for intrinsic self-reflection untapped. To address it, we propose Intrinsic Self-reflective Preference Optimization (InSPO), deriving a globally optimal policy conditioning on both context and alternative responses. We prove this formulation superior to DPO/RLHF while guaranteeing invariance to scalarization and reference choices. InSPO serves as a plug-and-play enhancement without architectural changes or inference overhead. Experiments demonstrate consistent improvements in win rates and length-controlled metrics, validating that unlocking self-reflection yields more robust, human-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13142v5">Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 Codebase: https://github.com/EvolvingLMMs-Lab/EASI/ ; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
    </div>
    <details class="paper-abstract">
      Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.04235v3">Addressing Hallucinations with RAG and NMISS in Italian Healthcare LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      I combine detection and mitigation techniques to addresses hallucinations in Large Language Models (LLMs). Mitigation is achieved in a question-answering Retrieval-Augmented Generation (RAG) framework while detection is obtained by introducing the Negative Missing Information Scoring System (NMISS), which accounts for contextual relevance in responses. While RAG mitigates hallucinations by grounding answers in external data, NMISS refines the evaluation by identifying cases where traditional metrics incorrectly flag contextually accurate responses as hallucinations. I use Italian health news articles as context to evaluate LLM performance. Results show that Gemma2 and GPT-4 outperform the other models, with GPT-4 producing answers closely aligned with reference responses. Mid-tier models, such as Llama2, Llama3, and Mistral benefit significantly from NMISS, highlighting their ability to provide richer contextual information. This combined approach offers new insights into the reduction and more accurate assessment of hallucinations in LLMs, with applications in real-world healthcare tasks and other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15301v3">A Survey on LLM-Assisted Clinical Trial Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 Accepted to IJCNLP-AACl 2025
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24830v3">Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 18 pages, 7 figures and tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit extensive medical knowledge but are prone to hallucinations and inaccurate citations, which pose a challenge to their clinical adoption and regulatory compliance. Current methods, such as Retrieval Augmented Generation, partially address these issues by grounding answers in source documents, but hallucinations and low fact-level explainability persist. In this work, we introduce a novel atomic fact-checking framework designed to enhance the reliability and explainability of LLMs used in medical long-form question answering. This method decomposes LLM-generated responses into discrete, verifiable units called atomic facts, each of which is independently verified against an authoritative knowledge base of medical guidelines. This approach enables targeted correction of errors and direct tracing to source literature, thereby improving the factual accuracy and explainability of medical Q&A. Extensive evaluation using multi-reader assessments by medical experts and an automated open Q&A benchmark demonstrated significant improvements in factual accuracy and explainability. Our framework achieved up to a 40% overall answer improvement and a 50% hallucination detection rate. The ability to trace each atomic fact back to the most relevant chunks from the database provides a granular, transparent explanation of the generated responses, addressing a major gap in current medical AI applications. This work represents a crucial step towards more trustworthy and reliable clinical applications of LLMs, addressing key prerequisites for clinical application and fostering greater confidence in AI-assisted healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24120v1">Enhancing LLM-Based Neural Network Generation: Few-Shot Prompting and Efficient Validation for Automated Architecture Design</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Automated neural network architecture design remains a significant challenge in computer vision. Task diversity and computational constraints require both effective architectures and efficient search methods. Large Language Models (LLMs) present a promising alternative to computationally intensive Neural Architecture Search (NAS), but their application to architecture generation in computer vision has not been systematically studied, particularly regarding prompt engineering and validation strategies. Building on the task-agnostic NNGPT/LEMUR framework, this work introduces and validates two key contributions for computer vision. First, we present Few-Shot Architecture Prompting (FSAP), the first systematic study of the number of supporting examples (n = 1, 2, 3, 4, 5, 6) for LLM-based architecture generation. We find that using n = 3 examples best balances architectural diversity and context focus for vision tasks. Second, we introduce Whitespace-Normalized Hash Validation, a lightweight deduplication method (less than 1 ms) that provides a 100x speedup over AST parsing and prevents redundant training of duplicate computer vision architectures. In large-scale experiments across seven computer vision benchmarks (MNIST, CIFAR-10, CIFAR-100, CelebA, ImageNette, SVHN, Places365), we generated 1,900 unique architectures. We also introduce a dataset-balanced evaluation methodology to address the challenge of comparing architectures across heterogeneous vision tasks. These contributions provide actionable guidelines for LLM-based architecture search in computer vision and establish rigorous evaluation practices, making automated design more accessible to researchers with limited computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24103v1">Enhancing LLM Planning Capabilities through Intrinsic Self-Critique</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      We demonstrate an approach for LLMs to critique their \emph{own} answers with the goal of enhancing their performance that leads to significant improvements over established planning benchmarks. Despite the findings of earlier research that has cast doubt on the effectiveness of LLMs leveraging self critique methods, we show significant performance gains on planning datasets in the Blocksworld domain through intrinsic self-critique, without external source such as a verifier. We also demonstrate similar improvements on Logistics and Mini-grid datasets, exceeding strong baseline accuracies. We employ a few-shot learning technique and progressively extend it to a many-shot approach as our base method and demonstrate that it is possible to gain substantial improvement on top of this already competitive approach by employing an iterative process for correction and refinement. We illustrate how self-critique can significantly boost planning performance. Our empirical results present new state-of-the-art on the class of models considered, namely LLM model checkpoints from October 2024. Our primary focus lies on the method itself, demonstrating intrinsic self-improvement capabilities that are applicable regardless of the specific model version, and we believe that applying our method to more complex search techniques and more capable models will lead to even better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07017v2">Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in its automation. However, existing benchmarks for LLM-based code review face three major limitations. Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review. We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries. ContextCRBench supports three evaluation scenarios aligned with the review workflow: hunk-level quality assessment, line-level defect localization, and line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility. https://github.com/kinesiatricssxilm14/ContextCRBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24063v1">How and Why LLMs Generalize: A Fine-Grained Analysis of LLM Reasoning from Cognitive Behaviors to Low-Level Patterns</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display strikingly different generalization behaviors: supervised fine-tuning (SFT) often narrows capability, whereas reinforcement-learning (RL) tuning tends to preserve it. The reasons behind this divergence remain unclear, as prior studies have largely relied on coarse accuracy metrics. We address this gap by introducing a novel benchmark that decomposes reasoning into atomic core skills such as calculation, fact retrieval, simulation, enumeration, and diagnostic, providing a concrete framework for addressing the fundamental question of what constitutes reasoning in LLMs. By isolating and measuring these core skills, the benchmark offers a more granular view of how specific cognitive abilities emerge, transfer, and sometimes collapse during post-training. Combined with analyses of low-level statistical patterns such as distributional divergence and parameter statistics, it enables a fine-grained study of how generalization evolves under SFT and RL across mathematical, scientific reasoning, and non-reasoning tasks. Our meta-probing framework tracks model behavior at different training stages and reveals that RL-tuned models maintain more stable behavioral profiles and resist collapse in reasoning skills, whereas SFT models exhibit sharper drift and overfit to surface patterns. This work provides new insights into the nature of reasoning in LLMs and points toward principles for designing training strategies that foster broad, robust generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24044v1">Jailbreaking Attacks vs. Content Safety Filters: How Far Are We in the LLM Safety Arms Race?</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 26 pages,11 tables, 7 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed, ensuring their safe use is paramount. Jailbreaking, adversarial prompts that bypass model alignment to trigger harmful outputs, present significant risks, with existing studies reporting high success rates in evading common LLMs. However, previous evaluations have focused solely on the models, neglecting the full deployment pipeline, which typically incorporates additional safety mechanisms like content moderation filters. To address this gap, we present the first systematic evaluation of jailbreak attacks targeting LLM safety alignment, assessing their success across the full inference pipeline, including both input and output filtering stages. Our findings yield two key insights: first, nearly all evaluated jailbreak techniques can be detected by at least one safety filter, suggesting that prior assessments may have overestimated the practical success of these attacks; second, while safety filters are effective in detection, there remains room to better balance recall and precision to further optimize protection and user experience. We highlight critical gaps and call for further refinement of detection accuracy and usability in LLM safety systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20730v3">Do LLMs Understand Collaborative Signals? Diagnosis and Repair</a></div>
    <div class="paper-meta">
      📅 2025-12-30
    </div>
    <details class="paper-abstract">
      Collaborative information from user-item interactions is a fundamental source of signal in successful recommender systems. Recently, researchers have attempted to incorporate this knowledge into large language model-based recommender approaches (LLMRec) to enhance their performance. However, there has been little fundamental analysis of whether LLMs can effectively reason over collaborative information. In this paper, we analyze the ability of LLMs to reason about collaborative information in recommendation tasks, comparing their performance to traditional matrix factorization (MF) models. We propose a simple and effective method to improve LLMs' reasoning capabilities using retrieval-augmented generation (RAG) over the user-item interaction matrix with four different prompting strategies. Our results show that the LLM outperforms the MF model whenever we provide relevant information in a clear and easy-to-follow format, and prompt the LLM to reason based on it. We observe that with this strategy, in almost all cases, the more information we provide, the better the LLM performs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21635v2">Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-12-30
      | 💬 Published as a conference paper at KDD 2026
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) are commonly regarded as errors to be minimized. However, recent perspectives suggest that some hallucinations may encode creative or epistemically valuable content, a dimension that remains underquantified in current literature. Existing hallucination detection methods primarily focus on factual consistency, struggling to handle heterogeneous scientific tasks and balance creativity with accuracy. To address these challenges, we propose HIC-Bench, a novel evaluation framework that categorizes hallucinations into Intelligent Hallucinations (IH) and Defective Hallucinations (DH), enabling systematic investigation of their interplay in LLM creativity. HIC-Bench features three core characteristics: (1) Structured IH/DH Assessment. using a multi-dimensional metric matrix integrating Torrance Tests of Creative Thinking (TTCT) metrics (Originality, Feasibility, Value) with hallucination-specific dimensions (scientific plausibility, factual deviation); (2) Cross-Domain Applicability. spanning ten scientific domains with open-ended innovation tasks; and (3) Dynamic Prompt Optimization. leveraging the Dynamic Hallucination Prompt (DHP) to guide models toward creative and reliable outputs. The evaluation process employs multiple LLM judges, averaging scores to mitigate bias, with human annotators verifying IH/DH classifications. Experimental results reveal a nonlinear relationship between IH and DH, demonstrating that creativity and correctness can be jointly optimized. These insights position IH as a catalyst for creativity and reveal the ability of LLM hallucinations to drive scientific innovation.Additionally, the HIC-Bench offers a valuable platform for advancing research into the creative intelligence of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23684v1">Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23609v1">The Big Three in Marriage Talk: LLM-Assisted Analysis of Moral Ethics and Sentiment on Weibo and Xiaohongshu</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      China's marriage registrations have declined dramatically, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. Understanding public attitudes toward marriage requires examining not only emotional sentiment but also the moral reasoning underlying these evaluations. This study analyzed 219,358 marriage-related posts from two major Chinese social media platforms (Sina Weibo and Xiaohongshu) using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral dimensions (Autonomy, Community, Divinity). Results revealed platform differences: Weibo discourse skewed positive, while Xiaohongshu was predominantly neutral. Most posts across both platforms lacked explicit moral framing. However, when moral ethics were invoked, significant associations with sentiment emerged. Posts invoking Autonomy ethics and Community ethics were predominantly negative, whereas Divinity-framed posts tended toward neutral or positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations drive negative marriage attitudes in contemporary China. The study demonstrates LLMs' utility for scaling qualitative analysis and offers insights for developing culturally informed policies addressing marriage decline in Chinese contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02976v3">How Safe Are AI-Generated Patches? A Large-scale Study on Security Risks in LLM and Agentic Automated Program Repair on SWE-bench</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their agentic frameworks are increasingly adopted to perform development tasks such as automated program repair (APR). While prior work has identified security risks in LLM-generated code, most have focused on synthetic, simplified, or isolated tasks that lack the complexity of real-world program repair. In this study, we present the first large-scale security analysis of LLM-generated patches using 20,000+ GitHub issues. We evaluate patches proposed by developers, a standalone LLM (Llama 3.3 Instruct-70B), and three top-performing agentic frameworks (OpenHands, AutoCodeRover, HoneyComb). Finally, we analyze a wide range of code, issue, and project-level factors to understand the conditions under which generating insecure patches is more likely. Our findings reveal that Llama introduces many new vulnerabilities, exhibiting unique patterns not found in developers' code. Agentic workflows also generate a number of vulnerabilities, particularly when given more autonomy. We find that vulnerabilities in LLM-generated patches are associated with distinctive code characteristics and are commonly observed in issues missing specific types of information. These results suggest that contextual factors play a critical role in the security of the generated patches and point toward the need for proactive risk assessment methods that account for both issue and code-level information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23587v1">Can AI Recognize Its Own Reflection? Self-Detection Performance of LLMs in Computing Education</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 10 pages, 5 tables. Accepted for publication at the 59th Hawaii International Conference on System Sciences
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) presents a significant challenge to academic integrity within computing education. As educators seek reliable detection methods, this paper evaluates the capacity of three prominent LLMs (GPT-4, Claude, and Gemini) to identify AI-generated text in computing-specific contexts. We test their performance under both standard and 'deceptive' prompt conditions, where the models were instructed to evade detection. Our findings reveal a significant instability: while default AI-generated text was easily identified, all models struggled to correctly classify human-written work (with error rates up to 32%). Furthermore, the models were highly susceptible to deceptive prompts, with Gemini's output completely fooling GPT-4. Given that simple prompt alterations significantly degrade detection efficacy, our results demonstrate that these LLMs are currently too unreliable for making high-stakes academic misconduct judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23547v1">Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 Accepted to ICPRAM 2026 in Marbella, Spain
    </div>
    <details class="paper-abstract">
      Hallucinations, the generation of apparently convincing yet false statements, remain a major barrier to the safe deployment of LLMs. Building on the strong performance of self-detection methods, we examine the use of structured knowledge representations, namely knowledge graphs, to improve hallucination self-detection. Specifically, we propose a simple yet powerful approach that enriches hallucination self-detection by (i) converting LLM responses into knowledge graphs of entities and relations, and (ii) using these graphs to estimate the likelihood that a response contains hallucinations. We evaluate the proposed approach using two widely used LLMs, GPT-4o and Gemini-2.5-Flash, across two hallucination detection datasets. To support more reliable future benchmarking, one of these datasets has been manually curated and enhanced and is released as a secondary outcome of this work. Compared to standard self-detection methods and SelfCheckGPT, a state-of-the-art approach, our method achieves up to 16% relative improvement in accuracy and 20% in F1-score. Our results show that LLMs can better analyse atomic facts when they are structured as knowledge graphs, even when initial outputs contain inaccuracies. This low-cost, model-agnostic approach paves the way toward safer and more trustworthy language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23518v1">Single LLM Debate, MoLaCE: Mixture of Latent Concept Experts Against Confirmation Bias</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly vulnerable to input confirmation bias. When a prompt implies a preferred answer, models often reinforce that bias rather than explore alternatives. This phenomenon remains underexplored, yet it is already harmful in base models and poses an even greater risk in multi-agent debate, where echo chambers reinforce bias instead of correction. We introduce Mixture of Latent Concept Experts (MoLaCE), a lightweight inference-time framework that addresses confirmation bias by mixing experts instantiated as different activation strengths over latent concepts that shape model responses. Our key insight is that, due to the compositional nature of language, differently phrased prompts reweight latent concepts in prompt-specific ways that affect factual correctness, so no single fixed intervention can be applied universally across inputs. This design enables a single LLM to emulate the benefits of debate internally while remaining computationally efficient and scalable. It can also be integrated into multi-agent debate frameworks to diversify perspectives and reduce correlated errors. We empirically show that it consistently reduces confirmation bias, improves robustness, and matches or surpasses multi-agent debate while requiring only a fraction of the computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23515v1">Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Signal decay and regime shifts pose recurring challenges for data-driven investment strategies in non-stationary markets. Conventional time-series and machine learning approaches, which rely primarily on historical correlations, often struggle to generalize when the economic environment changes. While large language models (LLMs) offer strong capabilities for processing unstructured information, their potential to support quantitative factor screening through explicit economic reasoning remains underexplored. Existing factor-based methods typically reduce alphas to numerical time series, overlooking the semantic rationale that determines when a factor is economically relevant. We propose Alpha-R1, an 8B-parameter reasoning model trained via reinforcement learning for context-aware alpha screening. Alpha-R1 reasons over factor logic and real-time news to evaluate alpha relevance under changing market conditions, selectively activating or deactivating factors based on contextual consistency. Empirical results across multiple asset pools show that Alpha-R1 consistently outperforms benchmark strategies and exhibits improved robustness to alpha decay. The full implementation and resources are available at https://github.com/FinStep-AI/Alpha-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23511v1">Beyond Correctness: Exposing LLM-generated Logical Flaws in Reasoning via Multi-step Automated Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, leading to their adoption in high-stakes domains such as healthcare, law, and scientific research. However, their reasoning often contains subtle logical errors masked by fluent language, posing significant risks for critical applications. While existing approaches like fact-checking, self-consistency methods, and rule-based validation provide partial solutions, they fail to detect complex logical flaws in multi-step reasoning. To overcome these challenges, we present MATP, an evaluation framework for systematically verifying LLM reasoning via Multi-step Automatic Theorem Proving. MATP translates natural language reasoning into First-Order Logic (FOL) and applies automated theorem provers to assess step-by-step logical validity. This approach identifies hidden logical errors and provides fine-grained classifications of reasoning correctness. Evaluations on a benchmark comprising 10,830 reasoning instances generated by 10 LLMs across tasks from PrOntoQA-OOD, ProofWriter, and FOLIO show that MATP surpasses prompting-based baselines by over 42 percentage points in reasoning step verification. It further reveals model-level disparities, with reasoning models generating more logically coherent outputs than general models. These results demonstrate MATP's potential to enhance the trustworthiness of LLM-generated reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23489v1">The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: https://anonymous.4open.science/r/MIRAGE-VC-323F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23430v1">C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Bias in Large Language Models (LLMs) poses significant risks to trustworthiness, manifesting primarily as stereotypical biases (e.g., gender or racial stereotypes) and structural biases (e.g., lexical overlap or position preferences). However, prior paradigms typically address these in isolation, often mitigating one at the expense of exacerbating the other. To address this, we conduct a systematic exploration of these reasoning failures and identify a primary inducement: the latent spurious feature correlations within the input that drive these erroneous reasoning shortcuts. Driven by these findings, we introduce Causal-Contrastive Preference Optimization (C2PO), a unified alignment framework designed to tackle these specific failures by simultaneously discovering and suppressing these correlations directly within the optimization process. Specifically, C2PO leverages causal counterfactual signals to isolate bias-inducing features from valid reasoning paths, and employs a fairness-sensitive preference update mechanism to dynamically evaluate logit-level contributions and suppress shortcut features. Extensive experiments across multiple benchmarks covering stereotypical bias (BBQ, Unqover), structural bias (MNLI, HANS, Chatbot, MT-Bench), out-of-domain fairness (StereoSet, WinoBias), and general utility (MMLU, GSM8K) demonstrate that C2PO effectively mitigates stereotypical and structural biases while preserving robust general reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21580v2">Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parameter LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality English enrichment to transfer performance gains across languages. Our model supports 12 languages, with special focus on Russian. Despite a significantly smaller training budget than comparable models, Gamayun outperforms LLaMA3.2-1B (9T tokens) on all considered benchmarks, and surpasses Qwen2.5-1.5B (18T tokens) on a wide range of English and multilingual tasks. It matches or exceeds Qwen3 (36T tokens) on most tasks outside advanced STEM, achieving state-of-the-art results in Russian, including the MERA benchmark, among the models of comparable size (1-2B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12620v3">Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 9 pages, 4 figures, 5 tables. Accepted at AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
    </div>
    <details class="paper-abstract">
      We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25987v2">R-Log: Incentivizing Log Analysis Capability in LLMs via Reasoning-based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 Accepted by ICSE 2026 (SEIP Track)
    </div>
    <details class="paper-abstract">
      The growing complexity of log data in modern software systems has prompted the use of Large Language Models (LLMs) for automated log analysis. Current approaches typically rely on direct supervised fine-tuning (SFT) on log-label pairs. However, this exacerbates the domain discrepancy between general-purpose LLMs and specialized log data, causing overfitting. Furthermore, SFT's imbalanced loss computation often allows lengthy contexts to overwhelm critical, concise details in model answers, leading to hallucinations. To address these limitations, we propose R-Log, a novel reasoning-based paradigm that mirrors the structured, step-by-step analytical process of human engineers. This approach enhances generalizability by learning the underlying rules behind conclusions. We further employ Reinforcement Learning (RL) to optimize the model within a simulated O&M environment, thereby reducing hallucinations by directly rewarding correct outcomes. R-Log is first cold-started on a curated dataset of 2k+ reasoning trajectories, guided by 13 strategies from manual O&M practices, to establish an initial reasoning capability. This ability is then refined via RL using a joint reward function. Empirical evaluations on real-world logs show that R-Log outperforms existing methods across five log analysis tasks, particularly in unseen scenarios (by 228.05%). We also designed R-Log-fast with 5x speedup while keeping 93% of the efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05822v3">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01412v2">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23310v1">Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 11 pages, 9 figures. Accepted by ACM for presentation at UCC '25 (18th International Conference on Utility and Cloud Computing), December 1-4, 2025, France. Proceedings publication pending
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge devices is challenging due to their limited memory and power resources. Cloud-only inference reduces device burden but introduces high latency and cost. Static edge-cloud partitions optimize a single metric and struggle when bandwidth fluctuates. We propose Splitwise, a novel Lyapunov-assisted deep reinforcement learning (DRL) framework for fine-grained, adaptive partitioning of LLMs across edge and cloud environments. Splitwise decomposes transformer layers into attention heads and feed-forward sub-blocks, exposing more partition choices than layer-wise schemes. A hierarchical DRL policy, guided by Lyapunov optimization, jointly minimizes latency, energy consumption, and accuracy degradation while guaranteeing queue stability under stochastic workloads and variable network bandwidth. Splitwise also guarantees robustness via partition checkpoints with exponential backoff recovery in case of communication failures. Experiments on Jetson Orin NX, Galaxy S23, and Raspberry Pi 5 with GPT-2 (1.5B), LLaMA-7B, and LLaMA-13B show that Splitwise reduces end-to-end latency by 1.4x-2.8x and cuts energy consumption by up to 41% compared with existing partitioners. It lowers the 95th-percentile latency by 53-61% relative to cloud-only execution, while maintaining accuracy and modest memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06435v2">Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer new opportunities for scalable analysis of online discourse. Yet their use in multilingual social science research remains constrained by model size, cost and linguistic bias. We develop a lightweight, open-source LLM framework using fine-tuned LLaMA 3.2-3B models to classify immigration-related tweets across 13 languages. Unlike prior work relying on BERT style models or translation pipelines, we combine topic classification with stance detection and demonstrate that LLMs fine-tuned in just one or two languages can generalize topic understanding to unseen languages. Capturing ideological nuance, however, benefits from multilingual fine-tuning. Our approach corrects pretraining biases with minimal data from under-represented languages and avoids reliance on proprietary systems. With 26-168x faster inference and over 1000x cost savings compared to commercial LLMs, our method supports real-time analysis of billions of tweets. This scale-first framework enables inclusive, reproducible research on public attitudes across linguistic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09427v2">ODMA: On-Demand Memory Allocation Framework for LLM Serving on LPDDR-Class Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Device-memory management is a key bottleneck for serving large language models (LLMs) on accelerators whose memory has poor small-granularity random-access bandwidth (e.g., LPDDR5-class). Existing approaches either statically pre-allocate worst-case KV-cache per request, wasting substantial device memory, or rely on fine-grained paging that assumes high random-access tolerance and is therefore ill-suited to LPDDR-style systems. We present ODMA, an on-demand memory allocation framework for LLM serving on random-access-constrained device memory (RACM) platforms such as LPDDR5-based Cambricon MLUs. ODMA builds on generation-length prediction while addressing distribution drift and heavy-tailed request lengths via dynamic bucket partitioning and a large-bucket safeguard: bucket boundaries are periodically re-learned from online histograms, and high-uncertainty or overflowed requests fall back to a reserved large bucket for robustness. On Alpaca and Google-NQ, ODMA improves S3's predictor accuracy from 98.60% to 99.55% and from 82.68% to 93.36%, respectively. Serving DeepSeek-R1-Distill-Qwen-7B on four Cambricon MLU370-X4 accelerators, ODMA increases device-memory utilization from 55.05% to 72.45% on Alpaca and from 42.54% to 61.79% on Google-NQ, and boosts throughput by 23% and 27% over a static pre-allocation baseline. These results show that predictor-driven, hardware-aware allocation can unlock efficient LLM serving on RACM accelerators without hardware changes, complementing paging-centric designs tailored to HBM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23214v1">Anka: A Domain-Specific Language for Reliable LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 11 pages, 1 figure, 4 tables. Code and benchmarks available at https://github.com/BleBlo/Anka
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet they exhibit systematic errors on complex, multi-step programming tasks. We hypothesize that these errors stem from the flexibility of general-purpose languages, which permits multiple valid approaches and requires implicit state management. To test this hypothesis, we introduce Anka, a domain-specific language (DSL) for data transformation pipelines designed with explicit, constrained syntax that reduces ambiguity in code generation. Despite having zero prior training exposure to Anka, Claude 3.5 Haiku achieves 99.9% parse success and 95.8% overall task accuracy across 100 benchmark problems. Critically, Anka demonstrates a 40 percentage point accuracy advantage over Python on multi-step pipeline tasks (100% vs. 60%), where Python's flexible syntax leads to frequent errors in operation sequencing and variable management. Cross-model validation with GPT-4o-mini confirms this advantage (+26.7 percentage points on multi-step tasks). Our results demonstrate that: (1) LLMs can learn novel DSLs entirely from in-context prompts, achieving near-native accuracy; (2) constrained syntax significantly reduces errors on complex tasks; and (3) domain-specific languages purposefully designed for LLM generation can outperform general-purpose languages on which the LLM has extensive training. We release the complete language implementation, benchmark suite, and evaluation framework to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23206v1">Not too long do read: Evaluating LLM-generated extreme scientific summaries</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      High-quality scientific extreme summary (TLDR) facilitates effective science communication. How do large language models (LLMs) perform in generating them? How are LLM-generated summaries different from those written by human experts? However, the lack of a comprehensive, high-quality scientific TLDR dataset hinders both the development and evaluation of LLMs' summarization ability. To address these, we propose a novel dataset, BiomedTLDR, containing a large sample of researcher-authored summaries from scientific papers, which leverages the common practice of including authors' comments alongside bibliography items. We then test popular open-weight LLMs for generating TLDRs based on abstracts. Our analysis reveals that, although some of them successfully produce humanoid summaries, LLMs generally exhibit a greater affinity for the original text's lexical choices and rhetorical structures, hence tend to be more extractive rather than abstractive in general, compared to humans. Our code and datasets are available at https://github.com/netknowledge/LLM_summarization (Lyu and Ke, 2025).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23184v1">From Model Choice to Model Belief: Establishing a New Measure for LLM-Based Research</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate human behavior, but common practices to use LLM-generated data are inefficient. Treating an LLM's output ("model choice") as a single data point underutilizes the information inherent to the probabilistic nature of LLMs. This paper introduces and formalizes "model belief," a measure derived from an LLM's token-level probabilities that captures the model's belief distribution over choice alternatives in a single generation run. The authors prove that model belief is asymptotically equivalent to the mean of model choices (a non-trivial property) but forms a more statistically efficient estimator, with lower variance and a faster convergence rate. Analogous properties are shown to hold for smooth functions of model belief and model choice often used in downstream applications. The authors demonstrate the performance of model belief through a demand estimation study, where an LLM simulates consumer responses to different prices. In practical settings with limited numbers of runs, model belief explains and predicts ground-truth model choice better than model choice itself, and reduces the computation needed to reach sufficiently accurate estimates by roughly a factor of 20. The findings support using model belief as the default measure to extract more information from LLM-generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23167v1">SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often falter at complex planning tasks that require exploration and self-correction, as their linear reasoning process struggles to recover from early mistakes. While search algorithms like Monte Carlo Tree Search (MCTS) can explore alternatives, they are often ineffective when guided by sparse rewards and fail to leverage the rich semantic capabilities of LLMs. We introduce SPIRAL (Symbolic LLM Planning via Grounded and Reflective Search), a novel framework that embeds a cognitive architecture of three specialized LLM agents into an MCTS loop. SPIRAL's key contribution is its integrated planning pipeline where a Planner proposes creative next steps, a Simulator grounds the search by predicting realistic outcomes, and a Critic provides dense reward signals through reflection. This synergy transforms MCTS from a brute-force search into a guided, self-correcting reasoning process. On the DailyLifeAPIs and HuggingFace datasets, SPIRAL consistently outperforms the default Chain-of-Thought planning method and other state-of-the-art agents. More importantly, it substantially surpasses other state-of-the-art agents; for example, SPIRAL achieves 83.6% overall accuracy on DailyLifeAPIs, an improvement of over 16 percentage points against the next-best search framework, while also demonstrating superior token efficiency. Our work demonstrates that structuring LLM reasoning as a guided, reflective, and grounded search process yields more robust and efficient autonomous planners. The source code, full appendices, and all experimental data are available for reproducibility at the official project repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25743v2">Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2306.05499v3">Prompt Injection attack against LLM-integrated Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23136v1">Understanding EFL Learners' Code-Switching and Teachers' Pedagogical Approaches in LLM-Supported Speaking Practice</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      For English as a Foreign Language (EFL) learners, code-switching (CSW), or alternating between their native language and the target language (English), can lower anxiety and ease communication barriers. Large language models (LLMs), with their multilingual abilities, offer new opportunities to support CSW in speaking practice. Yet, the pedagogical design of LLM-based tutors remains underexplored. To this end, we conducted a six-week study of LLM-mediated speaking practice with 20 Korean EFL learners, alongside a qualitative study with nine English teachers who designed and refined responses to learner CSW. Findings show that learners used CSW not only to bridge lexical gaps but also to express cultural and emotional nuance, prompting teachers to employ selective interventions and dynamic scaffolding strategies. We conclude with design implications for bilingual LLM-powered tutors that leverage teachers' expertise to transform CSW into meaningful learning opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23126v1">InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) and its variants have become standard for aligning Large Language Models due to their simplicity and offline stability. However, we identify two fundamental limitations. First, the optimal policy depends on arbitrary modeling choices (scalarization function, reference policy), yielding behavior reflecting parameterization artifacts rather than true preferences. Second, treating response generation in isolation fails to leverage comparative information in pairwise data, leaving the model's capacity for intrinsic self-reflection untapped. To address it, we propose Intrinsic Self-reflective Preference Optimization (\q), deriving a globally optimal policy conditioning on both context and alternative responses. We prove this formulation superior to DPO/RLHF while guaranteeing invariance to scalarization and reference choices. \q~serves as a plug-and-play enhancement without architectural changes or inference overhead. Experiments demonstrate consistent improvements in win rates and length-controlled metrics, validating that unlocking self-reflection yields more robust, human-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04245v4">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls. Our code is available at: https://github.com/EricGLan/CI-RL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23858v1">Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Speculative decoding improves LLM inference by generating and verifying multiple tokens in parallel, but existing systems suffer from suboptimal performance due to a mismatch between dynamic speculation and static runtime assumptions. We present Yggdrasil, a co-designed system that enables latency-optimal speculative decoding through context-aware tree drafting and compiler-friendly execution. Yggdrasil introduces an equal-growth tree structure for static graph compatibility, a latency-aware optimization objective for draft selection, and stage-based scheduling to reduce overhead. Yggdrasil supports unmodified LLMs and achieves up to $3.98\times$ speedup over state-of-the-art baselines across multiple hardware setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23848v1">Integrating Domain Knowledge for Financial QA: A Multi-Retriever RAG Approach with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      This research project addresses the errors of financial numerical reasoning Question Answering (QA) tasks due to the lack of domain knowledge in finance. Despite recent advances in Large Language Models (LLMs), financial numerical questions remain challenging because they require specific domain knowledge in finance and complex multi-step numeric reasoning. We implement a multi-retriever Retrieval Augmented Generators (RAG) system to retrieve both external domain knowledge and internal question contexts, and utilize the latest LLM to tackle these tasks. Through comprehensive ablation experiments and error analysis, we find that domain-specific training with the SecBERT encoder significantly contributes to our best neural symbolic model surpassing the FinQA paper's top model, which serves as our baseline. This suggests the potential superior performance of domain-specific training. Furthermore, our best prompt-based LLM generator achieves the state-of-the-art (SOTA) performance with significant improvement (>7%), yet it is still below the human expert performance. This study highlights the trade-off between hallucinations loss and external knowledge gains in smaller models and few-shot examples. For larger models, the gains from external facts typically outweigh the hallucination loss. Finally, our findings confirm the enhanced numerical reasoning capabilities of the latest LLM, optimized for few-shot learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23847v1">A Test of Lookahead Bias in LLM Forecasts</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      We develop a statistical test to detect lookahead bias in economic forecasts generated by large language models (LLMs). Using state-of-the-art pre-training data detection techniques, we estimate the likelihood that a given prompt appeared in an LLM's training corpus, a statistic we term Lookahead Propensity (LAP). We formally show that a positive correlation between LAP and forecast accuracy indicates the presence and magnitude of lookahead bias, and apply the test to two forecasting tasks: news headlines predicting stock returns and earnings call transcripts predicting capital expenditures. Our test provides a cost-efficient, diagnostic tool for assessing the validity and reliability of LLM-generated forecasts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23836v1">Retrieval Augmented Question Answering: When Should LLMs Admit Ignorance?</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      The success of expanded context windows in Large Language Models (LLMs) has driven increased use of broader context in retrieval-augmented generation. We investigate the use of LLMs for retrieval augmented question answering. While longer contexts make it easier to incorporate targeted knowledge, they introduce more irrelevant information that hinders the model's generation process and degrades its performance. To address the issue, we design an adaptive prompting strategy which involves splitting the retrieved information into smaller chunks and sequentially prompting a LLM to answer the question using each chunk. Adjusting the chunk size allows a trade-off between incorporating relevant information and reducing irrelevant information. Experimental results on three open-domain question answering datasets demonstrate that the adaptive strategy matches the performance of standard prompting while using fewer tokens. Our analysis reveals that when encountering insufficient information, the LLM often generates incorrect answers instead of declining to respond, which constitutes a major source of error. This finding highlights the need for further research into enhancing LLMs' ability to effectively decline requests when faced with inadequate information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23765v1">Entropy-Aware Speculative Decoding Toward Improved LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD) accelerates large language model (LLM) reasoning by using a small draft model to generate candidate tokens, which the target LLM either accepts directly or regenerates upon rejection. However, excessive alignment between the draft and target models constrains SD to the performance of the target LLM. To address this limitation, we propose Entropy-Aware Speculative Decoding (EASD), a training-free enhancement. Building on standard SD, EASD incorporates a dynamic entropy-based penalty. At each decoding step, we employ the entropy of the sampling distribution to quantify model uncertainty. When both models exhibit high entropy with substantial overlap among their top-N predictions, the corresponding token is rejected and re-sampled by the target LLM. This penalty prevents low-confidence errors from propagating. By incorporating draft-model verification, EASD enables the possibility of surpassing the target model's inherent performance. Experiments across multiple reasoning benchmarks demonstrate that EASD consistently outperforms existing SD methods and, in most cases, surpasses the target LLM itself. We further prove that the efficiency of EASD is comparable to that of SD. The code can be found in the Supplementary Materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00845v1">Enhancing Temporal Awareness in LLMs for Temporal Point Processes</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are crucial for analyzing events over time and are widely used in fields such as finance, healthcare, and social systems. These processes are particularly valuable for understanding how events unfold over time, accounting for their irregularity and dependencies. Despite the success of large language models (LLMs) in sequence modeling, applying them to temporal point processes remains challenging. A key issue is that current methods struggle to effectively capture the complex interaction between temporal information and semantic context, which is vital for accurate event modeling. In this context, we introduce TPP-TAL (Temporal Point Processes with Enhanced Temporal Awareness in LLMs), a novel plug-and-play framework designed to enhance temporal reasoning within LLMs. Rather than using the conventional method of simply concatenating event time and type embeddings, TPP-TAL explicitly aligns temporal dynamics with contextual semantics before feeding this information into the LLM. This alignment allows the model to better perceive temporal dependencies and long-range interactions between events and their surrounding contexts. Through comprehensive experiments on several benchmark datasets, it is shown that TPP-TAL delivers substantial improvements in temporal likelihood estimation and event prediction accuracy, highlighting the importance of enhancing temporal awareness in LLMs for continuous-time event modeling. The code is made available at https://github.com/chenlilil/TPP-TAL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23097v1">A Note on Hybrid Online Reinforcement and Imitation Learning for LLMs: Formulations and Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      We present a unified framework for Large Language Model (LLM) fine-tuning that integrates Imitation Learning and Reinforcement Learning. By analyzing the gradient of a composite objective combining trajectory-level KL divergence with task rewards, we derive a natural decomposition into two components: (1) an analytically computable Dense Gradient for token-level imitation, and (2) a Monte Carlo estimated Sparse Gradient for long-horizon reward optimization. The Dense Gradient admits a closed-form logit-level formula, enabling efficient GPU implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23087v1">Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Reinforcement learning for large language models (LLMs) faces a fundamental tension: high-throughput inference engines and numerically-precise training systems produce different probability distributions from the same parameters, creating a training-inference mismatch. We prove this mismatch has an asymmetric effect: the bound on log-probability mismatch scales as $(1-p)$ where $p$ is the token probability. For high-probability tokens, this bound vanishes, contributing negligibly to sequence-level mismatch. For low-probability tokens in the tail, the bound remains large, and moreover, when sampled, these tokens exhibit systematically biased mismatches that accumulate over sequences, destabilizing gradient estimation. Rather than applying post-hoc corrections, we propose constraining the RL objective to a dynamically-pruned ``safe'' vocabulary that excludes the extreme tail. By pruning such tokens, we trade large, systematically biased mismatches for a small, bounded optimization bias. Empirically, our method achieves stable training; theoretically, we bound the optimization bias introduced by vocabulary pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23075v1">Trust Region Masking for Long-Horizon LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Policy gradient methods for large language models optimize a surrogate objective computed from samples of a rollout policy $π_{\text{roll}}$. When $π_{\text{roll}} \ne π_θ$, there is approximation error between the surrogate and the true objective. Prior work has shown that this off-policy mismatch is unavoidable in modern LLM-RL due to implementation divergence, mixture-of-experts routing discontinuities, and distributed training staleness. Classical trust region bounds on the resulting error scale as $O(T^2)$ with sequence length $T$, rendering them vacuous for long-horizon tasks. We derive two tighter bounds: a Pinsker-Marginal bound scaling as $O(T^{3/2})$ and a Mixed bound scaling as $O(T)$. Crucially, both bounds depend on $D_{kl}^{tok,max}$ -- the maximum token-level KL divergence across all positions in a sequence. This is inherently a sequence-level quantity: it requires examining the entire trajectory to compute, and therefore cannot be controlled by token-independent methods like PPO clipping. We propose Trust Region Masking (TRM), which excludes entire sequences from gradient computation if any token violates the trust region, providing the first non-vacuous monotonic improvement guarantees for long-horizon LLM-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23036v1">Problems With Large Language Models for Learner Modelling: Why LLMs Alone Fall Short for Responsible Tutoring in K--12 Education</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      The rapid rise of large language model (LLM)-based tutors in K--12 education has fostered a misconception that generative models can replace traditional learner modelling for adaptive instruction. This is especially problematic in K--12 settings, which the EU AI Act classifies as high-risk domain requiring responsible design. Motivated by these concerns, this study synthesises evidence on limitations of LLM-based tutors and empirically investigates one critical issue: the accuracy, reliability, and temporal coherence of assessing learners' evolving knowledge over time. We compare a deep knowledge tracing (DKT) model with a widely used LLM, evaluated zero-shot and fine-tuned, using a large open-access dataset. Results show that DKT achieves the highest discrimination performance (AUC = 0.83) on next-step correctness prediction and consistently outperforms the LLM across settings. Although fine-tuning improves the LLM's AUC by approximately 8\% over the zero-shot baseline, it remains 6\% below DKT and produces higher early-sequence errors, where incorrect predictions are most harmful for adaptive support. Temporal analyses further reveal that DKT maintains stable, directionally correct mastery updates, whereas LLM variants exhibit substantial temporal weaknesses, including inconsistent and wrong-direction updates. These limitations persist despite the fine-tuned LLM requiring nearly 198 hours of high-compute training, far exceeding the computational demands of DKT. Our qualitative analysis of multi-skill mastery estimation further shows that, even after fine-tuning, the LLM produced inconsistent mastery trajectories, while DKT maintained smooth and coherent updates. Overall, the findings suggest that LLMs alone are unlikely to match the effectiveness of established intelligent tutoring systems, and that responsible tutoring requires hybrid frameworks that incorporate learner modelling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23029v1">Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has been accompanied by a reliance on cloud-based, proprietary systems, raising significant concerns regarding data privacy, operational sovereignty, and escalating costs. This paper investigates the feasibility of deploying a high-performance, private LLM inference server at a cost accessible to Small and Medium Businesses (SMBs). We present a comprehensive benchmarking analysis of a locally hosted, quantized 30-billion parameter Mixture-of-Experts (MoE) model based on Qwen3, running on a consumer-grade server equipped with a next-generation NVIDIA GPU. Unlike cloud-based offerings, which are expensive and complex to integrate, our approach provides an affordable and private solution for SMBs. We evaluate two dimensions: the model's intrinsic capabilities and the server's performance under load. Model performance is benchmarked against academic and industry standards to quantify reasoning and knowledge relative to cloud services. Concurrently, we measure server efficiency through latency, tokens per second, and time to first token, analyzing scalability under increasing concurrent users. Our findings demonstrate that a carefully configured on-premises setup with emerging consumer hardware and a quantized open-source model can achieve performance comparable to cloud-based services, offering SMBs a viable pathway to deploy powerful LLMs without prohibitive costs or privacy compromises.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23025v1">LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 22 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14973v2">Attention Is All You Need for KV Cache in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Code at: https://github.com/VILA-Lab/Elastic-Cache
    </div>
    <details class="paper-abstract">
      This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), and $45.1\times$ on longer sequences, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23014v1">Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive performance across natural language tasks but incur substantial computational and storage costs due to their scale. Post-training structured pruning offers an efficient solution. However, when few-shot calibration sets fail to adequately reflect the pretraining data distribution, existing methods exhibit limited generalization to downstream tasks. To address this issue, we propose Function-Aware Neuron Grouping (FANG), a post-training pruning framework that alleviates calibration bias by identifying and preserving neurons critical to specific function. FANG groups neurons with similar function based on the type of semantic context they process and prunes each group independently. During importance estimation within each group, tokens that strongly correlate with the functional role of the neuron group are given higher weighting. Additionally, FANG also preserves neurons that contribute across multiple context types. To achieve a better trade-off between sparsity and performance, it allocates sparsity to each block adaptively based on its functional complexity. Experiments show that FANG improves downstream accuracy while preserving language modeling performance. It achieves the state-of-the-art (SOTA) results when combined with FLAP and OBC, two representative pruning methods. Specifically, FANG outperforms FLAP and OBC by 1.5%--8.5% in average accuracy under 30% and 40% sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22925v1">Argus: Token Aware Distributed LLM Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly being integrated into real-world applications, yet their autoregressive architectures introduce significant inference time variability, especially when deployed across heterogeneous edge-cloud systems. Existing solutions largely neglect the dynamic, stochastic, and heterogeneous nature of such environments, often ignoring the impact of variable output token lengths and device diversity. In this work, we present Argus, the first token-aware distributed edge-cloud LLM inference framework that conducts efficient task offloading. Argus features a Length-Aware Semantics (LAS) module, which predicts output token lengths for incoming prompts using a fine-tuned language model with token-length-sensitive feature modulation, enabling precise estimation. Building on this, our Lyapunov-guided Offloading Optimization (LOO) module formulates long-term Quality-of-Experience optimization that explicitly considers both LLM prefilling and decoding costs. We introduce a novel Iterative Offloading Algorithm with Damping and Congestion Control (IODCC) to effectively solve the resulting integer nonlinear programming problem under time-varying constraints. Extensive theoretical and empirical evaluations demonstrate that Argus achieves robust performance and superior efficiency in highly dynamic, heterogeneous settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22905v1">JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Accepted by NeurIPS as a Spotlight paper. Code: https://github.com/JavisVerse/JavisGPT
    </div>
    <details class="paper-abstract">
      This paper presents JavisGPT, the first unified multimodal large language model (MLLM) for Joint Audio-Video (JAV) comprehension and generation. JavisGPT adopts a concise encoder-LLM-decoder architecture, featuring a SyncFusion module for spatio-temporal audio-video fusion and synchrony-aware learnable queries to bridge a pretrained JAV-DiT generator. This design enables temporally coherent video-audio understanding and generation from multimodal instructions. We design an effective three-stage training pipeline consisting of multimodal pretraining, audio-video fine-tuning, and large-scale instruction-tuning, to progressively build multimodal comprehension and generation from existing vision-language models. To support this, we further construct JavisInst-Omni, a high-quality instruction dataset with over 200K GPT-4o-curated audio-video-text dialogues that span diverse and multi-level comprehension and generation scenarios. Extensive experiments on JAV comprehension and generation benchmarks show that JavisGPT outperforms existing MLLMs, particularly in complex and temporally synchronized settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22827v1">FasterPy: An LLM-based Code Execution Efficiency Optimization Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 32 pages, 5 images, 7 tables, Manuscript submitted to a Journal (2025)
    </div>
    <details class="paper-abstract">
      Code often suffers from performance bugs. These bugs necessitate the research and practice of code optimization. Traditional rule-based methods rely on manually designing and maintaining rules for specific performance bugs (e.g., redundant loops, repeated computations), making them labor-intensive and limited in applicability. In recent years, machine learning and deep learning-based methods have emerged as promising alternatives by learning optimization heuristics from annotated code corpora and performance measurements. However, these approaches usually depend on specific program representations and meticulously crafted training datasets, making them costly to develop and difficult to scale. With the booming of Large Language Models (LLMs), their remarkable capabilities in code generation have opened new avenues for automated code optimization. In this work, we proposed FasterPy, a low-cost and efficient framework that adapts LLMs to optimize the execution efficiency of Python code. FasterPy combines Retrieval-Augmented Generation (RAG), supported by a knowledge base constructed from existing performance-improving code pairs and corresponding performance measurements, with Low-Rank Adaptation (LoRA) to enhance code optimization performance. Our experimental results on the Performance Improving Code Edits (PIE) benchmark demonstrate that our method outperforms existing models on multiple metrics. The FasterPy tool and the experimental results are available at https://github.com/WuYue22/fasterpy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.15759v2">Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 21 pages, 7 figures; Accepted by IEEE TIP
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual understanding and reasoning, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance the overall capabilities of LLMs, which could be regarded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce Modular Visual Memory (MVM), a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixture of Multimodal Experts (MoMEs) architecture in LLMs to invoke multimodal knowledge collaboration during text generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on image-text understanding multimodal benchmarks. The codes will be available at: https://github.com/HITsz-TMG/MKS2-Multimodal-Knowledge-Storage-and-Sharing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v3">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22790v1">ChatGraPhT: A Visual Conversation Interface for Multi-Path Reflection with Agentic LLM Support</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Early-stage research prototype; exploratory study
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in complex knowledge work, yet linear transcript interfaces limit support for reflection. Schon's Reflective Practice distinguishes between reflection-in-action (during a task) and reflection-on-action (after a task), both benefiting from non-linear, revisitable representations of dialogue. ChatGraPhT is an interactive tool that shows dialogue as a visual map, allowing users to branch and merge ideas, edit past messages, and receive guidance that prompts deeper reflection. It supports non-linear, multi-path dialogue, while two agentic LLM assistants provide moment-to-moment and higher-level guidance. Our inquiry suggests that keeping the conversation structure visible, allowing branching and merging, and suggesting patterns or ways to combine ideas deepened user reflective engagement. Contributions are: (1) the design of a node-link, agentic LLM interface for reflective dialogue, and (2) transferable design knowledge on balancing structure and AI support to sustain reflection in complex, open-ended tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22753v1">From Rookie to Expert: Manipulating LLMs for Automated Vulnerability Exploitation in Enterprise Software</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      LLMs democratize software engineering by enabling non-programmers to create applications, but this same accessibility fundamentally undermines security assumptions that have guided software engineering for decades. We show in this work how publicly available LLMs can be socially engineered to transform novices into capable attackers, challenging the foundational principle that exploitation requires technical expertise. To that end, we propose RSA (Role-assignment, Scenario-pretexting, and Action-solicitation), a pretexting strategy that manipulates LLMs into generating functional exploits despite their safety mechanisms. Testing against Odoo -- a widely used ERP platform, we evaluated five mainstream LLMs (GPT-4o, Gemini, Claude, Microsoft Copilot, and DeepSeek) and achieved a 100% success rate: tested CVE yielded at least one working exploit within 3-4 prompting rounds. While prior work [13] found LLM-assisted attacks difficult and requiring manual effort, we demonstrate that this overhead can be eliminated entirely. Our findings invalidate core software engineering security principles: the distinction between technical and non-technical actors no longer provides valid threat models; technical complexity of vulnerability descriptions offers no protection when LLMs can abstract it away; and traditional security boundaries dissolve when the same tools that build software can be manipulated to break it. This represents a paradigm shift in software engineering -- we must redesign security practices for an era where exploitation requires only the ability to craft prompts, not understand code. Artifacts available at: https://anonymous.4open.science/r/From-Rookie-to-Attacker-D8B3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22742v1">Robust LLM-based Column Type Annotation via Prompt Augmentation with LoRA Tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Column Type Annotation (CTA) is a fundamental step towards enabling schema alignment and semantic understanding of tabular data. Existing encoder-only language models achieve high accuracy when fine-tuned on labeled columns, but their applicability is limited to in-domain settings, as distribution shifts in tables or label spaces require costly re-training from scratch. Recent work has explored prompting generative large language models (LLMs) by framing CTA as a multiple-choice task, but these approaches face two key challenges: (1) model performance is highly sensitive to subtle changes in prompt wording and structure, and (2) annotation F1 scores remain modest. A natural extension is to fine-tune large language models. However, fully fine-tuning these models incurs prohibitive computational costs due to their scale, and the sensitivity to prompts is not eliminated. In this paper, we present a parameter-efficient framework for CTA that trains models over prompt-augmented data via Low-Rank Adaptation (LoRA). Our approach mitigates sensitivity to prompt variations while drastically reducing the number of necessary trainable parameters, achieving robust performance across datasets and templates. Experimental results on recent benchmarks demonstrate that models fine-tuned with our prompt augmentation strategy maintain stable performance across diverse prompt patterns during inference and yield higher weighted F1 scores than those fine-tuned on a single prompt template. These results highlight the effectiveness of parameter-efficient training and augmentation strategies in developing practical and adaptable CTA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10321v2">AdvPrefix: An Objective for Nuanced LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix ``Sure, here is (harmful request)''. While straightforward, this objective has two limitations: limited control over model behaviors, yielding incomplete or unrealistic jailbroken responses, and a rigid format that hinders optimization. We introduce AdvPrefix, a plug-and-play prefix-forcing objective that selects one or more model-dependent prefixes by combining two criteria: high prefilling attack success rates and low negative log-likelihood. AdvPrefix integrates seamlessly into existing jailbreak attacks to mitigate the previous limitations for free. For example, replacing GCG's default prefixes on Llama-3 improves nuanced attack success rates from 14% to 80%, revealing that current safety alignment fails to generalize to new prefixes. Code and selected prefixes are released at github.com/facebookresearch/jailbreak-objectives.
    </details>
</div>
