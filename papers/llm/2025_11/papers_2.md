# llm - 2025_11

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22275v1">RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language models are revolutionizing the conversational recommender systems through their impressive capabilities in instruction comprehension, reasoning, and human interaction. A core factor underlying effective recommendation dialogue is the ability to infer and reason about users' mental states (such as desire, intention, and belief), a cognitive capacity commonly referred to as Theory of Mind. Despite growing interest in evaluating ToM in LLMs, current benchmarks predominantly rely on synthetic narratives inspired by Sally-Anne test, which emphasize physical perception and fail to capture the complexity of mental state inference in realistic conversational settings. Moreover, existing benchmarks often overlook a critical component of human ToM: behavioral prediction, the ability to use inferred mental states to guide strategic decision-making and select appropriate conversational actions for future interactions. To better align LLM-based ToM evaluation with human-like social reasoning, we propose RecToM, a novel benchmark for evaluating ToM abilities in recommendation dialogues. RecToM focuses on two complementary dimensions: Cognitive Inference and Behavioral Prediction. The former focus on understanding what has been communicated by inferring the underlying mental states. The latter emphasizes what should be done next, evaluating whether LLMs can leverage these inferred mental states to predict, select, and assess appropriate dialogue strategies. Extensive experiments on state-of-the-art LLMs demonstrate that RecToM poses a significant challenge. While the models exhibit partial competence in recognizing mental states, they struggle to maintain coherent, strategic ToM reasoning throughout dynamic recommendation dialogues, particularly in tracking evolving intentions and aligning conversational strategies with inferred mental states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24765v3">From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Logical reasoning is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between logical complexity and semantic complexity, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances, which are central to human reasoning. For this gap, we propose LogicAgent, a semiotic-square-guided framework designed to jointly address logical complexity and semantic complexity. LogicAgent explicitly performs multi-perspective deduction in first-order logic (FOL), while mitigating vacuous reasoning through existential import checks that incorporate a three-valued decision scheme (True, False, Uncertain) to handle boundary cases more faithfully. Furthermore, to overcome the semantic simplicity and low logical complexity of existing datasets, we introduce RepublicQA, a benchmark that reaches college-level difficulty (FKGL = 11.94) and exhibits substantially greater lexical and structural diversity than prior benchmarks. RepublicQA is grounded in philosophical concepts, featuring abstract propositions and systematically organized contrary and contradictory relations, making it the most semantically rich resource for evaluating logical reasoning. Experiments demonstrate that LogicAgent achieves state-of-the-art performance on RepublicQA, with a 6.25% average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05% average gain. These results highlight the strong effectiveness of our semiotic-grounded multi-perspective reasoning in boosting LLMs' logical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16607v3">MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Text-to-SQL is a fundamental yet challenging task in the NLP area, aiming at translating natural language questions into SQL queries. While recent advances in large language models have greatly improved performance, most existing approaches depend on models with tens of billions of parameters or costly APIs, limiting their applicability in resource-constrained environments. For real world, especially on edge devices, it is crucial for Text-to-SQL to ensure cost-effectiveness. Therefore, enabling the light-weight models for Text-to-SQL is of great practical significance. However, smaller LLMs often struggle with complicated user instruction, redundant schema linking or syntax correctness. To address these challenges, we propose MCTS-SQL, a novel framework that uses Monte Carlo Tree Search to guide SQL generation through multi-step refinement. Since the light-weight models' weak performance of single-shot prediction, we generate better results through several trials with feedback. However, directly applying MCTS-based methods inevitably leads to significant time and computational overhead. Driven by this issue, we propose a token-level prefix-cache mechanism that stores prior information during iterations, effectively improved the execution speed. Experiments results on the SPIDER and BIRD benchmarks demonstrate the effectiveness of our approach. Using a small open-source Qwen2.5-Coder-1.5B, our method outperforms ChatGPT-3.5. When leveraging a more powerful model Gemini 2.5 to explore the performance upper bound, we achieved results competitive with the SOTA. Our findings demonstrate that even small models can be effectively deployed in practical Text-to-SQL systems with the right strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.19657v6">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted as International Conference of Computational Optimization 2025 Oral
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedup in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13142v4">Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Codebase: https://github.com/EvolvingLMMs-Lab/EASI/; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
    </div>
    <details class="paper-abstract">
      Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09936v2">KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 14 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Efficient inference of large language models (LLMs) is hindered by an ever-growing key-value (KV) cache, making KV cache compression a critical research direction. Traditional methods selectively evict less important KV cache entries, which leads to information loss and hallucinations. Recently, merging-based strategies have been explored to retain more information by merging KV pairs that would be discarded; however, these existing approaches inevitably introduce inconsistencies in attention distributions before and after merging, causing degraded generation quality. To overcome this challenge, we propose KeepKV, a novel adaptive KV cache merging method designed to preserve performance under strict memory constraints, achieving single-step lossless compression and providing error bounds for multi-step compression. KeepKV introduces the Electoral Votes mechanism that records merging history and adaptively adjusts attention scores. Moreover, it further leverages a novel Zero Inference-Perturbation Merging method, compensating for attention loss resulting from cache merging. Extensive experiments on various benchmarks and LLM architectures demonstrate that KeepKV substantially reduces memory usage while successfully retaining essential context information, achieving over 2x inference throughput improvement and maintaining superior generation quality even with only 10% KV cache budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22176v1">Focused Chain-of-Thought: Efficient LLM Reasoning via Structured Input Information</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Recent large language models achieve strong reasoning performance by generating detailed chain-of-thought traces, but this often leads to excessive token use and high inference latency. Existing efficiency approaches typically focus on model-centric interventions, such as reinforcement learning or supervised fine-tuning, to reduce verbosity. In contrast, we propose a training-free, input-centric approach. Inspired by cognitive psychology, we introduce Focused Chain-of-Thought (F-CoT), which separates information extraction from the reasoning process. F-CoT first organizes the essential information from a query into a concise, structured context and then guides the model to reason exclusively over this context. By preventing attention to irrelevant details, F-CoT naturally produces shorter reasoning paths. On arithmetic word problems, F-CoT reduces generated tokens by 2-3x while maintaining accuracy comparable to standard zero-shot CoT. These results highlight structured input as a simple yet effective lever for more efficient LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22153v1">A Theoretically Grounded Hybrid Ensemble for Reliable Detection of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has blurred the line between human and machine authorship, creating practical risks for academic integrity and information reliability. Existing text detectors typically rely on a single methodological paradigm and suffer from poor generalization and high false positive rates (FPR), especially on high-stakes academic text. We propose a theoretically grounded hybrid ensemble that systematically fuses three complementary detection paradigms: (i) a RoBERTa-based transformer classifier for deep semantic feature extraction, (ii) a GPT-2-based probabilistic detector using perturbation-induced likelihood curvature, and (iii) a statistical linguistic feature analyzer capturing stylometric patterns. The core novelty lies in an optimized weighted voting framework, where ensemble weights are learned on the probability simplex to maximize F1-score rather than set heuristically. We provide a bias-variance analysis and empirically demonstrate low inter-model correlation (rho ~ 0.35-0.42), a key condition for variance reduction. Evaluated on a large-scale, multigenerator corpus of 30,000 documents, our system achieves 94.2% accuracy and an AUC of 0.978, with a 35% relative reduction in false positives on academic text. This yields a more reliable and ethically responsible detector for real-world deployment in education and other high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v4">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20942v1">Improving Procedural Skill Explanations via Constrained Generation: A Symbolic-LLM Hybrid Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      In procedural skill learning, instructional explanations must convey not just steps, but the causal, goal-directed, and compositional logic behind them. Large language models (LLMs) often produce fluent yet shallow responses that miss this structure. We present Ivy, an AI coaching system that delivers structured, multi-step explanations by combining symbolic Task-Method-Knowledge (TMK) models with a generative interpretation layer-an LLM that constructs explanations while being constrained by TMK structure. TMK encodes causal transitions, goal hierarchies, and problem decompositions, and guides the LLM within explicit structural bounds. We evaluate Ivy against responses against GPT and retrieval-augmented GPT baselines using expert and independent annotations across three inferential dimensions. Results show that symbolic constraints consistently improve the structural quality of explanations for "how" and "why" questions. This study demonstrates a scalable AI for education approach that strengthens the pedagogical value of AI-generated explanations in intelligent coaching systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21930v1">A Comparative Study of LLM Prompting and Fine-Tuning for Cross-genre Authorship Attribution on Chinese Lyrics</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      We propose a novel study on authorship attribution for Chinese lyrics, a domain where clean, public datasets are sorely lacking. Our contributions are twofold: (1) we create a new, balanced dataset of Chinese lyrics spanning multiple genres, and (2) we develop and fine-tune a domain-specific model, comparing its performance against zero-shot inference using the DeepSeek LLM. We test two central hypotheses. First, we hypothesize that a fine-tuned model will outperform a zero-shot LLM baseline. Second, we hypothesize that performance is genre-dependent. Our experiments strongly confirm Hypothesis 2: structured genres (e.g. Folklore & Tradition) yield significantly higher attribution accuracy than more abstract genres (e.g. Love & Romance). Hypothesis 1 receives only partial support: fine-tuning improves robustness and generalization in Test1 (real-world data and difficult genres), but offers limited or ambiguous gains in Test2, a smaller, synthetically-augmented set. We show that the design limitations of Test2 (e.g., label imbalance, shallow lexical differences, and narrow genre sampling) can obscure the true effectiveness of fine-tuning. Our work establishes the first benchmark for cross-genre Chinese lyric attribution, highlights the importance of genre-sensitive evaluation, and provides a public dataset and analytical framework for future research. We conclude with recommendations: enlarge and diversify test sets, reduce reliance on token-level data augmentation, balance author representation across genres, and investigate domain-adaptive pretraining as a pathway for improved attribution performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21928v1">Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 In The Thirty-ninth Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) traditionally relies on scalar reward signals, limiting its ability to leverage the rich semantic knowledge often available in real-world tasks. In contrast, humans learn efficiently by combining numerical feedback with language, prior knowledge, and common sense. We introduce Prompted Policy Search (ProPS), a novel RL method that unifies numerical and linguistic reasoning within a single framework. Unlike prior work that augment existing RL components with language, ProPS places a large language model (LLM) at the center of the policy optimization loop-directly proposing policy updates based on both reward feedback and natural language input. We show that LLMs can perform numerical optimization in-context, and that incorporating semantic signals, such as goals, domain knowledge, and strategy hints can lead to more informed exploration and sample-efficient learning. ProPS is evaluated across fifteen Gymnasium tasks, spanning classic control, Atari games, and MuJoCo environments, and compared to seven widely-adopted RL algorithms (e.g., PPO, SAC, TRPO). It outperforms all baselines on eight out of fifteen tasks and demonstrates substantial gains when provided with domain knowledge. These results highlight the potential of unifying semantics and numerics for transparent, generalizable, and human-aligned RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21920v1">Toward Automated and Trustworthy Scientific Analysis and Visualization with LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      As modern science becomes increasingly data-intensive, the ability to analyze and visualize large-scale, complex datasets is critical to accelerating discovery. However, many domain scientists lack the programming expertise required to develop custom data analysis workflows, creating barriers to timely and effective insight. Large language models (LLMs) offer a promising solution by generating executable code from natural language descriptions. In this paper, we investigate the trustworthiness of open-source LLMs in autonomously producing Python scripts for scientific data analysis and visualization. We construct a benchmark suite of domain-inspired prompts that reflect real-world research tasks and systematically evaluate the executability and correctness of the generated code. Our findings show that, without human intervention, the reliability of LLM-generated code is limited, with frequent failures caused by ambiguous prompts and the models' insufficient understanding of domain-specific contexts. To address these challenges, we design and assess three complementary strategies: data-aware prompt disambiguation, retrieval-augmented prompt enhancement, and iterative error repair. While these methods significantly improve execution success rates and output quality, further refinement is needed. This work highlights both the promise and current limitations of LLM-driven automation in scientific workflows and introduces actionable techniques and a reusable benchmark for building more inclusive, accessible, and trustworthy AI-assisted research tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02789v2">Strong Memory, Weak Control: An Empirical Study of Executive Functioning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Working memory, or the ability to hold and manipulate information in the mind, is a critical component of human intelligence and executive functioning. It is correlated with performance on various cognitive tasks, including measures of fluid intelligence, which encompasses reasoning and problem solving. We use a comprehensive set of classic working memory tasks to estimate the working memory capacity of large language models (LLMs). We find that in most cases, LLMs exceed normative human scores. However, we do not find that the increased capacity of working memory is associated with higher performance on other executive functioning tasks or problem solving benchmarks. These results suggest that LLMs may have deficits in attentional control and cognitive flexibility, which result in difficulties with inhibiting automatic responses and adapting to shifting information. Our findings suggest that current reasoning models have mixed results in compensating for these deficits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17554v2">Beyond the Rubric: Cultural Misalignment in LLM Benchmarks for Sexual and Reproductive Health</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 https://github.com/Sumon/healthbench-srh-eval/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been positioned as having the potential to expand access to health information in the Global South, yet their evaluation remains heavily dependent on benchmarks designed around Western norms. We present insights from a preliminary benchmarking exercise with a chatbot for sexual and reproductive health (SRH) for an underserved community in India. We evaluated using HealthBench, a benchmark for conversational health models by OpenAI. We extracted 637 SRH queries from the dataset and evaluated on the 330 single-turn conversations. Responses were evaluated using HealthBench's rubric-based automated grader, which rated responses consistently low. However, qualitative analysis by trained annotators and public health experts revealed that many responses were actually culturally appropriate and medically accurate. We highlight recurring issues, particularly a Western bias, such as for legal framing and norms (e.g., breastfeeding in public), diet assumptions (e.g., fish safe to eat during pregnancy), and costs (e.g., insurance models). Our findings demonstrate the limitations of current benchmarks in capturing the effectiveness of systems built for different cultural and healthcare contexts. We argue for the development of culturally adaptive evaluation frameworks that meet quality standards while recognizing needs of diverse populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21877v1">LLM-Empowered Event-Chain Driven Code Generation for ADAS in SDV systems</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      This paper presents an event-chain-driven, LLM-empowered workflow for generating validated, automotive code from natural-language requirements. A Retrieval-Augmented Generation (RAG) layer retrieves relevant signals from large and evolving Vehicle Signal Specification (VSS) catalogs as code generation prompt context, reducing hallucinations and ensuring architectural correctness. Retrieved signals are mapped and validated before being transformed into event chains that encode causal and timing constraints. These event chains guide and constrain LLM-based code synthesis, ensuring behavioral consistency and real-time feasibility. Based on our initial findings from the emergency braking case study, with the proposed approach, we managed to achieve valid signal usage and consistent code generation without LLM retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21862v1">OOCO: Latency-disaggregated Architecture for Online-Offline Co-locate LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in both latency-sensitive online services and cost-sensitive offline workloads. Co-locating these workloads on shared serving instances can improve resource utilization, but directly applying this approach to Prefill/Decode (P/D) disaggregated systems introduces severe load imbalance, as fluctuating request mixes alter the intrinsic P/D ratio. Existing dynamic adjustment techniques cannot keep up with the bursty traffic patterns of online services. We propose a latency-constraint disaggregated architecture, which separates cluster resources into latency-strict and latency-relaxed pools based on task latency requirements. This design enables flexible placement of offline decode tasks, mitigating P/D imbalance while preserving online performance. To fully exploit this flexibility, we propose (1) a bottleneck-based scheduler guided by a Roofline-based performance model for performance bottleneck based scheduling, and (2) a fast preemption mechanism that strictly enforces Service Level Objectives (SLOs) for online requests. Experiments on real-world traces show that compared to existing offline system approaches, our method improves offline throughput by up to 3x, while maintaining online request SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21788v1">Code Refactoring with LLM: A Comprehensive Evaluation With Few-Shot Settings</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      In today's world, the focus of programmers has shifted from writing complex, error-prone code to prioritizing simple, clear, efficient, and sustainable code that makes programs easier to understand. Code refactoring plays a critical role in this transition by improving structural organization and optimizing performance. However, existing refactoring methods are limited in their ability to generalize across multiple programming languages and coding styles, as they often rely on manually crafted transformation rules. The objectives of this study are to (i) develop an Large Language Models (LLMs)-based framework capable of performing accurate and efficient code refactoring across multiple languages (C, C++, C#, Python, Java), (ii) investigate the impact of prompt engineering (Temperature, Different shot algorithm) and instruction fine-tuning on refactoring effectiveness, and (iii) evaluate the quality improvements (Compilability, Correctness, Distance, Similarity, Number of Lines, Token, Character, Cyclomatic Complexity) in refactored code through empirical metrics and human assessment. To accomplish these goals, we propose a fine-tuned prompt-engineering-based model combined with few-shot learning for multilingual code refactoring. Experimental results indicate that Java achieves the highest overall correctness up to 99.99% the 10-shot setting, records the highest average compilability of 94.78% compared to the original source code and maintains high similarity (Approx. 53-54%) and thus demonstrates a strong balance between structural modifications and semantic preservation. Python exhibits the lowest structural distance across all shots (Approx. 277-294) while achieving moderate similarity ( Approx. 44-48%) that indicates consistent and minimally disruptive refactoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21783v1">NetworkGames: Simulating Cooperation in Network Games with Personality-driven LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) presents a novel opportunity to build high-fidelity agent-based models for simulating complex social systems. However, the behavior of these LLM-based agents in game-theoretic network games remains surprisingly unexplored. In this work, we introduce "NetworkGames," a novel simulation framework designed to investigate how network topology and agent personality jointly shape the evolution of cooperation in network games. We instantiate a population of LLM agents, each endowed with a distinct personality from the MBTI taxonomy, and situate them in various network structures (e.g., small-world and scale-free). Through extensive simulations of the Iterated Prisoner's Dilemma, we first establish a baseline dyadic interaction matrix, revealing nuanced cooperative preferences between all 16 personality pairs. We then demonstrate that macro-level cooperative outcomes are not predictable from dyadic interactions alone; they are co-determined by the network's connectivity and the spatial distribution of personalities. For instance, we find that small-world networks are detrimental to cooperation, while strategically placing pro-social personalities in hub positions within scale-free networks can significantly promote cooperative behavior. Our findings offer significant implications for designing healthier online social environments and forecasting collective behavior. We open-source our framework to foster further research in network game simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10507v2">AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21638v1">Aligning LLMs Toward Multi-Turn Conversational Outcomes Using Iterative PPO</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Optimizing large language models (LLMs) for multi-turn conversational outcomes remains a significant challenge, especially in goal-oriented settings like AI marketing or sales agents who facilitate transactions via messaging platforms. The difficulty stems from sparse, long-horizon rewards and the discrepancy between response-level planning and token-level generation. In this technical note, we propose a formal reduction of the multi-turn RL problem into a sequence of single-turn RLHF-style problems. This is achieved by setting a learned multi-turn Q-function as the reward model for the single-turn problem. We demonstrate and prove a key insight: solving this single-turn RL problem with standard token-level PPO is equivalent to a policy improvement step within the multi-turn problem. This insight naturally leads to Iterative PPO, a batch online policy iteration algorithm that alternates between fitting Q-functions from logged conversation trajectories and improving the policy. A major practical advantage is that Iterative PPO directly leverages stable, off-the-shelf single-turn RLHF tools, making it straightforward to implement. Our method occupies a middle ground between fully online and fully offline approaches, retaining the adaptability of online updates while gaining the stability benefits of offline training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21624v1">TAGFN: A Text-Attributed Graph Dataset for Fake News Detection in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently revolutionized machine learning on text-attributed graphs, but the application of LLMs to graph outlier detection, particularly in the context of fake news detection, remains significantly underexplored. One of the key challenges is the scarcity of large-scale, realistic, and well-annotated datasets that can serve as reliable benchmarks for outlier detection. To bridge this gap, we introduce TAGFN, a large-scale, real-world text-attributed graph dataset for outlier detection, specifically fake news detection. TAGFN enables rigorous evaluation of both traditional and LLM-based graph outlier detection methods. Furthermore, it facilitates the development of misinformation detection capabilities in LLMs through fine-tuning. We anticipate that TAGFN will be a valuable resource for the community, fostering progress in robust graph-based outlier detection and trustworthy AI. The dataset is publicly available at https://huggingface.co/datasets/kayzliu/TAGFN and our code is available at https://github.com/kayzliu/tagfn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21613v1">Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal-URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v1">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 24 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings, the sequence-wise geometric patterns in LLaMa, and experiment with repeating token sequences. We aim to support systematic analysis of Transformer internals with the goal of enabling further reproducible interpretability research. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v2">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI 2026 Workshop on Graphs and more Complex Structures For Learning and Reasoning (GCLR). Version 2 fixes typos in author name and Figure 1
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.14746v5">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Leibler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^γ$ where $D$ is the size of the training data and $ γ\in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.17846v2">Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19218v2">Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21401v1">Can LLMs extract human-like fine-grained evidence for evidence-based fact-checking?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Misinformation frequently spreads in user comments under online news articles, highlighting the need for effective methods to detect factually incorrect information. To strongly support or refute claims extracted from such comments, it is necessary to identify relevant documents and pinpoint the exact text spans that justify or contradict each claim. This paper focuses on the latter task -- fine-grained evidence extraction for Czech and Slovak claims. We create new dataset, containing two-way annotated fine-grained evidence created by paid annotators. We evaluate large language models (LLMs) on this dataset to assess their alignment with human annotations. The results reveal that LLMs often fail to copy evidence verbatim from the source text, leading to invalid outputs. Error-rate analysis shows that the {llama3.1:8b model achieves a high proportion of correct outputs despite its relatively small size, while the gpt-oss-120b model underperforms despite having many more parameters. Furthermore, the models qwen3:14b, deepseek-r1:32b, and gpt-oss:20b demonstrate an effective balance between model size and alignment with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21334v1">Emergent Lexical Semantics in Neural Language Models: Testing Martin's Law on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 paper draft
    </div>
    <details class="paper-abstract">
      We present the first systematic investigation of Martin's Law - the empirical relationship between word frequency and polysemy - in text generated by neural language models during training. Using DBSCAN clustering of contextualized embeddings as an operationalization of word senses, we analyze four Pythia models (70M-1B parameters) across 30 training checkpoints. Our results reveal a non-monotonic developmental trajectory: Martin's Law emerges around checkpoint 100, reaches peak correlation (r > 0.6) at checkpoint 104, then degrades by checkpoint 105. Smaller models (70M, 160M) experience catastrophic semantic collapse at late checkpoints, while larger models (410M, 1B) show graceful degradation. The frequency-specificity trade-off remains stable (r $\approx$ -0.3) across all models. These findings suggest that compliance with linguistic regularities in LLM-generated text is not monotonically increasing with training, but instead follows a balanced trajectory with an optimal semantic window. This work establishes a novel methodology for evaluating emergent linguistic structure in neural language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19123v2">Facilitating the Integration of LLMs Into Online Experiments With Simple Chat</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, understanding human-LLM interactions is emerging as a central priority in psychological research. Online experiments offer an efficient means to study human-LLM interactions, yet integrating LLMs into established survey platforms remains technically demanding, particularly when aiming for ecologically valid, real-time conversational experiences with strong experimental control. We introduce Simple Chat, an open-source, research-focused chat interface that streamlines LLM integration for platforms such as Qualtrics, oTree, and LimeSurvey, while presenting a unified participant experience across conditions. Simple Chat connects to both commercial providers and open-weights models, supports streaming responses to preserve conversational flow, and offers an administrative interface for fine-grained control of prompts and interface features. By reducing technical barriers, standardizing interfaces, and improving participant experience, Simple Chat helps advance the study of human-LLM interaction. In this article, we outline Simple Chat's key features, provide a step-by-step tutorial, and demonstrate its utility through two illustrative case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21322v1">TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how such chatbots represent diverse cultures. At the same time, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2,925 annotations from 108 annotators with lived cultural experience from across 71 regions in India and 14 languages. Concerningly, we find that 88\% of the generated stories contain one or more cultural inaccuracies, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. Lastly, we transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of foundational models. Through this evaluation, we surprisingly discover that models often possess the requisite cultural knowledge despite generating stories rife with cultural misrepresentations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21270v1">Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17041v2">CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v1">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20403v2">LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted at 40th IEEE/ACM International Conference on Automated Software Engineering
    </div>
    <details class="paper-abstract">
      Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21140v1">How to Correctly Report LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as evaluators in lieu of humans. While scalable, their judgments are noisy due to imperfect specificity and sensitivity of LLMs, leading to biased accuracy estimates. Although bias-correction methods exist, they are underutilized in LLM research and typically assume exact knowledge of the model's specificity and sensitivity. Furthermore, in general we only have estimates of these values and it is not well known how to properly construct confidence intervals using only estimates. This work presents a simple plug-in framework that corrects such bias and constructs confidence intervals reflecting uncertainty from both test and calibration dataset, enabling practical and statistically sound LLM-based evaluation. Additionally, to reduce uncertainty in the accuracy estimate, we introduce an adaptive algorithm that efficiently allocates calibration sample sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04231v2">Empowering Time Series Forecasting with LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02833v4">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Published as a conference paper at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19933v2">Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being rapidly integrated into decision-support tools, automation workflows, and AI-enabled software systems. However, their behavior in production environments remains poorly understood, and their failure patterns differ fundamentally from those of traditional machine learning models. This paper presents a system-level taxonomy of fifteen hidden failure modes that arise in real-world LLM applications, including multi-step reasoning drift, latent inconsistency, context-boundary degradation, incorrect tool invocation, version drift, and cost-driven performance collapse. Using this taxonomy, we analyze the growing gap in evaluation and monitoring practices: existing benchmarks measure knowledge or reasoning but provide little insight into stability, reproducibility, drift, or workflow integration. We further examine the production challenges associated with deploying LLMs - including observability limitations, cost constraints, and update-induced regressions - and outline high-level design principles for building reliable, maintainable, and cost-aware LLM systems. Finally, we outline high-level design principles for building reliable, maintainable, and cost-aware LLM-based systems. By framing LLM reliability as a system-engineering problem rather than a purely model-centric one, this work provides an analytical foundation for future research on evaluation methodology, AI system robustness, and dependable LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21089v1">MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly deployed as dense transformers, where every parameter in every feed-forward block is activated for every token. While architecturally simple, this is computationally inefficient, since inference costs scale linearly with parameter count. Recent upcycling methods such as MoEfication, CMoE, ToMoE, and MoORE reveal that much of the useful computation lives in sparse, semi-modular substructures inside dense feed-forward networks, but these approaches typically rely on clustering, activation profiling, singular value decomposition, or custom routing that requires calibration data. This paper introduces MLPMoE (MLP Mixture-of-Experts), a training-free, deterministic transformation that restructures the dense MLP in transformer blocks into a static, high-cardinality mixture of experts. The transformation uses simple tensor slicing and summation, reinterpreting the algebra of tensor parallelism as a topological conversion rather than a distributed training pattern. We further introduce Fractal Fade (differential branch sparsity) and Compensated Pruning (variance-preserving branch reduction) as lightweight mechanisms for structured sparsity. On Qwen2.5-0.5B-Instruct and DeepSeek-R1-Distill-Llama-8B, the zero-shot MLPMoE transform changes a proxy perplexity metric by less than 0.05 percent while keeping the parameter count effectively constant. On the 8B model, differential sparsity removes about 20 percent of MLP parameters while keeping perplexity within about 2 percent of the dense baseline. The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training. Code is available at https://gist.github.com/iwallarm/fc2ef1eddf226ca7814f9e5e2ae9bad1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21075v1">Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Effective post-training is essential to align Large Language Models (LLMs) with specialized biomedical knowledge to accelerate life science research. However, current approaches face significant limitations. First, biomedical reasoning involves intricate mechanisms often represented by sparse textual data. Standard Supervised Fine-Tuning (SFT) tends to overfit to surface-level instruction patterns without effectively internalizing this fragmented scientific knowledge. Second, Reinforcement Learning (RL) is impractical for this domain, as defining meaningful rewards often necessitates prohibitive experimental validation (e.g., wet-lab verification of drug responses), rendering real-time feedback unfeasible. We propose Balanced Fine-Tuning (BFT), an efficient post-training method designed to learn complex reasoning from sparse data without external reward signals. BFT operates through a two-layer weighting mechanism: 1. At the token level, it scales loss via prediction probabilities to stabilize gradients and prevent overfitting; 2. At the sample level, it uses "minimum group confidence" to adaptively enhance the learning of hard samples. Experiments demonstrate that BFT significantly outperforms SFT. In medical tasks, it enables LLMs to acquire knowledge that SFT misses. In biological tasks, BFT-based LLMs surpass GeneAgent (an accurate agent for biology analysis) in biological process reasoning. Moreover, the text embeddings generated by BFT can be directly applied to downstream tasks, such as gene interaction and single-cell perturbation response prediction. These results indicate that BFT facilitates broad applications of LLMs in biomedical research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21056v1">A Unified Understanding of Offline Data Selection and Online Self-refining Generation for Post-training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Offline data selection and online self-refining generation, which enhance the data quality, are crucial steps in adapting large language models (LLMs) to specific downstream tasks. We tackle offline data selection and online self-refining generations through an optimization perspective. Specifically, bilevel data selection is used for offline data selection with respect to the validation dataset, and we treat online self-refining generation as a model adaptation step of selecting the model trained on current responses that best fits the validation data. Our framework offers a unified understanding of offline data selection and self-refining generation by assigning a learned data weight to each question and response, either explicitly or implicitly. For the first time, we theoretically demonstrate the effectiveness of the bilevel data selection framework and demonstrate its performance gains over unfiltered direct mixing baselines. By combining offline data with validation-weighted online generations, our method enhances fine-tuning performance. Experiments on quality enhancement and safety-aware LLM fine-tuning validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21050v1">Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI-26 Workshop on Post-AI Formal Methods
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21038v1">Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 13 pages total (7 pages main text, 3 pages references, 3 pages appendix), 2 figures, 14 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl
    </div>
    <details class="paper-abstract">
      Can in-context learning (ICL) override pre-trained label semantics, or does it merely refine an existing semantic backbone? We address this question by treating LLMs as prompt-induced classifiers and contrasting their behavior under \emph{natural} demonstrations (with correct labels) and \emph{inverted} demonstrations (systematically flipping label meanings). We decompose ICL behavior into three alignment metrics (truth, prior, and prompt alignment) and introduce a semantic override rate, defined as correctness under flipped semantics. Across eight classification tasks and eight open-source LLMs (1--12B parameters), we find consistent evidence for a semantic anchor view. With natural demonstrations, ICL improves accuracy while maintaining strong prior alignment; most correct predictions coincide with zero-shot behavior, even when the prior is weak. With inverted demonstrations, models cannot learn coherent anti-semantic classifiers: prompt alignment increases only by sacrificing accuracy, and semantic override rates remain exactly zero in our few-shot 1--12B setting. Rather than flexibly remapping label meanings, ICL primarily adjusts how inputs project onto stable semantic directions learned during pre-training, clarifying fundamental limits of few-shot prompting and suggesting that overriding label semantics at these scales requires interventions beyond ICL. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21037v1">LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted to the PerFM Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Foundation models are increasingly used to personalize learning, yet many systems still assume fixed curricula or coarse progress signals, limiting alignment with learners' day-to-day needs. At the other extreme, lightweight incidental systems offer flexible, in-the-moment content but rarely guide learners toward mastery. Prior work privileges either continuity (maintaining a plan across sessions) or initiative (reacting to the moment), not both, leaving learners to navigate the trade-off between recency and trajectory-immediate relevance versus cumulative, goal-aligned progress. We present LOOM, an agentic pipeline that infers evolving learner needs from recent LLM conversations and a dynamic learner memory graph, then assembles coherent learning materials personalized to the learner's current needs, priorities, and understanding. These materials link adjacent concepts and surface gaps as tightly scoped modules that cumulatively advance broader goals, providing guidance and sustained progress while remaining responsive to new interests. We describe LOOM's end-to-end architecture and working prototype, including conversation summarization, topic planning, course generation, and graph-based progress tracking. In a formative study with ten participants, users reported that LOOM's generated lessons felt relevant to their recent activities and helped them recognize knowledge gaps, though they also highlighted needs for greater consistency and control. We conclude with design implications for more robust, mixed-initiative learning pipelines that integrate structured learner modelling with everyday LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21033v1">Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The rationality of law manifests in two forms: substantive rationality, which concerns the fairness or moral desirability of outcomes, and formal rationality, which requires legal decisions to follow explicitly stated, general, and logically coherent rules. Existing LLM-based systems excel at surface-level text analysis but lack the guarantees required for principled jurisprudence. We introduce L4M, a novel framework that combines adversarial LLM agents with SMT-solver-backed proofs to unite the interpretive flexibility of natural language with the rigor of symbolic verification. The pipeline consists of three phases: (1) Statute Formalization, where domain-specific prompts convert legal provisions into logical formulae; (2) Dual Fact and Statute Extraction, in which prosecutor- and defense-aligned LLMs independently map case narratives to fact tuples and statutes, ensuring role isolation; and (3) Solver-Centric Adjudication, where an autoformalizer compiles both parties' arguments into logic constraints, and unsat cores trigger iterative self-critique until a satisfiable formula is achieved, which is then verbalized by a Judge-LLM into a transparent verdict and optimized sentence. Experimental results on public benchmarks show that our system surpasses advanced LLMs including GPT-o4-mini, DeepSeek-V3, and Claude 4 as well as state-of-the-art Legal AI baselines, while providing rigorous and explainable symbolic justifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21022v1">Lightweight Model Editing for LLMs to Correct Deprecated API Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Pre-trained or fine-tuned on large code corpora, Large Language Models (LLMs) have demonstrated strong performance in code completion tasks. However, their embedded knowledge is constrained by the timeliness of training data, which often includes code using deprecated APIs. Consequently, LLMs frequently generate deprecated APIs that will no longer be supported in future versions of third-party libraries. While retraining LLMs on updated codebases could refresh their API knowledge, this approach is computationally expensive. Recently, lightweight model editing methods have emerged to efficiently correct specific knowledge in LLMs. However, it remains unclear whether these methods can effectively update deprecated API knowledge and enable edited models to generate up-to-date APIs. To address this gap, we conduct the first systematic study applying 10 state-of-the-art model editing techniques to update deprecated API knowledge in three LLMs: Qwen2.5-Coder, StarCoder2, and DeepSeek-Coder. We introduce EDAPIBench, a dedicated benchmark featuring over 70 deprecated APIs from 8 popular Python libraries, with more than 3,000 editing instances. Our results show that the parameter-efficient fine-tuning method AdaLoRA achieves the best performance in enabling edited models to generate correct, up-to-date APIs, but falls short in Specificity (i.e., the editing influences untargeted knowledge). To resolve this, we propose AdaLoRA-L, which defines "Common API Layers" (layers within the LLMs with high importance across all APIs, storing general knowledge and excluded from editing) and restricts edits exclusively to "Specific API Layers" (layers with high importance only for the target API, storing the API-specific knowledge). Experimental results demonstrate that AdaLoRA-L significantly improves Specificity while maintaining comparable performance across other evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20993v1">Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18581v2">TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Many recent studies showed that LLMs are vulnerable to jailbreak attacks, where an attacker can perturb the input of an LLM to induce it to generate an output for a harmful question. In general, existing jailbreak techniques either optimize a semantic template intended to induce the LLM to produce harmful outputs or optimize a suffix that leads the LLM to initiate its response with specific tokens (e.g., "Sure"). In this work, we introduce TASO (Template and Suffix Optimization), a novel jailbreak method that optimizes both a template and a suffix in an alternating manner. Our insight is that suffix optimization and template optimization are complementary to each other: suffix optimization can effectively control the first few output tokens but cannot control the overall quality of the output, while template optimization provides guidance for the entire output but cannot effectively control the initial tokens, which significantly impact subsequent responses. Thus, they can be combined to improve the attack's effectiveness. We evaluate the effectiveness of TASO on benchmark datasets (including HarmBench and AdvBench) on 24 leading LLMs (including models from the Llama family, OpenAI, and DeepSeek). The results demonstrate that TASO can effectively jailbreak existing LLMs. We hope our work can inspire future studies in exploring this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20982v1">A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      To meet strict Service-Level Objectives (SLOs),contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks inherent to each phase. However, the heterogeneity of LLM workloads causes producerconsumer imbalance between the two instance types in such disaggregated architecture. To address this problem, we propose DOPD (Dynamic Optimal Prefill/Decoding), a dynamic LLM inference system that adjusts instance allocations to achieve an optimal prefill-to-decoding (P/D) ratio based on real-time load monitoring. Combined with an appropriate request-scheduling policy, DOPD effectively resolves imbalances between prefill and decoding instances and mitigates resource allocation mismatches due to mixed-length requests under high concurrency. Experimental evaluations show that, compared with vLLM and DistServe (representative aggregation-based and disaggregationbased approaches), DOPD improves overall system goodput by up to 1.5X, decreases P90 time-to-first-token (TTFT) by up to 67.5%, and decreases P90 time-per-output-token (TPOT) by up to 22.8%. Furthermore, our dynamic P/D adjustment technique performs proactive reconfiguration based on historical load, achieving over 99% SLOs attainment while using less additional resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.17264v5">No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Deploying million-token Large Language Models (LLMs) is challenging because production workloads are highly heterogeneous, mixing short queries and long documents. This heterogeneity, combined with the quadratic complexity of attention, creates severe convoy effects where long-running requests stall short, interactive ones, degrading system responsiveness. We present Medha, a serving system that eliminates these convoys by introducing fine-grained, preemptive scheduling to LLM inference. Medha makes preemption practical with a co-designed set of mechanisms -- including Adaptive Chunking and Stream Pipeline Parallel that overcome the perceived inefficiencies and scaling challenges of chunking. Additionally, we present a new parallelism strategy KV-Cache Parallelism to reduce the decode latency and afford interactivity despite very long context. These mechanisms are orchestrated by a Length-Aware Relative Slack (LARS) scheduler, a deadline and heterogeneity-aware scheduling policy that prevents both the convoy effect and the starvation that plagues simpler policies. Under a heterogeneous workload, Medha improves throughput by 5.7x while reducing median and 99th percentile latency by 30x and 174x, respectively, compared to state-of-the-art non-preemptive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20965v1">TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)
    </div>
    <details class="paper-abstract">
      Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14993v3">Multi-modal Generative AI: Multi-modal LLMs, Diffusions, and the Unification</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 21 pages, 10 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20613v1">Can Vibe Coding Beat Graduate CS Students? An LLM vs. Human Coding Tournament on Market-driven Strategic Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has revolutionized AI-assisted code generation. This rapid development of LLMs has outpaced our ability to properly benchmark them. Prevailing benchmarks emphasize unit-test pass rates and syntactic correctness. Such metrics understate the difficulty of many real-world problems that require planning, optimization, and strategic interaction. We introduce a multi-agent reasoning-driven benchmark based on a real-world logistics optimization problem (Auction, Pickup, and Delivery Problem) that couples competitive auctions with capacity-constrained routing. The benchmark requires building agents that can (i) bid strategically under uncertainty and (ii) optimize planners that deliver tasks while maximizing profit. We evaluate 40 LLM-coded agents (by a wide range of state-of-the-art LLMs under multiple prompting methodologies, including vibe coding) against 17 human-coded agents developed before the advent of LLMs. Our results over 12 double all-play-all tournaments and $\sim 40$k matches demonstrate (i) a clear superiority of human(graduate students)-coded agents: the top 5 spots are consistently won by human-coded agents, (ii) the majority of LLM-coded agents (33 out of 40) are beaten by very simple baselines, and (iii) given the best human solution as an input and prompted to improve upon, the best performing LLM makes the solution significantly worse instead of improving it. Our results highlight a gap in LLMs' ability to produce code that works competitively in the real-world, and motivate new evaluations that emphasize reasoning-driven code synthesis in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20604v1">On Evaluating LLM Alignment by Evaluating LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 NeurIPS 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20526v1">Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain underexplored.In China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20468v1">DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and problem-solving.Recent works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic learning.DRAFT-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20403v1">LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted at 40th IEEE/ACM International Conference on Automated Software Engineering
    </div>
    <details class="paper-abstract">
      Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21740v3">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 INLG25
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20340v1">Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 accepted by AAAI-2026
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by utilizing otherwise idle computational resources during memory-to-chip data transfer. Current speculative decoding methods typically assume a considerable amount of available computing power, then generate a complex and massive draft tree using a small autoregressive language model to improve overall prediction accuracy. However, methods like batching have been widely applied in mainstream model inference systems as a superior alternative to speculative decoding, as they compress the available idle computing power. Therefore, performing speculative decoding with low verification resources and low scheduling costs has become an important research problem. We believe that more capable models that allow for parallel generation on draft sequences are what we truly need. Recognizing the fundamental nature of draft models to only generate sequences of limited length, we propose SpecFormer, a novel architecture that integrates unidirectional and bidirectional attention mechanisms. SpecFormer combines the autoregressive model's ability to extract information from the entire input sequence with the parallel generation benefits of non-autoregressive models. This design eliminates the reliance on large prefix trees and achieves consistent acceleration, even in large-batch scenarios. Through lossless speculative decoding experiments across models of various scales, we demonstrate that SpecFormer sets a new standard for scaling LLM inference with lower training demands and reduced computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05130v3">Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI 2026 Workshop WMAC
    </div>
    <details class="paper-abstract">
      Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20284v1">Can LLMs Make (Personalized) Access Control Decisions?</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Precise access control decisions are crucial to the security of both traditional applications and emerging agent-based systems. Typically, these decisions are made by users during app installation or at runtime. Due to the increasing complexity and automation of systems, making these access control decisions can add a significant cognitive load on users, often overloading them and leading to suboptimal or even arbitrary access control decisions. To address this problem, we propose to leverage the processing and reasoning capabilities of large language models (LLMs) to make dynamic, context-aware decisions aligned with the user's security preferences. For this purpose, we conducted a user study, which resulted in a dataset of 307 natural-language privacy statements and 14,682 access control decisions made by users. We then compare these decisions against those made by two versions of LLMs: a general and a personalized one, for which we also gathered user feedback on 1,446 of its decisions. Our results show that in general, LLMs can reflect users' preferences well, achieving up to 86\% accuracy when compared to the decision made by the majority of users. Our study also reveals a crucial trade-off in personalizing such a system: while providing user-specific privacy preferences to the LLM generally improves agreement with individual user decisions, adhering to those preferences can also violate some security best practices. Based on our findings, we discuss design and risk considerations for implementing a practical natural-language-based access control system that balances personalization, security, and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00882v5">VULSOLVER: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, degraded performance with long context, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 97.85% accuracy, 97.97% F1-score, and 100% recall. Applied to widely-used open-source projects, VULSOLVER identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20172v1">Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 13 pages, accepted by SIGMOD'26
    </div>
    <details class="paper-abstract">
      The rapid increase in LLM model sizes and the growing demand for long-context inference have made memory a critical bottleneck in GPU-accelerated serving systems. Although high-bandwidth memory (HBM) on GPUs offers fast access, its limited capacity necessitates reliance on host memory (CPU DRAM) to support larger working sets such as the KVCache. However, the maximum DRAM capacity is constrained by the limited number of memory channels per CPU socket. To overcome this limitation, current systems often adopt RDMA-based disaggregated memory pools, which introduce significant challenges including high access latency, complex communication protocols, and synchronization overhead. Fortunately, the emerging CXL technology introduces new opportunities in KVCache design. In this paper, we propose Beluga, a novel memory architecture that enables GPUs and CPUs to access a shared, large-scale memory pool through CXL switches. By supporting native load/store access semantics over the CXL fabric, our design delivers near-local memory latency, while reducing programming complexity and minimizing synchronization overhead. We conduct a systematic characterization of a commercial CXL switch-based memory pool and propose a set of design guidelines. Based on Beluga, we design and implement Beluga-KVCache, a system tailored for managing the large-scale KVCache in LLM inference. Beluga-KVCache achieves an 89.6% reduction in Time-To-First-Token (TTFT) and 7.35x throughput improvement in the vLLM inference engine compared to RDMA-based solutions. To the best of our knowledge, Beluga is the first system that enables GPUs to directly access large-scale memory pools through CXL switches, marking a significant step toward low-latency, shared access to vast memory resources by GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20100v1">QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages, 2 figures, accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20080v1">Adaptive LLM Agents: Toward Personalized Empathetic Care</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted at workshop Future Wellbeing: Using Design Fiction to Explore Human-Agent Interaction and Mental Health at The 13th International Conference on Human-Agent Interaction (HAI 2025)
    </div>
    <details class="paper-abstract">
      Current mental-health conversational systems are usually based on fixed, generic dialogue patterns. This paper proposes an adaptive framework based on large language models that aims to personalize therapeutic interaction according to a user's psychological state, quantified with the Acceptance of Illness Scale (AIS). The framework defines three specialized agents, L, M, and H, each linked to a different level of illness acceptance, and adjusts conversational behavior over time using continuous feedback signals. The AIS-stratified architecture is treated as a diegetic prototype placed in a plausible near-future setting and examined through the method of design fiction. By embedding the architecture in narrative scenarios, the study explores how such agents might influence access to care and therapeutic relationship. The goal is to show how clinically informed personalization, technical feasibility, and speculative scenario analysis can together inform the responsible design of LLM-based companions for mental-health support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20048v1">Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      LLM-based search agents achieve strong performance but suffer from severe latency, as each step requires serialized LLM reasoning followed by action of tool execution. We revisit this bottleneck through the lens of speculation. While traditional predict-verify speculation paradigm can break serial execution, its benefit remains limited, as it retains the full original workload and adds extra inference overhead. We observe that early agent steps often involve simple evidence-gathering, where correct actions can often be predicted without full reasoning. Building on these observations, we present SPAgent, an algorithm-system co-design framework that expands the role of speculation in search agents to reduce latency. Algorithmically, SPAgent introduces a two-phase adaptive speculation mechanism that selectively omits verification when safe. System-wise, a two-level scheduler regulates speculative requests based on engine load to ensure speculation remains beneficial. We implement SPAgent in real-world systems. Across extensive experimental settings, SPAgent achieves up to $1.65\times$ end-to-end speedup while maintaining same or even achieving higher accuracy, enabling practical deployment of multi-step search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10037v2">Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Existing tool-augmented large language models (LLMs) encounter significant challenges when processing complex queries. Current frameworks such as ReAct are prone to local optimization traps due to their reliance on incremental decision-making processes. To address these limitations, we propose a novel Planner-centric Plan-Execute paradigm that fundamentally resolves local optimization bottlenecks through architectural innovation. Central to our approach is a novel Planner model that performs global Directed Acyclic Graph (DAG) planning for complex queries, enabling optimized execution beyond conventional tool coordination. We also introduce ComplexTool-Plan, a large-scale benchmark dataset featuring complex queries that demand sophisticated multi-tool composition and coordination capabilities. Additionally, we develop a two-stage training methodology that integrates Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), systematically enhancing the Planner's tool selection accuracy and global planning awareness through structured DAG-based planning. When integrated with a capable executor, our framework achieves state-of-the-art performance on the StableToolBench benchmark for complex user queries, demonstrating superior end-to-end execution capabilities and robust handling of intricate multi-tool workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19956v1">Prompt Fairness: Sub-group Disparities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), though shown to be effective in many applications, can vary significantly in their response quality. In this paper, we investigate this problem of prompt fairness: specifically, the phrasing of a prompt by different users/styles, despite the same question being asked in principle, may elicit different responses from an LLM. To quantify this disparity, we propose to use information-theoretic metrics that can capture two dimensions of bias: subgroup sensitivity, the variability of responses within a subgroup and cross group consistency, the variability of responses across subgroups. Our analysis reveals that certain subgroups exhibit both higher internal variability and greater divergence from others. Our empirical analysis reveals that certain demographic sub groups experience both higher internal variability and greater divergence from others, indicating structural inequities in model behavior. To mitigate these disparities, we propose practical interventions, including majority voting across multiple generations and prompt neutralization, which together improve response stability and enhance fairness across user populations. In the experiments, we observe clear prompt sensitivity disparities across demographic subgroups: before mitigation, cross-group divergence values reach 0.28 and typically fall in the from 0.14 to 0.22 range. After applying our neutralization and multi generation strategy, these divergences consistently decrease, with the largest gap reduced to 0.22 and many distances falling to 0.17 or below, indicating more stable and consistent outputs across subgroups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08726v2">Improved LLM Agents for Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 13 pages, 5 figures. Unlike the previous version, LLM names are now unmasked
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19940v1">Editing with AI: How Doctors Refine LLM-Generated Answers to Patient Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Patients frequently seek information during their medical journeys, but the rising volume of digital patient messages has strained healthcare systems. Large language models (LLMs) offer promise in generating draft responses for clinicians, yet how physicians refine these drafts remains underexplored. We present a mixed-methods study with nine ophthalmologists answering 144 cataract surgery questions across three conditions: writing from scratch, directly editing LLM drafts, and instruction-based indirect editing. Our quantitative and qualitative analyses reveal that while LLM outputs were generally accurate, occasional errors and automation bias revealed the need for human oversight. Contextualization--adapting generic answers to local practices and patient expectations--emerged as a dominant form of editing. Editing workflows revealed trade-offs: indirect editing reduced effort but introduced errors, while direct editing ensured precision but with higher workload. We conclude with design and policy implications for building safe, scalable LLM-assisted clinical communication systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19931v1">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {https://anonymous.4open.science/r/LLM-EDT-583F}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19877v1">It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19875v1">CodeFuse-CommitEval: Towards Benchmarking LLM's Power on Commit Message and Code Change Inconsistency Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Version control relies on commit messages to convey the rationale for code changes, but these messages are often low quality and, more critically, inconsistent with their diffs-known as message-code inconsistency (MCI). MCIs mislead reviewers, hinder maintenance, contaminate research datasets, and may obscure security patches. Yet, no dedicated benchmark exists to evaluate models for MCI detection. We introduce CODEFUSE-COMMITEVAL, the first benchmark designed for MCI detection using large language models (LLMs). Built on the ApacheCM dataset for diversity and quality, we generate seven types of inconsistent messages through rule-guided mutations of originally consistent commits and apply two-fold validation to verify both positive and negative samples. Using this labeled dataset of message-diff pairs, we evaluate six state-of-the-art open-source LLMs under a vanilla setting and with three augmentation strategies: few-shot prompting, chain-of-thought, and extended context. Results show models detect inconsistent commits more reliably than consistent ones (average Recall 85.95%, Precision 80.28%, Specificity 63.8%); gpt-oss-20B performs best overall but uses over twice the tokens of others. Augmentation effects vary: adjacent context helps larger models but adds noise for smaller ones; few-shot improves accuracy and reduces token use, yet increases universally incorrect predictions; chain-of-thought boosts precision and specificity at the cost of recall and higher token consumption. Type-wise analysis reveals higher detectability for component, file-path, and operation inconsistencies, but lower accuracy and higher token cost for intent-level "purpose" inconsistencies. CODEFUSE-COMMITEVAL provides a rigorous foundation for measuring, comparing, and advancing MCI detection, highlighting the need for richer context and balanced data to capture high-level semantic gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19874v1">Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 10 pages, 2 figures, 8 tables. Evaluation across 6 production LLMs with 1,198 traces
    </div>
    <details class="paper-abstract">
      As AI agents become integral to enterprise workflows, their reliance on shared tool libraries and pre-trained components creates significant supply chain vulnerabilities. While previous work has demonstrated behavioral backdoor detection within individual LLM architectures, the critical question of cross-LLM generalization remains unexplored, a gap with serious implications for organizations deploying multiple AI systems. We present the first systematic study of cross-LLM behavioral backdoor detection, evaluating generalization across six production LLMs (GPT-5.1, Claude Sonnet 4.5, Grok 4.1, Llama 4 Maverick, GPT-OSS 120B, and DeepSeek Chat V3.1). Through 1,198 execution traces and 36 cross-model experiments, we quantify a critical finding: single-model detectors achieve 92.7% accuracy within their training distribution but only 49.2% across different LLMs, a 43.4 percentage point generalization gap equivalent to random guessing. Our analysis reveals that this gap stems from model-specific behavioral signatures, particularly in temporal features (coefficient of variation > 0.8), while structural features remain stable across architectures. We show that model-aware detection incorporating model identity as an additional feature achieves 90.6% accuracy universally across all evaluated models. We release our multi-LLM trace dataset and detection framework to enable reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01265v3">AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning a decade of reporting from 2015 to 2025. Designed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a realistic benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we investigate the impact of domain specificity on abstractive summarisation of Arabic financial texts with large language models (LLMs). In particular, we evaluate transformer-based models: mT5, AraT5, and the domain-adapted FinAraT5 to examine how financial-domain pretraining influences accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more coherent summaries, especially in their handling of quantitative and entity-centric information. These findings highlight the importance of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-uk/AraFinNews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14977v2">SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19852v1">Profile-LLM: Dynamic Profile Optimization for Realistic Personality Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) have been shown to be an effective way to create more engaging and enjoyable user-AI interactions. While previous studies have explored using prompts to elicit specific personality traits in LLMs, they have not optimized these prompts to maximize personality expression. To address this limitation, we propose PersonaPulse: Dynamic Profile Optimization for Realistic Personality Expression in LLMs, a framework that leverages LLMs' inherent knowledge of personality traits to iteratively enhance role-play prompts while integrating a situational response benchmark as a scoring tool, ensuring a more realistic and contextually grounded evaluation to guide the optimization process. Quantitative evaluations demonstrate that the prompts generated by PersonaPulse outperform those of prior work, which were designed based on personality descriptions from psychological studies. Additionally, we explore the relationship between model size and personality modeling through extensive experiments. Finally, we find that, for certain personality traits, the extent of personality evocation can be partially controlled by pausing the optimization process. These findings underscore the importance of prompt optimization in shaping personality expression within LLMs, offering valuable insights for future research on adaptive AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16690v5">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $τ$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $τ$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $τ$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17648v4">Simulating Macroeconomic Expectations using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for simulating macroeconomic expectations using LLM Agents. By constructing LLM Agents equipped with various functional modules, we replicate three representative survey experiments involving several expectations across different types of economic agents. Our results show that although the expectations simulated by LLM Agents are more homogeneous than those of humans, they consistently outperform LLMs relying simply on prompt engineering, and possess human-like mental mechanisms. Evaluation reveals that these capabilities stem from the contributions of their components, offering guidelines for their architectural design. Our approach complements traditional methods and provides new insights into AI behavioral science in macroeconomic research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19830v1">Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native Query Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Multi-modal analytical processing has the potential to transform applications in e-commerce, healthcare, entertainment, and beyond. However, real-world adoption remains elusive due to the limited ability of traditional relational query operators to capture query semantics. The emergence of foundation models, particularly the large language models (LLMs), opens up new opportunities to develop flexible, semantic-aware data analytics systems that transcend the relational paradigm. We present Nirvana, a multi-modal data analytics framework that incorporates programmable semantic operators while leveraging both logical and physical query optimization strategies, tailored for LLM-driven semantic query processing. Nirvana addresses two key challenges. First, it features an agentic logical optimizer that uses natural language-specified transformation rules and random-walk-based search to explore vast spaces of semantically equivalent query plans -- far beyond the capabilities of conventional optimizers. Second, it introduces a cost-aware physical optimizer that selects the most effective LLM backend for each operator using a novel improvement-score metric. To further enhance efficiency, Nirvana incorporates computation reuse and evaluation pushdown techniques guided by model capability hypotheses. Experimental evaluations on three real-world benchmarks demonstrate that Nirvana is able to reduce end-to-end runtime by 10%--85% and reduces system processing costs by 76% on average, outperforming state-of-the-art systems at both efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v2">Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Polished the abstract and replaced the demonstration screenshots
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v3">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20910v1">Emergence and Localisation of Semantic Role Circuits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Despite displaying semantic competence, large language models' internal mechanisms that ground abstract semantic structure remain insufficiently characterised. We propose a method integrating role-cross minimal pairs, temporal emergence analysis, and cross-model comparison to study how LLMs implement semantic roles. Our analysis uncovers: (i) highly concentrated circuits (89-94% attribution within 28 nodes); (ii) gradual structural refinement rather than phase transitions, with larger models sometimes bypassing localised circuits; and (iii) moderate cross-scale conservation (24-59% component overlap) alongside high spectral similarity. These findings suggest that LLMs form compact, causally isolated mechanisms for abstract semantic structure, and these mechanisms exhibit partial transfer across scales and architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09606v2">How Can We Effectively Use LLMs for Phishing Detection?: Evaluating the Effectiveness of Large Language Model-based Phishing Detection Models</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a promising phishing detection mechanism, addressing the limitations of traditional deep learning-based detectors, including poor generalization to previously unseen websites and a lack of interpretability. However, LLMs' effectiveness for phishing detection remains unexplored. This study investigates how to effectively leverage LLMs for phishing detection (including target brand identification) by examining the impact of input modalities (screenshots, logos, HTML, and URLs), temperature settings, and prompt engineering strategies. Using a dataset of 19,131 real-world phishing websites and 243 benign sites, we evaluate seven LLMs -- two commercial models (GPT 4.1 and Gemini 2.0 flash) and five open-source models (Qwen, Llama, Janus, DeepSeek-VL2, and R1) -- alongside two deep learning (DL)-based baselines (PhishIntention and Phishpedia). Our findings reveal that commercial LLMs generally outperform open-source models in phishing detection, while DL models demonstrate better performance on benign samples. For brand identification, screenshot inputs achieve optimal results, with commercial LLMs reaching 93-95% accuracy and open-source models, particularly Qwen, achieving up to 92%. However, incorporating multiple input modalities simultaneously or applying one-shot prompts does not consistently enhance performance and may degrade results. Furthermore, higher temperature values reduce performance. Based on these results, we recommend using screenshot inputs with zero temperature to maximize accuracy for LLM-based detectors with HTML serving as auxiliary context when screenshot information is insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20878v1">Supporting Students in Navigating LLM-Generated Insecure Code</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      The advent of Artificial Intelligence (AI), particularly large language models (LLMs), has revolutionized software development by enabling developers to specify tasks in natural language and receive corresponding code, boosting productivity. However, this shift also introduces security risks, as LLMs may generate insecure code that can be exploited by adversaries. Current educational approaches emphasize efficiency while overlooking these risks, leaving students underprepared to identify and mitigate security issues in AI-assisted workflows. To address this gap, we present Bifröst, an educational framework that cultivates security awareness in AI-augmented development. Bifröst integrates (1) a Visual Studio Code extension simulating realistic environments, (2) adversarially configured LLMs that generate insecure code, and (3) a feedback system highlighting vulnerabilities. By immersing students in tasks with compromised LLMs and providing targeted security analysis, Bifröst cultivates critical evaluation skills; classroom deployments (n=61) show vulnerability to insecure code, while a post-intervention survey (n=21) indicates increased skepticism toward LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20872v1">Winning with Less for Low Resource Languages: Advantage of Cross-Lingual English_Persian Argument Mining Model over LLM Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Argument mining is a subfield of natural language processing to identify and extract the argument components, like premises and conclusions, within a text and to recognize the relations between them. It reveals the logical structure of texts to be used in tasks like knowledge extraction. This paper aims at utilizing a cross-lingual approach to argument mining for low-resource languages, by constructing three training scenarios. We examine the models on English, as a high-resource language, and Persian, as a low-resource language. To this end, we evaluate the models based on the English Microtext corpus \citep{PeldszusStede2015}, and its parallel Persian translation. The learning scenarios are as follow: (i) zero-shot transfer, where the model is trained solely with the English data, (ii) English-only training enhanced by synthetic examples generated by Large Language Models (LLMs), and (iii) a cross-lingual model that combines the original English data with manually translated Persian sentences. The zero-shot transfer model attains F1 scores of 50.2\% on the English test set and 50.7\% on the Persian test set. LLM-based augmentation model improves the performance up to 59.2\% on English and 69.3\% on Persian. The cross-lingual model, trained on both languages but evaluated solely on the Persian test set, surpasses the LLM-based variant, by achieving a F1 of 74.8\%. Results indicate that a lightweight cross-lingual blend can outperform considerably the more resource-intensive augmentation pipelines, and it offers a practical pathway for the argument mining task to overcome data resource shortage on low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20857v1">Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Statefulness is essential for large language model (LLM) agents to perform long-term planning and problem-solving. This makes memory a critical component, yet its management and evolution remain largely underexplored. Existing evaluations mostly focus on static conversational settings, where memory is passively retrieved from dialogue to answer queries, overlooking the dynamic ability to accumulate and reuse experience across evolving task streams. In real-world environments such as interactive problem assistants or embodied agents, LLMs are required to handle continuous task streams, yet often fail to learn from accumulated interactions, losing valuable contextual insights, a limitation that calls for test-time evolution, where LLMs retrieve, integrate, and update memory continuously during deployment. To bridge this gap, we introduce Evo-Memory, a comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents. Evo-Memory structures datasets into sequential task streams, requiring LLMs to search, adapt, and evolve memory after each interaction. We unify and implement over ten representative memory modules and evaluate them across 10 diverse multi-turn goal-oriented and single-turn reasoning and QA datasets. To better benchmark experience reuse, we provide a baseline method, ExpRAG, for retrieving and utilizing prior experience, and further propose ReMem, an action-think-memory refine pipeline that tightly integrates reasoning, task actions, and memory updates to achieve continual improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08532v3">Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20726v1">Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 24 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Autonomous driving faces critical challenges in rare long-tail events and complex multi-agent interactions, which are scarce in real-world data yet essential for robust safety validation. This paper presents a high-fidelity scenario generation framework that integrates a conditional variational autoencoder (CVAE) with a large language model (LLM). The CVAE encodes historical trajectories and map information from large-scale naturalistic datasets to learn latent traffic structures, enabling the generation of physically consistent base scenarios. Building on this, the LLM acts as an adversarial reasoning engine, parsing unstructured scene descriptions into domain-specific loss functions and dynamically guiding scenario generation across varying risk levels. This knowledge-driven optimization balances realism with controllability, ensuring that generated scenarios remain both plausible and risk-sensitive. Extensive experiments in CARLA and SMARTS demonstrate that our framework substantially increases the coverage of high-risk and long-tail events, improves consistency between simulated and real-world traffic distributions, and exposes autonomous driving systems to interactions that are significantly more challenging than those produced by existing rule- or data-driven methods. These results establish a new pathway for safety validation, enabling principled stress-testing of autonomous systems under rare but consequential events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11255v2">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19023v1">OrdMoE: Preference Alignment via Hierarchical Expert Group Ranking in Multimodal Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Preference learning has recently emerged as a pivotal strategy for post-training alignment of Multimodal Large Language Models (MLLMs). However, existing approaches predominantly rely on external human-annotated preference data, which is costly and labor-intensive to collect. In this work, we propose OrdMoE, a novel preference alignment framework that bypasses the reliance on external human preferences entirely by leveraging intrinsic signals within Mixture-of-Experts (MoE) architectures. Specifically, we observe that the router's expert selection scores implicitly encode a quality-aware ranking of responses (i.e. higher-scoring experts consistently generate higher-quality outputs). Building on this insight, OrdMoE constructs an internal preference hierarchy by grouping experts into ranked tiers based on their per-token routing scores and activating each tier separately to produce a sequence of responses with increasing quality. This yields a zero-cost, self-supervised preference ordering over generated responses, which can be directly optimized using standard preference learning objectives. Extensive experiments across multiple multimodal benchmarks demnstrate that OrdMoE significantly enhances both alignment and overall performance of multimodal Mixture-of-Experts LLMs, achieving competitive results without requiring any human-annotated preference data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v4">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18985v1">LLM Chatbots in High School Programming: Exploring Behaviors and Interventions</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This study uses a Design-Based Research (DBR) cycle to refine the integration of Large Language Models (LLMs) in high school programming education. The initial problem was identified in an Intervention Group where, in an unguided setting, a higher proportion of executive, solution-seeking queries correlated strongly and negatively with exam performance. A contemporaneous Comparison Group demonstrated that without guidance, these unproductive help-seeking patterns do not self-correct, with engagement fluctuating and eventually declining. This insight prompted a mid-course pedagogical intervention in the first group, designed to teach instrumental help-seeking. The subsequent evaluation confirmed the intervention's success, revealing a decrease in executive queries, as well as a shift toward more productive learning workflows. However, this behavioral change did not translate into a statistically significant improvement in exam grades, suggesting that altering tool-use strategies alone may be insufficient to overcome foundational knowledge gaps. The DBR process thus yields a more nuanced principle: the educational value of an LLM depends on a pedagogy that scaffolds help-seeking, but this is only one part of the complex process of learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18977v1">FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 5 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Pruning is an effective method for compressing Large Language Models, but finding an optimal, non-uniform layer-wise sparsity allocation remains a key challenge. While heuristic methods are fast but yield suboptimal performance, more powerful search-based approaches like Reinforcement Learning are often hindered by prohibitive computational costs on large-scale models. To overcome this efficiency barrier, we propose FastForward Pruning. Its core is a decoupled, single-step RL framework that separates policy optimization from the complex budget satisfaction problem. Such a decoupling is crucial for efficiently searching the vast policy space of LLMs. This curriculum-based strategy begins with low-cost, simple tasks and gradually increases in complexity, significantly reducing the search's computational overhead. Evaluated on the LLaMA, Mistral, and OPT model families, our framework discovers pruning policies that achieve superior performance over strong heuristic baselines. Crucially, when compared to other search-based algorithms, our method achieves competitive or superior results at a fraction of the computational cost, demonstrating a clear advantage in search efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18966v1">LLM-CSEC: Empirical Evaluation of Security in C/C++ Code Generated by Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The security of code generated by large language models (LLMs) is a significant concern, as studies indicate that such code often contains vulnerabilities and lacks essential defensive programming constructs. This work focuses on examining and evaluating the security of LLM-generated code, particularly in the context of C/C++. We categorized known vulnerabilities using the Common Weakness Enumeration (CWE) and, to study their criticality, mapped them to CVEs. We used ten different LLMs for code generation and analyzed the outputs through static analysis. The amount of CWEs present in AI-generated code is concerning. Our findings highlight the need for developers to be cautious when using LLM-generated code. This study provides valuable insights to advance automated code generation and encourage further research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v21">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18924v1">LLM-Driven Kernel Evolution: Automating Driver Updates in Linux</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Linux kernel evolution breaks drivers through API/ABI changes, semantic shifts, and security-hardening updates. We introduce DRIVEBENCH, an executable corpus of kernel$\rightarrow$driver co-evolution cases, and AUTODRIVER, a closed-loop, LLM-driven system for automating driver maintenance. The system integrates prompt engineering, multi-agent collaboration, static analysis, and iterative validation to ensure that generated patches are not only syntactically correct but also functionally and semantically consistent with kernel conventions. The corpus spans v5.10-v6.10 with 235 validated cases drawn from 612 candidates. In evaluation across 55 cases, AUTODRIVER achieves 56.4% compilation success; QEMU-based boot verification indicates that compiled patches preserve driver initialization in most instances. By releasing DRIVEBENCH and tooling, we enable reproducible research and a practical route to continuous, safe co-evolution of drivers with the Linux kernel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18903v1">How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Due to the scarcity of high-quality data, large language models (LLMs) are often trained on mixtures of data with varying quality levels, even after sophisticated data curation. A natural approach to better leverage high-quality data is curriculum-based pretraining, where the model is trained on data sorted in ascending order of quality as determined by a quality metric. However, prior studies have reported limited improvements from such curriculum-based pretraining strategies. This work identifies a critical factor constraining these methods: the incompatibility between the ascending data quality order and the decaying learning rate (LR) schedule. We find that while curriculum-based training substantially outperforms random shuffling when using a constant LR, its advantage diminishes under standard LR decay schedules. Our experiments show this incompatibility can be mitigated by two simple strategies: (1) employing a more moderate LR decay schedule, where the final LR is only moderately smaller than the peak LR, and (2) replacing LR decay with model averaging, i.e., computing a weighted average of the final few checkpoints. By combining these strategies, we improve the average score on a suite of standard benchmarks by 1.64% over random shuffling, without additional data refinement. Validated on 1.5B-parameter models trained over 30B tokens with various data-quality metrics, our findings call for a re-evaluation of curriculum-based LLM pretraining and underscore the potential of co-designing data curricula with optimization methods.
    </details>
</div>
