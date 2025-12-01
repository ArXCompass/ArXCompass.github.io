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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18889v1">CoreEval: Automatically Building Contamination-Resilient Datasets with Real-World Knowledge toward Reliable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 ACL'25
    </div>
    <details class="paper-abstract">
      Data contamination poses a significant challenge to the fairness of LLM evaluations in natural language processing tasks by inadvertently exposing models to test data during training. Current studies attempt to mitigate this issue by modifying existing datasets or generating new ones from freshly collected information. However, these methods fall short of ensuring contamination-resilient evaluation, as they fail to fully eliminate pre-existing knowledge from models or preserve the semantic complexity of the original datasets. To address these limitations, we propose \textbf{CoreEval}, a \textbf{Co}ntamination-\textbf{re}silient \textbf{Eval}uation strategy for automatically updating data with real-world knowledge. This approach begins by extracting entity relationships from the original data and leveraging the GDELT database to retrieve relevant, up-to-date knowledge. The retrieved knowledge is then recontextualized and integrated with the original data, which is refined and restructured to ensure semantic coherence and enhanced task relevance. Ultimately, a robust data reflection mechanism is employed to iteratively verify and refine labels, ensuring consistency between the updated and original datasets. Extensive experiments on updated datasets validate the robustness of CoreEval, demonstrating its effectiveness in mitigating performance overestimation caused by data contamination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18868v1">KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      High quality kernels are critical for reducing training and inference costs of Large Language Models (LLMs), yet they traditionally require significant expertise in hardware architecture and software optimization. While recent advances in LLM-based code generation show promise for complex optimization, existing methods struggle with the vast optimization space due to insufficient hardware domain knowledge, failing to effectively balance exploration and exploitation. We present KernelBand, a novel framework that formulates kernel optimization as a hierarchical multi-armed bandit problem, enabling LLM agents to strategically navigate the optimization space by treating kernel selection and optimization strategy application as sequential decision-making processes. Our approach leverages hardware profiling information to identify promising optimization strategies and employs runtime behavior clustering to reduce exploration overhead across kernel candidates. Extensive experiments on TritonBench demonstrate that KernelBand significantly outperforms state-of-the-art methods, achieving superior performance with fewer tokens while exhibiting consistent improvement without saturation as computational resources increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18854v1">Time Travel: LLM-Assisted Semantic Behavior Localization with Git Bisect</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 submitted to Git Bisect SCALCOM 2025 Calgary (to be published)
    </div>
    <details class="paper-abstract">
      We present a novel framework that integrates Large Language Models (LLMs) into the Git bisect process for semantic fault localization. Traditional bisect assumes deterministic predicates and binary failure states assumptions often violated in modern software development due to flaky tests, nonmonotonic regressions, and semantic divergence from upstream repositories. Our system augments bisect traversal with structured chain of thought reasoning, enabling commit by commit analysis under noisy conditions. We evaluate multiple open source and proprietary LLMs for their suitability and fine tune DeepSeekCoderV2 using QLoRA on a curated dataset of semantically labeled diffs. We adopt a weak supervision workflow to reduce annotation overhead, incorporating human in the loop corrections and self consistency filtering. Experiments across multiple open source projects show a 6.4 point absolute gain in success rate from 74.2 to 80.6 percent, leading to significantly fewer failed traversals and by experiment up to 2x reduction in average bisect time. We conclude with discussions on temporal reasoning, prompt design, and finetuning strategies tailored for commit level behavior analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18850v1">Cognitive Alpha Mining via LLM-Driven Code-Based Evolution</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Discovering effective predictive signals, or ``alphas,'' from financial data with high dimensionality and extremely low signal-to-noise ratio remains a difficult open problem. Despite progress in deep learning, genetic programming, and, more recently, large language model (LLM)--based factor generation, existing approaches still explore only a narrow region of the vast alpha search space. Neural models tend to produce opaque and fragile patterns, while symbolic or formula-based methods often yield redundant or economically ungrounded expressions that generalize poorly. Although different in form, these paradigms share a key limitation: none can conduct broad, structured, and human-like exploration that balances logical consistency with creative leaps. To address this gap, we introduce the Cognitive Alpha Mining Framework (CogAlpha), which combines code-level alpha representation with LLM-driven reasoning and evolutionary search. Treating LLMs as adaptive cognitive agents, our framework iteratively refines, mutates, and recombines alpha candidates through multi-stage prompts and financial feedback. This synergistic design enables deeper thinking, richer structural diversity, and economically interpretable alpha discovery, while greatly expanding the effective search space. Experiments on A-share equities demonstrate that CogAlpha consistently discovers alphas with superior predictive accuracy, robustness, and generalization over existing methods. Our results highlight the promise of aligning evolutionary optimization with LLM-based reasoning for automated and explainable alpha discovery. All source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08426v8">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to IEEE TKDE2025
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18849v1">Pre-Filtering Code Suggestions using Developer Behavioral Telemetry to Optimize LLM-Assisted Programming</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into code editors to provide AI-powered code suggestions. Yet many of these suggestions are ignored, resulting in wasted computation, increased latency, and unnecessary interruptions. We introduce a lightweight pre-filtering model that predicts the likelihood of suggestion acceptance before invoking the LLM, using only real-time developer telemetry such as typing speed, file navigation, and editing activity. Deployed in a production-grade Visual Studio Code plugin over four months of naturalistic use, our approach nearly doubled acceptance rates (18.4% -> 34.2%) while suppressing 35% of low-value LLM calls. These findings demonstrate that behavioral signals alone can meaningfully improve both user experience and system efficiency in LLM-assisted programming, highlighting the value of timing-aware, privacy-preserving adaptation mechanisms. The filter operates solely on pre-invocation editor telemetry and never inspects code or prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18842v1">Optimizing LLM Code Suggestions: Feedback-Driven Timing with Lightweight State Bounds</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed code auto-completion by generating context-aware suggestions. Yet, deciding when to present these suggestions remains underexplored, often leading to interruptions or wasted inference calls. We propose an adaptive timing mechanism that dynamically adjusts the delay before offering a suggestion based on real-time developer feedback. Our suggested method combines a logistic transform of recent acceptance rates with a bounded delay range, anchored by a high-level binary prediction of the developer's cognitive state. In a two-month deployment with professional developers, our system improved suggestion acceptance from 4.9% with no delay to 15.4% with static delays, and to 18.6% with adaptive timing-while reducing blind rejections (rejections without being read) from 8.3% to 0.36%. Together, these improvements increase acceptance and substantially reduce wasted inference calls by 75%, making LLM-based code assistants more efficient and cost-effective in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.18436v5">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06447v2">SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Long-context inference for Large Language Models (LLMs) is heavily limited by high computational demands. While several existing methods optimize attention computation, they still process the full set of hidden states at each layer, limiting overall efficiency. In this work, we propose SlimInfer, an innovative framework that aims to accelerate inference by directly pruning less critical prompt tokens during the forward pass. Our key insight is an information diffusion phenomenon: As information from critical tokens propagates through layers, it becomes distributed across the entire sequence. This diffusion process suggests that LLMs can maintain their semantic integrity when excessive tokens, even including these critical ones, are pruned in hidden states. Motivated by this, SlimInfer introduces a dynamic fine-grained pruning mechanism that accurately removes redundant tokens of hidden state at intermediate layers. This layer-wise pruning naturally enables an asynchronous KV cache manager that prefetches required token blocks without complex predictors, reducing both memory usage and I/O costs. Extensive experiments show that SlimInfer can achieve up to $\mathbf{2.53\times}$ time-to-first-token (TTFT) speedup and $\mathbf{1.88\times}$ end-to-end latency reduction for LLaMA3.1-8B-Instruct on a single RTX 4090, without sacrificing performance on LongBench. Our code is available at https://github.com/Longxmas/SlimInfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13837v5">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 31 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18782v1">Summary-Mediated Repair: Can LLMs use code summarisation as a tool for program repair?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 6 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce code with subtle implementation-level bugs despite strong benchmark performance. These errors are hard for LLMs to spot and can have large behavioural effects; yet when asked to summarise code, LLMs can frequently surface high-level intent and sometimes overlook this low-level noise. Motivated by this, we propose summary-mediated repair, a prompt-only pipeline for program repair that leverages natural-language code summarisation as an explicit intermediate step, extending previous work that has already shown code summarisation to be a useful intermediary for downstream tasks. We evaluate our method across eight production-grade LLMs on two function level benchmarks (HumanEvalPack and MBPP), comparing several summary styles against a direct repair baseline. Error-aware diagnostic summaries consistently yield the largest gains - repairing up to 65% of unseen errors, on average of 5% more than the baseline - though overall improvements are modest and LLM-dependent. Our results position summaries as a cheap, human-interpretable diagnostic artefact that can be integrated into program-repair pipelines rather than a stand-alone fix-all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26213v2">OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 TL;DR: With the proposed OmniDocLayout-1M dataset and the LLM-based coarse-to-fine learning strategy, we enable diverse and complex document layout generation that achieves both strong condition consistency and adherence to fundamental aesthetic principles
    </div>
    <details class="paper-abstract">
      Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, layout generation, remains underexplored. Distinct from traditional graphic layout design and room layout planning, document layout generation typically involves a larger number of elements per page and exhibits greater structural diversity and complexity. Currently, a major obstacle lies in the scarcity of diverse document layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate OmniDocLayout-1M, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce OmniDocLayout-LLM, a 0.5B model with designed two-stage Coarse-to-Fine learning paradigm:1) learning universal layout principles from our dataset with coarse category definitions, and 2) transferring the knowledge to a specific domain with few fine-grained annotated samples. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M$^6$Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, dataset, and models will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18760v1">HERMES: Towards Efficient and Verifiable Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Informal mathematics has been central to modern large language model (LLM) reasoning, offering flexibility and enabling efficient construction of arguments. However, purely informal reasoning is prone to logical gaps and subtle errors that are difficult to detect and correct. In contrast, formal theorem proving provides rigorous, verifiable mathematical reasoning, where each inference step is checked by a trusted compiler in systems such as Lean, but lacks the exploratory freedom of informal problem solving. This mismatch leaves current LLM-based math agents without a principled way to combine the strengths of both paradigms. In this work, we introduce Hermes, the first tool-assisted agent that explicitly interleaves informal reasoning with formally verified proof steps in Lean. The framework performs intermediate formal checking to prevent reasoning drift and employs a memory module that maintains proof continuity across long, multi-step reasoning chains, enabling both exploration and verification within a single workflow. We evaluate Hermes on four challenging mathematical reasoning benchmarks using LLMs of varying parameter scales, from small models to state-of-the-art systems. Across all settings, Hermes reliably improves the reasoning accuracy of base models while substantially reducing token usage and computational cost compared to reward-based approaches. On difficult datasets such as AIME'25, Hermes achieves up to a 67% accuracy improvement while using 80% fewer total inference FLOPs. The implementation and codebase are publicly available at https://github.com/aziksh-ospanov/HERMES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15289v5">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 ACL Findings 2025. Welcome to employ SATA as a baseline
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18727v1">LogSyn: A Few-Shot LLM Framework for Structured Insight Extraction from Unstructured General Aviation Maintenance Logs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted in Proceedings of the 3rd INCOM 2026
    </div>
    <details class="paper-abstract">
      Aircraft maintenance logs hold valuable safety data but remain underused due to their unstructured text format. This paper introduces LogSyn, a framework that uses Large Language Models (LLMs) to convert these logs into structured, machine-readable data. Using few-shot in-context learning on 6,169 records, LogSyn performs Controlled Abstraction Generation (CAG) to summarize problem-resolution narratives and classify events within a detailed hierarchical ontology. The framework identifies key failure patterns, offering a scalable method for semantic structuring and actionable insight extraction from maintenance logs. This work provides a practical path to improve maintenance workflows and predictive analytics in aviation and related industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.16216v4">GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 9 pages, 25 figures
    </div>
    <details class="paper-abstract">
      The sparse Mixture-of-Experts (MoE) architecture of large language models (LLMs) confronts an inherent issue of load imbalance arising from the simplistic linear router strategy, which ultimately causes the instability and inefficient learning of LLMs. To address this challenge, we introduce a novel MoE graph-based framework $\textbf{GMoE}$, aimed at enhancing the collaboration among multiple experts. In GMoE, a graph router function is designed to capture the collaboration signals among experts. This enables all experts to dynamically allocate information derived from input data by sharing information with their neighboring experts. Moreover, we put forward two coordination strategies in GMoE: the $\textit{Poisson distribution-based distinction strategy}$ and the $\textit{Normal distribution-based balance strategy}$, to further release the capacity of each expert and increase the model stability in the fine-tuning of LLMs. Specifically, we leverage a parameter-efficient fine-tuning technique, i.e., Low-Rank Adaptation (LoRA), to implement the graph MoE architecture. Extensive experiments on four real-world benchmark datasets demonstrate the effectiveness of GMoE, showing the benefits of facilitating collaborations of multiple experts in LLM fine-tuning. The code of experimental implementation is available at https://github.com/BAI-LAB/GMoE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18305v3">Evolving Triple Knowledge-Augmented LLMs for Code Translation in Repository Context</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in function-level code translation without repository-level context. However, the performance of LLMs in repository-level context code translation remains suboptimal due to complex dependencies and context, hindering their adoption in industrial settings. In this work, we propose a novel LLM-based code translation technique K-Trans, which leverages triple knowledge augmentation to enhance LLM's translation quality under repository context in real-world software development. First, K-Trans constructs a evolving translation knowledge base by extracting relevant information from target-language codebases, the repository being translated, and prior translation results. Second, for each function to be translated, K-Trans retrieves relevant triple knowledge, including target-language code samples, dependency usage examples, and successful translation function pairs, serving as references to enhance LLM for translation. Third, K-Trans constructs a knowledge-augmented translation prompt using the retrieved triple knowledge and employs LLMs to generate the translated code while preserving repository context. It further leverages LLMs for self-debugging, enhancing translation correctness. Lastly, K-Trans continuously evolves the translation knowledge base. The experiments show that K-Trans substantially outperforms the baseline adapted from previous work by 19.4%/40.2% relative improvement in pass@1 and 0.138 in CodeBLEU. It is important to note that the results also demonstrate that each knowledge significantly contributes to K-Trans's effectiveness in handling repository-level context code translation, with dependency usage examples making the most notable contribution. Moreover, as the self-evolution process progresses, the knowledge base continuously enhances the LLM's performance across various aspects of the repository-level code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09598v6">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This paper introduces an infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models in commercial datacenters. The framework combines public API performance data with company-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost and provide a dynamically updated dashboard that visualizes model-level energy, water, and carbon metrics. Results show the most energy-intensive models exceed 29 Wh per long prompt, over 65 times the most efficient systems. Even a 0.42 Wh short query, when scaled to 700M queries/day, aggregates to annual electricity comparable to 35{,}000 U.S. homes, evaporative freshwater equal to the annual drinking needs of 1.2M people, and carbon emissions requiring a Chicago-sized forest to offset. These findings highlight a growing paradox: as AI becomes cheaper and faster, global adoption drives disproportionate resource consumption. Our methodology offers a standardized, empirically grounded basis for sustainability benchmarking and accountability in AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07061v3">Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets -- IEMOCAP and MELD -- show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12109v3">Personalized LLM Decoding via Contrasting Personal Preference</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02025v2">SAFE: Harnessing LLM for Scenario-Driven ADS Testing from Multimodal Crash Data</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 The paper has been accepted for publication in the proceedings of the IEEE/ACM 48th International Conference on Software Engineering to be held 12-18 April 2026 (ICSE2026)
    </div>
    <details class="paper-abstract">
      Ensuring the safety of Autonomous Driving Systems (ADS) requires realistic and reproducible test scenarios, yet extracting such scenarios from multimodal crash reports remains a major challenge. Large Language Models (LLMs) often hallucinate and lose map structure, resulting in unrealistic road layouts and vehicle behaviors. To address this, we introduce SAFE, a novel Scenario-based ADS testing Framework via multimodal Extraction, which leverages Retrieval-Augmented Generation (RAG), knowledge-grounded prompting, Chain-of-Thought (CoT) reasoning, and self-validation to improve scenario reconstruction from multimodal crash data. SAFE achieves 93.8% accuracy in extracting road network details, 80.0% for actor information, and 100% for environmental context. In human studies, SAFE outperforms LCTGen and AC3R in reconstructing consistent road networks and vehicle behaviors. Under identical ADS and simulator settings, SAFE detects 39 and 71 more safety violations than LCTGen and AC3R, respectively, and reproduces 12 more real-world crash cases than LCTGen. On 19 cases supported by AC3R, SAFE reproduces one additional crash case with statistically significant gains across five runs. It generates scenarios within 25 seconds and triggers violations after just 1 case (IDM) and 3 cases (PPO) in MetaDrive, as well as 1 case (Auto) in BeamNG. Code: https://github.com/Siwei-Luo-MQ/SAFE-ADS-Testing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14909v3">Mixture of Attention Spans: Optimizing LLM Inference Efficiency with Heterogeneous Sliding-Window Lengths</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Published at CoLM'25
    </div>
    <details class="paper-abstract">
      Sliding-window attention offers a hardware-efficient solution to the memory and throughput challenges of Large Language Models (LLMs) in long-context scenarios. Existing methods typically employ a single window length across all attention heads and input sizes. However, this uniform approach fails to capture the heterogeneous attention patterns inherent in LLMs, ignoring their distinct accuracy-latency trade-offs. To address this challenge, we propose *Mixture of Attention Spans* (MoA), which automatically tailors distinct sliding-window length configurations to different heads and layers. MoA constructs and navigates a search space of various window lengths and their scaling rules relative to input sizes. It profiles the model, evaluates potential configurations, and pinpoints the optimal length configurations for each head. MoA adapts to varying input sizes, revealing that some attention heads expand their focus to accommodate longer inputs, while other heads consistently concentrate on fixed-length local contexts. Experiments show that MoA increases the effective context length by 3.9x with the same average sliding-window length, boosting retrieval accuracy by 1.5-7.1x over the uniform-window baseline across Vicuna-{7B, 13B} and Llama3-{8B, 70B} models. Moreover, MoA narrows the performance gap with full attention, reducing the maximum relative performance drop from 9%-36% to within 5% across three long-context understanding benchmarks. MoA achieves a 1.2-1.4x GPU memory reduction, boosting decode throughput by 6.6-8.2x and 1.7-1.9x over FlashAttention2 and vLLM, with minimal performance impact. Our code is available at: https://github.com/thu-nics/MoA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19719v1">Can LLMs Faithfully Explain Themselves in Low-Resource Languages? A Case Study on Emotion Detection in Persian</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate self-explanations alongside their predictions, a practice that raises concerns about the faithfulness of these explanations, especially in low-resource languages. This study evaluates the faithfulness of LLM-generated explanations in the context of emotion classification in Persian, a low-resource language, by comparing the influential words identified by the model against those identified by human annotators. We assess faithfulness using confidence scores derived from token-level log-probabilities. Two prompting strategies, differing in the order of explanation and prediction (Predict-then-Explain and Explain-then-Predict), are tested for their impact on explanation faithfulness. Our results reveal that while LLMs achieve strong classification performance, their generated explanations often diverge from faithful reasoning, showing greater agreement with each other than with human judgments. These results highlight the limitations of current explanation methods and metrics, emphasizing the need for more robust approaches to ensure LLM reliability in multilingual and low-resource contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v5">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19654v1">Accuracy and Efficiency Trade-Offs in LLM-Based Malware Detection and Explanation: A Comparative Study of Parameter Tuning vs. Full Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted in IEEE Big Data 2025
    </div>
    <details class="paper-abstract">
      This study examines whether Low-Rank Adaptation (LoRA) fine-tuned Large Language Models (LLMs) can approximate the performance of fully fine-tuned models in generating human-interpretable decisions and explanations for malware classification. Achieving trustworthy malware detection, particularly when LLMs are involved, remains a significant challenge. We developed an evaluation framework using Bilingual Evaluation Understudy (BLEU), Recall-Oriented Understudy for Gisting Evaluation (ROUGE), and Semantic Similarity Metrics to benchmark explanation quality across five LoRA configurations and a fully fine-tuned baseline. Results indicate that full fine-tuning achieves the highest overall scores, with BLEU and ROUGE improvements of up to 10% over LoRA variants. However, mid-range LoRA models deliver competitive performance exceeding full fine-tuning on two metrics while reducing model size by approximately 81% and training time by over 80% on a LoRA model with 15.5% trainable parameters. These findings demonstrate that LoRA offers a practical balance of interpretability and resource efficiency, enabling deployment in resource-constrained environments without sacrificing explanation quality. By providing feature-driven natural language explanations for malware classifications, this approach enhances transparency, analyst confidence, and operational scalability in malware detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19648v1">Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19550v1">The Semiotic Channel Principle: Measuring the Capacity for Meaning in LLM Communication</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This paper proposes a novel semiotic framework for analyzing Large Language Models (LLMs), conceptualizing them as stochastic semiotic engines whose outputs demand active, asymmetric human interpretation. We formalize the trade-off between expressive richness (semiotic breadth) and interpretive stability (decipherability) using information-theoretic tools. Breadth is quantified as source entropy, and decipherability as the mutual information between messages and human interpretations. We introduce a generative complexity parameter (lambda) that governs this trade-off, as both breadth and decipherability are functions of lambda. The core trade-off is modeled as an emergent property of their distinct responses to $λ$. We define a semiotic channel, parameterized by audience and context, and posit a capacity constraint on meaning transmission, operationally defined as the maximum decipherability by optimizing lambda. This reframing shifts analysis from opaque model internals to observable textual artifacts, enabling empirical measurement of breadth and decipherability. We demonstrate the framework's utility across four key applications: (i) model profiling; (ii) optimizing prompt/context design; (iii) risk analysis based on ambiguity; and (iv) adaptive semiotic systems. We conclude that this capacity-based semiotic approach offers a rigorous, actionable toolkit for understanding, evaluating, and designing LLM-mediated communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19537v1">Cross-Domain Generalization of Multimodal LLMs for Global Photovoltaic Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 5 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The rapid expansion of distributed photovoltaic (PV) systems poses challenges for power grid management, as many installations remain undocumented. While satellite imagery provides global coverage, traditional computer vision (CV) models such as CNNs and U-Nets require extensive labeled data and fail to generalize across regions. This study investigates the cross-domain generalization of a multimodal large language model (LLM) for global PV assessment. By leveraging structured prompts and fine-tuning, the model integrates detection, localization, and quantification within a unified schema. Cross-regional evaluation using the $Δ$F1 metric demonstrates that the proposed model achieves the smallest performance degradation across unseen regions, outperforming conventional CV and transformer baselines. These results highlight the robustness of multimodal LLMs under domain shift and their potential for scalable, transferable, and interpretable global PV mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19536v1">AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19523v1">EAGER: Edge-Aligned LLM Defense for Robust, Efficient, and Accurate Cybersecurity Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly effective for cybersecurity question answering (QA) but are difficult to deploy on edge devices due to their size. Quantization reduces memory and compute requirements but often degrades accuracy and increases vulnerability to adversarial attacks. We present EAGER, an edge-aligned defense framework that integrates parameter-efficient quantization with domain-specific preference alignment to jointly optimize efficiency, robustness, and accuracy. Unlike prior methods that address these aspects separately, EAGER leverages Quantized Low-Rank Adaptation (QLoRA) for low-cost fine-tuning and Direct Preference Optimization (DPO) on a self-constructed cybersecurity preference dataset, eliminating the need for human labels. Experiments show that EAGER reduces adversarial attack success rates by up to 7.3x and improves QA accuracy by up to 55% over state-of-the-art defenses, while achieving the lowest response latency on a Jetson Orin, demonstrating its practical edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19517v1">Automating Deception: Scalable Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Multi-turn conversational attacks, which leverage psychological principles like Foot-in-the-Door (FITD), where a small initial request paves the way for a more significant one, to bypass safety alignments, pose a persistent threat to Large Language Models (LLMs). Progress in defending against these attacks is hindered by a reliance on manual, hard-to-scale dataset creation. This paper introduces a novel, automated pipeline for generating large-scale, psychologically-grounded multi-turn jailbreak datasets. We systematically operationalize FITD techniques into reproducible templates, creating a benchmark of 1,500 scenarios across illegal activities and offensive content. We evaluate seven models from three major LLM families under both multi-turn (with history) and single-turn (without history) conditions. Our results reveal stark differences in contextual robustness: models in the GPT family demonstrate a significant vulnerability to conversational history, with Attack Success Rates (ASR) increasing by as much as 32 percentage points. In contrast, Google's Gemini 2.5 Flash exhibits exceptional resilience, proving nearly immune to these attacks, while Anthropic's Claude 3 Haiku shows strong but imperfect resistance. These findings highlight a critical divergence in how current safety architectures handle conversational context and underscore the need for defenses that can resist narrative-based manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12491v3">Cost-Aware Contrastive Routing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16660v2">Cognitive Foundations for Reasoning and Their Manifestation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 40 pages, 4 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. To understand this gap, we synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning reasoning invariants, meta-cognitive controls, representations for organizing reasoning & knowledge, and transformation operations. We introduce a fine-grained evaluation framework and conduct the first large-scale empirical analysis of 192K traces from 18 models across text, vision, and audio, complemented by 54 human think-aloud traces, which we make publicly available. We find that models under-utilize cognitive elements correlated with success, narrowing to rigid sequential processing on ill-structured problems where diverse representations and meta-cognitive monitoring are critical. Human traces show more abstraction and conceptual processing, while models default to surface-level enumeration. Meta-analysis of 1.6K LLM reasoning papers reveals the research community concentrates on easily quantifiable elements (sequential organization: 55%, decomposition: 60%) but neglecting meta-cognitive controls (self-awareness: 16%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 66.7% on complex problems. By establishing a shared vocabulary between cognitive science and LLM research, our framework enables systematic diagnosis of reasoning failures and principled development of models that reason through robust cognitive mechanisms rather than spurious shortcuts, while providing tools to test theories of human cognition at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19423v1">Beyond Protein Language Models: An Agentic LLM Framework for Mechanistic Enzyme Design</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present Genie-CAT, a tool-augmented large-language-model (LLM) system designed to accelerate scientific hypothesis generation in protein design. Using metalloproteins (e.g., ferredoxins) as a case study, Genie-CAT integrates four capabilities -- literature-grounded reasoning through retrieval-augmented generation (RAG), structural parsing of Protein Data Bank files, electrostatic potential calculations, and machine-learning prediction of redox properties -- into a unified agentic workflow. By coupling natural-language reasoning with data-driven and physics-based computation, the system generates mechanistically interpretable, testable hypotheses linking sequence, structure, and function. In proof-of-concept demonstrations, Genie-CAT autonomously identifies residue-level modifications near [Fe--S] clusters that affect redox tuning, reproducing expert-derived hypotheses in a fraction of the time. The framework highlights how AI agents combining language models with domain-specific tools can bridge symbolic reasoning and numerical simulation, transforming LLMs from conversational assistants into partners for computational discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10659v2">Information Extraction From Fiscal Documents Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 6 pages. Presented at the AI for Financial Inclusion, Risk Modeling and Resilience in Emerging Markets workshop at ACM ICAIF 2025 Singapore
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in text comprehension, but their ability to process complex, hierarchical tabular data remains underexplored. We present a novel approach to extracting structured data from multi-page government fiscal documents using LLM-based techniques. Applied to annual fiscal documents from the State of Karnataka in India (200+ pages), our method achieves high accuracy through a multi-stage pipeline that leverages domain knowledge, sequential context, and algorithmic validation. A large challenge with traditional OCR methods is the inability to verify the accurate extraction of numbers. When applied to fiscal data, the inherent structure of fiscal tables, with totals at each level of the hierarchy, allows for robust internal validation of the extracted data. We use these hierarchical relationships to create multi-level validation checks. We demonstrate that LLMs can read tables and also process document-specific structural hierarchies, offering a scalable process for converting PDF-based fiscal disclosures into research-ready databases. Our implementation shows promise for broader applications across developing country contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03469v2">Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to AgenticSE Workshop at ASE 2025
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03463v2">ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to MAS-GAIN Workshop at ASE 2025
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19368v1">LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has been increasingly adopted in many real-world applications. While MARL enables decentralized deployment on resource-constrained edge devices, it suffers from severe non-stationarity due to the synchronous updates of agent policies. This non stationarity results in unstable training and poor policy con vergence, especially as the number of agents increases. In this paper, we propose RELED, a scalable MARL framework that integrates large language model (LLM)-driven expert demonstrations with autonomous agent exploration. RELED incorporates a Stationarity-Aware Expert Demonstration module, which leverages theoretical non-stationarity bounds to enhance the quality of LLM-generated expert trajectories, thus providing high reward and training-stable samples for each agent. Moreover, a Hybrid Expert-Agent Policy Optimization module adaptively balances each agent's learning from both expert-generated and agent-generated trajectories, accelerating policy convergence and improving generalization. Extensive experiments with real city networks based on OpenStreetMap demonstrate that RELED achieves superior performance compared to state-of-the-art MARL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03480v2">LLM Agents for Automated Dependency Upgrades</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to AISM Workshop at ASE 2005
    </div>
    <details class="paper-abstract">
      As a codebase expands over time, its library dependencies can become outdated and require updates to maintain innovation and security. However, updating a library can introduce breaking changes in the code, necessitating significant developer time for maintenance. To address this, we introduce a framework of LLM agents to be used in combination with migration documentation to automatically recommend and apply code updates and ensure compatibility with new versions. Our solution can automatically localize updated library usages in live Java codebases and implement recommended fixes in a user-friendly manner. The system architecture consists of multiple key components: a Summary Agent, Control Agent, and Code Agent. To validate our approach, we apply the framework on an industrial use case by which we create three synthetic code repositories with major Upgrade changes and benchmark our approach against state-of-the-art methods. Results show that our approach not only performs upgrades using fewer tokens across all cases but also achieves a precision of 71.4%, highlighting its efficiency and effectiveness compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19355v1">Leveraging LLMs for reward function design in reinforcement learning control tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19333v1">Learning to Reason: Training LLMs with GPT-OSS or DeepSeek R1 Reasoning Traces</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Test-time scaling, which leverages additional computation during inference to improve model accuracy, has enabled a new class of Large Language Models (LLMs) that are able to reason through complex problems by understanding the goal, turning this goal into a plan, working through intermediate steps, and checking their own work before answering . Frontier large language models with reasoning capabilities, such as DeepSeek-R1 and OpenAI's gpt-oss, follow the same procedure when solving complex problems by generating intermediate reasoning traces before giving the final answer. Today, these models are being increasingly used to generate reasoning traces that serve as high-quality supervised data for post-training of small and medium-sized language models to teach reasoning capabilities without requiring expensive human curation. In this work, we compare the performance of medium-sized LLMs on Math problems after post-training on two kinds of reasoning traces. We compare the impact of reasoning traces generated by DeepSeek-R1 and gpt-oss LLMs in terms of accuracy and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22006v2">Enhancing Domain-Specific Encoder Models with LLM-Generated Data: How to Leverage Ontologies, and How to Do Without Them</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Published in the Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      We investigate the use of LLM-generated data for continual pretraining of encoder models in specialized domains with limited training data, using the scientific domain of invasion biology as a case study. To this end, we leverage domain-specific ontologies by enriching them with LLM-generated data and pretraining the encoder model as an ontology-informed embedding model for concept definitions. To evaluate the effectiveness of this method, we compile a benchmark specifically designed for assessing model performance in invasion biology. After demonstrating substantial improvements over standard LLM pretraining, we investigate the feasibility of applying the proposed approach to domains without comprehensive ontologies by substituting ontological concepts with concepts automatically extracted from a small corpus of scientific abstracts and establishing relationships between concepts through distributional statistics. Our results demonstrate that this automated approach achieves comparable performance using only a small set of scientific abstracts, resulting in a fully automated pipeline for enhancing domain-specific understanding of small encoder models that is especially suited for application in low-resource settings and achieves performance comparable to masked language modeling pretraining on much larger datasets.
    </details>
</div>
