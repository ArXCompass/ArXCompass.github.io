# llm - 2025_06

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
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06391v1">From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05346v1">Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project Page: https://hsiung.cc/llm-similarity-risk/
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05334v1">Search Arena: Analyzing Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Preprint. Code: https://github.com/lmarena/search-arena. Dataset: https://huggingface.co/datasets/lmarena-ai/search-arena-24k
    </div>
    <details class="paper-abstract">
      Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research in this direction. Our dataset and code are available at: https://github.com/lmarena/search-arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05316v1">Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24102v2">Is LLM the Silver Bullet to Low-Resource Languages Machine Translation?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Low-Resource Languages (LRLs) present significant challenges in natural language processing due to their limited linguistic resources and underrepresentation in standard datasets. While recent advances in Large Language Models (LLMs) and Neural Machine Translation have substantially improved translation capabilities for high-resource languages, performance disparities persist for LRLs, particularly impacting privacy-sensitive and resource-constrained scenarios. This paper systematically evaluates current LLMs in 200 languages using the FLORES-200 benchmark and demonstrates their limitations in LRL translation capability. We also explore alternative data sources, including news articles and bilingual dictionaries, and demonstrate how knowledge distillation from large pre-trained teacher models can significantly improve the performance of small LLMs on LRL translation tasks. For example, this approach increases EN->LB with the LLM-as-a-Judge score on the validation set from 0.36 to 0.89 for Llama-3.2-3B. Furthermore, we examine different fine-tuning configurations, providing practical insights on optimal data scale, training efficiency, and the preservation of generalization capabilities of models under study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05309v1">Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are inherently asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns; therefore, the decision of when to speak forms a crucial part of the participant's decision making. In this work, we develop an adaptive asynchronous LLM-agent which, in addition to determining what to say, also decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, including both human participants, as well as our asynchronous agent. Overall, our agent performs on par with human players, both in game performance, as well as in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We release all our data and code to support and encourage further research for more realistic asynchronous communication between LLM agents. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05260v1">LeanPO: Lean Preference Optimization for Likelihood Alignment in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Code: https://github.com/Wang-Xiaodong1899/LeanPO
    </div>
    <details class="paper-abstract">
      Most Video Large Language Models (Video-LLMs) adopt preference alignment techniques, e.g., DPO~\citep{rafailov2024dpo}, to optimize the reward margin between a winning response ($y_w$) and a losing response ($y_l$). However, the likelihood displacement observed in DPO indicates that both $\log \pi_\theta (y_w\mid x)$ and $\log \pi_\theta (y_l\mid x) $ often decrease during training, inadvertently boosting the probabilities of non-target responses. In this paper, we systematically revisit this phenomenon from LLMs to Video-LLMs, showing that it intensifies when dealing with the redundant complexity of video content. To alleviate the impact of this phenomenon, we propose \emph{Lean Preference Optimization} (LeanPO), a reference-free approach that reformulates the implicit reward as the average likelihood of the response with respect to the policy model. A key component of LeanPO is the reward-trustworthiness correlated self-generated preference data pipeline, which carefully infuses relevant prior knowledge into the model while continuously refining the preference data via self-reflection. This allows the policy model to obtain high-quality paired data and accurately estimate the newly defined reward, thus mitigating the unintended drop. In addition, we introduce a dynamic label smoothing strategy that mitigates the impact of noise in responses from diverse video content, preventing the model from overfitting to spurious details. Extensive experiments demonstrate that LeanPO significantly enhances the performance of state-of-the-art Video-LLMs, consistently boosting baselines of varying capacities with minimal additional training overhead. Moreover, LeanPO offers a simple yet effective solution for aligning Video-LLM preferences with human trustworthiness, paving the way toward the reliable and efficient Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04075v2">Rethinking LLM Advancement: Compute-Dependent and Independent Paths to Progress</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Regulatory efforts to govern large language model (LLM) development have predominantly focused on restricting access to high-performance computational resources. This study evaluates the efficacy of such measures by examining whether LLM capabilities can advance through algorithmic innovation in compute-constrained environments. We propose a novel framework distinguishing compute-dependent innovations--which yield disproportionate benefits at high compute--from compute-independent innovations, which improve efficiency across compute scales. The impact is quantified using Compute-Equivalent Gain (CEG). Experimental validation with nanoGPT models confirms that compute-independent advancements yield significant performance gains (e.g., with combined CEG up to $3.5\times$) across the tested scales. In contrast, compute-dependent advancements were detrimental to performance at smaller experimental scales, but showed improved CEG (on par with the baseline) as model size increased, a trend consistent with their definition of yielding primary benefits at higher compute. Crucially, these findings indicate that restrictions on computational hardware, while potentially slowing LLM progress, are insufficient to prevent all capability gains driven by algorithmic advancements. We argue that effective AI oversight must therefore incorporate mechanisms for understanding, anticipating, and potentially guiding algorithmic research, moving beyond a singular focus on hardware. The proposed framework also serves as an analytical tool for forecasting AI progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05242v1">SECNEURON: Reliable and Flexible Abuse Control in Local LLMs via Hybrid Neuron Encryption</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with diverse capabilities are increasingly being deployed in local environments, presenting significant security and controllability challenges. These locally deployed LLMs operate outside the direct control of developers, rendering them more susceptible to abuse. Existing mitigation techniques mainly designed for cloud-based LLM services are frequently circumvented or ineffective in deployer-controlled environments. We propose SECNEURON, the first framework that seamlessly embeds classic access control within the intrinsic capabilities of LLMs, achieving reliable, cost-effective, flexible, and certified abuse control for local deployed LLMs. SECNEURON employs neuron-level encryption and selective decryption to dynamically control the task-specific capabilities of LLMs, limiting unauthorized task abuse without compromising others. We first design a task-specific neuron extraction mechanism to decouple logically related neurons and construct a layered policy tree for handling coupled neurons. We then introduce a flexible and efficient hybrid encryption framework for millions of neurons in LLMs. Finally, we developed a distribution-based decrypted neuron detection mechanism on ciphertext to ensure the effectiveness of partially decrypted LLMs. We proved that SECNEURON satisfies IND-CPA Security and Collusion Resistance Security under the Task Controllability Principle. Experiments on various task settings show that SECNEURON limits unauthorized task accuracy to below 25% while keeping authorized accuracy loss with 2%. Using an unauthorized Code task example, the accuracy of abuse-related malicious code generation was reduced from 59% to 15%. SECNEURON also mitigates unauthorized data leakage, reducing PII extraction rates to below 5% and membership inference to random guesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08503v2">Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      In this work, we identify the "2D-Cheating" problem in 3D LLM evaluation, where these tasks might be easily solved by VLMs with rendered images of point clouds, exposing ineffective evaluation of 3D LLMs' unique 3D capabilities. We test VLM performance across multiple 3D LLM benchmarks and, using this as a reference, propose principles for better assessing genuine 3D understanding. We also advocate explicitly separating 3D abilities from 1D or 2D aspects when evaluating 3D LLMs. Code and data are available at https://github.com/LLM-class-group/Revisiting-3D-LLM-Benchmarks .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05213v1">LLM-First Search: Self-Guided Exploration of the Solution Space</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 9 main pages, 2 figures, 2 tables, 36 appendix pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code is publicly available at \href{https://github.com/NathanHerr/LLM-First-Search}{LLM-First-Search}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02524v4">CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming a wide range of domains, yet verifying their outputs remains a significant challenge, especially for complex open-ended tasks such as consolidation, summarization, and knowledge extraction. To address this, we introduce CheckEmbed (CE): a simple, scalable, and accurate verification method. CE reduces each LLM answer to a single embedding vector using powerful modern embedding LLM models like SFR-Embedding-Mistral. Prior methods such as BERTScore and SelfCheckGPT relied on weaker encoders like BERT, forcing them to operate at token or sentence granularity. In contrast, CE performs fast, semantically rich comparisons directly at the whole-answer level, overcoming key limitations in both accuracy and scalability. We conduct a comprehensive design and time complexity analysis across 13 verification baselines, including classical text scorers (e.g., BLEU), stability-based methods (e.g., SelfCheckGPT), and generative evaluators (e.g., LLM-as-a-Judge), which highlights the effectiveness, efficiency, versatility, and simplicity of CE. Empirical results show that CE reliably detects hallucinations in both closed and open-ended tasks. We further present evidence that CE generalizes beyond text to other modalities such as vision, establishing it as a practical and versatile verification framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v2">A Multi-agent LLM-based JUnit Test Generation with Strong Oracles</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is laborious, especially for strong typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to generate tests that achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in large language models (LLMs) have enabled oracle generation from natural language descriptions. However, existing LLM-based methods often require LLM fine-tuning or rely on external tools such as EvoSuite for test prefix generation. In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM framework for automated JUnit test generation. CANDOR orchestrates multiple specialized LLM agents to generate JUnit tests, including both high-quality test prefixes and accurate oracles. To mitigate the notorious hallucinations in LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generate accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments on the HumanEvalJava and LeetCodeJava datasets show that CANDOR can generate accurate oracles and is slightly better than EvoSuite in generating tests with high line coverage and clearly superior in terms of mutation score. Moreover, CANDOR significantly outperforms the state-of-the-art, prompt-based test generator LLM-Empirical, achieving improvements of 15.8 to 25.1 percentage points in oracle correctness on both correct and faulty source code. Ablation studies confirm the critical contributions of key agents in improving test prefix quality and oracle accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v2">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 38 pages
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03939v1">Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      Graph Retrieval Augmented Generation (GraphRAG) effectively enhances external knowledge integration capabilities by explicitly modeling knowledge relationships, thereby improving the factual accuracy and generation quality of Large Language Models (LLMs) in specialized domains. However, existing methods suffer from two inherent limitations: 1) Inefficient Information Aggregation: They rely on a single agent and fixed iterative patterns, making it difficult to adaptively capture multi-level textual, structural, and degree information within graph data. 2) Rigid Reasoning Mechanism: They employ preset reasoning schemes, which cannot dynamically adjust reasoning depth nor achieve precise semantic correction. To overcome these limitations, we propose Graph Counselor, an GraphRAG method based on multi-agent collaboration. This method uses the Adaptive Graph Information Extraction Module (AGIEM), where Planning, Thought, and Execution Agents work together to precisely model complex graph structures and dynamically adjust information extraction strategies, addressing the challenges of multi-level dependency modeling and adaptive reasoning depth. Additionally, the Self-Reflection with Multiple Perspectives (SR) module improves the accuracy and semantic consistency of reasoning results through self-reflection and backward reasoning mechanisms. Experiments demonstrate that Graph Counselor outperforms existing methods in multiple graph reasoning tasks, exhibiting higher reasoning accuracy and generalization ability. Our code is available at https://github.com/gjq100/Graph-Counselor.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03930v1">VisCoder: Fine-Tuning LLMs for Executable Python Visualization Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with visualization tasks like plotting diagrams, charts, where success depends on both code correctness and visual semantics. Existing instruction-tuning datasets lack execution-grounded supervision and offer limited support for iterative code correction, resulting in fragile and unreliable plot generation. We present VisCode-200K, a large-scale instruction tuning dataset for Python-based visualization and self-correction. It contains over 200K examples from two sources: (1) validated plotting code from open-source repositories, paired with natural language instructions and rendered plots; and (2) 45K multi-turn correction dialogues from Code-Feedback, enabling models to revise faulty code using runtime feedback. We fine-tune Qwen2.5-Coder-Instruct on VisCode-200K to create VisCoder, and evaluate it on PandasPlotBench. VisCoder significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4o-mini. We further adopt a self-debug evaluation protocol to assess iterative repair, demonstrating the benefits of feedback-driven learning for executable, visually accurate code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v2">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the *NL-FL Problem Alignment* method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the *Mixed Problem Input* technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive experiments demonstrate that the **HybridReasoning** framework achieves **89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03923v1">More or Less Wrong: A Benchmark for Directional Bias in LLM Comparative Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to be sensitive to input phrasing, but the mechanisms by which semantic cues shape reasoning remain poorly understood. We investigate this phenomenon in the context of comparative math problems with objective ground truth, revealing a consistent and directional framing bias: logically equivalent questions containing the words ``more'', ``less'', or ``equal'' systematically steer predictions in the direction of the framing term. To study this effect, we introduce MathComp, a controlled benchmark of 300 comparison scenarios, each evaluated under 14 prompt variants across three LLM families. We find that model errors frequently reflect linguistic steering, systematic shifts toward the comparative term present in the prompt. Chain-of-thought prompting reduces these biases, but its effectiveness varies: free-form reasoning is more robust, while structured formats may preserve or reintroduce directional drift. Finally, we show that including demographic identity terms (e.g., ``a woman'', ``a Black person'') in input scenarios amplifies directional drift, despite identical underlying quantities, highlighting the interplay between semantic framing and social referents. These findings expose critical blind spots in standard evaluation and motivate framing-aware benchmarks for diagnosing reasoning robustness and fairness in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03921v1">Boosting Open-Source LLMs for Program Repair via Reasoning Transfer and LLM-Guided Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Several closed-source LLMs have consistently outperformed open-source alternatives in program repair tasks, primarily due to their superior reasoning capabilities and extensive pre-training. This paper introduces Repairity, a novel three-stage methodology that significantly narrows this performance gap through reasoning extraction and reinforcement learning. Our approach: (1) systematically filters high-quality reasoning traces from closed-source models using correctness verification, (2) transfers this reasoning knowledge to open-source models via supervised fine-tuning, and (3) develops reinforcement learning with LLM-based feedback to further optimize performance. Empirical evaluation across multiple program repair benchmarks demonstrates that Repairity improves the performance of Qwen2.5-Coder-32B-Instruct, a base open source LLM, by 8.68\% on average, reducing the capability gap with Claude-Sonnet3.7, a state-of-the-art closed-source model, from 10.05% to 1.35%. Ablation studies confirm that both reasoning extraction and LLM-guided reinforcement learning contribute significantly to these improvements. Our methodology generalizes effectively to additional code-related tasks, enabling organizations to leverage high-quality program repair capabilities while maintaining the customizability, transparency, and deployment flexibility inherent to open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04670v2">LLM Code Customization with Visual Results: A Benchmark on TikZ</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03887v1">Pre$^3$: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Published as a conference paper at ACL 2025
    </div>
    <details class="paper-abstract">
      Extensive LLM applications demand efficient structured generations, particularly for LR(1) grammars, to produce outputs in specified formats (e.g., JSON). Existing methods primarily parse LR(1) grammars into a pushdown automaton (PDA), leading to runtime execution overhead for context-dependent token processing, especially inefficient under large inference batches. To address these issues, we propose Pre$^3$ that exploits deterministic pushdown automata (DPDA) to optimize the constrained LLM decoding efficiency. First, by precomputing prefix-conditioned edges during the preprocessing, Pre$^3$ enables ahead-of-time edge analysis and thus makes parallel transition processing possible. Second, by leveraging the prefix-conditioned edges, Pre$^3$ introduces a novel approach that transforms LR(1) transition graphs into DPDA, eliminating the need for runtime path exploration and achieving edge transitions with minimal overhead. Pre$^3$ can be seamlessly integrated into standard LLM inference frameworks, reducing time per output token (TPOT) by up to 40% and increasing throughput by up to 36% in our experiments. Our code is available at https://github.com/ModelTC/lightllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03857v1">Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to ACL 2025 (Main conference)
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have demonstrated significant potential for data annotation, markedly reducing the labor costs associated with downstream applications. However, existing methods mostly adopt an aggressive strategy by prompting LLM to determine a single gold label for each unlabeled sample. Due to the inherent uncertainty within LLMs, they often produce incorrect labels for difficult samples, severely compromising the data quality for downstream applications. Motivated by ambiguity aversion in human behaviors, we propose a novel candidate annotation paradigm wherein large language models are encouraged to output all possible labels when incurring uncertainty. To ensure unique labels are provided for downstream tasks, we develop a teacher-student framework CanDist that distills candidate annotations with a Small Language Model (SLM). We further provide a rigorous justification demonstrating that distilling candidate annotations from the teacher LLM offers superior theoretical guarantees compared to directly using single annotations. Extensive experiments across six text classification tasks validate the effectiveness of our proposed method. The source code is available at https://github.com/MingxuanXia/CanDist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21082v3">LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04920v4">Enabling LLM Knowledge Analysis via Extensive Materialization</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 14 pages, 4 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have majorly advanced NLP and AI, and next to their ability to perform a wide range of procedural tasks, a major success factor is their internalized factual knowledge. Since Petroni et al. (2019), analyzing this knowledge has gained attention. However, most approaches investigate one question at a time via modest-sized pre-defined samples, introducing an ``availability bias'' (Tversky&Kahnemann, 1973) that prevents the analysis of knowledge (or beliefs) of LLMs beyond the experimenter's predisposition. To address this challenge, we propose a novel methodology to comprehensively materialize an LLM's factual knowledge through recursive querying and result consolidation. Our approach is a milestone for LLM research, for the first time providing constructive insights into the scope and structure of LLM knowledge (or beliefs). As a prototype, we build GPTKB, a knowledge base (KB) comprising 101 million relational triples for over 2.9 million entities from GPT-4o-mini. We use GPTKB to exemplarily analyze GPT-4o-mini's factual knowledge in terms of scale, accuracy, bias, cutoff and consistency, at the same time. GPTKB is accessible at https://gptkb.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03801v1">From Theory to Practice: Real-World Use Cases on Trustworthy LLM-Driven Process Modeling, Prediction and Automation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to the Next Gen Data and Process Management: Large Language Models and Beyond workshop at SIGMOD 2025
    </div>
    <details class="paper-abstract">
      Traditional Business Process Management (BPM) struggles with rigidity, opacity, and scalability in dynamic environments while emerging Large Language Models (LLMs) present transformative opportunities alongside risks. This paper explores four real-world use cases that demonstrate how LLMs, augmented with trustworthy process intelligence, redefine process modeling, prediction, and automation. Grounded in early-stage research projects with industrial partners, the work spans manufacturing, modeling, life-science, and design processes, addressing domain-specific challenges through human-AI collaboration. In manufacturing, an LLM-driven framework integrates uncertainty-aware explainable Machine Learning (ML) with interactive dialogues, transforming opaque predictions into auditable workflows. For process modeling, conversational interfaces democratize BPMN design. Pharmacovigilance agents automate drug safety monitoring via knowledge-graph-augmented LLMs. Finally, sustainable textile design employs multi-agent systems to navigate regulatory and environmental trade-offs. We intend to examine tensions between transparency and efficiency, generalization and specialization, and human agency versus automation. By mapping these trade-offs, we advocate for context-sensitive integration prioritizing domain needs, stakeholder values, and iterative human-in-the-loop workflows over universal solutions. This work provides actionable insights for researchers and practitioners aiming to operationalize LLMs in critical BPM environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v1">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14175v2">Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Query expansion methods powered by large language models (LLMs) have demonstrated effectiveness in zero-shot retrieval tasks. These methods assume that LLMs can generate hypothetical documents that, when incorporated into a query vector, enhance the retrieval of real evidence. However, we challenge this assumption by investigating whether knowledge leakage in benchmarks contributes to the observed performance gains. Using fact verification as a testbed, we analyze whether the generated documents contain information entailed by ground-truth evidence and assess their impact on performance. Our findings indicate that, on average, performance improvements consistently occurred for claims whose generated documents included sentences entailed by gold evidence. This suggests that knowledge leakage may be present in fact-verification benchmarks, potentially inflating the perceived performance of LLM-based query expansion methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03700v1">AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ICML 2025. Code: https://github.com/weizhepei/AdaDecode
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for long-content generation (e.g., long Chain-of-Thought reasoning) where decoding efficiency becomes a critical bottleneck: Autoregressive decoding is inherently limited by its sequential token generation process, where each token must be generated before the next can be processed. This sequential dependency restricts the ability to fully leverage modern hardware's parallel processing capabilities. Existing methods like speculative decoding and layer skipping offer potential speedups but have notable drawbacks: speculative decoding relies on an auxiliary "drafter" model, which can be challenging to acquire and increases memory overhead, while layer skipping may introduce discrepancies in the outputs due to the missing key-value cache at skipped layers. In this work, we propose AdaDecode, which accelerates LLM decoding without requiring auxiliary models or changes to the original model parameters, while ensuring output consistency. AdaDecode leverages the insight that many tokens can accurately be generated at intermediate layers, as further layers often do not significantly alter predictions once the model reaches a certain confidence. By adaptively generating tokens at intermediate layers when confidence is high, AdaDecode enables the next token's computation to begin immediately. The remaining layer computations for early-predicted tokens are deferred and executed in parallel with subsequent tokens when needed, maximizing hardware utilization and reducing decoding latency. A final verification step ensures that early predictions match the results of standard autoregressive decoding, preserving output parity. Experiments across diverse generation tasks shows that AdaDecode consistently achieves superior decoding throughput with up to 1.73x speedup, while guaranteeing output parity with standard autoregressive decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03691v1">A Two-Staged LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Continuous Integration and Continuous Deployment (CI/CD) pipelines are pivotal to modern software engineering, yet diagnosing and resolving their failures remains a complex and labor-intensive challenge. In this paper, we present LogSage, the first end-to-end LLM-powered framework that performs root cause analysis and solution generation from failed CI/CD pipeline logs. During the root cause analysis stage, LogSage employs a specialized log preprocessing pipeline tailored for LLMs, which extracts critical error logs and eliminates noise to enhance the precision of LLM-driven root cause analysis. In the solution generation stage, LogSage leverages RAG to integrate historical resolution strategies and utilizes tool-calling to deliver actionable, automated fixes. We evaluated the root cause analysis stage using a newly curated open-source dataset, achieving 98\% in precision and 12\% improvement over naively designed LLM-based log analysis baselines, while attaining near-perfect recall. The end-to-end system was rigorously validated in a large-scale industrial CI/CD environment of production quality, processing more than 3,000 executions daily and accumulating more than 1.07 million executions in its first year of deployment, with end-to-end precision exceeding 88\%. These two forms of evaluation confirm that LogSage providing a scalable and practical solution to manage CI/CD pipeline failures in real-world DevOps workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10515v2">Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 18 pages, 7 figures, 3 tables, code: https://github.com/mainlp/discourse_probes, camera-ready revision for ACL 2025
    </div>
    <details class="paper-abstract">
      Discourse understanding is essential for many NLP tasks, yet most existing work remains constrained by framework-dependent discourse representations. This work investigates whether large language models (LLMs) capture discourse knowledge that generalizes across languages and frameworks. We address this question along two dimensions: (1) developing a unified discourse relation label set to facilitate cross-lingual and cross-framework discourse analysis, and (2) probing LLMs to assess whether they encode generalizable discourse abstractions. Using multilingual discourse relation classification as a testbed, we examine a comprehensive set of 23 LLMs of varying sizes and multilingual capabilities. Our results show that LLMs, especially those with multilingual training corpora, can generalize discourse information across languages and frameworks. Further layer-wise analyses reveal that language generalization at the discourse level is most salient in the intermediate layers. Lastly, our error analysis provides an account of challenging relation classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00486v3">It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: https://huggingface.co/spaces/shifeng3711/gg_prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16502v3">RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Preprint. Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19449v3">VecTrans: Enhancing Compiler Auto-Vectorization through LLM-Assisted Code Transformations</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Auto-vectorization is a fundamental optimization for modern compilers to exploit SIMD parallelism. However, state-of-the-art approaches still struggle to handle intricate code patterns, often requiring manual hints or domain-specific expertise. Large language models (LLMs), with their ability to capture intricate patterns, provide a promising solution, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. In this paper, we present VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compilers auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. experimental results show that among all TSVC functions unvectorizable by GCC, ICC, Clang, and BiSheng Compiler, VecTrans achieves an geomean speedup of 1.77x and successfully vectorizes 24 of 51 test cases. This marks a significant advancement over state-of-the-art approaches while maintaining a cost efficiency of $0.012 per function optimization for LLM API usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03656v1">Client-Side Zero-Shot LLM Inference for Comprehensive In-Browser URL Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 46 pages , 5 figures
    </div>
    <details class="paper-abstract">
      Malicious websites and phishing URLs pose an ever-increasing cybersecurity risk, with phishing attacks growing by 40% in a single year. Traditional detection approaches rely on machine learning classifiers or rule-based scanners operating in the cloud, but these face significant challenges in generalization, privacy, and evasion by sophisticated threats. In this paper, we propose a novel client-side framework for comprehensive URL analysis that leverages zero-shot inference by a local large language model (LLM) running entirely in-browser. Our system uses a compact LLM (e.g., 3B/8B parameters) via WebLLM to perform reasoning over rich context collected from the target webpage, including static code analysis (JavaScript abstract syntax trees, structure, and code patterns), dynamic sandbox execution results (DOM changes, API calls, and network requests),and visible content. We detail the architecture and methodology of the system, which combines a real browser sandbox (using iframes) resistant to common anti-analysis techniques, with an LLM-based analyzer that assesses potential vulnerabilities and malicious behaviors without any task-specific training (zero-shot). The LLM aggregates evidence from multiple sources (code, execution trace, page content) to classify the URL as benign or malicious and to provide an explanation of the threats or security issues identified. We evaluate our approach on a diverse set of benign and malicious URLs, demonstrating that even a compact client-side model can achieve high detection accuracy and insightful explanations comparable to cloud-based solutions, while operating privately on end-user devices. The results show that client-side LLM inference is a feasible and effective solution to web threat analysis, eliminating the need to send potentially sensitive data to cloud services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03655v1">Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation necessitates scalable, automated fact-checking solutions. Yet, current benchmarks often overlook multilingual and topical diversity. This paper introduces a novel, dynamically extensible data set that includes 61,514 claims in multiple languages and topics, extending existing datasets up to 2024. Through a comprehensive evaluation of five prominent Large Language Models (LLMs), including GPT-4o, GPT-3.5 Turbo, LLaMA 3.1, and Mixtral 8x7B, we identify significant performance gaps between different languages and topics. While overall GPT-4o achieves the highest accuracy, it declines to classify 43% of claims. Across all models, factual-sounding claims are misclassified more often than opinions, revealing a key vulnerability. These findings underscore the need for caution and highlight challenges in deploying LLM-based fact-checking systems at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04173v2">Quantifying Prediction Consistency Under Fine-Tuning Multiplicity in Tabular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 International Conference on Machine Learning (ICML), 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning LLMs on tabular classification tasks can lead to the phenomenon of fine-tuning multiplicity where equally well-performing models make conflicting predictions on the same input. Fine-tuning multiplicity can arise due to variations in the training process, e.g., seed, weight initialization, minor changes to training data, etc., raising concerns about the reliability of Tabular LLMs in high-stakes applications such as finance, hiring, education, healthcare. Our work formalizes this unique challenge of fine-tuning multiplicity in Tabular LLMs and proposes a novel measure to quantify the consistency of individual predictions without expensive model retraining. Our measure quantifies a prediction's consistency by analyzing (sampling) the model's local behavior around that input in the embedding space. Interestingly, we show that sampling in the local neighborhood can be leveraged to provide probabilistic guarantees on prediction consistency under a broad class of fine-tuned models, i.e., inputs with sufficiently high local stability (as defined by our measure) also remain consistent across several fine-tuned models with high probability. We perform experiments on multiple real-world datasets to show that our local stability measure preemptively captures consistency under actual multiplicity across several fine-tuned models, outperforming competing measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09887v4">OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited their problem-solving abilities in mathematical reasoning. Solving realistic optimization (OPT) problems in application scenarios requires advanced and applied mathematics ability. However, current OPT benchmarks that merely solve linear programming are far from complex realistic situations. In this work, we propose OptiBench, a benchmark for End-to-end optimization problem-solving with human-readable inputs and outputs. OptiBench contains rich optimization problems, including linear and nonlinear programming with or without tabular data, which can comprehensively evaluate LLMs' solving ability. In our benchmark, LLMs are required to call a code solver to provide precise numerical answers. Furthermore, to alleviate the data scarcity for optimization problems, and to bridge the gap between open-source LLMs on a small scale (e.g., Llama-3-8b) and closed-source LLMs (e.g., GPT-4), we further propose a data synthesis method namely ReSocratic. Unlike general data synthesis methods that proceed from questions to answers, \ReSocratic first incrementally synthesizes formatted optimization demonstration with mathematical formulations step by step and then back-translates the generated demonstrations into questions. Based on this, we synthesize the ReSocratic-29k dataset. We further conduct supervised fine-tuning with ReSocratic-29k on multiple open-source models. Experimental results show that ReSocratic-29k significantly improves the performance of open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09429v3">From Intention To Implementation: Automating Biomedical Research via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 The paper involves material for which we have not yet obtained proper copyright permissions
    </div>
    <details class="paper-abstract">
      Conventional biomedical research is increasingly labor-intensive due to the exponential growth of scientific literature and datasets. Artificial intelligence (AI), particularly Large Language Models (LLMs), has the potential to revolutionize this process by automating various steps. Still, significant challenges remain, including the need for multidisciplinary expertise, logicality of experimental design, and performance measurements. This paper introduces BioResearcher, the first end-to-end automated system designed to streamline the entire biomedical research process involving dry lab experiments. BioResearcher employs a modular multi-agent architecture, integrating specialized agents for search, literature processing, experimental design, and programming. By decomposing complex tasks into logically related sub-tasks and utilizing a hierarchical learning approach, BioResearcher effectively addresses the challenges of multidisciplinary requirements and logical complexity. Furthermore, BioResearcher incorporates an LLM-based reviewer for in-process quality control and introduces novel evaluation metrics to assess the quality and automation of experimental protocols. BioResearcher successfully achieves an average execution success rate of 63.07% across eight previously unmet research objectives. The generated protocols averagely outperform typical agent systems by 22.0% on five quality metrics. The system demonstrates significant potential to reduce researchers' workloads and accelerate biomedical discoveries, paving the way for future innovations in automated research systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03610v1">Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are reshaping the game industry, particularly with more intelligent and human-preferable game characters. However, existing game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets for aligning pre-trained LLMs into gaming agents. To fill these gaps, we present \textbf{\benchname{}}, a foundational benchmark designed to train and evaluate LLM agents across diverse real-world video games. Unlike existing benchmarks, Orak includes 12 popular video games spanning all major genres, enabling comprehensive studies of LLM capabilities and agentic modules essential for intricate game scenarios. To support consistent evaluation of LLMs, we introduce a plug-and-play interface based on Model Context Protocol (MCP) that enables LLMs to seamlessly connect with games and manipulate agentic modules. Additionally, we propose a fine-tuning dataset, consisting of LLM gameplay trajectories across diverse game genres. Orak offers a comprehensive evaluation framework, encompassing general game score leaderboards, LLM battle arenas, and in-depth analyses of visual input state, agentic strategies, and fine-tuning effects, establishing a foundation towards building generic gaming agents. Code is available at https://github.com/krafton-ai/Orak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v2">MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14884v2">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02442v2">Should LLM Safety Be More Than Refusing Harmful Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02965v2">PC-MoE: Memory-Efficient and Privacy-Preserving Collaborative Training for Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) has been gaining popularity due to its successful adaptation to large language models (LLMs). In this work, we introduce Privacy-preserving Collaborative Mixture-of-Experts (PC-MoE), which leverages the sparsity of the MoE architecture for memory-efficient decentralized collaborative LLM training, enabling multiple parties with limited GPU-memory and data resources to collectively train more capable LLMs than they could achieve individually. At the same time, this approach protects training data privacy of each participant by keeping training data, as well as parts of the forward pass signal and gradients locally within each party. By design, PC-MoE synergistically combines the strengths of distributed computation with strong confidentiality assurances. Unlike most privacy-preserving schemes, which pay for confidentiality with lower task accuracy, our framework breaks that trade-off: across seven popular LLM benchmarks, it almost matches (and sometimes exceeds) the performance and convergence rate of a fully centralized model, enjoys near 70% peak GPU RAM reduction, while being fully robust against reconstruction attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03585v1">Improving LLM-Based Fault Localization with External Memory and Project Context</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 12 Pages, 7 figures
    </div>
    <details class="paper-abstract">
      Fault localization, the process of identifying the software components responsible for failures, is essential but often time-consuming. Recent advances in Large Language Models (LLMs) have enabled fault localization without extensive defect datasets or model fine-tuning. However, existing LLM-based methods rely only on general LLM capabilities and lack integration of project-specific knowledge, resulting in limited effectiveness, especially for complex software. We introduce MemFL, a novel approach that enhances LLM-based fault localization by integrating project-specific knowledge via external memory. This memory includes static summaries of the project and dynamic, iterative debugging insights gathered from previous attempts. By leveraging external memory, MemFL simplifies debugging into three streamlined steps, significantly improving efficiency and accuracy. Iterative refinement through dynamic memory further enhances reasoning quality over time. Evaluated on the Defects4J benchmark, MemFL using GPT-4o-mini localized 12.7% more bugs than current LLM-based methods, achieving this improvement with just 21% of the execution time (17.4 seconds per bug) and 33% of the API cost (0.0033 dollars per bug). On complex projects, MemFL's advantage increased to 27.6%. Additionally, MemFL with GPT-4.1-mini outperformed existing methods by 24.4%, requiring only 24.7 seconds and 0.0094 dollars per bug. MemFL thus demonstrates significant improvements by effectively incorporating project-specific knowledge into LLM-based fault localization, delivering high accuracy with reduced time and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20875v2">Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 27 pages, 6 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our code and datasets are publicly available at https://github.com/jiyounglee-0523/TransEnV and https://huggingface.co/collections/jiyounglee0523/transenv-681eadb3c0c8cf363b363fb1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03543v1">CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) agents lack authentic human psychological processes necessary for genuine digital twins and social AI applications. To address this limitation, we present a computational implementation of Global Workspace Theory (GNWT) that integrates human cognitive architecture principles into LLM agents, creating specialized sub-agents for emotion, memory, social norms, planning, and goal-tracking coordinated through a global workspace mechanism. However, authentic digital twins require accurate personality initialization. We therefore develop a novel adventure-based personality test that evaluates true personality through behavioral choices within interactive scenarios, bypassing self-presentation bias found in traditional assessments. Building on these innovations, our CogniPair platform enables digital twins to engage in realistic simulated dating interactions and job interviews before real encounters, providing bidirectional cultural fit assessment for both romantic compatibility and workplace matching. Validation using 551 GNWT-Agents and Columbia University Speed Dating dataset demonstrates 72% correlation with human attraction patterns, 77.8% match prediction accuracy, and 74% agreement in human validation studies. This work advances psychological authenticity in LLM agents and establishes a foundation for intelligent dating platforms and HR technology solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00095v3">ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at https://clinbench-hpb.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v3">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20099v2">Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03504v1">Beyond C/C++: Probabilistic and LLM Methods for Next-Generation Software Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      This proposal discusses the growing challenges in reverse engineering modern software binaries, particularly those compiled from newer system programming languages such as Rust, Go, and Mojo. Traditional reverse engineering techniques, developed with a focus on C and C++, fall short when applied to these newer languages due to their reliance on outdated heuristics and failure to fully utilize the rich semantic information embedded in binary programs. These challenges are exacerbated by the limitations of current data-driven methods, which are susceptible to generating inaccurate results, commonly referred to as hallucinations. To overcome these limitations, we propose a novel approach that integrates probabilistic binary analysis with fine-tuned large language models (LLMs). Our method systematically models the uncertainties inherent in reverse engineering, enabling more accurate reasoning about incomplete or ambiguous information. By incorporating LLMs, we extend the analysis beyond traditional heuristics, allowing for more creative and context-aware inferences, particularly for binaries from diverse programming languages. This hybrid approach not only enhances the robustness and accuracy of reverse engineering efforts but also offers a scalable solution adaptable to the rapidly evolving landscape of software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03483v1">APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often require domain-specific fine-tuning to address targeted tasks, which risks degrading their general capabilities. Maintaining a balance between domain-specific enhancements and general model utility is a key challenge. This paper proposes a novel approach named APT (Weakness Case Acquisition and Iterative Preference Training) to enhance domain-specific performance with self-generated dis-preferred weakness data (bad cases and similar cases). APT uniquely focuses on training the model using only those samples where errors occur, alongside a small, similar set of samples retrieved for this purpose. This targeted training minimizes interference with the model's existing knowledge base, effectively retaining generic capabilities. Experimental results on the LLama-2 and Mistral-V0.3 models across various benchmarks demonstrate that APT ensures no reduction in generic capacity and achieves superior performance on downstream tasks compared to various existing methods. This validates our method as an effective strategy for enhancing domain-specific capabilities without sacrificing the model's broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18173v2">UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 The experimental results of the paper require further validation
    </div>
    <details class="paper-abstract">
      Managing long texts is challenging for large language models (LLMs) due to limited context window sizes. This study introduces UIO-LLMs, an unbiased incremental optimization approach for memory-enhanced transformers under long-context settings. We initially conceptualize the process as a streamlined encoder-decoder framework where the weights-shared encoder and decoder respectively encapsulate a context segment into memories and leverage these memories to predict outputs of the subsequent segment. Subsequently, by treating our memory-enhanced transformers as fully-connected recurrent neural networks (RNNs), we refine the training process using the Truncated Backpropagation Through Time (TBPTT) algorithm, which incorporates innovative incremental optimization techniques. These techniques not only diminish time complexity but also address the bias in gradient computation through an unbiased optimization process. UIO-LLMs successfully handle long context, such as extending the context window of Llama2-7b-chat from 4K to 100K tokens with minimal 2% additional parameters, while keeping the inference cost nearly linear as context length increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20730v2">What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01821v2">On the Power of Context-Enhanced Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 77 pages, 18 figures; ICML 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We formalize a new concept for LLMs, context-enhanced learning. It involves standard gradient-based learning on text except that the context is enhanced with additional data on which no auto-regressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works. Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04482v1">Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Findings of the Association for Computational Linguistics: ACL 2025
    </div>
    <details class="paper-abstract">
      The NLP research community has made publicly available numerous instruments for measuring representational harms caused by large language model (LLM)-based systems. These instruments have taken the form of datasets, metrics, tools, and more. In this paper, we examine the extent to which such instruments meet the needs of practitioners tasked with evaluating LLM-based systems. Via semi-structured interviews with 12 such practitioners, we find that practitioners are often unable to use publicly available instruments for measuring representational harms. We identify two types of challenges. In some cases, instruments are not useful because they do not meaningfully measure what practitioners seek to measure or are otherwise misaligned with practitioner needs. In other cases, instruments - even useful instruments - are not used by practitioners due to practical and institutional barriers impeding their uptake. Drawing on measurement theory and pragmatic measurement, we provide recommendations for addressing these challenges to better meet practitioner needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04481v1">CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) show promise in solving complex mathematical tasks, existing evaluation paradigms rely solely on a coarse measure of overall answer accuracy, which are insufficient for assessing their authentic capabilities. In this paper, we propose \textbf{CogMath}, which comprehensively assesses LLMs' mathematical abilities through the lens of human cognition. Specifically, inspired by psychological theories, CogMath formalizes human reasoning process into 3 stages: \emph{problem comprehension}, \emph{problem solving}, and \emph{solution summarization}. Within these stages, we investigate perspectives such as numerical calculation, knowledge, and counterfactuals, and design a total of 9 fine-grained evaluation dimensions. In each dimension, we develop an ``\emph{Inquiry}-\emph{Judge}-\emph{Reference}'' multi-agent system to generate inquiries that assess LLMs' mastery from this dimension. An LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. By applying CogMath on three benchmarks, we reveal that the mathematical capabilities of 7 mainstream LLMs are overestimated by 30\%-40\%. Moreover, we locate their strengths and weaknesses across specific stages/dimensions, offering in-depth insights to further enhance their reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04478v1">Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v1">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 19 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v1">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Submitted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches this graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04418v1">Characterizing Multi-Hunk Patches: Divergence, Proximity, and LLM Repair Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Multi-hunk bugs, where fixes span disjoint regions of code, are common in practice, yet remain underrepresented in automated repair. Existing techniques and benchmarks pre-dominantly target single-hunk scenarios, overlooking the added complexity of coordinating semantically related changes across the codebase. In this work, we characterize HUNK4J, a dataset of multi-hunk patches derived from 372 real-world defects. We propose hunk divergence, a metric that quantifies the variation among edits in a patch by capturing lexical, structural, and file-level differences, while incorporating the number of hunks involved. We further define spatial proximity, a classification that models how hunks are spatially distributed across the program hierarchy. Our empirical study spanning six LLMs reveals that model success rates decline with increased divergence and spatial dispersion. Notably, when using the LLM alone, no model succeeds in the most dispersed Fragment class. These findings highlight a critical gap in LLM capabilities and motivate divergence-aware repair strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v3">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04405v1">MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v3">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by IJCAI 2025 (Survey Track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs, which can be categorized into the following two aspects: (1) Logical question answering: LLMs often fail to generate the correct answer within a complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency: LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art question-answering LLM Macaw, answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose a detailed taxonomy. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistencies, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extending to modal logic to account for uncertainty and developing efficient algorithms that simultaneously satisfy multiple logical consistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v3">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      In this work, we investigate the causal reasoning abilities of large language models (LLMs) through the representative problem of inferring causal relationships from narratives. We find that even state-of-the-art language models rely on unreliable shortcuts, both in terms of the narrative presentation and their parametric knowledge. For example, LLMs tend to determine causal relationships based on the topological ordering of events (i.e., earlier events cause later ones), resulting in lower performance whenever events are not narrated in their exact causal order. Similarly, we demonstrate that LLMs struggle with long-term causal reasoning and often fail when the narratives are long and contain many events. Additionally, we show LLMs appear to rely heavily on their parametric knowledge at the expense of reasoning over the provided narrative. This degrades their abilities whenever the narrative opposes parametric knowledge. We extensively validate these failure modes through carefully controlled synthetic experiments, as well as evaluations on real-world narratives. Finally, we observe that explicitly generating a causal graph generally improves performance while naive chain-of-thought is ineffective. Collectively, our results distill precise failure modes of current state-of-the-art models and can pave the way for future techniques to enhance causal reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v3">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Added benchmarks, baselines, author, appendix
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., "Sure, here it is") or otherwise unsafe content -- even for harmful prompts. In experiments on the SafeBench and MM-SafetyBench benchmarks, our method achieves higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 81% on certain models). We further demonstrate cross-model universality by training on several multimodal LLMs simultaneously. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04344v1">GEM: Empowering LLM for both Embedding Generation and Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large decoder-only language models (LLMs) have achieved remarkable success in generation and reasoning tasks, where they generate text responses given instructions. However, many applications, e.g., retrieval augmented generation (RAG), still rely on separate embedding models to generate text embeddings, which can complicate the system and introduce discrepancies in understanding of the query between the embedding model and LLMs. To address this limitation, we propose a simple self-supervised approach, Generative Embedding large language Model (GEM), that enables any large decoder-only LLM to generate high-quality text embeddings while maintaining its original text generation and reasoning capabilities. Our method inserts new special token(s) into a text body, and generates summarization embedding of the text by manipulating the attention mask. This method could be easily integrated into post-training or fine tuning stages of any existing LLMs. We demonstrate the effectiveness of our approach by applying it to two popular LLM families, ranging from 1B to 8B parameters, and evaluating the transformed models on both text embedding benchmarks (MTEB) and NLP benchmarks (MMLU). The results show that our proposed method significantly improves the original LLMs on MTEB while having a minimal impact on MMLU. Our strong results indicate that our approach can empower LLMs with state-of-the-art text embedding capabilities while maintaining their original NLP performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04290v1">Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 20 pages, 6 figures, preprint under review at Information Processing & Management
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM), which have developed in recent years, enable credit risk assessment through the analysis of financial texts such as analyst reports and corporate disclosures. This paper presents the first systematic review and taxonomy focusing on LLMbased approaches in credit risk estimation. We determined the basic model architectures by selecting 60 relevant papers published between 2020-2025 with the PRISMA research strategy. And we examined the data used for scenarios such as credit default prediction and risk analysis. Since the main focus of the paper is interpretability, we classify concepts such as explainability mechanisms, chain of thought prompts and natural language justifications for LLM-based credit models. The taxonomy organizes the literature under four main headings: model architectures, data types, explainability mechanisms and application areas. Based on this analysis, we highlight the main future trends and research gaps for LLM-based credit scoring systems. This paper aims to be a reference paper for artificial intelligence and financial researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04282v1">DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Symbolic regression is a fundamental tool for discovering interpretable mathematical expressions from data, with broad applications across scientific and engineering domains. Recently, large language models (LLMs) have demonstrated strong performance in this task, leveraging embedded scientific priors and reasoning capabilities to surpass traditional methods. However, existing LLM-based approaches, such as LLM-SR, often over-rely on internal priors, lacking explicit data understanding and systematic reflection during equation generation. To address these limitations, we propose DrSR (Dual Reasoning Symbolic Regression), a framework that combines data-driven insight with reflective learning to enhance both robustness and discovery capability. Specifically, DrSR guides LLMs to analyze structural relationships (e.g., monotonicity, nonlinearity, and correlation) within the data to generate structured descriptions. Simultaneously, it monitors equation performance and establishes a feedback loop to refine subsequent generations. By integrating data understanding and generation reflection in a closed loop, DrSR enables more efficient exploration of the symbolic expression space. Experiments across interdisciplinary datasets in physics, chemistry, biology, and materials science demonstrate that DrSR substantially improves the valid equation rate and consistently outperforms both classical and recent LLM-based methods in terms of accuracy, generalization, and search efficiency. These results underscore its potential for scientific equation discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05414v1">SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Project website with demo videos: https://zijuncui02.github.io/SAVVY/
    </div>
    <details class="paper-abstract">
      3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05413v1">SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 6 pages, 3 figures, 5 tables. Submitted to the IEEE SMC 2025 conference
    </div>
    <details class="paper-abstract">
      We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05410v1">Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 14 pages,7 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v1">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04185v1">R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have notably progressed in multi-step and long-chain reasoning. However, extending their reasoning capabilities to encompass deep interactions with search remains a non-trivial challenge, as models often fail to identify optimal reasoning-search interaction trajectories, resulting in suboptimal responses. We propose R-Search, a novel reinforcement learning framework for Reasoning-Search integration, designed to enable LLMs to autonomously execute multi-step reasoning with deep search interaction, and learn optimal reasoning search interaction trajectories via multi-reward signals, improving response quality in complex logic- and knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when to retrieve or reason, while globally integrating key evidence to enhance deep knowledge interaction between reasoning and search. During RL training, R-Search provides multi-stage, multi-type rewards to jointly optimize the reasoning-search trajectory. Experiments on seven datasets show that R-Search outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1% (out-of-domain). The code and data are available at https://github.com/QingFei1/R-Search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13865v2">A Survey on (M)LLM-Based GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04172v1">Does Prompt Design Impact Quality of Data Imputation by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Generating realistic synthetic tabular data presents a critical challenge in machine learning. It adds another layer of complexity when this data contain class imbalance problems. This paper presents a novel token-aware data imputation method that leverages the in-context learning capabilities of large language models. This is achieved through the combination of a structured group-wise CSV-style prompting technique and the elimination of irrelevant contextual information in the input prompt. We test this approach with two class-imbalanced binary classification datasets and evaluate the effectiveness of imputation using classification-based evaluation metrics. The experimental results demonstrate that our approach significantly reduces the input prompt size while maintaining or improving imputation quality compared to our baseline prompt, especially for datasets that are of relatively smaller in size. The contributions of this presented work is two-fold -- 1) it sheds light on the importance of prompt design when leveraging LLMs for synthetic data generation and 2) it addresses a critical gap in LLM-based data imputation for class-imbalanced datasets with missing data by providing a practical solution within computational constraints. We hope that our work will foster further research and discussions about leveraging the incredible potential of LLMs and prompt engineering techniques for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v2">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04167v1">Neural and Cognitive Impacts of AI: The Influence of Task Subjectivity on Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      AI-based interactive assistants are advancing human-augmenting technology, yet their effects on users' mental and physiological states remain under-explored. We address this gap by analyzing how Copilot for Microsoft Word, a LLM-based assistant, impacts users. Using tasks ranging from objective (SAT reading comprehension) to subjective (personal reflection), and with measurements including fNIRS, Empatica E4, NASA-TLX, and questionnaires, we measure Copilot's effects on users. We also evaluate users' performance with and without Copilot across tasks. In objective tasks, participants reported a reduction of workload and an increase in enjoyment, which was paired with objective performance increases. Participants reported reduced workload and increased enjoyment with no change in performance in a creative poetry writing task. However, no benefits due to Copilot use were reported in a highly subjective self-reflection task. Although no physiological changes were recorded due to Copilot use, task-dependent differences in prefrontal cortex activation offer complementary insights into the cognitive processes associated with successful and unsuccessful human-AI collaboration. These findings suggest that AI assistants' effectiveness varies with task type-particularly showing decreased usefulness in tasks that engage episodic memory-and presents a brain-network based hypothesis of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04142v1">Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) depends on trustworthy evaluation. However, most current evaluations rely on public benchmarks, which are prone to data contamination issues that significantly compromise fairness. Previous researches have focused on constructing dynamic benchmarks to address contamination. However, continuously building new benchmarks is costly and cyclical. In this work, we aim to tackle contamination by analyzing the mechanisms of contaminated models themselves. Through our experiments, we discover that the overestimation of contaminated models is likely due to parameters acquiring shortcut solutions in training. We further propose a novel method for identifying shortcut neurons through comparative and causal analysis. Building on this, we introduce an evaluation method called shortcut neuron patching to suppress shortcut neurons. Experiments validate the effectiveness of our approach in mitigating contamination. Additionally, our evaluation results exhibit a strong linear correlation with MixEval, a recently released trustworthy benchmark, achieving a Spearman coefficient ($\rho$) exceeding 0.95. This high correlation indicates that our method closely reveals true capabilities of the models and is trustworthy. We conduct further experiments to demonstrate the generalizability of our method across various benchmarks and hyperparameter settings. Code: https://github.com/GaryStack/Trustworthy-Evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10981v2">Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL 2025 Main, 33 pages, 51 figures
    </div>
    <details class="paper-abstract">
      Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a probabilistic method to efficiently predict scaling performance and identify the best prompting strategy under large sampling times, eliminating the need for resource-intensive inference processes in practical applications. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. Code is available at https://github.com/MraDonkey/rethinking_prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v1">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built on large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligent autonomy, collaboration and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based agentic multi-agent systems (AMAS). We begin by examining the conceptual foundations of agentic AI, its architectural differences from traditional AI agents, and the emerging system designs that enable scalable, tool-using autonomy. The TRiSM in the agentic AI framework is then detailed through four pillars governance, explainability, ModelOps, and privacy/security each contextualized for agentic LLMs. We identify unique threat vectors and introduce a comprehensive risk taxonomy for the agentic AI applications, supported by case studies illustrating real-world vulnerabilities. Furthermore, the paper also surveys trust-building mechanisms, transparency and oversight techniques, and state-of-the-art explainability strategies in distributed LLM agent systems. Additionally, metrics for evaluating trust, interpretability, and human-centered performance are reviewed alongside open benchmarking challenges. Security and privacy are addressed through encryption, adversarial defense, and compliance with evolving AI regulations. The paper concludes with a roadmap for responsible agentic AI, proposing research directions to align emerging multi-agent systems with robust TRiSM principles for safe, accountable, and transparent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19176v2">Assistant-Guided Mitigation of Teacher Preference Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge employs large language models (LLMs), such as GPT-4, to evaluate the quality of LLM-generated responses, gaining popularity for its cost-effectiveness and strong alignment with human evaluations. However, training proxy judge models using evaluation data generated by powerful teacher models introduces a critical yet previously overlooked issue: teacher preference bias, where the proxy judge model learns a biased preference for responses from the teacher model. To tackle this problem, we propose a novel setting that incorporates an additional assistant model, which is not biased toward the teacher model's responses, to complement the training data. Building on this setup, we introduce AGDe-Judge, a three-stage framework designed to debias from both the labels and feedbacks in the training data. Extensive experiments demonstrate that AGDe-Judge effectively reduces teacher preference bias while maintaining strong performance across six evaluation benchmarks. Code is available at https://github.com/Liuz233/AGDe-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17089v2">KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Inference for Large Language Models (LLMs) is computationally demanding. To reduce the cost of auto-regressive decoding, Key-Value (KV) cache is used to store intermediate activations, which significantly lowers the computational overhead for token generation. However, the memory required for the KV cache grows rapidly, often exceeding the capacity of GPU memory. A cost-effective alternative is to offload KV cache to CPU memory, which alleviates GPU memory pressure, but shifts the bottleneck to the limited bandwidth of the PCIe connection between the CPU and GPU. Existing methods attempt to address these issues by overlapping GPU computation with I/O or employing CPU-GPU heterogeneous execution, but they are hindered by excessive data movement and dependence on CPU capabilities. Fully overlapping PCIe communication latency gets challenging as the size of the KV cache grows and/or the GPU compute capabilities increase. In this paper, we introduce KVPR, an efficient I/O-aware LLM inference method where the CPU first transfers a partial set of activations, from which the GPU can start recomputing the KV cache values. While the GPU recomputes the partial KV cache, the remaining portion of the KV cache is transferred concurrently from the CPU. This approach overlaps GPU recomputation with KV cache transfer to minimize idle GPU time and maximize inference performance. KVPR is fully automated by integrating a profiler module that utilizes input characteristics and system hardware information, a scheduler module to optimize the distribution of computation and communication workloads, and a runtime module to efficiently execute the derived execution plan. Experimental results show that KVPR achieves up to 35.8% lower latency and 46.2% higher throughput during decoding compared to state-of-the-art approaches. The code is available at https://github.com/chaoyij/KVPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04078v1">LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04070v1">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study, hence, focuses on producing precise, in-situ, step-by-step navigation instructions that are practically usable by VI users. Concretely, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to generate rewards guiding the Vision-Language Model (VLM) post-training. This enhances instruction usability while reducing costly real-world data needs. To facilitate training and testing, we introduce NIG4VI, a 27k-sample open-sourced benchmark. It provides diverse navigation scenarios with accurate spatial coordinates, supporting detailed, open-ended in-situ instruction generation. Experiments on NIG4VI show the effectiveness of LaF-GRPO by quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU +14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o's 0.323) and yields more intuitive, safer instructions. Code and benchmark are available at \href{https://github.com/YiyiyiZhao/NIG4VI}{https://github.com/YiyiyiZhao/NIG4VI}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17169v4">REAL: Response Embedding-based Alignment for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) to human preferences is a crucial step in building helpful and safe AI tools, which usually involve training on supervised datasets. Popular algorithms such as Direct Preference Optimization (DPO) rely on pairs of AI-generated responses ranked according to human annotation. The response pair annotation process might bring human bias. Building a correct preference dataset is the costly part of the alignment pipeline. To improve annotation efficiency and quality in the LLMs alignment, we propose REAL: Response Embedding-based Alignment for LLMs, a strategy for constructing a high-quality training dataset that focuses on acquiring the less ambiguous preference pairs for labeling out of a set of response candidates. Our selection process is based on the similarity of embedding responses independently of prompts, which guarantees the selection process in an off-policy setting, avoiding adaptively measuring the similarity during the training. Experimental results on real-world dataset SHP2 and synthetic HH-RLHF benchmarks indicate that choosing dissimilar response pairs enhances the direct alignment of LLMs while reducing inherited labeling errors. The model aligned with dissimilar response pairs obtained a better margin and win rate on the dialogue task. Our findings suggest that focusing on distinct pairs can reduce the label error and improve LLM alignment efficiency, saving up to $65\%$ of annotators' work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04063v1">Crowd-SFT: Crowdsourcing for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) to align model responses with human preferences. While RLHF employs a reinforcement learning approach with a separate reward model, SFT uses human-curated datasets for supervised learning. Both approaches traditionally depend on small, vetted groups of annotators, making them costly, prone to bias, and limited in scalability. We propose an open, crowd-sourced fine-tuning framework that addresses these limitations by enabling broader feedback collection for SFT without extensive annotator training. Our framework promotes incentive fairness via a point-based reward system correlated with Shapley values and guides model convergence through iterative model updates. Our multi-model selection framework demonstrates up to a 55% reduction in target distance over single-model selection, enabling subsequent experiments that validate our point-based reward mechanism's close alignment with Shapley values (a well-established method for attributing individual contributions) thereby supporting fair and scalable participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04051v1">High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04050v1">Explainability-Based Token Replacement on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04044v1">Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to SemEval-2025, an ACL 2025 workshop
    </div>
    <details class="paper-abstract">
      This paper describes LIBU (LoRA enhanced influence-based unlearning), an algorithm to solve the task of unlearning - removing specific knowledge from a large language model without retraining from scratch and compromising its overall utility (SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models). The algorithm combines classical \textit{influence functions} to remove the influence of the data from the model and \textit{second-order optimization} to stabilize the overall utility. Our experiments show that this lightweight approach is well applicable for unlearning LLMs in different kinds of task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04043v1">Think Like a Person Before Responding: A Multi-Faceted Evaluation of Persona-Guided LLMs for Countering Hate</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted at ACL WOAH 2025
    </div>
    <details class="paper-abstract">
      Automated counter-narratives (CN) offer a promising strategy for mitigating online hate speech, yet concerns about their affective tone, accessibility, and ethical risks remain. We propose a framework for evaluating Large Language Model (LLM)-generated CNs across four dimensions: persona framing, verbosity and readability, affective tone, and ethical robustness. Using GPT-4o-Mini, Cohere's CommandR-7B, and Meta's LLaMA 3.1-70B, we assess three prompting strategies on the MT-Conan and HatEval datasets. Our findings reveal that LLM-generated CNs are often verbose and adapted for people with college-level literacy, limiting their accessibility. While emotionally guided prompts yield more empathetic and readable responses, there remain concerns surrounding safety and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02524v3">CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming a wide range of domains, yet verifying their outputs remains a significant challenge, especially for complex open-ended tasks such as consolidation, summarization, and knowledge extraction. To address this, we introduce CheckEmbed (CE): a simple, scalable, and accurate verification method. CE reduces each LLM answer to a single embedding vector using powerful modern embedding LLM models like SFR-Embedding-Mistral. Prior methods such as BERTScore and SelfCheckGPT relied on weaker encoders like BERT, forcing them to operate at token or sentence granularity. In contrast, CE performs fast, semantically rich comparisons directly at the whole-answer level, overcoming key limitations in both accuracy and scalability. We conduct a comprehensive design and time complexity analysis across 13 verification baselines, including classical text scorers (e.g., BLEU), stability-based methods (e.g., SelfCheckGPT), and generative evaluators (e.g., LLM-as-a-Judge), which highlights the effectiveness, efficiency, versatility, and simplicity of CE. Empirical results show that CE reliably detects hallucinations in both closed and open-ended tasks. We further present evidence that CE generalizes beyond text to other modalities such as vision, establishing it as a practical and versatile verification framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04019v1">CETBench: A Novel Dataset constructed via Transformations over Programs for Benchmarking LLMs for Code-Equivalence Checking</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      LLMs have been extensively used for the task of automated code generation. In this work, we examine the applicability of LLMs for the related but relatively unexplored task of code-equivalence checking, i.e., given two programs, whether they are functionally equivalent or not. This is an important problem since benchmarking code equivalence can play a critical role in evaluating LLM capabilities for tasks such as code re-writing and code translation. Towards this end, we present CETBench - Code Equivalence with Transformations Benchmark, constructed via a repository of programs, where two programs in the repository may be solving the same or different tasks. Each instance in our dataset is obtained by taking a pair of programs in the repository and applying a random series of pre-defined code transformations, resulting in (non-)equivalent pairs. Our analysis on this dataset reveals a surprising finding that very simple code transformations in the underlying pair of programs can result in a significant drop in performance of SOTA LLMs for the task of code-equivalence checking. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the transformed pairs of programs. Our approach for dataset generation is generic, and can be used with repositories with varying program difficulty levels and allows for applying varying numbers as well as kinds of transformations. In our experiments, we perform ablations over the difficulty level of original programs, as well as the kind of transformations used in generating pairs for equivalence checking. Our analysis presents deep insights into the working of LLMs for the task of code-equivalence, and points to the fact that they may still be far from what could be termed as a semantic understanding of the underlying code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04018v1">AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Prepint, under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. Prior work has examined agents' ability to enact misaligned behaviour (misalignment capability) and their compliance with harmful instructions (misuse propensity). However, the likelihood of agents attempting misaligned behaviours in real-world settings (misalignment propensity) remains poorly understood. We introduce a misalignment propensity benchmark, AgentMisalignment, consisting of a suite of realistic scenarios in which LLM agents have the opportunity to display misaligned behaviour. We organise our evaluations into subcategories of misaligned behaviours, including goal-guarding, resisting shutdown, sandbagging, and power-seeking. We report the performance of frontier models on our benchmark, observing higher misalignment on average when evaluating more capable models. Finally, we systematically vary agent personalities through different system prompts. We find that persona characteristics can dramatically and unpredictably influence misalignment tendencies -- occasionally far more than the choice of model itself -- highlighting the importance of careful system prompt engineering for deployed AI agents. Our work highlights the failure of current alignment methods to generalise to LLM agents, and underscores the need for further propensity evaluations as autonomous systems become more prevalent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04015v1">GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown great potential in recommender systems, the prohibitive computational costs for fine-tuning LLMs on entire datasets hinder their successful deployment in real-world scenarios. To develop affordable and effective LLM-based recommender systems, we focus on the task of coreset selection which identifies a small subset of fine-tuning data to optimize the test loss, thereby facilitating efficient LLMs' fine-tuning. Although there exist some intuitive solutions of subset selection, including distribution-based and importance-based approaches, they often lead to suboptimal performance due to the misalignment with downstream fine-tuning objectives or weak generalization ability caused by individual-level sample selection. To overcome these challenges, we propose GORACS, which is a novel Group-level Optimal tRAnsport-guided Coreset Selection framework for LLM-based recommender systems. GORACS is designed based on two key principles for coreset selection: 1) selecting the subsets that minimize the test loss to align with fine-tuning objectives, and 2) enhancing model generalization through group-level data selection. Corresponding to these two principles, GORACS has two key components: 1) a Proxy Optimization Objective (POO) leveraging optimal transport and gradient information to bound the intractable test loss, thus reducing computational costs by avoiding repeated LLM retraining, and 2) a two-stage Initialization-Then-Refinement Algorithm (ITRA) for efficient group-level selection. Our extensive experiments across diverse recommendation datasets and tasks validate that GORACS significantly reduces fine-tuning costs of LLMs while achieving superior performance over the state-of-the-art baselines and full data training. The source code of GORACS are available at https://github.com/Mithas-114/GORACS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02678v1">TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15319v2">GPTVQ: The Blessing of Dimensionality for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      In this work we show that the size versus accuracy trade-off of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE. Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B model, depending on quantization setting. Lastly, with on-device timings for VQ decompression on a mobile CPU we show that VQ leads to improved latency compared to using a 4-bit integer format.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02672v1">EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 47 pages, 24 figures
    </div>
    <details class="paper-abstract">
      We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02659v1">Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) are increasingly used in diverse applications, where they are assigned a specific persona - such as a happy high school teacher - to guide their responses. While prior research has examined how well LLMs adhere to predefined personas in writing style, a comprehensive analysis of consistency across different personas and task types is lacking. In this paper, we introduce a new standardized framework to analyze consistency in persona-assigned LLMs. We define consistency as the extent to which a model maintains coherent responses when assigned the same persona across different tasks and runs. Our framework evaluates personas across four different categories (happiness, occupation, personality, and political stance) spanning multiple task dimensions (survey writing, essay generation, social media post generation, single turn, and multi-turn conversations). Our findings reveal that consistency is influenced by multiple factors, including the assigned persona, stereotypes, and model design choices. Consistency also varies across tasks, increasing with more structured tasks and additional context. All code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21271v4">EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA, a novel fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{6.74\%}$ on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07965v5">SHARP: Unlocking Interactive Hallucination via Stance Transfer in Role-Playing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 28 pages, unfortunately accepted to findings with Meta 4, acknowledge and apologize to the reviewers and area chair who support our work in the discussion period
    </div>
    <details class="paper-abstract">
      The advanced role-playing capabilities of Large Language Models (LLMs) have enabled rich interactive scenarios, yet existing research in social interactions neglects hallucination while struggling with poor generalizability and implicit character fidelity judgments. To bridge this gap, motivated by human behaviour, we introduce a generalizable and explicit paradigm for uncovering interactive patterns of LLMs across diverse worldviews. Specifically, we first define interactive hallucination through stance transfer, then construct SHARP, a benchmark built by extracting relations from commonsense knowledge graphs and utilizing LLMs' inherent hallucination properties to simulate multi-role interactions. Extensive experiments confirm our paradigm's effectiveness and stability, examine the factors that influence these metrics, and challenge conventional hallucination mitigation solutions. More broadly, our work reveals a fundamental limitation in popular post-training methods for role-playing LLMs: the tendency to obscure knowledge beneath style, resulting in monotonous yet human-like behaviors - interactive hallucination.
    </details>
</div>
