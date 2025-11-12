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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- Part 14
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v3">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02873v1">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23908v2">Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 9 pages, 2 figures, accepted at ACL 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Discovering and evaluating long-form talk content such as videos and podcasts poses a significant challenge for users, as it requires a considerable time investment. Previews offer a practical solution by providing concise snippets that showcase key moments of the content, enabling users to make more informed and confident choices. We propose an LLM-based approach for generating podcast episode previews and deploy the solution at scale, serving hundreds of thousands of podcast previews in a real-world application. Comprehensive offline evaluations and online A/B testing demonstrate that LLM-generated previews consistently outperform a strong baseline built on top of various ML expert models, showcasing a significant reduction in the need for meticulous feature engineering. The offline results indicate notable enhancements in understandability, contextual clarity, and interest level, and the online A/B test shows a 4.6% increase in user engagement with preview content, along with a 5x boost in processing efficiency, offering a more streamlined and performant solution compared to the strong baseline of feature-engineered expert models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02847v1">CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by USENIX ATC 2025
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge devices is crucial for delivering fast responses and ensuring data privacy. However, the limited storage, weight, and power of edge devices make it difficult to deploy LLM-powered applications. These devices must balance latency requirements with energy consumption and model accuracy. In this paper, we first quantify the challenges of deploying LLMs on off-the-shelf edge devices and then we present CLONE, an in-depth algorithm-hardware co-design at both the model- and system-level that intelligently integrates real-time, energy optimization while maintaining robust generality. In order to maximize the synergistic benefits of these algorithms in always-on and intermediate edge computing settings, we specialize in a 28nm scalable hardware accelerator system. We implement and extensively evaluate CLONE on two off-the-shelf edge platforms. Experiments show that CLONE effectively accelerates the inference process up to 11.92x, and saves energy up to 7.36x, while maintaining high-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02818v1">ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive results in natural language processing tasks but require a significant amount of computational and memory resources. Structured matrix representations are a promising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM output is invariant under certain orthogonal transformations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structured classes. The proposed approach is applicable to various types of structured matrices that support efficient projection operations. Code is available at https://github.com/GrishKate/ProcrustesGPT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19146v5">Puzzle: Distillation-Based NAS for Inference-Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer remarkable capabilities, yet their high inference costs restrict wider adoption. While increasing parameter counts improves accuracy, it also broadens the gap between state-of-the-art capabilities and practical deployability. We present Puzzle, a hardware-aware framework that accelerates the inference of LLMs while preserving their capabilities. Using neural architecture search (NAS) at a large-scale, Puzzle optimizes models with tens of billions of parameters. Our approach utilizes blockwise local knowledge distillation (BLD) for parallel architecture exploration and employs mixed-integer programming for precise constraint optimization. We showcase our framework's impact via Llama-3.1-Nemotron-51B-Instruct (Nemotron-51B) and Llama-3.3-Nemotron-49B, two publicly available models derived from Llama-70B-Instruct. Both models achieve a 2.17x inference throughput speedup, fitting on a single NVIDIA H100 GPU while retaining 98.4% of the original model's benchmark accuracies. These are the most accurate models supporting single H100 GPU inference with large batch sizes, despite training on 45B tokens at most, far fewer than the 15T used to train Llama-70B. Lastly, we show that lightweight alignment on these derived models allows them to surpass the parent model in specific capabilities. Our work establishes that powerful LLM models can be optimized for efficient deployment with only negligible loss in quality, underscoring that inference performance, not parameter count alone, should guide model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10927v3">OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 This paper has been accepted to ACM ETRA 2025 and published on PACMHCI
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02758v1">Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications
    </div>
    <details class="paper-abstract">
      Vocabulary use is a fundamental aspect of second language (L2) proficiency. To date, its assessment by automated systems has typically examined the context-independent, or part-of-speech (PoS) related use of words. This paper introduces a novel approach to enable fine-grained vocabulary evaluation exploiting the precise use of words within a sentence. The scheme combines large language models (LLMs) with the English Vocabulary Profile (EVP). The EVP is a standard lexical resource that enables in-context vocabulary use to be linked with proficiency level. We evaluate the ability of LLMs to assign proficiency levels to individual words as they appear in L2 learner writing, addressing key challenges such as polysemy, contextual variation, and multi-word expressions. We compare LLMs to a PoS-based baseline. LLMs appear to exploit additional semantic information that yields improved performance. We also explore correlations between word-level proficiency and essay-level proficiency. Finally, the approach is applied to examine the consistency of the EVP proficiency levels. Results show that LLMs are well-suited for the task of vocabulary assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23754v2">DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01586v2">SubTrack++ : Gradient Subspace Tracking for Scalable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) is highly resource-intensive due to their massive number of parameters and the overhead of optimizer states. While recent work has aimed to reduce memory consumption, such efforts often entail trade-offs among memory efficiency, training time, and model performance. Yet, true democratization of LLMs requires simultaneous progress across all three dimensions. To this end, we propose SubTrack++ that leverages Grassmannian gradient subspace tracking combined with projection-aware optimizers, enabling Adam's internal statistics to adapt to changes in the optimization subspace. Additionally, employing recovery scaling, a technique that restores information lost through low-rank projections, further enhances model performance. Our method demonstrates SOTA convergence by exploiting Grassmannian geometry and achieves lowest evaluation loss, outperforming the current SOTA while reducing pretraining wall time by 43% and maintaining the memory footprint on a 1B-parameter Llama model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02742v1">Prompt-Unseen-Emotion: Zero-shot Expressive Speech Synthesis with Prompt-LLM Contextual Knowledge for Mixed Emotions</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Existing expressive text-to-speech (TTS) systems primarily model a limited set of categorical emotions, whereas human conversations extend far beyond these predefined emotions, making it essential to explore more diverse emotional speech generation for more natural interactions. To bridge this gap, this paper proposes a novel prompt-unseen-emotion (PUE) approach to generate unseen emotional speech via emotion-guided prompt learning. PUE is trained utilizing an LLM-TTS architecture to ensure emotional consistency between categorical emotion-relevant prompts and emotional speech, allowing the model to quantitatively capture different emotion weightings per utterance. During inference, mixed emotional speech can be generated by flexibly adjusting emotion proportions and leveraging LLM contextual knowledge, enabling the model to quantify different emotional styles. Our proposed PUE successfully facilitates expressive speech synthesis of unseen emotions in a zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02718v1">Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across diverse natural language processing tasks, yet their deployment in real-world applications is hindered by fixed knowledge cutoffs and difficulties in generating controllable, accurate outputs in a single inference. Multi-agent systems (MAS) built from specialized LLM agents offer a promising solution, enabling dynamic collaboration and iterative reasoning. However, optimizing these systems remains a challenge, as conventional methods such as prompt engineering and supervised fine-tuning entail high engineering overhead and limited adaptability. Reinforcement learning (RL), particularly multi-agent reinforcement learning (MARL), provides a scalable framework by refining agent policies based on system-level feedback. Nevertheless, existing MARL algorithms, such as Multi-Agent Proximal Policy Optimization (MAPPO), rely on Critic networks, which can cause training instability and increase computational burden. To address these limitations and target the prototypical Multi-Agent Search System (MASS), we propose Multi-Agent Heterogeneous Group Policy Optimization (MHGPO), a novel Critic-free algorithm that guides policy updates by estimating relative reward advantages across heterogeneous groups of rollouts. MHGPO eliminates the need for Critic networks, enhancing stability and reducing computational overhead. Additionally, we introduce three group rollout sampling strategies that trade off between efficiency and effectiveness. Experiments on a multi-agent LLM-based search system demonstrate that MHGPO consistently outperforms MAPPO in both task performance and computational efficiency, without requiring warm-up, underscoring its potential for stable and scalable optimization of complex LLM-based MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17878v2">Towards Enhanced Immersion and Agency for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL'2025
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00519v2">CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to Association for Computational Linguistics Findings (ACL) 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at https://github.com/peachch/CausalAbstain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.10424v3">LLM-as-an-Interviewer: Beyond Static Testing Through Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We introduce LLM-as-an-Interviewer, a novel paradigm for evaluating large language models (LLMs). This approach leverages multi-turn interactions where the LLM interviewer actively provides feedback on responses and poses follow-up questions to the evaluated LLM. At the start of the interview, the LLM interviewer dynamically modifies datasets to generate initial questions, mitigating data contamination. We apply the LLM-as-an-Interviewer framework to evaluate six models on the MATH and DepthQA tasks. Our results show that the framework effectively provides insights into LLM performance, including the quality of initial responses, adaptability to feedback, and ability to address follow-up queries like clarification or additional knowledge requests. The framework also addresses key limitations of conventional methods like LLM-as-a-Judge, including verbosity bias and inconsistency across runs. Finally, we propose the Interview Report, which aggregates insights from the interview process, providing examples and a comprehensive analysis of the LLM's strengths and weaknesses. This report offers a detailed snapshot of the model's real-world applicability. The code for our framework is publicly available at https://github.com/interview-eval/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.11674v2">LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Time series forecasting using LLMs
    </div>
    <details class="paper-abstract">
      Time series forecasting remains a challenging task, particularly in the context of complex multiscale temporal patterns. This study presents LLM-Mixer, a framework that improves forecasting accuracy through the combination of multiscale time-series decomposition with pre-trained LLMs (Large Language Models). LLM-Mixer captures both short-term fluctuations and long-term trends by decomposing the data into multiple temporal resolutions and processing them with a frozen LLM, guided by a textual prompt specifically designed for time-series data. Extensive experiments conducted on multivariate and univariate datasets demonstrate that LLM-Mixer achieves competitive performance, outperforming recent state-of-the-art models across various forecasting horizons. This work highlights the potential of combining multiscale analysis and LLMs for effective and scalable time-series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.18458v3">A Survey of LLM $\times$ DATA</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Please refer to the paper list at: https://github.com/weAIDB/awesome-data-llm
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.18547v5">Token-Budget-Aware LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning and enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework that dynamically adjusts the number of reasoning tokens based on the reasoning complexity of each problem. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02314v1">ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in transforming machine learning research, yet their capability to faithfully implement novel ideas from recent research papers-ideas unseen during pretraining-remains unclear. We introduce ResearchCodeBench, a benchmark of 212 coding challenges that evaluates LLMs' ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code. We assessed 30+ proprietary and open-source LLMs, finding that even the best models correctly implement less than 40% of the code. We find Gemini-2.5-Pro-Preview to perform best at 37.3% success rate, with O3 (High) and O4-mini (High) following behind at 32.3% and 30.8% respectively. We present empirical findings on performance comparison, contamination, and error patterns. By providing a rigorous and community-driven evaluation platform, ResearchCodeBench enables continuous understanding and advancement of LLM-driven innovation in research code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v9">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v4">SafeSwitch: Steering Unsafe LLM Behavior via Internal Activation Signals</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit exceptional capabilities across various tasks but also pose risks by generating harmful content. Existing safety mechanisms, while improving model safety, often lead to overly cautious behavior and fail to fully leverage LLMs' internal cognitive processes. Inspired by humans' reflective thinking capability, we first show that LLMs can similarly perform internal assessments about safety in their internal states. Building on this insight, we propose SafeSwitch, a dynamic framework that regulates unsafe outputs by utilizing the prober-based internal state monitor that actively detects harmful intentions, and activates a safety head that leads to safer and more conservative responses only when necessary. SafeSwitch reduces harmful outputs by approximately 80% on harmful queries while maintaining strong utility, reaching a Pareto optimal among several methods. Our method is also advantageous over traditional methods in offering more informative, context-aware refusals, and achieves these benefits while only tuning less than 6% of the original parameters. SafeSwitch demonstrates large language models' capacity for self-awareness and reflection regarding safety, offering a promising approach to more nuanced and effective safety controls. Codes for this work are available at https://github.com/Hanpx20/SafeSwitch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19501v2">Toward Scientific Reasoning in LLMs: Training from Expert Discussions via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We investigate how to teach large language models (LLMs) to perform scientific reasoning by leveraging expert discussions as a learning signal. Focusing on the genomics domain, we develop an automated pipeline to extract trainable data and introduce Genome-Bench, a new benchmark constructed from over a decade of scientific forum discussions on genome engineering. Our pipeline transforms raw interactions into a reinforcement learning-friendly multiple-choice questions format, supported by 3000+ high-quality question-answer pairs spanning foundational biology, experimental troubleshooting, tool usage, and beyond. We fine-tune an LLM using RL with a rule-based reward signal derived from the synthetic MCQ dataset to enhance domain-specific reasoning. Our results show that reinforcement learning from scientific discussions improves model performance by over 15% compared to the base model on Genome-Bench, narrowing the gap between open-source LLMs and expert-level reasoning. To our knowledge, this is the first end-to-end pipeline for teaching LLMs to reason from scientific discussions, with promising potential for generalization across scientific domains beyond biology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17579v2">Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between pronoun production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size-with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior. Our codes and results are available at https://github.com/LingMechLab/Production-Interpretation_Asymmetries_ACL2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02211v1">Improving LLM-Generated Code Quality with GRPO</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are gaining widespread use for code generation. Recent training procedures use execution feedback as a reward signal, typically focusing on the functional correctness of the code, using unit test pass rate as a reward signal. However, this reward signal fails to capture notions of maintainability, quality and safety of the code produced. We address this under-explored area and develop a comprehensive library to quantify various aspects of code quality, and use it as a reward in GRPO. We find GRPO increases code quality according to this measure, which is confirmed by expert, blinded human annotators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02208v1">KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) post-training have leveraged two distinct paradigms to enhance reasoning capabilities: reinforcement learning (RL) and knowledge distillation (KD). While RL enables the emergence of complex reasoning behaviors, it often suffers from low sample efficiency when the initial policy struggles to explore high-reward trajectories. Conversely, KD improves learning efficiency via mimicking the teacher model but tends to generalize poorly to out-of-domain scenarios. In this work, we present \textbf{KDRL}, a \textit{unified post-training framework} that jointly optimizes a reasoning model through teacher supervision (KD) and self-exploration (RL). Specifically, KDRL leverages policy gradient optimization to simultaneously minimize the reverse Kullback-Leibler divergence (RKL) between the student and teacher distributions while maximizing the expected rule-based rewards. We first formulate a unified objective that integrates GRPO and KD, and systematically explore how different KL approximations, KL coefficients, and reward-guided KD strategies affect the overall post-training dynamics and performance. Empirical results on multiple reasoning benchmarks demonstrate that KDRL outperforms GRPO and various KD baselines while achieving a favorable balance between performance and reasoning token efficiency. These findings indicate that integrating KD and RL serves as an effective and efficient strategy to train reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03525v4">UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      This paper introduces UnSeenTimeQA, a novel data contamination-free time-sensitive question-answering (TSQA) benchmark. It differs from existing TSQA benchmarks by avoiding web-searchable queries grounded in the real world. We present a series of time-sensitive event scenarios based on synthetically generated facts. It requires large language models (LLMs) to engage in genuine temporal reasoning without depending on the factual knowledge acquired during the pre-training phase. Our data generation framework enables on-demand generation of new samples, mitigating the risk of data leakage. We designed three types of time-sensitive questions to test LLMs' temporal reasoning abilities over sequential and parallel event occurrences. Our evaluation of five LLMs on synthetic fact-based TSQA reveals mixed results: while they perform well on simpler subsets, their overall performance remains inferior as compared to real world fact-based TSQA. Error analysis indicates that LLMs face difficulties in reasoning over long-range event dependencies and parallel events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02177v1">Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning, such as PPO and GRPO, has powered recent breakthroughs in LLM reasoning. Scaling rollout to sample more prompts enables models to selectively use higher-quality data for training, which can stabilize RL training and improve model performance. However, this comes at the cost of significant computational overhead. In this paper, we show that a substantial portion of this overhead can be avoided by skipping uninformative prompts before rollout. Our analysis of reward dynamics reveals a strong temporal consistency in prompt value: prompts that are uninformative in one epoch of training are likely to remain uninformative in future epochs. Based on these insights, we propose GRESO (GRPO with Efficient Selective Rollout), an online, lightweight pre-rollout filtering algorithm that predicts and skips uninformative prompts using reward training dynamics. By evaluating GRESO on a broad range of math reasoning benchmarks and models, such as Qwen2.5-Math-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, and Qwen2.5-Math-7B, we show that GRESO achieves up to 2.4x wall-clock time speedup in rollout and up to 2.0x speedup in total training time without accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16873v2">AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to ICML 2025. Code is available at http://github.com/facebookresearch/advprompter
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires a time-consuming search for adversarial prompts, whereas automatic adversarial prompt generation often leads to semantically meaningless attacks that do not scale well. In this paper, we present a novel method that uses another LLM, called AdvPrompter, to generate human-readable adversarial prompts in seconds. AdvPrompter, which is trained using an alternating optimization algorithm, generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show highly competitive results on the AdvBench and HarmBench datasets, that also transfer to closed-source black-box LLMs. We also show that training on adversarial suffixes generated by AdvPrompter is a promising strategy for improving the robustness of LLMs to jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06150v2">A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01954v1">DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 Main. Code is available at https://github.com/VILA-Lab/DRAG
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating hallucinated content from Humans. In this work, we introduce $\texttt{DRAG}$, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph-based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, $\texttt{DRAG}$ effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With $\texttt{DRAG}$, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-sized LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01939v1">Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 25 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01900v1">COALESCE: Economic and Security Dynamics of Skill-Based Task Outsourcing Among Team of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 2 figures, github linked
    </div>
    <details class="paper-abstract">
      The meteoric rise and proliferation of autonomous Large Language Model (LLM) agents promise significant capabilities across various domains. However, their deployment is increasingly constrained by substantial computational demands, specifically for Graphics Processing Unit (GPU) resources. This paper addresses the critical problem of optimizing resource utilization in LLM agent systems. We introduce COALESCE (Cost-Optimized and Secure Agent Labour Exchange via Skill-based Competence Estimation), a novel framework designed to enable autonomous LLM agents to dynamically outsource specific subtasks to specialized, cost-effective third-party LLM agents. The framework integrates mechanisms for hybrid skill representation, dynamic skill discovery, automated task decomposition, a unified cost model comparing internal execution costs against external outsourcing prices, simplified market-based decision-making algorithms, and a standardized communication protocol between LLM agents. Comprehensive validation through 239 theoretical simulations demonstrates 41.8\% cost reduction potential, while large-scale empirical validation across 240 real LLM tasks confirms 20.3\% cost reduction with proper epsilon-greedy exploration, establishing both theoretical viability and practical effectiveness. The emergence of proposed open standards like Google's Agent2Agent (A2A) protocol further underscores the need for frameworks like COALESCE that can leverage such standards for efficient agent interaction. By facilitating a dynamic market for agent capabilities, potentially utilizing protocols like A2A for communication, COALESCE aims to significantly reduce operational costs, enhance system scalability, and foster the emergence of specialized agent economies, making complex LLM agent functionalities more accessible and economically viable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03505v2">Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have extended the success of large language models (LLMs) to multiple data types, such as image, text and audio, achieving significant performance in various domains, including multimodal translation, visual question answering and content generation. Nonetheless, existing systems are inefficient to train MLLMs due to substantial GPU bubbles caused by the heterogeneous modality models and complex data dependencies in 3D parallelism. This paper proposes Optimus, a distributed MLLM training system that reduces end-to-end MLLM training time. Optimus is based on our principled analysis that scheduling the encoder computation within the LLM bubbles can reduce bubbles in MLLM training. To make scheduling encoder computation possible for all GPUs, Optimus searches the separate parallel plans for encoder and LLM, and adopts a bubble scheduling algorithm to enable exploiting LLM bubbles without breaking the original data dependencies in the MLLM model architecture. We further decompose encoder layer computation into a series of kernels, and analyze the common bubble pattern of 3D parallelism to carefully optimize the sub-millisecond bubble scheduling, minimizing the overall training time. Our experiments in a production cluster show that Optimus accelerates MLLM training by 20.5%-21.3% with ViT-22B and GPT-175B model over 3072 GPUs compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10918v2">INVARLLM: LLM-assisted Physical Invariant Extraction for Cyber-Physical Systems Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Cyber-Physical Systems (CPS) are vulnerable to cyber-physical attacks that violate physical laws. While invariant-based anomaly detection is effective, existing methods are limited: data-driven approaches lack semantic context, and physics-based models require extensive manual work. We propose INVARLLM, a hybrid framework that uses large language models (LLMs) to extract semantic information from CPS documentation and generate physical invariants, then validates these against real system data using a PCMCI+-inspired K-means method. This approach combines LLM semantic understanding with empirical validation to ensure both interpretability and reliability. We evaluate INVARLLM on SWaT and WADI datasets, achieving 100% precision in anomaly detection with no false alarms, outperforming all existing methods. Our results demonstrate that integrating LLM-derived semantics with statistical validation provides a scalable and dependable solution for CPS security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01839v1">Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from static tools to fully agentic systems, their potential for transforming social science research has become increasingly evident. This paper introduces a structured framework for understanding the diverse applications of LLM-based agents, ranging from simple data processors to complex, multi-agent systems capable of simulating emergent social dynamics. By mapping this developmental continuum across six levels, the paper clarifies the technical and methodological boundaries between different agentic architectures, providing a comprehensive overview of current capabilities and future potential. It highlights how lower-tier systems streamline conventional tasks like text classification and data annotation, while higher-tier systems enable novel forms of inquiry, including the study of group dynamics, norm formation, and large-scale social processes. However, these advancements also introduce significant challenges, including issues of reproducibility, ethical oversight, and the risk of emergent biases. The paper critically examines these concerns, emphasizing the need for robust validation protocols, interdisciplinary collaboration, and standardized evaluation metrics. It argues that while LLM-based agents hold transformative potential for the social sciences, realizing this promise will require careful, context-sensitive deployment and ongoing methodological refinement. The paper concludes with a call for future research that balances technical innovation with ethical responsibility, encouraging the development of agentic systems that not only replicate but also extend the frontiers of social science, offering new insights into the complexities of human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01825v1">Which Factors Make Code LLMs More Vulnerable to Backdoor Attacks? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Code LLMs are increasingly employed in software development. However, studies have shown that they are vulnerable to backdoor attacks: when a trigger (a specific input pattern) appears in the input, the backdoor will be activated and cause the model to generate malicious outputs. Researchers have designed various triggers and demonstrated the feasibility of implanting backdoors by poisoning a fraction of the training data. Some basic conclusions have been made, such as backdoors becoming easier to implant when more training data are modified. However, existing research has not explored other factors influencing backdoor attacks on Code LLMs, such as training batch size, epoch number, and the broader design space for triggers, e.g., trigger length. To bridge this gap, we use code summarization as an example to perform an empirical study that systematically investigates the factors affecting backdoor effectiveness and understands the extent of the threat posed. Three categories of factors are considered: data, model, and inference, revealing previously overlooked findings. We find that the prevailing consensus -- that attacks are ineffective at extremely low poisoning rates -- is incorrect. The absolute number of poisoned samples matters as well. Specifically, poisoning just 20 out of 454K samples (0.004\% poisoning rate -- far below the minimum setting of 0.1\% in prior studies) successfully implants backdoors! Moreover, the common defense is incapable of removing even a single poisoned sample from it. Additionally, small batch sizes increase the risk of backdoor attacks. We also uncover other critical factors such as trigger types, trigger length, and the rarity of tokens in the triggers, leading to valuable insights for assessing Code LLMs' vulnerability to backdoor attacks. Our study highlights the urgent need for defense mechanisms against extremely low poisoning rate settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01814v1">Analysis of LLM Bias (Chinese Propaganda & Anti-US Sentiment) in DeepSeek-R1 vs. ChatGPT o3-mini-high</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly shape public understanding and civic decisions, yet their ideological neutrality is a growing concern. While existing research has explored various forms of LLM bias, a direct, cross-lingual comparison of models with differing geopolitical alignments-specifically a PRC-system model versus a non-PRC counterpart-has been lacking. This study addresses this gap by systematically evaluating DeepSeek-R1 (PRC-aligned) against ChatGPT o3-mini-high (non-PRC) for Chinese-state propaganda and anti-U.S. sentiment. We developed a novel corpus of 1,200 de-contextualized, reasoning-oriented questions derived from Chinese-language news, presented in Simplified Chinese, Traditional Chinese, and English. Answers from both models (7,200 total) were assessed using a hybrid evaluation pipeline combining rubric-guided GPT-4o scoring with human annotation. Our findings reveal significant model-level and language-dependent biases. DeepSeek-R1 consistently exhibited substantially higher proportions of both propaganda and anti-U.S. bias compared to ChatGPT o3-mini-high, which remained largely free of anti-U.S. sentiment and showed lower propaganda levels. For DeepSeek-R1, Simplified Chinese queries elicited the highest bias rates; these diminished in Traditional Chinese and were nearly absent in English. Notably, DeepSeek-R1 occasionally responded in Simplified Chinese to Traditional Chinese queries and amplified existing PRC-aligned terms in its Chinese answers, demonstrating an "invisible loudspeaker" effect. Furthermore, such biases were not confined to overtly political topics but also permeated cultural and lifestyle content, particularly in DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01804v1">A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01770v1">ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at https://github.com/weizeming/ReGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01734v1">Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law -- a statistical pattern where lower digits occur more frequently as leading digits -- we hypothesize that the long-tailed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark with uniformly distributed ground-truth digits across seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01732v1">Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. These data most often contain trillions of tokens with large portions of copyrighted or proprietary content, which hinders the usage of such models under AI legislation. This raises the need for truly open pre-training data that is compliant with the data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for language model pre-training. The data assembled in Common Corpus are either uncopyrighted or under permissible licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the main European languages to low-resource ones rarely present in pre-training datasets; in addition, it includes a large portion of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs in diverse areas of knowledge. In this technical report, we present the detailed provenance of data assembling and the details of dataset filtering and curation. Being already used by such industry leaders as Anthropic and multiple LLM training projects, we believe that Common Corpus will become a critical infrastructure for open science research in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v1">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01702v1">mdok of KInIT: Robustly Fine-tuned LLM for Binary and Multiclass AI-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) are able to generate high-quality texts in multiple languages. Such texts are often not recognizable by humans as generated, and therefore present a potential of LLMs for misuse (e.g., plagiarism, spams, disinformation spreading). An automated detection is able to assist humans to indicate the machine-generated texts; however, its robustness to out-of-distribution data is still challenging. This notebook describes our mdok approach in robust detection, based on fine-tuning smaller LLMs for text classification. It is applied to both subtasks of Voight-Kampff Generative AI Detection 2025, providing remarkable performance in binary detection as well as in multiclass (1st rank) classification of various cases of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01698v1">When LLMs Team Up: The Emergence of Collaborative Affective Computing</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 7 figures, and 3 tables
    </div>
    <details class="paper-abstract">
      Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v1">SALAD: Systematic Assessment of Machine Unlearing on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01687v1">StochasTok: Improving Fine-Grained Subword Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still often struggle with seemingly simple subword-level tasks like How many 'r's in 'strawberry'?. A key factor behind these failures is tokenization which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to 'see' their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs' downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok's simplicity allows seamless integration at any stage of the training pipeline; and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models. Code open-sourced at: https://github.com/anyasims/stochastok.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01646v1">ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 37 pages, 8 figures, 11 tables
    </div>
    <details class="paper-abstract">
      We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01631v1">Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance. To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01616v1">MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The emergence of multimodal LLM-based agents (MLAs) has transformed interaction paradigms by seamlessly integrating vision, language, action and dynamic environments, enabling unprecedented autonomous capabilities across GUI applications ranging from web automation to mobile systems. However, MLAs introduce critical trustworthiness challenges that extend far beyond traditional language models' limitations, as they can directly modify digital states and trigger irreversible real-world consequences. Existing benchmarks inadequately tackle these unique challenges posed by MLAs' actionable outputs, long-horizon uncertainty and multimodal attack vectors. In this paper, we introduce MLA-Trust, the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: truthfulness, controllability, safety and privacy. We utilize websites and mobile applications as realistic testbeds, designing 34 high-risk interactive tasks and curating rich evaluation datasets. Large-scale experiments involving 13 state-of-the-art agents reveal previously unexplored trustworthiness vulnerabilities unique to multimodal interactive scenarios. For instance, proprietary and open-source GUI-interacting MLAs pose more severe trustworthiness risks than static MLLMs, particularly in high-stakes domains; the transition from static MLLMs into interactive MLAs considerably compromises trustworthiness, enabling harmful content generation in multi-step interactions that standalone MLLMs would typically prevent; multi-step execution, while enhancing the adaptability of MLAs, involves latent nonlinear risk accumulation across successive interactions, circumventing existing safeguards and resulting in unpredictable derived risks. Moreover, we present an extensible toolbox to facilitate continuous evaluation of MLA trustworthiness across diverse interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v1">EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01484v1">LLM in the Loop: Creating the PARADEHATE Dataset for Hate Speech Detoxification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct PARADEHATE, a large-scale parallel dataset specifically for hatespeech detoxification. We release PARADEHATE as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on PARADEHATE, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01407v1">Comparing LLM-generated and human-authored news text using formal syntactic theory</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 15 figures, 13 tables; accepted to ACL-2025 main
    </div>
    <details class="paper-abstract">
      This study provides the first comprehensive comparison of New York Times-style text generated by six large language models against real, human-authored NYT writing. The comparison is based on a formal syntactic theory. We use Head-driven Phrase Structure Grammar (HPSG) to analyze the grammatical structure of the texts. We then investigate and illustrate the differences in the distributions of HPSG grammar types, revealing systematic distinctions between human and LLM-generated writing. These findings contribute to a deeper understanding of the syntactic behavior of LLMs as well as humans, within the NYT genre.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01386v1">ThinkEval: Practical Evaluation of Knowledge Preservation and Consistency in LLM Editing with Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Model editing has become an important tool for addressing privacy, bias, and misinformation in large language models (LLMs) by enabling updates to knowledge without the need for retraining from scratch. However, existing editing techniques often target isolated facts, ignoring ripple effects on related knowledge, allowing edited facts to remain deducible and compromising broader contextual integrity. For example, changing Harry Potter's school from Hogwarts to Ilvermorny requires reassigning his house from Gryffindor to a suitable alternative while preserving Gryffindor's relationship with Hogwarts. In this work, we present a new model-editing setting, deep editing, to show: (1) how editing techniques fail to handle connected facts, evaluating how original knowledge sneaks through unchanged causal links, and (2) their impact on broader contextual knowledge. We introduce ThinkEval, a framework to systematically evaluate model-editing techniques by building model-specific knowledge graphs to analyze pre- and post-edit effects on fact persistence and catastrophic forgetting. We present KnowGIC, a benchmark created with ThinkEval, consisting of sequentially linked queries to measure these effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. We find that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge. Our dataset is available at: https://anonymous.4open.science/r/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01374v1">Compiler Optimization via LLM Reasoning for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimization to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process, guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-aware transformations that reflect the current program state and accumulated performance feedback. Monte Carlo tree search (MCTS) incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01370v1">PointT2I: LLM-based text-to-image generation via keypoints</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generation model has made significant advancements, resulting in high-quality images aligned with an input prompt. However, despite T2I generation's ability to generate fine-grained images, it still faces challenges in accurately generating images when the input prompt contains complex concepts, especially human pose. In this paper, we propose PointT2I, a framework that effectively generates images that accurately correspond to the human pose described in the prompt by using a large language model (LLM). PointT2I consists of three components: Keypoint generation, Image generation, and Feedback system. The keypoint generation uses an LLM to directly generate keypoints corresponding to a human pose, solely based on the input prompt, without external references. Subsequently, the image generation produces images based on both the text prompt and the generated keypoints to accurately reflect the target pose. To refine the outputs of the preceding stages, we incorporate an LLM-based feedback system that assesses the semantic consistency between the generated contents and the given prompts. Our framework is the first approach to leveraging LLM for keypoints-guided image generation without any fine-tuning, producing accurate pose-aligned images based solely on textual prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01369v1">Incentivizing LLMs to Self-Verify Their Answers</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance during inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating its capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling. Our code is available at https://github.com/mansicer/self-verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01339v1">Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Machine unlearning offers a promising solution to privacy and safety concerns in large language models (LLMs) by selectively removing targeted knowledge while preserving utility. However, current methods are highly sensitive to downstream fine-tuning, which can quickly recover forgotten information-even from unrelated tasks. To address this, we introduce invariance into unlearning for the first time, inspired by invariant risk minimization (IRM). Building on this principle, we propose invariant LLM unlearning (ILU), a regularization-based framework that enhances robustness. Notably, ILU generalizes well to diverse fine-tuning tasks, even when trained using a single dataset. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP and MUSE benchmark, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01334v1">Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) decompose image classification into a process governed by interpretable, human-readable concepts. Recent advances in CBMs have used Large Language Models (LLMs) to generate candidate concepts. However, a critical question remains: What is the optimal number of concepts to use? Current concept banks suffer from redundancy or insufficient coverage. To address this issue, we introduce a dynamic, agent-based approach that adjusts the concept bank in response to environmental feedback, optimizing the number of concepts for sufficiency yet concise coverage. Moreover, we propose Conditional Concept Bottleneck Models (CoCoBMs) to overcome the limitations in traditional CBMs' concept scoring mechanisms. It enhances the accuracy of assessing each concept's contribution to classification tasks and feature an editable matrix that allows LLMs to correct concept scores that conflict with their internal knowledge. Our evaluations across 6 datasets show that our method not only improves classification accuracy by 6% but also enhances interpretability assessments by 30%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01290v1">TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating is built on the assumption that LLMs inherit ample knowledge, acquired during their extensive pretraining, enabling them to comprehend and discern quality differences in diverse TS data. We verify this assumption by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions via TSRater's inference on future TS samples. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01262v1">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Personalized AI assistants, a hallmark of the human-like capabilities of Large Language Models (LLMs), are a challenging application that intertwines multiple problems in LLM research. Despite the growing interest in the development of personalized assistants, the lack of an open-source conversational dataset tailored for personalization remains a significant obstacle for researchers in the field. To address this research gap, we introduce HiCUPID, a new benchmark to probe and unleash the potential of LLMs to deliver personalized responses. Alongside a conversational dataset, HiCUPID provides a Llama-3.2-based automated evaluation model whose assessment closely mirrors human preferences. We release our dataset, evaluation model, and code at https://github.com/12kimih/HiCUPID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01252v1">MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: https://github.com/Wayyuanyuan/MTCMB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01245v1">Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01237v1">Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: https://github.com/SungHo3268/KoGEM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23799v2">Estimating LLM Consistency: A User Baseline vs Surrogate Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations and sensitive to prompt perturbations, often resulting in inconsistent or unreliable generated text. Different methods have been proposed to mitigate such hallucinations and fragility -- one of them being measuring the consistency (the model's confidence in the response, or likelihood of generating a similar response when resampled) of LLM responses. In previous work, measuring consistency often relied on the probability of a response appearing within a pool of resampled responses, or internal states or logits of responses. However, it is not yet clear how well these approaches approximate how humans perceive the consistency of LLM responses. We performed a user study (n=2,976) and found current methods typically do not approximate users' perceptions of LLM consistency very well. We propose a logit-based ensemble method for estimating LLM consistency, and we show that this method matches the performance of the best-performing existing metric in estimating human ratings of LLM consistency. Our results suggest that methods of estimating LLM consistency without human evaluation are sufficiently imperfect that we suggest evaluation with human input be more broadly used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07071v3">Value Compass Benchmarks: A Platform for Fundamental and Validated Evaluation of LLMs Values</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) achieve remarkable breakthroughs, aligning their values with humans has become imperative for their responsible development and customized applications. However, there still lack evaluations of LLMs values that fulfill three desirable goals. (1) Value Clarification: We expect to clarify the underlying values of LLMs precisely and comprehensively, while current evaluations focus narrowly on safety risks such as bias and toxicity. (2) Evaluation Validity: Existing static, open-source benchmarks are prone to data contamination and quickly become obsolete as LLMs evolve. Additionally, these discriminative evaluations uncover LLMs' knowledge about values, rather than valid assessments of LLMs' behavioral conformity to values. (3) Value Pluralism: The pluralistic nature of human values across individuals and cultures is largely ignored in measuring LLMs value alignment. To address these challenges, we presents the Value Compass Benchmarks, with three correspondingly designed modules. It (i) grounds the evaluation on motivationally distinct \textit{basic values to clarify LLMs' underlying values from a holistic view; (ii) applies a \textit{generative evolving evaluation framework with adaptive test items for evolving LLMs and direct value recognition from behaviors in realistic scenarios; (iii) propose a metric that quantifies LLMs alignment with a specific value as a weighted sum over multiple dimensions, with weights determined by pluralistic values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07453v2">How well do LLMs reason over tabular data, really?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07784v2">Domain Regeneration: How well do LLMs match syntactic properties of text domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Recent improvement in large language model performance have, in all likelihood, been accompanied by improvement in how well they can approximate the distribution of their training data. In this work, we explore the following question: which properties of text domains do LLMs faithfully approximate, and how well do they do so? Applying observational approaches familiar from corpus linguistics, we prompt a commonly used, opensource LLM to regenerate text from two domains of permissively licensed English text which are often contained in LLM training data -- Wikipedia and news text. This regeneration paradigm allows us to investigate whether LLMs can faithfully match the original human text domains in a fairly semantically-controlled setting. We investigate varying levels of syntactic abstraction, from more simple properties like sentence length, and article readability, to more complex and higher order properties such as dependency tag distribution, parse depth, and parse complexity. We find that the majority of the regenerated distributions show a shifted mean, a lower standard deviation, and a reduction of the long tail, as compared to the human originals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04503v3">When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Code available at https://github.com/jeremyperez2/TelephoneGameLLM. Companion website with a Data Explorer tool at https://sites.google.com/view/telephone-game-llm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) start interacting with each other and generating an increasing amount of text online, it becomes crucial to better understand how information is transformed as it passes from one LLM to the next. While significant research has examined individual LLM behaviors, existing studies have largely overlooked the collective behaviors and information distortions arising from iterated LLM interactions. Small biases, negligible at the single output level, risk being amplified in iterated interactions, potentially leading the content to evolve towards attractor states. In a series of telephone game experiments, we apply a transmission chain design borrowed from the human cultural evolution literature: LLM agents iteratively receive, produce, and transmit texts from the previous to the next agent in the chain. By tracking the evolution of text toxicity, positivity, difficulty, and length across transmission chains, we uncover the existence of biases and attractors, and study their dependence on the initial text, the instructions, language model, and model size. For instance, we find that more open-ended instructions lead to stronger attraction effects compared to more constrained tasks. We also find that different text properties display different sensitivity to attraction effects, with toxicity leading to stronger attractors than length. These findings highlight the importance of accounting for multi-step transmission dynamics and represent a first step towards a more comprehensive understanding of LLM cultural dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03340v2">EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Theory-of-Mind (ToM), the ability to infer others' perceptions and mental states, is fundamental to human interaction but remains challenging for Large Language Models (LLMs). While existing ToM reasoning methods show promise with reasoning via perceptual perspective-taking, they often rely excessively on off-the-shelf LLMs, reducing their efficiency and limiting their applicability to high-order ToM reasoning. To address these issues, we present EnigmaToM, a novel neuro-symbolic framework that enhances ToM reasoning by integrating a Neural Knowledge Base of entity states (Enigma) for (1) a psychology-inspired iterative masking mechanism that facilitates accurate perspective-taking and (2) knowledge injection that elicits key entity information. Enigma generates structured knowledge of entity states to build spatial scene graphs for belief tracking across various ToM orders and enrich events with fine-grained entity state details. Experimental results on ToMi, HiToM, and FANToM benchmarks show that EnigmaToM significantly improves ToM reasoning across LLMs of varying sizes, particularly excelling in high-order reasoning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09439v2">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01150v2">MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 (Findings) Code and data available at https://github.com/luciusssss/MiLiC-Eval
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in high-resource languages but struggle with low-resource languages (LRLs), particularly those spoken by minority communities in China, such as Tibetan, Uyghur, Kazakh, and Mongolian. To systematically track the progress in these languages, we introduce MiLiC-Eval, a benchmark designed for minority languages in China, featuring 24K instances across 9 tasks. MiLiC-Eval focuses on underrepresented writing systems. Its parallelism between tasks and languages can provide a faithful and fine-grained assessment of linguistic and problem-solving skills. Our evaluation reveals that open-source LLMs perform poorly on syntax-intensive tasks and multi-script languages. We further demonstrate how MiLiC-Eval can help advance LRL research in handling diverse writing systems and understanding the process of language adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15268v4">Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 8 pages of content
    </div>
    <details class="paper-abstract">
      The rapid growth of social media platforms has raised significant concerns regarding online content toxicity. When Large Language Models (LLMs) are used for toxicity detection, two key challenges emerge: 1) the absence of domain-specific toxic knowledge leads to false negatives; 2) the excessive sensitivity of LLMs to toxic speech results in false positives, limiting freedom of speech. To address these issues, we propose a novel method called MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance hatred and toxicity detection. First, we construct a comprehensive meta-toxic knowledge graph by utilizing LLMs to extract toxic information through a three-step pipeline, with toxic benchmark datasets serving as corpora. Second, we query the graph via retrieval and ranking processes to supplement accurate, relevant toxic knowledge. Extensive experiments and in-depth case studies across multiple datasets demonstrate that our MetaTox significantly decreases the false positive rate while boosting overall toxicity detection performance. Our code is available at https://github.com/YiboZhao624/MetaTox.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18403v3">LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to the main conference of ACL 2025
    </div>
    <details class="paper-abstract">
      There is an increasing trend towards evaluating NLP models with LLMs instead of human judgments, raising questions about the validity of these evaluations, as well as their reproducibility in the case of proprietary models. We provide JUDGE-BENCH, an extensible collection of 20 NLP datasets with human annotations covering a broad range of evaluated properties and types of data, and comprehensively evaluate 11 current LLMs, covering both open-weight and proprietary models, for their ability to replicate the annotations. Our evaluations show substantial variance across models and datasets. Models are reliable evaluators on some tasks, but overall display substantial variability depending on the property being evaluated, the expertise level of the human judges, and whether the language is human or model-generated. We conclude that LLMs should be carefully validated against human judgments before being used as evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14050v4">Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to the Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. We reduce biases by finetuning on curated non-harmful text, but find only direct preference optimization to be effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07217v3">ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Project page: https://vincent2311.github.io/ReelWave_demo
    </div>
    <details class="paper-abstract">
      Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24362v2">Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We investigate whether the success of a zero-shot Chain-of-Thought (CoT) process can be predicted before completion. We discover that a probing classifier, based on LLM representations, performs well \emph{even before a single token is generated}, suggesting that crucial information about the reasoning process is already present in the initial steps representations. In contrast, a strong BERT-based baseline, which relies solely on the generated tokens, performs worse, likely because it depends on shallow linguistic cues rather than deeper reasoning dynamics. Surprisingly, using later reasoning steps does not always improve classification. When additional context is unhelpful, earlier representations resemble later ones more, suggesting LLMs encode key information early. This implies reasoning can often stop early without loss. To test this, we conduct early stopping experiments, showing that truncating CoT reasoning still improves performance over not using CoT at all, though a gap remains compared to full reasoning. However, approaches like supervised learning or reinforcement learning designed to shorten CoT chains could leverage our classifier's guidance to identify when early stopping is effective. Our findings provide insights that may support such methods, helping to optimize CoT's efficiency while preserving its benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15865v2">Standard Benchmarks Fail -- Auditing LLM Agents in Finance Must Prioritize Risk</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 46 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Standard benchmarks fixate on how well large language model (LLM) agents perform in finance, yet say little about whether they are safe to deploy. We argue that accuracy metrics and return-based scores provide an illusion of reliability, overlooking vulnerabilities such as hallucinated facts, stale data, and adversarial prompt manipulation. We take a firm position: financial LLM agents should be evaluated first and foremost on their risk profile, not on their point-estimate performance. Drawing on risk-engineering principles, we outline a three-level agenda: model, workflow, and system, for stress-testing LLM agents under realistic failure modes. To illustrate why this shift is urgent, we audit six API-based and open-weights LLM agents on three high-impact tasks and uncover hidden weaknesses that conventional benchmarks miss. We conclude with actionable recommendations for researchers, practitioners, and regulators: audit risk-aware metrics in future studies, publish stress scenarios alongside datasets, and treat ``safety budget'' as a primary success criterion. Only by redefining what ``good'' looks like can the community responsibly advance AI-driven finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16310v4">An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 21 pages, 5 images, 8 tables, Manuscript submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Security code review is a time-consuming and labor-intensive process typically requiring integration with automated security defect detection tools. However, existing security analysis tools struggle with poor generalization, high false positive rates, and coarse detection granularity. Large Language Models (LLMs) have been considered promising candidates for addressing those challenges. In this study, we conducted an empirical study to explore the potential of LLMs in detecting security defects during code review. Specifically, we evaluated the performance of six LLMs under five different prompts and compared them with state-of-the-art static analysis tools. We also performed linguistic and regression analyses for the best-performing LLM to identify quality problems in its responses and factors influencing its performance. Our findings showthat: (1) existing pre-trained LLMs have limited capability in security code review but significantly outperformthe state-of-the-art static analysis tools. (2) GPT-4 performs best among all LLMs when provided with a CWE list for reference. (3) GPT-4 frequently generates verbose or non-compliant responses with the task requirements given in the prompts. (4) GPT-4 is more adept at identifying security defects in code files with fewer tokens, containing functional logic, or written by developers with less involvement in the project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13899v2">Schemato -- An LLM for Netlist-to-Schematic Conversion</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Machine learning models are advancing circuit design, particularly in analog circuits. They typically generate netlists that lack human interpretability. This is a problem as human designers heavily rely on the interpretability of circuit diagrams or schematics to intuitively understand, troubleshoot, and develop designs. Hence, to integrate domain knowledge effectively, it is crucial to translate ML-generated netlists into interpretable schematics quickly and accurately. We propose Schemato, a large language model (LLM) for netlist-to-schematic conversion. In particular, we consider our approach in converting netlists to .asc files, text-based schematic description used in LTSpice. Experiments on our circuit dataset show that Schemato achieves up to 76% compilation success rate, surpassing 63% scored by the state-of-the-art LLMs. Furthermore, our experiments show that Schemato generates schematics with an average graph edit distance score and mean structural similarity index measure, scaled by the compilation success rate that are 1.8x and 4.3x higher than the best performing LLMs respectively, demonstrating its ability to generate schematics that are more accurately connected and are closer to the reference human design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08219v2">Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks, leading to widespread adoption in both research and industry. However, their inference workloads are computationally and energy intensive, raising concerns about sustainability and environmental impact. As LLMs continue to scale, it becomes essential to identify and optimize the factors that influence their runtime efficiency without compromising performance. In this work, we systematically investigate the energy-performance trade-offs of LLMs during inference. We benchmark models of varying sizes and architectures, including Falcon-7B, Mistral-7B-v0.1, LLaMA-3.2-1B, LLaMA-3.2-3B, and GPT-Neo-2.7B, across tasks such as question answering, commonsense reasoning, and factual generation. We analyze the effect of input characteristics, such as sequence length, entropy, named entity density and so on. Furthermore, we examine the impact of hardware-level optimizations through Dynamic Voltage and Frequency Scaling (DVFS), measuring how different GPU clock settings affect latency and power consumption. Our empirical findings show that model architecture, input complexity, and clock configuration significantly influence inference efficiency. By correlating input features with energy metrics and evaluating DVFS behavior, we identify practical strategies that reduce energy consumption by up to 30% while preserving model quality. This study provides actionable insights for designing energy-efficient and sustainable LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v3">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00113v3">Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted for ACL 2025 Main Conference and NeurIPS 2024 FM-EduAssess Workshop
    </div>
    <details class="paper-abstract">
      This work introduces a novel framework for evaluating LLMs' capacity to balance instruction-following with critical reasoning when presented with multiple-choice questions containing no valid answers. Through systematic evaluation across arithmetic, domain-specific knowledge, and high-stakes medical decision tasks, we demonstrate that post-training aligned models often default to selecting invalid options, while base models exhibit improved refusal capabilities that scale with model size. Our analysis reveals that alignment techniques, though intended to enhance helpfulness, can inadvertently impair models' reflective judgment--the ability to override default behaviors when faced with invalid options. We additionally conduct a parallel human study showing similar instruction-following biases, with implications for how these biases may propagate through human feedback datasets used in alignment. We provide extensive ablation studies examining the impact of model size, training techniques, and prompt engineering. Our findings highlight fundamental tensions between alignment optimization and preservation of critical reasoning capabilities, with important implications for developing more robust AI systems for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09923v2">Guiding Reasoning in Small Language Models with LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The limited reasoning capabilities of small language models (SLMs) cast doubt on their suitability for tasks demanding deep, multi-step logical deduction. This paper introduces a framework called Small Reasons, Large Hints (SMART), which selectively augments SLM reasoning with targeted guidance from large language models (LLMs). Inspired by the concept of cognitive scaffolding, SMART employs a score-based evaluation to identify uncertain reasoning steps and injects corrective LLM-generated reasoning only when necessary. By framing structured reasoning as an optimal policy search, our approach steers the reasoning trajectory toward correct solutions without exhaustive sampling. Our experiments on mathematical reasoning datasets demonstrate that targeted external scaffolding significantly improves performance, paving the way for collaborative use of both SLM and LLM to tackle complex reasoning tasks that are currently unsolvable by SLMs alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10201v2">Prediction hubs are context-informed frequent tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Published as a conference paper at ACL 2025
    </div>
    <details class="paper-abstract">
      Hubness, the tendency for a few points to be among the nearest neighbours of a disproportionate number of other points, commonly arises when applying standard distance measures to high-dimensional data, often negatively impacting distance-based analysis. As autoregressive large language models (LLMs) operate on high-dimensional representations, we ask whether they are also affected by hubness. We first prove that the only large-scale representation comparison operation performed by LLMs, namely that between context and unembedding vectors to determine continuation probabilities, is not characterized by the concentration of distances phenomenon that typically causes the appearance of nuisance hubness. We then empirically show that this comparison still leads to a high degree of hubness, but the hubs in this case do not constitute a disturbance. They are rather the result of context-modulated frequent tokens often appearing in the pool of likely candidates for next token prediction. However, when other distances are used to compare LLM representations, we do not have the same theoretical guarantees, and, indeed, we see nuisance hubs appear. There are two main takeaways. First, hubness, while omnipresent in high-dimensional spaces, is not a negative property that needs to be mitigated when LLMs are being used for next token prediction. Second, when comparing representations from LLMs using Euclidean or cosine distance, there is a high risk of nuisance hubs and practitioners should use mitigation techniques if relevant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02508v2">Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00212v3">Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 camera-ready
    </div>
    <details class="paper-abstract">
      Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at https://github.com/mingyin1/Agents_Failure_Attribution
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18970v2">Learning to Explain: Prototype-Based Surrogate Models for LLM Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on natural language tasks, but their decision-making processes remain largely opaque. Existing explanation methods either suffer from limited faithfulness to the model's reasoning or produce explanations that humans find difficult to understand. To address these challenges, we propose \textbf{ProtoSurE}, a novel prototype-based surrogate framework that provides faithful and human-understandable explanations for LLMs. ProtoSurE trains an interpretable-by-design surrogate model that aligns with the target LLM while utilizing sentence-level prototypes as human-understandable concepts. Extensive experiments show that ProtoSurE consistently outperforms SOTA explanation methods across diverse LLMs and datasets. Importantly, ProtoSurE demonstrates strong data efficiency, requiring relatively few training examples to achieve good performance, making it practical for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11693v3">BridG MT: Enhancing LLMs' Machine Translation Capabilities with Sentence Bridging and Gradual MT</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Recent Large Language Models (LLMs) have demonstrated impressive translation performance without requiring fine-tuning on additional parallel corpora. However, they still face significant challenges in certain scenarios, particularly when translating low-resource languages. A common approach to address this issue is to provide external knowledge, such as few-shot examples, to assist LLMs in translating specific source sentences. However, this method is fundamentally limited by the quality or quantity of relevant sources, which cannot always be guaranteed. To reduce LLMs' reliance on external sources, we propose BridG MT, a method that combines Sentence Bridging, which generates a sequence of sentences as a bridge that gradually transition from easy-to-translate to more difficult, and Gradual MT, which sequentially translates these sentences using earlier translations as few-shot examples for subsequent ones. Experiments conducted on four LLMs across seven languages demonstrate that our method effectively enhances translation performance, even outperforming translation methods that rely on a large number of few-shot examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14847v2">Red-Teaming LLM Multi-Agent Systems via Communication Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24034v2">LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become the most effective post-training approach for improving the capabilities of Large Language Models (LLMs). In practice, because of the high demands on latency and memory, it is particularly challenging to develop an efficient RL framework that reliably manages policy models with hundreds to thousands of billions of parameters. In this paper, we present LlamaRL, a fully distributed, asynchronous RL framework optimized for efficient training of large-scale LLMs with various model sizes (8B, 70B, and 405B parameters) on GPU clusters ranging from a handful to thousands of devices. LlamaRL introduces a streamlined, single-controller architecture built entirely on native PyTorch, enabling modularity, ease of use, and seamless scalability to thousands of GPUs. We also provide a theoretical analysis of LlamaRL's efficiency, including a formal proof that its asynchronous design leads to strict RL speed-up. Empirically during the Llama 3 post-training, by leveraging best practices such as colocated model offloading, asynchronous off-policy training, and distributed direct memory access for weight synchronization, LlamaRL achieves significant efficiency gains -- up to 10.7x speed-up compared to DeepSpeed-Chat-like systems on a 405B-parameter policy model. Furthermore, the efficiency advantage continues to grow with increasing model scale, demonstrating the framework's suitability for future large-scale RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08291v4">CleanAgent: Automating Data Standardization with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Data standardization is a crucial part of the data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing different column types, simplifying the LLM's code generation with concise API calls. We first propose Dataprep.Clean, a component of the Dataprep Python Library, significantly reduces the coding complexity by enabling the standardization of specific column types with a single line of code. Then, we introduce the CleanAgent framework integrating Dataprep.Clean and LLM-based agents to automate the data standardization process. With CleanAgent, data scientists only need to provide their requirements once, allowing for a hands-free process. To demonstrate the practical utility of CleanAgent, we developed a user-friendly web application, allowing users to interact with it using real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18547v5">Token-Budget-Aware LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning and enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework that dynamically adjusts the number of reasoning tokens based on the reasoning complexity of each problem. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.24840v1">Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 28 pages, 13 figures
    </div>
    <details class="paper-abstract">
      This paper reveals that many state-of-the-art large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual understanding (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect to some extent because the VQA tasks improve the LLM's hierarchical consistency more than the vision LLM's. We conjecture that one cannot make vision LLMs understand visual concepts fully hierarchical until LLMs possess corresponding taxonomy knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14127v2">Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing, where LLMs construct and explain answers, better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA, robustness, biases, and unfaithful explanations, showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01195v1">CoBRA: Quantifying Strategic Language Use and LLM Pragmatics</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Language is often used strategically, particularly in high-stakes, adversarial settings, yet most work on pragmatics and LLMs centers on cooperativity. This leaves a gap in systematic understanding of non-cooperative discourse. To address this, we introduce CoBRA (Cooperation-Breach Response Assessment), along with three interpretable metrics -- Benefit at Turn (BaT), Penalty at Turn (PaT), and Normalized Relative Benefit at Turn (NRBaT) -- to quantify the perceived strategic effects of discourse moves. We also present CHARM, an annotated dataset of real courtroom cross-examinations, to demonstrate the framework's effectiveness. Using these tools, we evaluate a range of LLMs and show that LLMs generally exhibit limited pragmatic understanding of strategic language. While model size shows an increase in performance on our metrics, reasoning ability does not help and largely hurts, introducing overcomplication and internal confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01190v1">Culturally-Grounded Chain-of-Thought (CG-CoT):Enhancing LLM Performance on Culturally-Specific Tasks in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with culturally-specific reasoning tasks, particularly in low-resource languages, hindering their global applicability. Addressing this gap is crucial for equitable AI deployment. We introduce Culturally-Grounded Chain-of-Thought (CG-CoT), a novel prompting strategy that combines dense vector retrieval of cultural context with explicit reasoning sequences. Our extensive experiments on Yoruba proverb interpretation demonstrate that CG-CoT provides significantly higher culturally-aligned accuracy and depth than traditional prompting methods, validated through both automated metrics and LLM-based evaluations. Notably, we uncover stark disparities between token-level translation metrics like BLEU and human-judged cultural relevance, suggesting a rethinking of evaluation approaches for low-resource NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01140v1">ReTern: Exploiting Natural Redundancy and Sign Transformations for Enhanced Fault Tolerance in Compute-in-Memory based Ternary LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Ternary large language models (LLMs), which utilize ternary precision weights and 8-bit activations, have demonstrated competitive performance while significantly reducing the high computational and memory requirements of full-precision LLMs. The energy efficiency and performance of Ternary LLMs can be further improved by deploying them on ternary computing-in-memory (TCiM) accelerators, thereby alleviating the von-Neumann bottleneck. However, TCiM accelerators are prone to memory stuck-at faults (SAFs) leading to degradation in the model accuracy. This is particularly severe for LLMs due to their low weight sparsity. To boost the SAF tolerance of TCiM accelerators, we propose ReTern that is based on (i) fault-aware sign transformations (FAST) and (ii) TCiM bit-cell reprogramming exploiting their natural redundancy. The key idea is to utilize FAST to minimize computations errors due to SAFs in +1/-1 weights, while the natural bit-cell redundancy is exploited to target SAFs in 0 weights (zero-fix). Our experiments on BitNet b1.58 700M and 3B ternary LLMs show that our technique furnishes significant fault tolerance, notably 35% reduction in perplexity on the Wikitext dataset in the presence of faults. These benefits come at the cost of < 3%, < 7%, and < 1% energy, latency and area overheads respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01116v1">ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01114v1">Reconsidering LLM Uncertainty Estimation Methods in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: https://github.com/duygunuryldz/uncertainty_in_the_wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01104v1">Contextual Candor: Enhancing LLM Trustworthiness Through Hierarchical Unanswerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The pervasive deployment of large language models (LLMs) in conversational AI systems has revolutionized information access, yet their propensity for generating factually unsupported or hallucinated responses remains a critical impediment to trustworthiness and widespread adoption. This paper introduces Reinforced Unanswerability Learning (RUL), a novel hybrid training paradigm designed to imbue LLMs with the intrinsic capability to accurately detect unanswerable questions and generate reliably appropriate responses. Unlike conventional approaches that rely on external classifiers or simple prompting, RUL integrates a discriminative unanswerability prediction head with the LLM's generative core, guided by a multi-stage learning strategy. This includes supervised fine-tuning on a novel, richly annotated dataset, Enhanced-CAsT-Answerability (ECA), which features hierarchical answerability labels and ground-truth refusal responses. Crucially, RUL incorporates a subsequent reinforcement learning with human feedback (RLHF) phase to refine the nuance, helpfulness, and informativeness of refusal responses. Extensive experiments demonstrate RUL's superior performance, achieving significantly higher accuracy in unanswerability detection across sentence, paragraph, and ranking levels, and substantially increasing the generation of appropriate refusals for unanswerable queries, alongside strong performance on answerable questions. Human evaluations further corroborate RUL's effectiveness, highlighting a marked improvement in perceived helpfulness and trustworthiness, ultimately paving the way for more reliable and user-centric conversational AI.
    </details>
</div>
