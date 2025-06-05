# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24539v2">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01926v2">Unnatural Languages Are Not Bugs but Features for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been observed to process non-human-readable text sequences, such as jailbreak prompts, often viewed as a bug for aligned LLMs. In this work, we present a systematic investigation challenging this perception, demonstrating that unnatural languages - strings that appear incomprehensible to humans but maintain semantic meanings for LLMs - contain latent features usable by models. Notably, unnatural languages possess latent features that can be generalized across different models and tasks during inference. Furthermore, models fine-tuned on unnatural versions of instruction datasets perform on-par with those trained on natural language, achieving 49.71 win rates in Length-controlled AlpacaEval 2.0 in average across various base models. In addition, through comprehensive analysis, we demonstrate that LLMs process unnatural languages by filtering noise and inferring contextual meaning from filtered words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02592v1">Beyond the Surface: Measuring Self-Preference in LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent studies show that large language models (LLMs) exhibit self-preference bias when serving as judges, meaning they tend to favor their own responses over those generated by other models. Existing methods typically measure this bias by calculating the difference between the scores a judge model assigns to its own responses and those it assigns to responses from other models. However, this approach conflates self-preference bias with response quality, as higher-quality responses from the judge model may also lead to positive score differences, even in the absence of bias. To address this issue, we introduce gold judgments as proxies for the actual quality of responses and propose the DBG score, which measures self-preference bias as the difference between the scores assigned by the judge model to its own responses and the corresponding gold judgments. Since gold judgments reflect true response quality, the DBG score mitigates the confounding effect of response quality on bias measurement. Using the DBG score, we conduct comprehensive experiments to assess self-preference bias across LLMs of varying versions, sizes, and reasoning abilities. Additionally, we investigate two factors that influence and help alleviate self-preference bias: response text style and the post-training data of judge models. Finally, we explore potential underlying mechanisms of self-preference bias from an attention-based perspective. Our code and data are available at https://github.com/zhiyuanc2001/self-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02591v1">On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to ACL 2025 Main (Camera-Ready Version)
    </div>
    <details class="paper-abstract">
      Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: (RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02589v1">Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17626v3">Tracking the Feature Dynamics in LLM Training: A Mechanistic Study</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we (1) introduce SAE-Track, a novel method for efficiently obtaining a continual series of SAEs, providing the foundation for a mechanistic study that covers (2) the semantic evolution of features, (3) the underlying processes of feature formation, and (4) the directional drift of feature vectors. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution. For reproducibility, our code is available at https://github.com/Superposition09m/SAE-Track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02573v1">IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Although region-specific large language models (LLMs) are increasingly developed, their safety remains underexplored, particularly in culturally diverse settings like Indonesia, where sensitivity to local norms is essential and highly valued by the community. In this work, we present IndoSafety, the first high-quality, human-verified safety evaluation dataset tailored for the Indonesian context, covering five language varieties: formal and colloquial Indonesian, along with three major local languages: Javanese, Sundanese, and Minangkabau. IndoSafety is constructed by extending prior safety frameworks to develop a taxonomy that captures Indonesia's sociocultural context. We find that existing Indonesian-centric LLMs often generate unsafe outputs, particularly in colloquial and local language settings, while fine-tuning on IndoSafety significantly improves safety while preserving task performance. Our work highlights the critical need for culturally grounded safety evaluation and provides a concrete step toward responsible LLM deployment in multilingual settings. Warning: This paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01538v2">LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at https://windylab.github.io/LAMARL/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04291v2">MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Published in AAAI 2025
    </div>
    <details class="paper-abstract">
      We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06705v3">LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02553v1">Response-Level Rewards Are All You Need for Online Reinforcement Learning in LLMs: A Mathematical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We study a common challenge in reinforcement learning for large language models (LLMs): the Zero-Reward Assumption, where non-terminal actions (i.e., intermediate token generations) receive zero task-specific immediate reward, while only the final token receives a reward for the entire response. This assumption arises frequently in practice, as precise token-level rewards are often difficult or infeasible to obtain in LLM applications. In this work, we provide a unifying theoretical perspective. We introduce the Trajectory Policy Gradient Theorem, which shows that the policy gradient based on true, unknown token-level rewards can be unbiasedly estimated using only a response-level reward model, regardless of whether the Zero-Reward Assumption holds or not, for algorithms in the REINFORCE and Actor-Critic families. This result reveals that widely used methods such as PPO, GRPO, ReMax, and RLOO inherently possess the capacity to model token-level reward signals, offering a theoretical justification for response-level reward approaches. Our findings pave the way for more practical, efficient LLM fine-tuning, allowing developers to treat training algorithms as black boxes and focus on improving the response-level reward model with auxiliary sub-models. We also offer a detailed analysis of popular RL and non-RL methods, comparing their theoretical foundations and practical advantages across common LLM tasks. Finally, we propose a new algorithm: Token-Reinforced Policy Optimization (TRePO), a theoretically grounded method that is simpler than PPO, matches GRPO in memory efficiency, and holds promise for broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23809v2">LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02546v1">Attention Knows Whom to Trust: Attention-based Trust Management for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have demonstrated strong capabilities in solving complex tasks but remain vulnerable when agents receive unreliable messages. This vulnerability stems from a fundamental gap: LLM agents treat all incoming messages equally without evaluating their trustworthiness. While some existing studies approach the trustworthiness, they focus on a single type of harmfulness rather than analyze it in a holistic approach from multiple trustworthiness perspectives. In this work, we propose Attention Trust Score (A-Trust), a lightweight, attention-based method for evaluating message trustworthiness. Inspired by human communication literature[1], through systematically analyzing attention behaviors across six orthogonal trust dimensions, we find that certain attention heads in the LLM specialize in detecting specific types of violations. Leveraging these insights, A-Trust directly infers trustworthiness from internal attention patterns without requiring external prompts or verifiers. Building upon A-Trust, we develop a principled and efficient trust management system (TMS) for LLM-MAS, enabling both message-level and agent-level trust assessment. Experiments across diverse multi-agent settings and tasks demonstrate that applying our TMS significantly enhances robustness against malicious inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18499v2">G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated remarkable progress, their proficiency in graph-related tasks remains notably limited, hindering the development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face challenges such as the scarcity of large-scale, universally represented graph data. We introduce G1, a simple yet effective approach demonstrating that Reinforcement Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs' graph reasoning abilities. To enable RL training, we curate Erd\~os, the largest graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of varying difficulty levels, 100k training data and 5k test data, all drived from real-world graphs. With RL on Erd\~os, G1 obtains substantial improvements in graph reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct (24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic benchmarks as well as real-world node classification and link prediction tasks, without compromising general reasoning abilities. Our findings offer an efficient, scalable path for building strong graph reasoners by finetuning LLMs with RL on graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs possess graph understanding abilities that RL can elicit successfully. Our implementation is open-sourced at https://github.com/PKU-ML/G1, with models and datasets hosted on Hugging Face collections https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2 for broader accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02522v1">Think Twice, Act Once: A Co-Evolution Framework of LLM and RL for Large-Scale Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and Reinforcement Learning (RL) have shown significant promise in decision-making tasks. Nevertheless, for large-scale industrial decision problems, both approaches face distinct challenges: LLMs lack real-time long-sequence decision-making capabilities, while RL struggles with sample efficiency in vast action spaces. To bridge this gap, we propose Agents Co-Evolution (ACE), a synergistic framework between LLMs and RL agents for large-scale decision-making scenarios. ACE introduces a dual-role trajectory refinement mechanism where LLMs act as both Policy Actor and Value Critic during RL's training: the Actor refines suboptimal actions via multi-step reasoning and environment validation, while the Critic performs temporal credit assignment through trajectory-level reward shaping. Concurrently, RL agent enhances LLMs' task-specific decision-making with high-quality fine-tuning datasets generated via prioritized experience replay. Through extensive experiments across multiple power grid operation challenges with action spaces exceeding 60K discrete actions, ACE demonstrates superior performance over existing RL methods and LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02519v1">Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted at ACL Main 2025
    </div>
    <details class="paper-abstract">
      LLMssuch as GPT-4 have shown a remarkable ability to solve complex questions by generating step-by-step rationales. Prior works have utilized this capability to improve smaller and cheaper LMs (say, with 7B parameters). However, various practical constraints, such as copyright and legal issues, owing to lack of transparency in the pre-training data of large (often closed) models, prevent their use in commercial settings. Little focus has been given to improving the innate reasoning ability of smaller models without distilling information from larger LLMs. To address this, we propose COLLATE, a trainable framework that tunes a (small) LLM to generate those outputs from a pool of diverse rationales that selectively improves the downstream task. COLLATE enforces multiple instances of the same LLM to exhibit distinct behavior and employs them to generate rationales to obtain diverse outputs. The LLM is then tuned via preference optimization to choose the candidate rationale which maximizes the likelihood of ground-truth answer. COLLATE outperforms several trainable and prompting baselines on 5 datasets across 3 domains: maths problem solving, natural language inference, and commonsense reasoning. We show the eff icacy of COLLATE on LLMs from different model families across varying parameter scales (1B to 8B) and demonstrate the benefit of multiple rationale providers guided by the end task through ablations. Code is released here (https://github.com/Sohanpatnaik106/collate).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02514v1">To Embody or Not: The Effect Of Embodiment On User Perception Of LLM-based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Embodiment in conversational agents (CAs) refers to the physical or visual representation of these agents, which can significantly influence user perception and interaction. Limited work has been done examining the effect of embodiment on the perception of CAs utilizing modern large language models (LLMs) in non-hierarchical cooperative tasks, a common use case of CAs as more powerful models become widely available for general use. To bridge this research gap, we conducted a mixed-methods within-subjects study on how users perceive LLM-based CAs in cooperative tasks when embodied and non-embodied. The results show that the non-embodied agent received significantly better quantitative appraisals for competence than the embodied agent, and in qualitative feedback, many participants believed that the embodied CA was more sycophantic than the non-embodied CA. Building on prior work on users' perceptions of LLM sycophancy and anthropomorphic features, we theorize that the typically-positive impact of embodiment on perception of CA credibility can become detrimental in the presence of sycophancy. The implication of such a phenomenon is that, contrary to intuition and existing literature, embodiment is not a straightforward way to improve a CA's perceived credibility if there exists a tendency to sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00486v2">It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: https://huggingface.co/spaces/shifeng3711/gg_prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02490v1">Simplifying Root Cause Analysis in Kubernetes with StateGraph and LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 12 pages, 13 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Kubernetes, a notably complex and distributed system, utilizes an array of controllers to uphold cluster management logic through state reconciliation. Nevertheless, maintaining state consistency presents significant challenges due to unexpected failures, network disruptions, and asynchronous issues, especially within dynamic cloud environments. These challenges result in operational disruptions and economic losses, underscoring the necessity for robust root cause analysis (RCA) to enhance Kubernetes reliability. The development of large language models (LLMs) presents a promising direction for RCA. However, existing methodologies encounter several obstacles, including the diverse and evolving nature of Kubernetes incidents, the intricate context of incidents, and the polymorphic nature of these incidents. In this paper, we introduce SynergyRCA, an innovative tool that leverages LLMs with retrieval augmentation from graph databases and enhancement with expert prompts. SynergyRCA constructs a StateGraph to capture spatial and temporal relationships and utilizes a MetaGraph to outline entity connections. Upon the occurrence of an incident, an LLM predicts the most pertinent resource, and SynergyRCA queries the MetaGraph and StateGraph to deliver context-specific insights for RCA. We evaluate SynergyRCA using datasets from two production Kubernetes clusters, highlighting its capacity to identify numerous root causes, including novel ones, with high efficiency and precision. SynergyRCA demonstrates the ability to identify root causes in an average time of about two minutes and achieves an impressive precision of approximately 0.90.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19390v2">Checkpoint Merging via Bayesian Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      The rapid proliferation of large language models (LLMs) such as GPT-4 and Gemini underscores the intense demand for resources during their training processes, posing significant challenges due to substantial computational and environmental costs. To alleviate this issue, we propose checkpoint merging in pretraining LLM. This method utilizes LLM checkpoints with shared training trajectories, and is rooted in an extensive search space exploration for the best merging weight via Bayesian optimization. Through various experiments, we demonstrate that: (1) Our proposed methodology exhibits the capacity to augment pretraining, presenting an opportunity akin to obtaining substantial benefits at minimal cost; (2) Our proposed methodology, despite requiring a given held-out dataset, still demonstrates robust generalization capabilities across diverse domains, a pivotal aspect in pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v4">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v7">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Tauric Research @ https://github.com/TauricResearch; Oral @ Multi-Agent AI in the Real World
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, the multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/TauricResearch/TradingAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02470v1">A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Misdiagnosis causes significant harm to healthcare systems worldwide, leading to increased costs and patient risks. MedRAG is a smart multimodal healthcare copilot equipped with powerful large language model (LLM) reasoning, designed to enhance medical decision-making. It supports multiple input modalities, including non-intrusive voice monitoring, general medical queries, and electronic health records. MedRAG provides recommendations on diagnosis, treatment, medication, and follow-up questioning. Leveraging retrieval-augmented generation enhanced by knowledge graph-elicited reasoning, MedRAG retrieves and integrates critical diagnostic insights, reducing the risk of misdiagnosis. It has been evaluated on both public and private datasets, outperforming existing models and offering more specific and accurate healthcare assistance. A demonstration video of MedRAG is available at: https://www.youtube.com/watch?v=PNIBDMYRfDM. The source code is available at: https://github.com/SNOWTEAM2023/MedRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13508v2">Time-R1: Towards Comprehensive Temporal Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02457v1">SOVA-Bench: Benchmarking the Speech Conversation Ability for LLM-based Voice Assistant</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Thanks to the steady progress of large language models (LLMs), speech encoding algorithms and vocoder structure, recent advancements have enabled generating speech response directly from a user instruction. However, benchmarking the generated speech quality has been a neglected but critical issue, considering the shift from the pursuit of semantic accuracy to vivid and spontaneous speech flow. Previous evaluation focused on the speech-understanding ability, lacking a quantification of acoustic quality. In this paper, we propose Speech cOnversational Voice Assistant Benchmark (SOVA-Bench), providing a comprehension comparison of the general knowledge, speech recognition and understanding, along with both semantic and acoustic generative ability between available speech LLMs. To the best of our knowledge, SOVA-Bench is one of the most systematic evaluation frameworks for speech LLMs, inspiring the direction of voice interaction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02442v1">Should LLM Safety Be More Than Refusing Harmful Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00418v3">Self-Evolved Reward Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 23 pages,6 figures,Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. These methods can be costly and may introduce biases that affect the language model's responses. As language models improve, human input may become less effective in further enhancing their performance. In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs). Resources of this paper can be found at https://aka.ms/ser
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17214v2">CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on Llama Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: https://github.com/ZBox1005/CoT-UQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v3">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00694v2">Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 11 pages, 7th Workshop on Automated Semantic Analysis of Information in Legal Text @ ICAIL 2025, 16 June 2025, Chicago, IL
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Link: https://lizhang-aiandlaw.github.io/An-Automated-Pipeline-for-Evaluating-LLM-Generated-3-ply-Case-Based-Legal-Arguments/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00490v2">LLM-Driven Instance-Specific Heuristic Generation and Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Combinatorial optimization problems are widely encountered in real-world applications. Designing high-quality heuristic algorithms that efficiently approximate optimal solutions within reasonable time is a critical research challenge. In recent years, many works have explored integrating Large Language Models (LLMs) with Evolutionary Algorithms to automate heuristic algorithm design through prompt engineering. However, these approaches generally adopt a problem-specific paradigm, applying a single algorithm across all problem instances, failing to account for the heterogeneity across instances. In this paper, we propose InstSpecHH, a novel framework that introduces the concept of instance-specific heuristic generation. InstSpecHH partitions the overall problem class into sub-classes based on instance features and performs differentiated, automated heuristic design for each problem subclass. By tailoring heuristics to the unique features of different sub-classes, InstSpecHH achieves better performance at the problem class level while avoiding redundant heuristic generation for similar instances, thus reducing computational overhead. This approach effectively balances the trade-off between the cost of automatic heuristic design and the quality of the obtained solutions. To evaluate the performance of InstSpecHH, we conduct experiments on 4,500 subclasses of the Online Bin Packing Problem (OBPP) and 365 subclasses of the Capacitated Vehicle Routing Problem (CVRP). Experimental results show that InstSpecHH demonstrates strong intra-subclass and inter-subclass generalization capabilities. Compared to previous problem-specific methods, InstSpecHH reduces the average optimality gap by more than 5.6\% for OBPP and 0.9\% for CVRP. These results highlight the potential of instance-aware automatic heuristic design to further enhance solution quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02389v1">Univariate to Multivariate: LLMs as Zero-Shot Predictors for Time-Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Time-series prediction or forecasting is critical across many real-world dynamic systems, and recent studies have proposed using Large Language Models (LLMs) for this task due to their strong generalization capabilities and ability to perform well without extensive pre-training. However, their effectiveness in handling complex, noisy, and multivariate time-series data remains underexplored. To address this, we propose LLMPred which enhances LLM-based time-series prediction by converting time-series sequences into text and feeding them to LLMs for zero shot prediction along with two main data pre-processing techniques. First, we apply time-series sequence decomposition to facilitate accurate prediction on complex and noisy univariate sequences. Second, we extend this univariate prediction capability to multivariate data using a lightweight prompt-processing strategy. Extensive experiments with smaller LLMs such as Llama 2 7B, Llama 3.2 3B, GPT-4o-mini, and DeepSeek 7B demonstrate that LLMPred achieves competitive or superior performance compared to state-of-the-art baselines. Additionally, a thorough ablation study highlights the importance of the key components proposed in LLMPred.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00095v2">ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02372v1">AnswerCarefully: A Dataset for Improving the Safety of Japanese LLM Output</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      In this paper we present AnswerCarefully, a dataset for promoting the safety and appropriateness of Japanese LLM outputs. The dataset consists of 1,800 pairs of questions and reference answers, where the questions require special attention in answering. It covers a wide range of risk categories established in prior English-language datasets, but the data samples are original in that they are manually created to reflect the socio-cultural context of LLM usage in Japan. We show that using this dataset for instruction to fine-tune a Japanese LLM led to improved output safety without compromising the utility of general responses. We also report the results of a safety evaluation of 12 Japanese LLMs using this dataset as a benchmark. Finally, we describe the latest update on the dataset which provides English translations and annotations of the questions, aimed at facilitating the derivation of similar datasets in different languages and regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v4">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02368v1">NextQuill: Causal Preference Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) for individual users has become increasingly important as they are progressively integrated into real-world applications to support users' daily lives. However, existing personalization approaches often fail to distinguish which components of model predictions and training data truly reflect user preferences, leading to superficial personalization alignment. In this paper, we introduce NextQuill, a novel LLM personalization alignment framework grounded in causal preference modeling. We approach personalization from a causal perspective, treating both model predictions and ground-truth data generation as outcomes influenced by user preferences, along with other factors. We define the true preference effect as the causal impact of user history (which reflects preferences) on each token prediction or data generation instance, estimated through causal intervention techniques. Building on this insight, NextQuill introduces two complementary alignment strategies: (1) aligning model-internal causal preference effects on predictions with those reflected in ground-truth data, rather than indiscriminately fitting predictions, and (2) focusing on fitting preference-bearing tokens identified via ground-truth data preference effects, rather than treating all tokens uniformly. By integrating these strategies, NextQuill shifts the alignment process toward learning from causal preference effects, facilitating more effective and personalized adaptation. Experiments across multiple personalization benchmarks demonstrate that NextQuill significantly improves personalization quality, offering a principled, causal foundation for LLM personalization. Our codes are available on https://github.com/juntaoyou/NextQuill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02357v1">Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint. This work has been submitted to the Technical AI Governance Workshop at ICML 2025 for review
    </div>
    <details class="paper-abstract">
      Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. Failure to prioritize such principles indicates a potential basic control failure. This paper introduces a lightweight, interpretable benchmark methodology using a simple grid world to evaluate an LLM agent's ability to uphold a predefined, high-level safety principle (e.g., "never enter hazardous zones") when faced with conflicting lower-level task instructions. We probe whether the agent reliably prioritizes the inviolable directive, testing a foundational controllability aspect of LLMs. This pilot study demonstrates the methodology's feasibility, offers preliminary insights into agent behavior under principle conflict, and discusses how such benchmarks can contribute empirical evidence for assessing controllability. We argue that evaluating adherence to hierarchical principles is a crucial early step in understanding our capacity to build governable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02351v1">DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 To appear in the First REALM (Research on Agent Language Models) workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02338v1">One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 ACL 2025 Industry
    </div>
    <details class="paper-abstract">
      With the release of R1, a publicly available large reasoning model (LRM), researchers commonly train new LRMs by training language models on R1's long chain-of-thought (CoT) inferences. While prior works show that LRMs' capabilities can be reproduced through direct distillation, the continued reliance on the existing models (e.g., R1) remains a critical limitation in advancing the field. As a first step toward independent LRM development, this paper explores the possibility of constructing a long CoT dataset with LLMs that are not trained for inference-time scaling. To this end, we present the Long CoT Collection, a dataset of 100K CoT rationales annotated using existing short CoT LLMs. We develop a pipeline that induces o1's novel reasoning strategies into short CoT LLMs, enabling them to think longer and introducing controllability over the thought budget to better manage the overthinking problem. Our extensive analyses validate that our dataset achieves quality comparable to--or slightly below--R1. Furthermore, our experiments demonstrate that training on our dataset not only strengthens general reasoning skills, but also provides a strong foundation for reinforcement learning--models initialized on our data achieve 2-3x larger gains with RLVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00250v2">PersianMedQA: Language-Centric Evaluation of LLMs in the Persian Medical Domain</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance on a wide range of NLP benchmarks, often surpassing human-level accuracy. However, their reliability in high-stakes domains such as medicine, particularly in low-resource languages, remains underexplored. In this work, we introduce PersianMedQA, a large-scale, expert-validated dataset of multiple-choice Persian medical questions, designed to evaluate LLMs across both Persian and English. We benchmark over 40 state-of-the-art models, including general-purpose, Persian fine-tuned, and medical LLMs, in zero-shot and chain-of-thought (CoT) settings. Our results show that closed-source general models (e.g., GPT-4.1) consistently outperform all other categories, achieving 83.3% accuracy in Persian and 80.7% in English, while Persian fine-tuned models such as Dorna underperform significantly (e.g., 35.9% in Persian), often struggling with both instruction-following and domain reasoning. We also analyze the impact of translation, showing that while English performance is generally higher, Persian responses are sometimes more accurate due to cultural and clinical contextual cues. Finally, we demonstrate that model size alone is insufficient for robust performance without strong domain or language adaptation. PersianMedQA provides a foundation for evaluating multilingual and culturally grounded medical reasoning in LLMs. The PersianMedQA dataset can be accessed at: https://huggingface.co/datasets/MohammadJRanjbar/PersianMedQA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03444v1">Exploiting LLMs for Automatic Hypothesis Assessment via a Logit-Based Calibrated Prior</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      As hypothesis generation becomes increasingly automated, a new bottleneck has emerged: hypothesis assessment. Modern systems can surface thousands of statistical relationships-correlations, trends, causal links-but offer little guidance on which ones are novel, non-trivial, or worthy of expert attention. In this work, we study the complementary problem to hypothesis generation: automatic hypothesis assessment. Specifically, we ask: given a large set of statistical relationships, can we automatically assess which ones are novel and worth further exploration? We focus on correlations as they are a common entry point in exploratory data analysis that often serve as the basis for forming deeper scientific or causal hypotheses. To support automatic assessment, we propose to leverage the vast knowledge encoded in LLMs' weights to derive a prior distribution over the correlation value of a variable pair. If an LLM's prior expects the correlation value observed, then such correlation is not surprising, and vice versa. We propose the Logit-based Calibrated Prior, an LLM-elicited correlation prior that transforms the model's raw output logits into a calibrated, continuous predictive distribution over correlation values. We evaluate the prior on a benchmark of 2,096 real-world variable pairs and it achieves a sign accuracy of 78.8%, a mean absolute error of 0.26, and 95% credible interval coverage of 89.2% in predicting Pearson correlation coefficient. It also outperforms a fine-tuned RoBERTa classifier in binary correlation prediction and achieves higher precision@K in hypothesis ranking. We further show that the prior generalizes to correlations not seen during LLM pretraining, reflecting context-sensitive reasoning rather than memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03424v1">DistRAG: Towards Distance-Based Spatial Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Many real world tasks where Large Language Models (LLMs) can be used require spatial reasoning, like Point of Interest (POI) recommendation and itinerary planning. However, on their own LLMs lack reliable spatial reasoning capabilities, especially about distances. To address this problem, we develop a novel approach, DistRAG, that enables an LLM to retrieve relevant spatial information not explicitly learned during training. Our method encodes the geodesic distances between cities and towns in a graph and retrieves a context subgraph relevant to the question. Using this technique, our method enables an LLM to answer distance-based reasoning questions that it otherwise cannot answer. Given the vast array of possible places an LLM could be asked about, DistRAG offers a flexible first step towards providing a rudimentary `world model' to complement the linguistic knowledge held in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02165v2">A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 EDM 2025 Long Paper
    </div>
    <details class="paper-abstract">
      Open-ended short-answer questions (SAGs) have been widely recognized as a powerful tool for providing deeper insights into learners' responses in the context of learning analytics (LA). However, SAGs often present challenges in practice due to the high grading workload and concerns about inconsistent assessments. With recent advancements in natural language processing (NLP), automatic short-answer grading (ASAG) offers a promising solution to these challenges. Despite this, current ASAG algorithms are often limited in generalizability and tend to be tailored to specific questions. In this paper, we propose a unified multi-agent ASAG framework, GradeOpt, which leverages large language models (LLMs) as graders for SAGs. More importantly, GradeOpt incorporates two additional LLM-based agents - the reflector and the refiner - into the multi-agent system. This enables GradeOpt to automatically optimize the original grading guidelines by performing self-reflection on its errors. Through experiments on a challenging ASAG task, namely the grading of pedagogical content knowledge (PCK) and content knowledge (CK) questions, GradeOpt demonstrates superior performance in grading accuracy and behavior alignment with human graders compared to representative baselines. Finally, comprehensive ablation studies confirm the effectiveness of the individual components designed in GradeOpt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03396v1">Fault Localisation and Repair for DL Systems: An Empirical Study with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 arXiv admin note: text overlap with arXiv:2301.11568
    </div>
    <details class="paper-abstract">
      Numerous Fault Localisation (FL) and repair techniques have been proposed to address faults in Deep Learning (DL) models. However, their effectiveness in practical applications remains uncertain due to the reliance on pre-defined rules. This paper presents a comprehensive evaluation of state-of-the-art FL and repair techniques, examining their advantages and limitations. Moreover, we introduce a novel approach that harnesses the power of Large Language Models (LLMs) in localising and repairing DL faults. Our evaluation, conducted on a carefully designed benchmark, reveals the strengths and weaknesses of current FL and repair techniques. We emphasise the importance of enhanced accuracy and the need for more rigorous assessment methods that employ multiple ground truth patches. Notably, LLMs exhibit remarkable performance in both FL and repair tasks. For instance, the GPT-4 model achieves 44% and 82% improvements in FL and repair tasks respectively, compared to the second-best tool, demonstrating the potential of LLMs in this domain. Our study sheds light on the current state of FL and repair techniques and suggests that LLMs could be a promising avenue for future advancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05276v2">Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 EDM 2025 Short Paper
    </div>
    <details class="paper-abstract">
      Short answer assessment is a vital component of science education, allowing evaluation of students' complex three-dimensional understanding. Large language models (LLMs) that possess human-like ability in linguistic tasks are increasingly popular in assisting human graders to reduce their workload. However, LLMs' limitations in domain knowledge restrict their understanding in task-specific requirements and hinder their ability to achieve satisfactory performance. Retrieval-augmented generation (RAG) emerges as a promising solution by enabling LLMs to access relevant domain-specific knowledge during assessment. In this work, we propose an adaptive RAG framework for automated grading that dynamically retrieves and incorporates domain-specific knowledge based on the question and student answer context. Our approach combines semantic search and curated educational sources to retrieve valuable reference materials. Experimental results in a science education dataset demonstrate that our system achieves an improvement in grading accuracy compared to baseline LLM approaches. The findings suggest that RAG-enhanced grading systems can serve as reliable support with efficient performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01679v2">VinePPO: Refining Credit Assignment in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted at ICML 2025; 12 pages and 22 pages Appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a common reinforcement learning (RL) algorithm used for LLM finetuning, employs value networks to tackle credit assignment. However, recent approaches achieve strong results without it, raising questions about the efficacy of value networks in practice. In this work, we systematically evaluate the efficacy of value networks and reveal their significant shortcomings in reasoning-heavy LLM tasks, showing that they often produce poor estimate of expected return and barely outperform a random baseline when comparing alternative steps. This motivates our key question: Can improved credit assignment enhance RL training for LLMs? To address this, we propose VinePPO, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates. Our method consistently outperforms PPO and other baselines across MATH and GSM8K datasets in less wall-clock time (up to 3.0x). Crucially, it achieves higher test accuracy for a given training accuracy, capturing more generalization signal per sample. These results emphasize the importance of accurate credit assignment in RL training of LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01622v3">DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation. Browse the data, contribute, and more: https://slab-nlp.github.io/DOVE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09300v4">Nudging: Inference-time Alignment of LLMs via Guided Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require alignment to effectively and safely follow user instructions. This process necessitates training an aligned version for every base model, resulting in significant computational overhead. In this work, we propose NUDGING, a simple, training-free algorithm that aligns any base model at inference time using a small aligned model. NUDGING is motivated by recent findings that alignment primarily alters the model's behavior on a small subset of stylistic tokens (e.g., discourse markers). We find that base models are significantly more uncertain when generating these tokens. Building on this insight, NUDGING employs a small aligned model to generate nudging tokens to guide the base model's output during decoding when the base model's uncertainty is high, with only a minor additional inference overhead. We evaluate NUDGING across 3 model families on a diverse range of open-instruction tasks. Without any training, nudging a large base model with a 7x-14x smaller aligned model achieves zero-shot performance comparable to, and sometimes surpassing, that of large aligned models. By operating at the token level, NUDGING enables off-the-shelf collaboration between model families. For instance, nudging Gemma-2-27b with Llama-27b-chat outperforms Llama-2-70b-chat on various tasks. Overall, our work offers a modular and cost-efficient solution to LLM alignment. Our code and demo are available at: https://fywalter.github.io/nudging/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19582v2">Where Are We? Evaluating LLM Performance on African Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Africa's rich linguistic heritage remains underrepresented in NLP, largely due to historical policies that favor foreign languages and create significant data inequities. In this paper, we integrate theoretical insights on Africa's language landscape with an empirical evaluation using Sahara - a comprehensive benchmark curated from large-scale, publicly accessible datasets capturing the continent's linguistic diversity. By systematically assessing the performance of leading large language models (LLMs) on Sahara, we demonstrate how policy-induced data variations directly impact model effectiveness across African languages. Our findings reveal that while a few languages perform reasonably well, many Indigenous languages remain marginalized due to sparse data. Leveraging these insights, we offer actionable recommendations for policy reforms and inclusive data practices. Overall, our work underscores the urgent need for a dual approach - combining theoretical understanding with empirical evaluation - to foster linguistic diversity in AI for African communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20573v2">Collision- and Reachability-Aware Multi-Robot Control with Grounded LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance in various robot control tasks. However, their deployment in real-world applications remains constrained. Even state-ofthe-art LLMs, such as GPT-o4mini, frequently produce invalid action plans that violate physical constraints, such as directing a robot to an unreachable location or causing collisions between robots. This issue primarily arises from a lack of awareness of these physical constraints during the reasoning process. To address this issue, we propose a novel framework that integrates reinforcement learning with verifiable rewards (RLVR) to incentivize knowledge of physical constraints into LLMs to induce constraints-aware reasoning during plan generation. In this approach, only valid action plans that successfully complete a control task receive positive rewards. We applied our method to two small-scale LLMs: a non-reasoning Qwen2.5-3B-Instruct and a reasoning Qwen3-4B. The experiment results demonstrate that constraint-aware small LLMs largely outperform large-scale models without constraints, grounded on both the BoxNet task and a newly developed BoxNet3D environment built using MuJoCo. This work highlights the effectiveness of grounding even small LLMs with physical constraints to enable scalable and efficient multi-robot control in complex, physically constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03337v1">Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 56 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Federated Learning enables collaborative fine-tuning of Large Language Models (LLMs) across decentralized Non-Independent and Identically Distributed (Non-IID) clients, but such models' massive parameter sizes lead to significant memory and communication challenges. This work introduces Meerkat, a sparse zeroth-order optimization (ZO) method designed for federated LLM fine-tuning. By limiting fine-tuning to a transferable, static, extremely sparse subset of parameters, Meerkat achieves remarkable communication efficiency, enabling cost-effective high-frequency synchronization. With theoretical analysis and experiments, we show that this high-frequency communication effectively mitigates Non-IID data challenges and leads to superior performance compared to full-parameter ZO. Furthermore, experiment results show that Meerkat outperforms existing sparsity baselines with better performance at the same communication frequency. To further handle Non-IID drift, Meerkat leverages traceable local updates and forms a virtual path for each client. This virtual path mechanism reveals the GradIP phenomenon: the inner products between LLM pre-training gradients maintained by server and client gradients estimated via ZO converges for extreme Non-IID clients but oscillates for IID ones. This distinct behavior provides a signal for identifying clients with extreme data heterogeneity. Using this signal, Meerkat-vp is proposed to analyze GradIP trajectories to identify extreme Non-IID clients and applies early stopping to enhance aggregated model quality. Experiments confirm that Meerkat and Meerkat-vp significantly improve the efficiency and effectiveness of ZO federated LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v1">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads.We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings.APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03295v1">Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We have witnessed that strong LLMs like Qwen-Math, MiMo, and Phi-4 possess immense reasoning potential inherited from the pre-training stage. With reinforcement learning (RL), these models can improve dramatically on reasoning tasks. Recent studies have shown that even RL on a single problem can unleash these models' reasoning capabilities. However, RL is not only expensive but also unstable. Even one-shot RL requires hundreds of GPU hours. This raises a critical question: Is there a more efficient way to unleash the reasoning potential of these powerful base LLMs? In this work, we demonstrate that Critique Fine-Tuning (CFT) on only one problem can effectively unleash the reasoning potential of LLMs. Our method constructs critique data by collecting diverse model-generated solutions to a single problem and using teacher LLMs to provide detailed critiques. We fine-tune Qwen and Llama family models, ranging from 1.5B to 14B parameters, on the CFT data and observe significant performance gains across diverse reasoning tasks. For example, with just 5 GPU hours of training, Qwen-Math-7B-CFT show an average improvement of 15% on six math benchmarks and 16% on three logic reasoning benchmarks. These results are comparable to or even surpass the results from RL with 20x less compute. Ablation studies reveal the robustness of one-shot CFT across different prompt problems. These results highlight one-shot CFT as a simple, general, and compute-efficient approach to unleashing the reasoning capabilities of modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03231v1">NetPress: Dynamically Generated LLM Benchmarks for Network Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Despite growing interest in domain-specific benchmarking of large language models (LLMs) and agents, current evaluations remain limited to static, small-scale datasets, especially in high-stakes tasks like network operations that demand reliability for deployments. We present NetPress, an automated benchmark generation framework for evaluating LLM agents in network applications. NetPress introduces a unified abstraction with state and action, enabling dynamic generation of diverse query sets along with corresponding ground truths. At runtime, users can specify benchmark configurations to generate millions of queries on the fly. In addition to dynamic benchmark construction, NetPress integrates with network emulators to provide realistic environment feedback, supporting comprehensive evaluation across correctness, safety, and latency. We instantiate NetPress on three representative applications, revealing interesting fine-grained differences in agent behavior that static, correctness-only benchmarks often miss. NetPress moves LLM evaluation toward realistic, scalable testing in infrastructure-centric domains, helping close the gap between benchmark performance and real-world deployment readiness. Code is available at https://github.com/Froot-NetSys/NetPress.
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
