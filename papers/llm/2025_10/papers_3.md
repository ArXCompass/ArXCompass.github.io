# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13423v2">AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09719v2">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10601v5">When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11062v2">Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07146v2">Attention-Aware GNN-based Input Defense against Multi-Turn LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained significant traction in various applications, yet their capabilities present risks for both constructive and malicious exploitation. Despite extensive training and fine-tuning efforts aimed at enhancing safety, LLMs remain susceptible to jailbreak attacks. Recently, the emergence of multi-turn attacks has intensified this vulnerability. Unlike single-turn attacks, multi-turn attacks incrementally escalate dialogue complexity, rendering them more challenging to detect and mitigate. In this study, we introduce G-Guard, an innovative attention-aware Graph Neural Network (GNN)-based input classifier specifically designed to defend against multi-turn jailbreak attacks targeting LLMs. G-Guard constructs an entity graph for multi-turn queries, which captures the interrelationships between queries and harmful keywords that present in multi-turn queries. Furthermore, we propose an attention-aware augmentation mechanism that retrieves the most relevant single-turn query based on the ongoing multi-turn conversation. The retrieved query is incorporated as a labeled node within the graph, thereby enhancing the GNN's capacity to classify the current query as harmful or benign. Evaluation results show that G-Guard consistently outperforms all baselines across diverse datasets and evaluation metrics, demonstrating its efficacy as a robust defense mechanism against multi-turn jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12120v1">Towards Engineering Multi-Agent LLMs: A Protocol-Driven Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The increasing demand for software development has driven interest in automating software engineering (SE) tasks using Large Language Models (LLMs). Recent efforts extend LLMs into multi-agent systems (MAS) that emulate collaborative development workflows, but these systems often fail due to three core deficiencies: under-specification, coordination misalignment, and inappropriate verification, arising from the absence of foundational SE structuring principles. This paper introduces Software Engineering Multi-Agent Protocol (SEMAP), a protocol-layer methodology that instantiates three core SE design principles for multi-agent LLMs: (1) explicit behavioral contract modeling, (2) structured messaging, and (3) lifecycle-guided execution with verification, and is implemented atop Google's Agent-to-Agent (A2A) infrastructure. Empirical evaluation using the Multi-Agent System Failure Taxonomy (MAST) framework demonstrates that SEMAP effectively reduces failures across different SE tasks. In code development, it achieves up to a 69.6% reduction in total failures for function-level development and 56.7% for deployment-level development. For vulnerability detection, SEMAP reduces failure counts by up to 47.4% on Python tasks and 28.2% on C/C++ tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15732v2">Can LLMs Reconcile Knowledge Conflicts in Counterfactual Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 ICML 2025 Workshop on Scaling up Intervention Models
    </div>
    <details class="paper-abstract">
      Large Language Models have been shown to contain extensive world knowledge in their parameters, enabling impressive performance on many knowledge intensive tasks. However, when deployed in novel settings, LLMs often encounter situations where they must integrate parametric knowledge with new or unfamiliar information. In this work, we explore whether LLMs can combine knowledge in-context with their parametric knowledge through the lens of counterfactual reasoning. Through synthetic and real experiments in multi-hop reasoning problems, we show that LLMs generally struggle with counterfactual reasoning, often resorting to exclusively using their parametric knowledge. Moreover, we show that simple post-hoc finetuning can struggle to instill counterfactual reasoning ability -- often leading to degradation in stored parametric knowledge. Ultimately, our work reveals important limitations of current LLM's abilities to re-purpose parametric knowledge in novel settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01656v2">Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12095v1">IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 9 pages main paper; 15 pages references and appendix
    </div>
    <details class="paper-abstract">
      In this study, we present IL3D, a large-scale dataset meticulously designed for large language model (LLM)-driven 3D scene generation, addressing the pressing demand for diverse, high-quality training data in indoor layout design. Comprising 27,816 indoor layouts across 18 prevalent room types and a library of 29,215 high-fidelity 3D object assets, IL3D is enriched with instance-level natural language annotations to support robust multimodal learning for vision-language tasks. We establish rigorous benchmarks to evaluate LLM-driven scene generation. Experimental results show that supervised fine-tuning (SFT) of LLMs on IL3D significantly improves generalization and surpasses the performance of SFT on other datasets. IL3D offers flexible multimodal data export capabilities, including point clouds, 3D bounding boxes, multiview images, depth maps, normal maps, and semantic masks, enabling seamless adaptation to various visual tasks. As a versatile and robust resource, IL3D significantly advances research in 3D scene generation and embodied intelligence, by providing high-fidelity scene data to support environment perception tasks of embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12078v1">FedLoDrop: Federated LoRA with Dropout for Generalized LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Fine-tuning (FT) large language models (LLMs) is crucial for adapting general-purpose models to specific tasks, enhancing accuracy and relevance with minimal resources. To further enhance generalization ability while reducing training costs, this paper proposes Federated LoRA with Dropout (FedLoDrop), a new framework that applies dropout to the rows and columns of the trainable matrix in Federated LoRA. A generalization error bound and convergence analysis under sparsity regularization are obtained, which elucidate the fundamental trade-off between underfitting and overfitting. The error bound reveals that a higher dropout rate increases model sparsity, thereby lowering the upper bound of pointwise hypothesis stability (PHS). While this reduces the gap between empirical and generalization errors, it also incurs a higher empirical error, which, together with the gap, determines the overall generalization error. On the other hand, though dropout reduces communication costs, deploying FedLoDrop at the network edge still faces challenges due to limited network resources. To address this issue, an optimization problem is formulated to minimize the upper bound of the generalization error, by jointly optimizing the dropout rate and resource allocation subject to the latency and per-device energy consumption constraints. To solve this problem, a branch-and-bound (B\&B)-based method is proposed to obtain its globally optimal solution. Moreover, to reduce the high computational complexity of the B\&B-based method, a penalized successive convex approximation (P-SCA)-based algorithm is proposed to efficiently obtain its high-quality suboptimal solution. Finally, numerical results demonstrate the effectiveness of the proposed approach in mitigating overfitting and improving the generalization capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v3">EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for enhancing vision-language navigation (VLN) performance, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches predominantly adopt straightforward input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. To address these issues, we propose EvolveNav, a novel sElf-improving embodied reasoning paradigm that realizes adaptable and generalizable navigational reasoning for boosting LLM-based vision-language Navigation. Specifically, EvolveNav involves a two-stage training process: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with curated formalized CoT labels to first activate the model's navigational reasoning capabilities, and simultaneously increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also designed to encourage the model to learn correct reasoning patterns by contrasting with wrong ones. Experimental results under both task-specific and cross-task training paradigms demonstrate the consistent superiority of EvolveNav over previous LLM-based VLN approaches on various popular benchmarks, including R2R, REVERIE, CVDN, and SOON. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12064v1">GeoPipe: a Geo-distributed LLM Training Framework with enhanced Pipeline Parallelism in a Lossless RDMA-enabled Datacenter Optical Transport Network</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) with exponentially growing parameters is making cross-data center (DC) training an inevitable trend. However, viable strategies for extending single-DC training frameworks to multi-DC environments remain underdeveloped. We experimentally demonstrate, for the first time, a high-performance geo-distributed LLMs training framework across multiple DCs interconnected by a lossless, remote direct memory access (RDMA) enabled Datacenter Optical Transport Network (DC-OTN). An enhanced pipeline parallelism scheme is implemented within the Ascend full-stack environment of Huawei, which effectively eliminates the impact of cross-DC communication overhead on training efficiency. The overlapped computation and cross-DC communication is achieved with constraint cross-DC bandwidth and High Bandwidth Memory (HBM), reducing computation bubble ratio by up to 78.91%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12061v1">Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Effective disaster response is essential for safeguarding lives and property. Existing statistical approaches often lack semantic context, generalize poorly across events, and offer limited interpretability. While Large language models (LLMs) provide few-shot generalization, they remain text-bound and blind to geography. To bridge this gap, we introduce a Geospatial Awareness Layer (GAL) that grounds LLM agents in structured earth data. Starting from raw wildfire detections, GAL automatically retrieves and integrates infrastructure, demographic, terrain, and weather information from external geodatabases, assembling them into a concise, unit-annotated perception script. This enriched context enables agents to produce evidence-based resource-allocation recommendations (e.g., personnel assignments, budget allocations), further reinforced by historical analogs and daily change signals for incremental updates. We evaluate the framework in real wildfire scenarios across multiple LLM models, showing that geospatially grounded agents can outperform baselines. The proposed framework can generalize to other hazards such as floods and hurricanes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12004v3">The Open Source Advantage in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 9 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly advanced natural language processing, driving significant breakthroughs in tasks such as text generation, machine translation, and domain-specific reasoning. The field now faces a critical dilemma in its approach: closed-source models like GPT-4 deliver state-of-the-art performance but restrict reproducibility, accessibility, and external oversight, while open-source frameworks like LLaMA and Mixtral democratize access, foster collaboration, and support diverse applications, achieving competitive results through techniques like instruction tuning and LoRA. Hybrid approaches address challenges like bias mitigation and resource accessibility by combining the scalability of closed-source systems with the transparency and inclusivity of open-source framework. However, in this position paper, we argue that open-source remains the most robust path for advancing LLM research and ethical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24069v2">Can LLMs Reason Structurally? An Evaluation via the Lens of Data Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) take on increasingly complex tasks, understanding their algorithmic reasoning abilities has become essential. However, existing evaluations focus on distinct and isolated tasks. We propose a unified diagnostic lens: structural reasoning--understanding and manipulating relationships like order, hierarchy, and connectivity. We introduce DSR-Bench, the first benchmark to systematically evaluate LLM structural reasoning through canonical data structures, which serve as interpretable, algorithmically meaningful abstractions. DSR-Bench spans 20 data structures, 35 operations, and 4,140 synthetically generated problem instances with minimal contamination. The benchmark's hierarchical design pinpoints specific failure modes, while its fully automated evaluation ensures objective and consistent assessment. Benchmarking ten state-of-the-art LLMs reveals critical limitations: the top-performing model scores only 0.498 out of 1 on challenging instances. Three additional evaluation suites reveal further weaknesses: models perform poorly on spatial data and natural language scenarios, and fail to reason over their own generated code. DSR-Bench offers a principled diagnostic tool for structural reasoning, helping expose reasoning bottlenecks and guide the development of more capable and reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12023v1">Information Extraction from Conversation Transcripts: Neuro-Symbolic vs. LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The current trend in information extraction (IE) is to rely extensively on large language models, effectively discarding decades of experience in building symbolic or statistical IE systems. This paper compares a neuro-symbolic (NS) and an LLM-based IE system in the agricultural domain, evaluating them on nine interviews across pork, dairy, and crop subdomains. The LLM-based system outperforms the NS one (F1 total: 69.4 vs. 52.7; core: 63.0 vs. 47.2), where total includes all extracted information and core focuses on essential details. However, each system has trade-offs: the NS approach offers faster runtime, greater control, and high accuracy in context-free tasks but lacks generalizability, struggles with contextual nuances, and requires significant resources to develop and maintain. The LLM-based system achieves higher performance, faster deployment, and easier maintenance but has slower runtime, limited control, model dependency and hallucination risks. Our findings highlight the "hidden cost" of deploying NLP systems in real-world applications, emphasizing the need to balance performance, efficiency, and control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17871v2">LLM Probability Concentration: How Alignment Shrinks the Generative Horizon</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Codebase: https://github.com/yangalan123/LLMBranchingFactor. V2: Rewrite the theory part for a broader audience. Add experiments to verify the necessity of the AEP estimator. Generalize findings to multilingual tasks and Qwen models. Add discussions on practical implications, and on which alignment stage contributes most to BF reduction. Add ethical statements connecting pluralistic alignment
    </div>
    <details class="paper-abstract">
      Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this consistency in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the *Branching Factor* (BF)--a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this consistency has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., ``Sure'') that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12997v1">Max It or Miss It: Benchmarking LLM On Solving Extremal Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Our benchmark dataset is available at https://huggingface.co/datasets/binxingao/extrem-bench
    </div>
    <details class="paper-abstract">
      Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12993v1">A Multilingual, Large-Scale Study of the Interplay between LLM Safeguards, Personalisation, and Disinformation</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The human-like proficiency of Large Language Models (LLMs) has brought concerns about their potential misuse for generating persuasive and personalised disinformation at scale. While prior work has demonstrated that LLMs can generate disinformation, specific questions around persuasiveness and personalisation (generation of disinformation tailored to specific demographic attributes) remain largely unstudied. This paper presents the first large-scale, multilingual empirical study on persona-targeted disinformation generation by LLMs. Employing a red teaming methodology, we systematically evaluate the robustness of LLM safety mechanisms to persona-targeted prompts. A key novel result is AI-TRAITS (AI-generaTed peRsonAlIsed disinformaTion dataSet), a new dataset of around 1.6 million texts generated by eight state-of-the-art LLMs. AI-TRAITS is seeded by prompts that combine 324 disinformation narratives and 150 distinct persona profiles, covering four major languages (English, Russian, Portuguese, Hindi) and key demographic dimensions (country, generation, political orientation). The resulting personalised narratives are then assessed quantitatively and compared along the dimensions of models, languages, jailbreaking rate, and personalisation attributes. Our findings demonstrate that the use of even simple personalisation strategies in the prompts significantly increases the likelihood of jailbreaks for all studied LLMs. Furthermore, personalised prompts result in altered linguistic and rhetorical patterns and amplify the persuasiveness of the LLM-generated false narratives. These insights expose critical vulnerabilities in current state-of-the-art LLMs and offer a foundation for improving safety alignment and detection strategies in multilingual and cross-demographic contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12985v1">SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      We present Sentinel, the first framework for formally evaluating the physical safety of Large Language Model(LLM-based) embodied agents across the semantic, plan, and trajectory levels. Unlike prior methods that rely on heuristic rules or subjective LLM judgments, Sentinel grounds practical safety requirements in formal temporal logic (TL) semantics that can precisely specify state invariants, temporal dependencies, and timing constraints. It then employs a multi-level verification pipeline where (i) at the semantic level, intuitive natural language safety requirements are formalized into TL formulas and the LLM agent's understanding of these requirements is probed for alignment with the TL formulas; (ii) at the plan level, high-level action plans and subgoals generated by the LLM agent are verified against the TL formulas to detect unsafe plans before execution; and (iii) at the trajectory level, multiple execution trajectories are merged into a computation tree and efficiently verified against physically-detailed TL specifications for a final safety check. We apply Sentinel in VirtualHome and ALFRED, and formally evaluate multiple LLM-based embodied agents against diverse safety requirements. Our experiments show that by grounding physical safety in temporal logic and applying verification methods across multiple levels, Sentinel provides a rigorous foundation for systematically evaluating LLM-based embodied agents in physical environments, exposing safety violations overlooked by previous methods and offering insights into their failure modes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10423v2">PAL: Probing Audio Encoders via LLMs - Audio Information Transfer into LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 17 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Integration of audio perception into large language models (LLMs) is an emerging research area for enabling machine listening applications, yet efficient transfer of rich audio semantics from audio encoders to LLMs remains underexplored. The most widely used integration paradigm projects the audio encoder output tokens into the LLM input space (e.g., via an MLP or a Q-Former), then prepends or inserts them to the text tokens. We refer to this generic scheme as Prepend to the LLM's input token space (PLITS) integration. We propose an efficient alternative, Lightweight Audio LLM Integration (LAL). LAL introduces audio representations solely via the attention mechanism within different layers of the LLM, bypassing its feedforward module. LAL encodes rich audio semantics at an appropriate level of abstraction for integration into different blocks of LLMs. Our design significantly reduces computational overhead compared to existing integration approaches. Observing with Whisper that the speech encoder benefits from PLITS integration, we propose an audio encoder aware approach for efficiently Probing Audio encoders via LLM (PAL), which employs PLITS integration for Whisper and LAL for general audio encoders. Under an identical training curriculum, LAL consistently maintains performance or outperforms existing integration approaches across multiple base LLMs and tasks. For general audio tasks, LAL improvement is up to 30% over a strong PLITS baseline while reducing memory usage by up to 64.1% and increasing throughput by up to 247.5%. Furthermore, for general audio-music-speech LLM, PAL performs on par with a fully PLITS integration-based system but with substantially improved computational and memory efficiency. Project page: https://ta012.github.io/PAL/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12943v1">The Curious Case of Curiosity across Human Cultures and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Preprint (Paper under review)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have expanded their role in human interaction, yet curiosity -- a central driver of inquiry -- remains underexplored in these systems, particularly across cultural contexts. In this work, we investigate cultural variation in curiosity using Yahoo! Answers, a real-world multi-country dataset spanning diverse topics. We introduce CUEST (CUriosity Evaluation across SocieTies), an evaluation framework that measures human-model alignment in curiosity through linguistic (style), topic preference (content) analysis and grounding insights in social science constructs. Across open- and closed-source models, we find that LLMs flatten cross-cultural diversity, aligning more closely with how curiosity is expressed in Western countries. We then explore fine-tuning strategies to induce curiosity in LLMs, narrowing the human-model alignment gap by up to 50\%. Finally, we demonstrate the practical value of curiosity for LLM adaptability across cultures, showing its importance for future NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12925v1">Who's Asking? Evaluating LLM Robustness to Inquiry Personas in Factual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) should answer factual questions truthfully, grounded in objective knowledge, regardless of user context such as self-disclosed personal information, or system personalization. In this paper, we present the first systematic evaluation of LLM robustness to inquiry personas, i.e. user profiles that convey attributes like identity, expertise, or belief. While prior work has primarily focused on adversarial inputs or distractors for robustness testing, we evaluate plausible, human-centered inquiry persona cues that users disclose in real-world interactions. We find that such cues can meaningfully alter QA accuracy and trigger failure modes such as refusals, hallucinated limitations, and role confusion. These effects highlight how model sensitivity to user framing can compromise factual reliability, and position inquiry persona testing as an effective tool for robustness evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02737v2">Early Signs of Steganographic Capabilities in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances like using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v6">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 30 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Robust workflow composition is critical for effective agent performance, yet progress in Large Language Model (LLM) planning and reasoning is hindered by a scarcity of scalable evaluation data. This work introduces NL2Flow, a fully automated pipeline for generating and evaluating workflow planning problems. NL2Flow generates problems parametrically in a structured intermediate representation, translating them into both natural language and formal PDDL. I evaluate several open-source, instruct-tuned LLMs on a dataset of 2296 low-difficulty problems generated by NL2Flow. Results demonstrate that the best-performing model achieved 86% success in generating valid plans and 69% in generating optimal plans (for solvable problems). Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Importantly, translating natural language problems into a structured JSON representation prior to symbolic planning significantly improved success rates, suggesting a benefit from neuro-symbolic integration. These findings underscore the importance of understanding error sources within LLM reasoning as systems scale to more complex tasks. As LLM reasoning scales to increasingly complex problems, understanding the shifting bottlenecks and sources of error within these systems will be crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21908v2">Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12872v1">KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted for publication in NeurIPS2025. Code is available at \url{https://github.com/HankYe/KVCOMM}
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems are increasingly adopted for complex language processing tasks that require communication and coordination among agents. However, these systems often suffer substantial overhead from repeated reprocessing of overlapping contexts across agents. In typical pipelines, once an agent receives a message from its predecessor, the full context-including prior turns-must be reprocessed from scratch, leading to inefficient processing. While key-value (KV) caching is an effective solution for avoiding redundant computation in single-agent settings where prefixes remain unchanged, it cannot be directly reused in multi-agent scenarios due to diverging prefixes introduced by agent-specific context extensions. We identify that the core challenge lies in the offset variance of KV-caches across agents. To address this, we propose KVCOMM, a training-free framework that enables efficient prefilling in multi-agent inference by reusing KV-caches and aligning cache offsets of overlapping contexts under diverse prefix contexts. KVCOMM estimates and adjusts KV-caches for shared content by referencing a pool of cached examples-termed anchors-that store observed cache deviations under varying prefixes. The anchor pool is maintained and updated online, allowing dynamic adaptation to distinct user requests and context structures. KVCOMM achieves over 70% reuse rate across diverse multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding tasks, all without quality degradation. Particularly, when each fully-connected agent receives 1K input tokens with 512 prefix tokens and 512 output tokens under a five-agent setting, KVCOMM achieves up to 7.8x speedup compared to the standard prefill pipeline, reducing TTFT from ~430 ms to ~55 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12857v1">Adaptive Generation of Bias-Eliciting Questions for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now widely deployed in user-facing applications, reaching hundreds of millions worldwide. As they become integrated into everyday tasks, growing reliance on their outputs raises significant concerns. In particular, users may unknowingly be exposed to model-inherent biases that systematically disadvantage or stereotype certain groups. However, existing bias benchmarks continue to rely on templated prompts or restrictive multiple-choice questions that are suggestive, simplistic, and fail to capture the complexity of real-world user interactions. In this work, we address this gap by introducing a counterfactual bias evaluation framework that automatically generates realistic, open-ended questions over sensitive attributes such as sex, race, or religion. By iteratively mutating and selecting bias-inducing questions, our approach systematically explores areas where models are most susceptible to biased behavior. Beyond detecting harmful biases, we also capture distinct response dimensions that are increasingly relevant in user interactions, such as asymmetric refusals and explicit acknowledgment of bias. Leveraging our framework, we construct CAB, a human-verified benchmark spanning diverse topics, designed to enable cross-model comparisons. Using CAB, we analyze a range of LLMs across multiple bias dimensions, revealing nuanced insights into how different models manifest bias. For instance, while GPT-5 outperforms other models, it nonetheless exhibits persistent biases in specific scenarios. These findings underscore the need for continual improvements to ensure fair model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12801v1">DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) in real-world applications require access to external knowledge sources and must remain responsive to the dynamic and ever-changing real-world information in order to address information-seeking and knowledge-intensive user queries. Existing approaches, such as retrieval augmented generation (RAG) methods, search agents, and search equipped MLLMs, often suffer from rigid pipelines, excessive search calls, and poorly constructed search queries, which result in inefficiencies and suboptimal outcomes. To address these limitations, we present DeepMMSearch-R1, the first multimodal LLM capable of performing on-demand, multi-turn web searches and dynamically crafting queries for both image and text search tools. Specifically, DeepMMSearch-R1 can initiate web searches based on relevant crops of the input image making the image search more effective, and can iteratively adapt text search queries based on retrieved information, thereby enabling self-reflection and self-correction. Our approach relies on a two-stage training pipeline: a cold start supervised finetuning phase followed by an online reinforcement learning optimization. For training, we introduce DeepMMSearchVQA, a novel multimodal VQA dataset created through an automated pipeline intermixed with real-world information from web search tools. This dataset contains diverse, multi-hop queries that integrate textual and visual information, teaching the model when to search, what to search for, which search tool to use and how to reason over the retrieved information. We conduct extensive experiments across a range of knowledge-intensive benchmarks to demonstrate the superiority of our approach. Finally, we analyze the results and provide insights that are valuable for advancing multimodal web-search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12773v1">Dr.LLM: Dynamic Layer Routing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 17 pages, Under submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce Dr.LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), Dr.LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr.LLM shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12728v1">Data-Model Co-Evolution: Growing Test Sets to Refine LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      A long-standing challenge in machine learning has been the rigid separation between data work and model refinement, enforced by slow fine-tuning cycles. The rise of Large Language Models (LLMs) overcomes this historical barrier, allowing applications developers to instantly govern model behavior by editing prompt instructions. This shift enables a new paradigm: data-model co-evolution, where a living test set and a model's instructions evolve in tandem. We operationalize this paradigm in an interactive system designed to address the critical challenge of encoding subtle, domain-specific policies into prompt instructions. The system's structured workflow guides people to discover edge cases, articulate rationales for desired behavior, and iteratively evaluate instruction revisions against a growing test set. A user study shows our workflow helps participants refine instructions systematically and specify ambiguous policies more concretely. This work points toward more robust and responsible LLM applications through human-in-the-loop development aligned with local preferences and policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12721v1">CARVQ: Corrective Adaptor with Group Residual Vector Quantization for LLM Embedding Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted at EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) typically rely on a large number of parameters for token embedding, leading to substantial storage requirements and memory footprints. In particular, LLMs deployed on edge devices are memory-bound, and reducing the memory footprint by compressing the embedding layer not only frees up the memory bandwidth but also speeds up inference. To address this, we introduce CARVQ, a post-training novel Corrective Adaptor combined with group Residual Vector Quantization. CARVQ relies on the composition of both linear and non-linear maps and mimics the original model embedding to compress to approximately 1.6 bits without requiring specialized hardware to support lower-bit storage. We test our method on pre-trained LLMs such as LLaMA-3.2-1B, LLaMA-3.2-3B, LLaMA-3.2-3B-Instruct, LLaMA-3.1-8B, Qwen2.5-7B, Qwen2.5-Math-7B and Phi-4, evaluating on common generative, discriminative, math and reasoning tasks. We show that in most cases, CARVQ can achieve lower average bitwidth-per-parameter while maintaining reasonable perplexity and accuracy compared to scalar quantization. Our contributions include a novel compression technique that is compatible with state-of-the-art transformer quantization methods and can be seamlessly integrated into any hardware supporting 4-bit memory to reduce the model's memory footprint in memory-constrained devices. This work demonstrates a crucial step toward the efficient deployment of LLMs on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12699v1">Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Different open-ended generation tasks require different degrees of output diversity. However, current LLMs are often miscalibrated. They collapse to overly homogeneous outputs for creative tasks and hallucinate diverse but incorrect responses for factual tasks. We argue that these two failure modes are unified by, and can both be addressed by, the notion of effective generation space size (GSS) -- the set of semantically distinct outputs a model considers for a prompt. We present GSSBench, a task suite of prompt pairs with ground-truth GSS relationships to assess different metrics and understand where models diverge from desired behavior. We find that hallucination detection metrics, particularly EigenScore, consistently outperform standard diversity and uncertainty quantification metrics, while using only model internals, providing interpretable insights into a model's internal task representations. We demonstrate three applications of GSS: (1) detecting prompt ambiguity and predicting clarification questions for better grounding, (2) interpreting overthinking and underthinking in reasoning models, and (3) steering models to expand their generation space to yield high-quality and diverse outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12697v1">Multi-Agent Debate for LLM Judges with Adaptive Stability Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      With advancements in reasoning capabilities, Large Language Models (LLMs) are increasingly employed for automated judgment tasks. While LLMs-as-Judges offer promise in automating evaluations, current approaches often rely on simplistic aggregation methods (e.g., majority voting), which can fail even when individual agents provide correct answers. To address this, we propose a multi-agent debate judge framework where agents collaboratively reason and iteratively refine their responses. We formalize the debate process mathematically, analyzing agent interactions and proving that debate amplifies correctness compared to static ensembles. To enhance efficiency, we introduce a stability detection mechanism that models judge consensus dynamics via a time-varying Beta-Binomial mixture, with adaptive stopping based on distributional similarity (Kolmogorov-Smirnov test). This mechanism models the judges' collective correct rate dynamics using a time-varying mixture of Beta-Binomial distributions and employs an adaptive stopping criterion based on distributional similarity (Kolmogorov-Smirnov statistic). Experiments across multiple benchmarks and models demonstrate that our framework improves judgment accuracy over majority voting while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12689v1">From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising accuracy in predicting survey responses and policy preferences, which has increased interest in their potential to represent human interests in various domains. Most existing research has focused on behavioral cloning, effectively evaluating how well models reproduce individuals' expressed preferences. Drawing on theories of political representation, we highlight an underexplored design trade-off: whether AI systems should act as delegates, mirroring expressed preferences, or as trustees, exercising judgment about what best serves an individual's interests. This trade-off is closely related to issues of LLM sycophancy, where models can encourage behavior or validate beliefs that may be aligned with a user's short-term preferences, but is detrimental to their long-term interests. Through a series of experiments simulating votes on various policy issues in the U.S. context, we apply a temporal utility framework that weighs short and long-term interests (simulating a trustee role) and compare voting outcomes to behavior-cloning models (simulating a delegate). We find that trustee-style predictions weighted toward long-term interests produce policy decisions that align more closely with expert consensus on well-understood issues, but also show greater bias toward models' default stances on topics lacking clear agreement. These findings reveal a fundamental trade-off in designing AI systems to represent human interests. Delegate models better preserve user autonomy but may diverge from well-supported policy positions, while trustee models can promote welfare on well-understood issues yet risk paternalism and bias on subjective topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12680v1">Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Hybrid thinking enables LLMs to switch between reasoning and direct answering, offering a balance between efficiency and reasoning capability. Yet our experiments reveal that current hybrid thinking LLMs only achieve partial mode separation: reasoning behaviors often leak into the no-think mode. To understand and mitigate this, we analyze the factors influencing controllability and identify four that matter most: (1) larger data scale, (2) using think and no-think answers from different questions rather than the same question, (3) a moderate increase in no-think data number, and (4) a two-phase strategy that first trains reasoning ability and then applies hybrid think training. Building on these findings, we propose a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from $1085$ to $585$ on MATH500) and occurrences of reasoning-supportive tokens such as ``\texttt{wait}'' (from $5917$ to $522$ on MATH500). Our findings highlight the limitations of current hybrid thinking and offer directions for strengthening its controllability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14214v2">Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted to AAAI 2026 Innovative Applications of AI
    </div>
    <details class="paper-abstract">
      LLM-based autonomous agents have recently shown strong capabilities in solving complex industrial design tasks. However, in domains aiming for carbon neutrality and high-performance renewable energy systems, current AI-assisted design automation methods face critical challenges in explainability, scalability, and practical usability. To address these limitations, we introduce PHIA (Physics-Informed Autonomous Agent), an LLM-driven system that automates modulation design for power converters in Power Electronics Systems with minimal human intervention. In contrast to traditional pipeline-based methods, PHIA incorporates an LLM-based planning module that interactively acquires and verifies design requirements via a user-friendly chat interface. This planner collaborates with physics-informed simulation and optimization components to autonomously generate and iteratively refine modulation designs. The interactive interface also supports interpretability by providing textual explanations and visual outputs throughout the design process. Experimental results show that PHIA reduces standard mean absolute error by 63.2% compared to the second-best benchmark and accelerates the overall design process by over 33 times. A user study involving 20 domain experts further confirms PHIA's superior design efficiency and usability, highlighting its potential to transform industrial design workflows in power electronics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v5">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem. The code is available at \href{https://github.com/NVIDIA/kvpress}{link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02392v2">KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: https://github.com/AIFrontierLab/KnowledgeSmith.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12608v1">StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      With the increasing integration of large language models (LLMs) into open-domain writing, detecting machine-generated text has become a critical task for ensuring content authenticity and trust. Existing approaches rely on statistical discrepancies or model-specific heuristics to distinguish between LLM-generated and human-written text. However, these methods struggle in real-world scenarios due to limited generalization, vulnerability to paraphrasing, and lack of explainability, particularly when facing stylistic diversity or hybrid human-AI authorship. In this work, we propose StyleDecipher, a robust and explainable detection framework that revisits LLM-generated text detection using combined feature extractors to quantify stylistic differences. By jointly modeling discrete stylistic indicators and continuous stylistic representations derived from semantic embeddings, StyleDecipher captures distinctive style-level divergences between human and LLM outputs within a unified representation space. This framework enables accurate, explainable, and domain-agnostic detection without requiring access to model internals or labeled segments. Extensive experiments across five diverse domains, including news, code, essays, reviews, and academic abstracts, demonstrate that StyleDecipher consistently achieves state-of-the-art in-domain accuracy. Moreover, in cross-domain evaluations, it surpasses existing baselines by up to 36.30%, while maintaining robustness against adversarial perturbations and mixed human-AI content. Further qualitative and quantitative analysis confirms that stylistic signals provide explainable evidence for distinguishing machine-generated text. Our source code can be accessed at https://github.com/SiyuanLi00/StyleDecipher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23564v2">Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: https://github.com/deeplearning-wisc/PrefCleanBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18389v4">ALLSTaR: Automated LLM-Driven Scheduler Generation and Testing for Intent-Based RAN</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Under submission to an IEEE journal, copyright may change without notice
    </div>
    <details class="paper-abstract">
      The evolution toward open, programmable O-RAN and AI-RAN 6G networks creates unprecedented opportunities for Intent-Based Networking (IBN) to dynamically optimize RAN[...]. However, applying IBN effectively to the RAN scheduler [...] remains a significant challenge. Current approaches predominantly rely on coarse-grained network slicing, lacking the granularity for dynamic adaptation to individual user conditions and traffic patterns. Despite the existence of a vast body of scheduling algorithms [...], their practical utilization is hindered by implementation heterogeneity, insufficient systematic evaluation in production environments, and the complexity of developing high-performance scheduler implementations.[...] To address these limitations, we propose ALLSTaR (Automated LLm-driven Scheduler generation and Testing for intent-based RAN), a novel framework leveraging LLMs for automated, intent-driven scheduler design, implementation, and evaluation. ALLSTaR interprets NL intents, automatically generates functional scheduler code from the research literature using OCR and LLMs, and intelligently matches operator intents to the most suitable scheduler(s). Our implementation deploys these schedulers as O-RAN dApps, enabling on-the-fly deployment and testing on a production-grade, 5G-compliant testbed. This approach has enabled the largest-scale OTA experimental comparison of 18 scheduling algorithms automatically synthesized from the academic literature. The resulting performance profiles serve as the input for our Intent-Based Scheduling (IBS) framework, which dynamically selects and deploys appropriate schedulers that optimally satisfy operator intents. We validate our approach through multiple use cases unattainable with current slicing-based optimization techniques, demonstrating fine-grained control based on buffer status, physical layer conditions, and heterogeneous traffic types
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08621v2">MooseAgent: A LLM Based Multi-agent Framework for Automating Moose Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 11 pages, 3 Figs
    </div>
    <details class="paper-abstract">
      The Finite Element Method (FEM) is widely used in engineering and scientific computing, but its pre-processing, solver configuration, and post-processing stages are often time-consuming and require specialized knowledge. This paper proposes an automated solution framework, MooseAgent, for the multi-physics simulation framework MOOSE, which combines large-scale pre-trained language models (LLMs) with a multi-agent system. The framework uses LLMs to understand user-described simulation requirements in natural language and employs task decomposition and multi-round iterative verification strategies to automatically generate MOOSE input files. To improve accuracy and reduce model hallucinations, the system builds and utilizes a vector database containing annotated MOOSE input cards and function documentation. We conducted experimental evaluations on several typical cases, including heat transfer, mechanics, phase field, and multi-physics coupling. The results show that MooseAgent can automate the MOOSE simulation process to a certain extent, especially demonstrating a high success rate when dealing with relatively simple single-physics problems. The main contribution of this research is the proposal of a multi-agent automated framework for MOOSE, which validates its potential in simplifying finite element simulation processes and lowering the user barrier, providing new ideas for the development of intelligent finite element simulation software. The code for the MooseAgent framework proposed in this paper has been open-sourced and is available at https://github.com/taozhan18/MooseAgent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02038v2">Persuasion at Play: Understanding Misinformation Dynamics in Demographic-Aware Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Existing challenges in misinformation exposure and susceptibility vary across demographic groups, as some populations are more vulnerable to misinformation than others. Large language models (LLMs) introduce new dimensions to these challenges through their ability to generate persuasive content at scale and reinforcing existing biases. This study investigates the bidirectional persuasion dynamics between LLMs and humans when exposed to misinformative content. We analyze human-to-LLM influence using human-stance datasets and assess LLM-to-human influence by generating LLM-based persuasive arguments. Additionally, we use a multi-agent LLM framework to analyze the spread of misinformation under persuasion among demographic-oriented LLM agents. Our findings show that demographic factors influence susceptibility to misinformation in LLMs, closely reflecting the demographic-based patterns seen in human susceptibility. We also find that, similar to human demographic groups, multi-agent LLMs exhibit echo chamber behavior. This research explores the interplay between humans and LLMs, highlighting demographic differences in the context of misinformation and offering insights for future interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12462v1">Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to autonomously evaluate the quality of content in communication systems, e.g., to assess responses in telecom customer support chatbots. However, the impartiality of these AI "judges" is not guaranteed, and any biases in their evaluation criteria could skew outcomes and undermine user trust. In this paper, we systematically investigate judgment biases in two LLM-as-a-judge models (i.e., GPT-Judge and JudgeLM) under the point-wise scoring setting, encompassing 11 types of biases that cover both implicit and explicit forms. We observed that state-of-the-art LLM judges demonstrate robustness to biased inputs, generally assigning them lower scores than the corresponding clean samples. Providing a detailed scoring rubric further enhances this robustness. We further found that fine-tuning an LLM on high-scoring yet biased responses can significantly degrade its performance, highlighting the risk of training on biased data. We also discovered that the judged scores correlate with task difficulty: a challenging dataset like GPQA yields lower average scores, whereas an open-ended reasoning dataset (e.g., JudgeLM-val) sees higher average scores. Finally, we proposed four potential mitigation strategies to ensure fair and reliable AI judging in practical communication scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02613v3">ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Training LLMs relies on distributed implementations using multiple GPUs to compute gradients in parallel with sharded optimizers. However, synchronizing gradients in data parallel setups introduces communication overhead that grows with the number of workers, limiting parallelization efficiency. Local optimization algorithms reduce communications but incur high memory costs as they prevent optimizer state sharding, hindering scalability. To address this, we propose \textbf{AC}cumulate while \textbf{CO}mmunicate (ACCO), a memory-efficient optimization algorithm for distributed LLM training. By synchronizing delayed gradients while computing new ones, ACCO reduces GPU idle time and supports heterogeneous hardware. To mitigate the convergence issues caused by delayed updates, we introduce a novel technique ensuring training dynamics align with standard distributed optimization. Compared to ZeRO-1, our approach is significantly faster and scales effectively across heterogeneous hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12423v1">MTOS: A LLM-Driven Multi-topic Opinion Simulation Framework for Exploring Echo Chamber Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 14 pages, 11figures
    </div>
    <details class="paper-abstract">
      The polarization of opinions, information segregation, and cognitive biases on social media have attracted significant academic attention. In real-world networks, information often spans multiple interrelated topics, posing challenges for opinion evolution and highlighting the need for frameworks that simulate interactions among topics. Existing studies based on large language models (LLMs) focus largely on single topics, limiting the capture of cognitive transfer in multi-topic, cross-domain contexts. Traditional numerical models, meanwhile, simplify complex linguistic attitudes into discrete values, lacking interpretability, behavioral consistency, and the ability to integrate multiple topics. To address these issues, we propose Multi-topic Opinion Simulation (MTOS), a social simulation framework integrating multi-topic contexts with LLMs. MTOS leverages LLMs alongside short-term and long-term memory, incorporates multiple user-selection interaction mechanisms and dynamic topic-selection strategies, and employs a belief decay mechanism to enable perspective updates across topics. We conduct extensive experiments on MTOS, varying topic numbers, correlation types, and performing ablation studies to assess features such as group polarization and local consistency. Results show that multi-topic settings significantly alter polarization trends: positively correlated topics amplify echo chambers, negatively correlated topics inhibit them, and irrelevant topics also mitigate echo chamber effects through resource competition. Compared with numerical models, LLM-based agents realistically simulate dynamic opinion changes, reproduce linguistic features of news texts, and capture complex human reasoning, improving simulation interpretability and system stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12409v1">PricingLogic: Evaluating LLMs Reasoning on Complex Tourism Pricing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      We present PricingLogic, the first benchmark that probes whether Large Language Models(LLMs) can reliably automate tourism-related prices when multiple, overlapping fare rules apply. Travel agencies are eager to offload this error-prone task onto AI systems; however, deploying LLMs without verified reliability could result in significant financial losses and erode customer trust. PricingLogic comprises 300 natural-language questions based on booking requests derived from 42 real-world pricing policies, spanning two levels of difficulty: (i) basic customer-type pricing and (ii)bundled-tour calculations involving interacting discounts. Evaluations of a line of LLMs reveal a steep performance drop on the harder tier,exposing systematic failures in rule interpretation and arithmetic reasoning.These results highlight that, despite their general capabilities, today's LLMs remain unreliable in revenue-critical applications without further safeguards or domain adaptation. Our code and dataset are available at https://github.com/EIT-NLP/PricingLogic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07140v6">Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant success in reasoning tasks, including mathematical reasoning and logical deduction. Among these reasoning tasks, graph problems stand out due to their complexity and unique structural characteristics, attracting considerable attention from researchers. Previous studies have explored LLMs' graph reasoning abilities through various techniques, such as different encoding methods for graph structures and the use of carefully designed prompts. However, a critical factor has been mostly overlooked: the prompt sequential order in which graph descriptions are presented to the models. In this study, we present the first comprehensive analysis of how the order of graph descriptions impacts LLM performance. Specifically, we comprehensively evaluate four graph description orders across six graph problems using six mainstream LLMs. The results reveal that: (1) ordered graph descriptions significantly improve LLMs' comprehension of graph structures; (2) the robustness of LLMs to graph description order varies across different tasks; and (3) the impact of graph order on performance is closely related to the inherent characteristics of tasks. This study provides a critical advancement in the application of LLMs for solving graph-related problems, paving the way for future research to optimize model performance through strategic graph description ordering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12389v1">Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 6 pages 4 figures
    </div>
    <details class="paper-abstract">
      Tokenization disparities pose a significant barrier to achieving equitable access to artificial intelligence across linguistically diverse populations. This study conducts a large-scale cross-linguistic evaluation of tokenization efficiency in over 200 languages to systematically quantify computational inequities in large language models (LLMs). Using a standardized experimental framework, we applied consistent preprocessing and normalization protocols, followed by uniform tokenization through the tiktoken library across all language samples. Comprehensive tokenization statistics were collected using established evaluation metrics, including Tokens Per Sentence (TPS) and Relative Tokenization Cost (RTC), benchmarked against English baselines. Our cross-linguistic analysis reveals substantial and systematic disparities: Latin-script languages consistently exhibit higher tokenization efficiency, while non-Latin and morphologically complex languages incur significantly greater token inflation, often 3-5 times higher RTC ratios. These inefficiencies translate into increased computational costs and reduced effective context utilization for underrepresented languages. Overall, the findings highlight structural inequities in current AI systems, where speakers of low-resource and non-Latin languages face disproportionate computational disadvantages. Future research should prioritize the development of linguistically informed tokenization strategies and adaptive vocabulary construction methods that incorporate typological diversity, ensuring more inclusive and computationally equitable multilingual AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12367v1">LLM-REVal: Can We Trust LLM Reviewers Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation. This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments: (1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones; (2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions. Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements. These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12355v1">Fine-grained Analysis of Brain-LLM Alignment through Input Attribution</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Understanding the alignment between large language models (LLMs) and human brain activity can reveal computational principles underlying language processing. We introduce a fine-grained input attribution method to identify the specific words most important for brain-LLM alignment, and leverage it to study a contentious research question about brain-LLM alignment: the relationship between brain alignment (BA) and next-word prediction (NWP). Our findings reveal that BA and NWP rely on largely distinct word subsets: NWP exhibits recency and primacy biases with a focus on syntax, while BA prioritizes semantic and discourse-level information with a more targeted recency effect. This work advances our understanding of how LLMs relate to human language processing and highlights differences in feature reliance between BA and NWP. Beyond this study, our attribution method can be broadly applied to explore the cognitive relevance of model predictions in diverse language processing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12350v1">O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12306v1">A large-scale, unsupervised pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable, unsupervised pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English consider construction. Using GPT-5 through the OpenAI API, we annotate 143,933 sentences from the Corpus of Historical American English (COHA) in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, opening new possibilities for corpus-based research, though implementation requires attention to costs, licensing, and other ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12294v1">Show Your Title! A Scoping Review on Verbalization in Software Engineering with LLM-Assisted Screening</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 preprint of a paper under publication in Quality of Information and Communications Technology 2025
    </div>
    <details class="paper-abstract">
      Understanding how software developers think, make decisions, and behave remains a key challenge in software engineering (SE). Verbalization techniques (methods that capture spoken or written thought processes) offer a lightweight and accessible way to study these cognitive aspects. This paper presents a scoping review of research at the intersection of SE and psychology (PSY), focusing on the use of verbal data. To make large-scale interdisciplinary reviews feasible, we employed a large language model (LLM)-assisted screening pipeline using GPT to assess the relevance of over 9,000 papers based solely on titles. We addressed two questions: what themes emerge from verbalization-related work in SE, and how effective are LLMs in supporting interdisciplinary review processes? We validated GPT's outputs against human reviewers and found high consistency, with a 13\% disagreement rate. Prominent themes mainly were tied to the craft of SE, while more human-centered topics were underrepresented. The data also suggests that SE frequently draws on PSY methods, whereas the reverse is rare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26072v2">The Silent Judge: Unacknowledged Shortcut Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 9 Pages, 5 Tables, 1 Figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as automatic judges to evaluate system outputs in tasks such as summarization, dialogue, and creative writing. A faithful judge should base its verdicts solely on response quality and explicitly acknowledge the factors shaping its decision. We show that current LLM judges fail on both counts by relying on shortcuts introduced in the prompt. Our study uses two evaluation datasets: ELI5, a benchmark for long-form question answering, and LitBench, a recent benchmark for creative writing. Both datasets provide pairwise comparisons, where the evaluator must choose which of two responses is better. From each dataset we construct 100 pairwise judgment tasks and employ two widely used models, GPT-4o and Gemini-2.5-Flash, as evaluators in the role of LLM-as-a-judge. For each pair, we assign superficial cues to the responses, provenance cues indicating source identity (Human, Expert, LLM, or Unknown) and recency cues indicating temporal origin (Old, 1950 vs. New, 2025), while keeping the rest of the prompt fixed. Results reveal consistent verdict shifts: both models exhibit a strong recency bias, systematically favoring new responses over old, as well as a clear provenance hierarchy (Expert > Human > LLM > Unknown). These biases are especially pronounced in GPT-4o and in the more subjective and open-ended LitBench domain. Crucially, cue acknowledgment is rare: justifications almost never reference the injected cues, instead rationalizing decisions in terms of content qualities. These findings demonstrate that current LLM-as-a-judge systems are shortcut-prone and unfaithful, undermining their reliability as evaluators in both research and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26041v2">Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 5 Pages, 4 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12255v1">Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Dataset and code: https://huggingface.co/datasets/dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust ; https://github.com/bmanczak/MedQA-MultiTurnRobustness Accepted as a poster at NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly transitioning into medical clinical use, yet their reliability under realistic, multi-turn interactions remains poorly understood. Existing evaluation frameworks typically assess single-turn question answering under idealized conditions, overlooking the complexities of medical consultations where conflicting input, misleading context, and authority influence are common. We introduce MedQA-Followup, a framework for systematically evaluating multi-turn robustness in medical question answering. Our approach distinguishes between shallow robustness (resisting misleading initial context) and deep robustness (maintaining accuracy when answers are challenged across turns), while also introducing an indirect-direct axis that separates contextual framing (indirect) from explicit suggestion (direct). Using controlled interventions on the MedQA dataset, we evaluate five state-of-the-art LLMs and find that while models perform reasonably well under shallow perturbations, they exhibit severe vulnerabilities in multi-turn settings, with accuracy dropping from 91.2% to as low as 13.5% for Claude Sonnet 4. Counterintuitively, indirect, context-based interventions are often more harmful than direct suggestions, yielding larger accuracy drops across models and exposing a significant vulnerability for clinical deployment. Further compounding analyses reveal model differences, with some showing additional performance drops under repeated interventions while others partially recovering or even improving. These findings highlight multi-turn robustness as a critical but underexplored dimension for safe and reliable deployment of medical LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v3">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12245v1">MoRA: On-the-fly Molecule-aware Low-Rank Adaptation Framework for LLM-based Multi-Modal Molecular Assistant</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Effectively integrating molecular graph structures with Large Language Models (LLMs) is a key challenge in drug discovery. Most existing multi-modal alignment methods typically process these structures by fine-tuning the LLM or adding a static adapter simultaneously. However, these approaches have two main limitations: (1) it optimizes a shared parameter space across all molecular inputs, limiting the model's ability to capture instance-specific structural features; and (2) fine-tuning the LLM for molecular tasks can lead to catastrophic forgetting, undermining its general reasoning capabilities. In this paper, instead of static task-oriented adaptation, we propose an instance-specific parameter space alignment approach for each molecule on-the-fly. To this end, we introduce Molecule-aware Low-Rank Adaptation (MoRA) that produces a unique set of low-rank adaptation weights for each input molecular graph. These weights are then dynamically injected into a frozen LLM, allowing the model to adapt its reasoning to the structure of each molecular input, while preserving the LLM's core knowledge. Extensive experiments demonstrate that on key molecular tasks, such as chemical reaction prediction and molecular captioning, MoRA's instance-specific dynamic adaptation outperforms statically adapted baselines, including a 14.1% relative improvement in reaction prediction exact match and a 22% reduction in error for quantum property prediction. The code is available at https://github.com/jk-sounds/MoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12229v1">Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12224v1">MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The reliable evaluation of large language models (LLMs) in medical applications remains an open challenge, particularly in capturing the complexity of multi-turn doctor-patient interactions that unfold in real clinical environments. Existing evaluation methods typically rely on post hoc review of full conversation transcripts, thereby neglecting the dynamic, context-sensitive nature of medical dialogues and the evolving informational needs of patients. In this work, we present MedKGEval, a novel multi-turn evaluation framework for clinical LLMs grounded in structured medical knowledge. Our approach introduces three key contributions: (1) a knowledge graph-driven patient simulation mechanism, where a dedicated control module retrieves relevant medical facts from a curated knowledge graph, thereby endowing the patient agent with human-like and realistic conversational behavior. This knowledge graph is constructed by integrating open-source resources with additional triples extracted from expert-annotated datasets; (2) an in-situ, turn-level evaluation framework, where each model response is assessed by a Judge Agent for clinical appropriateness, factual correctness, and safety as the dialogue progresses using a suite of fine-grained, task-specific metrics; (3) a comprehensive multi-turn benchmark of eight state-of-the-art LLMs, demonstrating MedKGEval's ability to identify subtle behavioral flaws and safety risks that are often overlooked by conventional evaluation pipelines. Although initially designed for Chinese and English medical applications, our framework can be readily extended to additional languages by switching the input knowledge graphs, ensuring seamless bilingual support and domain-specific applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12217v1">HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13907v1">LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly sensitive to their input prompts, making prompt design a central challenge. While automatic prompt optimization (APO) reduces manual engineering, most approaches assume access to ground-truth references such as labeled validation data. In practice, however, collecting high-quality labels is costly and slow. We propose the Prompt Duel Optimizer (PDO), a sample-efficient framework for label-free prompt optimization. PDO formulates the problem as a dueling-bandit setting, where supervision signal comes from pairwise preference feedback provided by an LLM judge. The framework combines Double Thompson Sampling (D-TS), which prioritizes informative prompt comparisons, with Top-Performer Guided Mutation, which expands the candidate pool by mutating high-performing prompts. PDO naturally operates in label-free settings and can also incorporate partial labels to mitigate judge noise. Experiments on BIG-bench Hard (BBH) and MS MARCO show that PDO consistently outperforms baseline methods. Ablation studies further demonstrate the effectiveness of both D-TS and prompt mutation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13901v1">RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13898v1">Attribution Quality in AI-Generated Content:Benchmarking Style Embeddings and LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted for publication at the 2025 IEEE ICDM Workshop on "Grounding Documents with Reasoning, Agents, Retrieval, and Attribution". This is author submitted version. Not yet published
    </div>
    <details class="paper-abstract">
      Attributing authorship in the era of large language models (LLMs) is increasingly challenging as machine-generated prose rivals human writing. We benchmark two complementary attribution mechanisms , fixed Style Embeddings and an instruction-tuned LLM judge (GPT-4o) on the Human AI Parallel Corpus, an open dataset of 600 balanced instances spanning six domains (academic, news, fiction, blogs, spoken transcripts, and TV/movie scripts). Each instance contains a human prompt with both a gold continuation and an LLM-generated continuation from either GPT-4o or LLaMA-70B-Instruct. The Style Embedding baseline achieves stronger aggregate accuracy on GPT continuations (82 pct vs. 68 pct). The LLM Judge is slightly better than the Style embeddings on LLaMA continuations (85 pct vs. 81 pct) but the results are not statistically significant. Crucially, the LLM judge significantly outperforms in fiction and academic prose, indicating semantic sensitivity, whereas embeddings dominate in spoken and scripted dialogue, reflecting structural strengths. These complementary patterns highlight attribution as a multidimensional problem requiring hybrid strategies. To support reproducibility we provide code on GitHub and derived data on Hugging Face under the MIT license. This open framework provides a reproducible benchmark for attribution quality assessment in AI-generated content, along with a review of related literature influencing this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11076v1">DebugTA: An LLM-Based Agent for Simplifying Debugging and Teaching in Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      In programming education, Debugging and Teaching (DT) task is a common scenario where students receive assistance in correcting their erroneous code. The task involves multiple inputs, including erroneous code, error messages, reference solutions, and the question description, with the goal of generating modification suggestions to the erroneous code. However, two key challenges hinder the effectiveness of existing approaches. Firstly, the complexity and heterogeneity of inputs inherent in DT tasks significantly elevate the reasoning challenges faced by LLMs. Second, existing approaches often fail to fully leverage the availability of standard code in DT tasks, forcing models to rely solely on complex multi-step reasoning, which limits the potential of LLMs in addressing DT tasks effectively. To address these challenges, we propose DebugTA, a novel LLM-based debugging and teaching agent with specialized tools for standard code retrieval, variable substitution to align reference code, and an external compiler for real-time code analysis. Guided by explicit pedagogical and debugging principles, DebugTA acts as an agent that decomposes a complex task into sequential LLM interactions, each utilizing distinct tools for specific subtasks, thereby simplifying the logical reasoning at each step and reducing overall reasoning complexity. Furthermore, DebugTA utilizes tool calls to align the standard code with the erroneous code as much as possible, allowing the LLM to focus on logic errors within the erroneous code and improving the accuracy of the generated suggestions. To rigorously assess the quality of modification suggestions, we introduce a student simulator-teacher interaction paradigm. Experimental results on three real-world code datasets demonstrate that DebugTA consistently improves teaching effectiveness while significantly reducing computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v3">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12839v1">FaStFACT: Faster, Stronger Long-Form Factuality Evaluations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to accuracy issues and costly human assessment. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to complex pipeline components unsuitable for long LLM outputs, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence collection of one-line snippets. To address these limitations, we propose \name, a fast and strong evaluation framework that achieves the highest alignment with human evaluation and efficiency among existing baselines. \name first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the cost of web searching and inference calling while ensuring reliability. For searching and verification, it collects document-level evidence from crawled webpages and selectively retrieves it during verification, addressing the evidence insufficiency problem in previous pipelines. Extensive experiments based on an aggregated and manually annotated benchmark demonstrate the reliability of \name in both efficiently and effectively evaluating the factuality of long-form LLM generations. Code and benchmark data is available at https://github.com/Yingjia-Wan/FastFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12836v1">BanglaMATH : A Bangla benchmark dataset for testing LLM mathematical reasoning at grades 6, 7, and 8</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have tremendous potential to play a key role in supporting mathematical reasoning, with growing use in education and AI research. However, most existing benchmarks are limited to English, creating a significant gap for low-resource languages. For example, Bangla is spoken by nearly 250 million people who would collectively benefit from LLMs capable of native fluency. To address this, we present BanglaMATH, a dataset of 1.7k Bangla math word problems across topics such as Arithmetic, Algebra, Geometry, and Logical Reasoning, sourced from Bangla elementary school workbooks and annotated with details like grade level and number of reasoning steps. We have designed BanglaMATH to evaluate the mathematical capabilities of both commercial and open-source LLMs in Bangla, and we find that Gemini 2.5 Flash and DeepSeek V3 are the only models to achieve strong performance, with $\ge$ 80\% accuracy across three elementary school grades. Furthermore, we assess the robustness and language bias of these top-performing LLMs by augmenting the original problems with distracting information, and translating the problems into English. We show that both LLMs fail to maintain robustness and exhibit significant performance bias in Bangla. Our study underlines current limitations of LLMs in handling arithmetic and mathematical reasoning in low-resource languages, and highlights the need for further research on multilingual and equitable mathematical understanding. Dataset link: \href{https://github.com/TabiaTanzin/BanglaMATH-A-Bangla-benchmark-dataset-for-testing-LLM-mathematical-reasoning-at-grades-6-7-and-8.git}{https://github.com/BanglaMATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12835v1">Repurposing Annotation Guidelines to Instruct LLM Annotators: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 11 pages, 2 figures, 3 tables, This is a preprint of the article accepted at NLDB 2025 (Springer LNCS). The final version is available at https://doi.org/10.1007/978-3-031-97144-0_13
    </div>
    <details class="paper-abstract">
      This study investigates how existing annotation guidelines can be repurposed to instruct large language model (LLM) annotators for text annotation tasks. Traditional guidelines are written for human annotators who internalize training, while LLMs require explicit, structured instructions. We propose a moderation-oriented guideline repurposing method that transforms guidelines into clear directives for LLMs through an LLM moderation process. Using the NCBI Disease Corpus as a case study, our experiments show that repurposed guidelines can effectively guide LLM annotators, while revealing several practical challenges. The results highlight the potential of this workflow to support scalable and cost-effective refinement of annotation guidelines and automated annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v4">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Need to address additional data or methodological concerns
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine reasoning or superficial pattern recognition. Common evaluation methods, which focus on the either the final answer or the reasoning process, fail to assess the entire problem-solving procedure. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework, together with its corresponding benchmark, SMART-Bench. SMART decomposes the entire problem solving process into four distinct cognitive dimensions: Understanding, Reasoning, Arithmetic, and Reflection \& Refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings reveal genuine weaknesses in current LLMs and motivate a new metric, the All-Pass Score, to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11062v1">Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11056v1">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11040v1">Enabling Doctor-Centric Medical AI with LLMs through Workflow-Aligned Tasks and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has transformed healthcare by offering clinical guidance, yet their direct deployment to patients poses safety risks due to limited domain expertise. To mitigate this, we propose repositioning LLMs as clinical assistants that collaborate with experienced physicians rather than interacting with patients directly. We conduct a two-stage inspiration-feedback survey to identify real-world needs in clinical workflows. Guided by this, we construct DoctorFLAN, a large-scale Chinese medical dataset comprising 92,000 Q&A instances across 22 clinical tasks and 27 specialties. To evaluate model performance in doctor-facing applications, we introduce DoctorFLAN-test (550 single-turn Q&A items) and DotaBench (74 multi-turn conversations). Experimental results with over ten popular LLMs demonstrate that DoctorFLAN notably improves the performance of open-source LLMs in medical contexts, facilitating their alignment with physician workflows and complementing existing patient-oriented models. This work contributes a valuable resource and framework for advancing doctor-centered medical LLM development
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12575v2">DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at https://github.com/whfeLingYu/DemonAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05431v2">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25279v2">RL in the Wild: Characterizing RLVR Training in LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 20 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used across many domains. With their rapid development, Reinforcement Learning with Verifiable Rewards (RLVR) has surged in recent months to enhance their reasoning and understanding abilities. However, its complex data flows and diverse tasks pose substantial challenges to RL training systems, and there is limited understanding of RLVR from a system perspective. To thoroughly understand the system challenges introduced by RLVR, we present a characterization study of RLVR tasks in our LLM deployment. Specifically, we investigate the distribution and variation trends of workloads across different RL tasks across training steps. We identify issues such as GPU idling caused by skewed sequence length distribution, inefficient parallel strategies in dynamically varying workloads, inefficient data management mechanisms, and load imbalance. We describe our observations and call for further investigation into the remaining open challenges. Furthermore, we propose PolyTrace benchmark suite to conduct evaluation with realistic workloads, and a practical use case validates that PolyTrace benchmark suite exhibits 94.7% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10998v1">ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 28 pages, 11 figures, 16 tables. In submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were further amplified by sharp increases in intersectional harms (e.g., Tokenism) for gender and caste-marginalized disabled candidates, highlighting critical blind spots in current safety tools and the need for intersectional safety evaluations of frontier models in high-stakes domains like hiring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10987v1">DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 14 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at https://github.com/hsannn/ditto.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10978v1">Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Language Bias, whereby the model over-relies on auxiliary tokens-such as task descriptions and prefix-generated tokens-while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating language bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates language bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04492v2">Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The advancement of LLMs and their accessibility have triggered renewed interest in multi-agent reinforcement learning as robust and adaptive frameworks for dynamically changing environments. This paper introduces RL-Focal, a two-stage RL agent framework that routes and ensembles LLMs. First, we develop the Decider RL-agent, which learns to dynamically select an ensemble of small size ($m_i$) among $N$ LLMs ($m_i \ll N$) for incoming queries from a user-defined downstream task $i$, by maximizing both error-diversity and reasoning-performance of the selected ensemble through iterative updates of task-adaptive rewards and policy. Second, to enable effective fusion of dynamically selected LLMs, we develop the stage-2 Fusion RL-agent, which learns to resolve reasoning conflicts from different LLMs and dynamically adapts to different ensemble teams composed by the Decider Agent for different downstream tasks. Third, we introduce the focal diversity metric to better model the error correlations among multiple LLMs, further improving the generalization performance of the Decider Agent, which actively prunes the ensemble combinations. By focal diversity, we enhance performance across tasks by effectively promoting reward-aware and policy-adaptive ensemble selection and inference fusion. Extensive evaluations on five benchmarks show that RL-Focal achieves the performance improvement of 8.48\% with an ensemble of small size compared to the best individual LLM in a pool and offers stronger robustness. Code is available at https://github.com/sftekin/rl-focal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v1">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10955v1">HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a surge of research on leveraging large language models (LLMs) for sequential recommendation. LLMs have demonstrated remarkable potential in inferring users' nuanced preferences through fine-grained semantic reasoning. However, they also exhibit a notable limitation in effectively modeling collaborative signals, i.e., behavioral correlations inherent in users' historical interactions. Our empirical analysis further reveals that the attention mechanisms in LLMs tend to disproportionately focus on tokens within the same item, thereby impeding the capture of cross-item correlations. To address this limitation, we propose a novel hierarchical attention masking strategy for LLM-based recommendation, termed HatLLM. Specifically, in shallow layers, HatLLM masks attention between tokens from different items, facilitating intra-item semantic understanding; in contrast, in deep layers, HatLLM masks attention within items, thereby compelling the model to capture cross-item correlations. This progressive, layer-wise approach enables LLMs to jointly model both token-level and item-level dependencies. Extensive experiments on three real-world datasets demonstrate that HatLLM achieves significant performance gains (9.13% on average) over existing LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09393v2">ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender systems. Existing approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity users. To address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity users. With a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce platform. ChoirRec improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v2">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD dot com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD dot com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v4">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15507v5">Steering LLMs for Formal Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent advances in automated theorem proving use Large Language Models (LLMs) to translate informal mathematical statements into formal proofs. However, informal cues are often ambiguous or lack strict logical structure, making it hard for models to interpret them precisely. While existing methods achieve strong performance, little is known about how LLMs internally represent informal cues, or how these influence proof generation. To address this, we explore \textit{activation steering}, an inference-time intervention that identifies linear directions in residual activations associated with informal reasoning traces and adjusts them to improve proof construction without fine-tuning. This mechanism also yields interpretable information about how reasoning is internally encoded in the activation space of LLMs. We test our method for generating formal proofs from already-formalized theorems. Our contributions are twofold: (1) a novel activation-based intervention for guiding proof synthesis in LLMs; and (2) demonstration that this intervention improves performance under two decoding strategies (sampling and best-first search) without any further training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10895v1">LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 This work has been submitted to IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Medium Access Control (MAC) protocols, essential for wireless networks, are typically manually configured. While deep reinforcement learning (DRL)-based protocols enhance task-specified network performance, they suffer from poor generalizability and resilience, demanding costly retraining to adapt to dynamic environments. To overcome this limitation, we introduce a game-theoretic LLM-empowered multi-agent DRL (MARL) framework, in which the uplink transmission between a base station and a varying number of user equipments is modeled as a dynamic multi-follower Stackelberg game (MFSG), capturing the network's natural hierarchical structure. Within this game, LLM-driven agents, coordinated through proximal policy optimization (PPO), synthesize adaptive, semantic MAC protocols in response to network dynamics. Protocol action grammar (PAG) is employed to ensure the reliability and efficiency of this process. Under this system, we further analyze the existence and convergence behavior in terms of a Stackelberg equilibrium by studying the learning dynamics of LLM-empowered unified policies in response to changing followers. Simulations corroborate that our framework achieves a 77.6% greater throughput and a 65.2% fairness improvement over conventional baselines. Besides, our framework generalizes excellently to a fluctuating number of users without requiring retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07680v3">Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue, but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MoE model, our method speeds up the training by 2.4$\times$ with competitive performance. Codes will be released at https://github.com/ModelTC/HBP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10890v1">LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted by EMNLP2025 System Demonstration
    </div>
    <details class="paper-abstract">
      We introduce LLM x MapReduce-V3, a hierarchically modular agent system designed for long-form survey generation. Building on the prior work, LLM x MapReduce-V2, this version incorporates a multi-agent architecture where individual functional components, such as skeleton initialization, digest construction, and skeleton refinement, are implemented as independent model-context-protocol (MCP) servers. These atomic servers can be aggregated into higher-level servers, creating a hierarchically structured system. A high-level planner agent dynamically orchestrates the workflow by selecting appropriate modules based on their MCP tool descriptions and the execution history. This modular decomposition facilitates human-in-the-loop intervention, affording users greater control and customization over the research process. Through a multi-turn interaction, the system precisely captures the intended research perspectives to generate a comprehensive skeleton, which is then developed into an in-depth survey. Human evaluations demonstrate that our system surpasses representative baselines in both content depth and length, highlighting the strength of MCP-based modular planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v4">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06994v3">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern application codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.19\% (upto 65\%) reduction in application binary size, and an average of 6.08\% (upto 20\%) performance improvement over the traditional profile-guided optimization, without any execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12470v2">Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2208.08067v3">K-ASTRO: Structure-Aware Adaptation of LLMs for Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming software engineering tasks, including code vulnerability detection-a critical area of software security. However, existing methods often rely on resource-intensive models or graph-based techniques, limiting their accessibility and practicality. This paper introduces K-ASTRO, a lightweight Transformer model that combines semantic embeddings from LLMs with structural features of Abstract Syntax Trees (ASTs) to improve both efficiency and accuracy in code vulnerability detection. Our approach introduces an AST-based augmentation technique inspired by mutation testing, a structure-aware attention mechanism that incorporates augmented AST features, and a joint adaptation pipeline to unify code semantics and syntax. Experimental results on three large-scale datasets, including BigVul, DiverseVul, and PrimeVul-demonstrate state-of-the-art performance while enabling rapid inference on CPUs with minimal training time. By offering a scalable, interpretable, and efficient solution, K-ASTRO bridges the gap between LLM advancements and practical software vulnerability detection, providing open-sourced tools to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11974v1">CTIArena: Benchmarking LLM Knowledge and Reasoning Across Heterogeneous Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Under peer-review
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) is central to modern cybersecurity, providing critical insights for detecting and mitigating evolving threats. With the natural language understanding and reasoning capabilities of large language models (LLMs), there is increasing interest in applying them to CTI, which calls for benchmarks that can rigorously evaluate their performance. Several early efforts have studied LLMs on some CTI tasks but remain limited: (i) they adopt only closed-book settings, relying on parametric knowledge without leveraging CTI knowledge bases; (ii) they cover only a narrow set of tasks, lacking a systematic view of the CTI landscape; and (iii) they restrict evaluation to single-source analysis, unlike realistic scenarios that require reasoning across multiple sources. To fill these gaps, we present CTIArena, the first benchmark for evaluating LLM performance on heterogeneous, multi-source CTI under knowledge-augmented settings. CTIArena spans three categories, structured, unstructured, and hybrid, further divided into nine tasks that capture the breadth of CTI analysis in modern security operations. We evaluate ten widely used LLMs and find that most struggle in closed-book setups but show noticeable gains when augmented with security-specific knowledge through our designed retrieval-augmented techniques. These findings highlight the limitations of general-purpose LLMs and the need for domain-tailored techniques to fully unlock their potential for CTI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11967v1">Scaling Long-Horizon LLM Agent via Context-Folding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05670v2">Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v4">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11938v1">FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EuroSys 26
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) in production faces significant challenges from highly variable request patterns and severe resource fragmentation in serverless clusters. Current systems rely on static pipeline configurations that struggle to adapt to dynamic workload conditions, leading to substantial inefficiencies. We present FlexPipe, a novel system that dynamically reconfigures pipeline architectures during runtime to address these fundamental limitations. FlexPipe decomposes models into fine-grained stages and intelligently adjusts pipeline granularity based on real-time request pattern analysis, implementing three key innovations: fine-grained model partitioning with preserved computational graph constraints, inflight pipeline refactoring with consistent cache transitions, and topology-aware resource allocation that navigates GPU fragmentation. Comprehensive evaluation on an 82-GPU cluster demonstrates that FlexPipe achieves up to 8.5x better resource efficiency while maintaining 38.3% lower latency compared to state-of-the-art systems, reducing GPU reservation requirements from 75% to 30% of peak capacity.
    </details>
</div>
