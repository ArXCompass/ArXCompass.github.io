# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17479v2">DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across various natural language processing tasks but often struggle to excel uniformly in diverse or complex domains. We propose a novel ensemble method - Diverse Fingerprint Ensemble (DFPE), which leverages the complementary strengths of multiple LLMs to achieve more robust performance. Our approach involves: (1) clustering models based on response "fingerprints" patterns, (2) applying a quantile-based filtering mechanism to remove underperforming models at a per-subject level, and (3) assigning adaptive weights to remaining models based on their subject-wise validation accuracy. In experiments on the Massive Multitask Language Understanding (MMLU) benchmark, DFPE outperforms the best single model by 3% overall accuracy and 5% in discipline-level accuracy. This method increases the robustness and generalization of LLMs and underscores how model selection, diversity preservation, and performance-driven weighting can effectively address challenging, multi-faceted language understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04510v1">Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles and weights. We represent multi-LLM systems as directed acyclic graphs (DAGs) of LLMs with topological message passing for collaborative generation. Given a pool of LLM experts and a utility function, Heterogeneous Swarms employs two iterative steps: role-step and weight-step. For role-step, we interpret model roles as learning a DAG that specifies the flow of inputs and outputs between LLMs. Starting from a swarm of random continuous adjacency matrices, we decode them into discrete DAGs, call the LLMs in topological order, evaluate on the utility function (e.g. accuracy on a task), and optimize the adjacency matrices with particle swarm optimization based on the utility score. For weight-step, we assess the contribution of individual LLMs in the multi-LLM systems and optimize model weights with swarm intelligence. We propose JFK-score to quantify the individual contribution of each LLM in the best-found DAG of the role-step, then optimize model weights with particle swarm optimization based on the JFK-score. Experiments demonstrate that Heterogeneous Swarms outperforms 15 role- and/or weight-based baselines by 18.5% on average across 12 tasks. Further analysis reveals that Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04506v1">When One LLM Drools, Multi-LLM Collaboration Rules</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      This position paper argues that in many realistic (i.e., complex, contextualized, subjective) scenarios, one LLM is not enough to produce a reliable output. We challenge the status quo of relying solely on a single general-purpose LLM and argue for multi-LLM collaboration to better represent the extensive diversity of data, skills, and people. We first posit that a single LLM underrepresents real-world data distributions, heterogeneous skills, and pluralistic populations, and that such representation gaps cannot be trivially patched by further training a single LLM. We then organize existing multi-LLM collaboration methods into a hierarchy, based on the level of access and information exchange, ranging from API-level, text-level, logit-level, to weight-level collaboration. Based on these methods, we highlight how multi-LLM collaboration addresses challenges that a single LLM struggles with, such as reliability, democratization, and pluralism. Finally, we identify the limitations of existing multi-LLM methods and motivate future work. We envision multi-LLM collaboration as an essential path toward compositional intelligence and collaborative AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21035v2">Beyond Autoregression: Fast LLMs via Self-Distillation Through Time</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Autoregressive (AR) Large Language Models (LLMs) have demonstrated significant success across numerous tasks. However, the AR modeling paradigm presents certain limitations; for instance, contemporary autoregressive LLMs are trained to generate one token at a time, which can result in noticeable latency. Recent advances have indicated that search and repeated sampling can enhance performance in various applications, such as theorem proving, code generation, and alignment, by utilizing greater computational resources during inference. In this study, we demonstrate that diffusion language models are capable of generating at least 32 tokens simultaneously, while exceeding the performance of AR models in text quality and on the LAMBADA natural language understanding benchmark. This outcome is achieved through a novel distillation method for discrete diffusion models, which reduces the number of inference steps by a factor of 32-64. Practically, at the 1.3B parameters scale, diffusion models, even without caching, can generate tokens at a rate that is up to 8 times faster than AR models employing KV-caching, and we anticipate further improvements with the inclusion of caching. Moreover, we demonstrate the efficacy of our approach for diffusion language models with up to 860M parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04485v1">Active Task Disambiguation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of large language models (LLMs) across various benchmarks, their ability to address ambiguously specified problems--frequent in real-world interactions--remains underexplored. To address this gap, we introduce a formal definition of task ambiguity and frame the problem of task disambiguation through the lens of Bayesian Experimental Design. By posing clarifying questions, LLM agents can acquire additional task specifications, progressively narrowing the space of viable solutions and reducing the risk of generating unsatisfactory outputs. Yet, generating effective clarifying questions requires LLM agents to engage in a form of meta-cognitive reasoning, an ability LLMs may presently lack. Our proposed approach of active task disambiguation enables LLM agents to generate targeted questions maximizing the information gain. Effectively, this approach shifts the load from implicit to explicit reasoning about the space of viable solutions. Empirical results demonstrate that this form of question selection leads to more effective task disambiguation in comparison to approaches relying on reasoning solely within the space of questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04428v1">Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed and democratized on edge devices. To improve the efficiency of on-device deployment, small language models (SLMs) are often adopted due to their efficient decoding latency and reduced energy consumption. However, these SLMs often generate inaccurate responses when handling complex queries. One promising solution is uncertainty-based SLM routing, offloading high-stakes queries to stronger LLMs when resulting in low-confidence responses on SLM. This follows the principle of "If you lack confidence, seek stronger support" to enhance reliability. Relying on more powerful LLMs is yet effective but increases invocation costs. Therefore, striking a routing balance between efficiency and efficacy remains a critical challenge. Additionally, efficiently generalizing the routing strategy to new datasets remains under-explored. In this paper, we conduct a comprehensive investigation into benchmarking and generalization of uncertainty-driven routing strategies from SLMs to LLMs over 1500+ settings. Our findings highlight: First, uncertainty-correctness alignment in different uncertainty quantification (UQ) methods significantly impacts routing performance. Second, uncertainty distributions depend more on both the specific SLM and the chosen UQ method, rather than downstream data. Building on the insight, we propose a calibration data construction instruction pipeline and open-source a constructed hold-out set to enhance routing generalization on new downstream scenarios. The experimental results indicate calibration data effectively bootstraps routing performance without any new data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v1">Decoding AI Judgment: How LLMs Assess News Credibility and Bias</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to assess news credibility, yet little is known about how they make these judgments. While prior research has examined political bias in LLM outputs or their potential for automated fact-checking, their internal evaluation processes remain largely unexamined. Understanding how LLMs assess credibility provides insights into AI behavior and how credibility is structured and applied in large-scale language models. This study benchmarks the reliability and political classifications of state-of-the-art LLMs - Gemini 1.5 Flash (Google), GPT-4o mini (OpenAI), and LLaMA 3.1 (Meta) - against structured, expert-driven rating systems such as NewsGuard and Media Bias Fact Check. Beyond assessing classification performance, we analyze the linguistic markers that shape LLM decisions, identifying which words and concepts drive their evaluations. We uncover patterns in how LLMs associate credibility with specific linguistic features by examining keyword frequency, contextual determinants, and rank distributions. Beyond static classification, we introduce a framework in which LLMs refine their credibility assessments by retrieving external information, querying other models, and adapting their responses. This allows us to investigate whether their assessments reflect structured reasoning or rely primarily on prior learned associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v1">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04419v1">Understanding and Mitigating the Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Technical report; 31 pages
    </div>
    <details class="paper-abstract">
      Generating synthetic datasets via large language models (LLMs) themselves has emerged as a promising approach to improve LLM performance. However, LLMs inherently reflect biases present in their training data, leading to a critical challenge: when these models generate synthetic data for training, they may propagate and amplify their inherent biases that can significantly impact model fairness and robustness on downstream tasks--a phenomenon we term bias inheritance. This work presents the first systematic investigation in understanding, analyzing, and mitigating bias inheritance. We study this problem by fine-tuning LLMs with a combined dataset consisting of original and LLM-augmented data, where bias ratio represents the proportion of augmented data. Through systematic experiments across 10 classification and generation tasks, we analyze how 6 different types of biases manifest at varying bias ratios. Our results reveal that bias inheritance has nuanced effects on downstream tasks, influencing both classification tasks and generation tasks differently. Then, our analysis identifies three key misalignment factors: misalignment of values, group data, and data distributions. Based on these insights, we propose three mitigation strategies: token-based, mask-based, and loss-based approaches. Experiments demonstrate that these strategies also work differently on various tasks and bias, indicating the substantial challenges to fully mitigate bias inheritance. We hope this work can provide valuable insights to the research of LLM data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04416v1">CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance by scaling model parameters, but this comes with significant inference overhead. Feed-forward networks (FFNs), which dominate LLM parameters, exhibit high activation sparsity in hidden neurons. To exploit this, researchers have proposed using a mixture-of-experts (MoE) architecture, where only a subset of parameters is activated. However, existing approaches often require extensive training data and resources, limiting their practicality. We propose CMoE (Carved MoE), a novel framework to efficiently carve MoE models from dense models. CMoE achieves remarkable performance through efficient expert grouping and lightweight adaptation. First, neurons are grouped into shared and routed experts based on activation rates. Next, we construct a routing mechanism without training from scratch, incorporating a differentiable routing process and load balancing. Using modest data, CMoE produces a well-designed, usable MoE from a 7B dense model within five minutes. With lightweight fine-tuning, it achieves high-performance recovery in under an hour. We make our code publicly available at https://github.com/JarvisPei/CMoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04412v1">Decoder-Only LLMs are Better Controllers for Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Groundbreaking advancements in text-to-image generation have recently been achieved with the emergence of diffusion models. These models exhibit a remarkable ability to generate highly artistic and intricately detailed images based on textual prompts. However, obtaining desired generation outcomes often necessitates repetitive trials of manipulating text prompts just like casting spells on a magic mirror, and the reason behind that is the limited capability of semantic understanding inherent in current image generation models. Specifically, existing diffusion models encode the text prompt input with a pre-trained encoder structure, which is usually trained on a limited number of image-caption pairs. The state-of-the-art large language models (LLMs) based on the decoder-only structure have shown a powerful semantic understanding capability as their architectures are more suitable for training on very large-scale unlabeled data. In this work, we propose to enhance text-to-image diffusion models by borrowing the strength of semantic understanding from large language models, and devise a simple yet effective adapter to allow the diffusion models to be compatible with the decoder-only structure. Meanwhile, we also provide a supporting theoretical analysis with various architectures (e.g., encoder-only, encoder-decoder, and decoder-only), and conduct extensive empirical evaluations to verify its effectiveness. The experimental results show that the enhanced models with our adapter module are superior to the stat-of-the-art models in terms of text-to-image generation quality and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04411v1">Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 work in progress. arXiv admin note: text overlap with arXiv:2405.09673 by other authors
    </div>
    <details class="paper-abstract">
      Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts. To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Considering the out-of-distribution samples, we select and merge appropriate experts based on the task uncertainty of the input data. We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04394v1">DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Disease</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Alzheimer's Disease (AD) is an irreversible neurodegenerative disease affecting 50 million people worldwide. Low-cost, accurate identification of key markers of AD is crucial for timely diagnosis and intervention. Language impairment is one of the earliest signs of cognitive decline, which can be used to discriminate AD patients from normal control individuals. Patient-interviewer dialogues may be used to detect such impairments, but they are often mixed with ambiguous, noisy, and irrelevant information, making the AD detection task difficult. Moreover, the limited availability of AD speech samples and variability in their speech styles pose significant challenges in developing robust speech-based AD detection models. To address these challenges, we propose DECT, a novel speech-based domain-specific approach leveraging large language models (LLMs) for fine-grained linguistic analysis and label-switched label-preserved data generation. Our study presents four novelties: We harness the summarizing capabilities of LLMs to identify and distill key Cognitive-Linguistic information from noisy speech transcripts, effectively filtering irrelevant information. We leverage the inherent linguistic knowledge of LLMs to extract linguistic markers from unstructured and heterogeneous audio transcripts. We exploit the compositional ability of LLMs to generate AD speech transcripts consisting of diverse linguistic patterns to overcome the speech data scarcity challenge and enhance the robustness of AD detection models. We use the augmented AD textual speech transcript dataset and a more fine-grained representation of AD textual speech transcript data to fine-tune the AD detection model. The results have shown that DECT demonstrates superior model performance with an 11% improvement in AD detection accuracy on the datasets from DementiaBank compared to the baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05227v1">Robotouille: An Asynchronous Planning Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 11 pages (not including references or appendix); 41 figures (7 main paper, 34 appendix); (v1) preprint
    </div>
    <details class="paper-abstract">
      Effective asynchronous planning, or the ability to efficiently reason and plan over states and actions that must happen in parallel or sequentially, is essential for agents that must account for time delays, reason over diverse long-horizon tasks, and collaborate with other agents. While large language model (LLM) agents show promise in high-level task planning, current benchmarks focus primarily on short-horizon tasks and do not evaluate such asynchronous planning capabilities. We introduce Robotouille, a challenging benchmark environment designed to test LLM agents' ability to handle long-horizon asynchronous scenarios. Our synchronous and asynchronous datasets capture increasingly complex planning challenges that go beyond existing benchmarks, requiring agents to manage overlapping tasks and interruptions. Our results show that ReAct (gpt4-o) achieves 47% on synchronous tasks but only 11% on asynchronous tasks, highlighting significant room for improvement. We further analyze failure modes, demonstrating the need for LLM agents to better incorporate long-horizon feedback and self-audit their reasoning during task execution. Code is available at https://github.com/portal-cornell/robotouille.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05224v1">A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significantly advanced capabilities in understanding and generating human language text, which have gained increasing popularity over recent years. Apart from their state-of-the-art natural language processing (NLP) performance, considering their widespread usage in many industries, including medicine, finance, education, etc., security concerns over their usage grow simultaneously. In recent years, the evolution of backdoor attacks has progressed with the advancement of defense mechanisms against them and more well-developed features in the LLMs. In this paper, we adapt the general taxonomy for classifying machine learning attacks on one of the subdivisions - training-time white-box backdoor attacks. Besides systematically classifying attack methods, we also consider the corresponding defense methods against backdoor attacks. By providing an extensive summary of existing works, we hope this survey can serve as a guideline for inspiring future research that further extends the attack scenarios and creates a stronger defense against them for more robust LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06843v1">Vision-Integrated LLMs for Autonomous Driving Assistance : Human Performance Comparison and Trust Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Traditional autonomous driving systems often struggle with reasoning in complex, unexpected scenarios due to limited comprehension of spatial relationships. In response, this study introduces a Large Language Model (LLM)-based Autonomous Driving (AD) assistance system that integrates a vision adapter and an LLM reasoning module to enhance visual understanding and decision-making. The vision adapter, combining YOLOv4 and Vision Transformer (ViT), extracts comprehensive visual features, while GPT-4 enables human-like spatial reasoning and response generation. Experimental evaluations with 45 experienced drivers revealed that the system closely mirrors human performance in describing situations and moderately aligns with human decisions in generating appropriate responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03450v1">A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Scene graphs have emerged as a structured and serializable environment representation for grounded spatial reasoning with Large Language Models (LLMs). In this work, we propose SG-RwR, a Schema-Guided Retrieve-while-Reason framework for reasoning and planning with scene graphs. Our approach employs two cooperative, code-writing LLM agents: a (1) Reasoner for task planning and information queries generation, and a (2) Retriever for extracting corresponding graph information following the queries. Two agents collaborate iteratively, enabling sequential reasoning and adaptive attention to graph information. Unlike prior works, both agents are prompted only with the scene graph schema rather than the full graph data, which reduces the hallucination by limiting input tokens, and drives the Reasoner to generate reasoning trace abstractly.Following the trace, the Retriever programmatically query the scene graph data based on the schema understanding, allowing dynamic and global attention on the graph that enhances alignment between reasoning and retrieval. Through experiments in multiple simulation environments, we show that our framework surpasses existing LLM-based approaches in numerical Q\&A and planning tasks, and can benefit from task-level few-shot examples, even in the absence of agent-level demonstrations. Project code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03447v1">Designing LLM-simulated Immersive Spaces to Enhance Autistic Children's Social Affordances Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 iui2025
    </div>
    <details class="paper-abstract">
      One of the key challenges faced by autistic children is understanding social affordances in complex environments, which further impacts their ability to respond appropriately to social signals. In traffic scenarios, this impairment can even lead to safety concerns. In this paper, we introduce an LLM-simulated immersive projection environment designed to improve this ability in autistic children while ensuring their safety. We first propose 17 design considerations across four major categories, derived from a comprehensive review of previous research. Next, we developed a system called AIroad, which leverages LLMs to simulate drivers with varying social intents, expressed through explicit multimodal social signals. AIroad helps autistic children bridge the gap in recognizing the intentions behind behaviors and learning appropriate responses through various stimuli. A user study involving 14 participants demonstrated that this technology effectively engages autistic children and leads to significant improvements in their comprehension of social affordances in traffic scenarios. Additionally, parents reported high perceived usability of the system. These findings highlight the potential of combining LLM technology with immersive environments for the functional rehabilitation of autistic children in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v1">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating proof search spaces. While the existing approaches primarily rely on value functions and Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Search (BFS) remains underexplored. This paper investigates whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present \texttt{BFS-Prover}, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. \texttt{BFS-Prover} achieves a score of $71.31$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02542v2">OverThink: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 18x slowdown on FreshQA dataset and 46x slowdown on SQuAD dataset. The attack also shows high transferability across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third-party applications operating reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03373v1">Demystifying Long Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Scaling inference compute enhances reasoning in large language models (LLMs), with long chains-of-thought (CoTs) enabling strategies like backtracking and error correction. Reinforcement learning (RL) has emerged as a crucial method for developing these capabilities, yet the conditions under which long CoTs emerge remain unclear, and RL training requires careful design choices. In this study, we systematically investigate the mechanics of long CoT reasoning, identifying the key factors that enable models to generate long CoT trajectories. Through extensive supervised fine-tuning (SFT) and RL experiments, we present four main findings: (1) While SFT is not strictly necessary, it simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL. We find that leveraging noisy, web-extracted solutions with filtering mechanisms shows strong potential, particularly for out-of-distribution (OOD) tasks such as STEM reasoning; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute, and measuring their emergence requires a nuanced approach. These insights provide practical guidance for optimizing training strategies to enhance long CoT reasoning in LLMs. Our code is available at: https://github.com/eddycmu/demystify-long-cot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00326v8">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 19 pages, 12 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v1">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose \textbf{Di}vergence-driven \textbf{Z}eroth-\textbf{O}rder (\textbf{DiZO}) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v2">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as annotators and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure -- the Alternative Annotator Test (alt-test) -- that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming open-source LLMs, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15230v3">PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 32 pages, 7 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Neural Networks can be effectively compressed through pruning, significantly reducing storage and compute demands while maintaining predictive performance. Simple yet effective methods like magnitude pruning remove less important parameters and typically require a costly retraining procedure to restore performance. However, with the rise of LLMs, full retraining has become infeasible due to memory and compute constraints. This study challenges the practice of retraining all parameters by showing that updating a small subset of highly expressive parameters can suffice to recover or even enhance performance after pruning. Surprisingly, retraining just 0.01%-0.05% of the parameters in GPT-architectures can match the performance of full retraining across various sparsity levels, significantly reducing compute and memory requirements, and enabling retraining of models with up to 30 billion parameters on a single GPU in minutes. To bridge the gap to full retraining in the high sparsity regime, we introduce two novel LoRA variants that, unlike standard LoRA, allow merging adapters back without compromising sparsity. Going a step further, we show that these methods can be applied for memory-efficient layer-wise reconstruction, significantly enhancing state-of-the-art retraining-free methods like Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023). Our findings present a promising alternative to avoiding retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03159v1">PICBench: Benchmarking LLMs for Photonic Integrated Circuits Design</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable potential in automating various tasks in digital chip design, the field of Photonic Integrated Circuits (PICs)-a promising solution to advanced chip designs-remains relatively unexplored in this context. The design of PICs is time-consuming and prone to errors due to the extensive and repetitive nature of code involved in photonic chip design. In this paper, we introduce PICBench, the first benchmarking and evaluation framework specifically designed to automate PIC design generation using LLMs, where the generated output takes the form of a netlist. Our benchmark consists of dozens of meticulously crafted PIC design problems, spanning from fundamental device designs to more complex circuit-level designs. It automatically evaluates both the syntax and functionality of generated PIC designs by comparing simulation outputs with expert-written solutions, leveraging an open-source simulator. We evaluate a range of existing LLMs, while also conducting comparative tests on various prompt engineering techniques to enhance LLM performance in automated PIC design. The results reveal the challenges and potential of LLMs in the PIC design domain, offering insights into the key areas that require further research and development to optimize automation in this field. Our benchmark and evaluation code is available at https://github.com/PICDA/PICBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03080v1">IAO Prompting: Making Knowledge Flow Explicit in LLMs through Structured Reasoning Templates</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted as Oral at KnowFM @ AAAI 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, understanding and validating their knowledge utilization remains challenging. Chain-of-thought (CoT) prompting partially addresses this by revealing intermediate reasoning steps, but the knowledge flow and application remain implicit. We introduce IAO (Input-Action-Output) prompting, a structured template-based method that explicitly models how LLMs access and apply their knowledge during complex reasoning tasks. IAO decomposes problems into sequential steps, each clearly identifying the input knowledge being used, the action being performed, and the resulting output. This structured decomposition enables us to trace knowledge flow, verify factual consistency, and identify potential knowledge gaps or misapplications. Through experiments across diverse reasoning tasks, we demonstrate that IAO not only improves zero-shot performance but also provides transparency in how LLMs leverage their stored knowledge. Human evaluation confirms that this structured approach enhances our ability to verify knowledge utilization and detect potential hallucinations or reasoning errors. Our findings provide insights into both knowledge representation within LLMs and methods for more reliable knowledge application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05498v3">SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted by USENIX Security Symposium 2025. Please cite the conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner. In Proc. USENIX Security, 2025."
    </div>
    <details class="paper-abstract">
      Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02988v1">Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 accepted at WWW'25 (Industrial Track), extended version
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for their adoption as evaluative judges. This paper introduces Themis, a fine-tuned LLM judge that delivers sophisticated context-aware evaluations. We provide a comprehensive overview of the development pipeline for Themis, highlighting its scenario-dependent evaluation prompts and two novel methods for controlled instruction generation. These designs enable Themis to effectively distill evaluative skills from teacher models, while retaining flexibility for continuous development. We introduce two human-labeled benchmarks for meta-evaluation, demonstrating that Themis can achieve high alignment with human preferences in an economical manner. Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing nuances in performance and the varied effects of reference answers. Notably, we observe that pure knowledge distillation from strong LLMs, though common, does not guarantee performance improvement through scaling. We propose a mitigation strategy based on instruction-following difficulty. Furthermore, we provide practical guidelines covering data balancing, prompt customization, multi-objective training, and metric aggregation. We aim for our method and findings, along with the fine-tuning data, benchmarks, and model checkpoints, to support future research and development in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14442v2">A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to NAACL2025 main conference
    </div>
    <details class="paper-abstract">
      Recently, sharing key-value (KV) cache across layers has been found effective in efficient inference of large language models (LLMs). To systematically investigate different techniques of cross-layer KV sharing, we propose a unified framework that covers several recent methods and their novel variants. We conduct comprehensive experiments on all the configurations of the framework, evaluating their generation throughput and performance in language modeling and downstream tasks. We find that when reducing the size of the KV cache by 2$\times$, most configurations can achieve higher throughput than standard transformers while maintaining competitive performance. When further reducing the size of the KV cache, however, pairing queries of all layers with KVs of upper layers performs better, at the expense of additional training cost and prefilling latency. We hope that this work will help users make more informed choices of cross-layer KV sharing approaches and facilitate future research on efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10020v3">Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Long Paper, 9 pages, accepted at NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13331v2">Qrazor: Reliable and Effortless 4-bit LLM Quantization by Significant Data Razoring</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) excel in language processing tasks but face deployment challenges due to high memory and computational demands. While low-bit quantization, such as 4-bit techniques, offers a potential solution, these methods often suffer from significant accuracy loss or require considerable effort for implementation such as reordering, rotation, etc. To address these challenges, we propose QRazor, a simple yet effective quantization scheme that enables 4-bit quantization of weights, activations, and KV cache in transformer-based LLMs. QRazor operates in two stages: first, quantizing data using 8 or 16-bit integers as a basis with absolute max scaling to preserve accuracy close to full-precision models, and second, compressing the quantized data to 4-bit using our significant data razoring (SDR) technique, which retains only the four most salient bits. Without any additional requirment of fine-tuning or additional training, QRazor achieves performance similar or better compared to state-of-the-art in 4-bit quantization method, surpassing Smoothquant and QLLM by over 12 points and Quarot(RTN) by more than 2.9 points in zero-shot reasoning task accuracy on the LLaMA2-7B model. Additionally, we introduce an integer-based arithmetic unit optimized for QRazor, allowing direct low-precision operations on SDR data without decompression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02966v1">FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We propose FACTER, a fairness-aware framework for LLM-based recommendation systems that integrates conformal prediction with dynamic prompt engineering. By introducing an adaptive semantic variance threshold and a violation-triggered mechanism, FACTER automatically tightens fairness constraints whenever biased patterns emerge. We further develop an adversarial prompt generator that leverages historical violations to reduce repeated demographic biases without retraining the LLM. Empirical results on MovieLens and Amazon show that FACTER substantially reduces fairness violations (up to 95.5%) while maintaining strong recommendation accuracy, revealing semantic variance as a potent proxy of bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14202v3">Rationale Behind Essay Scores: Enhancing S-LLM's Multi-Trait Essay Scoring with Rationale Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Existing automated essay scoring (AES) has solely relied on essay text without using explanatory rationales for the scores, thereby forgoing an opportunity to capture the specific aspects evaluated by rubric indicators in a fine-grained manner. This paper introduces Rationale-based Multiple Trait Scoring (RMTS), a novel approach for multi-trait essay scoring that integrates prompt-engineering-based large language models (LLMs) with a fine-tuning-based essay scoring model using a smaller large language model (S-LLM). RMTS uses an LLM-based trait-wise rationale generation system where a separate LLM agent generates trait-specific rationales based on rubric guidelines, which the scoring model uses to accurately predict multi-trait scores. Extensive experiments on benchmark datasets, including ASAP, ASAP++, and Feedback Prize, show that RMTS significantly outperforms state-of-the-art models and vanilla S-LLMs in trait-specific scoring. By assisting quantitative assessment with fine-grained qualitative rationales, RMTS enhances the trait-wise reliability, providing partial explanations about essays. The code is available at https://github.com/BBeeChu/RMTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00165v3">TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to WWW 2025 Research Track
    </div>
    <details class="paper-abstract">
      Hierarchical text classification aims to categorize each document into a set of classes in a label taxonomy, which is a fundamental web text mining task with broad applications such as web content analysis and semantic indexing. Most earlier works focus on fully or semi-supervised methods that require a large amount of human annotated data which is costly and time-consuming to acquire. To alleviate human efforts, in this paper, we work on hierarchical text classification with a minimal amount of supervision: using the sole class name of each node as the only supervision. Recently, large language models (LLM) have shown competitive performance on various tasks through zero-shot prompting, but this method performs poorly in the hierarchical setting because it is ineffective to include the large and structured label space in a prompt. On the other hand, previous weakly-supervised hierarchical text classification methods only utilize the raw taxonomy skeleton and ignore the rich information hidden in the text corpus that can serve as additional class-indicative features. To tackle the above challenges, we propose TELEClass, Taxonomy Enrichment and LLM-Enhanced weakly-supervised hierarchical text Classification, which combines the general knowledge of LLMs and task-specific features mined from an unlabeled corpus. TELEClass automatically enriches the raw taxonomy with class-indicative features for better label space understanding and utilizes novel LLM-based data annotation and generation methods specifically tailored for the hierarchical setting. Experiments show that TELEClass can significantly outperform previous baselines while achieving comparable performance to zero-shot prompting of LLMs with drastically less inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11629v6">Can Many-Shot In-Context Learning Help LLMs as Evaluators? A Preliminary Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Utilizing Large Language Models (LLMs) as evaluators to assess the performance of LLMs has garnered attention. However, this kind of evaluation approach is affected by potential biases within LLMs, raising concerns about the accuracy and reliability of the evaluation results of LLMs. To address this problem, we propose and study two many-shot In-Context Learning (ICL) prompt templates to help LLM evaluators mitigate potential biases: Many-Shot with Reference (MSwR) and Many-Shot without Reference (MSoR). Specifically, the former utilizes in-context examples with model-generated evaluation rationales as references, while the latter does not include these references. Using these prompt designs, we investigate the impact of increasing the number of in-context examples on the consistency and quality of the evaluation results. Experimental results show that advanced LLMs, such as GPT-4o, perform better in the many-shot regime than in the zero-shot and few-shot regimes. Furthermore, when using GPT-4o as an evaluator in the many-shot regime, adopting MSwR as the prompt template performs better than MSoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02909v1">SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We propose SPARC, a lightweight continual learning framework for large language models (LLMs) that enables efficient task adaptation through prompt tuning in a lower-dimensional space. By leveraging principal component analysis (PCA), we identify a compact subspace of the training data. Optimizing prompts in this lower-dimensional space enhances training efficiency, as it focuses updates on the most relevant features while reducing computational overhead. Furthermore, since the model's internal structure remains unaltered, the extensive knowledge gained from pretraining is fully preserved, ensuring that previously learned information is not compromised during adaptation. Our method achieves high knowledge retention in both task-incremental and domain-incremental continual learning setups while fine-tuning only 0.04% of the model's parameters. Additionally, by integrating LoRA, we enhance adaptability to computational constraints, allowing for a tradeoff between accuracy and training cost. Experiments on the SuperGLUE benchmark demonstrate that our PCA-based prompt tuning combined with LoRA maintains full knowledge retention while improving accuracy, utilizing only 1% of the model's parameters. These results establish our approach as a scalable and resource-efficient solution for continual learning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02896v1">A Benchmark for the Detection of Metalinguistic Disagreements between LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 6 pages, 2 tables, to appear in Reham Alharbi, Jacopo de Berardinis, Paul Groth, Albert Mero\~no-Pe\~nuela, Elena Simperl, Valentina Tamma (eds.), ISWC 2024 Special Session on Harmonising Generative AI and Semantic Web Technologies. CEUR-WS.org (forthcoming), for associated code and data see https://github.com/bradleypallen/trex-metalinguistic-disagreement
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for tasks like fact extraction in support of knowledge graph construction frequently involves computing accuracy metrics using a ground truth benchmark based on a knowledge graph (KG). These evaluations assume that errors represent factual disagreements. However, human discourse frequently features metalinguistic disagreement, where agents differ not on facts but on the meaning of the language used to express them. Given the complexity of natural language processing and generation using LLMs, we ask: do metalinguistic disagreements occur between LLMs and KGs? Based on an investigation using the T-REx knowledge alignment dataset, we hypothesize that metalinguistic disagreement does in fact occur between LLMs and KGs, with potential relevance for the practice of knowledge graph engineering. We propose a benchmark for evaluating the detection of factual and metalinguistic disagreements between LLMs and KGs. An initial proof of concept of such a benchmark is available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02893v1">Lowering the Barrier of Machine Learning: Achieving Zero Manual Labeling in Review Classification Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to 2025 11th International Conference on Computing and Artificial Intelligence (ICCAI 2025)
    </div>
    <details class="paper-abstract">
      With the internet's evolution, consumers increasingly rely on online reviews for service or product choices, necessitating that businesses analyze extensive customer feedback to enhance their offerings. While machine learning-based sentiment classification shows promise in this realm, its technical complexity often bars small businesses and individuals from leveraging such advancements, which may end up making the competitive gap between small and large businesses even bigger in terms of improving customer satisfaction. This paper introduces an approach that integrates large language models (LLMs), specifically Generative Pre-trained Transformer (GPT) and Bidirectional Encoder Representations from Transformers (BERT)-based models, making it accessible to a wider audience. Our experiments across various datasets confirm that our approach retains high classification accuracy without the need for manual labeling, expert knowledge in tuning and data annotation, or substantial computational power. By significantly lowering the barriers to applying sentiment classification techniques, our methodology enhances competitiveness and paves the way for making machine learning technology accessible to a broader audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06540v4">Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Scaling laws for large language models (LLMs) predict model performance based on parameters like size and training data. However, differences in training configurations and data processing across model families lead to significant variations in benchmark performance, making it difficult for a single scaling law to generalize across all LLMs. On the other hand, training family-specific scaling laws requires training models of varying sizes for every family. In this work, we propose Skills Scaling Laws (SSLaws, pronounced as Sloth), a novel scaling law that leverages publicly available benchmark data and assumes LLM performance is driven by low-dimensional latent skills, such as reasoning and instruction following. These latent skills are influenced by computational resources like model size and training tokens but with varying efficiencies across model families. Sloth exploits correlations across benchmarks to provide more accurate and interpretable predictions while alleviating the need to train multiple LLMs per family. We present both theoretical results on parameter identification and empirical evaluations on 12 prominent benchmarks, from Open LLM Leaderboard v1/v2, demonstrating that Sloth predicts LLM performance efficiently and offers insights into scaling behaviors for complex downstream tasks and increased test-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01991v2">Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)
    </div>
    <details class="paper-abstract">
      Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks, such as identifying morality frames, make relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02818v1">Accessible and Portable LLM Inference by Compiling Computational Graphs into SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) often demands specialized hardware, dedicated frameworks, and substantial development efforts, which restrict their accessibility, especially for edge devices and organizations with limited technical resources. We propose a novel compiler that translates LLM inference graphs into SQL queries, enabling relational databases, one of the most widely used and mature software systems globally, to serve as the runtime. By mapping neural operators such as matrix multiplication and attention into relational primitives like joins and aggregations, our approach leverages database capabilities, including disk-based data management and native caching. Supporting key transformer components, such as attention mechanisms and key-value caching, our system generates SQL pipelines for end-to-end LLM inference. Using the Llama3 family as a case study, we demonstrate up to 30x speedup in token generation for memory-constrained scenarios comparable to competitive CPU-based frameworks. Our work offers an accessible, portable, and efficient solution, facilitating the serving of LLMs across diverse deployment environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02810v1">Mol-LLM: Generalist Molecular LLM with Improved Graph Utilization</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have motivated the development of general LLMs for molecular tasks. While several studies have demonstrated that fine-tuned LLMs can achieve impressive benchmark performances, they are far from genuine generalist molecular LLMs due to a lack of fundamental understanding of molecular structure. Specifically, when given molecular task instructions, LLMs trained with naive next-token prediction training assign similar likelihood scores to both original and negatively corrupted molecules, revealing their lack of molecular structure understanding that is crucial for reliable and general molecular LLMs. To overcome this limitation and obtain a true generalist molecular LLM, we introduce a novel multi-modal training method based on a thorough multi-modal instruction tuning as well as a molecular structure preference optimization between chosen and rejected graphs. On various molecular benchmarks, the proposed generalist molecular LLM, called Mol-LLM, achieves state-of-the-art performances among generalist LLMs on most tasks, at the same time, surpassing or comparable to state-of-the-art specialist LLMs. Moreover, Mol-LLM also shows superior generalization performances in reaction prediction tasks, demonstrating the effect of the molecular structure understanding for generalization perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02794v1">METAMON: Finding Inconsistencies between Program Documentation and Behavior using Metamorphic LLM Queries</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 8 pages and 7 figures, accepted to LLM4Code 2025
    </div>
    <details class="paper-abstract">
      Code documentation can, if written precisely, help developers better understand the code they accompany. However, unlike code, code documentation cannot be automatically verified via execution, potentially leading to inconsistencies between documentation and the actual behavior. While such inconsistencies can be harmful for the developer's understanding of the code, checking and finding them remains a costly task due to the involvement of human engineers. This paper proposes METAMON, which uses an existing search-based test generation technique to capture the current program behavior in the form of test cases, and subsequently uses LLM-based code reasoning to identify the generated regression test oracles that are not consistent with the program specifications in the documentation. METAMON is supported in this task by metamorphic testing and self-consistency. An empirical evaluation against 9,482 pairs of code documentation and code snippets, generated using five open-source projects from Defects4J v2.0.1, shows that METAMON can classify the code-and-documentation inconsistencies with a precision of 0.72 and a recall of 0.48.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02790v1">Leveraging the true depth of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled without impacting performance significantly, these findings have not been employed to reduce the computational cost of inference. We investigate several potential ways to reduce the depth of pre-trained LLMs without significantly affecting performance. Leveraging our insights, we present a novel approach that exploits this decoupling between layers by grouping some of them into pairs that can be evaluated in parallel. This modification of the computational graph -- through better parallelism -- results in an average improvement of around 1.20x on the number of tokens generated per second, without re-training nor fine-tuning, while retaining 95%-99% of the original accuracy. Empirical evaluation demonstrates that this approach significantly improves serving efficiency while maintaining model performance, offering a practical improvement for large-scale LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15483v3">Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Post-training of pre-trained LLMs, which typically consists of the supervised fine-tuning (SFT) stage and the preference learning (RLHF or DPO) stage, is crucial to effective and safe LLM applications. The widely adopted approach in post-training popular open-source LLMs is to sequentially perform SFT and RLHF/DPO. However, sequential training is sub-optimal in terms of SFT and RLHF/DPO trade-off: the LLM gradually forgets about the first stage's training when undergoing the second stage's training. We theoretically prove the sub-optimality of sequential post-training. Furthermore, we propose a practical joint post-training framework with theoretical convergence guarantees and empirically outperforms sequential post-training framework, while having similar computational cost. Our code is available at https://github.com/heshandevaka/XRIGHT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03647v1">Looking for the Inner Music: Probing LLMs' Understanding of Literary Style</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated that language models can be trained to identify the author of much shorter literary passages than has been thought feasible for traditional stylometry. We replicate these results for authorship and extend them to a new dataset measuring novel genre. We find that LLMs are able to distinguish authorship and genre, but they do so in different ways. Some models seem to rely more on memorization, while others benefit more from training to learn author/genre characteristics. We then use three methods to probe one high-performing LLM for features that define style. These include direct syntactic ablations to input text as well as two methods that look at model internals. We find that authorial style is easier to define than genre-level style and is more impacted by minor syntactic decisions and contextual word usage. However, some traits like pronoun usage and word order prove significant for defining both kinds of literary style.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15676v3">Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been a widely adopted prompting method, eliciting impressive reasoning abilities of Large Language Models (LLMs). Inspired by the sequential thought structure of CoT, a number of Chain-of-X (CoX) methods have been developed to address various challenges across diverse domains and tasks involving LLMs. In this paper, we provide a comprehensive survey of Chain-of-X methods for LLMs in different contexts. Specifically, we categorize them by taxonomies of nodes, i.e., the X in CoX, and application tasks. We also discuss the findings and implications of existing CoX methods, as well as potential future directions. Our survey aims to serve as a detailed and up-to-date resource for researchers seeking to apply the idea of CoT to broader scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17406v2">ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      In cybersecurity, security analysts face the challenge of mitigating newly discovered vulnerabilities in real-time, with over 300,000 Common Vulnerabilities and Exposures (CVEs) identified since 1999. The sheer volume of known vulnerabilities complicates the detection of patterns for unknown threats. While LLMs can assist, they often hallucinate and lack alignment with recent threats. Over 25,000 vulnerabilities have been identified so far in 2024, which are introduced after popular LLMs' (e.g., GPT-4) training data cutoff. This raises a major challenge of leveraging LLMs in cybersecurity, where accuracy and up-to-date information are paramount. In this work, we aim to improve the adaptation of LLMs in vulnerability analysis by mimicking how analysts perform such tasks. We propose ProveRAG, an LLM-powered system designed to assist in rapidly analyzing CVEs with automated retrieval augmentation of web data while self-evaluating its responses with verifiable evidence. ProveRAG incorporates a self-critique mechanism to help alleviate omission and hallucination common in the output of LLMs applied in cybersecurity applications. The system cross-references data from verifiable sources (NVD and CWE), giving analysts confidence in the actionable insights provided. Our results indicate that ProveRAG excels in delivering verifiable evidence to the user with over 99% and 97% accuracy in exploitation and mitigation strategies, respectively. This system outperforms direct prompting and chunking retrieval in vulnerability analysis by overcoming temporal and context-window limitations. ProveRAG guides analysts to secure their systems more effectively while documenting the process for future audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03604v1">Bilevel ZOFO: Bridging Parameter-Efficient and Zeroth-Order Techniques for Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning(PEFT) methods have been proposed to address these challenges by freezing most model parameters and training only a small subset. While PEFT is efficient, it may not outperform full fine-tuning when high task-specific performance is required. Zeroth-Order (ZO) methods offer an alternative for fine-tuning the entire pre-trained model by approximating gradients using only the forward pass, thus eliminating the computational burden of back-propagation in first-order methods. However, when implementing ZO methods, a hard prompt is crucial, and relying on simple, fixed hard prompts may not be optimal. In this paper, we propose a bilevel optimization framework that complements ZO methods with PEFT to mitigate sensitivity to hard prompts while efficiently and effectively fine-tuning LLMs. Our Bilevel ZOFO (Zeroth-Order-First-Order) method employs a double-loop optimization strategy, where only the gradient of the PEFT model and the forward pass of the base model are required. We provide convergence guarantees for Bilevel ZOFO. Empirically, we demonstrate that Bilevel ZOFO outperforms both PEFT and ZO methods in single-task settings while maintaining similar memory efficiency. Additionally, we show its strong potential for multitask learning. Compared to current first-order meta-training algorithms for multitask learning, our method has significantly lower computational demands while maintaining or improving performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03589v1">HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Disaggregated Large Language Model (LLM) inference has gained popularity as it separates the computation-intensive prefill stage from the memory-intensive decode stage, avoiding the prefill-decode interference and improving resource utilization. However, transmitting Key-Value (KV) data between the two stages can be a bottleneck, especially for long prompts. Additionally, the computation time overhead for prefill and decode is key for optimizing Job Completion Time (JCT), and KV data size can become prohibitive for long prompts and sequences. Existing KV quantization methods can alleviate the transmission bottleneck and reduce memory requirements, but they introduce significant dequantization overhead, exacerbating the computation time. We propose Homomorphic Acceleration via Compression of the KV cache (HACK) for disaggregated LLM inference. HACK eliminates the heavy KV dequantization step, and directly performs computations on quantized KV data to approximate and reduce the cost of the expensive matrix-multiplication step. Extensive trace-driven experiments show that HACK reduces JCT by up to 70.9% compared to disaggregated LLM inference baseline and by up to 52.3% compared to state-of-the-art KV quantization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03579v1">A Mixed-Methods Evaluation of LLM-Based Chatbots for Menopause</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into healthcare settings has gained significant attention, particularly for question-answering tasks. Given the high-stakes nature of healthcare, it is essential to ensure that LLM-generated content is accurate and reliable to prevent adverse outcomes. However, the development of robust evaluation metrics and methodologies remains a matter of much debate. We examine the performance of publicly available LLM-based chatbots for menopause-related queries, using a mixed-methods approach to evaluate safety, consensus, objectivity, reproducibility, and explainability. Our findings highlight the promise and limitations of traditional evaluation metrics for sensitive health topics. We propose the need for customized and ethically grounded evaluation frameworks to assess LLMs to advance safe and effective use in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04390v1">In Praise of Stubbornness: The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Despite remarkable capabilities, large language models (LLMs) struggle to continually update their knowledge without catastrophic forgetting. In contrast, humans effortlessly integrate new information, detect conflicts with existing beliefs, and selectively update their mental models. This paper introduces a cognitive-inspired investigation paradigm to study continual knowledge updating in LLMs. We implement two key components inspired by human cognition: (1) Dissonance and Familiarity Awareness, analyzing model behavior to classify information as novel, familiar, or dissonant; and (2) Targeted Network Updates, which track neural activity to identify frequently used (stubborn) and rarely used (plastic) neurons. Through carefully designed experiments in controlled settings, we uncover a number of empirical findings demonstrating the potential of this approach. First, dissonance detection is feasible using simple activation and gradient features, suggesting potential for cognitive-inspired training. Second, we find that non-dissonant updates largely preserve prior knowledge regardless of targeting strategy, revealing inherent robustness in LLM knowledge integration. Most critically, we discover that dissonant updates prove catastrophically destructive to the model's knowledge base, indiscriminately affecting even information unrelated to the current updates. This suggests fundamental limitations in how neural networks handle contradictions and motivates the need for new approaches to knowledge updating that better mirror human cognitive mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04387v1">FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Federated learning (FL) has enabled the training of multilingual large language models (LLMs) on diverse and decentralized multilingual data, especially on low-resource languages. To improve client-specific performance, personalization via the use of parameter-efficient fine-tuning (PEFT) modules such as LoRA is common. This involves a personalization strategy (PS), such as the design of the PEFT adapter structures (e.g., in which layers to add LoRAs and what ranks) and choice of hyperparameters (e.g., learning rates) for fine-tuning. Instead of manual PS configuration, we propose FedP$^2$EFT, a federated learning-to-personalize method for multilingual LLMs in cross-device FL settings. Unlike most existing PEFT structure selection methods, which are prone to overfitting low-data regimes, FedP$^2$EFT collaboratively learns the optimal personalized PEFT structure for each client via Bayesian sparse rank selection. Evaluations on both simulated and real-world multilingual FL benchmarks demonstrate that FedP$^2$EFT largely outperforms existing personalized fine-tuning methods, while complementing a range of existing FL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v1">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 26 pages, 15 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this paper, we study the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations for both inter- and intra-diversity. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-development for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04376v1">MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      In contemporary workplaces, meetings are essential for exchanging ideas and ensuring team alignment but often face challenges such as time consumption, scheduling conflicts, and inefficient participation. Recent advancements in Large Language Models (LLMs) have demonstrated their strong capabilities in natural language generation and reasoning, prompting the question: can LLMs effectively delegate participants in meetings? To explore this, we develop a prototype LLM-powered meeting delegate system and create a comprehensive benchmark using real meeting transcripts. Our evaluation reveals that GPT-4/4o maintain balanced performance between active and cautious engagement strategies. In contrast, Gemini 1.5 Pro tends to be more cautious, while Gemini 1.5 Flash and Llama3-8B/70B display more active tendencies. Overall, about 60\% of responses address at least one key point from the ground-truth. However, improvements are needed to reduce irrelevant or repetitive content and enhance tolerance for transcription errors commonly found in real-world settings. Additionally, we implement the system in practical settings and collect real-world feedback from demos. Our findings underscore the potential and challenges of utilizing LLMs as meeting delegates, offering valuable insights into their practical application for alleviating the burden of meetings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04362v1">LLMs can be easily Confused by Instructional Distractions</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Despite the fact that large language models (LLMs) show exceptional skill in instruction following tasks, this strength can turn into a vulnerability when the models are required to disregard certain instructions. Instruction-following tasks typically involve a clear task description and input text containing the target data to be processed. However, when the input itself resembles an instruction, confusion may arise, even if there is explicit prompting to distinguish between the task instruction and the input. We refer to this phenomenon as instructional distraction. In this paper, we introduce a novel benchmark, named DIM-Bench, specifically designed to assess LLMs' performance under instructional distraction. The benchmark categorizes real-world instances of instructional distraction and evaluates LLMs across four instruction tasks: rewriting, proofreading, translation, and style transfer -- alongside five input tasks: reasoning, code generation, mathematical reasoning, bias detection, and question answering. Our experimental results reveal that even the most advanced LLMs are susceptible to instructional distraction, often failing to accurately follow user intent in such cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05223v1">KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Jailbreak attacks exploit specific prompts to bypass LLM safeguards, causing the LLM to generate harmful, inappropriate, and misaligned content. Current jailbreaking methods rely heavily on carefully designed system prompts and numerous queries to achieve a single successful attack, which is costly and impractical for large-scale red-teaming. To address this challenge, we propose to distill the knowledge of an ensemble of SOTA attackers into a single open-source model, called Knowledge-Distilled Attacker (KDA), which is finetuned to automatically generate coherent and diverse attack prompts without the need for meticulous system prompt engineering. Compared to existing attackers, KDA achieves higher attack success rates and greater cost-time efficiency when targeting multiple SOTA open-source and commercial black-box LLMs. Furthermore, we conducted a quantitative diversity analysis of prompts generated by baseline methods and KDA, identifying diverse and ensemble attacks as key factors behind KDA's effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02542v1">OVERTHINKING: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 46x slowdown and high transferability of the attack across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third party applications operating reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02539v1">LLMs for Generation of Architectural Components: An Exploratory Empirical Study in the Serverless World</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Software Architecture (ICSA) 2025 Main Track (https://conf.researchr.org/home/icsa-2025)
    </div>
    <details class="paper-abstract">
      Recently, the exponential growth in capability and pervasiveness of Large Language Models (LLMs) has led to significant work done in the field of code generation. However, this generation has been limited to code snippets. Going one step further, our desideratum is to automatically generate architectural components. This would not only speed up development time, but would also enable us to eventually completely skip the development phase, moving directly from design decisions to deployment. To this end, we conduct an exploratory study on the capability of LLMs to generate architectural components for Functions as a Service (FaaS), commonly known as serverless functions. The small size of their architectural components make this architectural style amenable for generation using current LLMs compared to other styles like monoliths and microservices. We perform the study by systematically selecting open source serverless repositories, masking a serverless function and utilizing state of the art LLMs provided with varying levels of context information about the overall system to generate the masked function. We evaluate correctness through existing tests present in the repositories and use metrics from the Software Engineering (SE) and Natural Language Processing (NLP) domains to evaluate code quality and the degree of similarity between human and LLM generated code respectively. Along with our findings, we also present a discussion on the path forward for using GenAI in architectural component generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02534v1">Adaptive Self-improvement LLM Agentic System for ML Library Development</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      ML libraries, often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures, are key to efficient ML systems. However, writing these high-performance ML libraries is challenging because it requires expert knowledge of ML algorithms and the ASPL. Large language models (LLMs), on the other hand, have shown general coding capabilities. However, challenges remain when using LLMs for generating ML libraries using ASPLs because 1) this task is complicated even for experienced human programmers and 2) there are limited code examples because of the esoteric and evolving nature of ASPLs. Therefore, LLMs need complex reasoning with limited data in order to complete this task. To address these challenges, we introduce an adaptive self-improvement agentic system. In order to evaluate the effectiveness of our system, we construct a benchmark of a typical ML library and generate ASPL code with both open and closed-source LLMs on this benchmark. Our results show improvements of up to $3.9\times$ over a baseline single LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02508v1">Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models will be fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18626v3">The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies. Warning: this paper contains examples of unethical inquiries used solely for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v2">Internal Activation as the Polar Star for Steering Unsafe LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities across a wide range of tasks but also pose significant risks due to their potential to generate harmful content. Although existing safety mechanisms can improve model safety, they often lead to overly cautious behavior and fail to fully utilize LLMs' internal cognitive processes. Drawing inspiration from cognitive science, where humans rely on reflective reasoning (System 2 thinking) to regulate language and behavior, we empirically demonstrate that LLMs also possess a similar capacity for internal assessment and regulation, which can be actively detected. Building on this insight, we introduce SafeSwitch, a framework that dynamically regulates unsafe outputs by monitoring and utilizing the model's internal states. Our empirical results show that SafeSwitch reduces harmful outputs by over 80% on safety benchmarks while maintaining strong utility. Compared to traditional safety alignment methods, SafeSwitch delivers more informative and context-aware refusals, demonstrates resilience to unseen queries, and achieves these benefits while only tuning less than 6% of the original parameters. These features make SafeSwitch a promising approach for implementing nuanced safety controls in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00903v2">Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      This study attempts to advancing content analysis methodology from consensus-oriented to coordination-oriented practices, thereby embracing diverse coding outputs and exploring the dynamics among differential perspectives. As an exploratory investigation of this approach, we evaluate six GPT-4o configurations to analyze sentiment in Fox News and MSNBC transcripts on Biden and Trump during the 2020 U.S. presidential campaign, examining patterns across these models. By assessing each model's alignment with ideological perspectives, we explore how partisan selective processing could be identified in LLM-Assisted Content Analysis (LACA). Findings reveal that partisan persona LLMs exhibit stronger ideological biases when processing politically congruent content. Additionally, intercoder reliability is higher among same-partisan personas compared to cross-partisan pairs. This approach enhances the nuanced understanding of LLM outputs and advances the integrity of AI-driven social science research, enabling simulations of real-world implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12091v3">Is poisoning a real threat to LLM alignment? Maybe more so than you think</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01618v2">A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code and further information is available at https://probabilistic-inference-scaling.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19297v2">Analysis of LLMs vs Human Experts in Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 8 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The majority of research around Large Language Models (LLM) application to software development has been on the subject of code generation. There is little literature on LLMs' impact on requirements engineering (RE), which deals with the process of developing and verifying the system requirements. Within RE, there is a subdiscipline of requirements elicitation, which is the practice of discovering and documenting requirements for a system from users, customers, and other stakeholders. In this analysis, we compare LLM's ability to elicit requirements of a software system, as compared to that of a human expert in a time-boxed and prompt-boxed study. We found LLM-generated requirements were evaluated as more aligned (+1.12) than human-generated requirements with a trend of being more complete (+10.2%). Conversely, we found users tended to believe that solutions they perceived as more aligned had been generated by human experts. Furthermore, while LLM-generated documents scored higher and performed at 720x the speed, their cost was, on average, only 0.06% that of a human expert. Overall, these findings indicate that LLMs will play an increasingly important role in requirements engineering by improving requirements definitions, enabling more efficient resource allocation, and reducing overall project timelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02412v1">AI-Powered, But Power-Hungry? Energy Efficiency of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in software development to assist in various tasks, e.g., code generation and code completion, but empirical evaluations of the quality of the results produced by these models focus on correctness and ignore other relevant aspects, such as their performance and energy efficiency. Studying the performance of LLM-produced programs is essential to understand how well LLMs can support the construction of performance- and energy-critical software, such as operating systems, servers, and mobile applications. This paper presents the first study analyzing the energy efficiency and performance of LLM-generated code for three programming languages Python, Java, and C++, on two platforms, a Mac and a PC, leveraging three frontier LLMs, Github Copilot, GPT-4o, and the recently-released OpenAI o1-mini, and targeting ``hard'' programming problems from LeetCode. Our results show that the models are much more successful in generating Python and Java than C++ code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07668v3">Towards Evaluation Guidelines for Empirical Studies involving LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 4 pages, 2nd IEEE/ACM International Workshop on Methodological Issues with Empirical Studies in Software Engineering (WSESE 2025)
    </div>
    <details class="paper-abstract">
      In the short period since the release of ChatGPT, large language models (LLMs) have changed the software engineering research landscape. While there are numerous opportunities to use LLMs for supporting research or software engineering tasks, solid science needs rigorous empirical evaluations. However, so far, there are no specific guidelines for conducting and assessing studies involving LLMs in software engineering research. Our focus is on empirical studies that either use LLMs as part of the research process or studies that evaluate existing or new tools that are based on LLMs. This paper contributes the first set of holistic guidelines for such studies. Our goal is to start a discussion in the software engineering research community to reach a common understanding of our standards for high-quality empirical studies involving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02368v1">Evaluating the Effectiveness of LLMs in Fixing Maintainability Issues in Real-World Projects</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained attention for addressing coding problems, but their effectiveness in fixing code maintainability remains unclear. This study evaluates LLMs capability to resolve 127 maintainability issues from 10 GitHub repositories. We use zero-shot prompting for Copilot Chat and Llama 3.1, and few-shot prompting with Llama only. The LLM-generated solutions are assessed for compilation errors, test failures, and new maintainability problems. Llama with few-shot prompting successfully fixed 44.9% of the methods, while Copilot Chat and Llama zero-shot fixed 32.29% and 30%, respectively. However, most solutions introduced errors or new maintainability issues. We also conducted a human study with 45 participants to evaluate the readability of 51 LLM-generated solutions. The human study showed that 68.63% of participants observed improved readability. Overall, while LLMs show potential for fixing maintainability issues, their introduction of errors highlights their current limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v1">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01600v2">Reinforcement Learning for Long-Horizon Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive LOOP, a data- and memory-efficient variant of proximal policy optimization. LOOP uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with LOOP in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02342v1">SHIELD: APT Detection and Intelligent Explanation Using LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Advanced persistent threats (APTs) are sophisticated cyber attacks that can remain undetected for extended periods, making their mitigation particularly challenging. Given their persistence, significant effort is required to detect them and respond effectively. Existing provenance-based attack detection methods often lack interpretability and suffer from high false positive rates, while investigation approaches are either supervised or limited to known attacks. To address these challenges, we introduce SHIELD, a novel approach that combines statistical anomaly detection and graph-based analysis with the contextual analysis capabilities of large language models (LLMs). SHIELD leverages the implicit knowledge of LLMs to uncover hidden attack patterns in provenance data, while reducing false positives and providing clear, interpretable attack descriptions. This reduces analysts' alert fatigue and makes it easier for them to understand the threat landscape. Our extensive evaluation demonstrates SHIELD's effectiveness and computational efficiency in real-world scenarios. SHIELD was shown to outperform state-of-the-art methods, achieving higher precision and recall. SHIELD's integration of anomaly detection, LLM-driven contextual analysis, and advanced graph-based correlation establishes a new benchmark for APT detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02337v1">Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00029v2">AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Financial metrics like the Sharpe ratio are pivotal in evaluating investment performance by balancing risk and return. However, traditional metrics often struggle with robustness and generalization, particularly in dynamic and volatile market conditions. This paper introduces AlphaSharpe, a novel framework leveraging large language models (LLMs) to iteratively evolve and optimize financial metrics to discover enhanced risk-return metrics that outperform traditional approaches in robustness and correlation with future performance metrics by employing iterative crossover, mutation, and evaluation. Key contributions of this work include: (1) a novel use of LLMs to generate and refine financial metrics with implicit domain-specific knowledge, (2) a scoring mechanism to ensure that evolved metrics generalize effectively to unseen data, and (3) an empirical demonstration of 3x predictive power for future risk-returns, and 2x portfolio performance. Experimental results in a real-world dataset highlight the superiority of discovered metrics, making them highly relevant to portfolio managers and financial decision-makers. This framework not only addresses the limitations of existing metrics but also showcases the potential of LLMs in advancing financial analytics, paving the way for informed and robust investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02329v1">ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Creating data reports is time-consuming, as it requires iterative exploration and understanding of data, followed by summarizing the insights. While large language models (LLMs) are powerful tools for data processing and text generation, they often struggle to produce complete data reports that fully meet user expectations. One significant challenge is effectively communicating the entire analysis logic to LLMs. Moreover, determining a comprehensive analysis logic can be mentally taxing for users. To address these challenges, we propose ReSpark, an LLM-based method that leverages existing data reports as references for creating new ones. Given a data table, ReSpark searches for similar-topic reports, parses them into interdependent segments corresponding to analytical objectives, and executes them with new data. It identifies inconsistencies and customizes the objectives, data transformations, and textual descriptions. ReSpark allows users to review real-time outputs, insert new objectives, and modify report content. Its effectiveness was evaluated through comparative and user studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14630v2">Extracting Problem Structure with LLMs for Optimized SAT Local Search</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Local search preprocessing makes Conflict-Driven Clause Learning (CDCL) solvers faster by providing high-quality starting points and modern SAT solvers have incorporated this technique into their preprocessing steps. However, these tools rely on basic strategies that miss the structural patterns in problems. We present a method that applies Large Language Models (LLMs) to analyze Python-based encoding code. This reveals hidden structural patterns in how problems convert into SAT. Our method automatically generates specialized local search algorithms that find these patterns and use them to create strong initial assignments. This works for any problem instance from the same encoding type. Our tests show encouraging results, achieving faster solving times compared to baseline preprocessing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02201v1">Can You Move These Over There? An LLM-based VR Mover for Supporting Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 64 pages (30 in main text), 22 figures (19 in main text)
    </div>
    <details class="paper-abstract">
      In our daily lives, we can naturally convey instructions for the spatial manipulation of objects using words and gestures. Transposing this form of interaction into virtual reality (VR) object manipulation can be beneficial. We propose VR Mover, an LLM-empowered solution that can understand and interpret the user's vocal instruction to support object manipulation. By simply pointing and speaking, the LLM can manipulate objects without structured input. Our user study demonstrates that VR Mover enhances user usability, overall experience and performance on multi-object manipulation, while also reducing workload and arm fatigue. Users prefer the proposed natural interface for broad movements and may complementarily switch to gizmos or virtual hands for finer adjustments. These findings are believed to contribute to design implications for future LLM-based object manipulation interfaces, highlighting the potential for more intuitive and efficient user interactions in VR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02199v1">When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable success in language modelling due to scaling laws found in model size and the hidden dimension of the model's text representation. Yet, we demonstrate that compressed representations of text can yield better performance in LLM-based regression tasks. In this paper, we compare the relative performance of embedding compression in three different signal-to-noise contexts: financial return prediction, writing quality assessment and review scoring. Our results show that compressing embeddings, in a minimally supervised manner using an autoencoder's hidden representation, can mitigate overfitting and improve performance on noisy tasks, such as financial return prediction; but that compression reduces performance on tasks that have high causal dependencies between the input and target data. Our results suggest that the success of interpretable compressed representations such as sentiment may be due to a regularising effect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18712v3">Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Fingerprinting refers to the process of identifying underlying Machine Learning (ML) models of AI Systemts, such as Large Language Models (LLMs), by analyzing their unique characteristics or patterns, much like a human fingerprint. The fingerprinting of Large Language Models (LLMs) has become essential for ensuring the security and transparency of AI-integrated applications. While existing methods primarily rely on access to direct interactions with the application to infer model identity, they often fail in real-world scenarios involving multi-agent systems, frequent model updates, and restricted access to model internals. In this paper, we introduce a novel fingerprinting framework designed to address these challenges by integrating static and dynamic fingerprinting techniques. Our approach identifies architectural features and behavioral traits, enabling accurate and robust fingerprinting of LLMs in dynamic environments. We also highlight new threat scenarios where traditional fingerprinting methods are ineffective, bridging the gap between theoretical techniques and practical application. To validate our framework, we present an extensive evaluation setup that simulates real-world conditions and demonstrate the effectiveness of our methods in identifying and monitoring LLMs in Gen-AI applications. Our results highlight the framework's adaptability to diverse and evolving deployment contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14202v2">Rationale Behind Essay Scores: Enhancing S-LLM's Multi-Trait Essay Scoring with Rationale Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Existing automated essay scoring (AES) has solely relied on essay text without using explanatory rationales for the scores, thereby forgoing an opportunity to capture the specific aspects evaluated by rubric indicators in a fine-grained manner. This paper introduces Rationale-based Multiple Trait Scoring (RMTS), a novel approach for multi-trait essay scoring that integrates prompt-engineering-based large language models (LLMs) with a fine-tuning-based essay scoring model using a smaller large language model (S-LLM). RMTS uses an LLM-based trait-wise rationale generation system where a separate LLM agent generates trait-specific rationales based on rubric guidelines, which the scoring model uses to accurately predict multi-trait scores. Extensive experiments on benchmark datasets, including ASAP, ASAP++, and Feedback Prize, show that RMTS significantly outperforms state-of-the-art models and vanilla S-LLMs in trait-specific scoring. By assisting quantitative assessment with fine-grained qualitative rationales, RMTS enhances the trait-wise reliability, providing partial explanations about essays. The code is available at https://github.com/BBeeChu/RMTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17178v2">Tuning LLM Judge Design Decisions for 1/1000 of the Cost</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune hyperparameter of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trades accuracy for cost and also reduce significantly the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19200v2">On Behalf of the Stakeholders: Trends in NLP Model Interpretability in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent advancements in NLP systems, particularly with the introduction of LLMs, have led to widespread adoption of these systems by a broad spectrum of users across various domains, impacting decision-making, the job market, society, and scientific research. This surge in usage has led to an explosion in NLP model interpretability and analysis research, accompanied by numerous technical surveys. Yet, these surveys often overlook the needs and perspectives of explanation stakeholders. In this paper, we address three fundamental questions: Why do we need interpretability, what are we interpreting, and how? By exploring these questions, we examine existing interpretability paradigms, their properties, and their relevance to different stakeholders. We further explore the practical implications of these paradigms by analyzing trends from the past decade across multiple research fields. To this end, we retrieved thousands of papers and employed an LLM to characterize them. Our analysis reveals significant disparities between NLP developers and non-developer users, as well as between research fields, underscoring the diverse needs of stakeholders. For example, explanations of internal model components are rarely used outside the NLP field. We hope this paper informs the future design, development, and application of methods that align with the objectives and requirements of various stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14175v2">QMOS: Enhancing LLMs for Telecommunication with Question Masked loss and Option Shuffling</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have brought about substantial advancements in the field of Question Answering (QA) systems. These models do remarkably well in addressing intricate inquiries in a variety of disciplines. However, because of domain-specific vocabulary, complex technological concepts, and the requirement for exact responses applying LLMs to specialized sectors like telecommunications presents additional obstacles. GPT-3.5 has been used in recent work, to obtain noteworthy accuracy for telecom-related questions in a Retrieval Augmented Generation (RAG) framework. Notwithstanding these developments, the practical use of models such as GPT-3.5 is restricted by their proprietary nature and high computing demands. This paper introduces QMOS, an innovative approach which uses a Question-Masked loss and Option Shuffling trick to enhance the performance of LLMs in answering Multiple-Choice Questions in the telecommunications domain. Our focus was on using opensource, smaller language models (Phi-2 and Falcon-7B) within an enhanced RAG framework. Our multi-faceted approach involves several enhancements to the whole LLM-RAG pipeline of finetuning, retrieval, prompt engineering and inference. Our approaches significantly outperform existing results, achieving accuracy improvements from baselines of 24.70% to 49.30% with Falcon-7B and from 42.07% to 84.65% with Phi-2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02067v1">AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
    </div>
    <details class="paper-abstract">
      Embodied agents assisting humans are often asked to complete a new task in a new scenario. An agent preparing a particular dish in the kitchen based on a known recipe may be asked to prepare a new dish or to perform cleaning tasks in the storeroom. There may not be sufficient resources, e.g., time or labeled examples, to train the agent for these new situations. Large Language Models (LLMs) trained on considerable knowledge across many domains are able to predict a sequence of abstract actions for such new tasks and scenarios, although it may not be possible for the agent to execute this action sequence due to task-, agent-, or domain-specific constraints. Our framework addresses these challenges by leveraging the generic predictions provided by LLM and the prior domain-specific knowledge encoded in a Knowledge Graph (KG), enabling an agent to quickly adapt to new tasks and scenarios. The robot also solicits and uses human input as needed to refine its existing knowledge. Based on experimental evaluation over cooking and cleaning tasks in simulation domains, we demonstrate that the interplay between LLM, KG, and human input leads to substantial performance gains compared with just using the LLM output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02066v1">Anticipate & Act : Integrating LLMs and Classical Planning for Efficient Task Execution in Household Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2024
    </div>
    <details class="paper-abstract">
      Assistive agents performing household tasks such as making the bed or cooking breakfast often compute and execute actions that accomplish one task at a time. However, efficiency can be improved by anticipating upcoming tasks and computing an action sequence that jointly achieves these tasks. State-of-the-art methods for task anticipation use data-driven deep networks and Large Language Models (LLMs), but they do so at the level of high-level tasks and/or require many training examples. Our framework leverages the generic knowledge of LLMs through a small number of prompts to perform high-level task anticipation, using the anticipated tasks as goals in a classical planning system to compute a sequence of finer-granularity actions that jointly achieve these goals. We ground and evaluate our framework's abilities in realistic scenarios in the VirtualHome environment and demonstrate a 31% reduction in execution time compared with a system that does not consider upcoming tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20242v4">BadRobot: Jailbreaking Embodied LLMs in the Physical World</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to ICLR 2025. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00212v2">STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 19.8 billion tokens generated during the training in Lean, STP proves 26.3% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (61.1%, pass@3200), Proofnet-test (23.1%, pass@3200) and PutnamBench (8/644, pass@64).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v2">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14304v2">MASTER: A Multi-Agent System with LLM Specialized MCTS</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted by main NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) are increasingly being explored for problem-solving tasks. However, their strategic planning capability is often viewed with skepticism. Recent studies have incorporated the Monte Carlo Tree Search (MCTS) algorithm to augment the planning capacity of LLM. Despite its potential, MCTS relies on extensive sampling simulations to approximate the true reward distribution, which leads to two primary issues. Firstly, MCTS is effective for tasks like the Game of Go, where simulation results can yield objective rewards (e.g., 1 for a win and 0 for a loss). However, for tasks such as question answering, the result of a simulation is the answer to the question, which cannot yield an objective reward without the ground truth. Secondly, obtaining statistically significant reward estimations typically requires a sample size exceeding 30 simulations, resulting in excessive token usage and time consumption. To address these challenges, we present the Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS (MASTER), a novel framework that coordinates agent recruitment and communication through LLM specialized MCTS. This system autonomously adjusts the number of agents based on task complexity and ensures focused communication among them. Comprehensive experiments across various tasks demonstrate the effectiveness of our proposed framework. It achieves 76% accuracy on HotpotQA and 80% on WebShop, setting new state-of-the-art performance on these datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v2">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 32 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but, even with domain-specific fine-tuning, often produce hallucinations for complex ones. While integrating LLMs with tools can mitigate this reliability issue, models finetuned on tool usage only often over-rely on them, incurring unnecessary costs from resource-intensive scientific tools even for simpler problems. Inspired by how human experts assess the complexity of the problem before choosing the solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tools-generated solutions. In the second component, Tool Usage Adaptation (TUA), we classify questions as easy or hard based on the WKL-trained model's accuracy, and train it to maintain direct reasoning for simple problems while switching to tools for challenging ones. We validate our method on 6 scientific benchmark datasets in climate science, epidemiology, and mathematics. Compared to the base 8B model, our trained models achieve 28.27% higher answer accuracy and 13.76% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4 and Claude-3.5 on 4 custom-created datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02009v1">LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Security misconfigurations in Container Orchestrators (COs) can pose serious threats to software systems. While Static Analysis Tools (SATs) can effectively detect these security vulnerabilities, the industry currently lacks automated solutions capable of fixing these misconfigurations. The emergence of Large Language Models (LLMs), with their proven capabilities in code understanding and generation, presents an opportunity to address this limitation. This study introduces LLMSecConfig, an innovative framework that bridges this gap by combining SATs with LLMs. Our approach leverages advanced prompting techniques and Retrieval-Augmented Generation (RAG) to automatically repair security misconfigurations while preserving operational functionality. Evaluation of 1,000 real-world Kubernetes configurations achieved a 94\% success rate while maintaining a low rate of introducing new misconfigurations. Our work makes a promising step towards automated container security management, reducing the manual effort required for configuration maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01992v1">FinRLlama: A Solution to LLM-Engineered Signals Challenge at FinRL Contest 2024</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Competition Track FinRL, ICAIF 2024
    </div>
    <details class="paper-abstract">
      In response to Task II of the FinRL Challenge at ACM ICAIF 2024, this study proposes a novel prompt framework for fine-tuning large language models (LLM) with Reinforcement Learning from Market Feedback (RLMF). Our framework incorporates market-specific features and short-term price dynamics to generate more precise trading signals. Traditional LLMs, while competent in sentiment analysis, lack contextual alignment for financial market applications. To bridge this gap, we fine-tune the LLaMA-3.2-3B-Instruct model using a custom RLMF prompt design that integrates historical market data and reward-based feedback. Our evaluation shows that this RLMF-tuned framework outperforms baseline methods in signal consistency and achieving tighter trading outcomes; awarded as winner of Task II. You can find the code for this project on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01991v1">Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)
    </div>
    <details class="paper-abstract">
      Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks such as identifying morality frames makes relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17017v3">Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Self-Consistency mitigates hallucinations in Large Language Models (LLMs) by sampling multiple reasoning paths,but it lacks a systematic approach to determine the optimal number of samples or select the most faithful rationale. To address this limitation, we introduce Reasoning-Aware Self-Consistency (RASC), a novel framework that enhances sampling efficiency and reasoning faithfulness by dynamically evaluating both outputs and rationales. RASC assesses the quality of reasoning and the consistency of answers for each generated sample, using these assessments to guide early stopping decisions and rationale selection. The framework employs criteria-based stopping and weighted majority voting, enabling more informed choices on when to halt sampling and which rationale to select. Our comprehensive experiments across diverse question-answering datasets demonstrate that RASC outperforms existing methods, reducing sample usage by approximately 70% while maintaining accuracy. Moreover, RASC facilitates the selection of high-fidelity rationales, thereby improving the faithfulness of LLM outputs. Our approach effectively addresses the efficiency-accuracy trade-off in LLM reasoning tasks, offering a new perspective for more nuanced, faithful, and effective utilization of LLMs in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01066v2">From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 15 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      LLMs when used with Retrieval Augmented Generation (RAG), are greatly improving the SOTA of translating natural language queries to structured and correct SQL. Unlike previous reviews, this survey provides a comprehensive study of the evolution of LLM-based text-to-SQL systems, from early rule-based models to advanced LLM approaches that use (RAG) systems. We discuss benchmarks, evaluation methods, and evaluation metrics. Also, we uniquely study the use of Graph RAGs for better contextual accuracy and schema linking in these systems. Finally, we highlight key challenges such as computational efficiency, model robustness, and data privacy toward improvements of LLM-based text-to-SQL systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01977v1">AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      User interface understanding with vision-language models has received much attention due to its potential for enabling next-generation software automation. However, existing UI datasets either only provide large-scale context-free element annotations or contextualized functional descriptions for elements at a much smaller scale. In this work, we propose the \methodname{} pipeline for automatically annotating UI elements with detailed functionality descriptions at scale. Specifically, we leverage large language models (LLMs) to infer element functionality by comparing the UI content changes before and after simulated interactions with specific UI elements. To improve annotation quality, we propose LLM-aided rejection and verification, eliminating invalid and incorrect annotations without human labor. We construct an \methodname{}-704k dataset using the proposed pipeline, featuring multi-resolution, multi-device screenshots, diverse data domains, and detailed functionality annotations that have never been provided by previous datasets. Human evaluation shows that the AutoGUI pipeline achieves annotation correctness comparable to trained human annotators. Extensive experimental results show that our \methodname{}-704k dataset remarkably enhances VLM's UI grounding capabilities, exhibits significant scaling effects, and outperforms existing web pre-training data types. We envision AutoGUI as a scalable pipeline for generating massive data to build GUI-oriented VLMs. AutoGUI dataset can be viewed at this anonymous URL: https://autogui-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01968v1">Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant or uninformative. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves performance across multiple downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.00948v3">Large Linguistic Models: Investigating LLMs' metalinguistic abilities</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that -- for the first time -- the models can also generate valid metalinguistic analyses of language data. We outline a research program where the behavioral interpretability of LLMs on these tasks is tested via prompting. LLMs are trained primarily on text -- as such, evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. We show that OpenAI's o1 vastly outperforms other models on tasks involving drawing syntactic trees and phonological generalization. We speculate that OpenAI o1's unique advantage over other models may result from the model's chain-of-thought mechanism, which mimics the structure of human reasoning used in complex cognitive tasks, such as linguistic analysis.
    </details>
</div>
