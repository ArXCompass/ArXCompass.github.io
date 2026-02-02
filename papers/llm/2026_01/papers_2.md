# llm - 2026_01

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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14038v6">OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 5 pages, 1 figure, 1 table, 1 code snippet
    </div>
    <details class="paper-abstract">
      Hallucinations of large language models (LLMs) commonly occur in domain-specific downstream tasks, with no exception in ontology matching (OM). The prevalence of using LLMs for OM raises the need for benchmarks to better understand LLM hallucinations. The OAEI-LLM dataset is an extended version of the Ontology Alignment Evaluation Initiative (OAEI) datasets that evaluate LLM-specific hallucinations in OM tasks. We outline the methodology used in dataset construction and schema extension, and provide examples of potential use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21824v1">DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Determinism is indispensable for reproducibility in large language model (LLM) training, yet it often exacts a steep performance cost. In widely used attention implementations such as FlashAttention-3, the deterministic backward pass can incur up to a 37.9% throughput reduction relative to its non-deterministic counterpart, primarily because gradient accumulation operations must be serialized to guarantee numerical consistency. This performance loss stems from suboptimal scheduling of compute and gradient-reduction phases, leading to significant hardware underutilization. To address this challenge, we formulate the backward pass of deterministic attention as a scheduling problem on a Directed Acyclic Graph (DAG) and derive schedules that minimize the critical path length. Building on this formulation, we present DASH (Deterministic Attention Scheduling for High-Throughput), which encapsulates two complementary scheduling strategies: (i) Descending Q-Tile Iteration, a reversed query-block traversal that shrinks pipeline stalls in causal attention, and (ii) Shift Scheduling, a theoretically optimal schedule within our DAG model that reduces pipeline stalls for both full and causal masks. Our empirical evaluations on NVIDIA H800 GPUs demonstrate that DASH narrows the performance gap of deterministic attention. The proposed strategies improve the throughput of the attention backward pass by up to 1.28$\times$ compared to the baseline, significantly advancing the efficiency of reproducible LLM training. Our code is open-sourced at https://github.com/SJTU-Liquid/deterministic-FA3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18352v2">LLM-based Few-Shot Early Rumor Detection with Imitation Agent</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at KDD 2026
    </div>
    <details class="paper-abstract">
      Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21816v1">Nonparametric LLM Evaluation from Preference Data</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating the performance of large language models (LLMs) from human preference data is crucial for obtaining LLM leaderboards. However, many existing approaches either rely on restrictive parametric assumptions or lack valid uncertainty quantification when flexible machine learning methods are used. In this paper, we propose a nonparametric statistical framework, DMLEval, for comparing and ranking LLMs from preference data using debiased machine learning (DML). For this, we introduce generalized average ranking scores (GARS), which generalize commonly used ranking models, including the Bradley-Terry model or PageRank/ Rank centrality, with complex human responses such as ties. DMLEval comes with the following advantages: (i) It produces statistically efficient estimates of GARS ranking scores. (ii) It naturally allows the incorporation of black-box machine learning methods for estimation. (iii) It can be combined with pre-trained LLM evaluators (e.g., using LLM-as-a-judge). (iv) It suggests optimal policies for collecting preference data under budget constraints. We demonstrate these advantages both theoretically and empirically using both synthetic and real-world preference datasets. In summary, our framework provides practitioners with powerful, state-of-the-art methods for comparing or ranking LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21802v1">A Unified XAI-LLM Approach for EndotrachealSuctioning Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Endotracheal suctioning (ES) is an invasive yet essential clinical procedure that requires a high degree of skill to minimize patient risk - particularly in home care and educational settings, where consistent supervision may be limited. Despite its critical importance, automated recognition and feedback systems for ES training remain underexplored. To address this gap, this study proposes a unified, LLM-centered framework for video-based activity recognition benchmarked against conventional machine learning and deep learning approaches, and a pilot study on feedback generation. Within this framework, the Large Language Model (LLM) serves as the central reasoning module, performing both spatiotemporal activity recognition and explainable decision analysis from video data. Furthermore, the LLM is capable of verbalizing feedback in natural language, thereby translating complex technical insights into accessible, human-understandable guidance for trainees. Experimental results demonstrate that the proposed LLM-based approach outperforms baseline models, achieving an improvement of approximately 15-20\% in both accuracy and F1 score. Beyond recognition, the framework incorporates a pilot student-support module built upon anomaly detection and explainable AI (XAI) principles, which provides automated, interpretable feedback highlighting correct actions and suggesting targeted improvements. Collectively, these contributions establish a scalable, interpretable, and data-driven foundation for advancing nursing education, enhancing training efficiency, and ultimately improving patient safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21758v1">EWSJF: An Adaptive Scheduler with Hybrid Partitioning for Mixed-Workload LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) under mixed workloads--short, latency-sensitive interactive queries alongside long, throughput-oriented batch requests--poses a fundamental scheduling challenge. Standard First-Come, First-Served (FCFS) policies suffer from severe head-of-line blocking, leading to high tail latency and underutilized hardware. We introduce EWSJF (Effective Workload-based Shortest Job First), an adaptive request-level scheduler that learns workload structure in real time to jointly improve fairness and throughput. EWSJF operates upstream of execution-level schedulers and integrates four components: (1) Refine-and-Prune, an unsupervised partitioning algorithm that discovers performance-homogeneous request groups; (2) Dynamic Queue Routing for assigning requests to these groups; (3) Density-Weighted Scoring, a context-aware prioritization function balancing urgency and fairness; and (4) Bayesian Meta-Optimization, which continuously tunes scoring and partitioning parameters based on live performance feedback. Implemented in vLLM, EWSJF improves end-to-end throughput by over 30% and reduces average Time-To-First-Token for short requests by up to 4x compared to FCFS. These results demonstrate that adaptive, learning-based request scheduling is a critical missing layer for efficient and responsive LLM serving. Implementation available at https://anonymous.4open.science/r/vllm_0110-32D8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v25">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 31 pages - VLDB 2025 (Page 1-20), OM 2025 (Page 21-31)
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21742v1">Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Codes and data are available at https://github.com/skyriver-2000/epistemic-context-learning
    </div>
    <details class="paper-abstract">
      Individual agents in multi-agent (MA) systems often lack robustness, tending to blindly conform to misleading peers. We show this weakness stems from both sycophancy and inadequate ability to evaluate peer reliability. To address this, we first formalize the learning problem of history-aware reference, introducing the historical interactions of peers as additional input, so that agents can estimate peer reliability and learn from trustworthy peers when uncertain. This shifts the task from evaluating peer reasoning quality to estimating peer reliability based on interaction history. We then develop Epistemic Context Learning (ECL): a reasoning framework that conditions predictions on explicitly-built peer profiles from history. We further optimize ECL by reinforcement learning using auxiliary rewards. Our experiments reveal that our ECL enables small models like Qwen 3-4B to outperform a history-agnostic baseline 8x its size (Qwen 3-30B) by accurately identifying reliable peers. ECL also boosts frontier models to near-perfect (100%) performance. We show that ECL generalizes well to various MA configurations and we find that trust is modeled well by LLMs, revealing a strong correlation in trust modeling accuracy and final answer quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20539v2">PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21733v1">CE-GOCD: Central Entity-Guided Graph Optimization for Community Detection to Augment LLM Scientific Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted by IEEE ICASSP 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for question answering over scientific research papers. Existing retrieval augmentation methods often rely on isolated text chunks or concepts, but overlook deeper semantic connections between papers. This impairs the LLM's comprehension of scientific literature, hindering the comprehensiveness and specificity of its responses. To address this, we propose Central Entity-Guided Graph Optimization for Community Detection (CE-GOCD), a method that augments LLMs' scientific question answering by explicitly modeling and leveraging semantic substructures within academic knowledge graphs. Our approach operates by: (1) leveraging paper titles as central entities for targeted subgraph retrieval, (2) enhancing implicit semantic discovery via subgraph pruning and completion, and (3) applying community detection to distill coherent paper groups with shared themes. We evaluated the proposed method on three NLP literature-based question-answering datasets, and the results demonstrate its superiority over other retrieval-augmented baseline approaches, confirming the effectiveness of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21714v1">E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09101v3">A Survey of LLM Alignment: Instruction Understanding, Intention Reasoning, and Reliable Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 37 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated exceptional capabilities in understanding and generation. However, in real-world scenarios, users' natural language expressions are often inherently fuzzy, ambiguous, and uncertain, leading to challenges such as vagueness, polysemy, and contextual ambiguity. This paper focuses on three challenges in LLM-based text generation tasks: instruction understanding, intention reasoning, and reliable dialog generation. Regarding human complex instruction, LLMs have deficiencies in understanding long contexts and instructions in multi-round conversations. For intention reasoning, LLMs may have inconsistent command reasoning, difficulty reasoning about commands containing incorrect information, difficulty understanding user ambiguous language commands, and a weak understanding of user intention in commands. Besides, In terms of Reliable Dialog Generation, LLMs may have unstable generated content and unethical generation. To this end, we classify and analyze the performance of LLMs in challenging scenarios and conduct a comprehensive evaluation of existing solutions. Furthermore, we introduce benchmarks and categorize them based on the aforementioned three core challenges. Finally, we explore potential directions for future research to enhance the reliability and adaptability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21700v1">Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21682v1">FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 20 Pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive capabilities across diverse tasks but raise concerns about privacy, copyright, and harmful materials. Existing LLM unlearning methods rarely consider the continual and high-volume nature of real-world deletion requests, which can cause utility degradation and catastrophic forgetting as requests accumulate. To address this challenge, we introduce \fit, a framework for continual unlearning that handles large numbers of deletion requests while maintaining robustness against both catastrophic forgetting and post-unlearning recovery. \fit mitigates degradation through rigorous data \underline{F}iltering, \underline{I}mportance-aware updates, and \underline{T}argeted layer attribution, enabling stable performance across long sequences of unlearning operations and achieving a favorable balance between forgetting effectiveness and utility retention. To support realistic evaluation, we present \textbf{PCH}, a benchmark covering \textbf{P}ersonal information, \textbf{C}opyright, and \textbf{H}armful content in sequential deletion scenarios, along with two symmetric metrics, Forget Degree (F.D.) and Retain Utility (R.U.), which jointly assess forgetting quality and utility preservation. Extensive experiments on four open-source LLMs with hundreds of deletion requests show that \fit achieves the strongest trade-off between F.D. and R.U., surpasses existing methods on MMLU, CommonsenseQA, and GSM8K, and remains resistant against both relearning and quantization recovery attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19423v2">UniRec: Unified Multimodal Encoding for LLM-Based Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models have recently shown promise for multimodal recommendation, particularly with text and image inputs. Yet real-world recommendation signals extend far beyond these modalities. To reflect this, we formalize recommendation features into four modalities: text, images, categorical features, and numerical attributes, and highlight the unique challenges this heterogeneity poses for LLMs in understanding multimodal information. In particular, these challenges arise not only across modalities but also within them, as attributes such as price, rating, and time may all be numeric yet carry distinct semantic meanings. Beyond this intra-modality ambiguity, another major challenge is the nested structure of recommendation signals, where user histories are sequences of items, each associated with multiple attributes. To address these challenges, we propose UniRec, a unified multimodal encoder for LLM-based recommendation. UniRec first employs modality-specific encoders to produce consistent embeddings across heterogeneous signals. It then adopts a triplet representation, comprising attribute name, type, and value, to separate schema from raw inputs and preserve semantic distinctions. Finally, a hierarchical Q-Former models the nested structure of user interactions while maintaining their layered organization. Across multiple real-world benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based recommenders by up to 15%, and extensive ablation studies further validate the contributions of each component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19684v2">LLM-Assisted Authentication and Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 20 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      User authentication and fraud detection face growing challenges as digital systems expand and adversaries adopt increasingly sophisticated tactics. Traditional knowledge-based authentication remains rigid, requiring exact word-for-word string matches that fail to accommodate natural human memory and linguistic variation. Meanwhile, fraud-detection pipelines struggle to keep pace with rapidly evolving scam behaviors, leading to high false-positive rates and frequent retraining cycles required. This work introduces two complementary LLM-enabled solutions, namely, an LLM-assisted authentication mechanism that evaluates semantic correctness rather than exact wording, supported by document segmentation and a hybrid scoring method combining LLM judgement with cosine-similarity metrics and a RAG-based fraud-detection pipeline that grounds LLM reasoning in curated evidence to reduce hallucinations and adapt to emerging scam patterns without model retraining. Experiments show that the authentication system accepts 99.5% of legitimate non-exact answers while maintaining a 0.1% false-acceptance rate, and that the RAG-enhanced fraud detection reduces false positives from 17.2% to 3.5%. Together, these findings demonstrate that LLMs can significantly improve both usability and robustness in security workflows, offering a more adaptive , explainable, and human-aligned approach to authentication and fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02901v3">Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in $\{\pm 1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence, yet their massive memory and computational demands necessitate aggressive quantization, increasingly pushing representations toward the theoretical limit of a single bit. While complex-valued LLMs, such as iFairy, offer a superior chance for low-bit representation compared to real-valued counterparts, they require training from scratch, preventing the utilization of the vast ecosystem of pre-trained real-valued foundation models. Here we present Fairy2i, a universal framework that transforms pre-trained real-valued layers into an equivalent widely-linear complex form, enabling extremely low-bit quantization while reusing existing checkpoints. By proving a lossless mathematical equivalence between real and widely-linear maps, we convert standard Transformers into the complex domain and employ a phase-aware quantization scheme with a highly efficient codebook of fourth roots of unity. Furthermore, we introduce a recursive residual quantization mechanism that iteratively minimizes quantization error, allowing inference to proceed via efficient multiplication-free accumulation. We demonstrate that Fairy2i restores the performance of LLaMA-2 7B at an effective 2-bit precision to levels nearly comparable with full-precision baselines, significantly outperforming state-of-the-art real-valued binary and ternary quantization methods. This work bridges the gap between the representational efficiency of complex-valued arithmetic and the practical utility of pre-trained models, paving a new way for efficient inference on commodity hardware. We open-source the Fairy2i model and code at https://huggingface.co/PKU-DS-LAB/Fairy2i-W2 and https://github.com/PKULab1806/Fairy2i-W2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13258v3">Towards Transparent RAG: Fostering Evidence Traceability in LLM Generation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) delivers substantial value in knowledge-intensive applications. However, its generated responses often lack transparent reasoning paths that trace back to source evidence from retrieved documents. This opacity not only compromises the interpretability of the output but also limits the model's ability to fully exploit the provided context. To address this, we propose TRACE (Transparent RAG with evidenCE tracing), a framework designed to enhance evidence traceability in Large Language Models (LLMs) through reinforcement learning (RL). TRACE guides LLMs to produce structured outputs with explicit evidence citations by prompting and rewarding evidence relevance and proper formatting, alongside accuracy, to optimize structured traceability. To ensure training stability with multiple reward signals, we further introduce an adaptive strategy for merging rewards and adopt a stabilized KL-divergence estimator. Experiments on three multi-hop QA datasets using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct show that TRACE achieves both transparent, evidence-attributed outputs and accuracy improvements of 10-30%. The resulting performance is comparable to advanced commercial LLMs (e.g., OpenAI o1, DeepSeek-R1). Further analyses demonstrate strong generalization capabilities to unseen tasks. Our code is publicly available now.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10233v2">Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 21 pages; Published in AAAI 2026, Main Technical Track
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Combinatorial Optimization (NCO) methods have significantly improved the capability of neural solvers to handle synthetic routing instances. Nonetheless, existing neural solvers typically struggle to generalize effectively from synthetic, uniformly-distributed training data to real-world VRP scenarios, including widely recognized benchmark instances from TSPLib and CVRPLib. To bridge this generalization gap, we present Evolutionary Realistic Instance Synthesis (EvoReal), which leverages an evolutionary module guided by large language models (LLMs) to generate synthetic instances characterized by diverse and realistic structural patterns. Specifically, the evolutionary module produces synthetic instances whose structural attributes statistically mimics those observed in authentic real-world instances. Subsequently, pre-trained NCO models are progressively refined, firstly aligning them with these structurally enriched synthetic distributions and then further adapting them through direct fine-tuning on actual benchmark instances. Extensive experimental evaluations demonstrate that EvoReal markedly improves the generalization capabilities of state-of-the-art neural solvers, yielding a notable reduced performance gap compared to the optimal solutions on the TSPLib (1.05%) and CVRPLib (2.71%) benchmarks across a broad spectrum of problem scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06411v2">SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on https://github.com/zz1358m/SofT-GRPO-master
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21590v1">Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is a dominant approach for improving the reasoning performance of large language models (LLMs), yet growing evidence suggests that its gains arise primarily from distribution sharpening rather than the acquisition of new capabilities. Recent work has shown that sampling from the power distribution of LLMs using Markov chain Monte Carlo (MCMC) can recover performance comparable to RL post-training without relying on external rewards; however, the high computational cost of MCMC makes such approaches impractical for widespread adoption. In this work, we propose a theoretically grounded alternative that eliminates the need for iterative MCMC. We derive a novel formulation showing that the global power distribution can be approximated by a token-level scaled low-temperature one, where the scaling factor captures future trajectory quality. Leveraging this insight, we introduce a training-free and verifier-free algorithm that sharpens the base model's generative distribution autoregressively. Empirically, we evaluate our method on math, QA, and code tasks across four LLMs, and show that our method matches or surpasses one-shot GRPO without relying on any external rewards, while reducing inference latency by over 10x compared to MCMC-based sampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.10996v3">RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the unstructured nature of retrieved context. While retrieval-augmented generation (RAG) methods provide external information, the lack of explicit organization among retrieved passages limits their effectiveness, leading to brittle reasoning pathways. Recent interpretability studies highlighting the importance of structured intermediate reasoning further align with this perspective. We propose Retrieval-And-Structuring (RAS), a framework that dynamically constructs question-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query. On seven knowledge-intensive benchmarks, RAS consistently outperforms strong baselines, achieving up to 8.7\% and 7.0\% gains with proprietary and open-source LLMs, respectively. Our results demonstrate that dynamic, question-specific knowledge structuring offers a robust path to improving reasoning accuracy and robustness in language model generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.04503v4">When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Code available at https://github.com/jeremyperez2/TelephoneGameLLM. Companion website with a Data Explorer tool at https://sites.google.com/view/telephone-game-llm . This paper was published at the 2025 International Conference on Learning Representations (ICLR2025) https://iclr.cc/virtual/2025/poster/28880
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) start interacting with each other and generating an increasing amount of text online, it becomes crucial to better understand how information is transformed as it passes from one LLM to the next. While significant research has examined individual LLM behaviors, existing studies have largely overlooked the collective behaviors and information distortions arising from iterated LLM interactions. Small biases, negligible at the single output level, risk being amplified in iterated interactions, potentially leading the content to evolve towards attractor states. In a series of telephone game experiments, we apply a transmission chain design borrowed from the human cultural evolution literature: LLM agents iteratively receive, produce, and transmit texts from the previous to the next agent in the chain. By tracking the evolution of text toxicity, positivity, difficulty, and length across transmission chains, we uncover the existence of biases and attractors, and study their dependence on the initial text, the instructions, language model, and model size. For instance, we find that more open-ended instructions lead to stronger attraction effects compared to more constrained tasks. We also find that different text properties display different sensitivity to attraction effects, with toxicity leading to stronger attractors than length. These findings highlight the importance of accounting for multi-step transmission dynamics and represent a first step towards a more comprehensive understanding of LLM cultural dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21551v1">Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Effective clinical history taking is a foundational yet underexplored component of clinical reasoning. While large language models (LLMs) have shown promise on static benchmarks, they often fall short in dynamic, multi-turn diagnostic settings that require iterative questioning and hypothesis refinement. To address this gap, we propose \method{}, a note-driven framework that trains LLMs to conduct structured history taking and diagnosis by learning from widely available medical notes. Instead of relying on scarce and sensitive dialogue data, we convert real-world medical notes into high-quality doctor-patient dialogues using a decision tree-guided generation and refinement pipeline. We then propose a three-stage fine-tuning strategy combining supervised learning, simulated data augmentation, and preference learning. Furthermore, we propose a novel single-turn reasoning paradigm that reframes history taking as a sequence of single-turn reasoning problems. This design enhances interpretability and enables local supervision, dynamic adaptation, and greater sample efficiency. Experimental results show that our method substantially improves clinical reasoning, achieving gains of +16.9 F1 and +21.0 Top-1 diagnostic accuracy over GPT-4o. Our code and dataset can be found at https://github.com/zhentingsheng/Note2Chat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21545v1">ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Agentic large language model (LLM) systems rely on external memory for long-horizon state and concurrent multi-agent execution, but centralized indexes and heuristic partitions become bottlenecks as memory volume and parallel access grow. We present ShardMemo, a budgeted tiered memory service with Tier A per-agent working state, Tier B sharded evidence with shard-local approximate nearest neighbor (ANN) indexes, and Tier C, a versioned skill library. Tier B enforces scope-before-routing: structured eligibility constraints mask ineligible shards before routing or ANN search. We cast shard probing as masked mixture-of-experts (MoE) routing over eligible shards, probing up to $B_{\mathrm{probe}}$ shards via Top-$B_{\mathrm{probe}}$ or adaptive Top-$P$, and use cost-aware gating over profile/observation/session shard families; the router is trained from evidence-to-shard supervision. On LoCoMo, ShardMemo improves over the strongest baseline (GAM) by +5.11 to +6.82 F1 across question categories. Under a fixed-budget routing setting ($B_{\mathrm{probe}}=3$), ShardMemo improves over cosine-to-prototype shard routing by +6.87 F1 while reducing retrieval work (VecScan 521->414, -20.5%) and p95 latency (95->76 ms). On long-context HotpotQA, ShardMemo achieves 63.41/61.88/57.95 F1 at 56K/224K/448K tokens. On ToolBench, Tier C reaches 0.97 Precision@3 and 1.94 StepRed (+10.2% and +7.2% over embedding-similarity retrieval).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23694v4">SafeSearch: Automated Red-Teaming of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Preprint. Updated with new experiments
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling them to access broader and more up-to-date information. However, this also introduces a new threat surface: unreliable search results can mislead agents into producing unsafe outputs. Real-world incidents and our two in-the-wild observations show that such failures can occur in practice. To study this threat systematically, we propose SafeSearch, an automated red-teaming framework that is scalable, cost-efficient, and lightweight, enabling harmless safety evaluation of search agents. Using this, we generate 300 test cases spanning five risk categories (e.g., misinformation and prompt injection) and evaluate three search agent scaffolds across 17 representative LLMs. Our results reveal substantial vulnerabilities in LLM-based search agents, with the highest ASR reaching 90.5% for GPT-4.1-mini in a search-workflow setting. Moreover, we find that common defenses, such as reminder prompting, offer limited protection. Overall, SafeSearch provides a practical way to measure and improve the safety of LLM-based search agents. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16651v2">Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Added acknowledgments section and related work on random projection. 8 pages
    </div>
    <details class="paper-abstract">
      Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21533v1">ARGORA: Orchestrated Argumentation for Causally Grounded LLM Reasoning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 58 pages
    </div>
    <details class="paper-abstract">
      Existing multi-expert LLM systems gather diverse perspectives but combine them through simple aggregation, obscuring which arguments drove the final decision. We introduce ARGORA, a framework that organizes multi-expert discussions into explicit argumentation graphs showing which arguments support or attack each other. By casting these graphs as causal models, ARGORA can systematically remove individual arguments and recompute outcomes, identifying which reasoning chains were necessary and whether decisions would change under targeted modifications. We further introduce a correction mechanism that aligns internal reasoning with external judgments when they disagree. Across diverse benchmarks and an open-ended use case, ARGORA achieves competitive accuracy and demonstrates corrective behavior: when experts initially disagree, the framework resolves disputes toward correct answers more often than it introduces new errors, while providing causal diagnostics of decisive arguments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20755v2">ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted in the 9th Annual Conference on Machine Learning and Systems (MLSys 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research to production, understanding how inference engines behave in real time has become both essential and elusive. Unlike general-purpose engines such as ONNX Runtime, today's LLM inference systems offer little operator-level visibility, leaving developers blind to where time and resources go. Even basic questions -- is this workload memory-bound or compute-bound? -- often remain unanswered. To close this gap, we develop a fine-grained, non-intrusive profiling framework for modern LLM inference engines, exemplified by llama-cpp but applicable to similar runtime architectures. Built on extended Berkeley Packet Filter (eBPF) technology, our system dynamically attaches probes to runtime functions across multiple layers -- without modifying or recompiling the source. It transforms collected traces into rich visualizations of operators, graphs, timelines, and hardware counter trends, exposing how dense inference, Mixture-of-Experts routing, and operator offloading behave in practice. With less than 4% runtime overhead and high profiling fidelity, our framework makes LLM inference both transparent and diagnosable, turning performance profiling into a practical tool for optimization, scheduling, and resource-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v4">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Watermarking offers a promising solution for detecting LLM-generated content, yet its robustness under realistic query-free (black-box) evasion remains an open challenge. Existing query-free attacks often achieve limited success or severely distort semantic meaning. We bridge this gap by theoretically analyzing rewriting-based evasion, demonstrating that reducing the average conditional probability of sampling green tokens by a small margin causes the detection probability to decay exponentially. Guided by this insight, we propose the Bias-Inversion Rewriting Attack (BIRA), a practical query-free method that applies a negative logit bias to a proxy suppression set identified via token surprisal. Empirically, BIRA achieves state-of-the-art evasion rates (>99%) across diverse watermarking schemes while preserving semantic fidelity substantially better than prior baselines. Our findings reveal a fundamental vulnerability in current watermarking methods and highlight the need for rigorous stress tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04418v2">The Illusion of Certainty: Uncertainty Quantification for LLMs Fails under Ambiguity</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification (UQ) in Large Language Models (LLMs) is critical for trustworthy deployment. While real-world language is inherently ambiguous, reflecting aleatoric uncertainty, existing UQ methods are typically benchmarked against tasks with no ambiguity. In this work, we demonstrate that while current uncertainty estimators perform well under the restrictive assumption of no ambiguity, they degrade to close-to-random performance on ambiguous data. To this end, we introduce MAQA* and AmbigQA*, the first ambiguous question-answering (QA) datasets equipped with ground-truth answer distributions estimated from factual co-occurrence. We find this performance deterioration to be consistent across different estimation paradigms: using the predictive distribution itself, internal representations throughout the model, and an ensemble of models. We show that this phenomenon can be theoretically explained, revealing that predictive-distribution and ensemble-based estimators are fundamentally limited under ambiguity. Overall, our study reveals a key shortcoming of current UQ methods for LLMs and motivates a rethinking of current modeling paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21500v1">Task-Awareness Improves LLM Generations and Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      In many applications of LLMs, natural language responses often have an underlying structure such as representing discrete labels, numerical values, or graphs. Yet, existing decoding and uncertainty estimation methods operate only in language space and largely disregard structural information. We address this by modeling LLM outputs directly in a task-dependent latent structure. By equipping this structure with a dissimilarity measure, we can compute Bayes-optimal responses. These are not selected from sampled generations but are newly synthesized by combining individual responses in the latent space. Across different tasks, Bayes-optimal responses consistently outperform standard decoding methods like beam search. Moreover, quantifying uncertainty via the induced Bayesian risk captures variations in terms of the latent structure and improves alignment with output quality and correctness. Our decision-theoretic framework is applicable to any problem that admits a latent response structure and enables reliable task-aware LLM predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21471v1">Best Arm Identification with LLM Judges and Limited Human</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 22 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We study fixed-confidence best-arm identification (BAI) where a cheap but potentially biased proxy (e.g., LLM judge) is available for every sample, while an expensive ground-truth label can only be acquired selectively when using a human for auditing. Unlike classical multi-fidelity BAI, the proxy is biased (arm- and context-dependent) and ground truth is selectively observed. Consequently, standard multi-fidelity methods can mis-select the best arm, and uniform auditing, though accurate, wastes scarce resources and is inefficient. We prove that without bias correction and propensity adjustment, mis-selection probability may not vanish (even with unlimited proxy data). We then develop an estimator for the mean of each arm that combines proxy scores with inverse-propensity-weighted residuals and form anytime-valid confidence sequences for that estimator. Based on the estimator and confidence sequence, we propose an algorithm that adaptively selects and audits arms. The algorithm concentrates audits on unreliable contexts and close arms and we prove that a plug-in Neyman rule achieves near-oracle audit efficiency. Numerical experiments confirm the theoretical guarantees and demonstrate the superior empirical performance of the proposed algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21464v1">Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for non-verifiable tasks, such as creative writing, dialogue, and ethical reasoning, remains challenging due to the absence of ground-truth labels. While LLM-as-Judge approaches offer a scalable alternative to human feedback, they face a fundamental limitation: performance is constrained by the evaluator's own quality. If the judge cannot recognize good solutions, it cannot provide useful training signals, and evaluation biases (e.g., favoring verbosity over quality) remain unaddressed. This motivates meta-evaluation: the ability to evaluate and improve the evaluator itself. We introduce CoNL, a framework that unifies generation, evaluation, and meta-evaluation through multi-agent self-play. Our key insight: critique quality can be measured by whether it helps others improve their solutions. In CoNL, multiple agents sharing the same policy engage in structured conversations to propose, critique, and revise solutions. Critiques that enable solution improvements earn a diagnostic reward, creating explicit supervision for meta-evaluation and enabling joint optimization of generation and judging capabilities through self-play, without external judges or ground truth. Experiments on five benchmarks show that CoNL achieves consistent improvements over self-rewarding baselines while maintaining stable training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21459v1">HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 41pages, 10 figures
    </div>
    <details class="paper-abstract">
      LLM role-playing, i.e., using LLMs to simulate specific personas, has emerged as a key capability in various applications, such as companionship, content creation, and digital games. While current models effectively capture character tones and knowledge, simulating the inner thoughts behind their behaviors remains a challenge. Towards cognitive simulation in LLM role-play, previous efforts mainly suffer from two deficiencies: data with high-quality reasoning traces, and reliable reward signals aligned with human preferences. In this paper, we propose HER, a unified framework for cognitive-level persona simulation. HER introduces dual-layer thinking, which distinguishes characters' first-person thinking from LLMs' third-person thinking. To bridge these gaps, we curate reasoning-augmented role-playing data via reverse engineering and construct human-aligned principles and reward models. Leveraging these resources, we train \method models based on Qwen3-32B via supervised and reinforcement learning. Extensive experiments validate the effectiveness of our approach. Notably, our models significantly outperform the Qwen3-32B baseline, achieving a 30.26 improvement on the CoSER benchmark and a 14.97 gain on the Minimax Role-Play Bench. Our datasets, principles, and models will be released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21448v1">ChipBench: A Next-Step Benchmark for Evaluating LLM Performance in AI-Aided Chip Design</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) show significant potential in hardware engineering, current benchmarks suffer from saturation and limited task diversity, failing to reflect LLMs' performance in real industrial workflows. To address this gap, we propose a comprehensive benchmark for AI-aided chip design that rigorously evaluates LLMs across three critical tasks: Verilog generation, debugging, and reference model generation. Our benchmark features 44 realistic modules with complex hierarchical structures, 89 systematic debugging cases, and 132 reference model samples across Python, SystemC, and CXXRTL. Evaluation results reveal substantial performance gaps, with state-of-the-art Claude-4.5-opus achieving only 30.74\% on Verilog generation and 13.33\% on Python reference model generation, demonstrating significant challenges compared to existing saturated benchmarks where SOTA models achieve over 95\% pass rates. Additionally, to help enhance LLM reference model generation, we provide an automated toolbox for high-quality training data generation, facilitating future research in this underexplored domain. Our code is available at https://github.com/zhongkaiyu/ChipBench.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21437v1">Accurate Network Traffic Matrix Prediction via LEAD: an LLM-Enhanced Adapter-Based Conditional Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Driven by the evolution toward 6G and AI-native edge intelligence, network operations increasingly require predictive and risk-aware adaptation under stringent computation and latency constraints. Network Traffic Matrix (TM), which characterizes flow volumes between nodes, is a fundamental signal for proactive traffic engineering. However, accurate TM forecasting remains challenging due to the stochastic, non-linear, and bursty nature of network dynamics. Existing discriminative models often suffer from over-smoothing and provide limited uncertainty awareness, leading to poor fidelity under extreme bursts. To address these limitations, we propose LEAD, a Large Language Model (LLM)-Enhanced Adapter-based conditional Diffusion model. First, LEAD adopts a "Traffic-to-Image" paradigm to transform traffic matrices into RGB images, enabling global dependency modeling via vision backbones. Then, we design a "Frozen LLM with Trainable Adapter" model, which efficiently captures temporal semantics with limited computational cost. Moreover, we propose a Dual-Conditioning Strategy to precisely guide a diffusion model to generate complex, dynamic network traffic matrices. Experiments on the Abilene and GEANT datasets demonstrate that LEAD outperforms all baselines. On the Abilene dataset, LEAD attains a remarkable 45.2% reduction in RMSE against the best baseline, with the error margin rising only marginally from 0.1098 at one-step to 0.1134 at 20-step predictions. Meanwhile, on the GEANT dataset, LEAD achieves a 0.0258 RMSE at 20-step prediction horizon which is 27.3% lower than the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01027v2">DBellQuant: Breaking the Bell with Double-Bell Transformation for LLMs Post Training Binarization</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 19 pages; Appendix added
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance but face substantial computational and memory challenges that limit their practical deployment. Quantization has emerged as a promising solution; however, its effectiveness is often limited by quantization errors arising from weight distributions that are not quantization-friendly and the presence of activation outliers. To address these challenges, we introduce DBellQuant, an innovative post-training quantization (PTQ) framework that achieves nearly 1-bit weight compression and 6-bit activation quantization with minimal performance degradation. DBellQuant uses Learnable Transformation for Dual-Bell (LTDB) algorithm, which transforms single-bell weight distributions into dual-bell forms to reduce binarization errors and applies inverse transformations to smooth activations. DBellQuant sets a new state-of-the-art by preserving superior model performance under aggressive weight and activation quantization. For example, on the Wikitext2 dataset, DBellQuant achieves a perplexity of 14.39 on LLaMA2-13B with 6-bit activation quantization, significantly outperforming BiLLM's 21.35 without activation quantization, underscoring its potential in compressing LLMs for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14466v2">Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to struggle with low-resource languages, primarily due to limited training data, translation noise, and unstable cross-lingual alignment. To address these challenges, we propose LiRA (Linguistic Robust Anchoring for LLMs)-a plug-and-play framework that requires only lightweight fine-tuning on top of existing pretrained backbones. LiRA jointly optimizes representation stability and cross-lingual semantic consistency by combining two key components: Arca (Anchored Representation Composition Architecture), which aligns low-resource inputs to a shared English semantic space through anchor-based alignment and collaborative encoding; and LaSR (Language-coupled Semantic Reasoner), a lightweight, language-aware head that enforces consistency regularization for unified cross-lingual understanding, retrieval, and reasoning. We theoretically show that under controlled anchoring error and translation-induced bias, LiRA guarantees bounded representation deviation and stable downstream performance under local Lipschitz continuity. To facilitate research, we release a new multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Extensive experiments across diverse low-resource benchmarks demonstrate consistent improvements in retrieval, ranking, question answering, and reasoning tasks. Code will be publicly available on GitHub, and the dataset will be hosted on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21410v1">Statsformer: Validated Ensemble Learning with LLM-Derived Semantic Priors</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      We introduce Statsformer, a principled framework for integrating large language model (LLM)-derived knowledge into supervised statistical learning. Existing approaches are limited in adaptability and scope: they either inject LLM guidance as an unvalidated heuristic, which is sensitive to LLM hallucination, or embed semantic information within a single fixed learner. Statsformer overcomes both limitations through a guardrailed ensemble architecture. We embed LLM-derived feature priors within an ensemble of linear and nonlinear learners, adaptively calibrating their influence via cross-validation. This design yields a flexible system with an oracle-style guarantee that it performs no worse than any convex combination of its in-library base learners, up to statistical error. Empirically, informative priors yield consistent performance improvements, while uninformative or misspecified LLM guidance is automatically downweighted, mitigating the impact of hallucinations across a diverse range of prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21380v1">RerouteGuard: Understanding and Mitigating Adversarial Risks for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in multi-model AI systems have leveraged LLM routers to reduce computational cost while maintaining response quality by assigning queries to the most appropriate model. However, as classifiers, LLM routers are vulnerable to novel adversarial attacks in the form of LLM rerouting, where adversaries prepend specially crafted triggers to user queries to manipulate routing decisions. Such attacks can lead to increased computational cost, degraded response quality, and even bypass safety guardrails, yet their security implications remain largely underexplored. In this work, we bridge this gap by systematizing LLM rerouting threats based on the adversary's objectives (i.e., cost escalation, quality hijacking, and safety bypass) and knowledge. Based on the threat taxonomy, we conduct a measurement study of real-world LLM routing systems against existing LLM rerouting attacks. The results reveal that existing routing systems are vulnerable to rerouting attacks, especially in the cost escalation scenario. We then characterize existing rerouting attacks using interpretability techniques, revealing that they exploit router decision boundaries through confounder gadgets that prepend queries to force misrouting. To mitigate these risks, we introduce RerouteGuard, a flexible and scalable guardrail framework for LLM rerouting. RerouteGuard filters adversarial rerouting prompts via dynamic embedding-based detection and adaptive thresholding. Extensive evaluations in three attack settings and four benchmarks demonstrate that RerouteGuard achieves over 99% detection accuracy against state-of-the-art rerouting attacks, while maintaining negligible impact on legitimate queries. The experimental results indicate that RerouteGuard offers a principled and practical solution for safeguarding multi-model AI systems against adversarial rerouting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19402v2">PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Production LLM deployments serve diverse workloads where cost and quality requirements vary by customer tier, time of day, and query criticality. Model serving systems accept latency SLOs directly. LLM routers do not. They force operators to tune parameters offline and guess what accuracy might result. The relationship between parameters and outcomes is indirect, non-monotonic, and dataset-dependent. Operators need to specify accuracy targets, not infer them from opaque settings. We present PROTEUS (Polymorphic Router for Operational Target Enforcement with Unified SLA), a router that accepts accuracy targets tau as runtime input. PROTEUS uses Lagrangian dual control. A learned dual variable lambda tracks constraint violations during training and conditions the policy network. This lets the router translate specified tau values into routing decisions that satisfy them. A single trained model serves the full accuracy spectrum without retraining.We evaluate on RouterBench (11 models, 405K queries) and SPROUT (14 models, 45K queries). PROTEUS achieves consistent floor compliance where accuracy meets or exceeds tau. The target-response correlation reaches 0.97 to 0.98. The closest baseline, OmniRouter, meets floors only 22% of the time despite also using Lagrangian optimization. PROTEUS operates across tau in [0.85, 0.95] from a single model. On RouterBench it achieves 90.1% accuracy, within 1.3% of oracle. On SPROUT it achieves 94.0% accuracy, within 4.6% of oracle. Cost savings reach 89.8% versus the best fixed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.06787v5">FLAME: Empowering Frozen LLMs for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information and struggle with sparsity, while Large Language Models (LLMs) address these limitations through rich world knowledge and strong context modeling. Fine-tuning LLMs is effective but costly, while non-fine-tuned LLMs are efficient but suboptimal. To address this trade-off, we propose \textbf{FLAME}, a framework that extracts context-aware hidden states from intermediate layers of frozen LLMs to train data-efficient KGC classifiers. We bridge LLM-KG semantic gaps via subgraph-based entity descriptions and employ sliced mutual information (SMI) to quantify task-relevant information in representations. Experiments demonstrate that FLAME achieves 47\% improvement over non-fine-tuned LLM baselines and, to our knowledge, is the first to achieve fine-tuned performance with $188\times$ memory efficiency and $26.11\times$ speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22137v4">HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Edge-cloud collaborative inference is becoming a practical necessity for LLM-powered edge devices: on-device models often cannot afford the required reasoning capability, while cloud-only inference could be prohibitively costly and slow under strict latency and token/API budgets. However, existing edge-cloud collaboration methods often route per query or fixed steps simply based-on the estimated difficulty. Such coarse and static heuristics overlook subtask dependencies, missing opportunities for parallel execution and budget-adaptive routing. To this end, we propose \textbf{HybridFlow}, a resource-adaptive edge-cloud inference framework that (i) builds a dependency-aware DAG for each query and executes newly unlocked subtasks in parallel, reducing end-to-end latency; (ii) routes each subtask online to the edge or cloud via a learned benefit--cost utility model that dynamically trades accuracy gains against token/API and latency budgets, thereby reducing unnecessary cloud usage while preserving reasoning quality. Across GPQA, MMLU-Pro, AIME24, and LiveBench-Reasoning, HybridFlow improves the cost-accuracy trade-off, reducing latency and cloud API usage while maintaining competitive accuracy against strong structured reasoning baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24050v3">Bridging On-Device and Cloud LLMs for Collaborative Reasoning: A Unified Methodology for Local Routing and Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 We propose a unified post-training framework that integrates routing optimization, enabling the on-device LLM to improve its problem-solving ability while learning routing strategies
    </div>
    <details class="paper-abstract">
      Device-cloud collaboration holds promise for deploying large language models (LLMs), leveraging lightweight on-device models for efficiency while relying on powerful cloud models for superior reasoning. A central challenge in this setting is determining, for each incoming query, whether it should be processed locally or offloaded to the cloud. Existing approaches typically rely on external routers, which often struggle to determine difficulty from the prompt itself, especially for tasks involving complex reasoning. Motivated by this limitation, we propose enabling on-device LLMs to decide internally whether to invoke cloud assistance at inference time, with this capability instilled through reinforcement learning based post-training. Casting on-device LLM post-training as a reward maximization problem, we design hierarchical rewards to encourage local problem solving and judicious cloud offloading. To solve the resulting problem, we develop an algorithm featuring a group-level policy gradient that stabilizes optimization, together with adaptive prompt filtering that provides complementary learning signals to mitigate policy collapse (i.e., exclusive local execution or exclusive cloud offloading). Extensive experiments on on-device-scale LLaMA and Qwen models across multiple reasoning benchmarks show that our method consistently outperforms baselines and significantly narrows the gap to full cloud LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21351v1">Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Submitted to ICML 2026
    </div>
    <details class="paper-abstract">
      Attention-FFN disaggregation (AFD) is an emerging architecture for LLM decoding that separates state-heavy, KV-cache-dominated Attention computation from stateless, compute-intensive FFN computation, connected by per-step communication. While AFD enables independent scaling of memory and compute resources, its performance is highly sensitive to the Attention/FFN provisioning ratio: mis-sizing induces step-level blocking and costly device idle time. We develop a tractable analytical framework for sizing AFD bundles in an $r$A-$1$F topology, where the key difficulty is that Attention-side work is nonstationary-token context grows and requests are continuously replenished with random lengths-while FFN work is stable given the aggregated batch. Using a probabilistic workload model, we derive closed-form rules for the optimal A/F ratio that maximize average throughput per instance across the system. A trace-calibrated AFD simulator validates the theory: across workloads, the theoretical optimal A/F ratio matches the simulation-optimal within 10%, and consistently reduces idle time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01203v3">HetGCoT: Heterogeneous Graph-Enhanced Chain-of-Thought LLM Reasoning for Academic Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      Academic question answering (QA) in heterogeneous scholarly networks presents unique challenges requiring both structural understanding and interpretable reasoning. While graph neural networks (GNNs) capture structured graph information and large language models (LLMs) demonstrate strong capabilities in semantic comprehension, current approaches lack integration at the reasoning level. We propose HetGCoT, a framework enabling LLMs to effectively leverage and learn information from graphs to reason interpretable academic QA results. Our framework introduces three technical contributions: (1) a framework that transforms heterogeneous graph structural information into LLM-processable reasoning chains, (2) an adaptive metapath selection mechanism identifying relevant subgraphs for specific queries, and (3) a multi-step reasoning strategy systematically incorporating graph contexts into the reasoning process. Experiments on OpenAlex and DBLP datasets show our approach outperforms all sota baselines. The framework demonstrates adaptability across different LLM architectures and applicability to various scholarly question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21347v1">Towards Robust Dysarthric Speech Recognition: LLM-Agent Post-ASR Correction Beyond WER</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      While Automatic Speech Recognition (ASR) is typically benchmarked by word error rate (WER), real-world applications ultimately hinge on semantic fidelity. This mismatch is particularly problematic for dysarthric speech, where articulatory imprecision and disfluencies can cause severe semantic distortions. To bridge this gap, we introduce a Large Language Model (LLM)-based agent for post-ASR correction: a Judge-Editor over the top-k ASR hypotheses that keeps high-confidence spans, rewrites uncertain segments, and operates in both zero-shot and fine-tuned modes. In parallel, we release SAP-Hypo5, the largest benchmark for dysarthric speech correction, to enable reproducibility and future exploration. Under multi-perspective evaluation, our agent achieves a 14.51% WER reduction alongside substantial semantic gains, including a +7.59 pp improvement in MENLI and +7.66 pp in Slot Micro F1 on challenging samples. Our analysis further reveals that WER is highly sensitive to domain shift, whereas semantic metrics correlate more closely with downstream task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21344v1">Dynamic Framework for Collaborative Learning: Leveraging Advanced LLM with Adaptive Feedback Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Publication Link: https://ieeexplore.ieee.org/document/11118419
    </div>
    <details class="paper-abstract">
      This paper presents a framework for integrating LLM into collaborative learning platforms to enhance student engagement, critical thinking, and inclusivity. The framework employs advanced LLMs as dynamic moderators to facilitate real-time discussions and adapt to learners' evolving needs, ensuring diverse and inclusive educational experiences. Key innovations include robust feedback mechanisms that refine AI moderation, promote reflective learning, and balance participation among users. The system's modular architecture featuring ReactJS for the frontend, Flask for backend operations, and efficient question retrieval supports personalized and engaging interactions through dynamic adjustments to prompts and discussion flows. Testing demonstrates that the framework significantly improves student collaboration, fosters deeper comprehension, and scales effectively across various subjects and user groups. By addressing limitations in static moderation and personalization in existing systems, this work establishes a strong foundation for next-generation AI-driven educational tools, advancing equitable and impactful learning outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19139v2">Native LLM and MLLM Inference at Scale on Apple Silicon</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The growing adoption of Apple Silicon for machine learning development has created demand for efficient inference solutions that leverage its unique unified memory architecture. However, existing tools either lack native optimization (PyTorch MPS) or focus solely on text models, leaving multimodal workloads underserved. We present vllm-mlx, a framework for efficient LLM and MLLM inference on Apple Silicon built natively on MLX. For text models, we achieve 21\% to 87\% higher throughput than llama-cpp across models ranging from Qwen3-0.6B to Nemotron-30B, while providing continuous batching that scales to 4.3x aggregate throughput at 16 concurrent requests. For multimodal models, we introduce content-based prefix caching that eliminates redundant vision encoding by identifying identical images through content hashing, regardless of input format. Our evaluation on Apple M4 Max demonstrates throughput of up to 525 tokens per second on text models and 28x speedup on repeated image queries, reducing multimodal latency from 21.7 seconds to under 1 second. Video analysis with up to 64 frames achieves 24.7x cache speedup. We release our implementation as open source to support efficient inference on consumer Apple hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20090v2">Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents can translate high-level user intents into plans and actions in an environment. Yet after observing an outcome, users may wonder: What if I had phrased my intent differently? We introduce a framework that enables such counterfactual reasoning in agentic LLM-driven control scenarios, while providing formal reliability guarantees. Our approach models the closed-loop interaction between a user, an LLM-based agent, and an environment as a structural causal model (SCM), and leverages test-time scaling to generate multiple candidate counterfactual outcomes via probabilistic abduction. Through an offline calibration phase, the proposed conformal counterfactual generation (CCG) yields sets of counterfactual outcomes that are guaranteed to contain the true counterfactual outcome with high probability. We showcase the performance of CCG on a wireless network control use case, demonstrating significant advantages compared to naive re-execution baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19510v2">ALRM: Agentic LLM for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21283v1">DUET: Distilled LLM Unlearning from an Efficiently Contextualized Teacher</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      LLM unlearning is a technique to remove the impacts of undesirable knowledge from the model without retraining from scratch, which is indispensable towards trustworthy AI. Existing unlearning methods face significant limitations: conventional tuning-based unlearning is computationally heavy and prone to catastrophic forgetting. In contrast, in-contextualized unlearning is lightweight for precise unlearning but vulnerable to prompt removal or reverse engineering attacks. In response, we propose Distilled Unlearning from an Efficient Teacher (DUET), a novel distillation-based unlearning method that combines the merits of these two lines of work. It learns a student model to imitate the behavior of a prompt-steered teacher that effectively refuses undesirable knowledge generation while preserving general domain knowledge. Extensive evaluations on existing benchmarks with our enriched evaluation protocols demonstrate that DUET achieves higher performance in both forgetting and utility preservation, while being orders of magnitude more data-efficient than state-of-the-art unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20802v3">SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICASSP 2026
    </div>
    <details class="paper-abstract">
      The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18987v3">LLMs versus the Halting Problem: Revisiting Program Termination Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18196v2">LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at https://llm-symbol.github.io/LogicReward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21261v1">User-Centric Phishing Detection: A RAG and LLM-Based Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      The escalating sophistication of phishing emails necessitates a shift beyond traditional rule-based and conventional machine-learning-based detectors. Although large language models (LLMs) offer strong natural language understanding, using them as standalone classifiers often yields elevated falsepositive (FP) rates, which mislabel legitimate emails as phishing and create significant operational burden. This paper presents a personalized phishing detection framework that integrates LLMs with retrieval-augmented generation (RAG). For each message, the system constructs user-specific context by retrieving a compact set of the user's historical legitimate emails and enriching it with real-time domain and URL reputation from a cyber-threat intelligence platform, then conditions the LLM's decision on this evidence. We evaluate four open-source LLMs (Llama4-Scout, DeepSeek-R1, Mistral-Saba, and Gemma2) on an email dataset collected from public and institutional sources. Results show high performance; for example, Llama4-Scout attains an F1-score of 0.9703 and achieves a 66.7% reduction in FPs with RAG. These findings validate that a RAG-based, user-profiling approach is both feasible and effective for building high-precision, low-friction email security systems that adapt to individual communication patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20420v2">Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Developing human understandable interpretation of large language models (LLMs) becomes increasingly critical for their deployment in essential domains. Mechanistic interpretability seeks to mitigate the issues through extracts human-interpretable process and concepts from LLMs' activations. Sparse autoencoders (SAEs) have emerged as a popular approach for extracting interpretable and monosemantic concepts by decomposing the LLM internal representations into a dictionary. Despite their empirical progress, SAEs suffer from a fundamental theoretical ambiguity: the well-defined correspondence between LLM representations and human-interpretable concepts remains unclear. This lack of theoretical grounding gives rise to several methodological challenges, including difficulties in principled method design and evaluation criteria. In this work, we show that, under mild assumptions, LLM representations can be approximated as a {linear mixture} of the log-posteriors over concepts given the input context, through the lens of a latent variable model where concepts are treated as latent variables. This motivates a principled framework for concept extraction, namely Concept Component Analysis (ConCA), which aims to recover the log-posterior of each concept from LLM representations through a {unsupervised} linear unmixing process. We explore a specific variant, termed sparse ConCA, which leverages a sparsity prior to address the inherent ill-posedness of the unmixing problem. We implement 12 sparse ConCA variants and demonstrate their ability to extract meaningful concepts across multiple LLMs, offering theory-backed advantages over SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21239v1">TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Although Large Language Models have advanced Automated Heuristic Design, treating algorithm evolution as a monolithic text generation task overlooks the coupling between discrete algorithmic structures and continuous numerical parameters. Consequently, existing methods often discard promising algorithms due to uncalibrated constants and suffer from premature convergence resulting from simple similarity metrics. To address these limitations, we propose TIDE, a Tuning-Integrated Dynamic Evolution framework designed to decouple structural reasoning from parameter optimization. TIDE features a nested architecture where an outer parallel island model utilizes Tree Similarity Edit Distance to drive structural diversity, while an inner loop integrates LLM-based logic generation with a differential mutation operator for parameter tuning. Additionally, a UCB-based scheduler dynamically prioritizes high-yield prompt strategies to optimize resource allocation. Extensive experiments across nine combinatorial optimization problems demonstrate that TIDE discovers heuristics that significantly outperform state-of-the-art baselines in solution quality while achieving improved search efficiency and reduced computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21233v1">Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 24 pages, 6 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present \textbf{\textsc{JustAsk}}, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, \textsc{JustAsk} requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound-based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on \textbf{41} black-box commercial models across multiple providers, \textsc{JustAsk} consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21210v1">Uncovering Hidden Correctness in LLM Causal Reasoning via Symbolic Verification</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 EACL 2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being applied to tasks that involve causal reasoning. However, current benchmarks often rely on string matching or surface-level metrics that do not capture whether the output of a model is formally valid under the semantics of causal reasoning. To address this, we propose DoVerifier, a simple symbolic verifier that checks whether LLM-generated causal expressions are derivable from a given causal graph using rules from do-calculus and probability theory. This allows us to recover correct answers to causal queries that would otherwise be marked incorrect due to superficial differences in their causal semantics. Our evaluations on synthetic data and causal QA benchmarks show that DoVerifier more accurately captures semantic correctness of causal reasoning traces, offering a more rigorous and informative way to evaluate LLMs on causal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18874v3">When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 The ACM Web Conference 2026
    </div>
    <details class="paper-abstract">
      Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 Facebook ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception at only a fraction of the cost (223x lower) and time (52x faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20251v2">Efficient Evaluation of LLM Performance with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Exhaustively evaluating many large language models (LLMs) on a large suite of benchmarks is expensive. We cast benchmarking as finite-population inference and, under a fixed query budget, seek tight confidence intervals (CIs) for model accuracy with valid frequentist coverage. We propose Factorized Active Querying (FAQ), which (a) leverages historical information through a Bayesian factor model; (b) adaptively selects questions using a hybrid variance-reduction/active-learning sampling policy; and (c) maintains validity through Proactive Active Inference -- a finite-population extension of active inference (Zrnic & Candès, 2024) that enables direct question selection while preserving coverage. With negligible overhead cost, FAQ delivers up to $5\times$ effective sample size gains over strong baselines on two benchmark suites, across varying historical-data missingness levels: this means that it matches the CI width of uniform sampling while using up to $5\times$ fewer queries. We release our source code and our curated datasets to support reproducible evaluation and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.10163v2">Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance across various tasks, prompting researchers to develop diverse evaluation benchmarks. However, most benchmarks typically measure the ability of LLMs to respond to individual questions, neglecting the complex interactions in real-world applications. We introduce Compound Question Synthesis (CQ-Syn) to build Compound-QA, a benchmark targeting questions composed of multiple interrelated sub-questions. This benchmark is derived from existing QA datasets, annotated with proprietary LLMs, and verified by humans for accuracy. It encompasses five categories: Factual-Statement, Cause-and-Effect, Hypothetical-Analysis, Comparison-and-Selection, and Evaluation-and-Suggestion. It evaluates the LLM capability in terms of three dimensions, including understanding, reasoning, and knowledge. Evaluating nine open-source LLMs on Compound-QA reveals that their performance on compound questions is notably lower than on non-compound questions. We further explore strategies to enhance LLMs' handling of compound questions, and our results show that these methods substantially improve models' comprehension and reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21169v1">Output-Space Search: Targeting LLM Generations in a Frozen Encoder-Defined Output Space</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      We introduce Output-Space Search (OS-Search), which turns LLM generation into endpoint search. An outer loop selects a target z* in a frozen encoder-defined 3D output space Z, and a retrieval-grounded policy trained with sequence-level RL generates outputs whose coordinates land near z* under standard autoregressive decoding. This enables parallel sweeps and black-box optimization in Z without path-dependent token/program search. On stories, sweeping Z (text) yields 3.1x higher LLM-scored diversity than prompt-chaining. On code, Bayesian optimization over Z (code) improves an objective withheld from the controller under matched inference budgets while preserving validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21164v1">Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Plane Geometry Problem Solving (PGPS) is a multimodal reasoning task that aims to solve a plane geometric problem based on a geometric diagram and problem textual descriptions. Although Large Language Models (LLMs) possess strong reasoning skills, their direct application to PGPS is hindered by their inability to process visual diagrams. Existing works typically fine-tune Multimodal LLMs (MLLMs) end-to-end on large-scale PGPS data to enhance visual understanding and reasoning simultaneously. However, such joint optimization may compromise base LLMs' inherent reasoning capability. In this work, we observe that LLM itself is potentially a powerful PGPS solver when appropriately formulating visual information as textual descriptions. We propose to train a MLLM Interpreter to generate geometric descriptions for the visual diagram, and an off-the-shelf LLM is utilized to perform reasoning. Specifically, we choose Conditional Declaration Language (CDL) as the geometric description as its conciseness eases the MLLM Interpreter training. The MLLM Interpreter is fine-tuned via CoT (Chain-of-Thought)-augmented SFT followed by GRPO to generate CDL. Instead of using a conventional solution-based reward that compares the reasoning result with the ground-truth answer, we design CDL matching rewards to facilitate more effective GRPO training, which provides more direct and denser guidance for CDL generation. To support training, we construct a new dataset, Formalgeo7k-Rec-CoT, by manually reviewing Formalgeo7k v2 and incorporating CoT annotations. Extensive experiments on Formalgeo7k-Rec-CoT, Unigeo, and MathVista show our method (finetuned on only 5.5k data) performs favorably against leading open-source and closed-source MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20900v2">When Experts Speak:Sequential LLM-Bayesian Learning for Startup Success Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Evaluating startups is inherently challenging in entrepreneurial finance, where investors confront severe information asymmetry and limited quantitative data. Leveraging a novel expert network call data, we develop an LLM-Bayesian model that analyzes these conversations at the question-answer turn level, extracting semantic and evaluative signals via large language models (LLMs) and aggregating them in a sequential Bayesian architecture. The model dynamically updates beliefs as additional expert calls occur and attenuates contradictory assessments, which are absent from existing text-based screening tools. Empirically, our model outperforms state-of-the-art benchmarks by 6.691% in F1-score and increases portfolio-level Return on Investment by 15.255%. Attention and ablation analyses reveal that conversational cues are particularly informative for technologically complex startups, young firms, diverse founding teams, and firms with low public visibility. By converting expert dialogue into continually updated probabilities, our model advances research in entrepreneurial finance and information systems and offers policy implications for improving funding outcomes for informationally disadvantaged startups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.06160v4">Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from an established stigmatization framework, our analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03217v2">Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Accepted to the 2026 IEEE/ACM 48th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP '26)
    </div>
    <details class="paper-abstract">
      Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately these patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing patches unlikely to be accepted can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patches rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13725v3">AI Kill Switch for malicious web-based LLM agent</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Recently, web-based Large Language Model (LLM) agents autonomously perform increasingly complex tasks, thereby bringing significant convenience. However, they also amplify the risks of malicious misuse cases such as unauthorized collection of personally identifiable information (PII), generation of socially divisive content, and even automated web hacking. To address these threats, we propose an AI Kill Switch technique that can immediately halt the operation of malicious web-based LLM agents. To achieve this, we introduce AutoGuard - the key idea is generating defensive prompts that trigger the safety mechanisms of malicious LLM agents. In particular, generated defense prompts are transparently embedded into the website's DOM so that they remain invisible to human users but can be detected by the crawling process of malicious agents, triggering its internal safety mechanisms to abort malicious actions once read. To evaluate our approach, we constructed a dedicated benchmark consisting of three representative malicious scenarios. Experimental results show that AutoGuard achieves over 80% Defense Success Rate (DSR) across diverse malicious agents, including GPT-4o, Claude-4.5-Sonnet and generalizes well to advanced models like GPT-5.1, Gemini-2.5-flash, and Gemini-3-pro. Also, our approach demonstrates robust defense performance in real-world website environments without significant performance degradation for benign agents. Through this research, we demonstrate the controllability of web-based LLM agents, thereby contributing to the broader effort of AI control and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09378v3">Can you map it to English? The Role of Cross-Lingual Alignment in Multilingual Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can answer prompts in many languages, despite being trained predominantly on English; yet, the mechanisms driving this generalization remain poorly understood. This work asks: How does an LLM's ability to align representations of non-English inputs to English impact its performance on natural language understanding (NLU) tasks? We study the role of representation alignment in instance-level task decisions, complementing prior analyses conducted both at the language level and task-independently. We introduce the Discriminative Alignment Index ($\DALI$) to quantify instance-level alignment across 24 languages other than English and three distinct NLU tasks. Results show that incorrect NLU predictions are strongly associated with lower representation alignment with English in the model's middle layers. Through activation patching, we show that incorrect predictions in languages other than English can be fixed by patching their parallel English activations in the middle layers, thereby demonstrating the causal role of representation (mis)alignment in cross-lingual correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18902v2">Evaluating LLMs for Career Guidance: Comparative Analysis of Computing Competency Recommendations Across Ten African Countries</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 42 pages, 5 figures, 5 tables. Submitted to Computers & Education Open
    </div>
    <details class="paper-abstract">
      Employers increasingly expect graduates to utilize large language models (LLMs) in the workplace, yet the competencies needed for computing roles across Africa remain unclear given varying national contexts. This study examined how six LLMs, namely ChatGPT 4, DeepSeek, Gemini, Claude 3.5, Llama 3, and Mistral AI, describe entry-level computing career expectations across ten African countries. Using the Computing Curricula 2020 framework and drawing on Digital Colonialism Theory and Ubuntu Philosophy, content analysis of 60 LLM responses to standardized prompts reveals consistent coverage of technical competencies such as cloud computing and programming, but notable differences in non-technical competencies, particularly ethics and responsible AI use. Models vary considerably in recognizing country-specific factors, including local technology ecosystems, language requirements, and national policies averaging only 35.4% contextual awareness overall. Open-source models demonstrated stronger contextual awareness and better balance between technical and professional skills, with Llama (4.47/5) and DeepSeek (4.25/5) outperforming proprietary alternatives ChatGPT-4 (3.90/5) and Claude (3.46/5). However, Mistral's poor contextual performance (0.00/4) despite being open-source indicates that development philosophy alone does not guarantee contextual responsiveness. This first comprehensive comparison of LLM career guidance for African computing students uncovers entrenched infrastructure assumptions and Western-centric biases that create gaps between technical recommendations and local realities. The findings challenge assumptions about AI tool quality in resource-constrained settings and underscore the need for decolonial approaches to AI in education, emphasizing contextual relevance and hybrid human-AI guidance models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11062v5">Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22386v1">Specialists or Generalists? Multi-Agent and Single-Agent LLMs for Essay Grading</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Automated essay scoring (AES) systems increasingly rely on large language models, yet little is known about how architectural choices shape their performance across different essay quality levels. This paper evaluates single-agent and multi-agent LLM architectures for essay grading using the ASAP 2.0 corpus. Our multi-agent system decomposes grading into three specialist agents (Content, Structure, Language) coordinated by a Chairman Agent that implements rubric-aligned logic including veto rules and score capping. We test both architectures in zero-shot and few-shot conditions using GPT-5.1. Results show that the multi-agent system is significantly better at identifying weak essays while the single-agent system performs better on mid-range essays. Both architectures struggle with high-quality essays. Critically, few-shot calibration emerges as the dominant factor in system performance -- providing just two examples per score level improves QWK by approximately 26% for both architectures. These findings suggest architectural choice should align with specific deployment priorities, with multi-agent AI particularly suited for diagnostic screening of at-risk students, while single-agent models provide a cost-effective solution for general assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22362v1">Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in production, contributing towards shifting the burden in terms of computational resources and energy demands from training to inference. While prior work has examined the energy cost of inference per prompt or per token, we highlight how \emph{system-level design choices} - such as numerical precision, batching strategy, and request scheduling - can lead to orders-of-magnitude differences in energy consumption for the same model. We perform a detailed empirical study of LLM inference energy and latency on NVIDIA H100 GPUs, analyzing the impact of quantization, batch size, and serving configuration (e.g., with Hugging Face's Text Generation Inference server). Our results reveal that lower-precision formats only yield energy gains in compute-bound regimes; that batching improves energy efficiency, especially in memory-bound phases like decoding; and that structured request timing (arrival shaping) can reduce per-request energy by up to 100 times. We argue that sustainable LLM deployment depends not only on model internals, but also on the orchestration of the serving stack. Our findings motivate phase-aware energy profiling and system-level optimizations for greener AI services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16679v2">PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21282v2">It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with reinforcement learning (RL) methods such as PPO and GRPO commonly relies on ratio clipping to stabilise updates. While effective at preventing instability, clipping discards information, introduces gradient discontinuities and can prevent exploration of better policies. Inspired by label smoothing, we propose Probability Smoothing Policy Optimisation (PSPO). PSPO smooths current policy probabilities toward the behaviour policy before computing importance ratios, creating a soft trust region that preserves gradients while preventing destabilising updates. Unlike prior soft clipping approaches that use sigmoid-based transformations which can suffer from vanishing gradients and saturation, our method uses a linear interpolation, providing simpler and more robust gradient preservation. Empirically, GR-PSPO outperforms clipping and sigmoid-based alternatives on mathematical reasoning benchmarks when refining models with prior domain knowledge, achieving an accuracy of 79.9% on GSM8K and 59.6% on MATH for Qwen2-Math-1.5B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22336v1">Dependence-Aware Label Aggregation for LLM-as-a-Judge via Ising Models</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large-scale AI evaluation increasingly relies on aggregating binary judgments from $K$ annotators, including LLMs used as judges. Most classical methods, e.g., Dawid-Skene or (weighted) majority voting, assume annotators are conditionally independent given the true label $Y\in\{0,1\}$, an assumption often violated by LLM judges due to shared data, architectures, prompts, and failure modes. Ignoring such dependencies can yield miscalibrated posteriors and even confidently incorrect predictions. We study label aggregation through a hierarchy of dependence-aware models based on Ising graphical models and latent factors. For class-dependent Ising models, the Bayes log-odds is generally quadratic in votes; for class-independent couplings, it reduces to a linear weighted vote with correlation-adjusted parameters. We present finite-$K$ examples showing that methods based on conditional independence can flip the Bayes label despite matching per-annotator marginals. We prove separation results demonstrating that these methods remain strictly suboptimal as the number of judges grows, incurring nonvanishing excess risk under latent factors. Finally, we evaluate the proposed method on three real-world datasets, demonstrating improved performance over the classical baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22329v1">Sparks of Rationality: Do Reasoning LLMs Align with Human Judgment and Choice?</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly positioned as decision engines for hiring, healthcare, and economic judgment, yet real-world human judgment reflects a balance between rational deliberation and emotion-driven bias. If LLMs are to participate in high-stakes decisions or serve as models of human behavior, it is critical to assess whether they exhibit analogous patterns of (ir)rationalities and biases. To this end, we evaluate multiple LLM families on (i) benchmarks testing core axioms of rational choice and (ii) classic decision domains from behavioral economics and social norms where emotions are known to shape judgment and choice. Across settings, we show that deliberate "thinking" reliably improves rationality and pushes models toward expected-value maximization. To probe human-like affective distortions and their interaction with reasoning, we use two emotion-steering methods: in-context priming (ICP) and representation-level steering (RLS). ICP induces strong directional shifts that are often extreme and difficult to calibrate, whereas RLS produces more psychologically plausible patterns but with lower reliability. Our results suggest that the same mechanisms that improve rationality also amplify sensitivity to affective interventions, and that different steering methods trade off controllability against human-aligned behavior. Overall, this points to a tension between reasoning and affective steering, with implications for both human simulation and the safe deployment of LLM-based decision systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22311v1">Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents exhibit strong step-by-step reasoning capabilities over short horizons, yet often fail to sustain coherent behavior over long planning horizons. We argue that this failure reflects a fundamental mismatch: step-wise reasoning induces a form of step-wise greedy policy that is adequate for short horizons but fails in long-horizon planning, where early actions must account for delayed consequences. From this planning-centric perspective, we study LLM-based agents in deterministic, fully structured environments with explicit state transitions and evaluation signals. Our analysis reveals a core failure mode of reasoning-based policies: locally optimal choices induced by step-wise scoring lead to early myopic commitments that are systematically amplified over time and difficult to recover from. We introduce FLARE (Future-aware Lookahead with Reward Estimation) as a minimal instantiation of future-aware planning to enforce explicit lookahead, value propagation, and limited commitment in a single model, allowing downstream outcomes to influence early decisions. Across multiple benchmarks, agent frameworks, and LLM backbones, FLARE consistently improves task performance and planning-level behavior, frequently allowing LLaMA-8B with FLARE to outperform GPT-4o with standard step-by-step reasoning. These results establish a clear distinction between reasoning and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21322v2">TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26), April 13--17, 2026, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how they represent diverse cultures. However, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2925 annotations from 108 annotators with lived experience and native language proficiency from across 71 regions in India and 14 languages. Concerningly, we find that 88% of the generated stories contain misrepresentations, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. We also transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22290v1">The Six Sigma Agent: Achieving Enterprise-Grade Reliability in LLM Systems Through Consensus-Driven Decomposed Execution</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 25 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate remarkable capabilities yet remain fundamentally probabilistic, presenting critical reliability challenges for enterprise deployment. We introduce the Six Sigma Agent, a novel architecture that achieves enterprise-grade reliability through three synergistic components: (1) task decomposition into a dependency tree of atomic actions; (2) micro-agent sampling where each task is executed n times in parallel across diverse LLMs to generate independent outputs; and (3) consensus voting with dynamic scaling, clustering outputs and selecting the answer from the winning cluster with maximum votes. We prove that sampling n independent outputs with error rate p achieves system error O(p^{ceil(n/2)}), enabling exponential reliability gains. Even using cheaper models with 5% per-action error, consensus voting with 5 agents reduces error to 0.11%; dynamic scaling to 13 agents achieves 3.4 DPMO (Defects Per Million Opportunities), the Six Sigma standard. Evaluation across three enterprise use cases demonstrates a 14,700x reliability improvement over single-agent execution while reducing costs by 80%. Our work establishes that reliability in AI systems emerges from principled redundancy and consensus rather than model scaling alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22240v1">A Systematic Literature Review on LLM Defenses Against Prompt Injection and Jailbreaking: Expanding NIST Taxonomy</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 27 pages, 14 figures, 11 tables, submitted to Elsevier Computer Science Review
    </div>
    <details class="paper-abstract">
      The rapid advancement and widespread adoption of generative artificial intelligence (GenAI) and large language models (LLMs) has been accompanied by the emergence of new security vulnerabilities and challenges, such as jailbreaking and other prompt injection attacks. These maliciously crafted inputs can exploit LLMs, causing data leaks, unauthorized actions, or compromised outputs, for instance. As both offensive and defensive prompt injection techniques evolve quickly, a structured understanding of mitigation strategies becomes increasingly important. To address that, this work presents the first systematic literature review on prompt injection mitigation strategies, comprehending 88 studies. Building upon NIST's report on adversarial machine learning, this work contributes to the field through several avenues. First, it identifies studies beyond those documented in NIST's report and other academic reviews and surveys. Second, we propose an extension to NIST taxonomy by introducing additional categories of defenses. Third, by adopting NIST's established terminology and taxonomy as a foundation, we promote consistency and enable future researchers to build upon the standardized taxonomy proposed in this work. Finally, we provide a comprehensive catalog of the reviewed prompt injection defenses, documenting their reported quantitative effectiveness across specific LLMs and attack datasets, while also indicating which solutions are open-source and model-agnostic. This catalog, together with the guidelines presented herein, aims to serve as a practical resource for researchers advancing the field of adversarial machine learning and for developers seeking to implement effective defenses in production systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22230v1">DAJ: Data-Reweighted LLM Judge for Test-Time Scaling in Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Test-time scaling for code generation commonly relies on Best-of-N selection, in which multiple candidate solutions are sampled from a base model, and the best one is selected by an LLM judge. However, training reliable LLM judges is challenging due to severe distribution shifts, including imbalances between easy and hard problems, mismatches between training tasks and evaluation benchmarks, and trajectory mismatch arising from training data generated by cheaper models whose behavior differs from that of inference-time models. We propose DAJ, a reasoning-based LLM judge trained with verifiable rewards under a bi-level data-reweighted learning framework. The proposed framework learns data-importance weights (either domain-level or instance-level) to optimize generalization performance on a held-out meta set aligned with target benchmarks. To the best of our knowledge, this is the first application of data reweighting to LLM-as-a-Judge training for test-time scaling. Our approach automatically emphasizes hard problems, in-distribution samples, and trajectory-aligned data, without relying on hand-crafted heuristics. Empirically, DAJ achieves state-of-the-art performance on LiveCodeBench and BigCodeBench, outperforming strong test-time scaling baselines as well as leading proprietary models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22208v1">Stalled, Biased, and Confused: Uncovering Reasoning Failures in LLMs for Cloud-Based Root Cause Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 FORGE 2026
    </div>
    <details class="paper-abstract">
      Root cause analysis (RCA) is essential for diagnosing failures within complex software systems to ensure system reliability. The highly distributed and interdependent nature of modern cloud-based systems often complicates RCA efforts, particularly for multi-hop fault propagation, where symptoms appear far from their true causes. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance automated RCA. However, their practical value for RCA depends on the fidelity of reasoning and decision-making. Existing work relies on historical incident corpora, operates directly on high-volume telemetry beyond current LLM capacity, or embeds reasoning inside complex multi-agent pipelines -- conditions that obscure whether failures arise from reasoning itself or from peripheral design choices. We present a focused empirical evaluation that isolates an LLM's reasoning behavior. We design a controlled experimental framework that foregrounds the LLM by using a simplified experimental setting. We evaluate six LLMs under two agentic workflows (ReAct and Plan-and-Execute) and a non-agentic baseline on two real-world case studies (GAIA and OpenRCA). In total, we executed 48,000 simulated failure scenarios, totaling 228 days of execution time. We measure both root-cause accuracy and the quality of intermediate reasoning traces. We produce a labeled taxonomy of 16 common RCA reasoning failures and use an LLM-as-a-Judge for annotation. Our results clarify where current open-source LLMs succeed and fail in multi-hop RCA, quantify sensitivity to input data modalities, and identify reasoning failures that predict final correctness. Together, these contributions provide transparent and reproducible empirical results and a failure taxonomy to guide future work on reasoning-driven system diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.08862v2">LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted at AAAI 2025
    </div>
    <details class="paper-abstract">
      We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06052v3">DCP-Bench-Open: Evaluating LLMs for Constraint Modelling of Discrete Combinatorial Problems</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 This version is currently submitted and it is under review. For CP-Bench (the paper accepted at ECAI25), please refer to the previous version of this entry (v2)
    </div>
    <details class="paper-abstract">
      Discrete Combinatorial Problems (DCPs) are prevalent in industrial decision-making and optimisation. However, while constraint solving technologies for DCPs have advanced significantly, the core process of formalising them, namely constraint modelling, requires significant expertise and remains a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) to transform combinatorial problem descriptions into executable constraint models. However, the existing evaluation datasets for discrete constraint modelling are often limited to small, homogeneous, or domain-specific problems, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing DCP-Bench-Open, a novel benchmark that includes a diverse set of well-known discrete combinatorial problems sourced from the Constraint Programming (CP) and Operations Research (OR) communities, structured explicitly for evaluating LLM-driven constraint modelling. With this dataset, and given the variety of modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax. Notably, the results show higher performance when modelling with a high-level Python-based framework. Additionally, we systematically evaluate the use of prompt-based and inference-time compute methods across different LLMs, which further increase accuracy, reaching up to 91% on this highly challenging benchmark. DCP-Bench-Open is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07972v2">HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted to ICLR'26
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20573v3">Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 1.7$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09630v2">In-Context Bias Propagation in LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for synthetic tabular data generation through in-context learning (ICL), offering a practical solution for data augmentation in data scarce scenarios. While prior work has shown the potential of LLMs to improve downstream task performance through augmenting underrepresented groups, these benefits often assume access to a subset of unbiased in-context examples, representative of the real dataset. In real-world settings, however, data is frequently noisy and demographically skewed. In this paper, we systematically study how statistical biases within in-context examples propagate to the distribution of synthetic tabular data, showing that even mild in-context biases lead to global statistical distortions. We further introduce an adversarial scenario where a malicious contributor can inject bias into the synthetic dataset via a subset of in-context examples, ultimately compromising the fairness of downstream classifiers for a targeted and protected subgroup. Finally, we evaluate mitigation strategies based on preprocessing in-context examples, demonstrating that while such interventions can attenuate disparity, the inherent sensitivity of LLMs to adversarial prompts remains a persistent challenge. Our findings highlight a critical new vulnerability in LLM-based data generation pipelines within sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09785v2">AI Annotation Orchestration: Evaluating LLM verifiers to Improve the Quality of LLM Annotations in Learning Analytics</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to annotate learning interactions, yet concerns about reliability limit their utility. We test whether verification-oriented orchestration-prompting models to check their own labels (self-verification) or audit one another (cross-verification)-improves qualitative coding of tutoring discourse. Using transcripts from 30 one-to-one math sessions, we compare three production LLMs (GPT, Claude, Gemini) under three conditions: unverified annotation, self-verification, and cross-verification across all orchestration configurations. Outputs are benchmarked against a blinded, disagreement-focused human adjudication using Cohen's kappa. Overall, orchestration yields a 58 percent improvement in kappa. Self-verification nearly doubles agreement relative to unverified baselines, with the largest gains for challenging tutor moves. Cross-verification achieves a 37 percent improvement on average, with pair- and construct-dependent effects: some verifier-annotator pairs exceed self-verification, while others reduce alignment, reflecting differences in verifier strictness. We contribute: (1) a flexible orchestration framework instantiating control, self-, and cross-verification; (2) an empirical comparison across frontier LLMs on authentic tutoring data with blinded human "gold" labels; and (3) a concise notation, verifier(annotator) (e.g., Gemini(GPT) or Claude(Claude)), to standardize reporting and make directional effects explicit for replication. Results position verification as a principled design lever for reliable, scalable LLM-assisted annotation in Learning Analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16033v2">Helping Johnny Make Sense of Privacy Policies with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 21 pages, 3 figures, 3 tables, ACM CHI 2026
    </div>
    <details class="paper-abstract">
      Understanding and engaging with privacy policies is crucial for online privacy, yet these documents remain notoriously complex and difficult to navigate. We present PRISMe, an interactive browser extension that combines LLM-based policy assessment with a dashboard and customizable chat interface, enabling users to skim quick overviews or explore policy details in depth while browsing. We conduct a user study (N=22) with participants of diverse privacy knowledge to investigate how users interpret the tool's explanations and how it shapes their engagement with privacy policies, identifying distinct interaction patterns. Participants valued the clear overviews and conversational depth, but flagged some issues, particularly adversarial robustness and hallucination risks. Thus, we investigate how a retrieval-augmented generation (RAG) approach can alleviate issues by re-running the chat queries from the study. Our findings surface design challenges as well as technical trade-offs, contributing actionable insights for developing future user-centered, trustworthy privacy policy analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11558v4">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20757v1">Persona Prompting as a Lens on LLM Social Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 9 Pages, EACL main
    </div>
    <details class="paper-abstract">
      For socially sensitive tasks like hate speech detection, the quality of explanations from Large Language Models (LLMs) is crucial for factors like user trust and model alignment. While Persona prompting (PP) is increasingly used as a way to steer model towards user-specific generation, its effect on model rationales remains underexplored. We investigate how LLM-generated rationales vary when conditioned on different simulated demographic personas. Using datasets annotated with word-level rationales, we measure agreement with human annotations from different demographic groups, and assess the impact of PP on model bias and human alignment. Our evaluation across three LLMs results reveals three key findings: (1) PP improving classification on the most subjective task (hate speech) but degrading rationale quality. (2) Simulated personas fail to align with their real-world demographic counterparts, and high inter-persona agreement shows models are resistant to significant steering. (3) Models exhibit consistent demographic biases and a strong tendency to over-flag content as harmful, regardless of PP. Our findings reveal a critical trade-off: while PP can improve classification in socially-sensitive tasks, it often comes at the cost of rationale quality and fails to mitigate underlying biases, urging caution in its application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20755v1">ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted in the 9th Annual Conference on Machine Learning and Systems (MLSys 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research to production, understanding how inference engines behave in real time has become both essential and elusive. Unlike general-purpose engines such as ONNX Runtime, today's LLM inference systems offer little operator-level visibility, leaving developers blind to where time and resources go. Even basic questions -- is this workload memory-bound or compute-bound? -- often remain unanswered. To close this gap, we develop a fine-grained, non-intrusive profiling framework for modern LLM inference engines, exemplified by llama.cpp but applicable to similar runtime architectures. Built on extended Berkeley Packet Filter (eBPF) technology, our system dynamically attaches probes to runtime functions across multiple layers -- without modifying or recompiling the source. It transforms collected traces into rich visualizations of operators, graphs, timelines, and hardware counter trends, exposing how dense inference, Mixture-of-Experts routing, and operator offloading behave in practice. With less than 4% runtime overhead and high profiling fidelity, our framework makes LLM inference both transparent and diagnosable, turning performance profiling into a practical tool for optimization, scheduling, and resource-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14979v3">What Matters in LLM-Based Feature Extractor for Recommender? A Systematic Analysis of Prompts, Models, and Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 9 pages. Keywords: Recommender Systems, Large Language Models, Sequential Recommendation, Feature Extraction
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to generate semantic features has been demonstrated as a powerful paradigm for enhancing Sequential Recommender Systems (SRS). This typically involves three stages: processing item text, extracting features with LLMs, and adapting them for downstream models. However, existing methods vary widely in prompting, architecture, and adaptation strategies, making it difficult to fairly compare design choices and identify what truly drives performance. In this work, we propose RecXplore, a modular analytical framework that decomposes the LLM-as-feature-extractor pipeline into four modules: data processing, semantic feature extraction, feature adaptation, and sequential modeling. Instead of proposing new techniques, RecXplore revisits and organizes established methods, enabling systematic exploration of each module in isolation. Experiments on four public datasets show that simply combining the best designs from existing techniques without exhaustive search yields up to 18.7% relative improvement in NDCG@5 and 12.7% in HR@5 over strong baselines. These results underscore the utility of modular benchmarking for identifying effective design patterns and promoting standardized research in LLM-enhanced recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20745v1">HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework for Extremely Low-Bit LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, deployment is increasingly bottlenecked by the memory wall, motivating a shift toward extremely low-bit quantization. However, most quantization-aware training (QAT) methods apply hard rounding and the straight-through estimator (STE) from the beginning of the training, which prematurely discretizes the optimization landscape and induces persistent gradient mismatch between latent weights and quantized weights, hindering effective optimization of quantized models. To address this, we propose Hestia, a Hessian-guided differentiable QAT framework for extremely low-bit LLMs, which replaces the rigid step function with a temperature-controlled softmax relaxation to maintain gradient flow early in training while progressively hardening quantization. Furthermore, Hestia leverages a tensor-wise Hessian trace metric as a lightweight curvature signal to drive fine-grained temperature annealing, enabling sensitivity-aware discretization across the model. Evaluations on Llama-3.2 show that Hestia consistently outperforms existing ternary QAT baselines, yielding average zero-shot improvements of 5.39% and 4.34% for the 1B and 3B models. These results indicate that Hessian-guided relaxation effectively recovers representational capacity, establishing a more robust training path for 1.58-bit LLMs. The code is available at https://github.com/hestia2026/Hestia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20731v1">QueerGen: How LLMs Reflect Societal Norms on Gender and Sexuality in Sentence Completion Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      This paper examines how Large Language Models (LLMs) reproduce societal norms, particularly heterocisnormativity, and how these norms translate into measurable biases in their text generations. We investigate whether explicit information about a subject's gender or sexuality influences LLM responses across three subject categories: queer-marked, non-queer-marked, and the normalized "unmarked" category. Representational imbalances are operationalized as measurable differences in English sentence completions across four dimensions: sentiment, regard, toxicity, and prediction diversity. Our findings show that Masked Language Models (MLMs) produce the least favorable sentiment, higher toxicity, and more negative regard for queer-marked subjects. Autoregressive Language Models (ARLMs) partially mitigate these patterns, while closed-access ARLMs tend to produce more harmful outputs for unmarked subjects. Results suggest that LLMs reproduce normative social assumptions, though the form and degree of bias depend strongly on specific model characteristics, which may redistribute, but not eliminate, representational harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20704v1">Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 34 pages, 20 figures. Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to curate bibliographies, raising the question: are their reference lists distinguishable from human ones? We build paired citation graphs, ground truth and GPT-4o-generated (from parametric knowledge), for 10,000 focal papers ($\approx$ 275k references) from SciSciNet, and added a field-matched random baseline that preserves out-degree and field distributions while breaking latent structure. We compare (i) structure-only node features (degree/closeness/eigenvector centrality, clustering, edge count) with (ii) 3072-D title/abstract embeddings, using an RF on graph-level aggregates and Graph Neural Networks with node features. Structure alone barely separates GPT from ground truth (RF accuracy $\approx$ 0.60) despite cleanly rejecting the random baseline ($\approx$ 0.89--0.92). By contrast, embeddings sharply increase separability: RF on aggregated embeddings reaches $\approx$ 0.83, and GNNs with embedding node features achieve 93\% test accuracy on GPT vs.\ ground truth. We show the robustness of our findings by replicating the pipeline with Claude Sonnet 4.5 and with multiple embedding models (OpenAI and SPECTER), with RF separability for ground truth vs.\ Claude $\approx 0.77$ and clean rejection of the random baseline. Thus, LLM bibliographies, generated purely from parametric knowledge, closely mimic human citation topology, but leave detectable semantic fingerprints; detection and debiasing should target content signals rather than global graph structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23609v2">Marriage Discourse on Chinese Social Media: An LLM-assisted Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      China's marriage registrations have declined substantially, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. This study examined sentiment and moral elements underlying 219,358 marriage-related posts from Weibo and Xiaohongshu using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral elements (autonomy, community, divinity). Results revealed platform differences: Weibo leaned toward positive sentiment, while Xiaohongshu was predominantly neutral. Most posts lacked explicit moral framing. However, when moral elements were invoked, significant associations with sentiment emerged. Posts invoking autonomy and community were predominantly negative, whereas divinity-framed posts tended toward positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations contribute to negative marriage attitudes in contemporary China, offering insights for culturally informed policies addressing marriage decline.
    </details>
</div>
