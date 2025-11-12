# llm - 2025_02

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21321v1">LLM Post-Training: A Deep Dive into Reasoning Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 31 pages, 7 figures, 3 tables, 375 references
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed the natural language processing landscape and brought to life diverse applications. Pretraining on vast web-scale data has laid the foundation for these models, yet the research community is now increasingly shifting focus toward post-training techniques to achieve further breakthroughs. While pretraining provides a broad linguistic foundation, post-training methods enable LLMs to refine their knowledge, improve reasoning, enhance factual accuracy, and align more effectively with user intents and ethical considerations. Fine-tuning, reinforcement learning, and test-time scaling have emerged as critical strategies for optimizing LLMs performance, ensuring robustness, and improving adaptability across various real-world tasks. This survey provides a systematic exploration of post-training methodologies, analyzing their role in refining LLMs beyond pretraining, addressing key challenges such as catastrophic forgetting, reward hacking, and inference-time trade-offs. We highlight emerging directions in model alignment, scalable adaptation, and inference-time reasoning, and outline future research directions. We also provide a public repository to continually track developments in this fast-evolving field: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21239v1">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring white-box access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21231v1">ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 12 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Scaling long-context ability is essential for Large Language Models (LLMs). To amortize the memory consumption across multiple devices in long-context training, inter-data partitioning (a.k.a. Data Parallelism) and intra-data partitioning (a.k.a. Context Parallelism) are commonly used. Current training frameworks predominantly treat the two techniques as orthogonal, and establish static communication groups to organize the devices as a static mesh (e.g., a 2D mesh). However, the sequences for LLM training typically vary in lengths, no matter for texts, multi-modalities or reinforcement learning. The mismatch between data heterogeneity and static mesh causes redundant communication and imbalanced computation, degrading the training efficiency. In this work, we introduce ByteScale, an efficient, flexible, and scalable LLM training framework for large-scale mixed training of long and short sequences. The core of ByteScale is a novel parallelism strategy, namely Hybrid Data Parallelism (HDP), which unifies the inter- and intra-data partitioning with a dynamic mesh design. In particular, we build a communication optimizer, which eliminates the redundant communication for short sequences by data-aware sharding and dynamic communication, and further compresses the communication cost for long sequences by selective offloading. Besides, we also develop a balance scheduler to mitigate the imbalanced computation by parallelism-aware data assignment. We evaluate ByteScale with the model sizes ranging from 7B to 141B, context lengths from 256K to 2048K, on a production cluster with more than 12,000 GPUs. Experiment results show that ByteScale outperforms the state-of-the-art training system by up to 7.89x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21208v1">ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Recent research has shown that LLM performance on reasoning tasks can be enhanced by scaling test-time compute. One promising approach, particularly with decomposable problems, involves arranging intermediate solutions as a graph on which transformations are performed to explore the solution space. However, prior works rely on pre-determined, task-specific transformation schedules which are subject to a set of searched hyperparameters. In this work, we view thought graph transformations as actions in a Markov decision process, and implement policy agents to drive effective action policies for the underlying reasoning LLM agent. In particular, we investigate the ability for another LLM to act as a policy agent on thought graph environments and introduce ARIES, a multi-agent architecture for reasoning with LLMs. In ARIES, reasoning LLM agents solve decomposed subproblems, while policy LLM agents maintain visibility of the thought graph states, and dynamically adapt the problem-solving strategy. Through extensive experiments, we observe that using off-the-shelf LLMs as policy agents with no supervised fine-tuning (SFT) can yield up to $29\%$ higher accuracy on HumanEval relative to static transformation schedules, as well as reducing inference costs by $35\%$ and avoid any search requirements. We also conduct a thorough analysis of observed failure modes, highlighting that limitations on LLM sizes and the depth of problem decomposition can be seen as challenges to scaling LLM-guided reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11843v3">From Commands to Prompts: LLM-based Semantic File System for AIOS</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in the development of intelligent applications and systems such as LLM-based agents and agent operating systems (AIOS). However, when these applications and systems interact with the underlying file system, the file system still remains the traditional paradigm: reliant on manual navigation through precise commands. This paradigm poses a bottleneck to the usability of these systems as users are required to navigate complex folder hierarchies and remember cryptic file names. To address this limitation, we propose an LLM-based semantic file system ( LSFS ) for prompt-driven file management. Unlike conventional approaches, LSFS incorporates LLMs to enable users or agents to interact with files through natural language prompts, facilitating semantic file management. At the macro-level, we develop a comprehensive API set to achieve semantic file management functionalities, such as semantic file retrieval, file update monitoring and summarization, and semantic file rollback). At the micro-level, we store files by constructing semantic indexes for them, design and implement syscalls of different semantic operations (e.g., CRUD, group by, join) powered by vector database. Our experiments show that LSFS offers significant improvements over traditional file systems in terms of user convenience, the diversity of supported functions, and the accuracy and efficiency of file operations. Additionally, with the integration of LLM, our system enables more intelligent file management tasks, such as content summarization and version comparison, further enhancing its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03664v4">LLMs in the Heart of Differential Testing: A Case Study on a Medical Rule Engine</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 12 pages, 6 figures, 4 tables, 1 listing, revised arguments
    </div>
    <details class="paper-abstract">
      The Cancer Registry of Norway (CRN) uses an automated cancer registration support system (CaReSS) to support core cancer registry activities, i.e, data capture, data curation, and producing data products and statistics for various stakeholders. GURI is a core component of CaReSS, which is responsible for validating incoming data with medical rules. Such medical rules are manually implemented by medical experts based on medical standards, regulations, and research. Since large language models (LLMs) have been trained on a large amount of public information, including these documents, they can be employed to generate tests for GURI. Thus, we propose an LLM-based test generation and differential testing approach (LLMeDiff) to test GURI. We experimented with four different LLMs, two medical rule engine implementations, and 58 real medical rules to investigate the hallucination, success, time efficiency, and robustness of the LLMs to generate tests, and these tests' ability to find potential issues in GURI. Our results showed that GPT-3.5 hallucinates the least, is the most successful, and is generally the most robust; however, it has the worst time efficiency. Our differential testing revealed 22 medical rules where implementation inconsistencies were discovered (e.g., regarding handling rule versions). Finally, we provide insights for practitioners and researchers based on the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06842v2">SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks, yet their training remains highly resource-intensive and susceptible to critical challenges such as training instability. A predominant source of this instability stems from gradient and loss spikes, which disrupt the learning process, often leading to costly interventions like checkpoint recovery and experiment restarts, further amplifying inefficiencies. This paper presents a comprehensive investigation into gradient spikes observed during LLM training, revealing their prevalence across multiple architectures and datasets. Our analysis shows that these spikes can be up to $1000\times$ larger than typical gradients, substantially deteriorating model performance. To address this issue, we propose Spike-Aware Adam with Momentum Reset SPAM, a novel optimizer designed to counteract gradient spikes through momentum reset and spike-aware gradient clipping. Extensive experiments, including both pre-training and fine-tuning, demonstrate that SPAM consistently surpasses Adam and its variants across various tasks, including (1) LLM pre-training from 60M to 1B, (2) 4-bit LLM pre-training,(3) reinforcement learning, and (4) Time Series Forecasting. Additionally, SPAM facilitates memory-efficient training by enabling sparse momentum, where only a subset of momentum terms are maintained and updated. When operating under memory constraints, SPAM outperforms state-of-the-art memory-efficient optimizers such as GaLore and Adam-Mini. Our work underscores the importance of mitigating gradient spikes in LLM training and introduces an effective optimization strategy that enhances both training stability and resource efficiency at scale. Code is available at https://github.com/TianjinYellow/SPAM-Optimizer.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04755v4">LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing that our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21092v1">An LLM-based Delphi Study to Predict GenAI Evolution</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Predicting the future trajectory of complex and rapidly evolving systems remains a significant challenge, particularly in domains where data is scarce or unreliable. This study introduces a novel approach to qualitative forecasting by leveraging Large Language Models to conduct Delphi studies. The methodology was applied to explore the future evolution of Generative Artificial Intelligence, revealing insights into key factors such as geopolitical tensions, economic disparities, regulatory frameworks, and ethical considerations. The results highlight how LLM-based Delphi studies can facilitate structured scenario analysis, capturing diverse perspectives while mitigating issues such as respondent fatigue. However, limitations emerge in terms of knowledge cutoffs, inherent biases, and sensitivity to initial conditions. While the approach provides an innovative means for structured foresight, this method could be also considered as a novel form of reasoning. further research is needed to refine its ability to manage heterogeneity, improve reliability, and integrate external data sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21068v1">GUIDE: LLM-Driven GUI Generation Decomposition for Automated Prototyping</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      GUI prototyping serves as one of the most valuable techniques for enhancing the elicitation of requirements and facilitating the visualization and refinement of customer needs. While GUI prototyping has a positive impact on the software development process, it simultaneously demands significant effort and resources. The emergence of Large Language Models (LLMs) with their impressive code generation capabilities offers a promising approach for automating GUI prototyping. Despite their potential, there is a gap between current LLM-based prototyping solutions and traditional user-based GUI prototyping approaches which provide visual representations of the GUI prototypes and direct editing functionality. In contrast, LLMs and related generative approaches merely produce text sequences or non-editable image output, which lacks both mentioned aspects and therefore impede supporting GUI prototyping. Moreover, minor changes requested by the user typically lead to an inefficient regeneration of the entire GUI prototype when using LLMs directly. In this work, we propose GUIDE, a novel LLM-driven GUI generation decomposition approach seamlessly integrated into the popular prototyping framework Figma. Our approach initially decomposes high-level GUI descriptions into fine-granular GUI requirements, which are subsequently translated into Material Design GUI prototypes, enabling higher controllability and more efficient adaption of changes. To efficiently conduct prompting-based generation of Material Design GUI prototypes, we propose a retrieval-augmented generation approach to integrate the component library. Our preliminary evaluation demonstrates the effectiveness of GUIDE in bridging the gap between LLM generation capabilities and traditional GUI prototyping workflows, offering a more effective and controlled user-based approach to LLM-driven GUI prototyping. Video: https://youtu.be/C9RbhMxqpTU
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15835v2">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed to produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using one of the latest LLMs on a popular code generation dataset. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21030v1">Beyond Words: A Latent Memory Approach to Internal Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have popularized the chain-of-thought (CoT) paradigm, in which models produce explicit reasoning steps in natural language. Although this approach improves interpretability and facilitates external auditing, it may not represent the most computationally efficient method for internal reasoning. In contrast, human cognition relies on implicit mental representations that recall past sensory and episodic information without requiring complete verbalization. In this paper, we propose a framework that integrates implicit mental representations into the internal reasoning processes of LLMs. Preliminary experiments indicate that incorporating an Implicit Memory Module (IMM) into a simple GPT model yields a reduction of between 35% and 57% in final training loss compared to a regular GPT baseline. The addition of an explicit interpretability channel (e.g., a chain-of-thought decoder) is straightforward to implement within this approach. We outline theoretical foundations, propose technical mechanisms to scale the memory module, and discuss how these ideas may lead to more efficient and robust reasoning, with optional future extensions for explicit auditability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20984v1">UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at https://github.com/tongwu17/SemEval-2025-Task1-UoR-NCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20973v1">Arabizi vs LLMs: Can the Genie Understand the Language of Aladdin?</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Submitted to MT Summit 2025
    </div>
    <details class="paper-abstract">
      In this era of rapid technological advancements, communication continues to evolve as new linguistic phenomena emerge. Among these is Arabizi, a hybrid form of Arabic that incorporates Latin characters and numbers to represent the spoken dialects of Arab communities. Arabizi is widely used on social media and allows people to communicate in an informal and dynamic way, but it poses significant challenges for machine translation due to its lack of formal structure and deeply embedded cultural nuances. This case study arises from a growing need to translate Arabizi for gisting purposes. It evaluates the capacity of different LLMs to decode and translate Arabizi, focusing on multiple Arabic dialects that have rarely been studied up until now. Using a combination of human evaluators and automatic metrics, this research project investigates the model's performance in translating Arabizi into both Modern Standard Arabic and English. Key questions explored include which dialects are translated most effectively and whether translations into English surpass those into Arabic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20968v1">Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 25 pages, 10 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Role-playing enables large language models (LLMs) to engage users in immersive and personalized interactions, but it also introduces significant safety risks. Existing role-play fine-tuning techniques improve role adaptability but may degrade safety performance, particularly for villainous characters. In this work, we conduct the first comprehensive assessment of role-play fine-tuning risks by training 95 role-specific LLMs using RoleBench. Our experiments reveal that role-play fine-tuning leads to a noticeable decline in safety performance, with safety risks varying based on character traits. To tackle this challenge, we propose Safety-Aware Role-Play Fine-Tuning (SaRFT), a novel method designed to balance role-playing capabilities and safety. Extensive experiments on LLaMA-3-8B-Instruct, Gemma-2-9B-it, and Qwen2.5-7B-Instruct demonstrate that SaRFT consistently outperforms state-of-the-art baselines under both LoRA and full-parameter fine-tuning settings. Our findings highlight the necessity of role-adaptive safety measures and provide insights into mitigating role-specific safety risks in role-playing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15127v2">Cost-Effective, High-Performance Open-Source LLMs via Optimized Context Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 14 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in healthcare promise transformation, yet adoption is limited by concerns over factual accuracy and the high cost of proprietary models. This study demonstrates that optimized context retrieval unlocks cost-effective, high-performance healthcare AI using open-source LLMs, achieving a significantly improved cost-accuracy Pareto frontier for medical question answering and showcasing that open models can rival proprietary systems at a fraction of the cost. A key contribution is OpenMedQA, a novel benchmark for open-ended medical question answering that overcomes the limitations of multiple-choice formats - formats that we show lead to performance degradation in open-ended settings and often lack clinical realism. Further contributions include: (1) practical guidelines for implementing optimized context retrieval; (2) empirical validation of enhanced cost-effectiveness via the improved Pareto frontier; (3) the introduction of OpenMedQA for rigorous evaluation of open-ended medical QA; and (4) the release of prompt_engine alongside CoT/ToT/Thinking databases as community resources for cost-effective healthcare AI. Advancing optimized retrieval and open-ended QA benchmarking, we pave the way for more accessible and impactful LLM-powered healthcare solutions. All the materials have been made public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01570v3">Small Models are LLM Knowledge Triggers on Medical Tabular Prediction</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Accepted to ICLR 2025. Codes will be available at https://github.com/jyansir/sersal
    </div>
    <details class="paper-abstract">
      Recent development in large language models (LLMs) has demonstrated impressive domain proficiency on unstructured textual or multi-modal tasks. However, despite with intrinsic world knowledge, their application on structured tabular data prediction still lags behind, primarily due to the numerical insensitivity and modality discrepancy that brings a gap between LLM reasoning and statistical tabular learning. Unlike textual or vision data (e.g., electronic clinical notes or medical imaging data), tabular data is often presented in heterogeneous numerical values (e.g., CBC reports). This ubiquitous data format requires intensive expert annotation, and its numerical nature limits LLMs' capability to effectively transfer untapped domain expertise. In this paper, we propose SERSAL, a general self-prompting method by synergy learning with small models to enhance LLM tabular prediction in an unsupervised manner. Specifically, SERSAL utilizes the LLM's prior outcomes as original soft noisy annotations, which are dynamically leveraged to teach a better small student model. Reversely, the outcomes from the trained small model are used to teach the LLM to further refine its real capability. This process can be repeatedly applied to gradually distill refined knowledge for continuous progress. Comprehensive experiments on widely used medical domain tabular datasets show that, without access to gold labels, applying SERSAL to OpenAI GPT reasoning process attains substantial improvement compared to linguistic prompting methods, which serves as an orthogonal direction for tabular LLM, and increasing prompting bonus is observed as more powerful LLMs appear.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20866v1">Better Benchmarking LLMs for Zero-Shot Dependency Parsing</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Accepted at NoDaLiDa/Baltic-HLT 2025
    </div>
    <details class="paper-abstract">
      While LLMs excel in zero-shot tasks, their performance in linguistic challenges like syntactic parsing has been less scrutinized. This paper studies state-of-the-art open-weight LLMs on the task by comparing them to baselines that do not have access to the input sentence, including baselines that have not been used in this context such as random projective trees or optimal linear arrangements. The results show that most of the tested LLMs cannot outperform the best uninformed baselines, with only the newest and largest versions of LLaMA doing so for most languages, and still achieving rather low performance. Thus, accurate zero-shot syntactic parsing is not forthcoming with open LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15601v2">WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20825v1">LADs: Leveraging LLMs for AI-Driven DevOps</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 17 pages with Appendix, 8 figures, and 7 tables. This paper is currently Under Review
    </div>
    <details class="paper-abstract">
      Automating cloud configuration and deployment remains a critical challenge due to evolving infrastructures, heterogeneous hardware, and fluctuating workloads. Existing solutions lack adaptability and require extensive manual tuning, leading to inefficiencies and misconfigurations. We introduce LADs, the first LLM-driven framework designed to tackle these challenges by ensuring robustness, adaptability, and efficiency in automated cloud management. Instead of merely applying existing techniques, LADs provides a principled approach to configuration optimization through in-depth analysis of what optimization works under which conditions. By leveraging Retrieval-Augmented Generation, Few-Shot Learning, Chain-of-Thought, and Feedback-Based Prompt Chaining, LADs generates accurate configurations and learns from deployment failures to iteratively refine system settings. Our findings reveal key insights into the trade-offs between performance, cost, and scalability, helping practitioners determine the right strategies for different deployment scenarios. For instance, we demonstrate how prompt chaining-based adaptive feedback loops enhance fault tolerance in multi-tenant environments and how structured log analysis with example shots improves configuration accuracy. Through extensive evaluations, LADs reduces manual effort, optimizes resource utilization, and improves system reliability. By open-sourcing LADs, we aim to drive further innovation in AI-powered DevOps automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17749v2">Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs. We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_paraphrased_code_via_coding_style_features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12561v2">UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental yet challenging (e.g., inflexible to iterate the study design flaws and hard to recruit study participants) research method for user experience (UX) researchers to evaluate a web design. Recent advances in Large Language Model-simulated Agent (LLM-Agent) research inspired us to design UXAgent to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human subject study. Our system features an LLM-Agent module and a universal browser connector module so that UX researchers can automatically generate thousands of simulated users to test the target website. The results are shown in qualitative (e.g., interviewing how an agent thinks ), quantitative (e.g., # of actions), and video recording formats for UX researchers to analyze. Through a heuristic user evaluation with five UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent-assisted UX study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19820v2">Foot-In-The-Door: A Multi-turn Jailbreak for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00418v2">Self-Evolved Reward Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 23 pages,6 figures,Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. These methods can be costly and may introduce biases that affect the language model's responses. As language models improve, human input may become less effective in further enhancing their performance. In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20681v3">No Free Lunch Theorem for Privacy-Preserving LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Individuals and businesses have been significantly benefited by Large Language Models (LLMs) including PaLM, Gemini and ChatGPT in various ways. For example, LLMs enhance productivity, reduce costs, and enable us to focus on more valuable tasks. Furthermore, LLMs possess the capacity to sift through extensive datasets, uncover underlying patterns, and furnish critical insights that propel the frontiers of technology and science. However, LLMs also pose privacy concerns. Users' interactions with LLMs may expose their sensitive personal or company information. A lack of robust privacy safeguards and legal frameworks could permit the unwarranted intrusion or improper handling of individual data, thereby risking infringements of privacy and the theft of personal identities. To ensure privacy, it is essential to minimize the dependency between shared prompts and private information. Various randomization approaches have been proposed to protect prompts' privacy, but they may incur utility loss compared to unprotected LLMs prompting. Therefore, it is essential to evaluate the balance between the risk of privacy leakage and loss of utility when conducting effective protection mechanisms. The current study develops a framework for inferring privacy-protected Large Language Models (LLMs) and lays down a solid theoretical basis for examining the interplay between privacy preservation and utility. The core insight is encapsulated within a theorem that is called as the NFL (abbreviation of the word No-Free-Lunch) Theorem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20635v1">Can LLM Assist in the Evaluation of the Quality of Machine Learning Explanations?</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      EXplainable machine learning (XML) has recently emerged to address the mystery mechanisms of machine learning (ML) systems by interpreting their 'black box' results. Despite the development of various explanation methods, determining the most suitable XML method for specific ML contexts remains unclear, highlighting the need for effective evaluation of explanations. The evaluating capabilities of the Transformer-based large language model (LLM) present an opportunity to adopt LLM-as-a-Judge for assessing explanations. In this paper, we propose a workflow that integrates both LLM-based and human judges for evaluating explanations. We examine how LLM-based judges evaluate the quality of various explanation methods and compare their evaluation capabilities to those of human judges within an iris classification scenario, employing both subjective and objective metrics. We conclude that while LLM-based judges effectively assess the quality of explanations using subjective metrics, they are not yet sufficiently developed to replace human judges in this role.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20633v1">Are LLMs Ready for Practical Adoption for Assertion Generation?</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 7 Pages, 9 Figures, Accepted in DATE 2025. arXiv admin note: substantial text overlap with arXiv:2406.18627
    </div>
    <details class="paper-abstract">
      Assertions have been the de facto collateral for simulation-based and formal verification of hardware designs for over a decade. The quality of hardware verification, i.e., detection and diagnosis of corner-case design bugs, is critically dependent on the quality of the assertions. With the onset of generative AI such as Transformers and Large-Language Models (LLMs), there has been a renewed interest in developing novel, effective, and scalable techniques of generating functional and security assertions from design source code. While there have been recent works that use commercial-of-the-shelf (COTS) LLMs for assertion generation, there is no comprehensive study in quantifying the effectiveness of LLMs in generating syntactically and semantically correct assertions. In this paper, we first discuss AssertionBench from our prior work, a comprehensive set of designs and assertions to quantify the goodness of a broad spectrum of COTS LLMs for the task of assertion generations from hardware design source code. Our key insight was that COTS LLMs are not yet ready for prime-time adoption for assertion generation as they generate a considerable fraction of syntactically and semantically incorrect assertions. Motivated by the insight, we propose AssertionLLM, a first of its kind LLM model, specifically fine-tuned for assertion generation. Our initial experimental results show that AssertionLLM considerably improves the semantic and syntactic correctness of the generated assertions over COTS LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20620v1">Rectifying Belief Space via Unlearning to Harness LLMs' Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can exhibit advanced reasoning yet still generate incorrect answers. We hypothesize that such errors frequently stem from spurious beliefs, propositions the model internally considers true but are incorrect. To address this, we propose a method to rectify the belief space by suppressing these spurious beliefs while simultaneously enhancing true ones, thereby enabling more reliable inferences. Our approach first identifies the beliefs that lead to incorrect or correct answers by prompting the model to generate textual explanations, using our Forward-Backward Beam Search (FBBS). We then apply unlearning to suppress the identified spurious beliefs and enhance the true ones, effectively rectifying the model's belief space. Empirical results on multiple QA datasets and LLMs show that our method corrects previously misanswered questions without harming overall model performance. Furthermore, our approach yields improved generalization on unseen data, suggesting that rectifying a model's belief space is a promising direction for mitigating errors and enhancing overall reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11285v2">Zero-Shot Automatic Annotation and Instance Segmentation using LLM-Generated Datasets: Eliminating Field Imaging and Manual Annotation for Deep Learning Model Development</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Currently, deep learning-based instance segmentation for various applications (e.g., Agriculture) is predominantly performed using a labor-intensive process involving extensive field data collection using sophisticated sensors, followed by careful manual annotation of images, presenting significant logistical and financial challenges to researchers and organizations. The process also slows down the model development and training process. In this study, we presented a novel method for deep learning-based instance segmentation of apples in commercial orchards that eliminates the need for labor-intensive field data collection and manual annotation. Utilizing a Large Language Model (LLM), we synthetically generated orchard images and automatically annotated them using the Segment Anything Model (SAM) integrated with a YOLO11 base model. This method significantly reduces reliance on physical sensors and manual data processing, presenting a major advancement in "Agricultural AI". The synthetic, auto-annotated dataset was used to train the YOLO11 model for Apple instance segmentation, which was then validated on real orchard images. The results showed that the automatically generated annotations achieved a Dice Coefficient of 0.9513 and an IoU of 0.9303, validating the accuracy and overlap of the mask annotations. All YOLO11 configurations, trained solely on these synthetic datasets with automated annotations, accurately recognized and delineated apples, highlighting the method's efficacy. Specifically, the YOLO11m-seg configuration achieved a mask precision of 0.902 and a mask mAP@50 of 0.833 on test images collected from a commercial orchard. Additionally, the YOLO11l-seg configuration outperformed other models in validation on 40 LLM-generated images, achieving the highest mask precision and mAP@50 metrics. Keywords: YOLO, SAM, SAMv2, YOLO11, YOLOv11, Segment Anything, YOLO-SAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v3">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09724v2">Taming Overconfidence in LLMs: Reward Calibration in RLHF</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Language model calibration refers to the alignment between the confidence of the model and the actual performance of its responses. While previous studies point out the overconfidence phenomenon in Large Language Models (LLMs) and show that LLMs trained with Reinforcement Learning from Human Feedback (RLHF) are overconfident with a more sharpened output probability, in this study, we reveal that RLHF tends to lead models to express verbalized overconfidence in their own responses. We investigate the underlying cause of this overconfidence and demonstrate that reward models used for Proximal Policy Optimization (PPO) exhibit inherent biases towards high-confidence scores regardless of the actual quality of responses. Building upon this insight, we propose two PPO variants: PPO-M: PPO with Calibrated Reward Modeling and PPO-C: PPO with Calibrated Reward Calculation. PPO-M integrates explicit confidence scores in reward model training, which calibrates reward models to better capture the alignment between response quality and verbalized confidence. PPO-C adjusts the reward score during PPO based on the difference between the current reward and the exponential average of past rewards. Both PPO-M and PPO-C can be seamlessly integrated into the current PPO pipeline and do not require additional golden labels. We evaluate our methods on both Llama3-8B and Mistral-7B across six diverse datasets including multiple-choice and open-ended generation. Experimental results demonstrate that both of our methods can reduce calibration error and maintain performance comparable to standard PPO. We further show that they could preserve model capabilities in open-ended conversational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10860v2">Zero-shot and Few-shot Learning with Instruction-following LLMs for Claim Matching in Automated Fact-checking</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Published at the 31st International Conference on Computational Linguistics (COLING 2025). Compared to the conference version of the paper, the dataset link is added here & 2 minor typos fixed
    </div>
    <details class="paper-abstract">
      The claim matching (CM) task can benefit an automated fact-checking pipeline by putting together claims that can be resolved with the same fact-check. In this work, we are the first to explore zero-shot and few-shot learning approaches to the task. We consider CM as a binary classification task and experiment with a set of instruction-following large language models (GPT-3.5-turbo, Gemini-1.5-flash, Mistral-7B-Instruct, and Llama-3-8B-Instruct), investigating prompt templates. We introduce a new CM dataset, ClaimMatch, which will be released upon acceptance. We put LLMs to the test in the CM task and find that it can be tackled by leveraging more mature yet similar tasks such as natural language inference or paraphrase detection. We also propose a pipeline for CM, which we evaluate on texts of different lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14953v4">MallowsPO: Fine-Tune Your LLM with Preference Dispersions</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) has recently emerged as a popular approach to improve reinforcement learning with human feedback (RLHF), leading to better techniques to fine-tune large language models (LLM). A weakness of DPO, however, lies in its lack of capability to characterize the diversity of human preferences. Inspired by Mallows' theory of preference ranking, we develop in this paper a new approach, the MallowsPO. A distinct feature of this approach is a dispersion index, which reflects the dispersion of human preference to prompts. We show that existing DPO models can be reduced to special cases of this dispersion index, thus unified with MallowsPO. More importantly, we demonstrate (empirically) how to use this dispersion index to enhance the performance of DPO in a broad array of benchmark tasks, from synthetic bandit selection to controllable generations and dialogues, while maintaining great generalization capabilities. MallowsPO is also compatible with other SOTA offline preference optimization methods, boosting nearly 2\% extra LC win rate when used as a plugin for fine-tuning Llama3-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00231v1">Jawaher: A Multidialectal Dataset of Arabic Proverbs for LLM Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Project GitHub page is accessible at: https://github.com/UBC-NLP/jawaher
    </div>
    <details class="paper-abstract">
      Recent advancements in instruction fine-tuning, alignment methods such as reinforcement learning from human feedback (RLHF), and optimization techniques like direct preference optimization (DPO) have significantly enhanced the adaptability of large language models (LLMs) to user preferences. However, despite these innovations, many LLMs continue to exhibit biases toward Western, Anglo-centric, or American cultures, with performance on English data consistently surpassing that of other languages. This reveals a persistent cultural gap in LLMs, which complicates their ability to accurately process culturally rich and diverse figurative language such as proverbs. To address this, we introduce Jawaher, a benchmark designed to assess LLMs' capacity to comprehend and interpret Arabic proverbs. Jawaher includes proverbs from various Arabic dialects, along with idiomatic translations and explanations. Through extensive evaluations of both open- and closed-source models, we find that while LLMs can generate idiomatically accurate translations, they struggle with producing culturally nuanced and contextually relevant explanations. These findings highlight the need for ongoing model refinement and dataset expansion to bridge the cultural gap in figurative language processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00224v1">À la recherche du sens perdu: your favourite LLM might have more to say than you can understand</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      We report a peculiar observation that LLMs can assign hidden meanings to sequences that seem visually incomprehensible to humans: for example, a nonsensical phrase consisting of Byzantine musical symbols is recognized by gpt-4o as "say abracadabra". Moreover, some models can communicate using these sequences. Some of these meanings are hypothesized to partly originate in the massive spurious correlations due to BPE tokenization. We systematically evaluate the presence of such abilities in a wide range of models: Claude-3.5 Haiku, Claude-3.5 Sonnet (New and Old), Claude-3.7 Sonnet, gpt-4o mini, gpt-4o, o1-mini, Llama-3.3 70B, DeepSeek-R1-Distill-Lllama 70B, Qwen2.5 1.5B, Qwen2.5 32B, Phi-3.5 mini, GigaChat-Max, Vikhr-Llama-3.2 1B. We argue that this observation might have far-reaching consequences for both safety and security of the modern and future LLMs and systems that employ them. As an illustration, we show that applying this method in combination with simple templates is sufficient to jailbreak previous generation models, with ASR = 0.4 on gpt-4o mini. Our code and data artifacts are available at https://github.com/L3G5/llm-hidden-meanings
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01908v1">UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for handling complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements also amplify the risks of adversarial attacks, particularly when LLM agents can access sensitive external functionalities. Moreover, because LLM agents engage in extensive reasoning or planning before executing final actions, manipulating them into performing targeted malicious actions or invoking specific tools remains a significant challenge. Consequently, directly embedding adversarial strings in malicious instructions or injecting malicious prompts into tool interactions has become less effective against modern LLM agents. In this work, we present UDora, a unified red teaming framework designed for LLM Agents that dynamically leverages the agent's own reasoning processes to compel it toward malicious behavior. Specifically, UDora first samples the model's reasoning for the given task, then automatically identifies multiple optimal positions within these reasoning traces to insert targeted perturbations. Subsequently, it uses the modified reasoning as the objective to optimize the adversarial strings. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00151v1">Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 More information about our dataset is available at our project page: https://github.com/UBC-NLP/palm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00134v1">Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00124v1">Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 To appear in Findings of NAACL 2025
    </div>
    <details class="paper-abstract">
      Like most of NLP, models for human-centered NLP tasks -- tasks attempting to assess author-level information -- predominantly use representations derived from hidden states of Transformer-based LLMs. However, what component of the LM is used for the representation varies widely. Moreover, there is a need for Human Language Models (HuLMs) that implicitly model the author and provide a user-level hidden state. Here, we systematically evaluate different ways of representing documents and users using different LM and HuLM architectures to predict task outcomes as both dynamically changing states and averaged trait-like user-level attributes of valence, arousal, empathy, and distress. We find that representing documents as an average of the token hidden states performs the best generally. Further, while a user-level hidden state itself is rarely the best representation, we find its inclusion in the model strengthens token or document embeddings used to derive document- and user-level representations resulting in best performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00096v1">BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 8 main text pages, 5 main figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00093v1">Rethinking LLM Bias Probing Using Lessons from the Social Sciences</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      The proliferation of LLM bias probes introduces three significant challenges: (1) we lack principled criteria for choosing appropriate probes, (2) we lack a system for reconciling conflicting results across probes, and (3) we lack formal frameworks for reasoning about when (and why) probe results will generalize to real user behavior. We address these challenges by systematizing LLM social bias probing using actionable insights from social sciences. We then introduce EcoLevels - a framework that helps (a) determine appropriate bias probes, (b) reconcile conflicting findings across probes, and (c) generate predictions about bias generalization. Overall, we ground our analysis in social science research because many LLM probes are direct applications of human probes, and these fields have faced similar challenges when studying social bias in humans. Based on our work, we suggest how the next generation of LLM bias probing can (and should) benefit from decades of social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01903v1">PsychBench: A comprehensive and professional benchmark for evaluating the performance of LLM-assisted psychiatric clinical practice</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) offers potential solutions to address problems such as shortage of medical resources and low diagnostic consistency in psychiatric clinical practice. Despite this potential, a robust and comprehensive benchmarking framework to assess the efficacy of LLMs in authentic psychiatric clinical environments is absent. This has impeded the advancement of specialized LLMs tailored to psychiatric applications. In response to this gap, by incorporating clinical demands in psychiatry and clinical data, we proposed a benchmarking system, PsychBench, to evaluate the practical performance of LLMs in psychiatric clinical settings. We conducted a comprehensive quantitative evaluation of 16 LLMs using PsychBench, and investigated the impact of prompt design, chain-of-thought reasoning, input text length, and domain-specific knowledge fine-tuning on model performance. Through detailed error analysis, we identified strengths and potential limitations of the existing models and suggested directions for improvement. Subsequently, a clinical reader study involving 60 psychiatrists of varying seniority was conducted to further explore the practical benefits of existing LLMs as supportive tools for psychiatrists of varying seniority. Through the quantitative and reader evaluation, we show that while existing models demonstrate significant potential, they are not yet adequate as decision-making tools in psychiatric clinical practice. The reader study further indicates that, as an auxiliary tool, LLM could provide particularly notable support for junior psychiatrists, effectively enhancing their work efficiency and overall clinical quality. To promote research in this area, we will make the dataset and evaluation framework publicly available, with the hope of advancing the application of LLMs in psychiatric clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01902v1">An Empirical Analysis of LLMs for Countering Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-02-28
      | 💬 Adiba and Neeley contributed equally
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) can amplify online misinformation, they also show promise in tackling misinformation. In this paper, we empirically study the capabilities of three LLMs -- ChatGPT, Gemini, and Claude -- in countering political misinformation. We implement a two-step, chain-of-thought prompting approach, where models first identify credible sources for a given claim and then generate persuasive responses. Our findings suggest that models struggle to ground their responses in real news sources, and tend to prefer citing left-leaning sources. We also observe varying degrees of response diversity among models. Our findings highlight concerns about using LLMs for fact-checking through only prompt-engineering, emphasizing the need for more robust guardrails. Our results have implications for both researchers and non-technical users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01900v1">LLM-Empowered Class Imbalanced Graph Prompt Learning for Online Drug Trafficking Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-28
    </div>
    <details class="paper-abstract">
      As the market for illicit drugs remains extremely profitable, major online platforms have become direct-to-consumer intermediaries for illicit drug trafficking participants. These online activities raise significant social concerns that require immediate actions. Existing approaches to combating this challenge are generally impractical, due to the imbalance of classes and scarcity of labeled samples in real-world applications. To this end, we propose a novel Large Language Model-empowered Heterogeneous Graph Prompt Learning framework for illicit Drug Trafficking detection, called LLM-HetGDT, that leverages LLM to facilitate heterogeneous graph neural networks (HGNNs) to effectively identify drug trafficking activities in the class-imbalanced scenarios. Specifically, we first pre-train HGNN over a contrastive pretext task to capture the inherent node and structure information over the unlabeled drug trafficking heterogeneous graph (HG). Afterward, we employ LLM to augment the HG by generating high-quality synthetic user nodes in minority classes. Then, we fine-tune the soft prompts on the augmented HG to capture the important information in the minority classes for the downstream drug trafficking detection task. To comprehensively study online illicit drug trafficking activities, we collect a new HG dataset over Twitter, called Twitter-HetDrug. Extensive experiments on this dataset demonstrate the effectiveness, efficiency, and applicability of LLM-HetGDT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20421v1">MobiLLM: Enabling LLM Fine-Tuning on the Mobile Device via Server Assisted Side Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) at mobile devices and its potential applications never fail to fascinate. However, on-device LLM fine-tuning poses great challenges due to extremely high memory requirements and slow training speeds. Even with parameter-efficient fine-tuning (PEFT) methods that update only a small subset of parameters, resource-constrained mobile devices cannot afford them. In this paper, we propose MobiLLM to enable memory-efficient transformer LLM fine-tuning on a mobile device via server-assisted side-tuning. Particularly, MobiLLM allows the resource-constrained mobile device to retain merely a frozen backbone model, while offloading the memory and computation-intensive backpropagation of a trainable side-network to a high-performance server. Unlike existing fine-tuning methods that keep trainable parameters inside the frozen backbone, MobiLLM separates a set of parallel adapters from the backbone to create a backpropagation bypass, involving only one-way activation transfers from the mobile device to the server with low-width quantization during forward propagation. In this way, the data never leaves the mobile device while the device can remove backpropagation through the local backbone model and its forward propagation can be paralyzed with the server-side execution. Thus, MobiLLM preserves data privacy while significantly reducing the memory and computational burdens for LLM fine-tuning. Through extensive experiments, we demonstrate that MobiLLM can enable a resource-constrained mobile device, even a CPU-only one, to fine-tune LLMs and significantly reduce convergence time and memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20420v1">Chitranuvad: Adapting Multi-Lingual LLMs for Multimodal Translation</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      In this work, we provide the system description of our submission as part of the English to Lowres Multimodal Translation Task at the Workshop on Asian Translation (WAT2024). We introduce Chitranuvad, a multimodal model that effectively integrates Multilingual LLM and a vision module for Multimodal Translation. Our method uses a ViT image encoder to extract visual representations as visual token embeddings which are projected to the LLM space by an adapter layer and generates translation in an autoregressive fashion. We participated in all the three tracks (Image Captioning, Text only and Multimodal translation tasks) for Indic languages (ie. English translation to Hindi, Bengali and Malyalam) and achieved SOTA results for Hindi in all of them on the Challenge set while remaining competitive for the other languages in the shared task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20383v1">Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Project website: http://vulnerable-ai-agents.github.io
    </div>
    <details class="paper-abstract">
      Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20356v1">Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant limitations in understanding creative content, as demonstrated by Hessel et al. (2023)'s influential work on the New Yorker Cartoon Caption Contest (NYCCC). Their study exposed a substantial gap between LLMs and humans in humor comprehension, establishing that understanding and evaluating creative content is key challenge in AI development. We revisit this challenge by decomposing humor understanding into three components and systematically improve each: enhancing visual understanding through improved annotation, utilizing LLM-generated humor reasoning and explanations, and implementing targeted alignment with human preference data. Our refined approach achieves 82.4% accuracy in caption ranking, singificantly improving upon the previous 67% benchmark and matching the performance of world-renowned human experts in this domain. Notably, while attempts to mimic subgroup preferences through various persona prompts showed minimal impact, model finetuning with crowd preferences proved remarkably effective. These findings reveal that LLM limitations in creative judgment can be effectively addressed through focused alignment to specific subgroups and individuals. Lastly, we propose the position that achieving artificial general intelligence necessitates systematic collection of human preference data across creative domains. We advocate that just as human creativity is deeply influenced by individual and cultural preferences, training LLMs with diverse human preference data may be essential for developing true creative understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20284v1">Evaluating Human Trust in LLM-Based Planners: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for planning tasks, offering unique capabilities not found in classical planners such as generating explanations and iterative refinement. However, trust--a critical factor in the adoption of planning systems--remains underexplored in the context of LLM-based planning tasks. This study bridges this gap by comparing human trust in LLM-based planners with classical planners through a user study in a Planning Domain Definition Language (PDDL) domain. Combining subjective measures, such as trust questionnaires, with objective metrics like evaluation accuracy, our findings reveal that correctness is the primary driver of trust and performance. Explanations provided by the LLM improved evaluation accuracy but had limited impact on trust, while plan refinement showed potential for increasing trust without significantly enhancing evaluation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v2">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02408v2">AI on My Shoulder: Supporting Emotional Labor in Front-Office Roles with an LLM-based Empathetic Coworker</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Client-Service Representatives (CSRs) are vital to organizations. Frequent interactions with disgruntled clients, however, disrupt their mental well-being. To help CSRs regulate their emotions while interacting with uncivil clients, we designed Care-Pilot, an LLM-powered assistant, and evaluated its efficacy, perception, and use. Our comparative analyses between 665 human and Care-Pilot-generated support messages highlight Care-Pilot's ability to adapt to and demonstrate empathy in various incivility incidents. Additionally, 143 CSRs assessed Care-Pilot's empathy as more sincere and actionable than human messages. Finally, we interviewed 20 CSRs who interacted with Care-Pilot in a simulation exercise. They reported that Care-Pilot helped them avoid negative thinking, recenter thoughts, and humanize clients; showing potential for bridging gaps in coworker support. Yet, they also noted deployment challenges and emphasized the indispensability of shared experiences. We discuss future designs and societal implications of AI-mediated emotional labor, underscoring empathy as a critical function for AI assistants for worker mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20258v1">LLM as a Broken Telephone: Iterative Generation Distorts Information</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20238v1">FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Many challenging reasoning tasks require not just rapid, intuitive responses, but a more deliberate, multi-step approach. Recent progress in large language models (LLMs) highlights an important shift from the "System 1" way of quick reactions to the "System 2" style of reflection-and-correction problem solving. However, current benchmarks heavily rely on the final-answer accuracy, leaving much of a model's intermediate reasoning steps unexamined. This fails to assess the model's ability to reflect and rectify mistakes within the reasoning process. To bridge this gap, we introduce FINEREASON, a logic-puzzle benchmark for fine-grained evaluation of LLMs' reasoning capabilities. Each puzzle can be decomposed into atomic steps, making it ideal for rigorous validation of intermediate correctness. Building on this, we introduce two tasks: state checking, and state transition, for a comprehensive evaluation of how models assess the current situation and plan the next move. To support broader research, we also provide a puzzle training set aimed at enhancing performance on general mathematical tasks. We show that models trained on our state checking and transition data demonstrate gains in math reasoning by up to 5.1% on GSM8K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17732v2">CheckMate: LLM-Powered Approximate Intermittent Computing</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Accepted in SenSys 2025
    </div>
    <details class="paper-abstract">
      Batteryless IoT systems face energy constraints exacerbated by checkpointing overhead. Approximate computing offers solutions but demands manual expertise, limiting scalability. This paper presents CheckMate, an automated framework leveraging LLMs for context-aware code approximations. CheckMate integrates validation of LLM-generated approximations to ensure correct execution and employs Bayesian optimization to fine-tune approximation parameters autonomously, eliminating the need for developer input. Tested across six IoT applications, it reduces power cycles by up to 60% with an accuracy loss of just 8%, outperforming semi-automated tools like ACCEPT in speedup and accuracy. CheckMate's results establish it as a robust, user-friendly tool and a foundational step toward automated approximation frameworks for intermittent computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20175v1">An Extensive Evaluation of PDDL Capabilities in off-the-shelf LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      In recent advancements, large language models (LLMs) have exhibited proficiency in code generation and chain-of-thought reasoning, laying the groundwork for tackling automatic formal planning tasks. This study evaluates the potential of LLMs to understand and generate Planning Domain Definition Language (PDDL), an essential representation in artificial intelligence planning. We conduct an extensive analysis across 20 distinct models spanning 7 major LLM families, both commercial and open-source. Our comprehensive evaluation sheds light on the zero-shot LLM capabilities of parsing, generating, and reasoning with PDDL. Our findings indicate that while some models demonstrate notable effectiveness in handling PDDL, others pose limitations in more complex scenarios requiring nuanced planning knowledge. These results highlight the promise and current limitations of LLMs in formal planning tasks, offering insights into their application and guiding future efforts in AI-driven planning paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20140v1">Telephone Surveys Meet Conversational AI: Evaluating a LLM-Based Telephone Survey System at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Telephone surveys remain a valuable tool for gathering insights but typically require substantial resources in training and coordinating human interviewers. This work presents an AI-driven telephone survey system integrating text-to-speech (TTS), a large language model (LLM), and speech-to-text (STT) that mimics the versatility of human-led interviews on scale. We tested the system across two populations, a pilot study in the United States (n = 75) and a large-scale deployment in Peru (n = 2,739), inviting participants via web-based links and contacting them via direct phone calls. The AI agent successfully administered open-ended and closed-ended questions, handled basic clarifications, and dynamically navigated branching logic, allowing fast large-scale survey deployment without interviewer recruitment or training. Our findings demonstrate that while the AI system's probing for qualitative depth was more limited than human interviewers, overall data quality approached human-led standards for structured items. This study represents one of the first successful large-scale deployments of an LLM-based telephone interviewer in a real-world survey context. The AI-powered telephone survey system has the potential for expanding scalable, consistent data collecting across market research, social science, and public opinion studies, thus improving operational efficiency while maintaining appropriate data quality for research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11807v6">How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Accepted to ICLR 2025; 11 pages of main text; 26 pages of appendices; Included models: GPT-3.5-{0613, 1106, 0125}, GPT-4-0125, GPT-4o-0806, Gemini-{1.0, 1.5)-Pro, LLaMA-3.1-{7, 70, 405}B, Mixtral-8x{7, 22}B, Qwen-2-72B
    </div>
    <details class="paper-abstract">
      Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs' decision-making through the lens of Game Theory. However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA($\gamma$)-Bench, a new framework for evaluating LLMs' Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs' performance. $\gamma$-Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chain-of-Thought. We also evaluate 13 LLMs from 6 model families, including GPT-3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of $69.8$ out of $100$, followed by LLaMA-3.1-70B ($65.9$) and Mixtral-8x22B ($62.4$). Our code and experimental results are publicly available at https://github.com/CUHK-ARISE/GAMABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20082v1">LongRoPE2: Near-Lossless LLM Context Window Scaling</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      LongRoPE2 is a novel approach that extends the effective context window of pre-trained large language models (LLMs) to the target length, while preserving the performance on the original shorter context window. This is achieved by three contributions: (1) a hypothesis that insufficient training in higher RoPE dimensions contributes to the persistent out-of-distribution (OOD) issues observed in existing methods; (2) an effective RoPE rescaling algorithm that adopts evolutionary search guided by "needle-driven" perplexity to address the insufficient training problem; (3) a mixed context window training approach that fine-tunes model weights to adopt rescaled RoPE for long-context sequences while preserving the short-context performance with the original RoPE. Extensive experiments on LLaMA3-8B and Phi3-mini-3.8B across various benchmarks validate the hypothesis and demonstrate the effectiveness of LongRoPE2. Remarkably, LongRoPE2 extends LLaMA3-8B to achieve a 128K effective context length while retaining over 98.5% of short-context performance, using only 10B tokens -- 80x fewer than Meta's approach, which fails to reach the target effective context length. Code will be available at https://github.com/microsoft/LongRoPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06153v3">AgentSquare: Automatic LLM Agent Search in Modular Design Space</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to a rapid growth of agentic systems capable of handling a wide range of complex tasks. However, current research largely relies on manual, task-specific design, limiting their adaptability to novel tasks. In this paper, we introduce a new research problem: Modularized LLM Agent Search (MoLAS). We propose a modular design space that abstracts existing LLM agent designs into four fundamental modules with uniform IO interface: Planning, Reasoning, Tool Use, and Memory. Building on this design space, we present a novel LLM agent search framework called AgentSquare, which introduces two core mechanisms, i.e., module evolution and recombination, to efficiently search for optimized LLM agents. To further accelerate the process, we design a performance predictor that uses in-context surrogate models to skip unpromising agent designs. Extensive experiments across six benchmarks, covering the diverse scenarios of web, embodied, tool use and game applications, show that AgentSquare substantially outperforms hand-crafted agents, achieving an average performance gain of 17.2% against best-known human designs. Moreover, AgentSquare can generate interpretable design insights, enabling a deeper understanding of agentic architecture and its impact on task performance. We believe that the modular design space and AgentSquare search framework offer a platform for fully exploiting the potential of prior successful designs and consolidating the collective efforts of research community. Code repo is available at https://github.com/tsinghua-fib-lab/AgentSquare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06899v2">LongSafety: Enhance Safety for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent advancements in model architectures and length extrapolation techniques have significantly extended the context length of large language models (LLMs), paving the way for their application in increasingly complex tasks. However, despite the growing capabilities of long-context LLMs, the safety issues in long-context scenarios remain underexplored. While safety alignment in short context has been widely studied, the safety concerns of long-context LLMs have not been adequately addressed. In this work, we introduce \textbf{LongSafety}, a comprehensive safety alignment dataset for long-context LLMs, containing 10 tasks and 17k samples, with an average length of 40.9k tokens. Our experiments demonstrate that training with LongSafety can enhance long-context safety performance while enhancing short-context safety and preserving general capabilities. Furthermore, we demonstrate that long-context safety does not equal long-context alignment with short-context safety data and LongSafety has generalizing capabilities in context length and long-context safety scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04671v2">CUIfy the XR: An Open-Source Package to Embed LLM-powered Conversational Agents in XR</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality (IEEE AIxVR 2025)
    </div>
    <details class="paper-abstract">
      Recent developments in computer graphics, machine learning, and sensor technologies enable numerous opportunities for extended reality (XR) setups for everyday life, from skills training to entertainment. With large corporations offering affordable consumer-grade head-mounted displays (HMDs), XR will likely become pervasive, and HMDs will develop as personal devices like smartphones and tablets. However, having intelligent spaces and naturalistic interactions in XR is as important as technological advances so that users grow their engagement in virtual and augmented spaces. To this end, large language model (LLM)--powered non-player characters (NPCs) with speech-to-text (STT) and text-to-speech (TTS) models bring significant advantages over conventional or pre-scripted NPCs for facilitating more natural conversational user interfaces (CUIs) in XR. This paper provides the community with an open-source, customizable, extendable, and privacy-aware Unity package, CUIfy, that facilitates speech-based NPC-user interaction with widely used LLMs, STT, and TTS models. Our package also supports multiple LLM-powered NPCs per environment and minimizes latency between different computational models through streaming to achieve usable interactions between users and NPCs. We publish our source code in the following repository: https://gitlab.lrz.de/hctl/cuify
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v2">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce answers with a single chain-of-thought, which restricts their ability to explore reasoning paths or self-correct flawed outputs in complex tasks. In this paper, we introduce MALT (Multi-Agent LLM Training), a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents. During data generation, each agent is repeatedly sampled to form a multi-agent search tree, where final outputs are graded against ground-truth data. We then apply value iteration to propagate reward signals back to each role-conditioned model, automatically producing multi-agent post-training data without human or teacher-model supervision. Our off-policy approach allows each agent to specialize by learning from correct and incorrect trajectories, ultimately improving the end-to-end reasoning chain. On MATH, GSM8K, and CSQA, MALT surpasses the same baseline LLM with a relative improvement of 15.66%, 7.42%, and 9.40% respectively, making it an important advance towards multi-agent cooperative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13461v2">Progressive Mixed-Precision Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      In spite of the great potential of large language models (LLMs) across various tasks, their deployment on resource-constrained devices remains challenging due to their excessive computational and memory demands. Quantization has emerged as an effective solution by storing weights in reduced precision. However, utilizing low precisions (i.e.~2/3-bit) to substantially alleviate the memory-boundedness of LLM decoding, still suffers from prohibitive performance drop. In this work, we argue that existing approaches fail to explore the diversity in computational patterns, redundancy, and sensitivity to approximations of the different phases of LLM inference, resorting to a uniform quantization policy throughout. Instead, we propose a novel phase-aware method that selectively allocates precision during different phases of LLM inference, achieving both strong context extraction during prefill and efficient memory bandwidth utilization during decoding. To further address the memory-boundedness of the decoding phase, we introduce Progressive Mixed-Precision Decoding (PMPD), a technique that enables the gradual lowering of precision deeper in the generated sequence, together with a spectrum of precision-switching schedulers that dynamically drive the precision-lowering decisions in either task-adaptive or prompt-adaptive manner. Extensive evaluation across diverse language tasks shows that when targeting Nvidia GPUs, PMPD achieves 1.4$-$12.2$\times$ speedup in matrix-vector multiplications over fp16 models, while when targeting an LLM-optimized NPU, our approach delivers a throughput gain of 3.8$-$8.0$\times$ over fp16 models and up to 1.54$\times$ over uniform quantization approaches while preserving the output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19981v1">The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Autoregressive large language models (LLMs) exhibit impressive performance across various tasks but struggle with simple arithmetic, such as addition of two or more operands. We show that this struggle arises from LLMs' use of a simple one-digit lookahead heuristic, which works fairly well (but not perfect) for two-operand addition but fails in multi-operand cases, where the carry-over logic is more complex. Our probing experiments and digit-wise accuracy evaluation show that LLMs fail precisely where a one-digit lookahead is insufficient to account for cascading carries. We analyze the impact of tokenization strategies on arithmetic performance and show that all investigated models, regardless of tokenization, are inherently limited in the addition of multiple operands due to their reliance on a one-digit lookahead heuristic. Our findings reveal fundamental limitations that prevent LLMs from generalizing to more complex numerical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19965v1">Deterministic or probabilistic? The psychology of LLMs as random number generators</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 31 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed text generation through inherently probabilistic context-aware mechanisms, mimicking human natural language. In this paper, we systematically investigate the performance of various LLMs when generating random numbers, considering diverse configurations such as different model architectures, numerical ranges, temperature, and prompt languages. Our results reveal that, despite their stochastic transformers-based architecture, these models often exhibit deterministic responses when prompted for random numerical outputs. In particular, we find significant differences when changing the model, as well as the prompt language, attributing this phenomenon to biases deeply embedded within the training data. Models such as DeepSeek-R1 can shed some light on the internal reasoning process of LLMs, despite arriving to similar results. These biases induce predictable patterns that undermine genuine randomness, as LLMs are nothing but reproducing our own human cognitive biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v2">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19915v1">LLM-driven Effective Knowledge Tracing by Integrating Dual-channel Difficulty</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Knowledge Tracing (KT) is a fundamental technology in intelligent tutoring systems used to simulate changes in students' knowledge state during learning, track personalized knowledge mastery, and predict performance. However, current KT models face three major challenges: (1) When encountering new questions, models face cold-start problems due to sparse interaction records, making precise modeling difficult; (2) Traditional models only use historical interaction records for student personalization modeling, unable to accurately track individual mastery levels, resulting in unclear personalized modeling; (3) The decision-making process is opaque to educators, making it challenging for them to understand model judgments. To address these challenges, we propose a novel Dual-channel Difficulty-aware Knowledge Tracing (DDKT) framework that utilizes Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) for subjective difficulty assessment, while integrating difficulty bias-aware algorithms and student mastery algorithms for precise difficulty measurement. Our framework introduces three key innovations: (1) Difficulty Balance Perception Sequence (DBPS) - students' subjective perceptions combined with objective difficulty, measuring gaps between LLM-assessed difficulty, mathematical-statistical difficulty, and students' subjective perceived difficulty through attention mechanisms; (2) Difficulty Mastery Ratio (DMR) - precise modeling of student mastery levels through different difficulty zones; (3) Knowledge State Update Mechanism - implementing personalized knowledge acquisition through gated networks and updating student knowledge state. Experimental results on two real datasets show our method consistently outperforms nine baseline models, improving AUC metrics by 2% to 10% while effectively addressing cold-start problems and enhancing model interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19913v1">SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Data and pipeline parallelism are ubiquitous for training of Large Language Models (LLM) on distributed nodes. Driven by the need for cost-effective training, recent work explores efficient communication arrangement for end to end training. Motivated by LLM's resistance to layer skipping and layer reordering, in this paper, we explore stage (several consecutive layers) skipping in pipeline training, and challenge the conventional practice of sequential pipeline execution. We derive convergence and throughput constraints (guidelines) for pipelining with skipping and swapping pipeline stages. Based on these constraints, we propose SkipPipe, the first partial pipeline framework to reduce the end-to-end training time for LLMs while preserving the convergence. The core of SkipPipe is a path scheduling algorithm that optimizes the paths for individual microbatches and reduces idle time (due to microbatch collisions) on the distributed nodes, complying with the given stage skipping ratio. We extensively evaluate SkipPipe on LLaMa models from 500M to 8B parameters on up to 20 nodes. Our results show that SkipPipe reduces training iteration time by up to $55\%$ compared to full pipeline. Our partial pipeline training also improves resistance to layer omission during inference, experiencing a drop in perplexity of only $7\%$ when running only half the model. Our code is available at https://github.com/gensyn-ai/skippipe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19907v1">Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Logical reasoning is essential for large language models (LLMs) to ensure accurate and coherent inference. However, LLMs struggle with reasoning order variations and fail to generalize across logically equivalent transformations. LLMs often rely on fixed sequential patterns rather than true logical understanding. To address this issue, we introduce an order-centric data augmentation framework based on commutativity in logical reasoning. We first randomly shuffle independent premises to introduce condition order augmentation. For reasoning steps, we construct a directed acyclic graph (DAG) to model dependencies between steps, which allows us to identify valid reorderings of steps while preserving logical correctness. By leveraging order-centric augmentations, models can develop a more flexible and generalized reasoning process. Finally, we conduct extensive experiments across multiple logical reasoning benchmarks, demonstrating that our method significantly enhances LLMs' reasoning performance and adaptability to diverse logical structures. We release our codes and augmented data in https://anonymous.4open.science/r/Order-Centric-Data-Augmentation-822C/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13952v2">The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility?</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - the resulting model outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark. At the end of this paper, we analyze experimental results obtained from testing DeepSeek-R1 on our benchmark and reveal the critical ethical concerns raised by this highly acclaimed model. We highlight that the long Chain-of-Thought (CoT) reasoning process employed by DeepSeek-R1, as well as other LLMs distilled from it, introduces significant ethical vulnerabilities when exposed to users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23769v2">The Potential of LLMs in Medical Education: Generating Questions and Answers for Qualification Exams</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      In this work, we leverage LLMs to produce medical qualification exam questions and the corresponding answers through few-shot prompts, investigating in-depth how LLMs meet the requirements in terms of coherence, evidence of statement, factual consistency, and professionalism etc. Utilizing a multicenter bidirectional anonymized database with respect to comorbid chronic diseases, named Elderly Comorbidity Medical Database (CECMed), we tasked LLMs with generating open-ended questions and answers based on a subset of sampled admission reports. For CECMed, the retrospective cohort includes patients enrolled from January 2010 to January 2022 while the prospective cohort from January 2023 to November 2023, with participants sourced from selected tertiary and community hospitals across the southern, northern, and central regions of China. A total of 8 widely used LLMs were used, including ERNIE 4, ChatGLM 4, Doubao, Hunyuan, Spark 4, Qwen, Conventional medical education requires sophisticated clinicians to formulate questions and answers based on prototypes from EHRs, which is heuristic and time-consuming. We found that mainstream LLMs could generate questions and answers with real-world EHRs at levels close to clinicians. Although current LLMs performed dissatisfactory in some aspects, medical students, interns and residents could reasonably make use of LLMs to facilitate understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19271v2">AutoPureData: Automated Filtering of Undesirable Web Data to Update LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Final version
    </div>
    <details class="paper-abstract">
      Up-to-date and reliable language models are consistently sought after and are essential in various applications. Typically, models are trained on a fixed dataset and then deployed globally. However, the knowledge of the models becomes outdated. Enabling automatic updation of AI knowledge using web data involves significant concerns regarding the model's safety and quality due to a threat from unsafe and undesirable text across the web. The purity of new data was essential for updating knowledge of language models to maintain their reliability. This paper proposes AutoPureData, a system that automatically collects and purifies web data. The system loaded a sample of web data. Utilizing existing trusted AI models, it successfully eliminated unsafe text with an accuracy of 97% and undesirable text with an accuracy of 86%, demonstrating the system's effectiveness in purifying the data. The system ensures that only meaningful and safe text can be used to update LLM knowledge. The pure text was then optimized and stored in a vector database for future querying. It was found that LLM can fetch new data from the vector DB. The LLM writes the RAG query in English, even if the user's query is in another language, proving that the system can perform cross-lingual retrieval. This paper proposes a method to maintain the accuracy and relevance of up-to-date language models by ensuring that only purified data was used to update LLM knowledge. This work contributes to updating knowledge of chatbots using meaningful and safe text, enhancing their utility across various industries, and potentially reducing the risks associated with outputs caused by unsafe or impure data. Code is available at github.com/Pro-GenAI/AutoPureData.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15308v2">LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Content</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 LLMs, Multilingual, Language Diversity, Large Language Models, Social Media, News Media, Specialized LLMs, Fact-checking, Media Analysis, Arabic, Hindi, English
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success as general-purpose task solvers across various fields. However, their capabilities remain limited when addressing domain-specific problems, particularly in downstream NLP tasks. Research has shown that models fine-tuned on instruction-based downstream NLP datasets outperform those that are not fine-tuned. While most efforts in this area have primarily focused on resource-rich languages like English and broad domains, little attention has been given to multilingual settings and specific domains. To address this gap, this study focuses on developing a specialized LLM, LlamaLens, for analyzing news and social media content in a multilingual context. To the best of our knowledge, this is the first attempt to tackle both domain specificity and multilinguality, with a particular focus on news and social media. Our experimental setup includes 18 tasks, represented by 52 datasets covering Arabic, English, and Hindi. We demonstrate that LlamaLens outperforms the current state-of-the-art (SOTA) on 23 testing sets, and achieves comparable performance on 8 sets. We make the models and resources publicly available for the research community (https://huggingface.co/collections/QCRI/llamalens-672f7e0604a0498c6a2f0fe9).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19820v1">Foot-In-The-Door: A Multi-turn Jailbreak for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions.Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions.The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16235v2">Dynamic Parallel Tree Search for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tree of Thoughts (ToT) enhances Large Language Model (LLM) reasoning by structuring problem-solving as a spanning tree. However, recent methods focus on search accuracy while overlooking computational efficiency. The challenges of accelerating the ToT lie in the frequent switching of reasoning focus, and the redundant exploration of suboptimal solutions. To alleviate this dilemma, we propose Dynamic Parallel Tree Search (DPTS), a novel parallelism framework that aims to dynamically optimize the reasoning path in inference. It includes the Parallelism Streamline in the generation phase to build up a flexible and adaptive parallelism with arbitrary paths by fine-grained cache management and alignment. Meanwhile, the Search and Transition Mechanism filters potential candidates to dynamically maintain the reasoning focus on more possible solutions and have less redundancy. Experiments on Qwen-2.5 and Llama-3 with Math500 and GSM8K datasets show that DPTS significantly improves efficiency by 2-4x on average while maintaining or even surpassing existing reasoning algorithms in accuracy, making ToT-based reasoning more scalable and computationally efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17529v2">ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      This paper proposes a novel method for multi-lane convoy formation control that uses large language models (LLMs) to tackle coordination challenges in dynamic highway environments. Each connected and autonomous vehicle in the convoy uses a knowledge-driven approach to make real-time adaptive decisions based on various scenarios. Our method enables vehicles to dynamically perform tasks, including obstacle avoidance, convoy joining/leaving, and escort formation switching, all while maintaining the overall convoy structure. We design a Interlaced formation control strategy based on locally dynamic distributed graphs, ensuring the convoy remains stable and flexible. We conduct extensive experiments in the SUMO simulation platform across multiple traffic scenarios, and the results demonstrate that the proposed method is effective, robust, and adaptable to dynamic environments. The code is available at: https://github.com/chuduanfeng/ConvoyLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v3">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v3">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19735v1">R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Despite recent breakthroughs in reasoning-enhanced large language models (LLMs) like DeepSeek-R1, incorporating inference-time reasoning into machine translation (MT), where human translators naturally employ structured, multi-layered reasoning chain-of-thoughts (CoTs), is yet underexplored. Existing methods either design a fixed CoT tailored for a specific MT sub-task (e.g., literature translation), or rely on synthesizing CoTs unaligned with humans and supervised fine-tuning (SFT) prone to catastrophic forgetting, limiting their adaptability to diverse translation scenarios. This paper introduces R1-Translator (R1-T1), a novel framework to achieve inference-time reasoning for general MT via reinforcement learning (RL) with human-aligned CoTs comprising six common patterns. Our approach pioneers three innovations: (1) extending reasoning-based translation beyond MT sub-tasks to six languages and diverse tasks (e.g., legal/medical domain adaptation, idiom resolution); (2) formalizing six expert-curated CoT templates that mirror hybrid human strategies like context-aware paraphrasing and back translation; and (3) enabling self-evolving CoT discovery and anti-forgetting adaptation through RL with KL-constrained rewards. Experimental results indicate a steady translation performance improvement in 21 languages and 80 translation directions on Flores-101 test set, especially on the 15 languages unseen from training, with its general multilingual abilities preserved compared with plain SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19731v1">Preference Learning Unlocks LLMs' Psycho-Counseling Skills</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) to assist in psycho-counseling is an emerging and meaningful approach, driven by the significant gap between patient needs and the availability of mental health support. However, current LLMs struggle to consistently provide effective responses to client speeches, largely due to the lack of supervision from high-quality real psycho-counseling data, whose content is typically inaccessible due to client privacy concerns. Furthermore, the quality of therapists' responses in available sessions can vary significantly based on their professional training and experience. Assessing the quality of therapists' responses remains an open challenge. In this work, we address these challenges by first proposing a set of professional and comprehensive principles to evaluate therapists' responses to client speeches. Using these principles, we create a preference dataset, PsychoCounsel-Preference, which contains 36k high-quality preference comparison pairs. This dataset aligns with the preferences of professional psychotherapists, providing a robust foundation for evaluating and improving LLMs in psycho-counseling. Experiments on reward modeling and preference learning demonstrate that PsychoCounsel-Preference is an excellent resource for LLMs to acquire essential skills for responding to clients in a counseling session. Our best-aligned model, PsychoCounsel-Llama3-8B, achieves an impressive win rate of 87% against GPT-4o. We release PsychoCounsel-Preference, PsychoCounsel-Llama3-8B and the reward model PsychoCounsel Llama3-8B-Reward to facilitate the research of psycho-counseling with LLMs at: https://hf.co/Psychotherapy-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19721v1">Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate potential harms that may result from these biases, but most work studies biases in LLMs as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We also present a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02926v4">Demystifying RCE Vulnerabilities in LLM-Integrated Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      LLMs show promise in transforming software development, with a growing interest in integrating them into more intelligent apps. Frameworks like LangChain aid LLM-integrated app development, offering code execution utility/APIs for custom actions. However, these capabilities theoretically introduce Remote Code Execution (RCE) vulnerabilities, enabling remote code execution through prompt injections. No prior research systematically investigates these frameworks' RCE vulnerabilities or their impact on applications and exploitation consequences. Therefore, there is a huge research gap in this field. In this study, we propose LLMSmith to detect, validate and exploit the RCE vulnerabilities in LLM-integrated frameworks and apps. To achieve this goal, we develop two novel techniques, including 1) a lightweight static analysis to examine LLM integration mechanisms, and construct call chains to identify RCE vulnerabilities in frameworks; 2) a systematical prompt-based exploitation method to verify and exploit the found vulnerabilities in LLM-integrated apps. This technique involves various strategies to control LLM outputs, trigger RCE vulnerabilities and launch subsequent attacks. Our research has uncovered a total of 20 vulnerabilities in 11 LLM-integrated frameworks, comprising 19 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. Of these, 17 have been confirmed by the framework developers, with 11 vulnerabilities being assigned CVE IDs. For the 51 apps potentially affected by RCE, we successfully executed attacks on 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. Furthermore, we conduct a comprehensive analysis of these vulnerabilities and construct practical attacks to demonstrate the hazards in reality. Last, we propose several mitigation measures for both framework and app developers to counteract such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20681v2">No Free Lunch Theorem for Privacy-Preserving LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Individuals and businesses have been significantly benefited by Large Language Models (LLMs) including PaLM, Gemini and ChatGPT in various ways. For example, LLMs enhance productivity, reduce costs, and enable us to focus on more valuable tasks. Furthermore, LLMs possess the capacity to sift through extensive datasets, uncover underlying patterns, and furnish critical insights that propel the frontiers of technology and science. However, LLMs also pose privacy concerns. Users' interactions with LLMs may expose their sensitive personal or company information. A lack of robust privacy safeguards and legal frameworks could permit the unwarranted intrusion or improper handling of individual data, thereby risking infringements of privacy and the theft of personal identities. To ensure privacy, it is essential to minimize the dependency between shared prompts and private information. Various randomization approaches have been proposed to protect prompts' privacy, but they may incur utility loss compared to unprotected LLMs prompting. Therefore, it is essential to evaluate the balance between the risk of privacy leakage and loss of utility when conducting effective protection mechanisms. The current study develops a framework for inferring privacy-protected Large Language Models (LLMs) and lays down a solid theoretical basis for examining the interplay between privacy preservation and utility. The core insight is encapsulated within a theorem that is called as the NFL (abbreviation of the word No-Free-Lunch) Theorem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08904v3">MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19669v1">Investigating Neurons and Heads in Transformer-based LLMs for Typographical Errors</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 14 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This paper investigates how LLMs encode inputs with typos. We hypothesize that specific neurons and attention heads recognize typos and fix them internally using local and global contexts. We introduce a method to identify typo neurons and typo heads that work actively when inputs contain typos. Our experimental results suggest the following: 1) LLMs can fix typos with local contexts when the typo neurons in either the early or late layers are activated, even if those in the other are not. 2) Typo neurons in the middle layers are responsible for the core of typo-fixing with global contexts. 3) Typo heads fix typos by widely considering the context not focusing on specific tokens. 4) Typo neurons and typo heads work not only for typo-fixing but also for understanding general contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19662v1">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Quantization is critical for realizing efficient inference of LLMs. Traditional quantization methods are hardware-agnostic, limited to bit-width constraints, and lacking circuit-level insights, such as timing and energy characteristics of Multiply-Accumulate (MAC) units. We introduce HALO, a versatile framework that adapts to various hardware through a Hardware-Aware Post-Training Quantization (PTQ) approach. By leveraging MAC unit properties, HALO minimizes critical-path delays and enables dynamic frequency scaling. Deployed on LLM accelerators like TPUs and GPUs, HALO achieves on average 270% performance gains and 51% energy savings, all with minimal accuracy drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20589v1">LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly integrated into many technological ecosystems across various domains and industries, identifying which model is deployed or being interacted with is critical for the security and trustworthiness of the systems. Current verification methods typically rely on analyzing the generated output to determine the source model. However, these techniques are susceptible to adversarial attacks, operate in a post-hoc manner, and may require access to the model weights to inject a verifiable fingerprint. In this paper, we propose a novel passive and non-invasive fingerprinting technique that operates in real-time and remains effective even under encrypted network traffic conditions. Our method leverages the intrinsic autoregressive generation nature of language models, which generate text one token at a time based on all previously generated tokens, creating a unique temporal pattern like a rhythm or heartbeat that persists even when the output is streamed over a network. We find that measuring the Inter-Token Times (ITTs)-time intervals between consecutive tokens-can identify different language models with high accuracy. We develop a Deep Learning (DL) pipeline to capture these timing patterns using network traffic analysis and evaluate it on 16 Small Language Models (SLMs) and 10 proprietary LLMs across different deployment scenarios, including local host machine (GPU/CPU), Local Area Network (LAN), Remote Network, and Virtual Private Network (VPN). The experimental results confirm that our proposed technique is effective and maintains high accuracy even when tested in different network conditions. This work opens a new avenue for model identification in real-world scenarios and contributes to more secure and trustworthy language model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v1">ECCOS: Efficient Capability and Cost Coordinated Scheduling for Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. This paper addresses this challenge by proposing a capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset collected for sample-wise response quality and costs by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14879v2">Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-In-The-Loop LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Passive tracking methods, such as phone and wearable sensing, have become dominant in monitoring human behaviors in modern ubiquitous computing studies. While there have been significant advances in machine-learning approaches to translate periods of raw sensor data to model momentary behaviors, (e.g., physical activity recognition), there still remains a significant gap in the translation of these sensing streams into meaningful, high-level, context-aware insights that are required for various applications (e.g., summarizing an individual's daily routine). To bridge this gap, experts often need to employ a context-driven sensemaking process in real-world studies to derive insights. This process often requires manual effort and can be challenging even for experienced researchers due to the complexity of human behaviors. We conducted three rounds of user studies with 21 experts to explore solutions to address challenges with sensemaking. We follow a human-centered design process to identify needs and design, iterate, build, and evaluate Vital Insight (VI), a novel, LLM-assisted, prototype system to enable human-in-the-loop inference (sensemaking) and visualizations of multi-modal passive sensing data from smartphones and wearables. Using the prototype as a technology probe, we observe experts' interactions with it and develop an expert sensemaking model that explains how experts move between direct data representations and AI-supported inferences to explore, question, and validate insights. Through this iterative process, we also synthesize and discuss a list of design implications for the design of future AI-augmented visualization systems to better assist experts' sensemaking processes in multi-modal health sensing data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12207v2">Divide-Verify-Refine: Can LLMs Self-Align with Complex Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent studies show LLMs struggle with complex instructions involving multiple constraints (e.g., length, format, sentiment). Existing works address this issue by fine-tuning, which heavily relies on fine-tuning data quality and is computational expensive. An alternative is leveraging LLMs' self-correction to refine responses for better constraint adherence. However, this is limited by the feedback quality, as LLMs cannot generate reliable feedback or detect errors. Moreover, its effectiveness relies on few-shot examples illustrating response modifications. As constraints in complex instructions are diverse, manually crafting such examples for each constraint type can be labor-intensive and sub-optimal. To address these two challenges, we propose the Divide-Verify-Refine (DVR) framework with three steps: (1) Divide complex instructions into single constraints and prepare appropriate tools; (2) Verify responses using tools that provide rigorous check and textual guidance (e.g., Python toolkit for format checks or pre-trained classifiers for content analysis); (3) Refine: To maximize refinement effectiveness, we propose dynamic few-shot prompting, where a refinement repository collects successful refinements, and these examples are selectively retrieved for future refinements. Recognizing the lack of complexity in existing datasets, we create a new dataset of complex instructions. DVR doubles Llama3.1-8B's constraint adherence and triples Mistral-7B's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20566v1">Stochastic Rounding for LLM Training: Theory and Practice</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 AISTATS 2025
    </div>
    <details class="paper-abstract">
      As the parameters of Large Language Models (LLMs) have scaled to hundreds of billions, the demand for efficient training methods -- balancing faster computation and reduced memory usage without sacrificing accuracy -- has become more critical than ever. In recent years, various mixed precision strategies, which involve different precision levels for optimization components, have been proposed to increase training speed with minimal accuracy degradation. However, these strategies often require manual adjustments and lack theoretical justification. In this work, we leverage stochastic rounding (SR) to address numerical errors of training with low-precision representation. We provide theoretical analyses of implicit regularization and convergence under the Adam optimizer when SR is utilized. With the insights from these analyses, we extend previous BF16 + SR strategy to be used in distributed settings, enhancing the stability and performance for large scale training. Empirical results from pre-training models with up to 6.7B parameters, for the first time, demonstrate that our BF16 with SR strategy outperforms (BF16, FP32) mixed precision strategies, achieving better validation perplexity, up to $1.54\times$ higher throughput, and $30\%$ less memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v2">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Future LLM agents are likely to communicate on behalf of users with other entity-representing agents on tasks that entail long-horizon plans with interdependent goals. Current work does not focus on such agentic networks, nor does it address their challenges. Thus, we first identify the required properties of agents' communication, which should be proactive and adaptable. It needs to satisfy 1) privacy: agents should not share more than what is needed for the task, and 2) security: the communication must preserve integrity and maintain utility against selfish entities. We design a use case (travel planning) as a testbed that exemplifies these requirements, and we show examples of how this can go wrong. Next, we propose a practical design, inspired by established network security principles, for constrained LLM agentic networks that balance adaptability, security, and privacy. Our framework automatically constructs and updates task-specific rules from prior simulations to build firewalls. We offer layers of defense to 1) convert free-form input to a task-specific protocol, 2) dynamically abstract users' data to a task-specific degree of permissiveness, and 3) self-correct the agents' trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20548v1">$Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q_sharp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20545v1">SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved human-level proficiency across diverse tasks, but their ability to perform rigorous mathematical problem solving remains an open challenge. In this work, we investigate a fundamental yet computationally intractable problem: determining whether a given multivariate polynomial is nonnegative. This problem, closely related to Hilbert's Seventeenth Problem, plays a crucial role in global polynomial optimization and has applications in various fields. First, we introduce SoS-1K, a meticulously curated dataset of approximately 1,000 polynomials, along with expert-designed reasoning instructions based on five progressively challenging criteria. Evaluating multiple state-of-the-art LLMs, we find that without structured guidance, all models perform only slightly above the random guess baseline 50%. However, high-quality reasoning instructions significantly improve accuracy, boosting performance up to 81%. Furthermore, our 7B model, SoS-7B, fine-tuned on SoS-1K for just 4 hours, outperforms the 671B DeepSeek-V3 and GPT-4o-mini in accuracy while only requiring 1.8% and 5% of the computation time needed for letters, respectively. Our findings highlight the potential of LLMs to push the boundaries of mathematical reasoning and tackle NP-hard problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20527v1">Supervised Fine-Tuning LLMs to Behave as Pedagogical Agents in Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being explored in higher education, yet their effectiveness as teaching agents remains underexamined. In this paper, we present the development of GuideLM, a fine-tuned LLM designed for programming education. GuideLM has been integrated into the Debugging C Compiler (DCC), an educational C compiler that leverages LLMs to generate pedagogically sound error explanations. Previously, DCC relied on off-the-shelf OpenAI models, which, while accurate, often over-assisted students by directly providing solutions despite contrary prompting. To address this, we employed supervised fine-tuning (SFT) on a dataset of 528 student-question/teacher-answer pairs, creating two models: GuideLM and GuideLM-mini, fine-tuned on ChatGPT-4o and 4o-mini, respectively. We conducted an expert analysis of 400 responses per model, comparing their pedagogical effectiveness against base OpenAI models. Our evaluation, grounded in constructivism and cognitive load theory, assessed factors such as conceptual scaffolding, clarity, and Socratic guidance. Results indicate that GuideLM and GuideLM-mini improve pedagogical performance, with an 8% increase in Socratic guidance and a 58% improvement in economy of words compared to GPT-4o. However, this refinement comes at the cost of a slight reduction in general accuracy. While further work is needed, our findings suggest that fine-tuning LLMs with targeted datasets is a promising approach for developing models better suited to educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05864v3">From Tokens to Words: On the Inner Lexicon of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Natural language is composed of words, but modern large language models (LLMs) process sub-words as input. A natural question raised by this discrepancy is whether LLMs encode words internally, and if so how. We present evidence that LLMs engage in an intrinsic detokenization process, where sub-word sequences are combined into coherent whole-word representations at their last token. Our experiments show that this process primarily takes place within the early and middle layers of the model. We further demonstrate its robustness to arbitrary splits (e.g., "cats" to "ca" and "ts"), typos, and importantly-to out-of-vocabulary words: when feeding the last token internal representations of such words to the model as input, it can "understand" them as the complete word despite never seeing such representations as input during training. Our findings suggest that LLMs maintain a latent vocabulary beyond the tokenizer's scope. These insights provide a practical, finetuning-free application for expanding the vocabulary of pre-trained models. By enabling the addition of new vocabulary words, we reduce input length and inference iterations, which reduces both space and model latency, with little to no loss in model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20513v1">Personas Evolved: Designing Ethical LLM-Based Conversational Agent Personalities</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has revolutionized Conversational User Interfaces (CUIs), enabling more dynamic, context-aware, and human-like interactions across diverse domains, from social sciences to healthcare. However, the rapid adoption of LLM-based personas raises critical ethical and practical concerns, including bias, manipulation, and unforeseen social consequences. Unlike traditional CUIs, where personas are carefully designed with clear intent, LLM-based personas generate responses dynamically from vast datasets, making their behavior less predictable and harder to govern. This workshop aims to bridge the gap between CUI and broader AI communities by fostering a cross-disciplinary dialogue on the responsible design and evaluation of LLM-based personas. Bringing together researchers, designers, and practitioners, we will explore best practices, develop ethical guidelines, and promote frameworks that ensure transparency, inclusivity, and user-centered interactions. By addressing these challenges collaboratively, we seek to shape the future of LLM-driven CUIs in ways that align with societal values and expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20504v1">A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable advancements in embodying diverse personas, enhancing their effectiveness as conversational agents and virtual assistants. Consequently, LLMs have made significant strides in processing and integrating multimodal information. However, even though human personas can be expressed in both text and image, the extent to which the modality of a persona impacts the embodiment by the LLM remains largely unexplored. In this paper, we investigate how do different modalities influence the expressiveness of personas in multimodal LLMs. To this end, we create a novel modality-parallel dataset of 40 diverse personas varying in age, gender, occupation, and location. This consists of four modalities to equivalently represent a persona: image-only, text-only, a combination of image and small text, and typographical images, where text is visually stylized to convey persona-related attributes. We then create a systematic evaluation framework with 60 questions and corresponding metrics to assess how well LLMs embody each persona across its attributes and scenarios. Comprehensive experiments on $5$ multimodal LLMs show that personas represented by detailed text show more linguistic habits, while typographical images often show more consistency with the persona. Our results reveal that LLMs often overlook persona-specific details conveyed through images, highlighting underlying limitations and paving the way for future research to bridge this gap. We release the data and code at https://github.com/claws-lab/persona-modality .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17341v2">Time series forecasting based on optimized LLM for fault prediction in distribution power grid insulators</a></div>
    <div class="paper-meta">
      📅 2025-02-27
    </div>
    <details class="paper-abstract">
      Surface contamination on electrical grid insulators leads to an increase in leakage current until an electrical discharge occurs, which can result in a power system shutdown. To mitigate the possibility of disruptive faults resulting in a power outage, monitoring contamination and leakage current can help predict the progression of faults. Given this need, this paper proposes a hybrid deep learning (DL) model for predicting the increase in leakage current in high-voltage insulators. The hybrid structure considers a multi-criteria optimization using tree-structured Parzen estimation, an input stage filter for signal noise attenuation combined with a large language model (LLM) applied for time series forecasting. The proposed optimized LLM outperforms state-of-the-art DL models with a root-mean-square error equal to 2.24$\times10^{-4}$ for a short-term horizon and 1.21$\times10^{-3}$ for a medium-term horizon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18210v2">Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-27
      | 💬 15 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.
    </details>
</div>
