# llm - 2025_06

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06877v2">Right Is Not Enough: The Pitfalls of Outcome Supervision in Training LLMs for Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Outcome-rewarded Large Language Models (LLMs) have demonstrated remarkable success in mathematical problem-solving. However, this success often masks a critical issue: models frequently achieve correct answers through fundamentally unsound reasoning processes, a phenomenon indicative of reward hacking. We introduce MathOlympiadEval, a new dataset with fine-grained annotations, which reveals a significant gap between LLMs' answer correctness and their low process correctness. Existing automated methods like LLM-as-a-judge struggle to reliably detect these reasoning flaws. To address this, we propose ParaStepVerifier, a novel methodology for meticulous, step-by-step verification of mathematical solutions. ParaStepVerifier identifies incorrect reasoning steps. Empirical results demonstrate that ParaStepVerifier substantially improves the accuracy of identifying flawed solutions compared to baselines, especially for complex, multi-step problems. This offers a more robust path towards evaluating and training LLMs with genuine mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05153v2">A text-to-tabular approach to generate synthetic patient data using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 12 pages, 3 figures. Accepted to the 2025 IEEE International Conference on Healthcare Informatics (IEEE ICHI 2025), 2025, Rende (CS), Calabria, Italy
    </div>
    <details class="paper-abstract">
      Access to large-scale high-quality healthcare databases is key to accelerate medical research and make insightful discoveries about diseases. However, access to such data is often limited by patient privacy concerns, data sharing restrictions and high costs. To overcome these limitations, synthetic patient data has emerged as an alternative. However, synthetic data generation (SDG) methods typically rely on machine learning (ML) models trained on original data, leading back to the data scarcity problem. We propose an approach to generate synthetic tabular patient data that does not require access to the original data, but only a description of the desired database. We leverage prior medical knowledge and in-context learning capabilities of large language models (LLMs) to generate realistic patient data, even in a low-resource setting. We quantitatively evaluate our approach against state-of-the-art SDG models, using fidelity, privacy, and utility metrics. Our results show that while LLMs may not match the performance of state-of-the-art models trained on the original data, they effectively generate realistic patient data with well-preserved clinical correlations. An ablation study highlights key elements of our prompt contributing to high-quality synthetic patient data generation. This approach, which is easy to use and does not require original data or advanced ML skills, is particularly valuable for quickly generating custom-designed patient data, supporting project implementation and providing educational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19607v1">Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted to FEVER @ ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable capabilities to generate coherent text, they suffer from the issue of hallucinations -- factually inaccurate statements. Among numerous approaches to tackle hallucinations, especially promising are the self-correcting methods. They leverage the multi-turn nature of LLMs to iteratively generate verification questions inquiring additional evidence, answer them with internal or external knowledge, and use that to refine the original response with the new corrections. These methods have been explored for encyclopedic generation, but less so for domains like news summarization. In this work, we investigate two state-of-the-art self-correcting systems by applying them to correct hallucinated summaries using evidence from three search engines. We analyze the results and provide insights into systems' performance, revealing interesting practical findings on the benefits of search engine snippets and few-shot prompts, as well as high alignment of G-Eval and human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17728v2">KAG-Thinker: Interactive Thinking and Deep Reasoning in LLMs via Knowledge-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In this paper, we introduce KAG-Thinker, which upgrade KAG to a multi-turn interactive thinking and deep reasoning framework powered by a dedicated parameter-light large language model (LLM). Our approach constructs a structured thinking process for solving complex problems, enhancing the the logical coherence and contextual consistency of the reasoning process in question-answering (Q&A) tasks on domain-specific knowledge bases (KBs) within LLMs. Following the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG, this framework first decomposes complex questions into independently solvable sub-problems (which are also referred to as logical forms) through \textbf{breadth decomposition}. Each such logical form is represented in two equivalent forms-natural language and logical function-and subsequently classified as either a Knowledge Retrieval or Reasoning Analysis task. Dependencies and parameter passing between these tasks are explicitly modeled via logical function interfaces. In the solving process, the Retrieval function performs retrieval tasks. It retrieves one-hop structured and unstructured information of specified knowledge unit. While the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} module to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} module to enhance the comprehensiveness of knowledge acquisition...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19563v1">PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v2">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 15 pages, 19 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v2">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 I would like to request the withdrawal of this submission because the current version contains significant errors and incomplete results. I intend to revise the manuscript thoroughly before resubmitting. I apologize for the oversight and appreciate your understanding
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19527v1">KnowMap: Efficient Knowledge-Driven Task Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) possess significant capabilities in open-world agent tasks, they also face challenges in rapidly adapting to new, specialized tasks due to their reliance on static pre-trained knowledge. Traditional methods such as fine-tuning are often costly, data-intensive, and may lead to "catastrophic forgetting." Therefore, we present KnowMap, a novel approach that dynamically constructs a knowledge base from environmental and experiential data. KnowMap fine-tunes a small knowledge-embedding model to equip a larger LLM with valuable task-specific knowledge. Our experiments on the ScienceWorld benchmark demonstrate 17.71% improvement for the performance of gpt-4-turbo model. KnowMap not only provides an efficient and effective means for LLM task-adapting, but also highlights how integrating environmental and experiential knowledge can enhance LLMs' reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19525v1">Automatic Posology Structuration : What role for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06102v2">SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs) in Business Intelligence (BI), existing solutions face critical challenges in industrial deployments: functionality deficiencies from legacy systems failing to meet evolving LLM-era user demands, interaction limitations from single-round SQL generation paradigms inadequate for multi-round clarification, and cost for domain adaptation arising from cross-domain methods migration. We present SiriusBI, a practical LLM-powered BI system addressing the challenges of industrial deployments through three key innovations: (a) An end-to-end architecture integrating multi-module coordination to overcome functionality gaps in legacy systems; (b) A multi-round dialogue with querying mechanism, consisting of semantic completion, knowledge-guided clarification, and proactive querying processes, to resolve interaction constraints in SQL generation; (c) A data-conditioned SQL generation method selection strategy that supports both an efficient one-step Fine-Tuning approach and a two-step method leveraging Semantic Intermediate Representation for low-cost cross-domain applications. Experiments on both real-world datasets and public benchmarks demonstrate the effectiveness of SiriusBI. User studies further confirm that SiriusBI enhances both productivity and user experience. As an independent service on Tencent's data platform, SiriusBI is deployed across finance, advertising, and cloud sectors, serving dozens of enterprise clients. It achieves over 93% accuracy in SQL generation and reduces data analysts' query time from minutes to seconds in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15961v2">TrainVerify: Equivalence-Based Verification for Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at scale requires parallel execution across thousands of devices, incurring enormous computational costs. Yet, these costly distributed trainings are rarely verified, leaving them prone to silent errors and potentially wasting millions of GPU hours. We introduce TrainVerify, a system for verifiable distributed training of LLMs. Given a deep learning model's logical specification as the ground truth, TrainVerify formally verifies that a distributed parallel execution plan is mathematically equivalent to it. Direct verification is notoriously difficult due to the sheer scale of LLMs which often involves billions of variables and highly intricate computation graphs. Therefore, TrainVerify introduces shape-reduction techniques and a stage-wise parallel verification algorithm that significantly reduces complexity while preserving formal correctness. TrainVerify scales to frontier LLMs, including the successful verification of the Llama3 (405B) and DeepSeek-V3 (671B) training plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19502v1">MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at https://github.com/AlgazinovAleksandr/Multi-Agent-MATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19481v1">LLM-based Multi-Agent System for Intelligent Refactoring of Haskell Code</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 arXiv admin note: text overlap with arXiv:2502.07928
    </div>
    <details class="paper-abstract">
      Refactoring is a constant activity in software development and maintenance. Scale and maintain software systems are based on code refactoring. However, this process is still labor intensive, as it requires programmers to analyze the codebases in detail to avoid introducing new defects. In this research, we put forward a large language model (LLM)-based multi-agent system to automate the refactoring process on Haskell code. The objective of this research is to evaluate the effect of LLM-based agents in performing structured and semantically accurate refactoring on Haskell code. Our proposed multi-agent system based on specialized agents with distinct roles, including code analysis, refactoring execution, verification, and debugging. To test the effectiveness and practical applicability of the multi-agent system, we conducted evaluations using different open-source Haskell codebases. The results of the experiments carried out showed that the proposed LLM-based multi-agent system could average 11.03% decreased complexity in code, an improvement of 22.46% in overall code quality, and increase performance efficiency by an average of 13.27%. Furthermore, memory allocation was optimized by up to 14.57%. These results highlight the ability of LLM-based multi-agent in managing refactoring tasks targeted toward functional programming paradigms. Our findings hint that LLM-based multi-agent systems integration into the refactoring of functional programming languages can enhance maintainability and support automated development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19399v1">Automated Detection of Pre-training Text in Black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18082v2">Statistical Multicriteria Evaluation of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Assessing the quality of LLM-generated text remains a fundamental challenge in natural language processing. Current evaluation approaches often rely on isolated metrics or simplistic aggregations that fail to capture the nuanced trade-offs between coherence, diversity, fluency, and other relevant indicators of text quality. In this work, we adapt a recently proposed framework for statistical inference based on Generalized Stochastic Dominance (GSD) that addresses three critical limitations in existing benchmarking methodologies: the inadequacy of single-metric evaluation, the incompatibility between cardinal automatic metrics and ordinal human judgments, and the lack of inferential statistical guarantees. The GSD-front approach enables simultaneous evaluation across multiple quality dimensions while respecting their different measurement scales, building upon partial orders of decoding strategies, thus avoiding arbitrary weighting of the involved metrics. By applying this framework to evaluate common decoding strategies against human-generated text, we demonstrate its ability to identify statistically significant performance differences while accounting for potential deviations from the i.i.d. assumption of the sampling design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15066v2">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, the functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. To address these challenges, we introduce ChatModel, the first LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 55.02% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Furthermore, it accelerated the iterative process of reference model design and validation by an average of 5.90x compared to traditional approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18631v2">ReDit: Reward Dithering for Improved LLM Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 10 pages, 15 figures
    </div>
    <details class="paper-abstract">
      DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v4">MCP-Zero: Active Tool Discovery for Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      True intelligence requires active capability acquisition, yet current LLM agents inject pre-defined tool schemas into prompts, reducing models to passive selectors and falling short of robust general-purpose agency. We introduce MCP-Zero, an active agent framework that restores tool discovery autonomy to LLMs themselves. Instead of overwhelming models with all available tools, MCP-Zero enables agents to actively identify capability gaps, and request specific tools on-demand, transforming them from large-scale retrievers into genuine autonomous agents. The framework operates through three core mechanisms: (1) Active Tool Request, where models autonomously generate structured requests specifying their exact tool requirements; (2) Hierarchical Semantic Routing, a two-stage algorithm that matches requests to relevant servers and tools through improved semantic alignment; (3) Iterative Capability Extension, enabling agents to progressively build cross-domain toolchains while maintaining minimal context footprint. We construct MCP-tools, a comprehensive dataset of 308 MCP servers and 2,797 tools from the official Model-Context-Protocol repository. Experiments demonstrate that MCP-Zero preserves agent autonomy while achieving substantial efficiency gains: (i) accurate tool selection from nearly 3k candidates across 248.1k tokens; (ii) 98\% reduction in token consumption on APIBank while maintaining high accuracy; and (iii) consistent multi-turn performance that scales with tool ecosystem growth. This work establishes active tool discovery as a fundamental design pattern for scalable autonomous agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13816v3">Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 ACL 2025 main; camera ready
    </div>
    <details class="paper-abstract">
      While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on the knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19290v1">Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Software engineering (SWE) has recently emerged as a crucial testbed for next-generation LLM agents, demanding inherent capabilities in two critical dimensions: sustained iterative problem-solving (e.g., >50 interaction rounds) and long-context dependency resolution (e.g., >32k tokens). However, the data curation process in SWE remains notoriously time-consuming, as it heavily relies on manual annotation for code file filtering and the setup of dedicated runtime environments to execute and validate unit tests. Consequently, most existing datasets are limited to only a few thousand GitHub-sourced instances. To this end, we propose an incremental, automated data-curation pipeline that systematically scales both the volume and diversity of SWE datasets. Our dataset comprises 10,169 real-world Python task instances from 2,531 distinct GitHub repositories, each accompanied by a task specified in natural language and a dedicated runtime-environment image for automated unit-test validation. We have carefully curated over 8,000 successfully runtime-validated training trajectories from our proposed SWE dataset. When fine-tuning the Skywork-SWE model on these trajectories, we uncover a striking data scaling phenomenon: the trained model's performance for software engineering capabilities in LLMs continues to improve as the data size increases, showing no signs of saturation. Notably, our Skywork-SWE model achieves 38.0% pass@1 accuracy on the SWE-bench Verified benchmark without using verifiers or multiple rollouts, establishing a new state-of-the-art (SOTA) among the Qwen2.5-Coder-32B-based LLMs built on the OpenHands agent framework. Furthermore, with the incorporation of test-time scaling techniques, the performance further improves to 47.0% accuracy, surpassing the previous SOTA results for sub-32B parameter models. We release the Skywork-SWE-32B model checkpoint to accelerate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19287v1">Generating and Understanding Tests via Path-Aware Symbolic Execution with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Symbolic execution is a widely used technique for test generation, offering systematic exploration of program paths through constraint solving. However, it is fundamentally constrained by the capability to model the target code including library functions in terms of symbolic constraint and the capability of underlying constraint solvers. As a result, many paths involving complex features remain unanalyzed or insufficiently modeled. Recent advances in large language models (LLMs) have shown promise in generating diverse and valid test inputs. Yet, LLMs lack mechanisms for systematically enumerating program paths and often fail to cover subtle corner cases. We observe that directly prompting an LLM with the full program leads to missed coverage of interesting paths. In this paper, we present PALM, a test generation system that combines symbolic path enumeration with LLM-assisted test generation. PALM statically enumerates possible paths through AST-level analysis and transforms each into an executable variant with embedded assertions that specify the target path. This avoids the need to translate path constraints into SMT formulae, by instead constructing program variants that LLM can interpret. Importantly, PALM is the first to provide an interactive frontend that visualizes path coverage alongside generated tests, assembling tests based on the specific paths they exercise. A user study with 12 participants demonstrates that PALM's frontend helps users better understand path coverage and identify which paths are actually exercised by PALM-generated tests, through verification and visualization of their path profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19262v1">What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17539v2">Breaking Single-Tester Limits: Multi-Agent LLMs for Multi-User Feature Testing</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted to International Conference on Software Engineering (ICSE 2026). arXiv admin note: substantial text overlap with arXiv:2504.15474
    </div>
    <details class="paper-abstract">
      The growing dependence on mobile phones and their apps has made multi-user interactive features, like chat calls, live streaming, and video conferencing, indispensable for bridging the gaps in social connectivity caused by physical and situational barriers. However, automating these interactive features for testing is fraught with challenges, owing to their inherent need for timely, dynamic, and collaborative user interactions, which current automated testing methods inadequately address. Inspired by the concept of agents designed to autonomously and collaboratively tackle problems, we propose MAdroid, a novel multi-agent approach powered by the Large Language Models (LLMs) to automate the multi-user interactive task for app feature testing. Specifically, MAdroid employs two functional types of multi-agents: user agents (Operator) and supervisor agents (Coordinator and Observer). Each agent takes a specific role: the Coordinator directs the interactive task; the Operator mimics user interactions on the device; and the Observer monitors and reviews the task automation process. Our evaluation, which included 41 multi-user interactive tasks, demonstrates the effectiveness of our approach, achieving 82.9% of the tasks with 96.8% action similarity, outperforming the ablation studies and state-of-the-art baselines. Additionally, a preliminary investigation underscores MAdroid's practicality by helping identify 11 multi-user interactive bugs during regression app testing, confirming its potential value in real-world software development contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19993v1">CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Recommender systems play a pivotal role in providing relevant content to users. With the rapid development of large language models (LLMs), researchers have begun utilizing LLMs to build more powerful recommender systems. However, existing approaches that focus on aligning LLMs with recommendation tasks do not fully leverage their sequential information processing capabilities, leading to suboptimal performance. In this paper, we propose a novel system called compressed vocabulary expansion (CoVE). In CoVE, each item is assigned a unique ID within the expanded vocabulary. Our framework effectively capitalizes on sequence understanding abilities of LLMs, significantly enhancing their performance on recommendation tasks. Additionally, we compress the embedding layer, making CoVE practical for large-scale industrial applications. The effectiveness and performance of CoVE are demonstrated through comprehensive experiments on multiple recommendation datasets and comparisons with prior works. Our code can be found at https://github.com/HaochenZhang717/CoVE-official-Repo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19992v1">HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19952v1">CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00523v3">Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-To-Image Generation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generative models have revolutionized content creation by transforming textual descriptions into high-quality images. However, these models are vulnerable to jailbreaking attacks, where carefully crafted prompts bypass safety mechanisms to produce unsafe content. While researchers have developed various jailbreak attacks to expose this risk, these methods face significant limitations, including impractical access requirements, easily detectable unnatural prompts, restricted search spaces, and high query demands on the target system. In this paper, we propose JailFuzzer, a novel fuzzing framework driven by large language model (LLM) agents, designed to efficiently generate natural and semantically meaningful jailbreak prompts in a black-box setting. Specifically, JailFuzzer employs fuzz-testing principles with three components: a seed pool for initial and jailbreak prompts, a guided mutation engine for generating meaningful variations, and an oracle function to evaluate jailbreak success. Furthermore, we construct the guided mutation engine and oracle function by LLM-based agents, which further ensures efficiency and adaptability in black-box settings. Extensive experiments demonstrate that JailFuzzer has significant advantages in jailbreaking T2I models. It generates natural and semantically coherent prompts, reducing the likelihood of detection by traditional defenses. Additionally, it achieves a high success rate in jailbreak attacks with minimal query overhead, outperforming existing methods across all key metrics. This study underscores the need for stronger safety mechanisms in generative models and provides a foundation for future research on defending against sophisticated jailbreaking attacks. JailFuzzer is open-source and available at this repository: https://github.com/YingkaiD/JailFuzzer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16065v3">Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01484v2">LLM Watermarking Using Mixtures and Statistical-to-Computational Gaps</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Given a text, can we determine whether it was generated by a large language model (LLM) or by a human? A widely studied approach to this problem is watermarking. We propose an undetectable and elementary watermarking scheme in the closed setting. Also, in the harder open setting, where the adversary has access to most of the model, we propose an unremovable watermarking scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19897v1">Can LLMs Replace Humans During Code Chunking?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential tools in computer science, especially for tasks involving code understanding and generation. However, existing work does not address many of the unique challenges presented by code written for government applications. In particular, government enterprise software is often written in legacy languages like MUMPS or assembly language code (ALC) and the overall token lengths of these systems exceed the context window size for current commercially available LLMs. Additionally, LLMs are primarily trained on modern software languages and have undergone limited testing with legacy languages, making their ability to understand legacy languages unknown and, hence, an area for empirical study. This paper examines the application of LLMs in the modernization of legacy government code written in ALC and MUMPS, addressing the challenges of input limitations. We investigate various code-chunking methods to optimize the generation of summary module comments for legacy code files, evaluating the impact of code-chunking methods on the quality of documentation produced by different LLMs, including GPT-4o, Claude 3 Sonnet, Mixtral, and Llama 3. Our results indicate that LLMs can select partition points closely aligned with human expert partitioning. We also find that chunking approaches have significant impact on downstream tasks such as documentation generation. LLM-created partitions produce comments that are up to 20% more factual and up to 10% more useful than when humans create partitions. Therefore, we conclude that LLMs can be used as suitable replacements for human partitioning of large codebases during LLM-aided modernization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19884v1">MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      As the demand for on-device Large Language Model (LLM) inference grows, energy efficiency has become a major concern, especially for battery-limited mobile devices. Our analysis shows that the memory-bound LLM decode phase dominates energy use, and yet most existing works focus on accelerating the prefill phase, neglecting energy concerns. We introduce Adaptive Energy-Centric Core Selection (AECS) and integrate it into MNN to create the energy-efficient version, MNN-AECS, the first engine-level system solution without requiring root access or OS modifications for energy-efficient LLM decoding. MNN-AECS is designed to reduce LLM decoding energy while keeping decode speed within an acceptable slowdown threshold by dynamically selecting low-power CPU cores. MNN-AECS is evaluated across 5 Android and 2 iOS devices on 5 popular LLMs of various sizes. Compared to original MNN, MNN-AECS cuts down energy use by 23% without slowdown averaged over all 7 devices and 4 datasets. Against other engines, including llama.cpp, executorch, mllm, and MediaPipe, MNN-AECS delivers 39% to 78% energy saving and 12% to 363% speedup on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19109v1">Enhancing Security in LLM Applications: A Performance Evaluation of Early Detection Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 18 pages, 8 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Prompt injection threatens novel applications that emerge from adapting LLMs for various user tasks. The newly developed LLM-based software applications become more ubiquitous and diverse. However, the threat of prompt injection attacks undermines the security of these systems as the mitigation and defenses against them, proposed so far, are insufficient. We investigated the capabilities of early prompt injection detection systems, focusing specifically on the detection performance of techniques implemented in various open-source solutions. These solutions are supposed to detect certain types of prompt injection attacks, including the prompt leak. In prompt leakage attacks, an attacker maliciously manipulates the LLM into outputting its system instructions, violating the system's confidentiality. Our study presents analyzes of distinct prompt leakage detection techniques, and a comparative analysis of several detection solutions, which implement those techniques. We identify the strengths and weaknesses of these techniques and elaborate on their optimal configuration and usage in high-stake deployments. In one of the first studies on existing prompt leak detection solutions, we compared the performances of LLM Guard, Vigil, and Rebuff. We concluded that the implementations of canary word checks in Vigil and Rebuff were not effective at detecting prompt leak attacks, and we proposed improvements for them. We also found an evasion weakness in Rebuff's secondary model-based technique and proposed a mitigation. Then, the result of the comparison of LLM Guard, Vigil, and Rebuff at their peak performance revealed that Vigil is optimal for cases when minimal false positive rate is required, and Rebuff is the most optimal for average needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19095v1">Baba is LLM: Reasoning in a Game with Dynamic Rules</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perform well on language tasks, but struggle with reasoning tasks. This paper explores the ability of LLMs to play the 2D puzzle game Baba is You, in which players manipulate rules by rearranging text blocks that define object properties. Given that this rule-manipulation relies on language abilities and reasoning, it is a compelling challenge for LLMs. Six LLMs are evaluated using different prompt types, including (1) simple, (2) rule-extended and (3) action-extended prompts. In addition, two models (Mistral, OLMo) are finetuned using textual and structural data from the game. Results show that while larger models (particularly GPT-4o) perform better in reasoning and puzzle solving, smaller unadapted models struggle to recognize game mechanics or apply rule changes. Finetuning improves the ability to analyze the game levels, but does not significantly improve solution formulation. We conclude that even for state-of-the-art and finetuned LLMs, reasoning about dynamic rule changes is difficult (specifically, understanding the use-mention distinction). The results provide insights into the applicability of LLMs to complex problem-solving tasks and highlight the suitability of games with dynamically changing rules for testing reasoning and reflection by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18469v4">ADVLLM: Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to NAACL 2025 Main (oral)
    </div>
    <details class="paper-abstract">
      Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19082v1">FairCauseSyn: Towards Causally Fair LLM-Augmented Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to IEEE EMBC 2025
    </div>
    <details class="paper-abstract">
      Synthetic data generation creates data based on real-world data using generative models. In health applications, generating high-quality data while maintaining fairness for sensitive attributes is essential for equitable outcomes. Existing GAN-based and LLM-based methods focus on counterfactual fairness and are primarily applied in finance and legal domains. Causal fairness provides a more comprehensive evaluation framework by preserving causal structure, but current synthetic data generation methods do not address it in health settings. To fill this gap, we develop the first LLM-augmented synthetic data generation method to enhance causal fairness using real-world tabular health data. Our generated data deviates by less than 10% from real data on causal fairness metrics. When trained on causally fair predictors, synthetic data reduces bias on the sensitive attribute by 70% compared to real data. This work improves access to fair synthetic data, supporting equitable health research and healthcare delivery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v1">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19030v1">WiLLM: An Open Wireless LLM Communication System</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      The rapid evolution of LLMs threatens to overwhelm existing wireless infrastructure, necessitating architectural innovations for burgeoning mobile LLM services. This paper introduces WiLLM, the first open-source wireless system specifically designed for these services. First, we establish a new paradigm by deploying LLMs in core networks (CNs) with abundant GPUs. This enables distributed inference services, strategically positioning LLM inference at the convergence of backbone bandwidth and the cellular network's edge. Second, we propose an innovative "Tree-Branch-Fruit" extension to the conventional network slicing architecture. This specialized design allows telecom operators to monetize LLM services through slice subscriptions while maintaining infrastructure ownership. Finally, to realize this vision, WiLLM addresses critical limitations in current solutions with several novel capabilities. It features enhanced slice orchestration through a dual-layer slicing architecture, enabling coordinated multi-UE-multi-slice scheduling for finer-grained resource allocation. To ensure universal compatibility, an application-layer tunneling mechanism allows legacy devices without native slicing to access LLM slice services without hardware upgrades. Furthermore, its dual-mode scheduling and cross-layer APIs support flexible deployment from CNs to servers. Built on OpenAirInterface, WiLLM extends this established framework, lowering the adoption barrier for researchers. We also release the first LLM wireless communication dataset with 1,649,996 records and synchronized 58-dimensional metrics, alongside two benchmarks. A case study with smart glasses demonstrates practical viability for resource-constrained devices. WiLLM aims to foster an open platform for cross-layer optimization and AI-telecom convergence. The code, datasets, and hardware details are available at https://openwillm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v1">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18998v1">Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to the Pre-ACL Workshop 2025, Copenhagen
    </div>
    <details class="paper-abstract">
      When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/knowledge-verse-ai/LLM-Memorization_SK_Eval-.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18952v1">LLMs on a Budget? Say HOLA</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18951v1">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 26 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18896v1">ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Codes and Models: https://github.com/Gen-Verse/ReasonFlux
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Projects: https://github.com/Gen-Verse/ReasonFlux
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18880v1">OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13794v4">LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05123v2">A Survey on Data Selection for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted by JAIR
    </div>
    <details class="paper-abstract">
      Instruction tuning is a vital step of training large language models (LLM), so how to enhance the effect of instruction tuning has received increased attention. Existing works indicate that the quality of the dataset is more crucial than the quantity during instruction tuning of LLM. Therefore, recently a lot of studies focus on exploring the methods of selecting high-quality subset from instruction datasets, aiming to reduce training costs and enhance the instruction-following capabilities of LLMs. This paper presents a comprehensive survey on data selection for LLM instruction tuning. Firstly, we introduce the wildly used instruction datasets. Then, we propose a new taxonomy of the data selection methods and provide a detailed introduction of recent advances,and the evaluation strategies and results of data selection methods are also elaborated in detail. Finally, we emphasize the open challenges and present new frontiers of this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18795v1">FORGE: An LLM-driven Framework for Large-Scale Smart Contract Vulnerability Dataset Construction</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted for the 48th International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      High-quality smart contract vulnerability datasets are critical for evaluating security tools and advancing smart contract security research. Two major limitations of current manual dataset construction are (1) labor-intensive and error-prone annotation processes limiting the scale, quality, and evolution of the dataset, and (2) absence of standardized classification rules results in inconsistent vulnerability categories and labeling results across different datasets. To address these limitations, we present FORGE, the first automated approach for constructing smart contract vulnerability datasets. FORGE leverages an LLM-driven pipeline to extract high-quality vulnerabilities from real-world audit reports and classify them according to the CWE, the most widely recognized classification in software security. FORGE employs a divide-and-conquer strategy to extract structured and self-contained vulnerability information from these reports. Additionally, it uses a tree-of-thoughts technique to classify the vulnerability information into the hierarchical CWE classification. To evaluate FORGE's effectiveness, we run FORGE on 6,454 real-world audit reports and generate a dataset comprising 81,390 solidity files and 27,497 vulnerability findings across 296 CWE categories. Manual assessment of the dataset demonstrates high extraction precision and classification consistency with human experts (precision of 95.6% and inter-rater agreement k-$\alpha$ of 0.87). We further validate the practicality of our dataset by benchmarking 13 existing security tools on our dataset. The results reveal the significant limitations in current detection capabilities. Furthermore, by analyzing the severity-frequency distribution patterns through a unified CWE perspective in our dataset, we highlight inconsistency between current smart contract research focus and priorities identified from real-world vulnerabilities...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18783v1">TRIZ Agents: A Multi-Agent LLM Approach for TRIZ-Based Innovation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 12 pages, 10 figures, 2 tables, Accepted at the 17th International Conference on Agents and Artificial Intelligence (ICAART 2025). Final version published in Proceedings of ICAART 2025 (Vol. 1), pages 196-207
    </div>
    <details class="paper-abstract">
      TRIZ, the Theory of Inventive Problem Solving, is a structured, knowledge-based framework for innovation and abstracting problems to find inventive solutions. However, its application is often limited by the complexity and deep interdisciplinary knowledge required. Advancements in Large Language Models (LLMs) have revealed new possibilities for automating parts of this process. While previous studies have explored single LLMs in TRIZ applications, this paper introduces a multi-agent approach. We propose an LLM-based multi-agent system, called TRIZ agents, each with specialized capabilities and tool access, collaboratively solving inventive problems based on the TRIZ methodology. This multi-agent system leverages agents with various domain expertise to efficiently navigate TRIZ steps. The aim is to model and simulate an inventive process with language agents. We assess the effectiveness of this team of agents in addressing complex innovation challenges based on a selected case study in engineering. We demonstrate the potential of agent collaboration to produce diverse, inventive solutions. This research contributes to the future of AI-driven innovation, showcasing the advantages of decentralized problem-solving in complex ideation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18781v1">Existing LLMs Are Not Self-Consistent For Simple Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have grown increasingly powerful, yet ensuring their decisions remain transparent and trustworthy requires self-consistency -- no contradictions in their internal reasoning. Our study reveals that even on simple tasks, such as comparing points on a line or a plane, or reasoning in a family tree, all smaller models are highly inconsistent, and even state-of-the-art models like DeepSeek-R1 and GPT-o4-mini are not fully self-consistent. To quantify and mitigate these inconsistencies, we introduce inconsistency metrics and propose two automated methods -- a graph-based and an energy-based approach. While these fixes provide partial improvements, they also highlight the complexity and importance of self-consistency in building more reliable and interpretable AI. The code and data are available at https://github.com/scorpio-nova/llm-self-consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18777v1">Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24616v2">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 179 pages
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18711v1">LLM-enhanced Interactions in Human-Robot Collaborative Drawing with Older Adults</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      The goal of this study is to identify factors that support and enhance older adults' creative experiences in human-robot co-creativity. Because the research into the use of robots for creativity support with older adults remains underexplored, we carried out an exploratory case study. We took a participatory approach and collaborated with professional art educators to design a course Drawing with Robots for adults aged 65 and over. The course featured human-human and human-robot drawing activities with various types of robots. We observed collaborative drawing interactions, interviewed participants on their experiences, and analyzed collected data. Findings show that participants preferred acting as curators, evaluating creative suggestions from the robot in a teacher or coach role. When we enhanced a robot with a multimodal Large Language Model (LLM), participants appreciated its spoken dialogue capabilities. They reported however, that the robot's feedback sometimes lacked an understanding of the context, and sensitivity to their artistic goals and preferences. Our findings highlight the potential of LLM-enhanced robots to support creativity and offer future directions for advancing human-robot co-creativity with older adults.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18846v2">LLM-Driven APT Detection for 6G Wireless Networks: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 22 pages, 11 figures, 8 tables. Submitted to Computer Science Review (Elsevier), May 2025
    </div>
    <details class="paper-abstract">
      Sixth Generation (6G) wireless networks, which are expected to be deployed in the 2030s, have already created great excitement in academia and the private sector with their extremely high communication speed and low latency rates. However, despite the ultra-low latency, high throughput, and AI-assisted orchestration capabilities they promise, they are vulnerable to stealthy and long-term Advanced Persistent Threats (APTs). Large Language Models (LLMs) stand out as an ideal candidate to fill this gap with their high success in semantic reasoning and threat intelligence. In this paper, we present a comprehensive systematic review and taxonomy study for LLM-assisted APT detection in 6G networks. We address five research questions, namely, semantic merging of fragmented logs, encrypted traffic analysis, edge distribution constraints, dataset/modeling techniques, and reproducibility trends, by leveraging most recent studies on the intersection of LLMs, APTs, and 6G wireless networks. We identify open challenges such as explainability gaps, data scarcity, edge hardware limitations, and the need for real-time slicing-aware adaptation by presenting various taxonomies such as granularity, deployment models, and kill chain stages. We then conclude the paper by providing several research gaps in 6G infrastructures for future researchers. To the best of our knowledge, this paper is the first comprehensive systematic review and classification study on LLM-based APT detection in 6G networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18631v1">ReDit: Reward Dithering for Improved LLM Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 10 pages, 15 figures
    </div>
    <details class="paper-abstract">
      DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18628v1">AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 ICCS 2025 Workshops
    </div>
    <details class="paper-abstract">
      In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18600v1">Reply to "Emergent LLM behaviors are observationally equivalent to data leakage"</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Reply to arXiv:2505.23796
    </div>
    <details class="paper-abstract">
      A potential concern when simulating populations of large language models (LLMs) is data contamination, i.e. the possibility that training data may shape outcomes in unintended ways. While this concern is important and may hinder certain experiments with multi-agent models, it does not preclude the study of genuinely emergent dynamics in LLM populations. The recent critique by Barrie and T\"ornberg [1] of the results of Flint Ashery et al. [2] offers an opportunity to clarify that self-organisation and model-dependent emergent dynamics can be studied in LLM populations, highlighting how such dynamics have been empirically observed in the specific case of social conventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18576v1">A Modular Taxonomy for Hate Speech Definitions and Its Impact on Zero-Shot LLM Classification Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Detecting harmful content is a crucial task in the landscape of NLP applications for Social Good, with hate speech being one of its most dangerous forms. But what do we mean by hate speech, how can we define it, and how does prompting different definitions of hate speech affect model performance? The contribution of this work is twofold. At the theoretical level, we address the ambiguity surrounding hate speech by collecting and analyzing existing definitions from the literature. We organize these definitions into a taxonomy of 14 Conceptual Elements-building blocks that capture different aspects of hate speech definitions, such as references to the target of hate (individual or groups) or of the potential consequences of it. At the experimental level, we employ the collection of definitions in a systematic zero-shot evaluation of three LLMs, on three hate speech datasets representing different types of data (synthetic, human-in-the-loop, and real-world). We find that choosing different definitions, i.e., definitions with a different degree of specificity in terms of encoded elements, impacts model performance, but this effect is not consistent across all architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v2">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.06\pm15.3$ percentage points), up to 68.9pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15035v3">LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Inconsistencies</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we conduct a large-scale, comprehensive safety evaluation of the current LLM landscape. For this purpose, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, with category-wise annotations. Our extensive experiments on 39 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in category crime_tax for Italian but remains safe in other languages. Similar inconsistencies can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure responsible usage across diverse communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18510v1">Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to INTERSPEECH2025 workshop DISS2025
    </div>
    <details class="paper-abstract">
      Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15557v3">MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on LLM judges. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. Regarding the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches, and assist developers in evaluating the dialogue system performance more comprehensively with constrained test resources and budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v3">MCP-Zero: Active Tool Discovery for Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Current LLM agents inject thousands of tool schemas into prompts, creating massive context overhead and reducing models to passive tool selectors rather than autonomous agents. We introduce MCP-Zero, an active agent framework that restores tool discovery autonomy to LLMs themselves. Instead of overwhelming models with all available tools, MCP-Zero enables agents to actively identify capability gaps, and request specific tools on-demand, transforming them from large-scale retrievers into genuine autonomous agents. The framework operates through three core mechanisms: (1) Active Tool Request, where models autonomously generate structured requests specifying their exact tool requirements; (2) Hierarchical Semantic Routing, a two-stage algorithm that matches requests to relevant servers and tools through improved semantic alignment; (3) Iterative Capability Extension, enabling agents to progressively build cross-domain toolchains while maintaining minimal context footprint. We also construct MCP-tools, a comprehensive dataset of 308 MCP servers and 2,797 tools from the official Model-Context-Protocol repository. Experiments demonstrate that MCP-Zero preserves agent autonomy while achieving substantial efficiency gains: (i) accurate tool selection from nearly 3k candidates across 248.1k tokens; (ii) 98\% reduction in token consumption on APIBank while maintaining high accuracy; and (iii) consistent multi-turn performance that scales with tool ecosystem growth. This work establishes active tool discovery as a fundamental design pattern for scalable autonomous agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18387v1">Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 9 pages, presented at LLM4Eval Workshop, SIGIR 2025 Padova, Italy, July 17, 2025
    </div>
    <details class="paper-abstract">
      This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18383v1">LOGICPO: Efficient Translation of NL-based Logical Problems to FOL using LLMs and Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Logical reasoning is a key task for artificial intelligence due to it's role in major downstream tasks such as Question Answering, Summarization. Recent methods in improving the reasoning ability of LLMs fall short in correctly converting a natural language reasoning problem to an equivalent logical formulation, which hinders the framework's overall ability to reason. Towards this, we propose to use finetuning on a preference optimization dataset to learn to parse and represent a natural language problem as a whole to a consistent logical program by 1) introducing a new supervised and preference optimization dataset LogicPO, and 2) adopting popular techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky optimization (KTO) to finetune open-source LLMs. Our best model with Phi-3.5 consistently outperforms GPT-3.5-turbo's (8-shot) by producing 10% more logically correct and with 14% less syntax errors. Through the framework and our improved evaluation metrics, we offer a promising direction in improving the logical reasoning of LLMs by better representing them in their logical formulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12158v2">A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 21 pages, fixed typo
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09655v2">DipLLM: Fine-Tuning LLM for Strategic Decision-making in Diplomacy</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Diplomacy is a complex multiplayer game that requires both cooperation and competition, posing significant challenges for AI systems. Traditional methods rely on equilibrium search to generate extensive game data for training, which demands substantial computational resources. Large Language Models (LLMs) offer a promising alternative, leveraging pre-trained knowledge to achieve strong performance with relatively small-scale fine-tuning. However, applying LLMs to Diplomacy remains challenging due to the exponential growth of possible action combinations and the intricate strategic interactions among players. To address this challenge, we propose DipLLM, a fine-tuned LLM-based agent that learns equilibrium policies for Diplomacy. DipLLM employs an autoregressive factorization framework to simplify the complex task of multi-unit action assignment into a sequence of unit-level decisions. By defining an equilibrium policy within this framework as the learning objective, we fine-tune the model using only 1.5% of the data required by the state-of-the-art Cicero model, surpassing its performance. Our results demonstrate the potential of fine-tuned LLMs for tackling complex strategic decision-making in multiplayer games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15721v2">Bohdi: Heterogeneous LLM Fusion with Automatic Data Exploration</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Heterogeneous Large Language Model (LLM) fusion integrates the strengths of multiple source LLMs with different architectures into a target LLM with low computational overhead. While promising, existing methods suffer from two major limitations: 1) reliance on real data from limited domain for knowledge fusion, preventing the target LLM from fully acquiring knowledge across diverse domains, and 2) fixed data allocation proportions across domains, failing to dynamically adjust according to the target LLM's varying capabilities across domains, leading to a capability imbalance. To overcome these limitations, we propose Bohdi, a synthetic-data-only heterogeneous LLM fusion framework. Through the organization of knowledge domains into a hierarchical tree structure, Bohdi enables automatic domain exploration and multi-domain data generation through multi-model collaboration, thereby comprehensively extracting knowledge from source LLMs. By formalizing domain expansion and data sampling proportion allocation on the knowledge tree as a Hierarchical Multi-Armed Bandit problem, Bohdi leverages the designed DynaBranches mechanism to adaptively adjust sampling proportions based on the target LLM's performance feedback across domains. Integrated with our proposed Introspection-Rebirth (IR) mechanism, DynaBranches dynamically tracks capability shifts during target LLM's updates via Sliding Window Binomial Likelihood Ratio Testing (SWBLRT), further enhancing its online adaptation capability. Comparative experimental results on a comprehensive suite of benchmarks demonstrate that Bohdi significantly outperforms existing baselines on multiple target LLMs, exhibits higher data efficiency, and virtually eliminates the imbalance in the target LLM's capabilities. Our code is available at https://github.com/gjq100/Bohdi.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18341v1">Less Data Less Tokens: Multilingual Unification Learning for Efficient Test-Time Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      This paper explores the challenges of test-time scaling of large language models (LLMs), regarding both the data and inference efficiency. We highlight the diversity of multi-lingual reasoning based on our pilot studies, and then introduce a novel approach, \(L^2\) multi-lingual unification learning with a decoding intervention strategy for further investigation. The basic idea of \(L^2\) is that the reasoning process varies across different languages, which may be mutually beneficial to enhance both model performance and efficiency. In specific, there are two types of multi-lingual data: the entire long chain-of-thought annotations in different languages and the step-wise mixture of languages. By further tuning based on them, we show that even small amounts of data can significantly improve reasoning capabilities. Our findings suggest that multilingual learning reduces both the required data and the number of inference tokens while maintaining a comparable performance. Furthermore, \(L^2\) is orthogonal to other data efficient methods. Thus, we also emphasize the importance of diverse data selection. The \(L^2\) method offers a promising solution to the challenges of data collection and test-time compute efficiency in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21091v3">Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Published in Proceedings of ACM FAccT 2025 Update Comment: Fixed the error where user vs. system and implicit vs. explicit labels in the heatmaps were switched. The takeaways remain the same
    </div>
    <details class="paper-abstract">
      System prompts in Large Language Models (LLMs) are predefined directives that guide model behaviour, taking precedence over user inputs in text processing and generation. LLM deployers increasingly use them to ensure consistent responses across contexts. While model providers set a foundation of system prompts, deployers and third-party developers can append additional prompts without visibility into others' additions, while this layered implementation remains entirely hidden from end-users. As system prompts become more complex, they can directly or indirectly introduce unaccounted for side effects. This lack of transparency raises fundamental questions about how the position of information in different directives shapes model outputs. As such, this work examines how the placement of information affects model behaviour. To this end, we compare how models process demographic information in system versus user prompts across six commercially available LLMs and 50 demographic groups. Our analysis reveals significant biases, manifesting in differences in user representation and decision-making scenarios. Since these variations stem from inaccessible and opaque system-level configurations, they risk representational, allocative and potential other biases and downstream harms beyond the user's ability to detect or correct. Our findings draw attention to these critical issues, which have the potential to perpetuate harms if left unexamined. Further, we argue that system prompt analysis must be incorporated into AI auditing processes, particularly as customisable system prompts become increasingly prevalent in commercial AI deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18315v1">Use Property-Based Testing to Bridge LLM Code Generation and Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at code generation, but ensuring their outputs to be functionally correct, especially in complex programming tasks, is a persistent challenge. While traditional Test-Driven Development (TDD) offers a path for code refinement, its efficacy with LLMs is often undermined by the scarcity of high-quality test cases or the pitfalls of automated test generation, including biased tests or inaccurate output predictions that can misdirect the correction process. This paper introduces Property-Generated Solver, a novel framework that leverages Property-Based Testing (PBT) to validate high-level program properties or invariants, instead of relying on specific input-output examples. These properties are often simpler to define and verify than directly predicting exhaustive test oracles, breaking the "cycle of self-deception" where tests might share flaws with the code they are meant to validate. Property-Generated Solver employs two collaborative LLM-based agents: a Generator dedicated to code generation and iterative refinement, and a Tester that manages the PBT life-cycle and formulate semantically rich feedback from property violations. The resulting comprehensive and actionable feedback then guides the Generator in its refinement efforts. By establishing PBT as the core validation engine within this iterative, closed-loop paradigm, Property-Generated Solver provides a robust mechanism for steering LLMs towards more correct and generalizable code. Extensive experimental results on multiple code generation benchmarks demonstrate that Property-Generated Solver achieves substantial pass@1 improvements, ranging from 23.1% to 37.3% relative gains over established TDD methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11221v3">PlanGenLLMs: A Modern Survey of LLM Planning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18269v1">Co-persona: Leveraging LLMs and Expert Collaboration to Understand User Personas through Social Media Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 17pages,5figures,8tables
    </div>
    <details class="paper-abstract">
      This study introduces \textsc{Co-Persona}, a framework bridging large-scale social media analysis and user understanding via integration of Large Language Models (LLMs) and expert validation. Through a case study of B.Co, a Chinese manufacturer, we applied \textsc{Co-Persona} to bedside lamp development by analyzing 38 million posts from Xiao Hongshu. Our multi-stage NLP processing revealed five user personas based on nighttime behaviors: Health Aficionados, Night Owls, Interior Decorators, Child-care Workers, and Workaholics. These personas exhibit distinct pre-sleep activities and product preferences. The method enhances manufacturers' ability to interpret social data while preserving user-centric insights, offering actionable strategies for targeted marketing and product design. This work advances both theoretical persona development and practical consumer-driven innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18257v1">TableVault: Managing Dynamic Data Collections for LLM-Augmented Workflows</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for automating and executing complex data tasks. However, their integration into more complex data workflows introduces significant management challenges. In response, we present TableVault - a data management system designed to handle dynamic data collections in LLM-augmented environments. TableVault meets the demands of these workflows by supporting concurrent execution, ensuring reproducibility, maintaining robust data versioning, and enabling composable workflow design. By merging established database methodologies with emerging LLM-driven requirements, TableVault offers a transparent platform that efficiently manages both structured data and associated data artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13347v3">Craw4LLM: Efficient Web Crawling for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Web crawl is a main source of large language models' (LLMs) pretraining data, but the majority of crawled web pages are discarded in pretraining due to low data quality. This paper presents Craw4LLM, an efficient web crawling method that explores the web graph based on the preference of LLM pretraining. Specifically, it leverages the influence of a webpage in LLM pretraining as the priority score of the web crawler's scheduler, replacing the standard graph connectivity based priority. Our experiments on a web graph containing 900 million webpages from a commercial search engine's index demonstrate the efficiency of Craw4LLM in obtaining high-quality pretraining data. With just 21% URLs crawled, LLMs pretrained on Craw4LLM data reach the same downstream performances of previous crawls, significantly reducing the crawling waste and alleviating the burdens on websites. Our code is publicly available at https://github.com/cxcscmu/Craw4LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15911v2">From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Published at the 4th Muslims in Machine Learning (MusIML) Workshop (ICML-25)
    </div>
    <details class="paper-abstract">
      Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15690v2">LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      The increasing use of synthetic data from the public Internet has enhanced data usage efficiency in large language model (LLM) training. However, the potential threat of model collapse remains insufficiently explored. Existing studies primarily examine model collapse in a single model setting or rely solely on statistical surrogates. In this work, we introduce LLM Web Dynamics (LWD), an efficient framework for investigating model collapse at the network level. By simulating the Internet with a retrieval-augmented generation (RAG) database, we analyze the convergence pattern of model outputs. Furthermore, we provide theoretical guarantees for this convergence by drawing an analogy to interacting Gaussian Mixture Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19187v1">Prompt, Translate, Fine-Tune, Re-Initialize, or Instruction-Tune? Adapting LLMs for In-Context Learning in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Accepted to ACL GEM 2025
    </div>
    <details class="paper-abstract">
      LLMs are typically trained in high-resource languages, and tasks in lower-resourced languages tend to underperform the higher-resource language counterparts for in-context learning. Despite the large body of work on prompting settings, it is still unclear how LLMs should be adapted cross-lingually specifically for in-context learning in the low-resource target languages. We perform a comprehensive study spanning five diverse target languages, three base LLMs, and seven downstream tasks spanning over 4,100 GPU training hours (9,900+ TFLOPs) across various adaptation techniques: few-shot prompting, translate-test, fine-tuning, embedding re-initialization, and instruction fine-tuning. Our results show that the few-shot prompting and translate-test settings tend to heavily outperform the gradient-based adaptation methods. To better understand this discrepancy, we design a novel metric, Valid Output Recall (VOR), and analyze model outputs to empirically attribute the degradation of these trained models to catastrophic forgetting. To the extent of our knowledge, this is the largest study done on in-context learning for low-resource languages with respect to train compute and number of adaptation techniques considered. We make all our datasets and trained models available for public use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19185v1">Spiritual-LLM : Gita Inspired Mental Health Therapy In the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Traditional mental health support systems often generate responses based solely on the user's current emotion and situations, resulting in superficial interventions that fail to address deeper emotional needs. This study introduces a novel framework by integrating spiritual wisdom from the Bhagavad Gita with advanced large language model GPT-4o to enhance emotional well-being. We present the GITes (Gita Integrated Therapy for Emotional Support) dataset, which enhances the existing ExTES mental health dataset by including 10,729 spiritually guided responses generated by GPT-4o and evaluated by domain experts. We benchmark GITes against 12 state-of-the-art LLMs, including both mental health specific and general purpose models. To evaluate spiritual relevance in generated responses beyond what conventional n-gram based metrics capture, we propose a novel Spiritual Insight metric and automate assessment via an LLM as jury framework using chain-of-thought prompting. Integrating spiritual guidance into AI driven support enhances both NLP and spiritual metrics for the best performing LLM Phi3-Mini 3.2B Instruct, achieving improvements of 122.71% in ROUGE, 126.53% in METEOR, 8.15% in BERT score, 15.92% in Spiritual Insight, 18.61% in Sufficiency and 13.22% in Relevance compared to its zero-shot counterpart. While these results reflect substantial improvements across automated empathy and spirituality metrics, further validation in real world patient populations remains a necessary step. Our findings indicate a strong potential for AI systems enriched with spiritual guidance to enhance user satisfaction and perceived support outcomes. The code and dataset will be publicly available to advance further research in this emerging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00258v2">ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 ICML25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance in natural language processing tasks, yet their massive size makes serving them inefficient and costly. Semi-structured pruning has emerged as an effective method for model acceleration, but existing approaches are suboptimal because they focus on local, layer-wise optimizations using heuristic rules, failing to leverage global feedback. We present ProxSparse, a learning-based framework for mask selection enabled by regularized optimization. ProxSparse transforms the rigid, non-differentiable mask selection process into a smoother optimization procedure, allowing gradual mask exploration with flexibility. ProxSparse does not involve additional weight updates once the mask is determined. Our extensive evaluations on 7 widely used models show that ProxSparse consistently outperforms previously proposed semi-structured mask selection methods with significant improvement, demonstrating the effectiveness of our learned approach towards semi-structured pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19140v1">Command-V: Pasting LLM Behaviors via Activation Profiles</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Retrofitting large language models (LLMs) with new behaviors typically requires full finetuning or distillation-costly steps that must be repeated for every architecture. In this work, we introduce Command-V, a backpropagation-free behavior transfer method that copies an existing residual activation adapter from a donor model and pastes its effect into a recipient model. Command-V profiles layer activations on a small prompt set, derives linear converters between corresponding layers, and applies the donor intervention in the recipient's activation space. This process does not require access to the original training data and needs minimal compute. In three case studies-safety-refusal enhancement, jailbreak facilitation, and automatic chain-of-thought reasoning--Command-V matches or exceeds the performance of direct finetuning while using orders of magnitude less compute. Our code and data are accessible at https://github.com/GithuBarry/Command-V/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19113v1">Human-Aligned Faithfulness in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 21 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate \haf of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and nonsensical responses. We open-source our code and LLM-generated explanations at https://github.com/uofthcdslab/HAF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21621v1">The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18178v1">Integrating LLMs and Digital Twins for Adaptive Multi-Robot Task Allocation in Construction</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Multi-robot systems are emerging as a promising solution to the growing demand for productivity, safety, and adaptability across industrial sectors. However, effectively coordinating multiple robots in dynamic and uncertain environments, such as construction sites, remains a challenge, particularly due to unpredictable factors like material delays, unexpected site conditions, and weather-induced disruptions. To address these challenges, this study proposes an adaptive task allocation framework that strategically leverages the synergistic potential of Digital Twins, Integer Programming (IP), and Large Language Models (LLMs). The multi-robot task allocation problem is formally defined and solved using an IP model that accounts for task dependencies, robot heterogeneity, scheduling constraints, and re-planning requirements. A mechanism for narrative-driven schedule adaptation is introduced, in which unstructured natural language inputs are interpreted by an LLM, and optimization constraints are autonomously updated, enabling human-in-the-loop flexibility without manual coding. A digital twin-based system has been developed to enable real-time synchronization between physical operations and their digital representations. This closed-loop feedback framework ensures that the system remains dynamic and responsive to ongoing changes on site. A case study demonstrates both the computational efficiency of the optimization algorithm and the reasoning performance of several LLMs, with top-performing models achieving over 97% accuracy in constraint and parameter extraction. The results confirm the practicality, adaptability, and cross-domain applicability of the proposed methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10599v2">Generating Energy-efficient code with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      The increasing electricity demands of personal computers, communication networks, and data centers contribute to higher atmospheric greenhouse gas emissions, which in turn lead to global warming and climate change. Therefore the energy consumption of code must be minimized. Code can be generated by large language models. We look at the influence of prompt modification on the energy consumption of the code generated. We use three different Python code problems of varying difficulty levels. Prompt modification is done by adding the sentence ``Give me an energy-optimized solution for this problem'' or by using two Python coding best practices. The large language models used are CodeLlama-70b, CodeLlama-70b-Instruct, CodeLlama-70b-Python, DeepSeek-Coder-33b-base, and DeepSeek-Coder-33b-instruct. We find a decrease in energy consumption for a specific combination of prompt optimization, LLM, and Python code problem. However, no single optimization prompt consistently decreases energy consumption for the same LLM across the different Python code problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03705v2">Enhancing LLM Knowledge Learning through Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      As Large language models (LLMs) are increasingly deployed in diverse applications, faithfully integrating evolving factual knowledge into these models remains a critical challenge. Continued pre-training on paraphrased data has shown empirical promise for enhancing knowledge acquisition. However, this approach is often costly and unreliable, as it relies on external models or manual effort for rewriting, and may inadvertently alter the factual content. In this work, we hypothesize and empirically show that an LLM's ability to continually predict the same factual knowledge tokens given diverse paraphrased contexts is positively correlated with its capacity to extract that knowledge via question-answering. Based on this view and aiming to improve generalization to diverse paraphrased contexts, we introduce two strategies to enhance LLMs' ability to predict the same knowledge tokens given varied contexts, thereby enhancing knowledge acquisition. First, we propose formatting-based data augmentation, which diversifies documents conveying the same knowledge by altering document formats rather than their content, thereby preserving factual integrity. Second, we adopt sharpness-aware minimization as the optimizer to better improve generalization. Extensive experiments demonstrate our methods' effectiveness in both continued pre-training and instruction tuning, and further gains can be achieved by combining with paraphrased data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18116v1">Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 19 Pages, 7 Figures, 4 Tables (Note: Under Review)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18082v1">Statistical Multicriteria Evaluation of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Assessing the quality of LLM-generated text remains a fundamental challenge in natural language processing. Current evaluation approaches often rely on isolated metrics or simplistic aggregations that fail to capture the nuanced trade-offs between coherence, diversity, fluency, and other relevant indicators of text quality. In this work, we adapt a recently proposed framework for statistical inference based on Generalized Stochastic Dominance (GSD) that addresses three critical limitations in existing benchmarking methodologies: the inadequacy of single-metric evaluation, the incompatibility between cardinal automatic metrics and ordinal human judgments, and the lack of inferential statistical guarantees. The GSD-front approach enables simultaneous evaluation across multiple quality dimensions while respecting their different measurement scales, building upon partial orders of decoding strategies, thus avoiding arbitrary weighting of the involved metrics. By applying this framework to evaluate common decoding strategies against human-generated text, we demonstrate its ability to identify statistically significant performance differences while accounting for potential deviations from the i.i.d. assumption of the sampling design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10823v2">FinGPT: Enhancing Sentiment-Based Stock Movement Prediction with Dissemination-Aware and Context-Enriched LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 1st Workshop on Preparing Good Data for Generative AI: Challenges and Approaches@ AAAI 2025, ai4finance.org
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis is crucial for understanding the influence of news on stock prices. Recently, large language models (LLMs) have been widely adopted for this purpose due to their advanced text analysis capabilities. However, these models often only consider the news content itself, ignoring its dissemination, which hampers accurate prediction of short-term stock movements. Additionally, current methods often lack sufficient contextual data and explicit instructions in their prompts, limiting LLMs' ability to interpret news. In this paper, we propose a data-driven approach that enhances LLM-powered sentiment-based stock movement predictions by incorporating news dissemination breadth, contextual data, and explicit instructions. We cluster recent company-related news to assess its reach and influence, enriching prompts with more specific data and precise instructions. This data is used to construct an instruction tuning dataset to fine-tune an LLM for predicting short-term stock price movements. Our experimental results show that our approach improves prediction accuracy by 8\% compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18034v1">Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 Accepted by MICCAI 2025. Code: https://github.com/FengheTan9/LLM4Seg
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14429v2">LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 16 pages, 12 figures, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably stable perplexity during direct context extrapolation. Moreover, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct local perception phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first length extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. The code is available at https://github.com/OpenMOSS/LongLLaDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14562v2">AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. Our code is available at https://github.com/hed-ucas/AlphaDecay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17966v1">LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 arXiv admin note: substantial text overlap with arXiv:2504.15085
    </div>
    <details class="paper-abstract">
      Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12260v2">LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23989v2">Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 Accepted in ICER 2025
    </div>
    <details class="paper-abstract">
      Since the emergence of Large Language Models (LLMs) popularized by the release of GPT-3 and ChatGPT, LLMs have shown remarkable promise in programming-related tasks. While code generation using LLMs has become a popular field of research, code evaluation using LLMs remains under-explored. In this paper, we focus on LLM-based code evaluation and attempt to fill in the existing gaps. We propose multi-agentic novel approaches using \emph{question-specific rubrics} tailored to the problem statement, arguing that these perform better for logical assessment than the existing approaches that use \emph{question-agnostic rubrics}. To address the lack of suitable evaluation datasets, we introduce two datasets: a Data Structures and Algorithms dataset containing 150 student submissions from a popular Data Structures and Algorithms practice website, and an Object Oriented Programming dataset comprising 80 student submissions from undergraduate computer science courses. In addition to using standard metrics (Spearman Correlation, Cohen's Kappa), we additionally propose a new metric called as Leniency, which quantifies evaluation strictness relative to expert assessment. Our comprehensive analysis demonstrates that \emph{question-specific rubrics} significantly enhance logical assessment of code in educational settings, providing better feedback aligned with instructional goals beyond mere syntactic correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17865v1">LASA: Enhancing SoC Security Verification with LLM-Aided Property Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 9 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Ensuring the security of modern System-on-Chip (SoC) designs poses significant challenges due to increasing complexity and distributed assets across the intellectual property (IP) blocks. Formal property verification (FPV) provides the capability to model and validate design behaviors through security properties with model checkers; however, current practices require significant manual efforts to create such properties, making them time-consuming, costly, and error-prone. The emergence of Large Language Models (LLMs) has showcased remarkable proficiency across diverse domains, including HDL code generation and verification tasks. Current LLM-based techniques often produce vacuous assertions and lack efficient prompt generation, comprehensive verification, and bug detection. This paper presents LASA, a novel framework that leverages LLMs and retrieval-augmented generation (RAG) to produce non-vacuous security properties and SystemVerilog Assertions (SVA) from design specifications and related documentation for bus-based SoC designs. LASA integrates commercial EDA tool for FPV to generate coverage metrics and iteratively refines prompts through a feedback loop to enhance coverage. The effectiveness of LASA is validated through various open-source SoC designs, demonstrating high coverage values with an average of ~88\%, denoting comprehensive verification through efficient generation of security properties and SVAs. LASA also demonstrates bug detection capabilities, identifying five unique bugs in the buggy OpenTitan SoC from Hack@DAC'24 competition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17864v1">QueueEDIT: Structural Self-Correction for Sequential Model Editing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated impressive results but still suffer from hallucinations. Model editing has been proposed to correct factual inaccuracies in LLMs. A challenging case is sequential model editing (SME), which aims to rectify errors continuously rather than treating them as a one-time task. During SME, the general capabilities of LLMs can be negatively affected due to the introduction of new parameters. In this paper, we propose a queue-based self-correction framework (QueueEDIT) that not only enhances SME performance by addressing long-sequence dependency but also mitigates the impact of parameter bias on the general capabilities of LLMs. Specifically, we first introduce a structural mapping editing loss to map the triplets to the knowledge-sensitive neurons within the Transformer layers of LLMs. We then store the located parameters for each piece of edited knowledge in a queue and dynamically align previously edited parameters. In each edit, we select queue parameters most relevant to the currently located parameters to determine whether previous knowledge needs realignment. Irrelevant parameters in the queue are frozen, and we update the parameters at the queue head to the LLM to ensure they do not harm general abilities. Experiments show that our framework significantly outperforms strong baselines across various SME settings and maintains competitiveness in single-turn editing. The resulting LLMs also preserve high capabilities in general NLP tasks throughout the SME process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11976v2">How Visual Representations Map to Language Feature Space in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-22
    </div>
    <details class="paper-abstract">
      Effective multimodal reasoning depends on the alignment of visual and linguistic representations, yet the mechanisms by which vision-language models (VLMs) achieve this alignment remain poorly understood. Following the LiMBeR framework, we deliberately maintain a frozen large language model (LLM) and a frozen vision transformer (ViT), connected solely by training a linear adapter during visual instruction tuning. By keeping the language model frozen, we ensure it maintains its original language representations without adaptation to visual data. Consequently, the linear adapter must map visual features directly into the LLM's existing representational space rather than allowing the language model to develop specialized visual understanding through fine-tuning. Our experimental design uniquely enables the use of pre-trained sparse autoencoders (SAEs) of the LLM as analytical probes. These SAEs remain perfectly aligned with the unchanged language model and serve as a snapshot of the learned language feature-representations. Through systematic analysis of SAE reconstruction error, sparsity patterns, and feature SAE descriptions, we reveal the layer-wise progression through which visual representations gradually align with language feature representations, converging in middle-to-later layers. This suggests a fundamental misalignment between ViT outputs and early LLM layers, raising important questions about whether current adapter-based architectures optimally facilitate cross-modal representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17863v1">LLMs for Customized Marketing Content Generation and Evaluation at Scale</a></div>
    <div class="paper-meta">
      📅 2025-06-22
      | 💬 KDD LLM4ECommerce Workshop 2025
    </div>
    <details class="paper-abstract">
      Offsite marketing is essential in e-commerce, enabling businesses to reach customers through external platforms and drive traffic to retail websites. However, most current offsite marketing content is overly generic, template-based, and poorly aligned with landing pages, limiting its effectiveness. To address these limitations, we propose MarketingFM, a retrieval-augmented system that integrates multiple data sources to generate keyword-specific ad copy with minimal human intervention. We validate MarketingFM via offline human and automated evaluations and large-scale online A/B tests. In one experiment, keyword-focused ad copy outperformed templates, achieving up to 9% higher CTR, 12% more impressions, and 0.38% lower CPC, demonstrating gains in ad ranking and cost efficiency. Despite these gains, human review of generated ads remains costly. To address this, we propose AutoEval-Main, an automated evaluation system that combines rule-based metrics with LLM-as-a-Judge techniques to ensure alignment with marketing principles. In experiments with large-scale human annotations, AutoEval-Main achieved 89.57% agreement with human reviewers. Building on this, we propose AutoEval-Update, a cost-efficient LLM-human collaborative framework to dynamically refine evaluation prompts and adapt to shifting criteria with minimal human input. By selectively sampling representative ads for human review and using a critic LLM to generate alignment reports, AutoEval-Update improves evaluation consistency while reducing manual effort. Experiments show the critic LLM suggests meaningful refinements, improving LLM-human agreement. Nonetheless, human oversight remains essential for setting thresholds and validating refinements before deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17828v1">Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10786v3">Evaluating LLMs with Multiple Problems at once</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 22 pages, 9 figures, 12 tables
    </div>
    <details class="paper-abstract">
      This paper shows the benefits and fruitfulness of evaluating LLMs with multiple problems at once, a paradigm we call multi-problem evaluation (MPE). Unlike conventional single-problem evaluation, where a prompt presents a single problem and expects one specific answer, MPE places multiple problems together in a single prompt and assesses how well an LLM answers all these problems in a single output. Leveraging 6 classification and 12 reasoning benchmarks that already exist, we introduce a new benchmark called ZeMPE (Zero-shot Multi-Problem Evaluation), comprising 53,100 zero-shot multi-problem prompts. We experiment with a total of 13 LLMs from 5 model families on ZeMPE to present a comprehensive and systematic MPE. Our results show that LLMs are capable of handling multiple problems from a single data source as well as handling them separately, but there are conditions this multiple problem handling capability falls short. In addition, we perform in-depth further analyses and explore model-level factors that may enable multiple problem handling capabilities in LLMs. We release our corpus and code to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17782v1">Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 To appear at the Third Workshop on Large Language Models for Evaluation in Information Retrieval (LLM4Eval 2025), co-located with SIGIR 2025. 9 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks.
    </details>
</div>
