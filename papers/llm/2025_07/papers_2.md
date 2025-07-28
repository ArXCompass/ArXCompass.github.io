# llm - 2025_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17178v1">SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/Lza12a/SKA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16727v2">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 The inconsistency of base models undermines the fairness of evaluation comparisons and affects the validity of the paper's conclusions
    </div>
    <details class="paper-abstract">
      Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17168v1">Improving LLMs' Generalized Reasoning Abilities by Graph Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 COLM2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made remarkable strides in reasoning tasks, yet their performance often falters on novel and complex problems. Domain-specific continued pretraining (CPT) methods, such as those tailored for mathematical reasoning, have shown promise but lack transferability to broader reasoning tasks. In this work, we pioneer the use of Graph Problem Reasoning (GPR) to enhance the general reasoning capabilities of LLMs. GPR tasks, spanning pathfinding, network analysis, numerical computation, and topological reasoning, require sophisticated logical and relational reasoning, making them ideal for teaching diverse reasoning patterns. To achieve this, we introduce GraphPile, the first large-scale corpus specifically designed for CPT using GPR data. Spanning 10.9 billion tokens across 23 graph tasks, the dataset includes chain-of-thought, program-of-thought, trace of execution, and real-world graph data. Using GraphPile, we train GraphMind on popular base models Llama 3 and 3.1, as well as Gemma 2, achieving up to 4.9 percent higher accuracy in mathematical reasoning and up to 21.2 percent improvement in non-mathematical reasoning tasks such as logical and commonsense reasoning. By being the first to harness GPR for enhancing reasoning patterns and introducing the first dataset of its kind, our work bridges the gap between domain-specific pretraining and universal reasoning capabilities, advancing the adaptability and robustness of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17165v1">Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted at the 41st IEEE International Conference on Software Maintenance and Evolution 2025 (ICSME'25)
    </div>
    <details class="paper-abstract">
      Continuous Integration (CI) services, such as GitHub Actions, require developers to write YAML-based configurations, which can be tedious and error-prone. Despite the increasing use of Large Language Models (LLMs) to automate software engineering tasks, their ability to generate CI configurations remains underexplored. This paper presents a preliminary study evaluating six LLMs for generating GitHub Actions configurations from natural language descriptions. We assess three general-purpose foundation models (GPT-4o, Llama, and Gemma) and three code-pretrained models (GPT-4.1, Code Llama, and CodeGemma). We also introduce the first labeled dataset of its kind, constructed from GitHub Actions documentation, pairing descriptions with corresponding best-practice YAML configurations. Zero-shot prompting achieves up to 69% similarity with the ground truth, with only 3% perfect matches. Code-pretrained models slightly underperform compared to general-purpose ones in YAML-based CI tasks, revealing LLM limitations for CI configuration generation. Analyzing GPT-4o outputs reveals issues like missing or renamed steps, misinterpreted descriptions, and unnecessary additions that may affect structural and contextual correctness, indicating a gap between generation quality and the precision required for executable CI configurations. Our research offers insights for improving LLM alignment with configuration languages and guiding future efforts on CI automation and tooling support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17147v1">CogDual: Enhancing Dual Cognition of LLMs via Reinforcement Learning with Implicit Rule-Based Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Role-Playing Language Agents (RPLAs) have emerged as a significant application direction for Large Language Models (LLMs). Existing approaches typically rely on prompt engineering or supervised fine-tuning to enable models to imitate character behaviors in specific scenarios, but often neglect the underlying \emph{cognitive} mechanisms driving these behaviors. Inspired by cognitive psychology, we introduce \textbf{CogDual}, a novel RPLA adopting a \textit{cognize-then-respond } reasoning paradigm. By jointly modeling external situational awareness and internal self-awareness, CogDual generates responses with improved character consistency and contextual alignment. To further optimize the performance, we employ reinforcement learning with two general-purpose reward schemes designed for open-domain text generation. Extensive experiments on the CoSER benchmark, as well as Cross-MR and LifeChoice, demonstrate that CogDual consistently outperforms existing baselines and generalizes effectively across diverse role-playing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17134v1">Resilient Multi-Agent Negotiation for Medical Supply Chains:Integrating LLMs and Blockchain for Transparent Coordination</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 11 pages, 6 figure
    </div>
    <details class="paper-abstract">
      Global health emergencies, such as the COVID-19 pandemic, have exposed critical weaknesses in traditional medical supply chains, including inefficiencies in resource allocation, lack of transparency, and poor adaptability to dynamic disruptions. This paper presents a novel hybrid framework that integrates blockchain technology with a decentralized, large language model (LLM) powered multi-agent negotiation system to enhance the resilience and accountability of medical supply chains during crises. In this system, autonomous agents-representing manufacturers, distributors, and healthcare institutions-engage in structured, context-aware negotiation and decision-making processes facilitated by LLMs, enabling rapid and ethical allocation of scarce medical resources. The off-chain agent layer supports adaptive reasoning and local decision-making, while the on-chain blockchain layer ensures immutable, transparent, and auditable enforcement of decisions via smart contracts. The framework also incorporates a formal cross-layer communication protocol to bridge decentralized negotiation with institutional enforcement. A simulation environment emulating pandemic scenarios evaluates the system's performance, demonstrating improvements in negotiation efficiency, fairness of allocation, supply chain responsiveness, and auditability. This research contributes an innovative approach that synergizes blockchain trust guarantees with the adaptive intelligence of LLM-driven agents, providing a robust and scalable solution for critical supply chain coordination under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17133v1">BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      In recent years, the Mixture-of-Experts (MoE) architecture has been widely applied to large language models (LLMs), providing a promising solution that activates only a subset of the model's parameters during computation, thereby reducing overall memory requirements and allowing for faster inference compared to dense models. Despite these advantages, existing systems still face issues of low efficiency due to static model placement and lack of dynamic workloads adaptation. This leads to suboptimal resource utilization and increased latency, especially during bursty requests periods. To address these challenges, this paper introduces BrownoutServe, a novel serving framework designed to optimize inference efficiency and maintain service reliability for MoE-based LLMs under dynamic computational demands and traffic conditions. BrownoutServe introduces "united experts" that integrate knowledge from multiple experts, reducing the times of expert access and inference latency. Additionally, it proposes a dynamic brownout mechanism to adaptively adjust the processing of certain tokens, optimizing inference performance while guaranteeing service level objectives (SLOs) are met. Our evaluations show the effectiveness of BrownoutServe under various workloads: it achieves up to 2.07x throughput improvement compared to vLLM and reduces SLO violations by 90.28%, showcasing its robustness under bursty traffic while maintaining acceptable inference accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v2">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model. Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to thousands of GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17120v1">BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly popular in various areas, traditional business gradually shifting from rule-based systems to LLM-based solutions. However, the inference of LLMs is resource-intensive or latency-sensitive, posing significant challenges for serving systems. Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads. These methods may also struggle to adapt to dynamic workload fluctuations, resulting in suboptimal throughput and potential service level objective (SLO) violations. In this paper, we introduce BucketServe, a bucket-based dynamic batching framework designed to optimize LLM inference performance. By grouping requests into size-homogeneous buckets based on sequence length, BucketServe minimizes padding overhead and optimizes GPU memory usage through real-time batch size adjustments preventing out-of-memory (OOM) errors. It introduces adaptive bucket splitting/merging and priority-aware scheduling to mitigate resource fragmentation and ensure SLO compliance. Experiment shows that BucketServe significantly outperforms UELLM in throughput, achieving up to 3.58x improvement. It can also handle 1.93x more request load under the SLO attainment of 80% compared with DistServe and demonstrates 1.975x higher system load capacity compared to the UELLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08513v2">Advancing Multimodal LLMs by Large-Scale 3D Visual Instruction Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) struggle with accurately capturing camera-object relations, especially for object orientation, camera viewpoint, and camera shots. This stems from the fact that existing MLLMs are trained on images with limited diverse camera-object relations and corresponding textual descriptions. To address this, we propose a synthetic generation pipeline to create large-scale 3D visual instruction datasets. Our framework takes 3D assets as input and uses rendering and diffusion-based image generation models to create photorealistic images preserving precise camera-object relations. Additionally, large language models (LLMs) are used to generate text prompts for guiding visual instruction tuning and controlling image generation. We create Ultimate3D, a dataset of 240K VQAs with precise camera-object annotations, and corresponding benchmark. MLLMs fine-tuned on our proposed dataset outperform commercial models by a large margin, achieving an average accuracy improvement of 33.4% on camera-object relation recognition tasks. Our code, dataset, and benchmark will contribute to broad MLLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17962v1">TimelyHLS: LLM-Based Timing-Aware and Architecture-Specific FPGA HLS Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Achieving timing closure and design-specific optimizations in FPGA-targeted High-Level Synthesis (HLS) remains a significant challenge due to the complex interaction between architectural constraints, resource utilization, and the absence of automated support for platform-specific pragmas. In this work, we propose TimelyHLS, a novel framework integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to automatically generate and iteratively refine HLS code optimized for FPGA-specific timing and performance requirements. TimelyHLS is driven by a structured architectural knowledge base containing FPGA-specific features, synthesis directives, and pragma templates. Given a kernel, TimelyHLS generates HLS code annotated with both timing-critical and design-specific pragmas. The synthesized RTL is then evaluated using commercial toolchains, and simulation correctness is verified against reference outputs via custom testbenches. TimelyHLS iteratively incorporates synthesis logs and performance reports into the LLM engine for refinement in the presence of functional discrepancies. Experimental results across 10 FPGA architectures and diverse benchmarks show that TimelyHLS reduces the need for manual tuning by up to 70%, while achieving up to 4x latency speedup (e.g., 3.85x for Matrix Multiplication, 3.7x for Bitonic Sort) and over 50% area savings in certain cases (e.g., 57% FF reduction in Viterbi). TimelyHLS consistently achieves timing closure and functional correctness across platforms, highlighting the effectiveness of LLM-driven, architecture-aware synthesis in automating FPGA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13238v2">Multilingual LLMs Are Not Multilingual Thinkers: Evidence from Hindi Analogy Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17951v1">Are LLM Belief Updates Consistent with Bayes' Theorem?</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted at the ICML 2025 Workshop on Assessing World Models
    </div>
    <details class="paper-abstract">
      Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03699v3">LLM Alignment as Retriever Optimization: An Information Retrieval Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17927v1">SMARTAPS: Tool-augmented LLMs for Operations Management</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 https://aaai.org/conference/aaai/aaai-25/bridge-ai-orms/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) present intriguing opportunities to enhance user interaction with traditional algorithms and tools in real-world applications. An advanced planning system (APS) is a sophisticated software that leverages optimization to help operations planners create, interpret, and modify an operational plan. While highly beneficial, many customers are priced out of using an APS due to the ongoing costs of consultants responsible for customization and maintenance. To address the need for a more accessible APS expressed by supply chain planners, we present SmartAPS, a conversational system built on a tool-augmented LLM. Our system provides operations planners with an intuitive natural language chat interface, allowing them to query information, perform counterfactual reasoning, receive recommendations, and execute scenario analysis to better manage their operation. A short video demonstrating the system has been released: https://youtu.be/KtIrJjlDbyw
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08537v2">Chemical reasoning in LLMs unlocks strategy-aware synthesis planning and reaction mechanism elucidation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      While automated chemical tools excel at specific tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning. Here we demonstrate that large language models (LLMs) can serve as powerful tools enabling chemical analysis. When integrated with traditional search algorithms, they enable a new approach to computer-aided synthesis that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our system allows chemists to specify desired synthetic strategies in natural language -- from protecting group strategies to global feasibility assessment -- and uses traditional or LLM-guided Monte Carlo Tree Search to find routes that satisfy these constraints. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. This approach shows strong performance across diverse chemical tasks, with newer and larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical automation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17999v2">Streaming, Fast and Slow: Cognitive Load-Aware Streaming for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Generative conversational interfaces powered by large language models (LLMs) typically stream output token-by-token at a rate determined by computational budget, often neglecting actual human reading speeds and the cognitive load associated with the content. This mismatch frequently leads to inefficient use of computational resources. For example, in cloud-based services, streaming content faster than users can read appears unnecessary, resulting in wasted computational resources and potential delays for other users, particularly during peak usage periods. To address this issue, we propose an adaptive streaming method that dynamically adjusts the pacing of LLM streaming output in real-time based on inferred cognitive load. Our approach estimates the cognitive load associated with streaming content and strategically slows down the stream during complex or information-rich segments, thereby freeing computational resources for other users. We conducted a statistical analysis and simulation based on a statistical model derived from data collected in a crowdsourced user study across various types of LLM-generated content. Our results show that this adaptive method can effectively reduce computational consumption while largely maintaining streaming speed above user's normal reading speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17865v1">Talk with the Things: Integrating LLMs into IoT Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 arXiv admin note: text overlap with arXiv:2407.20970
    </div>
    <details class="paper-abstract">
      The convergence of Large Language Models (LLMs) and Internet of Things (IoT) networks open new opportunities for building intelligent, responsive, and user-friendly systems. This work presents an edge-centric framework that integrates LLMs into IoT architectures to enable natural language-based control, context-aware decision-making, and enhanced automation. The proposed modular and lightweight Retrieval Augmented Generation (RAG)-based LLMs are deployed on edge computing devices connected to IoT gateways, enabling local processing of user commands and sensor data for reduced latency, improved privacy, and enhanced inference quality. We validate the framework through a smart home prototype using LLaMA 3 and Gemma 2B models for controlling smart devices. Experimental results highlight the trade-offs between model accuracy and inference time with respect to models size. At last, we also discuss the potential applications that can use LLM-based IoT systems, and a few key challenges associated with such systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17842v1">Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17795v1">LSDM: LLM-Enhanced Spatio-temporal Diffusion Model for Service-Level Mobile Traffic Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Service-level mobile traffic prediction for individual users is essential for network efficiency and quality of service enhancement. However, current prediction methods are limited in their adaptability across different urban environments and produce inaccurate results due to the high uncertainty in personal traffic patterns, the lack of detailed environmental context, and the complex dependencies among different network services. These challenges demand advanced modeling techniques that can capture dynamic traffic distributions and rich environmental features. Inspired by the recent success of diffusion models in distribution modeling and Large Language Models (LLMs) in contextual understanding, we propose an LLM-Enhanced Spatio-temporal Diffusion Model (LSDM). LSDM integrates the generative power of diffusion models with the adaptive learning capabilities of transformers, augmented by the ability to capture multimodal environmental information for modeling service-level patterns and dynamics. Extensive evaluations on real-world service-level datasets demonstrate that the model excels in traffic usage predictions, showing outstanding generalization and adaptability. After incorporating contextual information via LLM, the performance improves by at least 2.83% in terms of the coefficient of determination. Compared to models of a similar type, such as CSDI, the root mean squared error can be reduced by at least 8.29%. The code and dataset will be available at: https://github.com/SoftYuaneR/LSDM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17788v1">Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      When using LLMs to rank items based on given criteria, or evaluate answers, the order of candidate items can influence the model's final decision. This sensitivity to item positioning in a LLM's prompt is known as position bias. Prior research shows that this bias exists even in large models, though its severity varies across models and tasks. In addition to position bias, LLMs also exhibit varying degrees of low repetition consistency, where repeating the LLM call with the same candidate ordering can lead to different rankings. To address both inconsistencies, a common approach is to prompt the model multiple times with different candidate orderings and aggregate the results via majority voting. However, this repetition strategy, significantly increases computational costs. Extending prior findings, we observe that both the direction -- favoring either the earlier or later candidate in the prompt -- and magnitude of position bias across instances vary substantially, even within a single dataset. This observation highlights the need for a per-instance mitigation strategy. To this end, we introduce a dynamic early-stopping method that adaptively determines the number of repetitions required for each instance. Evaluating our approach across three LLMs of varying sizes and on two tasks, namely re-ranking and alignment, we demonstrate that transitioning to a dynamic repetition strategy reduces the number of LLM calls by an average of 81%, while preserving the accuracy. Furthermore, we propose a confidence-based adaptation to our early-stopping method, reducing LLM calls by an average of 87% compared to static repetition, with only a slight accuracy trade-off relative to our original early-stopping method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16809v1">LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 41 pages, 17 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16808v1">Rethinking LLM-Based RTL Code Optimization Via Timing Logic Metamorphosis</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 13pages with 9 pictures and 2 tables
    </div>
    <details class="paper-abstract">
      Register Transfer Level(RTL) code optimization is crucial for achieving high performance and low power consumption in digital circuit design. However, traditional optimization methods often rely on manual tuning and heuristics, which can be time-consuming and error-prone. Recent studies proposed to leverage Large Language Models(LLMs) to assist in RTL code optimization. LLMs can generate optimized code snippets based on natural language descriptions, potentially speeding up the optimization process. However, existing approaches have not thoroughly evaluated the effectiveness of LLM-Based code optimization methods for RTL code with complex timing logic. To address this gap, we conducted a comprehensive empirical investigation to assess the capability of LLM-Based RTL code optimization methods in handling RTL code with complex timing logic. In this study, we first propose a new benchmark for RTL optimization evaluation. It comprises four subsets, each corresponding to a specific area of RTL code optimization. Then we introduce a method based on metamorphosis to systematically evaluate the effectiveness of LLM-Based RTL code optimization methods.Our key insight is that the optimization effectiveness should remain consistent for semantically equivalent but more complex code. After intensive experiments, we revealed several key findings. (1) LLM-Based RTL optimization methods can effectively optimize logic operations and outperform existing compiler-based methods. (2) LLM-Based RTL optimization methods do not perform better than existing compiler-based methods on RTL code with complex timing logic, particularly in timing control flow optimization and clock domain optimization. This is primarily attributed to the challenges LLMs face in understanding timing logic in RTL code. Based on these findings, we provide insights for further research in leveraging LLMs for RTL code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16799v1">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16773v1">When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16754v1">Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However, designing effective pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches, and (2) address new questions without any close prior matches by automatically lowering the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage for unseen cases. We find that implementing a RAG pipeline combining hypothetical-documentation-embedding (HyDE) with the full-answer context performs best in retrieving and answering similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG pipeline to 4 open-source LLMs and compare the results to their zero-shot performance. Our findings show that RAG with our optimal RAG pipeline consistently outperforms zero-shot baselines across models, achieving higher scores for helpfulness, correctness, and detail with LLM-as-a-judge. These findings demonstrate that our optimal RAG pipelines robustly enhance answer quality for a wide range of developer queries including both previously seen and novel questions across different LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16731v1">Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 35 pages, 9 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, deploying them solely in the cloud or compressing them for edge devices has become inadequate due to concerns about latency, privacy, cost, and personalization. This survey explores a collaborative paradigm in which cloud-based LLMs and edge-deployed small language models (SLMs) cooperate across both inference and training. We present a unified taxonomy of edge-cloud collaboration strategies. For inference, we categorize approaches into task assignment, task division, and mixture-based collaboration at both task and token granularity, encompassing adaptive scheduling, resource-aware offloading, speculative decoding, and modular routing. For training, we review distributed adaptation techniques, including parameter alignment, pruning, bidirectional distillation, and small-model-guided optimization. We further summarize datasets, benchmarks, and deployment cases, and highlight privacy-preserving methods and vertical applications. This survey provides the first systematic foundation for LLM-SLM collaboration, bridging system and algorithm co-design to enable efficient, scalable, and trustworthy edge-cloud intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16727v1">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16716v1">Enhancing Remote Sensing Vision-Language Models Through MLLM and LLM-Based High-Quality Image-Text Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 SUBMIT TO IEEE TRANSACTIONS
    </div>
    <details class="paper-abstract">
      The application of Vision-language foundation models (VLFMs) to remote sensing (RS) imagery has garnered significant attention due to their superior capability in various downstream tasks. A key challenge lies in the scarcity of high-quality, large-scale, image-text paired training data. Recently, several works introduced extensive image-text datasets for RS and trained their VLFMs. However, due to the rudimentary methods used for generating captions, the quality of datasets is suboptimal, requiring larger volumes of training data, while only yielding modest performance improvements. In this paper, we propose a two-stage method named MpGI(Multi-Perspective Generation and Integration) for generating high-quality text captions for RS images. Firstly, we generate distinct and detailed descriptions from different perspectives using Rule-MLLM(Multimodal Large Language Model) Relay Generation and MLLMs generation methods. Next, we utilize Large Language Models (LLMs) to integrate these diverse descriptions into comprehensive captions, capturing details from multiple perspectives. Finally, we have created the HQRS-IT-210K dataset, including about 210,000 RS images and 1.3 million captions. We fine-tuned two VLFMs using our dataset: CLIP, a discriminative model, and CoCa, an image-to-text generative model. This process resulted in our proposed HQRS-CLIP and RS-CoCa models. Experimental results demonstrate that HQRS-CLIP surpassed the previous SOTA RS CLIP model in various downstream tasks while using only 4.2\% of the training data. RS-CoCa outperforms other advanced approaches across benchmark datasets and can generate captions for RS images that rival or even exceed manual annotations. Dataset, pre-trained models, and codes will be released at https://github.com/YiguoHe/HQRS-210K-and-HQRS-CLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16708v1">Biases in LLM-Generated Musical Taste Profiles for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      One particularly promising use case of Large Language Models (LLMs) for recommendation is the automatic generation of Natural Language (NL) user taste profiles from consumption data. These profiles offer interpretable and editable alternatives to opaque collaborative filtering representations, enabling greater transparency and user control. However, it remains unclear whether users consider these profiles to be an accurate representation of their taste, which is crucial for trust and usability. Moreover, because LLMs inherit societal and data-driven biases, profile quality may systematically vary across user and item characteristics. In this paper, we study this issue in the context of music streaming, where personalization is challenged by a large and culturally diverse catalog. We conduct a user study in which participants rate NL profiles generated from their own listening histories. We analyze whether identification with the profiles is biased by user attributes (e.g., mainstreamness, taste diversity) and item features (e.g., genre, country of origin). We also compare these patterns to those observed when using the profiles in a downstream recommendation task. Our findings highlight both the potential and limitations of scrutable, LLM-based profiling in personalized systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08773v2">Universal Model Routing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Model routing is a simple technique for reducing the inference cost of large language models (LLMs), wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose UniRoute, a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective instantiations of UniRoute, relying on cluster-based routing and a learned cluster map respectively. We show that these are estimates of a theoretically optimal routing rule, and quantify their errors via an excess risk bound. Experiments on a range of public benchmarks show the effectiveness of UniRoute in routing amongst more than 30 unseen LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16679v1">PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16672v1">Meta-Learning for Cold-Start Personalization in Prompt-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Generative, explainable, and flexible recommender systems, derived using Large Language Models (LLM) are promising and poorly adapted to the cold-start user situation, where there is little to no history of interaction. The current solutions i.e. supervised fine-tuning and collaborative filtering are dense-user-item focused and would be expensive to maintain and update. This paper introduces a meta-learning framework, that can be used to perform parameter-efficient prompt-tuning, to effectively personalize LLM-based recommender systems quickly at cold-start. The model learns soft prompt embeddings with first-order (Reptile) and second-order (MAML) optimization by treating each of the users as the tasks. As augmentations to the input tokens, these learnable vectors are the differentiable control variables that represent user behavioral priors. The prompts are meta-optimized through episodic sampling, inner-loop adaptation, and outer-loop generalization. On MovieLens-1M, Amazon Reviews, and Recbole, we can see that our adaptive model outperforms strong baselines in NDCG@10, HR@10, and MRR, and it runs in real-time (i.e., below 300 ms) on consumer GPUs. Zero-history personalization is also supported by this scalable solution, and its 275 ms rate of adaptation allows successful real-time risk profiling of financial systems by shortening detection latency and improving payment network stability. Crucially, the 275 ms adaptation capability can enable real-time risk profiling for financial institutions, reducing systemic vulnerability detection latency significantly versus traditional compliance checks. By preventing contagion in payment networks (e.g., Fedwire), the framework strengthens national financial infrastructure resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16656v1">P-CoT: A Pedagogically-motivated Participatory Chain-of-Thought Prompting for Phonological Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      This study explores the potential of phonological reasoning within text-based large language models (LLMs). Utilizing the PhonologyBench benchmark, we assess tasks like rhyme word generation, g2p conversion, and syllable counting. Our evaluations across 12 LLMs reveal that while few-shot learning offers inconsistent gains, the introduction of a novel Pedagogically-motivated Participatory Chain-of-Thought (P-CoT) prompt, which is anchored in educational theories like scaffolding and discovery learning, consistently enhances performance. This method leverages structured guidance to activate latent phonological abilities, achieving up to 52% improvement and even surpassing human baselines in certain tasks. Future work could aim to optimize P-CoT prompts for specific models or explore their application across different linguistic domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16587v1">On the Effectiveness of LLM-as-a-judge for Code Generation and Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at TSE. IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Large Language Models have been recently exploited as judges for complex natural language processing tasks, such as Q&A. The basic idea is to delegate to an LLM the assessment of the "quality" of the output provided by an automated technique for tasks for which: (i) quantitative metrics would only tell part of the story, and; (ii) a large-scale human-based evaluation would be too expensive. LLMs-as-a-judge, if proven effective for a specific task, can also unlock new possibilities for automation, with several LLMs proposing a solution for a given instance of the task and others judging and deciding what is the best output to show the user. We study the effectiveness of LLMs-as-a-judge for two code-related tasks, namely code generation and code summarization. The rationale for choosing these tasks is two-fold. First, quantitative metrics are usually not enough for the assessment of code summarizers/generators. For example, it is well documented that metrics such as BLEU are quite weak proxies for the quality of the generated summaries. Second, even state-of-the-art techniques still struggle with handling complex instances of these tasks, making them good candidates for benefiting from more advanced solutions envisioning collaboration among LLMs. For code generation, we check whether eight LLMs are able to judge the correctness of 1,405 Java methods and 1,281 Python functions generated by the same LLMs or implemented by humans. For code summarization, we compare the judgment of five LLMs to those provided by nine humans for ~1.2k summaries, related to both Java and Python functions. Our findings show that GPT-4-turbo is the best LLM in terms of judging capabilities for both tasks, with "smaller" LLMs featuring tens of billions parameters not being able to cope with judging tasks. However, even the best-performing LLM frequently misjudges the correctness of the code and summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06821v3">Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 37 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v2">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v5">Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Project page: https://github.com/VITA-MLLM/Sparrow
    </div>
    <details class="paper-abstract">
      Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16488v1">ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at various natural language processing tasks, but their tendency to generate hallucinations undermines their reliability. Existing hallucination detection methods leveraging hidden states predominantly focus on static and isolated representations, overlooking their dynamic evolution across layers, which limits efficacy. To address this limitation, we shift the focus to the hidden state update process and introduce a novel metric, the ICR Score (Information Contribution to Residual Stream), which quantifies the contribution of modules to the hidden states' update. We empirically validate that the ICR Score is effective and reliable in distinguishing hallucinations. Building on these insights, we propose a hallucination detection method, the ICR Probe, which captures the cross-layer evolution of hidden states. Experimental results show that the ICR Probe achieves superior performance with significantly fewer parameters. Furthermore, ablation studies and case analyses offer deeper insights into the underlying mechanism of this method, improving its interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16456v1">An approach to measuring the performance of Automatic Speech Recognition (ASR) models in the context of Large Language Model (LLM) powered applications</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) plays a crucial role in human-machine interaction and serves as an interface for a wide range of applications. Traditionally, ASR performance has been evaluated using Word Error Rate (WER), a metric that quantifies the number of insertions, deletions, and substitutions in the generated transcriptions. However, with the increasing adoption of large and powerful Large Language Models (LLMs) as the core processing component in various applications, the significance of different types of ASR errors in downstream tasks warrants further exploration. In this work, we analyze the capabilities of LLMs to correct errors introduced by ASRs and propose a new measure to evaluate ASR performance for LLM-powered applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13246v2">Atomic Calibration of LLMs in Long-Form Generations</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 ACL 2025 KnowFM Oral
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, which estimates the underlying uncertainty of model predictions, is essential to enhance the LLMs' trustworthiness. Existing research on LLM calibration has primarily focused on short-form tasks, providing a single confidence score at the response level (macro calibration). However, this approach is insufficient for long-form generations, where responses often contain more complex statements and may include both accurate and inaccurate information. Therefore, we introduce atomic calibration, a novel approach that evaluates factuality calibration at a fine-grained level by breaking down long responses into atomic claims. We classify confidence elicitation methods into discriminative and generative types and demonstrate that their combination can enhance calibration. Our extensive experiments on various LLMs and datasets show that atomic calibration is well-suited for long-form generation and can also improve macro calibration results. Additionally, atomic calibration reveals insightful patterns in LLM confidence throughout the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16414v1">Identifying Pre-training Data in LLMs: A Neuron Activation-Based Detection Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) is closely tied to their training data, which can include copyrighted material or private information, raising legal and ethical concerns. Additionally, LLMs face criticism for dataset contamination and internalizing biases. To address these issues, the Pre-Training Data Detection (PDD) task was proposed to identify if specific data was included in an LLM's pre-training corpus. However, existing PDD methods often rely on superficial features like prediction confidence and loss, resulting in mediocre performance. To improve this, we introduce NA-PDD, a novel algorithm analyzing differential neuron activation patterns between training and non-training data in LLMs. This is based on the observation that these data types activate different neurons during LLM inference. We also introduce CCNewsPDD, a temporally unbiased benchmark employing rigorous data transformations to ensure consistent time distributions between training and non-training data. Our experiments demonstrate that NA-PDD significantly outperforms existing methods across three benchmarks and multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16407v1">Improving Code LLM Robustness to Prompt Perturbations via Layer-Aware Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, where the natural language prompt plays a crucial role in conveying user intent to the model. However, prior studies have shown that LLMs are highly sensitive to prompt perturbations. Minor modifications in wording, syntax, or formatting can significantly reduce the functional correctness of generated code. As perturbations frequently occur in real-world scenarios, improving the robustness of LLMs to prompt perturbations is essential for ensuring reliable performance in practical code generation. In this paper, we introduce CREME (Code Robustness Enhancement via Model Editing), a novel approach that enhances LLM robustness through targeted parameter updates. CREME first identifies robustness-sensitive layers by comparing hidden states between an original prompt and its perturbed variant. Then, it performs lightweight parameter editing at the identified layer to reduce performance degradation. We evaluate CREME on two widely used code generation benchmarks (HumanEval and MBPP) along with their perturbed counterparts. Experimental results show that CREME improves Pass@1 accuracy by 63% on perturbed prompts while maintaining stable performance on clean inputs, with accuracy deviations within 1%. Further analysis reveals that robustness-sensitive layers are primarily concentrated in the middle and deeper layers of the network, and their locations vary across different model architectures. These insights provide a valuable foundation for developing future robustness-oriented editing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16395v1">LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Atomic commits, each of which addresses a single development concern, are a best practice in software development. However, developers frequently produce tangled commits that mix unrelated changes due to practical constraints or unclear boundaries, negatively impacting code review and maintenance. Although prior commit untangling approaches: rule-based, feature-based, or graph-based, have made progress, they often rely on shallow signals and fail to distinguish between explicit dependencies (e.g., control/data flow) and implicit ones (e.g., semantic or conceptual relationships). In this paper, we propose ColaUntangle, a new collaborative consultation framework for commit untangling that models both explicit and implicit dependencies among code changes. ColaUntangle integrates Large Language Model (LLM)-driven agents in a multi-agent architecture: one agent specializes in explicit dependencies, another in implicit ones, and a reviewer agent synthesizes their perspectives through iterative consultation. To capture explicit and implicit contextual information, we construct multi-version Program Dependency Graphs (delta-PDG), enabling agents to reason over code relationships with both symbolic and semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C# and 14k Java tangled commits). Experimental results show that ColaUntangle outperforms the best-performing baseline, achieving an improvement of 44% on the C# dataset and 100% on the Java dataset. These findings highlight the potential of LLM-based collaborative frameworks for advancing automated commit untangling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16382v1">Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      Multi-Agent Systems (MAS) excel at accomplishing complex objectives through the collaborative efforts of individual agents. Among the methodologies employed in MAS, Multi-Agent Reinforcement Learning (MARL) stands out as one of the most efficacious algorithms. However, when confronted with the complex objective of Formation Control with Collision Avoidance (FCCA): designing an effective reward function that facilitates swift convergence of the policy network to an optimal solution. In this paper, we introduce a novel framework that aims to overcome this challenge. By giving large language models (LLMs) on the prioritization of tasks and the observable information available to each agent, our framework generates reward functions that can be dynamically adjusted online based on evaluation outcomes by employing more advanced evaluation metrics rather than the rewards themselves. This mechanism enables the MAS to simultaneously achieve formation control and obstacle avoidance in dynamic environments with enhanced efficiency, requiring fewer iterations to reach superior performance levels. Our empirical studies, conducted in both simulation and real-world settings, validate the practicality and effectiveness of our proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16372v1">Depth Gives a False Sense of Privacy: LLM Internal States Inversion</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by USENIX Security 2025. Please cite this paper as "Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu. Depth Gives a False Sense of Privacy: LLM Internal States Inversion. In the 34th USENIX Security Symposium (USENIX Security '25)."
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09164v6">ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08472v2">Pre-Training LLMs on a budget: A comparison of three optimizers</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v4">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 take a middle-ground approach, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfiniteHBD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level} using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfiniteHBD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt to variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh)-based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfiniteHBD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios are under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07457v2">LLMs syntactically adapt their language use to their conversational partner</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 5 pages, 1 table, 3 figures, accepted at ACL (main conference) 2025
    </div>
    <details class="paper-abstract">
      It has been frequently observed that human speakers align their language use with each other during conversations. In this paper, we study empirically whether large language models (LLMs) exhibit the same behavior of conversational adaptation. We construct a corpus of conversations between LLMs and find that two LLM agents end up making more similar syntactic choices as conversations go on, confirming that modern LLMs adapt their language use to their conversational partners in at least a rudimentary way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14430v2">X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16331v1">Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16322v1">Mind the Gap: Evaluating the Representativeness of Quantitative Medical Language Reasoning LLM Benchmarks for African Disease Burdens</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Preprint. 26 pages, includes appendix and tables
    </div>
    <details class="paper-abstract">
      Introduction: Existing medical LLM benchmarks largely reflect examination syllabi and disease profiles from high income settings, raising questions about their validity for African deployment where malaria, HIV, TB, sickle cell disease and other neglected tropical diseases (NTDs) dominate burden and national guidelines drive care. Methodology: We systematically reviewed 31 quantitative LLM evaluation papers (Jan 2019 May 2025) identifying 19 English medical QA benchmarks. Alama Health QA was developed using a retrieval augmented generation framework anchored on the Kenyan Clinical Practice Guidelines. Six widely used sets (AfriMedQA, MMLUMedical, PubMedQA, MedMCQA, MedQAUSMLE, and guideline grounded Alama Health QA) underwent harmonized semantic profiling (NTD proportion, recency, readability, lexical diversity metrics) and blinded expert rating across five dimensions: clinical relevance, guideline alignment, clarity, distractor plausibility, and language/cultural fit. Results: Alama Health QA captured >40% of all NTD mentions across corpora and the highest within set frequencies for malaria (7.7%), HIV (4.1%), and TB (5.2%); AfriMedQA ranked second but lacked formal guideline linkage. Global benchmarks showed minimal representation (e.g., sickle cell disease absent in three sets) despite large scale. Qualitatively, Alama scored highest for relevance and guideline alignment; PubMedQA lowest for clinical utility. Discussion: Quantitative medical LLM benchmarks widely used in the literature underrepresent African disease burdens and regulatory contexts, risking misleading performance claims. Guideline anchored, regionally curated resources such as Alama Health QA and expanded disease specific derivatives are essential for safe, equitable model evaluation and deployment across African health systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16307v1">Perovskite-R1: A Domain-Specialized LLM for Intelligent Discovery of Precursor Additives and Experimental Design</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 24 pages; 5 figures
    </div>
    <details class="paper-abstract">
      Perovskite solar cells (PSCs) have rapidly emerged as a leading contender in next-generation photovoltaic technologies, owing to their exceptional power conversion efficiencies and advantageous material properties. Despite these advances, challenges such as long-term stability, environmental sustainability, and scalable manufacturing continue to hinder their commercialization. Precursor additive engineering has shown promise in addressing these issues by enhancing both the performance and durability of PSCs. However, the explosive growth of scientific literature and the complex interplay of materials, processes, and device architectures make it increasingly difficult for researchers to efficiently access, organize, and utilize domain knowledge in this rapidly evolving field. To address this gap, we introduce Perovskite-R1, a specialized large language model (LLM) with advanced reasoning capabilities tailored for the discovery and design of PSC precursor additives. By systematically mining and curating 1,232 high-quality scientific publications and integrating a comprehensive library of 33,269 candidate materials, we constructed a domain-specific instruction-tuning dataset using automated question-answer generation and chain-of-thought reasoning. Fine-tuning the QwQ-32B model on this dataset resulted in Perovskite-R1, which can intelligently synthesize literature insights and generate innovative and practical solutions for defect passivation and the selection of precursor additives. Experimental validation of several model-proposed strategies confirms their effectiveness in improving material stability and performance. Our work demonstrates the potential of domain-adapted LLMs in accelerating materials discovery and provides a closed-loop framework for intelligent, data-driven advancements in perovskite photovoltaic research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22139v3">Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v4">OMNISEC: LLM-Driven Provenance-based Intrusion Detection via Retrieval-Augmented Behavior Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Recently, Provenance-based Intrusion Detection Systems (PIDSes) have been widely used for endpoint threat analysis. These studies can be broadly categorized into rule-based detection systems and learning-based detection systems. Among these, due to the evolution of attack techniques, rules cannot dynamically model all the characteristics of attackers. As a result, such systems often face false negatives. Learning-based detection systems are further divided into supervised learning and anomaly detection. The scarcity of attack samples hinders the usability and effectiveness of supervised learning-based detection systems in practical applications. Anomaly-based detection systems face a massive false positive problem because they cannot distinguish between changes in normal behavior and real attack behavior. The alert results of detection systems are closely related to the manual labor costs of subsequent security analysts. To reduce manual analysis time, we propose OMNISEC, which applies large language models (LLMs) to anomaly-based intrusion detection systems via retrieval-augmented behavior prompting. OMNISEC can identify abnormal nodes and corresponding abnormal events by constructing suspicious nodes and rare paths. By combining two external knowledge bases, OMNISEC uses Retrieval Augmented Generation (RAG) to enable the LLM to determine whether abnormal behavior is a real attack. Finally, OMNISEC can reconstruct the attack graph and restore the complete attack behavior chain of the attacker's intrusion. Experimental results show that OMNISEC outperforms state-of-the-art methods on public benchmark datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16291v1">Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by EAI ICDF2C 2025
    </div>
    <details class="paper-abstract">
      Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18527v3">Probing Ranking LLMs: A Mechanistic Analysis for Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Transformer networks, particularly those achieving performance comparable to GPT models, are well known for their robust feature extraction abilities. However, the nature of these extracted features and their alignment with human-engineered ones remain unexplored. In this work, we investigate the internal mechanisms of state-of-the-art, fine-tuned LLMs for passage reranking. We employ a probing-based analysis to examine neuron activations in ranking LLMs, identifying the presence of known human-engineered and semantic features. Our study spans a broad range of feature categories, including lexical signals, document structure, query-document interactions, and complex semantic representations, to uncover underlying patterns influencing ranking decisions. Through experiments on four different ranking LLMs, we identify statistical IR features that are prominently encoded in LLM activations, as well as others that are notably missing. Furthermore, we analyze how these models respond to out-of-distribution queries and documents, revealing distinct generalization behaviors. By dissecting the latent representations within LLM activations, we aim to improve both the interpretability and effectiveness of ranking models. Our findings offer crucial insights for developing more transparent and reliable retrieval systems, and we release all necessary scripts and code to support further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16252v1">Efficient RL for optimizing conversation level outcomes with an LLM-based tutor</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) built on existing reinforcement learning with human feedback (RLHF) frameworks typically optimize responses based on immediate turn-level human preferences. However, this approach falls short in multi-turn dialogue settings, such as online math tutoring. We propose a method to enhance LLM-based tutors by representing the dialogue history with a lower-dimensional latent state representation of a student and optimizing a long-term policy to determine high-level actions based on the latent state. The goal is to better align the tutor's behavior with the long-term objective of guiding the student towards solving a target math problem on their own. Our model is lightweight, requiring less computational resources than prior work of training the tutor policy end-to-end to directly output the tutor's next utterance. Our experiment results demonstrate that these modifications lead to improved long-term outcomes compared to prompting in LLM-simulated tutoring tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16237v1">LLM-Enhanced Reranking for Complementary Product Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Complementary product recommendation, which aims to suggest items that are used together to enhance customer value, is a crucial yet challenging task in e-commerce. While existing graph neural network (GNN) approaches have made significant progress in capturing complex product relationships, they often struggle with the accuracy-diversity tradeoff, particularly for long-tail items. This paper introduces a model-agnostic approach that leverages Large Language Models (LLMs) to enhance the reranking of complementary product recommendations. Unlike previous works that use LLMs primarily for data preprocessing and graph augmentation, our method applies LLM-based prompting strategies directly to rerank candidate items retrieved from existing recommendation models, eliminating the need for model retraining. Through extensive experiments on public datasets, we demonstrate that our approach effectively balances accuracy and diversity in complementary product recommendations, with at least 50% lift in accuracy metrics and 2% lift in diversity metrics on average for the top recommended items across datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v1">WakenLLM: A Fine-Grained Benchmark for Evaluating LLM Reasoning Potential and Reasoning Process Stability</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label \emph{Unknown}, yet current evaluations focus almost exclusively on whether such answers are \emph{honest} rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon \emph{Vague Perception}. And thus we introduce a framework that quantifies the proportion of \emph{Unknown} responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct (\emph{Known}) or intrinsically indeterminate outcomes. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the true reasoning ability of LLMs and providing a new perspective on solving the \emph{Vague Perception} phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16178v1">LLM Data Selection and Utilization via Dynamic Bi-level Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 The 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      While large-scale training data is fundamental for developing capable large language models (LLMs), strategically selecting high-quality data has emerged as a critical approach to enhance training efficiency and reduce computational costs. Current data selection methodologies predominantly rely on static, training-agnostic criteria, failing to account for the dynamic model training and data interactions. In this paper, we propose a new Data Weighting Model (DWM) to adjust the weight of selected data within each batch to achieve a dynamic data utilization during LLM training. Specially, to better capture the dynamic data preference of the trained model, a bi-level optimization framework is implemented to update the weighting model. Our experiments demonstrate that DWM enhances the performance of models trained with randomly-selected data, and the learned weighting model can be transferred to enhance other data selection methods and models of different sizes. Moreover, we further analyze how a model's data preferences evolve throughout training, providing new insights into the data preference of the model during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01661v2">R-Bot: An LLM-based Query Rewrite System</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Query rewrite is essential for optimizing SQL queries to improve their execution efficiency without changing their results. Traditionally, this task has been tackled through heuristic and learning-based methods, each with its limitations in terms of inferior quality and low robustness. Recent advancements in LLMs offer a new paradigm by leveraging their superior natural language and code comprehension abilities. Despite their potential, directly applying LLMs like GPT-4 has faced challenges due to problems such as hallucinations, where the model might generate inaccurate or irrelevant results. To address this, we propose R-Bot, an LLM-based query rewrite system with a systematic approach. We first design a multi-source rewrite evidence preparation pipeline to generate query rewrite evidences for guiding LLMs to avoid hallucinations. We then propose a hybrid structure-semantics retrieval method that combines structural and semantic analysis to retrieve the most relevant rewrite evidences for effectively answering an online query. We next propose a step-by-step LLM rewrite method that iteratively leverages the retrieved evidences to select and arrange rewrite rules with self-reflection. We conduct comprehensive experiments on real-world datasets and widely used benchmarks, and demonstrate the superior performance of our system, R-Bot, surpassing state-of-the-art query rewrite methods. The R-Bot system has been deployed at Huawei and with real customers, and the results show that the proposed R-Bot system achieves lower query latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16145v1">SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8980 (95% CI: 0.8820-0.9132). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16130v1">Disability Across Cultures: A Human-Centered Audit of Ableism in Western and Indic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      People with disabilities (PwD) experience disproportionately high levels of discrimination and hate online, particularly in India, where entrenched stigma and limited resources intensify these challenges. Large language models (LLMs) are increasingly used to identify and mitigate online hate, yet most research on online ableism focuses on Western audiences with Western AI models. Are these models adequately equipped to recognize ableist harm in non-Western places like India? Do localized, Indic language models perform better? To investigate, we adopted and translated a publicly available ableist speech dataset to Hindi, and prompted eight LLMs--four developed in the U.S. (GPT-4, Gemini, Claude, Llama) and four in India (Krutrim, Nanda, Gajendra, Airavata)--to score and explain ableism. In parallel, we recruited 175 PwD from both the U.S. and India to perform the same task, revealing stark differences between groups. Western LLMs consistently overestimated ableist harm, while Indic LLMs underestimated it. Even more concerning, all LLMs were more tolerant of ableism when it was expressed in Hindi and asserted Western framings of ableist harm. In contrast, Indian PwD interpreted harm through intention, relationality, and resilience--emphasizing a desire to inform and educate perpetrators. This work provides groundwork for global, inclusive standards of ableism, demonstrating the need to center local disability experiences in the design and evaluation of AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16124v1">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      Social robots are embodied agents that interact with people while following human communication norms. These robots interact using verbal and non-verbal cues, and share the physical environments of people. While social robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-empowered social robots for enhanced human-robot interaction. To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within home environments. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household social robots. In this study, we present a set of privacy-relevant scenarios crafted through the lens of Contextual Integrity (CI). We first survey users' privacy preferences regarding in-home social robot behaviors and then examine how their privacy orientation affects their choices of these behaviors (N = 450). We then provide the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and find that the agreement between humans and LLMs is low. To further investigate the capabilities of LLMs as a potential privacy controller, we implement four additional prompting strategies and compare their results. Finally, we discuss the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16117v1">BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 11 pages, 9 figures. Accepted to IEEE VIS 2025 (Full Papers Track, submission ID 1204)
    </div>
    <details class="paper-abstract">
      Biomedical data harmonization is essential for enabling exploratory analyses and meta-studies, but the process of schema matching - identifying semantic correspondences between elements of disparate datasets (schemas) - remains a labor-intensive and error-prone task. Even state-of-the-art automated methods often yield low accuracy when applied to biomedical schemas due to the large number of attributes and nuanced semantic differences between them. We present BDIViz, a novel visual analytics system designed to streamline the schema matching process for biomedical data. Through formative studies with domain experts, we identified key requirements for an effective solution and developed interactive visualization techniques that address both scalability challenges and semantic ambiguity. BDIViz employs an ensemble approach that combines multiple matching methods with LLM-based validation, summarizes matches through interactive heatmaps, and provides coordinated views that enable users to quickly compare attributes and their values. Our method-agnostic design allows the system to integrate various schema matching algorithms and adapt to application-specific needs. Through two biomedical case studies and a within-subject user study with domain experts, we demonstrate that BDIViz significantly improves matching accuracy while reducing cognitive load and curation time compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17080v1">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at RecSys 2025; DOI:https://doi.org/10.1145/3705328.3748064
    </div>
    <details class="paper-abstract">
      Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17075v1">LoRA is All You Need for Safety Alignment of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs -- with safety levels comparable to full-model fine-tuning -- without compromising their reasoning abilities. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. We also explore methods that further reduce such overlap -- via regularization or during weight merging -- and observe some improvement on certain tasks. We hope this result motivates designing approaches that yield more consistent improvements in the reasoning-safety trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17061v1">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10166v2">Fact-Checking with Contextual Narratives: Leveraging Retrieval-Augmented LLMs for Social Media Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      We propose CRAVE (Cluster-based Retrieval Augmented Verification with Explanation); a novel framework that integrates retrieval-augmented Large Language Models (LLMs) with clustering techniques to address fact-checking challenges on social media. CRAVE automatically retrieves multimodal evidence from diverse, often contradictory, sources. Evidence is clustered into coherent narratives, and evaluated via an LLM-based judge to deliver fact-checking verdicts explained by evidence summaries. By synthesizing evidence from both text and image modalities and incorporating agent-based refinement, CRAVE ensures consistency and diversity in evidence representation. Comprehensive experiments demonstrate CRAVE's efficacy in retrieval precision, clustering quality, and judgment accuracy, showcasing its potential as a robust decision-support tool for fact-checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17016v1">Causal Graph Fuzzy LLMs: A First Introduction and Applications in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted for publication at the Brazilian Congress of Artificial Intelligence (CBIC)
    </div>
    <details class="paper-abstract">
      In recent years, the application of Large Language Models (LLMs) to time series forecasting (TSF) has garnered significant attention among researchers. This study presents a new frame of LLMs named CGF-LLM using GPT-2 combined with fuzzy time series (FTS) and causal graph to predict multivariate time series, marking the first such architecture in the literature. The key objective is to convert numerical time series into interpretable forms through the parallel application of fuzzification and causal analysis, enabling both semantic understanding and structural insight as input for the pretrained GPT-2 model. The resulting textual representation offers a more interpretable view of the complex dynamics underlying the original time series. The reported results confirm the effectiveness of our proposed LLM-based time series forecasting model, as demonstrated across four different multivariate time series datasets. This initiative paves promising future directions in the domain of TSF using LLMs based on FTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17015v1">Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at https://github.com/apple/ml-agent-evaluator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05200v2">ORANSight-2.0: Foundational LLMs for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative to develop specialized foundational LLMs tailored for O-RAN. Built on 18 models spanning five open-source LLM frameworks -- Mistral, Qwen, Llama, Phi, and Gemma -- ORANSight-2.0 fine-tunes models ranging from 1B to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance in O-RAN-specific tasks. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG)-based instruction-tuning framework that employs two LLM agents -- a Mistral-based Question Generator and a Qwen-based Answer Generator -- to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16989v1">Obscured but Not Erased: Evaluating Nationality Bias in LLMs via Name-Based Bias Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can exhibit latent biases towards specific nationalities even when explicit demographic markers are not present. In this work, we introduce a novel name-based benchmarking approach derived from the Bias Benchmark for QA (BBQ) dataset to investigate the impact of substituting explicit nationality labels with culturally indicative names, a scenario more reflective of real-world LLM applications. Our novel approach examines how this substitution affects both bias magnitude and accuracy across a spectrum of LLMs from industry leaders such as OpenAI, Google, and Anthropic. Our experiments show that small models are less accurate and exhibit more bias compared to their larger counterparts. For instance, on our name-based dataset and in the ambiguous context (where the correct choice is not revealed), Claude Haiku exhibited the worst stereotypical bias scores of 9%, compared to only 3.5% for its larger counterpart, Claude Sonnet, where the latter also outperformed it by 117.7% in accuracy. Additionally, we find that small models retain a larger portion of existing errors in these ambiguous contexts. For example, after substituting names for explicit nationality references, GPT-4o retains 68% of the error rate versus 76% for GPT-4o-mini, with similar findings for other model providers, in the ambiguous context. Our research highlights the stubborn resilience of biases in LLMs, underscoring their profound implications for the development and deployment of AI systems in diverse, global contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16974v1">Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 15 pages, 9 tables, Appendix A-K
    </div>
    <details class="paper-abstract">
      Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16951v1">Harnessing RLHF for Robust Unanswerability Recognition and Trustworthy Response Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Conversational Information Retrieval (CIR) systems, while offering intuitive access to information, face a significant challenge: reliably handling unanswerable questions to prevent the generation of misleading or hallucinated content. Traditional approaches often rely on external classifiers, which can introduce inconsistencies with the core generative Large Language Models (LLMs). This paper introduces Self-Aware LLM for Unanswerability (SALU), a novel approach that deeply integrates unanswerability detection directly within the LLM's generative process. SALU is trained using a multi-task learning framework for both standard Question Answering (QA) and explicit abstention generation for unanswerable queries. Crucially, it incorporates a confidence-score-guided reinforcement learning with human feedback (RLHF) phase, which explicitly penalizes hallucinated responses and rewards appropriate abstentions, fostering intrinsic self-awareness of knowledge boundaries. Through extensive experiments on our custom-built C-IR_Answerability dataset, SALU consistently outperforms strong baselines, including hybrid LLM-classifier systems, in overall accuracy for correctly answering or abstaining from questions. Human evaluation further confirms SALU's superior reliability, achieving high scores in factuality, appropriate abstention, and, most importantly, a dramatic reduction in hallucination, demonstrating its ability to robustly "know when to say 'I don't know'."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18489v2">LLM as a code generator in Agile Model Driven Development</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLM) like GPT4 in the auto generation of code represents a significant advancement, yet it is not without its challenges. The ambiguity inherent in natural language descriptions of software poses substantial obstacles to generating deployable, structured artifacts. This research champions Model Driven Development (MDD) as a viable strategy to overcome these challenges, proposing an Agile Model Driven Development (AMDD) approach that employs GPT4 as a code generator. This approach enhances the flexibility and scalability of the code auto generation process and offers agility that allows seamless adaptation to changes in models or deployment environments. We illustrate this by modeling a multi agent Unmanned Vehicle Fleet (UVF) system using the Unified Modeling Language (UML), significantly reducing model ambiguity by integrating the Object Constraint Language (OCL) for code structure meta modeling, and the FIPA ontology language for communication semantics meta modeling. Applying GPT4 auto generation capabilities yields Java and Python code that is compatible with the JADE and PADE frameworks, respectively. Our thorough evaluation of the auto generated code verifies its alignment with expected behaviors and identifies enhancements in agent interactions. Structurally, we assessed the complexity of code derived from a model constrained solely by OCL meta models, against that influenced by both OCL and FIPA ontology meta models. The results indicate that the ontology constrained meta model produces inherently more complex code, yet its cyclomatic complexity remains within manageable levels, suggesting that additional meta model constraints can be incorporated without exceeding the high risk threshold for complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00178v2">Tournament of Prompts: Evolving LLM Instructions Through Structured Debates and Elo Ratings</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Prompt engineering represents a critical bottleneck to harness the full potential of Large Language Models (LLMs) for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention. This challenge is particularly pronounced for tasks involving subjective quality assessment, where defining explicit optimization objectives becomes fundamentally problematic. Existing automated prompt optimization methods falter in these scenarios, as they typically require well-defined task-specific numerical fitness functions or rely on generic templates that cannot capture the nuanced requirements of complex use cases. We introduce DEEVO (DEbate-driven EVOlutionary prompt optimization), a novel framework that guides prompt evolution through a debate-driven evaluation with an Elo-based selection. Contrary to prior work, DEEVOs approach enables exploration of the discrete prompt space while preserving semantic coherence through intelligent crossover and strategic mutation operations that incorporate debate-based feedback, combining elements from both successful and unsuccessful prompts based on identified strengths rather than arbitrary splicing. Using Elo ratings as a fitness proxy, DEEVO simultaneously drives improvement and preserves valuable diversity in the prompt population. Experimental results demonstrate that DEEVO significantly outperforms both manual prompt engineering and alternative state-of-the-art optimization approaches on open-ended tasks and close-ended tasks despite using no ground truth feedback. By connecting LLMs reasoning capabilities with adaptive optimization, DEEVO represents a significant advancement in prompt optimization research by eliminating the need of predetermined metrics to continuously improve AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15849v1">The Impact of Language Mixing on Bilingual LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15839v1">FASTGEN: Fast and Cost-Effective Synthetic Tabular Data Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Synthetic data generation has emerged as an invaluable solution in scenarios where real-world data collection and usage are limited by cost and scarcity. Large language models (LLMs) have demonstrated remarkable capabilities in producing high-fidelity, domain-relevant samples across various fields. However, existing approaches that directly use LLMs to generate each record individually impose prohibitive time and cost burdens, particularly when large volumes of synthetic data are required. In this work, we propose a fast, cost-effective method for realistic tabular data synthesis that leverages LLMs to infer and encode each field's distribution into a reusable sampling script. By automatically classifying fields into numerical, categorical, or free-text types, the LLM generates distribution-based scripts that can efficiently produce diverse, realistic datasets at scale without continuous model inference. Experimental results show that our approach outperforms traditional direct methods in both diversity and data realism, substantially reducing the burden of high-volume synthetic data generation. We plan to apply this methodology to accelerate testing in production pipelines, thereby shortening development cycles and improving overall system efficiency. We believe our insights and lessons learned will aid researchers and practitioners seeking scalable, cost-effective solutions for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15828v1">Investigating the Use of LLMs for Evidence Briefings Generation in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ESEM 2025 Registered Report with an IPA (In Principle Acceptance) for the Empirical Software Engineering journal
    </div>
    <details class="paper-abstract">
      [Context] An evidence briefing is a concise and objective transfer medium that can present the main findings of a study to software engineers in the industry. Although practitioners and researchers have deemed Evidence Briefings useful, their production requires manual labor, which may be a significant challenge to their broad adoption. [Goal] The goal of this registered report is to describe an experimental protocol for evaluating LLM-generated evidence briefings for secondary studies in terms of content fidelity, ease of understanding, and usefulness, as perceived by researchers and practitioners, compared to human-made briefings. [Method] We developed an RAG-based LLM tool to generate evidence briefings. We used the tool to automatically generate two evidence briefings that had been manually generated in previous research efforts. We designed a controlled experiment to evaluate how the LLM-generated briefings compare to the human-made ones regarding perceived content fidelity, ease of understanding, and usefulness. [Results] To be reported after the experimental trials. [Conclusion] Depending on the experiment results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15821v1">Just Put a Human in the Loop? Investigating LLM-Assisted Annotation for Subjective Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      LLM use in annotation is becoming widespread, and given LLMs' overall promising performance and speed, simply "reviewing" LLM annotations in interpretive tasks can be tempting. In subjective annotation tasks with multiple plausible answers, reviewing LLM outputs can change the label distribution, impacting both the evaluation of LLM performance, and analysis using these labels in a social science task downstream. We conducted a pre-registered experiment with 410 unique annotators and over 7,000 annotations testing three AI assistance conditions against controls, using two models, and two datasets. We find that presenting crowdworkers with LLM-generated annotation suggestions did not make them faster, but did improve their self-reported confidence in the task. More importantly, annotators strongly took the LLM suggestions, significantly changing the label distribution compared to the baseline. When these labels created with LLM assistance are used to evaluate LLM performance, reported model performance significantly increases. We believe our work underlines the importance of understanding the impact of LLM-assisted annotation on subjective, qualitative tasks, on the creation of gold data for training and testing, and on the evaluation of NLP systems on subjective tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15815v1">LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 27 pages, 6 figures, Code: https://github.com/sethkarten/LLM-Economist
    </div>
    <details class="paper-abstract">
      We present the LLM Economist, a novel framework that uses agent-based modeling to design and assess economic policies in strategic environments with hierarchical decision-making. At the lower level, bounded rational worker agents -- instantiated as persona-conditioned prompts sampled from U.S. Census-calibrated income and demographic statistics -- choose labor supply to maximize text-based utility functions learned in-context. At the upper level, a planner agent employs in-context reinforcement learning to propose piecewise-linear marginal tax schedules anchored to the current U.S. federal brackets. This construction endows economic simulacra with three capabilities requisite for credible fiscal experimentation: (i) optimization of heterogeneous utilities, (ii) principled generation of large, demographically realistic agent populations, and (iii) mechanism design -- the ultimate nudging problem -- expressed entirely in natural language. Experiments with populations of up to one hundred interacting agents show that the planner converges near Stackelberg equilibria that improve aggregate social welfare relative to Saez solutions, while a periodic, persona-level voting procedure furthers these gains under decentralized governance. These results demonstrate that large language model-based agents can jointly model, simulate, and govern complex economic systems, providing a tractable test bed for policy evaluation at the societal scale to help build better civilizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15788v1">Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15782v1">Interleaved LLM and Motion Planning for Generalized Multi-Object Collection in Large Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Household robots have been a longstanding research topic, but they still lack human-like intelligence, particularly in manipulating open-set objects and navigating large environments efficiently and accurately. To push this boundary, we consider a generalized multi-object collection problem in large scene graphs, where the robot needs to pick up and place multiple objects across multiple locations in a long mission of multiple human commands. This problem is extremely challenging since it requires long-horizon planning in a vast action-state space under high uncertainties. To this end, we propose a novel interleaved LLM and motion planning algorithm Inter-LLM. By designing a multimodal action cost similarity function, our algorithm can both reflect the history and look into the future to optimize plans, striking a good balance of quality and efficiency. Simulation experiments demonstrate that compared with latest works, our algorithm improves the overall mission performance by 30% in terms of fulfilling human commands, maximizing mission success rates, and minimizing mission costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v3">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15752v1">DialogueForge: LLM Simulation of Human-Chatbot Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 For our code and data, see https://github.com/nerchio/Human_Chatbot-Generation
    </div>
    <details class="paper-abstract">
      Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15717v1">BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15715v1">From Queries to Criteria: Understanding How Astronomers Evaluate LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15613v1">Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place. We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12601v2">CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to KDD 2025 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
    </div>
    <details class="paper-abstract">
      To broaden the dissemination of scientific knowledge to diverse audiences, it is desirable for scientific document summarization systems to simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, the first evaluation benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., conceptual or empirical focus), which are more subjective and abstract. We conduct extensive experiments using various large language models (LLMs) under various settings, including in-context learning, parameter-efficient fine-tuning, and two-stage modular methods for balancing control over different attributes. Our findings reveal significant limitations in LLMs capabilities in balancing trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15585v1">Unequal Voices: How LLMs Construct Constrained Queer Narratives</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      One way social groups are marginalized in discourse is that the narratives told about them often default to a narrow, stereotyped range of topics. In contrast, default groups are allowed the full complexity of human existence. We describe the constrained representations of queer people in LLM generations in terms of harmful representations, narrow representations, and discursive othering and formulate hypotheses to test for these phenomena. Our results show that LLMs are significantly limited in their portrayals of queer personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17181v2">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 13 pages, 8 tables, 2 figures. Paper was previously titled "LLMs Love Python"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate code, influencing users' choices of libraries and programming languages in critical real-world projects. However, little is known about their systematic biases or preferences toward certain libraries and programming languages, which can significantly impact software development practices. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. Our results reveal that LLMs exhibit a strong tendency to overuse widely adopted libraries such as NumPy; in up to 48% of cases, this usage is unnecessary and deviates from the ground-truth solutions. LLMs also exhibit a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used a single time. These results indicate that LLMs may prioritise familiarity and popularity over suitability and task-specific optimality. This will introduce security vulnerabilities and technical debt, and limit exposure to newly developed, better-suited tools and languages. Understanding and addressing these biases is essential for the responsible integration of LLMs into software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05445v2">clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15553v1">Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The rising demand for Large Language Model (LLM) inference services has intensified pressure on computational resources, resulting in latency and cost challenges. This paper introduces a novel routing algorithm based on the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to distribute inference requests across heterogeneous LLM instances in a cloud-edge computing environment. Formulated as a multi-objective optimization problem, the algorithm balances response quality, response time, and inference cost, adapting to request heterogeneity (e.g., varying complexity and prompt lengths) and node diversity (e.g., edge vs. cloud resources). This adaptive routing algorithm optimizes performance under dynamic workloads. We benchmark the approach using a testbed with datasets including Stanford Question Answering Dataset (SQuAD), Mostly Basic Python Problems (MBPP), Hella Situations With Adversarial Generations (HellaSwag), and Grade School Math 8K (GSM8K). Experimental results show our solution, compared to the baselines, achieves up to 95.2% and 34.9% improvements in terms of response time and cost, respectively. These findings validate the algorithm's effectiveness for scalable LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15550v1">PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 Pages
    </div>
    <details class="paper-abstract">
      Evaluating the scientific discovery capabilities of large language model based agents, particularly how they cope with varying environmental complexity and utilize prior knowledge, requires specialized benchmarks currently lacking in the landscape. To address this gap, we introduce PhysGym, a novel benchmark suite and simulation platform for rigorously assessing LLM-based scientific reasoning in interactive physics environments. PhysGym's primary contribution lies in its sophisticated control over the level of prior knowledge provided to the agent. This allows researchers to dissect agent performance along axes including the complexity of the problem and the prior knowledge levels. The benchmark comprises a suite of interactive simulations, where agents must actively probe environments, gather data sequentially under constraints and formulate hypotheses about underlying physical laws. PhysGym provides standardized evaluation protocols and metrics for assessing hypothesis accuracy and model fidelity. We demonstrate the benchmark's utility by presenting results from baseline LLMs, showcasing its ability to differentiate capabilities based on varying priors and task complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00024v2">Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ACL 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, human judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze whether 11 LLMs behave like humans in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15521v1">LLM world models are mental: Output layer evidence of brittle world model use in LLM mechanical reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Manuscript comprises 14 pages, 4 figures, 4 tables in the Technical Appendix and Supplementary Material, and is under review at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) construct and manipulate internal world models, or do they rely solely on statistical associations represented as output layer token probabilities? We adapt cognitive science methodologies from human mental models research to test LLMs on pulley system problems using TikZ-rendered stimuli. Study 1 examines whether LLMs can estimate mechanical advantage (MA). State-of-the-art models performed marginally but significantly above chance, and their estimates correlated significantly with ground-truth MA. Significant correlations between number of pulleys and model estimates suggest that models employed a pulley counting heuristic, without necessarily simulating pulley systems to derive precise values. Study 2 tested this by probing whether LLMs represent global features crucial to MA estimation. Models evaluated a functionally connected pulley system against a fake system with randomly placed components. Without explicit cues, models identified the functional system as having greater MA with F1=0.8, suggesting LLMs could represent systems well enough to differentiate jumbled from functional systems. Study 3 built on this by asking LLMs to compare functional systems with matched systems which were connected up but which transferred no force to the weight; LLMs identified the functional system with F1=0.46, suggesting random guessing. Insofar as they may generalize, these findings are compatible with the notion that LLMs manipulate internal world models, sufficient to exploit statistical associations between pulley count and MA (Study 1), and to approximately represent system components' spatial relations (Study 2). However, they may lack the facility to reason over nuanced structural connectivity (Study 3). We conclude by advocating the utility of cognitive scientific methods to evaluate the world-modeling capacities of artificial intelligence systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15502v1">FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Postoperative follow-up plays a crucial role in monitoring recovery and identifying complications. However, traditional approaches, typically involving bedside interviews and manual documentation, are time-consuming and labor-intensive. Although existing digital solutions, such as web questionnaires and intelligent automated calls, can alleviate the workload of nurses to a certain extent, they either deliver an inflexible scripted interaction or face private information leakage issues. To address these limitations, this paper introduces FollowUpBot, an LLM-powered edge-deployed robot for postoperative care and monitoring. It allows dynamic planning of optimal routes and uses edge-deployed LLMs to conduct adaptive and face-to-face conversations with patients through multiple interaction modes, ensuring data privacy. Moreover, FollowUpBot is capable of automatically generating structured postoperative follow-up reports for healthcare institutions by analyzing patient interactions during follow-up. Experimental results demonstrate that our robot achieves high coverage and satisfaction in follow-up interactions, as well as high report generation accuracy across diverse field types. The demonstration video is available at https://www.youtube.com/watch?v=_uFgDO7NoK0.
    </details>
</div>
