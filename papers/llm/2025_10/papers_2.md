# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20875v3">Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks (27 pages, 6 figures, 16 tables)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our code and datasets are publicly available at https://github.com/jiyounglee-0523/TransEnV and https://huggingface.co/collections/jiyounglee0523/transenv-681eadb3c0c8cf363b363fb1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01127v2">Can LLMs Grasp Implicit Cultural Values? Benchmarking LLMs' Cultural Intelligence with CQ-Bench</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Cultural Intelligence (CQ) refers to the ability to understand unfamiliar cultural contexts, a crucial skill for large language models (LLMs) to effectively engage with globally diverse users. Existing studies often focus on explicitly stated cultural norms, but fail to capture the subtle, implicit values that are common in daily conversation. To address this gap, we introduce CQBench, a benchmark specifically designed to assess LLMs' capability to infer implicit cultural values from natural conversational contexts. CQBench consists of multi character conversation based stories using values from the World Value Survey and the GlobalOpinions, with topics including ethical, religious, social, etc. Our automatic dataset construction pipeline integrates rigorous validation procedures (incorporation, consistency, and implicitness checks), achieving a 94.5% human model agreement in the final validation. To leverage CQBench data, we design three tasks of increasing complexity: attitude detection, value selection, and value extraction. These tasks evaluate whether models can detect attitude and recognize values embedded within natural dialogues rather than relying on explicit cultural knowledge. We find that while frontier models like o1 reach human level performance in value selection (0.809 F1), they still fall short in nuanced attitude detection (0.622 F1). Notably, finetuning a smaller LLaMA-3.2-3B on only 500 culturally rich examples improves performance by over 10%, even outperforming o3-mini in some cases. Using CQ-Bench, we provide insights into the current challenges in LLMs' CQ research and suggest practical pathways for enhancing LLMs' cross-cultural reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07931v1">Vision-Enabled LLMs in Historical Lexicography: Digitising and Enriching Estonian-German Dictionaries from the 17th and 18th Centuries</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      This article presents research conducted at the Institute of the Estonian Language between 2022 and 2025 on the application of large language models (LLMs) to the study of 17th and 18th century Estonian dictionaries. The authors address three main areas: enriching historical dictionaries with modern word forms and meanings; using vision-enabled LLMs to perform text recognition on sources printed in Gothic script (Fraktur); and preparing for the creation of a unified, cross-source dataset. Initial experiments with J. Gutslaff's 1648 dictionary indicate that LLMs have significant potential for semi-automatic enrichment of dictionary information. When provided with sufficient context, Claude 3.7 Sonnet accurately provided meanings and modern equivalents for 81% of headword entries. In a text recognition experiment with A. T. Helle's 1732 dictionary, a zero-shot method successfully identified and structured 41% of headword entries into error-free JSON-formatted output. For digitising the Estonian-German dictionary section of A. W. Hupel's 1780 grammar, overlapping tiling of scanned image files is employed, with one LLM being used for text recognition and a second for merging the structured output. These findings demonstrate that even for minor languages LLMs have a significant potential for saving time and financial resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07925v1">Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 8 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as the central control unit of AI agents, yet current approaches remain limited in their ability to deliver personalized interactions. While Retrieval Augmented Generation enhances LLM capabilities by improving context-awareness, it lacks mechanisms to combine contextual information with user-specific data. Although personalization has been studied in fields such as human-computer interaction or cognitive science, existing perspectives largely remain conceptual, with limited focus on technical implementation. To address these gaps, we build on a unified definition of personalization as a conceptual foundation to derive technical requirements for adaptive, user-centered LLM-based agents. Combined with established agentic AI patterns such as multi-agent collaboration or multi-source retrieval, we present a framework that integrates persistent memory, dynamic coordination, self-validation, and evolving user profiles to enable personalized long-term interactions. We evaluate our approach on three public datasets using metrics such as retrieval accuracy, response correctness, or BertScore. We complement these results with a five-day pilot user study providing initial insights into user feedback on perceived personalization. The study provides early indications that guide future work and highlights the potential of integrating persistent memory and user profiles to improve the adaptivity and perceived personalization of LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07920v1">Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      LLM-based financial agents have attracted widespread excitement for their ability to trade like human experts. However, most systems exhibit a "profit mirage": dazzling back-tested returns evaporate once the model's knowledge window ends, because of the inherent information leakage in LLMs. In this paper, we systematically quantify this leakage issue across four dimensions and release FinLake-Bench, a leakage-robust evaluation benchmark. Furthermore, to mitigate this issue, we introduce FactFin, a framework that applies counterfactual perturbations to compel LLM-based agents to learn causal drivers instead of memorized outcomes. FactFin integrates four core components: Strategy Code Generator, Retrieval-Augmented Generation, Monte Carlo Tree Search, and Counterfactual Simulator. Extensive experiments show that our method surpasses all baselines in out-of-sample generalization, delivering superior risk-adjusted performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07912v1">Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Automatic grading of subjective questions remains a significant challenge in examination assessment due to the diversity in question formats and the open-ended nature of student responses. Existing works primarily focus on a specific type of subjective question and lack the generality to support comprehensive exams that contain diverse question types. In this paper, we propose a unified Large Language Model (LLM)-enhanced auto-grading framework that provides human-like evaluation for all types of subjective questions across various domains. Our framework integrates four complementary modules to holistically evaluate student answers. In addition to a basic text matching module that provides a foundational assessment of content similarity, we leverage the powerful reasoning and generative capabilities of LLMs to: (1) compare key knowledge points extracted from both student and reference answers, (2) generate a pseudo-question from the student answer to assess its relevance to the original question, and (3) simulate human evaluation by identifying content-related and non-content strengths and weaknesses. Extensive experiments on both general-purpose and domain-specific datasets show that our framework consistently outperforms traditional and LLM-based baselines across multiple grading metrics. Moreover, the proposed system has been successfully deployed in real-world training and certification exams at a major e-commerce enterprise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06223v2">Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by EMNLP Industry Track 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been applied to reranking tasks in information retrieval, achieving strong performance. However, their high computational demands often hinder practical deployment. Existing studies evaluate the efficiency of LLM-based rerankers using proxy metrics such as latency, the number of forward passes, input tokens, and output tokens. However, these metrics depend on hardware and running-time choices (\eg parallel or not, batch size, etc), and often fail to account for model size, making it difficult to interpret and obscuring the evaluation of the efficiency-effectiveness tradeoff. To address this issue, we propose \ours\footnote{https://github.com/zhiyuanpeng/EER-FLOPs.} for LLM-based rerankers: RPP (ranking metrics per PetaFLOP), measuring how much ranking quality (e.g., NDCG or MRR) a method achieves per PetaFLOP, and QPP (queries per PetaFLOP), measuring how many queries can be processed per PetaFLOP. Accompanied by the new metrics, an interpretable FLOPs estimator is developed to estimate the FLOPs of an LLM-based reranker even without running any experiments. Based on the proposed metrics, we conduct comprehensive experiments to evaluate a wide range of LLM-based rerankers with different architectures, studying the efficiency-effectiveness trade-off and bringing this issue to the attention of the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16622v2">Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11885v5">Med-R$^2$: Crafting Trustworthy LLM Physicians via Retrieval and Reasoning of Evidence-Based Medicine</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. Despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 13.27\% improvement over vanilla RAG methods and even a 4.55\% enhancement compared to fine-tuning strategies, without incurring additional training costs. Furthermore, we find that our LLaMA3.1-70B + Med-R$^2$ surpasses frontier models, including GPT-4o, Claude3.5-Sonnet and DeepSeek-V3 by 1.05\%, 6.14\% and 1.91\%. Med-R$^2$ effectively enhances the capabilities of LLMs in the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v4">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07880v1">Do LLMs Really Need 10+ Thoughts for "Find the Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 30 pages, 41 figures, 10 tables. Preprint
    </div>
    <details class="paper-abstract">
      Models employing long chain-of-thought (CoT) reasoning have shown superior performance on complex reasoning tasks. Yet, this capability introduces a critical and often overlooked inefficiency -- overthinking -- models often engage in unnecessarily extensive reasoning even for simple queries, incurring significant computations without accuracy improvements. While prior work has explored solutions to mitigate overthinking, a fundamental gap remains in our understanding of its underlying causes. Most existing analyses are limited to superficial, profiling-based observations, failing to delve into LLMs' inner workings. This study introduces a systematic, fine-grained analyzer of LLMs' thought process to bridge the gap, TRACE. We first benchmark the overthinking issue, confirming that long-thinking models are five to twenty times slower on simple tasks with no substantial gains. We then use TRACE to first decompose the thought process into minimally complete sub-thoughts. Next, by inferring discourse relationships among sub-thoughts, we construct granular thought progression graphs and subsequently identify common thinking patterns for topically similar queries. Our analysis reveals two major patterns for open-weight thinking models -- Explorer and Late Landing. This finding provides evidence that over-verification and over-exploration are the primary drivers of overthinking in LLMs. Grounded in thought structures, we propose a utility-based definition of overthinking, which moves beyond length-based metrics. This revised definition offers a more insightful understanding of LLMs' thought progression, as well as practical guidelines for principled overthinking management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07877v1">Ready to Translate, Not to Represent? Bias and Performance Gaps in Multilingual LLMs Across Language Families and Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has redefined Machine Translation (MT), enabling context-aware and fluent translations across hundreds of languages and textual domains. Despite their remarkable capabilities, LLMs often exhibit uneven performance across language families and specialized domains. Moreover, recent evidence reveals that these models can encode and amplify different biases present in their training data, posing serious concerns for fairness, especially in low-resource languages. To address these gaps, we introduce Translation Tangles, a unified framework and dataset for evaluating the translation quality and fairness of open-source LLMs. Our approach benchmarks 24 bidirectional language pairs across multiple domains using different metrics. We further propose a hybrid bias detection pipeline that integrates rule-based heuristics, semantic similarity filtering, and LLM-based validation. We also introduce a high-quality, bias-annotated dataset based on human evaluations of 1,439 translation-reference pairs. The code and dataset are accessible on GitHub: https://github.com/faiyazabdullah/TranslationTangles
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05957v3">AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Code: https://github.com/HKUDS/AutoAgent
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce AutoAgent-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, AutoAgent comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, AutoAgent also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate AutoAgent's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, AutoAgent's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07024v2">Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02233v4">Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucination stemming from misaligned self-awareness, particularly when processing queries exceeding their knowledge boundaries. While existing mitigation strategies employ uncertainty estimation or query rejection mechanisms, they suffer from computational efficiency and sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate utilization of high-confidence outputs, whereas uncertain predictions trigger a slow refinement model for accuracy improvement. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. The framework establishes a scalable paradigm for deploying reliable LLMs in error-sensitive applications, effectively balancing accuracy and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13527v2">Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07825v1">An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The rise of Internet of Vehicles (IoV) technologies is transforming traffic management from isolated control to a collective, multi-vehicle process. At the heart of this shift is multi-vehicle dynamic navigation, which requires simultaneously routing large fleets under evolving traffic conditions. Existing path search algorithms and reinforcement learning methods struggle to scale to city-wide networks, often failing to capture the nonlinear, stochastic, and coupled dynamics of urban traffic. To address these challenges, we propose CityNav, a hierarchical, LLM-powered framework for large-scale multi-vehicle navigation. CityNav integrates a global traffic allocation agent, which coordinates strategic traffic flow distribution across regions, with local navigation agents that generate locally adaptive routes aligned with global directives. To enable effective cooperation, we introduce a cooperative reasoning optimization mechanism, in which agents are jointly trained with a dual-reward structure: individual rewards promote per-vehicle efficiency, while shared rewards encourage network-wide coordination and congestion reduction. Extensive experiments on four real-world road networks of varying scales (up to 1.6 million roads and 430,000 intersections) and traffic datasets demonstrate that CityNav consistently outperforms nine classical path search and RL-based baselines in city-scale travel efficiency and congestion mitigation. Our results highlight the potential of LLMs to enable scalable, adaptive, and cooperative city-wide traffic navigation, providing a foundation for intelligent, large-scale vehicle routing in complex urban environments. Our project is available at https://github.com/usail-hkust/CityNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19128v4">Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 On going work. Codes are released at https://github.com/zyc140345/FedAMoLE
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly powering web-based applications, whose effectiveness relies on fine-tuning with large-scale instruction data. However, such data often contains valuable or sensitive information that limits its public sharing among business organizations. Federated learning (FL) enables collaborative fine-tuning of LLMs without accessing raw data. Existing approaches to federated LLM fine-tuning usually adopt a uniform model architecture, making it challenging to fit highly heterogeneous client-side data in varying domains and tasks, e.g., hospitals and financial institutions conducting federated fine-tuning may require different LLM architectures due to the distinct nature of their domains and tasks. To address this, we propose FedAMoLE, a lightweight personalized FL framework that enables data-driven heterogeneous model architectures. It features a heterogeneous mixture of low-rank adaptation (LoRA) experts module to aggregate architecturally heterogeneous models and a reverse selection-based expert assignment strategy to tailor model architectures for each client based on data distributions. Experiments across seven scenarios demonstrate that FedAMoLE improves client-side performance by an average of 5.97% over existing approaches while maintaining practical memory, communication, and computation overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04768v3">DiMA: An LLM-Powered Ride-Hailing Assistant at DiDi</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 KDD 2025
    </div>
    <details class="paper-abstract">
      On-demand ride-hailing services like DiDi, Uber, and Lyft have transformed urban transportation, offering unmatched convenience and flexibility. In this paper, we introduce DiMA, an LLM-powered ride-hailing assistant deployed in DiDi Chuxing. Its goal is to provide seamless ride-hailing services and beyond through a natural and efficient conversational interface under dynamic and complex spatiotemporal urban contexts. To achieve this, we propose a spatiotemporal-aware order planning module that leverages external tools for precise spatiotemporal reasoning and progressive order planning. Additionally, we develop a cost-effective dialogue system that integrates multi-type dialog repliers with cost-aware LLM configurations to handle diverse conversation goals and trade-off response quality and latency. Furthermore, we introduce a continual fine-tuning scheme that utilizes real-world interactions and simulated dialogues to align the assistant's behavior with human preferred decision-making processes. Since its deployment in the DiDi application, DiMA has demonstrated exceptional performance, achieving 93% accuracy in order planning and 92% in response generation during real-world interactions. Offline experiments further validate DiMA capabilities, showing improvements of up to 70.23% in order planning and 321.27% in response generation compared to three state-of-the-art agent frameworks, while reducing latency by $0.72\times$ to $5.47\times$. These results establish DiMA as an effective, efficient, and intelligent mobile assistant for ride-hailing services. Our project is released at https://github.com/usail-hkust/DiMA and we also release the MCP service (https://mcp.didichuxing.com/api) to foster the ride-hailing research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07799v1">Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The efficiency of multi-agent systems driven by large language models (LLMs) largely hinges on their communication topology. However, designing an optimal topology is a non-trivial challenge, as it requires balancing competing objectives such as task performance, communication cost, and robustness. Existing frameworks often rely on static or hand-crafted topologies, which inherently fail to adapt to diverse task requirements, leading to either excessive token consumption for simple problems or performance bottlenecks for complex ones. To address this challenge, we introduce a novel generative framework called \textit{Guided Topology Diffusion (GTD)}. Inspired by conditional discrete graph diffusion models, GTD formulates topology synthesis as an iterative construction process. At each step, the generation is steered by a lightweight proxy model that predicts multi-objective rewards (e.g., accuracy, utility, cost), enabling real-time, gradient-free optimization towards task-adaptive topologies. This iterative, guided synthesis process distinguishes GTD from single-step generative frameworks, enabling it to better navigate complex design trade-offs. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods in LLM agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07796v1">HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The extraction and standardization of pharmacokinetic (PK) information from scientific literature remain significant challenges in computational pharmacology, which limits the reliability of data-driven models in drug development. Large language models (LLMs) have achieved remarkable progress in text understanding and reasoning, yet their adaptation to structured biomedical data, such as PK tables, remains constrained by heterogeneity, noise, and domain shift. To address these limitations, we propose HySim-LLM, a unified mathematical and computational framework that integrates embedding-weighted fine-tuning and manifold-aware denoising to enhance the robustness and interpretability of LLMs. We establish two theoretical results: (1) a similarity-weighted generalization bound that quantifies adaptation performance under embedding divergence, and (2) a manifold-based denoising guarantee that bounds loss contributions from noisy or off-manifold samples. These theorems provide a principled foundation for fine-tuning LLMs in structured biomedical settings. The framework offers a mathematically grounded pathway toward reliable and interpretable LLM adaptation for biomedical and data-intensive scientific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07777v1">Drift No More? Context Equilibria in Multi-Turn LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at single-turn tasks such as instruction following and summarization, yet real-world deployments require sustained multi-turn interactions where user goals and conversational context persist and evolve. A recurring challenge in this setting is context drift: the gradual divergence of a model's outputs from goal-consistent behavior across turns. Unlike single-turn errors, drift unfolds temporally and is poorly captured by static evaluation metrics. In this work, we present a study of context drift in multi-turn interactions and propose a simple dynamical framework to interpret its behavior. We formalize drift as the turn-wise KL divergence between the token-level predictive distributions of the test model and a goal-consistent reference model, and propose a recurrence model that interprets its evolution as a bounded stochastic process with restoring forces and controllable interventions. We instantiate this framework in both synthetic long-horizon rewriting tasks and realistic user-agent simulations such as in $\tau$-Bench, measuring drift for several open-weight LLMs that are used as user simulators. Our experiments consistently reveal stable, noise-limited equilibria rather than runaway degradation, and demonstrate that simple reminder interventions reliably reduce divergence in line with theoretical predictions. Together, these results suggest that multi-turn drift can be understood as a controllable equilibrium phenomenon rather than as inevitable decay, providing a foundation for studying and mitigating context drift in extended interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07775v1">The Unintended Trade-off of AI Alignment:Balancing Hallucination Mitigation and Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Hallucination in large language models (LLMs) has been widely studied in recent years, with progress in both detection and mitigation aimed at improving truthfulness. Yet, a critical side effect remains largely overlooked: enhancing truthfulness can negatively impact safety alignment. In this paper, we investigate this trade-off and show that increasing factual accuracy often comes at the cost of weakened refusal behavior. Our analysis reveals that this arises from overlapping components in the model that simultaneously encode hallucination and refusal information, leading alignment methods to suppress factual knowledge unintentionally. We further examine how fine-tuning on benign datasets, even when curated for safety, can degrade alignment for the same reason. To address this, we propose a method that disentangles refusal-related features from hallucination features using sparse autoencoders, and preserves refusal behavior during fine-tuning through subspace orthogonalization. This approach prevents hallucinations from increasing while maintaining safety alignment.We evaluate our method on commonsense reasoning tasks and harmful benchmarks (AdvBench and StrongReject). Results demonstrate that our approach preserves refusal behavior and task utility, mitigating the trade-off between truthfulness and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07772v1">An approach for systematic decomposition of complex llm tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from reliability issues on complex tasks, as existing decomposition methods are heuristic and rely on agent or manual decomposition. This work introduces a novel, systematic decomposition framework that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leveraging formal complexity measures to guide decomposition. On combinatorial (SATBench) and LLM database querying tasks (Spider), we find that by decomposing the tasks following the measure of complexity, agent can perform considerably better (10-40 percentage point).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07768v1">ToolLibGen: Scalable Automatic Tool Creation and Aggregation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) equipped with external tools have demonstrated enhanced performance on complex reasoning tasks. The widespread adoption of this tool-augmented reasoning is hindered by the scarcity of domain-specific tools. For instance, in domains such as physics question answering, suitable and specialized tools are often missing. Recent work has explored automating tool creation by extracting reusable functions from Chain-of-Thought (CoT) reasoning traces; however, these approaches face a critical scalability bottleneck. As the number of generated tools grows, storing them in an unstructured collection leads to significant retrieval challenges, including an expanding search space and ambiguity between function-related tools. To address this, we propose a systematic approach to automatically refactor an unstructured collection of tools into a structured tool library. Our system first generates discrete, task-specific tools and clusters them into semantically coherent topics. Within each cluster, we introduce a multi-agent framework to consolidate scattered functionalities: a code agent refactors code to extract shared logic and creates versatile, aggregated tools, while a reviewing agent ensures that these aggregated tools maintain the complete functional capabilities of the original set. This process transforms numerous question-specific tools into a smaller set of powerful, aggregated tools without loss of functionality. Experimental results demonstrate that our approach significantly improves tool retrieval accuracy and overall reasoning performance across multiple reasoning tasks. Furthermore, our method shows enhanced scalability compared with baselines as the number of question-specific increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07762v1">From Noisy to Native: LLM-driven Graph Restoration for Test-Time Graph Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Graph domain adaptation (GDA) has achieved great attention due to its effectiveness in addressing the domain shift between train and test data. A significant bottleneck in existing graph domain adaptation methods is their reliance on source-domain data, which is often unavailable due to privacy or security concerns. This limitation has driven the development of Test-Time Graph Domain Adaptation (TT-GDA), which aims to transfer knowledge without accessing the source examples. Inspired by the generative power of large language models (LLMs), we introduce a novel framework that reframes TT-GDA as a generative graph restoration problem, "restoring the target graph to its pristine, source-domain-like state". There are two key challenges: (1) We need to construct a reasonable graph restoration process and design an effective encoding scheme that an LLM can understand, bridging the modality gap. (2) We need to devise a mechanism to ensure the restored graph acquires the intrinsic features of the source domain, even without access to the source data. To ensure the effectiveness of graph restoration, we propose GRAIL, that restores the target graph into a state that is well-aligned with the source domain. Specifically, we first compress the node representations into compact latent features and then use a graph diffusion process to model the graph restoration process. Then a quantization module encodes the restored features into discrete tokens. Building on this, an LLM is fine-tuned as a generative restorer to transform a "noisy" target graph into a "native" one. To further improve restoration quality, we introduce a reinforcement learning process guided by specialized alignment and confidence rewards. Extensive experiments demonstrate the effectiveness of our approach across various datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v5">Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by the EMNLP25 main conference
    </div>
    <details class="paper-abstract">
      This paper explores the use of large language models (LLMs) for annotating document utility in training retrieval and retrieval-augmented generation (RAG) systems, aiming to reduce dependence on costly human annotations. We address the gap between retrieval relevance and generative utility by employing LLMs to annotate document utility. To effectively utilize multiple positive samples per query, we introduce a novel loss that maximizes their summed marginal likelihood. Using the Qwen-2.5-32B model, we annotate utility on the MS MARCO dataset and conduct retrieval experiments on MS MARCO and BEIR, as well as RAG experiments on MS MARCO QA, NQ, and HotpotQA. Our results show that LLM-generated annotations enhance out-of-domain retrieval performance and improve RAG outcomes compared to models trained solely on human annotations or downstream QA metrics. Furthermore, combining LLM annotations with just 20% of human labels achieves performance comparable to using full human annotations. Our study offers a comprehensive approach to utilizing LLM annotations for initializing QA systems on new corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01493v2">Sherkala-Chat: Building a State-of-the-Art LLM for Kazakh in a Moderately Resourced Setting</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Llama-3.1-Sherkala-8B-Chat, or Sherkala-Chat (8B) for short, is a state-of-the-art instruction-tuned open generative large language model (LLM) designed for Kazakh. Sherkala-Chat (8B) aims to enhance the inclusivity of LLM advancements for Kazakh speakers. Adapted from the LLaMA-3.1-8B model, Sherkala-Chat (8B) is trained on 45.3B tokens across Kazakh, English, Russian, and Turkish. With 8 billion parameters, it demonstrates strong knowledge and reasoning abilities in Kazakh, significantly outper-forming existing open Kazakh and multilingual models of similar scale while achieving competitive performance in English. To ensure effective and responsible alignment, we leverage translated instruction datasets, a Kazakhstan-specific instruction dataset that is automatically constructed and manually verified, and Kazakh-specific safety data. We release Sherkala-Chat (8B) as an open-weight model, along with a detailed description of its training, alignment, and evaluation, to support research and real-world applications for Kazakh speakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07743v1">OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Reward modeling lies at the core of reinforcement learning from human feedback (RLHF), yet most existing reward models rely on scalar or pairwise judgments that fail to capture the multifaceted nature of human preferences. Recent studies have explored rubrics-as-rewards (RaR) that uses structured natural language criteria that capture multiple dimensions of response quality. However, producing rubrics that are both reliable and scalable remains a key challenge. In this work, we introduce OpenRubrics, a diverse, large-scale collection of (prompt, rubric) pairs for training rubric-generation and rubric-based reward models. To elicit discriminative and comprehensive evaluation signals, we introduce Contrastive Rubric Generation (CRG), which derives both hard rules (explicit constraints) and principles (implicit qualities) by contrasting preferred and rejected responses. We further improve reliability by enforcing preference-label consistency via rejection sampling to remove noisy rubrics. Across multiple reward-modeling benchmarks, our rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines by 6.8%. These gains transfer to policy models on instruction-following and biomedical benchmarks. Our results show that rubrics provide scalable alignment signals that narrow the gap between costly human evaluation and automated reward modeling, enabling a new principle-driven paradigm for LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v5">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07737v1">ToolExpander: Extending the Frontiers of Tool-Using Reinforcement Learning to Weak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) with Group Relative Policy Optimization (GRPO) encounters a significant challenge: models often fail to produce accurate responses, particularly in small-scale architectures. This limitation not only diminishes performance improvements and undermines the potential of GRPO but also frequently leads to mid-training collapse, adversely affecting stability and final efficacy. To address these issues, we propose ToolExpander, a novel framework that advances tool-oriented reinforcement learning for resource-constrained LLMs through two key innovations:(1) Dynamic Multi-Round Hard Sampling, which dynamically substitutes challenging samples(those without correct outputs over 10 rollouts) with high-quality few-shot demonstrations during training, coupled with an exponential learning rate decay strategy to mitigate oscillations;(2) Self-Exemplifying Thinking, an enhanced GRPO framework that eliminates KL divergence and incorporates adjusted clipping coefficients, encouraging models to autonomously generate and analyze few-shot examples via a minimal additional reward (0.01).Experimental results demonstrate that ToolExpander significantly enhances tool-using capabilities in LLMs, especially in weaker small-scale models, improving both training stability and overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25034v2">MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on real-world USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07733v1">SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07731v1">oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Main Text: 8 pages, In total: 37 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Organic reaction mechanisms are the stepwise elementary reactions by which reactants form intermediates and products, and are fundamental to understanding chemical reactivity and designing new molecules and reactions. Although large language models (LLMs) have shown promise in understanding chemical tasks such as synthesis design, it is unclear to what extent this reflects genuine chemical reasoning capabilities, i.e., the ability to generate valid intermediates, maintain chemical consistency, and follow logically coherent multi-step pathways. We address this by introducing oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning in organic chemistry. It comprises over 10,000 annotated mechanistic steps with intermediates, type labels, and difficulty ratings. Furthermore, to evaluate LLM capability more precisely and enable fine-grained scoring, we propose oMeS, a dynamic evaluation framework that combines step-level logic and chemical similarity. We analyze the performance of state-of-the-art LLMs, and our results show that although current models display promising chemical intuition, they struggle with correct and consistent multi-step reasoning. Notably, we find that using prompting strategy and fine-tuning a specialist model on our proposed dataset increases performance by 50% over the leading closed-source model. We hope that oMeBench will serve as a rigorous foundation for advancing AI systems toward genuine chemical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07697v1">Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04596v2">LLM Applications: Current Paradigms and the Next Frontier</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has given rise to four major application paradigms: LLM app stores, LLM agents, self-hosted LLM services, and LLM-powered devices. Each has its advantages but also shares common challenges. LLM app stores lower the barrier to development but lead to platform lock-in; LLM agents provide autonomy but lack a unified communication mechanism; self-hosted LLM services enhance control but increase deployment complexity; and LLM-powered devices improve privacy and real-time performance but are limited by hardware. This paper reviews and analyzes these paradigms, covering architecture design, application ecosystem, research progress, as well as the challenges and open problems they face. Based on this, we outline the next frontier of LLM applications, characterizing them through three interconnected layers: infrastructure, protocol, and application. We describe their responsibilities and roles of each layer and demonstrate how to mitigate existing fragmentation limitations and improve security and scalability. Finally, we discuss key future challenges, identify opportunities such as protocol-driven cross-platform collaboration and device integration, and propose a research roadmap for openness, security, and sustainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18952v2">LLMs on a Budget? Say HOLA</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted at EMNLP 2025 (Industry Track)
    </div>
    <details class="paper-abstract">
      Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06674v2">Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables
    </div>
    <details class="paper-abstract">
      We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14053v2">MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05393v2">HiVeGen -- Hierarchical LLM-based Verilog Generation for Scalable Chip Design</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) recently demonstrating impressive proficiency in code generation, it is promising to extend their abilities to Hardware Description Language (HDL). However, LLMs tend to generate single HDL code blocks rather than hierarchical structures for hardware designs, leading to hallucinations, particularly in complex designs like Domain-Specific Accelerators (DSAs). To address this, we propose HiVeGen, a hierarchical LLM-based Verilog generation framework that decomposes generation tasks into LLM-manageable hierarchical submodules. HiVeGen further harnesses the advantages of such hierarchical structures by integrating automatic Design Space Exploration (DSE) into hierarchy-aware prompt generation, introducing weight-based retrieval to enhance code reuse, and enabling real-time human-computer interaction to lower error-correction cost, significantly improving the quality of generated designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07651v1">OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with extended context windows enable powerful downstream applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13933v2">Preprint: Poster: Did I Just Browse A Website Written by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 ACM Internet Measurement Conference 2025 Poster & ACM IMC 2025 Student Workshop. 2 pages. 3 figures
    </div>
    <details class="paper-abstract">
      Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are inaccurate on web content, because web content has low positive rates, complex markup, and diverse genres, instead of clean, prose-like benchmark data SoTA detectors are optimized for. We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages to boost accuracies. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08850v1">Repository-Aware File Path Retrieval via Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Modern codebases make it hard for developers and AI coding assistants to find the right source files when answering questions like "How does this feature work?" or "Where was the bug introduced?" Traditional code search (keyword or IR based) often misses semantic context and cross file links, while large language models (LLMs) understand natural language but lack repository specific detail. We present a method for file path retrieval that fine tunes a strong LLM (Qwen3-8B) with QLoRA and Unsloth optimizations to predict relevant file paths directly from a natural language query. To build training data, we introduce six code aware strategies that use abstract syntax tree (AST) structure and repository content to generate realistic question-answer pairs, where answers are sets of file paths. The strategies range from single file prompts to hierarchical repository summaries, providing broad coverage. We fine tune on Python projects including Flask, Click, Jinja, FastAPI, and PyTorch, and obtain high retrieval accuracy: up to 91\% exact match and 93\% recall on held out queries, clearly beating single strategy training. On a large codebase like PyTorch (about 4,000 Python files), the model reaches 59\% recall, showing scalability. We analyze how multi level code signals help the LLM reason over cross file context and discuss dataset design, limits (for example, context length in very large repos), and future integration of retrieval with LLM based code intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06094v2">ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Project page: https://conlangcrafter.github.io
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages - phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We evaluate ConlangCrafter on metrics measuring consistency and typological diversity, demonstrating its ability to produce coherent and varied conlangs without human linguistic expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v3">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06250v2">Scalable multilingual PII annotation for responsible AI in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v3">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08813v1">The Model's Language Matters: A Comparative Privacy Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across multilingual applications that handle sensitive data, yet their scale and linguistic variability introduce major privacy risks. Mostly evaluated for English, this paper investigates how language structure affects privacy leakage in LLMs trained on English, Spanish, French, and Italian medical corpora. We quantify six linguistic indicators and evaluate three attack vectors: extraction, counterfactual memorization, and membership inference. Results show that privacy vulnerability scales with linguistic redundancy and tokenization granularity: Italian exhibits the strongest leakage, while English shows higher membership separability. In contrast, French and Spanish display greater resilience due to higher morphological complexity. Overall, our findings provide the first quantitative evidence that language matters in privacy leakage, underscoring the need for language-aware privacy-preserving mechanisms in LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08779v1">Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to LM4Plan Workshop @ ICAPS 2025 (withdrawn before presentation due to lack of travel funding)
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08776v1">Measuring Moral LLM Responses in Multilingual Capacities</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 10 pages, 5 figures; referenced articles: arXiv:2303.08774, arXiv:2303.12528, arXiv:2308.14132, arXiv:2505.12201, arXiv:2406.04428, arXiv:2407.02273, arXiv:2404.01268, arXiv:2502.09747, arXiv:2507.13474, arXiv:2505.21479, arXiv:2306.05685
    </div>
    <details class="paper-abstract">
      With LLM usage becoming widespread across countries, languages, and humanity more broadly, the need to understand and guardrail their multilingual responses increases. Large-scale datasets for testing and benchmarking have been created to evaluate and facilitate LLM responses across multiple dimensions. In this study, we evaluate the responses of frontier and leading open-source models in five dimensions across low and high-resource languages to measure LLM accuracy and consistency across multilingual contexts. We evaluate the responses using a five-point grading rubric and a judge LLM. Our study shows that GPT-5 performed the best on average in each category, while other models displayed more inconsistency across language and category. Most notably, in the Consent & Autonomy and Harm Prevention & Safety categories, GPT scored the highest with averages of 3.56 and 4.73, while Gemini 2.5 Pro scored the lowest with averages of 1.39 and 1.98, respectively. These findings emphasize the need for further testing on how linguistic shifts impact LLM responses across various categories and improvement in these areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08755v1">Robust Heuristic Algorithm Design with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08741v1">Coordinates from Context: Using LLMs to Ground Complex Location References</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Under review at ARR
    </div>
    <details class="paper-abstract">
      Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06594v2">Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08710v1">Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08705v1">ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08664v1">Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middleware</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      LLM-based RTL generation is an interesting research direction, as it holds the potential to liberate the least automated stage in the current chip design. However, due to the substantial semantic gap between high-level specifications and RTL, coupled with limited training data, existing models struggle with generation accuracy. Drawing on human experience, design with verification helps improving accuracy. However, as the RTL testbench data are even more scarce, it is not friendly for LLMs. Although LLMs excel at higher-level languages like Python/C, they have a huge semantic gap from RTL. When implementing the same functionality, Python/C code and hardware code differ significantly in the spatiotemporal granularity, requiring the LLM not only to consider high-level functional semantics but also to ensure the low-level details align with the circuit code. It is not an easy task. In this paper, we propose a function abstracted verifiable middleware (Faver) that streamlines RTL verification in LLM-based workflows. By mixing LLM-friendly code structures with a rule-based template, Faver decouples the details of circuit verification, allowing the LLM to focus on the functionality itself. In our experiments on the SFT model and open-source models, Faver improved the model's generation accuracy by up to 14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08663v1">A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08640v1">Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v3">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 14 pages, 16 figures. For source code, see https://github.com/microsoft/tokenweave. In version 2, Figure 6 shows All Reduce bandwidth, not Reduce Scatter. The Multimem Reduce Scatter bandwidth formula differs slightly from the Ring-based version
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLink. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Furthermore, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The communication of one subset is then overlapped with the computation of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce--RMSNorm kernel that carefully leverages Multimem instruction support available on Hopper and Blackwell NVIDIA GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory-bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 1.29x speedup in latency and 1.26x higher throughput across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07083v1">All Claims Are Equal, but Some Claims Are More Equal Than Others: Importance-Sensitive Factuality Evaluation of LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Existing methods for evaluating the factuality of large language model (LLM) responses treat all claims as equally important. This results in misleading evaluations when vital information is missing or incorrect as it receives the same weight as peripheral details, raising the question: how can we reliably detect such differences when there are errors in key information? Current approaches that measure factuality tend to be insensitive to omitted or false key information. To investigate this lack of sensitivity, we construct VITALERRORS, a benchmark of 6,733 queries with minimally altered LLM responses designed to omit or falsify key information. Using this dataset, we demonstrate the insensitivities of existing evaluation metrics to key information errors. To address this gap, we introduce VITAL, a set of metrics that provide greater sensitivity in measuring the factuality of responses by incorporating the relevance and importance of claims with respect to the query. Our analysis demonstrates that VITAL metrics more reliably detect errors in key information than previous methods. Our dataset, metrics, and analysis provide a foundation for more accurate and robust assessment of LLM factuality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07073v1">VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, \VRPAgent is the first LLM-based paradigm to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04023v2">Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance on complex mathematical benchmarks yet sometimes fail on basic math reasoning while generating unnecessarily verbose responses. In this paper, we present a systematic benchmark and comprehensive empirical study to evaluate the efficiency of reasoning in LLMs, focusing on the fundamental tradeoff between accuracy and overthinking. First, we formalize the accuracy-verbosity tradeoff. Second, we introduce the Overthinking Score, a harmonic-mean metric combining accuracy and token-efficiency for holistic model evaluation. Third, we establish an evaluation protocol with dynamically-generated data across 14 basic math tasks. Fourth, we conduct a large-scale empirical study evaluating 53 LLMs, including reasoning and quantized variants across different reasoning budgets. Our findings reveal: 1) model performance on complex benchmarks does not translate directly to basic math reasoning; 2) reasoning models generate ~18 more tokens while sometimes achieving lower accuracy and exhibit catastrophic collapse when token is constrained, dropping by ~28; 3) the accuracy-verbosity relationship is non-monotonic with extended reasoning budgets yielding diminishing returns (GPT-5/o-series models show zero accuracy gain from low -> medium -> high reasoning effort). Our findings challenge the assumption that longer reasoning in LLMs necessarily improves mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19366v3">Grounding the Ungrounded: A Spectral-Graph Framework for Quantifying Hallucinations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 29 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Hallucinations in LLMs--especially in multimodal settings--undermine reliability. We present a rigorous, information-geometric framework in diffusion dynamics that quantifies hallucination in MLLMs: model outputs are embedded spectrally on multimodal graph Laplacians, and gaps to a truth manifold define a semantic-distortion metric. We derive Courant--Fischer bounds on a temperature-dependent hallucination energy and use RKHS eigenmodes to obtain modality-aware, interpretable measures that track evolution over prompts and time. This reframes hallucination as measurable and bounded, providing a principled basis for evaluation and mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07024v1">Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14161v2">MIST: Towards Multi-dimensional Implicit BiaS Evaluation of LLMs via Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06997v1">The Limits of Goal-Setting Theory in LLM-Driven Assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at T4E 2025 for poster
    </div>
    <details class="paper-abstract">
      Many users interact with AI tools like ChatGPT using a mental model that treats the system as human-like, which we call Model H. According to goal-setting theory, increased specificity in goals should reduce performance variance. If Model H holds, then prompting a chatbot with more detailed instructions should lead to more consistent evaluation behavior. This paper tests that assumption through a controlled experiment in which ChatGPT evaluated 29 student submissions using four prompts with increasing specificity. We measured consistency using intra-rater reliability (Cohen's Kappa) across repeated runs. Contrary to expectations, performance did not improve consistently with increased prompt specificity, and performance variance remained largely unchanged. These findings challenge the assumption that LLMs behave like human evaluators and highlight the need for greater robustness and improved input integration in future model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14146v4">MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06994v1">RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06974v1">Probing Social Identity Bias in Chinese LLMs with Gendered Pronouns and Social Groups</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in user-facing applications, raising concerns about their potential to reflect and amplify social biases. We investigate social identity framing in Chinese LLMs using Mandarin-specific prompts across ten representative Chinese LLMs, evaluating responses to ingroup ("We") and outgroup ("They") framings, and extending the setting to 240 social groups salient in the Chinese context. To complement controlled experiments, we further analyze Chinese-language conversations from a corpus of real interactions between users and chatbots. Across models, we observe systematic ingroup-positive and outgroup-negative tendencies, which are not confined to synthetic prompts but also appear in naturalistic dialogue, indicating that bias dynamics might strengthen in real interactions. Our study provides a language-aware evaluation framework for Chinese LLMs, demonstrating that social identity biases documented in English generalize cross-linguistically and intensify in user-facing contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06953v1">Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15828v2">Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Embodied agents need to plan and act reliably in real and complex 3D environments. Classical planning (e.g., PDDL) offers structure and guarantees, but in practice it fails under noisy perception and incorrect predicate grounding. On the other hand, Large Language Models (LLMs)-based planners leverage commonsense reasoning, yet frequently propose actions that are unfeasible or unsafe. Following recent works that combine the two approaches, we introduce ContextMatters, a framework that fuses LLMs and classical planning to perform hierarchical goal relaxation: the LLM helps ground symbols to the scene and, when the target is unreachable, it proposes functionally equivalent goals that progressively relax constraints, adapting the goal to the context of the agent's environment. Operating on 3D Scene Graphs, this mechanism turns many nominally unfeasible tasks into tractable plans and enables context-aware partial achievement when full completion is not achievable. Our experimental results show a +52.45% Success Rate improvement over state-of-the-art LLMs+PDDL baseline, demonstrating the effectiveness of our approach. Moreover, we validate the execution of ContextMatter in a real world scenario by deploying it on a TIAGo robot. Code, dataset, and supplementary materials are available to the community at https://lab-rococo-sapienza.github.io/context-matters/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06878v1">TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Iterative refinement has been a promising paradigm to enable large language models (LLMs) to resolve difficult reasoning and problem-solving tasks. One of the key challenges, however, is how to effectively search through the enormous search space of possible refinements. Existing methods typically fall back on predefined heuristics, which are troubled by the exploration-exploitation dilemma and cannot adapt based on past refinement outcomes. We introduce Tree-Guided Policy Refinement (TGPR), a novel framework that combines GRPO with a Thompson-Sampling-based tree search. TGPR explores both failed and successful refinement paths actively, with denser training trajectories and more adaptive policies. On HumanEval, MBPP, and APPS benchmarks, our method achieves up to +4.2 percentage points absolute improvement in pass@1 (on MBPP) and up to +12.51 percentage points absolute improvement in pass@10 (on APPS) compared to a competitive GRPO baseline. Apart from debugging code, TGPR focuses on a principled approach to combining learned policies with structured search methods, offering a general framework for enhancing iterative refinement and stateful reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06866v1">Unlocking Latent Discourse Translation in LLMs Through Quality-Aware Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as strong contenders in machine translation.Yet, they still struggle to adequately handle discourse phenomena, such as pronoun resolution and lexical cohesion at the document level. In this study, we thoroughly investigate the discourse phenomena performance of LLMs in context-aware translation. We demonstrate that discourse knowledge is encoded within LLMs and propose the use of quality-aware decoding (QAD) to effectively extract this knowledge, showcasing its superiority over other decoding approaches through comprehensive analysis. Furthermore, we illustrate that QAD enhances the semantic richness of translations and aligns them more closely with human preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v3">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We released our code and dataset at https://github.com/penfever/judgment-to-noise
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06843v1">SID: Multi-LLM Debate Driven by Self Signals</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{https://github.com/xuhang2019/SID}{\texttt{https://github.com/xuhang2019/SID}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06096v2">The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11557v2">AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Corporate LLMs are gaining traction for efficient knowledge dissemination and management within organizations. However, as current LLMs are vulnerable to leaking sensitive information, it has proven difficult to apply them in settings where strict access control is necessary. To this end, we design AC-LoRA, an end-to-end system for access control-aware corporate LLM chatbots that maintains a strong information isolation guarantee. AC-LoRA maintains separate LoRA adapters for permissioned datasets, along with the document embedding they are finetuned on. AC-LoRA retrieves a precise set of LoRA adapters based on the similarity score with the user query and their permission. This similarity score is later used to merge the responses if more than one LoRA is retrieved, without requiring any additional training for LoRA routing. We provide an end-to-end prototype of AC-LoRA, evaluate it on two datasets, and show that AC-LoRA matches or even exceeds the performance of state-of-the-art LoRA mixing techniques while providing strong isolation guarantees. Furthermore, we show that AC-LoRA design can be directly applied to different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18169v5">KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Serving LLMs with a cluster of GPUs is common nowadays, where the serving system must meet strict latency SLOs required by applications. However, the stateful nature of LLM serving requires maintaining huge states (i.e., KVCache) in limited GPU memory. Under spikes in real-world workloads, GPU memory can be easily throttled, leading to orders of magnitude higher response latency due to queuing introduced by waiting for KVCache to be reclaimed. Prior KVCache-centric approaches handle load throttling by dropping, migrating, or swapping KVCache. These methods fail to release sufficient memory quickly with requests still queued. This paper proposes the first parameter-centric approach to handling throttling by selectively dropping replicated parameters to instantly free memory for requests, based on an unnoticed observation that model parameters are commonly replicated across GPUs for serving LLMs. With additional memory, all requests can be served with a larger batch without queuing. To make the parameter-centric approach correct and efficient, we cooperatively execute requests on GPUs with a complete copy of parameters using pipeline parallelism, and derive an appropriate drop plan without unnecessary cooperation. We also design techniques to minimize the performance overhead due to pipeline parallelism with the execution patterns of requests under drop. Evaluations show that {\sys} reduces the tail TTFT of requests under throttling by up to 72.2 times compared to the state-of-the-art systems including Llumnix, vLLM and InferCept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12845v4">ExLLM: Experience-Enhanced LLM Optimization for Molecular Design and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, under review
    </div>
    <details class="paper-abstract">
      Molecular design involves an enormous and irregular search space, where traditional optimizers such as Bayesian optimization, genetic algorithms, and generative models struggle to leverage expert knowledge or handle complex feedback. Recently, LLMs have been used as optimizers, achieving promising results on benchmarks such as PMO. However, existing approaches rely only on prompting or extra training, without mechanisms to handle complex feedback or maintain scalable memory. In particular, the common practice of appending or summarizing experiences at every query leads to redundancy, degraded exploration, and ultimately poor final outcomes under large-scale iterative search. We introduce ExLLM (Experience-Enhanced LLM optimization), an LLM-as-optimizer framework with three components: (1) a compact, evolving experience snippet tailored to large discrete spaces that distills non-redundant cues and improves convergence at low cost; (2) a simple yet effective k-offspring scheme that widens exploration per call and reduces orchestration cost; and (3) a lightweight feedback adapter that normalizes objectives for selection while formatting constraints and expert hints for iteration. ExLLM sets new state-of-the-art results on PMO and generalizes strongly in our setup, it sets records on circle packing and stellarator design, and yields consistent gains across additional domains requiring only a task-description template and evaluation functions to transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06780v1">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06774v1">Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Neuro-symbolic NLP methods aim to leverage the complementary strengths of large language models and formal logical solvers. However, current approaches are mostly static in nature, i.e., the integration of a target solver is predetermined at design time, hindering the ability to employ diverse formal inference strategies. To address this, we introduce an adaptive, multi-paradigm, neuro-symbolic inference framework that: (1) automatically identifies formal reasoning strategies from problems expressed in natural language; and (2) dynamically selects and applies specialized formal logical solvers via autoformalization interfaces. Extensive experiments on individual and multi-paradigm reasoning tasks support the following conclusions: LLMs are effective at predicting the necessary formal reasoning strategies with an accuracy above 90 percent. This enables flexible integration with formal logical solvers, resulting in our framework outperforming competing baselines by 27 percent and 6 percent compared to GPT-4o and DeepSeek-V3.1, respectively. Moreover, adaptive reasoning can even positively impact pure LLM methods, yielding gains of 10, 5, and 6 percent on zero-shot, CoT, and symbolic CoT settings with GPT-4o. Finally, although smaller models struggle with adaptive neuro-symbolic reasoning, post-training offers a viable path to improvement. Overall, this work establishes the foundations for adaptive LLM-symbolic reasoning, offering a path forward for unifying material and formal inferences on heterogeneous reasoning challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06750v1">Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Reasoning Models (LRMs) excel in structured tasks by emulating deliberate human reasoning but often suffer from overthinking, degrading performance and wasting resources. One possible baseline is to deploy both LLM and LRM, then route input by predicting whether it requires reasoning and may cause overthinking. However, deploying multiple models can be costly or impractical. We propose a superposed deployment strategy with a lightweight, training-free regulation to optimize inference by switching one model on and off. Instead of routing, we selectively unlearn from LRM at inference, scaling down computation while preserving reasoning. By analyzing the cumulative energy of singular values, we identify optimal low-rank projections to adjust reasoning just right.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06747v1">TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06743v1">Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025
    </div>
    <details class="paper-abstract">
      Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15098v2">TextMine: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMine: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction in the HMA domain. TextMine structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we created the dataset in collaboration with Cambodian Mine Action Center (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06732v1">Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: https://github.com/glad-lab/RAF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06730v1">PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Current evaluations of sentence embedding models typically rely on static test beds such as the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported performance and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in semantic textual similarity gold ratings, we show that LLMs generate token-diverse but semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs and we extended our experiments to 3 multilingual datasets covering 10 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06727v1">Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06718v1">LLM Company Policies and Policy Implications in Software Organizations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at IEEE Software Special Issue on AIware in the Foundation Models Era
    </div>
    <details class="paper-abstract">
      The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04152v2">Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06708v1">AISysRev -- LLM-based Tool for Title-abstract Screening</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human reviewers.We conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: https://www.youtube.com/watch?v=jVbEj4Y4tQI Tool: https://github.com/EvoTestOps/AISysRev
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25033v2">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06695v1">Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06674v1">Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables
    </div>
    <details class="paper-abstract">
      We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05154v2">Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v3">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06663v1">Automated Discovery of Test Oracles for Database Management Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Since 2020, automated testing for Database Management Systems (DBMSs) has flourished, uncovering hundreds of bugs in widely-used systems. A cornerstone of these techniques is test oracle, which typically implements a mechanism to generate equivalent query pairs, thereby identifying bugs by checking the consistency between their results. However, while applying these oracles can be automated, their design remains a fundamentally manual endeavor. This paper explores the use of large language models (LLMs) to automate the discovery and instantiation of test oracles, addressing a long-standing bottleneck towards fully automated DBMS testing. Although LLMs demonstrate impressive creativity, they are prone to hallucinations that can produce numerous false positive bug reports. Furthermore, their significant monetary cost and latency mean that LLM invocations should be limited to ensure that bug detection is efficient and economical. To this end, we introduce Argus, a novel framework built upon the core concept of the Constrained Abstract Query - a SQL skeleton containing placeholders and their associated instantiation conditions (e.g., requiring a placeholder to be filled by a boolean column). Argus uses LLMs to generate pairs of these skeletons that are asserted to be semantically equivalent. This equivalence is then formally proven using a SQL equivalence solver to ensure soundness. Finally, the placeholders within the verified skeletons are instantiated with concrete, reusable SQL snippets that are also synthesized by LLMs to efficiently produce complex test cases. We implemented Argus and evaluated it on five extensively tested DBMSs, discovering 40 previously unknown bugs, 35 of which are logic bugs, with 36 confirmed and 26 already fixed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06658v1">Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at SIGIR-AP 2025
    </div>
    <details class="paper-abstract">
      Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable. We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06657v1">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 RecSys 2025 Industry Track
    </div>
    <details class="paper-abstract">
      This paper presents a case study on deploying Large Language Models (LLMs) as an advanced "annotation" mechanism to achieve nuanced content understanding (e.g., discerning content "vibe") at scale within a large-scale industrial short-form video recommendation system. Traditional machine learning classifiers for content understanding face protracted development cycles and a lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses these by significantly shortening development times and enabling the annotation of subtle attributes. This work details an end-to-end workflow encompassing: (1) iterative definition and robust evaluation of target attributes, refined by offline metrics and online A/B testing; (2) scalable offline bulk annotation of video corpora using LLMs with multimodal features, optimized inference, and knowledge distillation for broad application; and (3) integration of these rich annotations into the online recommendation serving system, for example, through personalized restrict retrieval. Experimental results demonstrate the efficacy of this approach, with LLMs outperforming human raters in offline annotation quality for nuanced attributes and yielding significant improvements of user participation and satisfied consumption in online A/B tests. The study provides insights into designing and scaling production-level LLM pipelines for rich content evaluation, highlighting the adaptability and benefits of LLM-generated nuanced understanding for enhancing content discovery, user satisfaction, and the overall effectiveness of modern recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02053v2">ProCut: LLM Prompt Compression via Attribution Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts.
    </details>
</div>
