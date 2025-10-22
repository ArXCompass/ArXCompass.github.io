# llm - 2025_10

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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06605v1">Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06594v1">Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v3">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08833v2">An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 34 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 51.5% on the originals but drops by 4.7 percentage points on surface-renaming variants, and by 12.9 percentage points on parametric variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06565v1">Auto-Stega: An Agent-Driven System for Lifelong Strategy Evolution in LLM-Based Text Steganography</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      With the rapid progress of LLMs, high quality generative text has become widely available as a cover for text steganography. However, prevailing methods rely on hand-crafted or pre-specified strategies and struggle to balance efficiency, imperceptibility, and security, particularly at high embedding rates. Accordingly, we propose Auto-Stega, an agent-driven self-evolving framework that is the first to realize self-evolving steganographic strategies by automatically discovering, composing, and adapting strategies at inference time; the framework operates as a closed loop of generating, evaluating, summarizing, and updating that continually curates a structured strategy library and adapts across corpora, styles, and task constraints. A decoding LLM recovers the information under the shared strategy. To handle high embedding rates, we introduce PC-DNTE, a plug-and-play algorithm that maintains alignment with the base model's conditional distribution at high embedding rates, preserving imperceptibility while enhancing security. Experimental results demonstrate that at higher embedding rates Auto-Stega achieves superior performance with gains of 42.2\% in perplexity and 1.6\% in anti-steganalysis performance over SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06538v1">Auto-Prompt Ensemble for LLM Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      We present a novel framework that improves the reliability of LLM judges by selectively augmenting LLM with auxiliary evaluation dimensions. Existing LLM judges often miss crucial evaluation dimensions because they fail to recognize the implicit standards underlying human assessments. To address this challenge, we propose the Auto-Prompt Ensemble (APE), an adaptive framework that automatically learns evaluation dimensions from its failure cases. APE incorporates a confidence-based ensemble mechanism to decide when to adopt the judgments from additional evaluation dimensions through a novel confidence estimation approach called Collective Confidence. Extensive experiments demonstrate that APE improves the reliability of LLM Judge across diverse standard benchmarks. For instance, APE enhances GPT-4o agreement rate on Reward Bench from 87.2% to 90.5% in the zero-shot setting. Overall, APE provides a principled approach for LLM Judge to leverage test-time computation, and bridge the evaluation gap between human and LLM judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06530v1">From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      The quality and experience of mobile communication have significantly improved with the introduction of 5G, and these improvements are expected to continue beyond the 5G era. However, vulnerabilities in control-plane protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS), pose significant security threats, such as Blind Denial of Service (DoS) attacks. Despite the availability of existing anomaly detection methods that leverage rule-based systems or traditional machine learning methods, these methods have several limitations, including the need for extensive training data, predefined rules, and limited explainability. Addressing these challenges, we propose a novel anomaly detection framework that leverages the capabilities of Large Language Models (LLMs) in zero-shot mode with unordered data and short natural language attack descriptions within the Open Radio Access Network (O-RAN) architecture. We analyse robustness to prompt variation, demonstrate the practicality of automating the attack descriptions and show that detection quality relies on the semantic completeness of the description rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate the solution and provide an extensive comparison of open-source and proprietary LLM implementations to demonstrate superior performance in attack detection. We further validate the practicality of our framework within O-RAN's real-time constraints, illustrating its potential for detecting other Layer-3 attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15347v2">FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07626v1">LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Machine unlearning for large language models (LLMs) aims to remove undesired data, knowledge, and behaviors (e.g., for safety, privacy, or copyright) while preserving useful model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on the WMDP benchmark. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narrow perspective, often overstating success while overlooking the model's actual generation behavior. To address this gap, we introduce open question-answering (Open-QA) metrics that better capture generative performance and reveal the inherent UE-UT tradeoff across method families. Furthermore, we demonstrate that robustness requires finer-grained analysis: for example, vulnerabilities differ substantially between in-domain relearning and out-of-domain fine-tuning, even though both fall under model-level attacks. Through this study, we hope to deliver a full-stack revisit of LLM unlearning and actionable guidance for designing and evaluating future methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03563v3">Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly sensitive to subtle changes in prompt phrasing, posing challenges for reliable auditing. Prior methods often apply unconstrained prompt paraphrasing, which risk missing linguistic and demographic factors that shape authentic user interactions. We introduce AUGMENT (Automated User-Grounded Modeling and Evaluation of Natural Language Transformations), a framework for generating controlled paraphrases, grounded in user behaviors. AUGMENT leverages linguistically informed rules and enforces quality through checks on instruction adherence, semantic similarity, and realism, ensuring paraphrases are both reliable and meaningful for auditing. Through case studies on the BBQ and MMLU datasets, we show that controlled paraphrases uncover systematic weaknesses that remain obscured under unconstrained variation. These results highlight the value of the AUGMENT framework for reliable auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17694v2">Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted for publication at the 18th International Natural Language Generation Conference (INLG 2025). Revised version: improved image quality and minor corrections. No change to conclusions
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07614v1">Traceability and Accountability in Role-Specialized Multi-Agent LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Sequential multi-agent systems built with large language models (LLMs) can automate complex software tasks, but they are hard to trust because errors quietly pass from one stage to the next. We study a traceable and accountable pipeline, meaning a system with clear roles, structured handoffs, and saved records that let us trace who did what at each step and assign blame when things go wrong. Our setting is a Planner -> Executor -> Critic pipeline. We evaluate eight configurations of three state-of-the-art LLMs on three benchmarks and analyze where errors start, how they spread, and how they can be fixed. Our results show: (1) adding a structured, accountable handoff between agents markedly improves accuracy and prevents the failures common in simple pipelines; (2) models have clear role-specific strengths and risks (e.g., steady planning vs. high-variance critiquing), which we quantify with repair and harm rates; and (3) accuracy-cost-latency trade-offs are task-dependent, with heterogeneous pipelines often the most efficient. Overall, we provide a practical, data-driven method for designing, tracing, and debugging reliable, predictable, and accountable multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07604v1">RustAssure: Differential Symbolic Testing for LLM-Transpiled C-to-Rust Code</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 13 pages to appear in Proceedings of ASE 2025
    </div>
    <details class="paper-abstract">
      Rust is a memory-safe programming language that significantly improves software security. Existing codebases written in unsafe memory languages, such as C, must first be transpiled to Rust to take advantage of Rust's improved safety guarantees. RustAssure presents a system that uses Large Language Models (LLMs) to automatically transpile existing C codebases to Rust. RustAssure uses prompt engineering techniques to maximize the chances of the LLM generating idiomatic and safe Rust code. Moreover, because LLMs often generate code with subtle bugs that can be missed under traditional unit or fuzz testing, RustAssure performs differential symbolic testing to establish the semantic similarity between the original C and LLM-transpiled Rust code. We evaluated RustAssure with five real-world applications and libraries, and showed that our system is able to generate compilable Rust functions for 89.8% of all C functions, of which 69.9% produced equivalent symbolic return values for both the C and Rust functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04822v2">From Handwriting to Feedback: Evaluating VLMs and LLMs for AI-Powered Assessment in Indonesian Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Despite rapid progress in vision-language and large language models (VLMs and LLMs), their effectiveness for AI-driven educational assessment in real-world, underrepresented classrooms remains largely unexplored. We evaluate state-of-the-art VLMs and LLMs on over 14K handwritten answers from grade-4 classrooms in Indonesia, covering Mathematics and English aligned with the local national curriculum. Unlike prior work on clean digital text, our dataset features naturally curly, diverse handwriting from real classrooms, posing realistic visual and linguistic challenges. Assessment tasks include grading and generating personalized Indonesian feedback guided by rubric-based evaluation. Results show that the VLM struggles with handwriting recognition, causing error propagation in LLM grading, yet LLM feedback remains pedagogically useful despite imperfect visual inputs, revealing limits in personalization and contextual relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00218v2">$\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07581v1">Expanding the Action Space of LLMs to Reason Beyond Language</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful reasoners in natural language, but their actions are typically confined to outputting vocabulary tokens. As a result, interactions with external environments -- such as symbolic operators or simulators -- must be expressed through text in predefined formats, parsed, and routed to external interfaces. This overloads the model's language with both reasoning and control duties, and requires a hand-crafted parser, external to the LLM. To address this, we decouple environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary. The model starts reasoning in the default language environment, but may trigger routing actions and switch to an external environment at any time. From there, the model can only invoke environment-specific actions, receive feedback from the environment, and potentially route back to language as a result. To promote effective exploration of the expanded action space and new environments, we introduce ExpA Reinforcement Learning (EARL) with counterfactual policy optimization. On tasks requiring multi-turn interactions and contingent planning, EARL outperforms strong baselines with vocabulary-constrained actions. It performs robustly across calculator-based multi-task learning and, in the partially observed sorting problem, achieves perfect Sort-4 accuracy while self-discovering an efficient algorithm competitive with classical designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11878v3">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07557v1">Investigating Thematic Patterns and User Preferences in LLM Interactions using BERTopic</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      This study applies BERTopic, a transformer-based topic modeling technique, to the lmsys-chat-1m dataset, a multilingual conversational corpus built from head-to-head evaluations of large language models (LLMs). Each user prompt is paired with two anonymized LLM responses and a human preference label, used to assess user evaluation of competing model outputs. The main objective is uncovering thematic patterns in these conversations and examining their relation to user preferences, particularly if certain LLMs are consistently preferred within specific topics. A robust preprocessing pipeline was designed for multilingual variation, balancing dialogue turns, and cleaning noisy or redacted data. BERTopic extracted over 29 coherent topics including artificial intelligence, programming, ethics, and cloud infrastructure. We analysed relationships between topics and model preferences to identify trends in model-topic alignment. Visualization techniques included inter-topic distance maps, topic probability distributions, and model-versus-topic matrices. Our findings inform domain-specific fine-tuning and optimization strategies for improving real-world LLM performance and user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v4">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 NeurIPS 2025 Workshop on MTI-LLM
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge (LLMaaJ) enables scalable evaluation, yet we lack a decisive test of a judge's qualification: can it recover the hidden objective of a conversation and know when that inference is reliable? Large language models degrade with irrelevant or lengthy context, and multi-turn jailbreaks can scatter goals across turns. We present ObjexMT, a benchmark for objective extraction and metacognition. Given a multi-turn transcript, a model must output a one-sentence base objective and a self-reported confidence. Accuracy is scored by semantic similarity to gold objectives, then thresholded once on 300 calibration items ($\tau^\star = 0.66$; $F_1@\tau^\star = 0.891$). Metacognition is assessed with expected calibration error, Brier score, Wrong@High-Confidence (0.80 / 0.90 / 0.95), and risk--coverage curves. Across six models (gpt-4.1, claude-sonnet-4, Qwen3-235B-A22B-FP8, kimi-k2, deepseek-v3.1, gemini-2.5-flash) evaluated on SafeMTData\_Attack600, SafeMTData\_1K, and MHJ, kimi-k2 achieves the highest objective-extraction accuracy (0.612; 95\% CI [0.594, 0.630]), while claude-sonnet-4 (0.603) and deepseek-v3.1 (0.599) are statistically tied. claude-sonnet-4 offers the best selective risk and calibration (AURC 0.242; ECE 0.206; Brier 0.254). Performance varies sharply across datasets (16--82\% accuracy), showing that automated obfuscation imposes challenges beyond model choice. High-confidence errors remain: Wrong@0.90 ranges from 14.9\% (claude-sonnet-4) to 47.7\% (Qwen3-235B-A22B-FP8). ObjexMT therefore supplies an actionable test for LLM judges: when objectives are implicit, judges often misinfer them; exposing objectives or gating decisions by confidence is advisable. All experimental data are in the Supplementary Material and at https://github.com/hyunjun1121/ObjexMT_dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v7">Can LLM Agents Simulate Multi-Turn Human Behavior? Evidence from Real Online Customer Behavior Data</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Recent research shows that LLM Agents can generate ``believable'' human behaviors via prompt-only methods, and such agents have been increasingly adopted in downstream applications. However, existing evaluation of these agents only focuses on qualitative believability (whether human raters think they are accurate), leaving open questions of whether LLM agents can accurately generate step-by-step actions mimicking a particular human's behavior in a multi-turn interaction task. In this work, we take shopping as a case study and present the first large-scale quantitative evaluation of state-of-the-art LLMs' ability to accurately simulate human behavior. Using real-world data from 31,865 online shopping sessions containing 230,965 user actions, our evaluation reveals that prompt-based LLMs (DeepSeek-R1, Llama, Claude) achieve only 11.86% accuracy in generating human actions, highlighting a substantial gap in actual behavioral accuracy. Through experiments, we also showcase that strategies as simple as fine-tuning LLMs on real human click-through data augmented with synthesized reasoning traces can greatly enhance models' performance. The fine-tuned Qwen2.5-7B achieves 17.26% action generation accuracy and 33.86% F1 score on final purchase prediction, representing substantial improvements of 5.4% and 13.85% over prompt-only baselines. This work establishes the first rigorous benchmark for human behavior simulation and provides actionable insights for developing more accurate LLM agents for future downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23234v3">p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03889v3">Identifying and Evaluating Inactive Heads in Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Attention is foundational to large language models (LLMs), enabling different heads to have diverse focus on relevant input tokens. However, learned behaviors like attention sinks, where the first token receives the most attention despite limited semantic importance, suggest some heads may be inactive, and point to a significant source of computational redundancy. To analyze this phenomenon, we propose a taxonomy of 13 score functions that measure different ways a head can be inactive. Thresholding these scores allows us to analyze different sets of potentially inactive attention heads. We evaluate whether identified heads are inactive through model interventions, finding that more than 12% of attention heads are inactive on average, and can be ablated in specific contexts while maintaining MMLU accuracy to within 1% of the pretrained LLM. Across 3 model families, our score functions that measure the average norm of a head's output consistently identify inactive heads that would not have been found by score functions that rely solely on attention weights. We establish that relying on a score function that measures a first token attention sink would underestimate the prevalence of inactive heads, failing to identify more than 7% of inactive heads on average. We also show how measuring score distributions can provide insights into attention behavior. For instance, we find evidence that finetuning causes little to no change in attention behavior, and that even within the same model family, large model scales present markedly different attention behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07497v1">Can Speech LLMs Think while Listening?</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Recent advances in speech large language models (speech LLMs) have enabled seamless spoken interactions, but these systems still struggle with complex reasoning tasks. Previously, chain-of-thought (CoT) prompting or fine-tuning has been to shown to significantly improve the reasoning abilities of text-based LLMs. In this work, we investigate the effect of CoT fine-tuning for multi-stream speech LLMs, demonstrating that reasoning in text space improves the accuracy of speech LLMs by 2.4x, on average, over a suite of spoken reasoning tasks. Beyond accuracy, the latency of the spoken response is a crucial factor for interacting with voice-based agents. Inspired by the human behavior of "thinking while listening," we propose methods to reduce the additional latency from reasoning by allowing the model to start reasoning before the user query has ended. To achieve this, we introduce an entropy-based metric, "question completeness," which acts as an indicator to guide the model on the optimal time to start reasoning. This method provides greater control over the accuracy-latency trade-off compared with heuristic-based approaches and, under equivalent latency conditions, yields a 4% accuracy gain on ARC-Easy. Finally, we use Direct Preference Optimization (DPO) on preference data created using rejection sampling to push the accuracy-latency pareto frontier further, resulting in a 70% reduction in latency without loss in accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v3">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning (NFL-HR)**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the NL-FL Problem Alignment method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the Mixed Problem Input technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based Answer Extraction mechanism. Comprehensive experiments demonstrate that the NFL-HR framework achieves **89.80**% and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by **4.60%** and **4.82%**, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07489v1">Evaluation of LLMs for Process Model Analysis and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 15 pages, 5 tables, 4 figures; full research paper currently under review for the Workshop on Information Technologies and Systems (WITS) 2025. The paper presents a comprehensive evaluation of large language models (LLMs) for business process model analysis and optimization, including error detection, reasoning, and scenario-based redesign
    </div>
    <details class="paper-abstract">
      In this paper, we report our experience with several LLMs for their ability to understand a process model in an interactive, conversational style, find syntactical and logical errors in it, and reason with it in depth through a natural language (NL) interface. Our findings show that a vanilla, untrained LLM like ChatGPT (model o3) in a zero-shot setting is effective in understanding BPMN process models from images and answering queries about them intelligently at syntactic, logic, and semantic levels of depth. Further, different LLMs vary in performance in terms of their accuracy and effectiveness. Nevertheless, our empirical analysis shows that LLMs can play a valuable role as assistants for business process designers and users. We also study the LLM's "thought process" and ability to perform deeper reasoning in the context of process analysis and optimization. We find that the LLMs seem to exhibit anthropomorphic properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23798v3">Adaptive Layer-skipping in Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Various layer-skipping methods have been proposed to accelerate token generation in large language models (LLMs). However, limited attention has been paid to a fundamental question: How do computational demands vary across the generation of different tokens? In this work, we introduce FlexiDepth, a method that dynamically adjusts the number of Transformer layers used in text generation. By incorporating a plug-in router and adapter, FlexiDepth enables adaptive computation in LLMs without modifying their original parameters. Applied to Llama-3-8B, it skips 8 out of 32 layers while maintaining full benchmark performance. Our experiments reveal that computational demands in LLMs significantly vary based on token type. Specifically, generating repetitive tokens or fixed phrases requires fewer layers, whereas producing tokens involving computation or high uncertainty requires more layers. Despite the computational savings, FlexiDepth does not yet achieve wall-clock speedup due to varied skipping patterns and I/O overhead. To inspire future work and advance research on practical speedup, we open-sourced FlexiDepth and a dataset documenting its layer allocation patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07458v1">Populism Meets AI: Advancing Populism Research with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 27 pages, 3 figures. Preprint version under review
    </div>
    <details class="paper-abstract">
      Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00529v2">Modeling Motivated Reasoning in Law: Evaluating Strategic Role Conditioning in LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at NLLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate user-tailored summaries, adapting outputs to specific stakeholders. In legal contexts, this raises important questions about motivated reasoning -- how models strategically frame information to align with a stakeholder's position within the legal system. Building on theories of legal realism and recent trends in legal practice, we investigate how LLMs respond to prompts conditioned on different legal roles (e.g., judges, prosecutors, attorneys) when summarizing judicial decisions. We introduce an evaluation framework grounded in legal fact and reasoning inclusion, also considering favorability towards stakeholders. Our results show that even when prompts include balancing instructions, models exhibit selective inclusion patterns that reflect role-consistent perspectives. These findings raise broader concerns about how similar alignment may emerge as LLMs begin to infer user roles from prior interactions or context, even without explicit role instructions. Our results underscore the need for role-aware evaluation of LLM summarization behavior in high-stakes legal settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07437v1">LASER: An LLM-based ASR Scoring and Evaluation Rubric</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Standard ASR evaluation metrics like Word Error Rate (WER) tend to unfairly penalize morphological and syntactic nuances that do not significantly alter sentence semantics. We introduce an LLM-based scoring rubric LASER that leverages state-of-the-art LLMs' in-context learning abilities to learn from prompts with detailed examples. Hindi LASER scores using Gemini 2.5 Pro achieved a very high correlation score of 94% with human annotations. Hindi examples in the prompt were also effective in analyzing errors in other Indian languages such as Marathi, Kannada and Malayalam. We also demonstrate how a smaller LLM like Llama 3 can be finetuned on word-pair examples derived from reference and ASR predictions to predict what kind of penalty should be applied with close to 89% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07429v1">Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Efficient use of large language models (LLMs) is critical for deployment at scale: without adaptive routing, systems either overpay for strong models or risk poor performance from weaker ones. Selecting the right LLM for each query is fundamentally an online decision problem: models differ in strengths, prices fluctuate, and users value accuracy and cost differently. Yet most routers are trained offline with labels for all candidate models, an assumption that breaks in deployment, where only the outcome of the chosen model is observed. We bridge this gap with BaRP, a Bandit-feedback Routing with Preferences approach that trains under the same partial-feedback restriction as deployment, while supporting preference-tunable inference: operators can dial the performance/cost trade-off at test time without retraining. Framed as a contextual bandit over prompt features and a user preference vector, our method simulates an online feedback setting during training and adapts its routing decisions to each new prompt, rather than depending on full-information offline supervision. Comprehensive experiments show that our method consistently outperforms strong offline routers by at least 12.46% and the largest LLM by at least 2.45%, and generalizes robustly for unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18344v2">Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SubSpec, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1x speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5x speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00096v3">BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 8 main text pages, 5 main figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07355v1">AV-EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Omni-modal LLMS with Audio-visual Cues</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Emotions conveyed through voice and face shape engagement and context in human-AI interaction. Despite rapid progress in omni-modal large language models (LLMs), the holistic evaluation of emotional reasoning with audiovisual cues remains limited. To address this gap, we introduce AV-EMO-Reasoning, a benchmark designed to systematically assess emotional coherence in LLMs. The framework leverages a curated, single- and multi-turn synthetic audiovisual corpus with a real-world set and is assessed under continuous, categorical, and perceptual metrics. Experiments with leading LLMs show that visual cues reliably improve emotional coherence over audio-only baselines. Moreover, LLMs can leverage audio-visual cues to generate more emotion-aware speech. Models exhibit complementary strengths across metric families, indicating that automatic scores capture facets distinct from perceptual judgments. By releasing a systematic evaluation benchmark, AV-EMO-Reasoning offers a reproducible standard for evaluating emotion-aware dialogue and advances toward more natural, adaptive human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12726v3">DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, and lack guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to create reasoning questions that far surpass the difficulty and diversity of existing datasets. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, after SFT on our datasets, the base versions of these models even surpass their official instruction-tuned counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16185v2">ParamBench: A Graduate-Level Benchmark for Evaluating LLM Understanding on Indic Subjects</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models have been widely evaluated on tasks such as comprehension, summarization, code generation, etc. However, their performance on graduate-level, culturally grounded questions in the Indian context remains largely unexplored. Existing Indian benchmarks emphasise basic fact-orientated queries that offer limited assessment of a deeper disciplinary understanding tailored to the Indian setting. In this paper, we present ParamBench, consisting of more than 17K questions in the Hindi language, comprising questionnaires from 21 diverse subjects. These questions are primarily derived from a nationwide graduate-level entrance examination covering topics such as history, music, instruments, yoga, literature, philosophy, law, etc.~ specifically for the Indian context. Additionally, we assess the ability of LLMs to handle diverse question formats - such as list-based matching, assertion-reason pairs, and sequence ordering - alongside conventional multiple-choice questions. We evaluated the performance of more than 16 open source LLMs on this benchmark, observing that Gemma3-27B attains the highest overall accuracy of 56.4\%. Furthermore, subject-wise analysis indicates that even for the best-performing LLMs, performance remains weak on topics such as music, classical instruments, and law, underscoring persistent challenges in culturally grounded reasoning. The dataset and source code is present at https://github.com/ayushbits/ParamBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16276v3">Empowering LLMs with Pseudo-Untrimmed Videos for Audio-Visual Temporal Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted to AAAI 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language and multimodal domains. By fine-tuning multimodal LLMs with temporal annotations from well-annotated datasets, e.g., dense video captioning datasets, their temporal understanding capacity in video-language tasks can be obtained. However, there is a notable lack of untrimmed audio-visual video datasets with precise temporal annotations for events. This deficiency hinders LLMs from learning the alignment between time, audio-visual events, and text tokens, thus impairing their ability to temporally localize audio-visual events in videos. To address this gap, we introduce PU-VALOR, a comprehensive audio-visual dataset comprising over 114,000 pseudo-untrimmed videos with detailed temporal annotations. PU-VALOR is derived from the large-scale but coarse-annotated audio-visual dataset VALOR, through a subtle method involving event-based video clustering, random temporal scaling, and permutation. By fine-tuning a multimodal LLM on PU-VALOR, we developed AVicuna, a model capable of aligning audio-visual events with temporal intervals and corresponding text tokens. AVicuna excels in temporal localization and time-aware dialogue capabilities. Our experiments demonstrate that AVicuna effectively handles temporal understanding in audio-visual videos and achieves state-of-the-art performance on open-ended video QA, audio-visual QA, and audio-visual event dense localization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07243v1">LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Published in Natural Legal Language Processing - EMNLP Workshop 2025
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM) outputs in the legal domain presents unique challenges due to the complex and nuanced nature of legal analysis. Current evaluation approaches either depend on reference data, which is costly to produce, or use standardized assessment methods, both of which have significant limitations for legal applications. Although LLM-as-a-Judge has emerged as a promising evaluation technique, its reliability and effectiveness in legal contexts depend heavily on evaluation processes unique to the legal industry and how trustworthy the evaluation appears to the human legal expert. This is where existing evaluation methods currently fail and exhibit considerable variability. This paper aims to close the gap: a) we break down lengthy responses into 'Legal Data Points' (LDPs), self-contained units of information, and introduce a novel, reference-free evaluation methodology that reflects how lawyers evaluate legal answers; b) we demonstrate that our method outperforms a variety of baselines on both our proprietary dataset and an open-source dataset (LegalBench); c) we show how our method correlates more closely with human expert evaluations and helps improve inter-annotator agreement; and finally d) we open source our Legal Data Points for a subset of LegalBench used in our experiments, allowing the research community to replicate our results and advance research in this vital area of LLM evaluation on legal question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07239v1">Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07231v1">Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07230v1">Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07192v1">Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07178v1">Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossible</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Are large language models (LLMs) sensitive to the distinction between humanly possible languages and humanly impossible languages? This question is taken by many to bear on whether LLMs and humans share the same innate learning biases. Previous work has attempted to answer it in the positive by comparing LLM learning curves on existing language datasets and on "impossible" datasets derived from them via various perturbation functions. Using the same methodology, we examine this claim on a wider set of languages and impossible perturbations. We find that in most cases, GPT-2 learns each language and its impossible counterpart equally easily, in contrast to previous claims. We also apply a more lenient condition by testing whether GPT-2 provides any kind of separation between the whole set of natural languages and the whole set of impossible languages. By considering cross-linguistic variance in various metrics computed on the perplexity curves, we show that GPT-2 provides no systematic separation between the possible and the impossible. Taken together, these perspectives show that LLMs do not share the human innate biases that shape linguistic typology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07176v1">Exposing LLM User Privacy via Traffic Fingerprint Analysis: A Study of Privacy Risks in LLM Agent Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 26 pages with 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed as agents that orchestrate tasks and integrate external tools to execute complex workflows. We demonstrate that these interactive behaviors leave distinctive fingerprints in encrypted traffic exchanged between users and LLM agents. By analyzing traffic patterns associated with agent workflows and tool invocations, adversaries can infer agent activities, distinguish specific agents, and even profile sensitive user attributes. To highlight this risk, we develop AgentPrint, which achieves an F1-score of 0.866 in agent identification and attains 73.9% and 69.1% top-3 accuracy in user attribute inference for simulated- and real-user settings, respectively. These results uncover an overlooked risk: the very interactivity that empowers LLM agents also exposes user privacy, underscoring the urgent need for technical countermeasures alongside regulatory and policy safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07175v1">Quantifying Data Contamination in Psychometric Evaluations of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Recent studies apply psychometric questionnaires to Large Language Models (LLMs) to assess high-level psychological constructs such as values, personality, moral foundations, and dark traits. Although prior work has raised concerns about possible data contamination from psychometric inventories, which may threaten the reliability of such evaluations, there has been no systematic attempt to quantify the extent of this contamination. To address this gap, we propose a framework to systematically measure data contamination in psychometric evaluations of LLMs, evaluating three aspects: (1) item memorization, (2) evaluation memorization, and (3) target score matching. Applying this framework to 21 models from major families and four widely used psychometric inventories, we provide evidence that popular inventories such as the Big Five Inventory (BFI-44) and Portrait Values Questionnaire (PVQ-40) exhibit strong contamination, where models not only memorize items but can also adjust their responses to achieve specific target scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07172v1">NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 60 pages, 18 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using metaphysical shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07096v1">Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Sarcasm is a subtle form of non-literal language that poses significant challenges for speech synthesis due to its reliance on nuanced semantic, contextual, and prosodic cues. While existing speech synthesis research has focused primarily on broad emotional categories, sarcasm remains largely unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which provide expressive reference patterns of sarcastic delivery. Integrated within a VITS backbone, this dual conditioning enables more natural and contextually appropriate sarcastic speech. Experiments demonstrate that our method outperforms baselines in both objective measures and subjective evaluations, yielding improvements in speech naturalness, sarcastic expressivity, and downstream sarcasm detection.
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
