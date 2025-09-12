# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03047v1">FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made a profound impact across various fields due to their advanced capabilities. However, training these models at unprecedented scales requires extensive AI accelerator clusters and sophisticated parallelism strategies, which pose significant challenges in maintaining system reliability over prolonged training periods. A major concern is the substantial loss of training time caused by inevitable hardware and software failures. To address these challenges, we present FlashRecovery, a fast and low-cost failure recovery system comprising three core modules: (1) Active and real-time failure detection. This module performs continuous training state monitoring, enabling immediate identification of hardware and software failures within seconds, thus ensuring rapid incident response; (2) Scale-independent task restart. By employing different recovery strategies for normal and faulty nodes, combined with an optimized communication group reconstruction protocol, our approach ensures that the recovery time remains nearly constant, regardless of cluster scale; (3) Checkpoint-free recovery within one step. Our novel recovery mechanism enables single-step restoration, completely eliminating dependence on traditional checkpointing methods and their associated overhead. Collectively, these innovations enable FlashRecovery to achieve optimal Recovery Time Objective (RTO) and Recovery Point Objective (RPO), substantially improving the reliability and efficiency of long-duration LLM training. Experimental results demonstrate that FlashRecovery system can achieve training restoration on training cluster with 4, 800 devices in 150 seconds. We also verify that the time required for failure recovery is nearly consistent for different scales of training tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01245v2">Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis. We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in https://github.com/eunomia-bpf/schedcp
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00971v2">CoreThink: A Symbolic Reasoning Layer to reason over Long Horizon Tasks with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      We introduce CoreThink, a state-of-the-art Reasoning Layer built upon a novel reasoning method called General Symbolics. This approach diverges from reasoning paradigms such as test-time scaling, Supervised Fine-Tuning (SFT), and Reinforcement Learning with Verifiable Rewards (RLVR). CoreThink General Symbolic Reasoner (GSR) is specifically structured around three key use cases: tool-calling, code generation, and planning, demonstrating exemplary performance across a total of seven benchmarks in their respective areas. Notably, we are achieving SOTA scores of 66.66% on Livecodebench v6, 89% on Instruction-Following Evals, and 24.4% on ARC-AGI-2. We also present an agentic coding IDE, developed using the principles of General Symbolics, which achieves a state-of-the-art accuracy of 62.3% on SWE-Bench Lite. We are able to achieve these improvements without any fine-tuning or training costs. Our Reasoning Layer is designed to provide a pure performance uplift, ensuring that a model's accuracy on reasoning tasks is never negatively impacted. We argue that incumbent methods will eventually lead to diminishing returns in LLM performance, necessitating the development of new reasoning techniques. This technical report details our approach at a high level and the availability of the CoreThink models for reasoning-intensive use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19992v2">HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v1">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03018v1">Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Reliability is essential for ensuring efficiency in LLM training. However, many real-world reliability issues remain difficult to resolve, resulting in wasted resources and degraded model performance. Unfortunately, today's collective communication libraries operate as black boxes, hiding critical information needed for effective root cause analysis. We propose Mycroft, a lightweight distributed tracing and root cause analysis system designed to address previously hidden reliability issues in collective communication. Mycroft's key idea is to trace collective communication states and leverage internal control and data dependencies to resolve reliability problems in LLM training. Mycroft has been deployed at ByteDance for over six months to debug collective communication related issues at runtime. It detected anomalies within 15 seconds in 90% of cases and identified the root cause within 20 seconds in 60% of cases. We also conducted extensive fault injection experiments to demonstrate Mycroft's capability and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13748v3">Learn and Unlearn: Addressing Misinformation in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      This paper investigates the propagation of harmful information in multilingual large language models (LLMs) and evaluates the efficacy of various unlearning methods. We demonstrate that fake information, regardless of the language it is in, once introduced into these models through training data, can spread across different languages, compromising the integrity and reliability of the generated content. Our findings reveal that standard unlearning techniques, which typically focus on English data, are insufficient in mitigating the spread of harmful content in multilingual contexts and could inadvertently reinforce harmful content across languages. We show that only by addressing harmful responses in both English and the original language of the harmful data can we effectively eliminate generations for all languages. This underscores the critical need for comprehensive unlearning strategies that consider the multilingual nature of modern LLMs to enhance their safety and reliability across diverse linguistic landscapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02998v1">Integrating Generative AI into Cybersecurity Education: A Study of OCR and Multimodal LLM-assisted Instruction</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 9 pages, 3 figures, accepted by IEEE FIE 2025
    </div>
    <details class="paper-abstract">
      This full paper describes an LLM-assisted instruction integrated with a virtual cybersecurity lab platform. The digital transformation of Fourth Industrial Revolution (4IR) systems is reshaping workforce needs, widening skill gaps, especially among older workers. With rising emphasis on robotics, automation, AI, and security, re-skilling and up-skilling are essential. Generative AI can help build this workforce by acting as an instructional assistant to support skill acquisition during experiential learning. We present a generative AI instructional assistant integrated into a prior experiential learning platform. The assistant employs a zero-shot OCR-LLM pipeline within the legacy Cybersecurity Labs-as-a-Service (CLaaS) platform (2015). Text is extracted from slide images using Tesseract OCR, then simplified instructions are generated via a general-purpose LLM, enabling real-time instructional support with minimal infrastructure. The system was evaluated in a live university course where student feedback (n=42) averaged 7.83/10, indicating strong perceived usefulness. A comparative study with multimodal LLMs that directly interpret slide images showed higher performance on visually dense slides, but the OCR-LLM pipeline provided comparable pedagogical value on text-centric slides with much lower computational overhead and cost. This work demonstrates that a lightweight, easily integrable pipeline can effectively extend legacy platforms with modern generative AI, offering scalable enhancements for student comprehension in technical education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v4">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15486v3">SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now support extremely long context windows, but the quadratic complexity of vanilla attention results in significantly long Time-to-First-Token (TTFT) latency. Existing approaches to address this complexity require additional pretraining or finetuning, and often sacrifice model accuracy. In this paper, we first provide both theoretical and empirical foundations for near-lossless sparse attention. We find dynamically capturing head-specific sparse patterns at runtime with low overhead is crucial. To address this, we propose SampleAttention, an adaptive structured and near-lossless sparse attention. Leveraging observed significant sparse patterns, SampleAttention attends to a fixed percentage of adjacent tokens to capture local window patterns, and employs a two-stage query-guided key-value filtering approach, which adaptively select a minimum set of key-values with low overhead, to capture column stripe patterns. Comprehensive evaluations show that SampleAttention can seamlessly replace vanilla attention in off-the-shelf LLMs with nearly no accuracy loss, and reduces TTFT by up to $2.42\times$ compared with FlashAttention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02910v1">The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act on people's behalf: they write emails, buy groceries, and book restaurants. While the outsourcing of human decision-making to AI can be both efficient and effective, it raises a fundamental question: how does delegating identity-defining choices to AI reshape who people become? We study the impact of agentic LLMs on two identity-relevant outcomes: interpersonal distinctiveness - how unique a person's choices are relative to others - and intrapersonal diversity - the breadth of a single person's choices over time. Using real choices drawn from social-media behavior of 1,000 U.S. users (110,000 choices in total), we compare a generic and personalized agent to a human baseline. Both agents shift people's choices toward more popular options, reducing the distinctiveness of their behaviors and preferences. While the use of personalized agents tempers this homogenization (compared to the generic AI), it also more strongly compresses the diversity of people's preference portfolios by narrowing what they explore across topics and psychological affinities. Understanding how AI agents might flatten human experience, and how using generic versus personalized agents involves distinctiveness-diversity trade-offs, is critical for designing systems that augment rather than constrain human agency, and for safeguarding diversity in thought, taste, and expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03768v1">RAGuard: A Novel Approach for in-context Safe Retrieval Augmented Generation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Accuracy and safety are paramount in Offshore Wind (OSW) maintenance, yet conventional Large Language Models (LLMs) often fail when confronted with highly specialised or unexpected scenarios. We introduce RAGuard, an enhanced Retrieval-Augmented Generation (RAG) framework that explicitly integrates safety-critical documents alongside technical manuals.By issuing parallel queries to two indices and allocating separate retrieval budgets for knowledge and safety, RAGuard guarantees both technical depth and safety coverage. We further develop a SafetyClamp extension that fetches a larger candidate pool, "hard-clamping" exact slot guarantees to safety. We evaluate across sparse (BM25), dense (Dense Passage Retrieval) and hybrid retrieval paradigms, measuring Technical Recall@K and Safety Recall@K. Both proposed extensions of RAG show an increase in Safety Recall@K from almost 0\% in RAG to more than 50\% in RAGuard, while maintaining Technical Recall above 60\%. These results demonstrate that RAGuard and SafetyClamp have the potential to establish a new standard for integrating safety assurance into LLM-powered decision support in critical maintenance contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03764v1">LLM-based Relevance Assessment for Web-Scale Search Evaluation at Pinterest</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 RecSys 2025 EARL Workshop
    </div>
    <details class="paper-abstract">
      Relevance evaluation plays a crucial role in personalized search systems to ensure that search results align with a user's queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present our approach at Pinterest Search to automate relevance evaluation for online experiments using fine-tuned LLMs. We rigorously validate the alignment between LLM-generated judgments and human annotations, demonstrating that LLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging LLM-based labeling further unlocks the opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effect (MDE) in online experiment measurements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03746v1">Efficient Item ID Generation for Large-Scale LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Integrating product catalogs and user behavior into LLMs can enhance recommendations with broad world knowledge, but the scale of real-world item catalogs, often containing millions of discrete item identifiers (Item IDs), poses a significant challenge. This contrasts with the smaller, tokenized text vocabularies typically used in LLMs. The predominant view within the LLM-based recommendation literature is that it is infeasible to treat item ids as a first class citizen in the LLM and instead some sort of tokenization of an item into multiple tokens is required. However, this creates a key practical bottleneck in serving these models for real-time low-latency applications. Our paper challenges this predominant practice and integrates item ids as first class citizens into the LLM. We provide simple, yet highly effective, novel training and inference modifications that enable single-token representations of items and single-step decoding. Our method shows improvements in recommendation quality (Recall and NDCG) over existing techniques on the Amazon shopping datasets while significantly improving inference efficiency by 5x-14x. Our work offers an efficiency perspective distinct from that of other popular approaches within LLM-based recommendation, potentially inspiring further research and opening up a new direction for integrating IDs into LLMs. Our code is available here https://drive.google.com/file/d/1cUMj37rV0Z1bCWMdhQ6i4q4eTRQLURtC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03736v1">Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 25 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The impressive capabilities of Large Language Models (LLMs) have fueled the notion that synthetic agents can serve as substitutes for real participants in human-subject research. In an effort to evaluate the merits of this claim, social science researchers have largely focused on whether LLM-generated survey data corresponds to that of a human counterpart whom the LLM is prompted to represent. In contrast, we address a more fundamental question: Do agents maintain internal consistency, retaining similar behaviors when examined under different experimental settings? To this end, we develop a study designed to (a) reveal the agent's internal state and (b) examine agent behavior in a basic dialogue setting. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversation behavior is consistent with what we would expect from their revealed internal state. Our findings on these hypotheses show significant internal inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be internally consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research. Our simulation code and data are publicly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09002v3">KNighter: Transforming Static Analysis with LLM-Synthesized Checkers</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 SOSP 2025
    </div>
    <details class="paper-abstract">
      Static analysis is a powerful technique for bug detection in critical systems like operating system kernels. However, designing and implementing static analyzers is challenging, time-consuming, and typically limited to predefined bug patterns. While large language models (LLMs) have shown promise for static analysis, directly applying them to scan large systems remains impractical due to computational constraints and contextual limitations. We present KNighter, the first approach that unlocks scalable LLM-based static analysis by automatically synthesizing static analyzers from historical bug patterns. Rather than using LLMs to directly analyze massive systems, our key insight is leveraging LLMs to generate specialized static analyzers guided by historical patch knowledge. KNighter implements this vision through a multi-stage synthesis pipeline that validates checker correctness against original patches and employs an automated refinement process to iteratively reduce false positives. Our evaluation on the Linux kernel demonstrates that KNighter generates high-precision checkers capable of detecting diverse bug patterns overlooked by existing human-written analyzers. To date, KNighter-synthesized checkers have discovered 92 new, critical, long-latent bugs (average 4.3 years) in the Linux kernel; 77 are confirmed, 57 fixed, and 30 have been assigned CVE numbers. This work establishes an entirely new paradigm for scalable, reliable, and traceable LLM-based static analysis for real-world systems via checker synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03730v1">The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion
    </div>
    <details class="paper-abstract">
      Personality traits have long been studied as predictors of human behavior.Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01228v2">ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving demands low latency and high throughput, but high load variability makes it challenging to achieve high GPU utilization. In this paper, we identify a synergetic but overlooked opportunity to co-serve latency-critical online requests alongside latency-tolerant offline tasks such as model benchmarking. While promising, existing serving systems fail to co-serve them efficiently, as their coarse-grained resource management at the request or iteration level cannot harvest millisecond-level GPU idle cycles without introducing interference that violates online latency objectives. ConServe is a new LLM co-serving system that achieves high throughput and strong online latency guarantees by managing resources at finer granularities. ConServe introduces three techniques: (1) a latency-aware token-level scheduler that precisely sizes offline batches and tokens to fit within online latency objectives; (2) sub-iteration, layer-wise preemption that allows offline tasks to yield to online load spikes; and (3) incremental KV cache management that enables preempting and resuming offline requests at near-zero cost. Evaluations with Llama-3.1 and Qwen-2.5 models on real-world workloads show that ConServe delivers an average of 2.2$\times$ higher throughput and reduces online serving tail latency by 2.9$\times$ on average compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02025v2">Evaluating the Efficacy of LLM-Based Reasoning for Multiobjective HPC Job Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 10 pages, 6 figures, work under review
    </div>
    <details class="paper-abstract">
      High-Performance Computing (HPC) job scheduling involves balancing conflicting objectives such as minimizing makespan, reducing wait times, optimizing resource use, and ensuring fairness. Traditional methods, including heuristic-based, e.g., First-Come-First-Served (FJFS) and Shortest Job First (SJF), or intensive optimization techniques, often lack adaptability to dynamic workloads and, more importantly, cannot simultaneously optimize multiple objectives in HPC systems. To address this, we propose a novel Large Language Model (LLM)-based scheduler using a ReAct-style framework (Reason + Act), enabling iterative, interpretable decision-making. The system incorporates a scratchpad memory to track scheduling history and refine decisions via natural language feedback, while a constraint enforcement module ensures feasibility and safety. We evaluate our approach using OpenAI's O4-Mini and Anthropic's Claude 3.7 across seven real-world HPC workload scenarios, including heterogeneous mixes, bursty patterns, and adversarial cases etc. Comparisons against FCFS, SJF, and Google OR-Tools (on 10 to 100 jobs) reveal that LLM-based scheduling effectively balances multiple objectives while offering transparent reasoning through natural language traces. The method excels in constraint satisfaction and adapts to diverse workloads without domain-specific training. However, a trade-off between reasoning quality and computational overhead challenges real-time deployment. This work presents the first comprehensive study of reasoning-capable LLMs for HPC scheduling, demonstrating their potential to handle multiobjective optimization while highlighting limitations in computational efficiency. The findings provide insights into leveraging advanced language models for complex scheduling problems in dynamic HPC environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03696v1">LLMs for estimating positional bias in logged interaction data</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Accepted at the CONSEQUENCES Workshop @ RecSys'25
    </div>
    <details class="paper-abstract">
      Recommender and search systems commonly rely on Learning To Rank models trained on logged user interactions to order items by predicted relevance. However, such interaction data is often subject to position bias, as users are more likely to click on items that appear higher in the ranking, regardless of their actual relevance. As a result, newly trained models may inherit and reinforce the biases of prior ranking models rather than genuinely improving relevance. A standard approach to mitigate position bias is Inverse Propensity Scoring (IPS), where the model's loss is weighted by the inverse of a propensity function, an estimate of the probability that an item at a given position is examined. However, accurate propensity estimation is challenging, especially in interfaces with complex non-linear layouts. In this paper, we propose a novel method for estimating position bias using Large Language Models (LLMs) applied to logged user interaction data. This approach offers a cost-effective alternative to online experimentation. Our experiments show that propensities estimated with our LLM-as-a-judge approach are stable across score buckets and reveal the row-column effects of Viator's grid layout that simpler heuristics overlook. An IPS-weighted reranker trained with these propensities matches the production model on standard NDCG@10 while improving weighted NDCG@10 by roughly 2%. We will verify these offline gains in forthcoming live-traffic experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05982v2">HamRaz: A Culture-Based Persian Conversation Dataset for Person-Centered Therapy Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      We present HamRaz, a culturally adapted Persian-language dataset for AI-assisted mental health support, grounded in Person-Centered Therapy (PCT). To reflect real-world therapeutic challenges, we combine script-based dialogue with adaptive large language models (LLM) role-playing, capturing the ambiguity and emotional nuance of Persian-speaking clients. We introduce HamRazEval, a dual-framework for assessing conversational and therapeutic quality using General Metrics and specialized psychological relationship measures. Human evaluations show HamRaz outperforms existing baselines in empathy, coherence, and realism. This resource contributes to the Digital Humanities by bridging language, culture, and mental health in underrepresented communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00406v3">Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 ECTEL 2025 Preprint. This is an extended version of a poster paper accepted and published at ECTEL-2025
    </div>
    <details class="paper-abstract">
      Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11671v2">Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications, strengthening sociological theory and measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03647v1">Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on legitimate self-preference and unbiased agreement, implying self-preference spans multiple or nonlinear directions. This underscores both their promise and limits as safeguards for LLM-as-judges and motivates more robust interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03646v1">Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03581v1">Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities. To our knowledge, this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks, paving the way for more efficient, adaptive, and controllable agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04523v1">Using LLMs to create analytical datasets: A case study of reconstructing the historical memory of Colombia</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Colombia has been submerged in decades of armed conflict, yet until recently, the systematic documentation of violence was not a priority for the Colombian government. This has resulted in a lack of publicly available conflict information and, consequently, a lack of historical accounts. This study contributes to Colombia's historical memory by utilizing GPT, a large language model (LLM), to read and answer questions about over 200,000 violence-related newspaper articles in Spanish. We use the resulting dataset to conduct both descriptive analysis and a study of the relationship between violence and the eradication of coca crops, offering an example of policy analyses that such data can support. Our study demonstrates how LLMs have opened new research opportunities by enabling examinations of large text corpora at a previously infeasible depth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03518v1">Can LLMs Lie? Investigation beyond Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Website at https://llm-liar.github.io/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across a variety of tasks, but their increasing autonomy in real-world applications raises concerns about their trustworthiness. While hallucinations-unintentional falsehoods-have been widely studied, the phenomenon of lying, where an LLM knowingly generates falsehoods to achieve an ulterior objective, remains underexplored. In this work, we systematically investigate the lying behavior of LLMs, differentiating it from hallucinations and testing it in practical scenarios. Through mechanistic interpretability techniques, we uncover the neural mechanisms underlying deception, employing logit lens analysis, causal interventions, and contrastive activation steering to identify and control deceptive behavior. We study real-world lying scenarios and introduce behavioral steering vectors that enable fine-grained manipulation of lying tendencies. Further, we explore the trade-offs between lying and end-task performance, establishing a Pareto frontier where dishonesty can enhance goal optimization. Our findings contribute to the broader discourse on AI ethics, shedding light on the risks and potential safeguards for deploying LLMs in high-stakes environments. Code and more illustrations are available at https://llm-liar.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03501v1">Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 This technical report serves as the archival version of our paper accepted at the ICCV 2025 Workshop. For more information, please visit our project website: https://strefer.github.io/
    </div>
    <details class="paper-abstract">
      Next-generation AI companions must go beyond general video understanding to resolve spatial and temporal references in dynamic, real-world environments. Existing Video Large Language Models (Video LLMs), while capable of coarse-level comprehension, struggle with fine-grained, spatiotemporal reasoning, especially when user queries rely on time-based event references for temporal anchoring, or gestural cues for spatial anchoring to clarify object references and positions. To bridge this critical gap, we introduce Strefer, a synthetic instruction data generation framework designed to equip Video LLMs with spatiotemporal referring and reasoning capabilities. Strefer produces diverse instruction-tuning data using a data engine that pseudo-annotates temporally dense, fine-grained video metadata, capturing rich spatial and temporal information in a structured manner, including subjects, objects, their locations as masklets, and their action descriptions and timelines. Our approach enhances the ability of Video LLMs to interpret spatial and temporal references, fostering more versatile, space-time-aware reasoning essential for real-world AI companions. Without using proprietary models, costly human annotation, or the need to annotate large volumes of new videos, experimental evaluations show that models trained with data produced by Strefer outperform baselines on tasks requiring spatial and temporal disambiguation. Additionally, these models exhibit enhanced space-time-aware reasoning, establishing a new foundation for perceptually grounded, instruction-tuned Video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03493v1">On Entropy Control in LLM-RL Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12100v2">LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for cybersecurity threat analysis, but their deployment in security-sensitive environments raises trust and safety concerns. With over 21,000 vulnerabilities disclosed in 2025, manual analysis is infeasible, making scalable and verifiable AI support critical. When querying LLMs, dealing with emerging vulnerabilities is challenging as they have a training cut-off date. While Retrieval-Augmented Generation (RAG) can inject up-to-date context to alleviate the cut-off date limitation, it remains unclear how much LLMs rely on retrieved evidence versus the model's internal knowledge, and whether the retrieved information is meaningful or even correct. This uncertainty could mislead security analysts, mis-prioritize patches, and increase security risks. Therefore, this work proposes LLM Embedding-based Attribution (LEA) to analyze the generated responses for vulnerability exploitation analysis. More specifically, LEA quantifies the relative contribution of internal knowledge vs. retrieved content in the generated responses. We evaluate LEA on 500 critical vulnerabilities disclosed between 2016 and 2025, across three RAG settings -- valid, generic, and incorrect -- using three state-of-the-art LLMs. Our results demonstrate LEA's ability to detect clear distinctions between non-retrieval, generic-retrieval, and valid-retrieval scenarios with over 95% accuracy on larger models. Finally, we demonstrate the limitations posed by incorrect retrieval of vulnerability information and raise a cautionary note to the cybersecurity community regarding the blind reliance on LLMs and RAG for vulnerability analysis. LEA offers security analysts with a metric to audit RAG-enhanced workflows, improving the transparent and trustworthy deployment of AI in cybersecurity threat analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03463v1">The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong potential for automating the generation of models from natural-language descriptions. A common approach is an iterative generate-critique-refine loop, where candidate models are produced, evaluated, and updated based on detected issues. This process needs to address: (1) structural correctness - compliance with well-formedness rules - and (2) semantic alignment - accurate reflection of the intended meaning in the source text. We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are always performed by an LLM. We design five ablated variants of LADEX to study: (i) the impact of the critique-refine loop itself, (ii) the role of LLM-based semantic checks, and (iii) the comparative effectiveness of algorithmic versus LLM-based structural checks. To evaluate LADEX, we compare the generated activity diagrams with expert-created ground truths using trace-based operational semantics. This enables automated measurement of correctness and completeness. Experiments on two datasets indicate that: (1) the critique-refine loop improves structural validity, correctness, and completeness compared to single-pass generation; (2) algorithmic structural checks eliminate inconsistencies that LLM-based checks fail to detect, improving correctness by an average of 17.81% and completeness by 13.24% over LLM-only checks; and (3) combining algorithmic structural checks with LLM-based semantic checks, implemented using the reasoning-focused O4 Mini, achieves the best overall performance - yielding average correctness of up to 86.37% and average completeness of up to 88.56% - while requiring fewer than five LLM calls on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10978v2">Group-in-Group Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on two challenging agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals and achieves performance gains of > 12\% on ALFWorld and > 9\% on WebShop over the GRPO baseline: all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18179v2">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 accepted at EMNLP'25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03419v1">Curse of Knowledge: When Complex Evaluation Context Benefits yet Biases LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 8 pages, 4 figures, conference
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow more capable, they face increasingly diverse and complex tasks, making reliable evaluation challenging. The paradigm of LLMs as judges has emerged as a scalable solution, yet prior work primarily focuses on simple settings. Their reliability in complex tasks--where multi-faceted rubrics, unstructured reference answers, and nuanced criteria are critical--remains understudied. In this paper, we constructed ComplexEval, a challenge benchmark designed to systematically expose and quantify Auxiliary Information Induced Biases. We systematically investigated and validated 6 previously unexplored biases across 12 basic and 3 advanced scenarios. Key findings reveal: (1) all evaluated models exhibit significant susceptibility to these biases, with bias magnitude scaling with task complexity; (2) notably, Large Reasoning Models (LRMs) show paradoxical vulnerability. Our in-depth analysis offers crucial insights for improving the accuracy and verifiability of evaluation signals, paving the way for more general and robust evaluation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00096v2">Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Accepted to EMNLP2025 findings (poster)
    </div>
    <details class="paper-abstract">
      Neural network pruning has emerged as a promising approach for deploying LLMs in low-resource scenarios while preserving downstream task performance. However, for the first time, we reveal that such pruning disrupts LLMs' internal activation features crucial for lie detection, where probing classifiers (typically small logistic regression models) trained on these features assess the truthfulness of LLM-generated statements. This discovery raises a crucial open question: how can we prune LLMs without sacrificing these critical lie detection capabilities? Our investigation further reveals that naively adjusting layer-wise pruning sparsity based on importance inadvertently removes crucial weights, failing to improve lie detection performance despite its reliance on the most crucial LLM layer. To address this issue, we propose Truthful Pruning aligned by Layer-wise Outliers (TPLO), which places greater emphasis on layers with more activation outliers and stronger discriminative features simultaneously. This preserves LLMs' original performance while retaining critical features of inner states needed for robust lie detection. Moreover, we introduce a prompting rule to enrich the TruthfulQA benchmark for better calibrating LLM pruning. Empirical results show that our approach improves the hallucination detection for pruned LLMs (achieving 88% accuracy at 50% sparsity) and enhances their performance on TruthfulQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00079v4">Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 23 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latency-related Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle 75% more requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03377v1">Amplifying Effective CXL Memory Bandwidth for LLM Inference via Transparent Near-Data Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference is bottlenecked by the limited bandwidth of CXL-based memory used for capacity expansion. We introduce CXL-NDP, a transparent near-data processing architecture that amplifies effective CXL bandwidth without requiring changes to the CXL.mem interface or AI models. CXL-NDP integrates a precision-scalable bit-plane layout for dynamic quantization with transparent lossless compression of weights and KV caches directly within the CXL device. In end-to-end serving, CXL-NDP improves throughput by 43%, extends the maximum context length by 87%, and reduces the KV cache footprint by 46.9% without accuracy loss. Hardware synthesis confirms its practicality with a modest silicon footprint, lowering the barrier for adopting efficient, scalable CXL-based memory in generative AI infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v5">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Determining the dynamic data dependency of a step that reads a variable $v$ is challenging. It typically requires either exhaustive instrumentation, which becomes prohibitively expensive when $v$ is defined within library calls, or repeated executions, which are impractical for non-deterministic programs. In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable. Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We evaluate RecovSlicing on 8300 data dependencies across three slicing benchmarks, comparing it with Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer. RecovSlicing achieves significantly higher accuracy (80.3%, 91.1%, 98.3%) and recall (up to 98.3%) than the best baseline (accuracy: 39.0%, 82.0%, 59.9%; recall: 53.4%, 79.1%, 87.1%). Integrated into a dual-slicing regression bug localizer, it identifies 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04387v2">FedP$^2$EFT: Federated Learning to Personalize PEFT for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Federated learning (FL) has enabled the training of multilingual large language models (LLMs) on diverse and decentralized multilingual data, especially on low-resource languages. To improve client-specific performance, personalization via the use of parameter-efficient fine-tuning (PEFT) modules such as LoRA is common. This involves a personalization strategy (PS), such as the design of the PEFT adapter structures (e.g., in which layers to add LoRAs and what ranks) and choice of hyperparameters (e.g., learning rates) for fine-tuning. Instead of manual PS configuration, we propose FedP$^2$EFT, a federated learning-to-personalize method for multilingual LLMs in cross-device FL settings. Unlike most existing PEFT structure selection methods, which are prone to overfitting low-data regimes, FedP$^2$EFT collaboratively learns the optimal personalized PEFT structure for each client via Bayesian sparse rank selection. Evaluations on both simulated and real-world multilingual FL benchmarks demonstrate that FedP$^2$EFT largely outperforms existing personalized fine-tuning methods, while complementing other existing FL methods. Code is available at https://github.com/SamsungLabs/fedp2eft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03312v1">AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems, often comprising multiple models, complex tool invocations, and orchestration protocols, substantially outperform monolithic agents. Yet this very sophistication amplifies their fragility, making them more prone to system failure. Pinpointing the specific agent or step responsible for an error within long execution traces defines the task of agentic system failure attribution. Current state-of-the-art reasoning LLMs, however, remain strikingly inadequate for this challenge, with accuracy generally below 10%. To address this gap, we propose AgenTracer, the first automated framework for annotating failed multi-agent trajectories via counterfactual replay and programmed fault injection, producing the curated dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a lightweight failure tracer trained with multi-granular reinforcement learning, capable of efficiently diagnosing errors in verbose multi-agent interactions. On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS with 4.8-14.2% performance gains, empowering self-correcting and self-evolving agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05225v2">QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Accepted by EMNLP 2025 Main Conference. Homepage: https://github.com/mengze-hong/QualBench
    </div>
    <details class="paper-abstract">
      The rapid advancement of Chinese LLMs underscores the need for vertical-domain evaluations to ensure reliable applications. However, existing benchmarks often lack domain coverage and provide limited insights into the Chinese working context. Leveraging qualification exams as a unified framework for expertise evaluation, we introduce QualBench, the first multi-domain Chinese QA benchmark dedicated to localized assessment of Chinese LLMs. The dataset includes over 17,000 questions across six vertical domains, drawn from 24 Chinese qualifications to align with national policies and professional standards. Results reveal an interesting pattern of Chinese LLMs consistently surpassing non-Chinese models, with the Qwen2.5 model outperforming the more advanced GPT-4o, emphasizing the value of localized domain knowledge in meeting qualification requirements. The average accuracy of 53.98% reveals the current gaps in domain coverage within model capabilities. Furthermore, we identify performance degradation caused by LLM crowdsourcing, assess data contamination, and illustrate the effectiveness of prompt engineering and model fine-tuning, suggesting opportunities for future improvements through multi-domain RAG and Federated Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09049v4">Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Discovering customer intentions is crucial for automated service agents, yet existing intent clustering methods often fall short due to their reliance on embedding distance metrics and neglect of underlying semantic structures. To address these limitations, we propose an LLM-in-the-loop (LLM-ITL) intent clustering framework, integrating the language understanding capabilities of LLMs into conventional clustering algorithms. Specifically, this paper (1) examines the effectiveness of fine-tuned LLMs in semantic coherence evaluation and intent cluster naming, achieving over 95% accuracy aligned with human judgments; (2) designs an LLM-ITL framework that facilitates the iterative discovery of coherent intent clusters and the optimal number of clusters; and (3) introduces context-aware techniques tailored for customer service dialogue. Since existing English benchmarks lack sufficient semantic diversity and intent coverage, we further present a comprehensive Chinese dialogue intent dataset comprising over 100k real customer service calls with 1,507 human-annotated clusters. The proposed approaches significantly outperform LLM-guided baselines, achieving notable improvements in clustering quality, cost efficiency, and downstream applications. Combined with several best practices, our findings highlight the prominence of LLM-in-the-loop techniques for scalable dialogue data mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03161v1">Domain Adaptation of LLMs for Process Data</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have emerged as a prominent area of interest across various research domains, including Process Mining (PM). Current applications in PM have predominantly centered on prompt engineering strategies or the transformation of event logs into narrative-style datasets, thereby exploiting the semantic capabilities of LLMs to address diverse tasks. In contrast, this study investigates the direct adaptation of pretrained LLMs to process data without natural language reformulation, motivated by the fact that these models excel in generating sequences of tokens, similar to the objective in PM. More specifically, we focus on parameter-efficient fine-tuning techniques to mitigate the computational overhead typically associated with such models. Our experimental setup focuses on Predictive Process Monitoring (PPM), and considers both single- and multi-task predictions. The results demonstrate a potential improvement in predictive performance over state-of-the-art recurrent neural network (RNN) approaches and recent narrative-style-based solutions, particularly in the multi-task setting. Additionally, our fine-tuned models exhibit faster convergence and require significantly less hyperparameter optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03117v1">PromptCOS: Towards System Prompt Copyright Auditing for LLMs via Content-level Output Similarity</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) has greatly enhanced reasoning tasks and facilitated the development of LLM-based applications. A critical factor in improving LLM-based applications is the design of effective system prompts, which significantly impact the behavior and output quality of LLMs. However, system prompts are susceptible to theft and misuse, which could undermine the interests of prompt owners. Existing methods protect prompt copyrights through watermark injection and verification but face challenges due to their reliance on intermediate LLM outputs (e.g., logits), which limits their practical feasibility. In this paper, we propose PromptCOS, a method for auditing prompt copyright based on content-level output similarity. It embeds watermarks by optimizing the prompt while simultaneously co-optimizing a special verification query and content-level signal marks. This is achieved by leveraging cyclic output signals and injecting auxiliary tokens to ensure reliable auditing in content-only scenarios. Additionally, it incorporates cover tokens to protect the watermark from malicious deletion. For copyright verification, PromptCOS identifies unauthorized usage by comparing the similarity between the suspicious output and the signal mark. Experimental results demonstrate that our method achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% greater than the best baseline), high fidelity (accuracy degradation of no more than 0.58%), robustness (resilience against three types of potential attacks), and computational efficiency (up to 98.1% reduction in computational cost). Our code is available at GitHub https://github.com/LianPing-cyber/PromptCOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03116v1">Measuring Scalar Constructs in Social Science with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Accepted to EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Many constructs that characterize language, like its complexity or emotionality, have a naturally continuous semantic structure; a public speech is not just "simple" or "complex," but exists on a continuum between extremes. Although large language models (LLMs) are an attractive tool for measuring scalar constructs, their idiosyncratic treatment of numerical outputs raises questions of how to best apply them. We address these questions with a comprehensive evaluation of LLM-based approaches to scalar construct measurement in social science. Using multiple datasets sourced from the political science literature, we evaluate four approaches: unweighted direct pointwise scoring, aggregation of pairwise comparisons, token-probability-weighted pointwise scoring, and finetuning. Our study yields actionable findings for applied researchers. First, LLMs prompted to generate pointwise scores directly from texts produce discontinuous distributions with bunching at arbitrary numbers. The quality of the measurements improves with pairwise comparisons made by LLMs, but it improves even more by taking pointwise scores and weighting them by token probability. Finally, finetuning smaller models with as few as 1,000 training pairs can match or exceed the performance of prompted LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17642v2">Banishing LLM Hallucinations Requires Rethinking Generalization</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 I want to revisit some of the experiments in this paper, specifically figure 5
    </div>
    <details class="paper-abstract">
      Despite their powerful chat, coding, and reasoning abilities, Large Language Models (LLMs) frequently hallucinate. Conventional wisdom suggests that hallucinations are a consequence of a balance between creativity and factuality, which can be mitigated, but not eliminated, by grounding the LLM in external knowledge sources. Through extensive systematic experiments, we show that these traditional approaches fail to explain why LLMs hallucinate in practice. Specifically, we show that LLMs augmented with a massive Mixture of Memory Experts (MoME) can easily memorize large datasets of random numbers. We corroborate these experimental findings with a theoretical construction showing that simple neural networks trained to predict the next token hallucinate when the training loss is above a threshold as it usually does in practice when training on internet scale data. We interpret our findings by comparing against traditional retrieval methods for mitigating hallucinations. We use our findings to design a first generation model for removing hallucinations -- Lamini-1 -- that stores facts in a massive mixture of millions of memory experts that are retrieved dynamically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02515v1">Contemporary Agent Technology: LLM-Driven Advancements vs Classic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 The paper has 33 pages and it contains 1 figure and 2 tables
    </div>
    <details class="paper-abstract">
      This contribution provides our comprehensive reflection on the contemporary agent technology, with a particular focus on the advancements driven by Large Language Models (LLM) vs classic Multi-Agent Systems (MAS). It delves into the models, approaches, and characteristics that define these new systems. The paper emphasizes the critical analysis of how the recent developments relate to the foundational MAS, as articulated in the core academic literature. Finally, it identifies key challenges and promising future directions in this rapidly evolving domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02494v1">GridMind: LLMs-Powered Agents for Power System Analysis and Operations</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 11 pages, 9 figures, 2 tables. Work under review
    </div>
    <details class="paper-abstract">
      The complexity of traditional power system analysis workflows presents significant barriers to efficient decision-making in modern electric grids. This paper presents GridMind, a multi-agent AI system that integrates Large Language Models (LLMs) with deterministic engineering solvers to enable conversational scientific computing for power system analysis. The system employs specialized agents coordinating AC Optimal Power Flow and N-1 contingency analysis through natural language interfaces while maintaining numerical precision via function calls. GridMind addresses workflow integration, knowledge accessibility, context preservation, and expert decision-support augmentation. Experimental evaluation on IEEE test cases demonstrates that the proposed agentic framework consistently delivers correct solutions across all tested language models, with smaller LLMs achieving comparable analytical accuracy with reduced computational latency. This work establishes agentic AI as a viable paradigm for scientific computing, demonstrating how conversational interfaces can enhance accessibility while preserving numerical rigor essential for critical engineering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02480v1">MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training to Break the GPU Memory Wall</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 SC'25: The International Conference for High Performance Computing, Networking, Storage and Analysis
    </div>
    <details class="paper-abstract">
      Training LLMs larger than the aggregated memory of multiple GPUs is increasingly necessary due to the faster growth of LLM sizes compared to GPU memory. To this end, multi-tier host memory or disk offloading techniques are proposed by state of art. Despite advanced asynchronous multi-tier read/write strategies, such offloading strategies result in significant I/O overheads in the critical path of training, resulting in slower iterations. To this end, we propose MLP-Offload, a novel multi-level, multi-path offloading engine specifically designed for optimizing LLM training on resource-constrained setups by mitigating I/O bottlenecks. We make several key observations that drive the design of MLP-Offload, such as I/O overheads during the update dominate the iteration time; I/O bandwidth of the third-level remote storage tier remains unutilized; and, contention due to concurrent offloading amplifies I/O bottlenecks. Driven by these insights, we design and implement MLP-Offload to offload the optimizer states across multiple tiers in a cache-efficient and concurrency-controlled fashion to mitigate I/O bottlenecks during the backward and update phases. Evaluations on models up to 280B parameters shows that MLP-Offload achieves 2.5$\times$ faster iterations compared to the state-of-the-art LLM training runtimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02452v1">Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 To appear in EMNLP 2025, Main Conference
    </div>
    <details class="paper-abstract">
      Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02449v1">KubeIntellect: A Modular LLM-Orchestrated Agent Framework for End-to-End Kubernetes Management</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Kubernetes has become the foundation of modern cloud-native infrastructure, yet its management remains complex and fragmented. Administrators must navigate a vast API surface, manage heterogeneous workloads, and coordinate tasks across disconnected tools - often requiring precise commands, YAML configuration, and contextual expertise. This paper presents KubeIntellect, a Large Language Model (LLM)-powered system for intelligent, end-to-end Kubernetes control. Unlike existing tools that focus on observability or static automation, KubeIntellect supports natural language interaction across the full spectrum of Kubernetes API operations, including read, write, delete, exec, access control, lifecycle, and advanced verbs. The system uses modular agents aligned with functional domains (e.g., logs, metrics, RBAC), orchestrated by a supervisor that interprets user queries, maintains workflow memory, invokes reusable tools, or synthesizes new ones via a secure Code Generator Agent. KubeIntellect integrates memory checkpoints, human-in-the-loop clarification, and dynamic task sequencing into a structured orchestration framework. Evaluation results show a 93% tool synthesis success rate and 100% reliability across 200 natural language queries, demonstrating the system's ability to operate efficiently under diverse workloads. An automated demo environment is provided on Azure, with additional support for local testing via kind. This work introduces a new class of interpretable, extensible, and LLM-driven systems for managing complex infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02408v1">Cache Management for Mixture-of-Experts LLMs -- extended version</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a variety of tasks. One of the main challenges towards the successful deployment of LLMs is memory management, since they typically involve billions of parameters. To this end, architectures based on Mixture-of-Experts have been proposed, which aim to reduce the size of the parameters that are activated when producing a token. This raises the equally critical issue of efficiently managing the limited cache of the system, in that frequently used experts should be stored in the fast cache rather than in the slower secondary memory. In this work, we introduce and study a new paging problem that models expert management optimization. Our formulation captures both the layered architecture of LLMs and the requirement that experts are cached efficiently. We first present lower bounds on the competitive ratio of both deterministic and randomized algorithms, which show that under mild assumptions, LRU-like policies have good theoretical competitive performance. We then propose a layer-based extension of LRU that is tailored to the problem at hand. Extensive simulations on both synthetic datasets and actual traces of MoE usage show that our algorithm outperforms policies for the classic paging problem, such as the standard LRU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02655v1">BioBlue: Notable runaway-optimiser-like LLM failure modes on biologically and economically aligned AI safety benchmarks for LLMs with simplified observation format</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 13 pages, 8 tables
    </div>
    <details class="paper-abstract">
      Relatively many past AI safety discussions have centered around the dangers of unbounded utility maximisation by RL agents, illustrated by scenarios like the "paperclip maximiser" or by specification gaming in general. Unbounded maximisation is problematic for many reasons. We wanted to verify whether these RL runaway optimisation problems are still relevant with LLMs as well. Turns out, strangely, this is indeed clearly the case. The problem is not that the LLMs just lose context or become incoherent. The problem is that in various scenarios, LLMs lose context in very specific ways, which systematically resemble runaway optimisers in the following distinct ways: 1) Ignoring homeostatic targets and "defaulting" to unbounded maximisation instead. 2) It is equally concerning that the "default" meant also reverting back to single-objective optimisation. Our findings also suggest that long-running scenarios are important. Systematic failures emerge after periods of initially successful behaviour. In some trials the LLMs were successful until the end. This means, while current LLMs do conceptually grasp biological and economic alignment, they exhibit randomly triggered problematic behavioural tendencies under sustained long-running conditions, particularly involving multiple or competing objectives. Once they flip, they usually do not recover. Even though LLMs look multi-objective and bounded on the surface, the underlying mechanisms seem to be actually still biased towards being single-objective and unbounded.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15567v3">Intelligent Assistants for the Semiconductor Failure Analysis with LLM-Based Planning Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 This technical report provides evaluation details of the experiments presented in the paper accepted to ISTFA 2025
    </div>
    <details class="paper-abstract">
      Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process. This paper investigates the design and implementation of an agentic AI system for semiconductor FA using a Large Language Model (LLM)-based Planning Agent (LPA). The LPA integrates LLMs with advanced planning capabilities and external tool utilization, allowing autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. The evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02372v1">Poisoned at Scale: A Scalable Audit Uncovers Hidden Scam Endpoints in Production LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become critical to modern software development, but their reliance on internet datasets for training introduces a significant security risk: the absorption and reproduction of malicious content. To evaluate this threat, this paper introduces a scalable, automated audit framework that synthesizes innocuous, developer-style prompts from known scam databases to query production LLMs and determine if they generate code containing harmful URLs. We conducted a large-scale evaluation across four production LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3), and found a systemic vulnerability, with all tested models generating malicious code at a non-negligible rate. On average, 4.2\% of programs generated in our experiments contained malicious URLs. Crucially, this malicious code is often generated in response to benign prompts. We manually validate the prompts which cause all four LLMs to generate malicious code, and resulting in 177 innocuous prompts that trigger all models to produce harmful outputs. These results provide strong empirical evidence that the training data of production LLMs has been successfully poisoned at scale, underscoring the urgent need for more robust defense mechanisms and post-generation safety checks to mitigate the propagation of hidden security threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19516v2">Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Modern LLM serving systems confront inefficient GPU utilization due to the fundamental mismatch between compute-intensive prefill and memory-bound decode phases. While current practices attempt to address this by organizing these phases into hybrid batches, such solutions create an inefficient tradeoff that sacrifices either throughput or latency, leaving substantial GPU resources underutilized. We identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks. 2) hybrid batches disproportionately prioritize latency over throughput, resulting in wasted compute and memory bandwidth. To mitigate the issues, we present Bullet, a novel spatial-temporal orchestration system that eliminates these inefficiencies through precise phase coordination. Bullet enables concurrent execution of prefill and decode phases, while dynamically provisioning GPU resources using real-time performance modeling. By integrating SLO-aware scheduling and adaptive resource allocation, Bullet maximizes utilization without compromising latency targets. Experimental evaluations on real-world workloads demonstrate that Bullet delivers 1.26x average throughput gains (up to 1.55x) over state-of-the-arts, while consistently meeting latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02330v1">ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Accepted by CIKM 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated impressive capabilities in code-related tasks, such as code generation and automated program repair. Despite their promising performance, most existing approaches for code repair suffer from high training costs or computationally expensive inference. Retrieval-augmented generation (RAG), with its efficient in-context learning paradigm, offers a more scalable alternative. However, conventional retrieval strategies, which are often based on holistic code-text embeddings, fail to capture the structural intricacies of code, resulting in suboptimal retrieval quality. To address the above limitations, we propose ReCode, a fine-grained retrieval-augmented in-context learning framework designed for accurate and efficient code repair. Specifically, ReCode introduces two key innovations: (1) an algorithm-aware retrieval strategy that narrows the search space using preliminary algorithm type predictions; and (2) a modular dual-encoder architecture that separately processes code and textual inputs, enabling fine-grained semantic matching between input and retrieved contexts. Furthermore, we propose RACodeBench, a new benchmark constructed from real-world user-submitted buggy code, which addresses the limitations of synthetic benchmarks and supports realistic evaluation. Experimental results on RACodeBench and competitive programming datasets demonstrate that ReCode achieves higher repair accuracy with significantly reduced inference cost, highlighting its practical value for real-world code repair scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02324v1">Language-Guided Long Horizon Manipulation with LLM-based Planning and Visual Perception</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Language-guided long-horizon manipulation of deformable objects presents significant challenges due to high degrees of freedom, complex dynamics, and the need for accurate vision-language grounding. In this work, we focus on multi-step cloth folding, a representative deformable-object manipulation task that requires both structured long-horizon planning and fine-grained visual perception. To this end, we propose a unified framework that integrates a Large Language Model (LLM)-based planner, a Vision-Language Model (VLM)-based perception system, and a task execution module. Specifically, the LLM-based planner decomposes high-level language instructions into low-level action primitives, bridging the semantic-execution gap, aligning perception with action, and enhancing generalization. The VLM-based perception module employs a SigLIP2-driven architecture with a bidirectional cross-attention fusion mechanism and weight-decomposed low-rank adaptation (DoRA) fine-tuning to achieve language-conditioned fine-grained visual grounding. Experiments in both simulation and real-world settings demonstrate the method's effectiveness. In simulation, it outperforms state-of-the-art baselines by 2.23, 1.87, and 33.3 on seen instructions, unseen instructions, and unseen tasks, respectively. On a real robot, it robustly executes multi-step folding sequences from language instructions across diverse cloth materials and configurations, demonstrating strong generalization in practical scenarios. Project page: https://language-guided.netlify.app/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02297v1">Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      The art of heuristic design has traditionally been a human pursuit. While Large Language Models (LLMs) can generate code for search heuristics, their application has largely been confined to adjusting simple functions within human-crafted frameworks, leaving their capacity for broader innovation an open question. To investigate this, we tasked an LLM with building a complete solver for the constrained 3D Packing Problem. Direct code generation quickly proved fragile, prompting us to introduce two supports: constraint scaffolding--prewritten constraint-checking code--and iterative self-correction--additional refinement cycles to repair bugs and produce a viable initial population. Notably, even within a vast search space in a greedy process, the LLM concentrated its efforts almost exclusively on refining the scoring function. This suggests that the emphasis on scoring functions in prior work may reflect not a principled strategy, but rather a natural limitation of LLM capabilities. The resulting heuristic was comparable to a human-designed greedy algorithm, and when its scoring function was integrated into a human-crafted metaheuristic, its performance rivaled established solvers, though its effectiveness waned as constraints tightened. Our findings highlight two major barriers to automated heuristic design with current LLMs: the engineering required to mitigate their fragility in complex reasoning tasks, and the influence of pretrained biases, which can prematurely narrow the search for novel solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02292v1">LLMs and their Limited Theory of Mind: Evaluating Mental State Annotations in Situated Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      What if large language models could not only infer human mindsets but also expose every blind spot in team dialogue such as discrepancies in the team members' joint understanding? We present a novel, two-step framework that leverages large language models (LLMs) both as human-style annotators of team dialogues to track the team's shared mental models (SMMs) and as automated discrepancy detectors among individuals' mental states. In the first step, an LLM generates annotations by identifying SMM elements within task-oriented dialogues from the Cooperative Remote Search Task (CReST) corpus. Then, a secondary LLM compares these LLM-derived annotations and human annotations against gold-standard labels to detect and characterize divergences. We define an SMM coherence evaluation framework for this use case and apply it to six CReST dialogues, ultimately producing: (1) a dataset of human and LLM annotations; (2) a reproducible evaluation framework for SMM coherence; and (3) an empirical assessment of LLM-based discrepancy detection. Our results reveal that, although LLMs exhibit apparent coherence on straightforward natural-language annotation tasks, they systematically err in scenarios requiring spatial reasoning or disambiguation of prosodic cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02241v1">LLMs for LLMs: A Structured Prompting Methodology for Long Legal Documents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 20 pages, 6 figures, 4 tables,
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has had a profoundly transformative effect on a number of fields and domains. However, their uptake in Law has proven more challenging due to the important issues of reliability and transparency. In this study, we present a structured prompting methodology as a viable alternative to the often expensive fine-tuning, with the capability of tacking long legal documents from the CUAD dataset on the task of information retrieval. Each document is first split into chunks via a system of chunking and augmentation, addressing the long document problem. Then, alongside an engineered prompt, the input is fed into QWEN-2 to produce a set of answers for each question. Finally, we tackle the resulting candidate selection problem with the introduction of the Distribution-based Localisation and Inverse Cardinality Weighting heuristics. This approach leverages a general purpose model to promote long term scalability, prompt engineering to increase reliability and the two heuristic strategies to reduce the impact of the black box effect. Whilst our model performs up to 9\% better than the previously presented method, reaching state-of-the-art performance, it also highlights the limiting factor of current automatic evaluation metrics for question answering, serving as a call to action for future research. However, the chief aim of this work is to underscore the potential of structured prompt engineering as a useful, yet under-explored, tool in ensuring accountability and responsibility of AI in the legal domain, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02198v1">FActBench: A Benchmark for Fine-grained Automatic Evaluation of LLM-Generated Text in the Medical Domain</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large Language Models tend to struggle when dealing with specialized domains. While all aspects of evaluation hold importance, factuality is the most critical one. Similarly, reliable fact-checking tools and data sources are essential for hallucination mitigation. We address these issues by providing a comprehensive Fact-checking Benchmark FActBench covering four generation tasks and six state-of-the-art Large Language Models (LLMs) for the Medical domain. We use two state-of-the-art Fact-checking techniques: Chain-of-Thought (CoT) Prompting and Natural Language Inference (NLI). Our experiments show that the fact-checking scores acquired through the Unanimous Voting of both techniques correlate best with Domain Expert Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12918v4">Query Rewriting via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      When complex SQL queries suffer slow executions despite query optimization, DBAs typically invoke automated query rewriting tools to recommend ``lean'' equivalents that are conducive to faster execution. The rewritings are usually achieved via transformation rules, but these rules are limited in scope and difficult to update in a production system. Recently, LLM-based techniques have also been suggested, but they are prone to semantic and syntactic errors. We investigate here how the remarkable cognitive capabilities of LLMs can be leveraged for performant query rewriting while incorporating safeguards and optimizations to ensure correctness and efficiency. Our study shows that these goals can be progressively achieved through incorporation of (a) an ensemble suite of basic prompts, (b) database-sensitive prompts via redundancy removal and selectivity-based rewriting rules, and (c) LLM token probability-guided rewrite paths. Further, a suite of logic-based and statistical tools can be used to check for semantic violations in the rewrites prior to DBA consideration. We have implemented the above LLM-infused techniques in the LITHE system, and evaluated complex analytic queries from standard benchmarks on contemporary database platforms. The results show significant performance improvements for slow queries, over both SOTA rewriters and the native optimizer. For instance, with TPC-DS on PostgreSQL, the GM of runtime speedups was a high 13.2 over the native optimizer, whereas SOTA only gave 4.9. Overall, LITHE is a promising step toward viable LLM-based advisory tools for ameliorating enterprise query performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02163v1">Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09012v3">Multimodal LLMs Can Reason about Aesthetics in Zero-Shot</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 ACM MM 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      The rapid technical progress of generative art (GenArt) has democratized the creation of visually appealing imagery. However, achieving genuine artistic impact - the kind that resonates with viewers on a deeper, more meaningful level - remains formidable as it requires a sophisticated aesthetic sensibility. This sensibility involves a multifaceted cognitive process extending beyond mere visual appeal, which is often overlooked by current computational methods. This paper pioneers an approach to capture this complex process by investigating how the reasoning capabilities of Multimodal LLMs (MLLMs) can be effectively elicited to perform aesthetic judgment. Our analysis reveals a critical challenge: MLLMs exhibit a tendency towards hallucinations during aesthetic reasoning, characterized by subjective opinions and unsubstantiated artistic interpretations. We further demonstrate that these hallucinations can be suppressed by employing an evidence-based and objective reasoning process, as substantiated by our proposed baseline, ArtCoT. MLLMs prompted by this principle produce multifaceted, in-depth aesthetic reasoning that aligns significantly better with human judgment. These findings have direct applications in areas such as AI art tutoring and as reward models for image generation. Ultimately, we hope this work paves the way for AI systems that can truly understand, appreciate, and contribute to art that aligns with human aesthetic values. Project homepage: https://github.com/songrise/MLLM4Art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02097v1">JudgeAgent: Dynamically Evaluate LLMs with Agent-as-Interviewer</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities of large language models (LLMs) is an essential step to ensure the successful application of LLMs across various domains. The current evaluation of LLMs is based on a paradigm that involves querying them with predefined question sets and assessing their outputs. This paradigm offers controllable processes and simplicity, but faces challenges such as limited interaction with targets, insufficient difficulty control, and difficulties in verifying the validity of evaluation results, making it hard to precisely determine the knowledge and capability boundaries of target models. To address these challenges, we propose JudgeAgent, a knowledge-target adaptive dynamic evaluation framework based on a new interviewer-style evaluation paradigm. JudgeAgent employs a comprehensive evaluation approach consisting of benchmark grading, interactive extension, and evaluation feedback. It utilizes knowledge-driven data synthesis and target-adaptive difficulty adjustment methods to conduct extended testing, providing accurate and effective evaluation results. We also introduce a novel insight into validating evaluation methods, demonstrating the effectiveness of JudgeAgent and its dynamic evaluation paradigm through extensive experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14314v3">Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 accepted by ACL'2025
    </div>
    <details class="paper-abstract">
      Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at https://read-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02040v1">Attributes as Textual Genes: Leveraging LLMs as Genetic Algorithm Simulators for Conditional Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Accepted to EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at generating synthetic data, but ensuring its quality and diversity remains challenging. We propose Genetic Prompt, a novel framework that combines genetic algorithms with LLMs to augment synthetic data generation. Our approach treats semantic text attributes as gene sequences and leverages the LLM to simulate crossover and mutation operations. This genetic process enhances data quality and diversity by creating novel attribute combinations, yielding synthetic distributions closer to real-world data. To optimize parent selection, we also integrate an active learning scheme that expands the offspring search space. Our experiments on multiple NLP tasks reveal several key findings: Genetic Prompt not only significantly outperforms state-of-the-art baselines but also shows robust performance across various generator model sizes and scales. Moreover, we demonstrate that fusing our synthetic data with the original training set significantly boosts downstream model performance, particularly for class-imbalanced scenarios. Our findings validate that Genetic Prompt is an effective method for producing high-quality synthetic data for a wide range of NLP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14146v2">MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18190v3">ST-Raptor: LLM-Powered Semi-Structured Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Extension of our SIGMOD 2026 paper. Please refer to source code available at: https://github.com/weAIDB/ST-Raptor
    </div>
    <details class="paper-abstract">
      Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17967v2">Agent Trading Arena: A Study on Numerical Understanding in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language tasks, yet their performance in dynamic, real-world financial environments remains underexplored. Existing approaches are limited to historical backtesting, where trading actions cannot influence market prices and agents train only on static data. To address this limitation, we present the Agent Trading Arena, a virtual zero-sum stock market in which LLM-based agents engage in competitive multi-agent trading and directly impact price dynamics. By simulating realistic bid-ask interactions, our platform enables training in scenarios that closely mirror live markets, thereby narrowing the gap between training and evaluation. Experiments reveal that LLMs struggle with numerical reasoning when given plain-text data, often overfitting to local patterns and recent values. In contrast, chart-based visualizations significantly enhance both numerical reasoning and trading performance. Furthermore, incorporating a reflection module yields additional improvements, especially with visual inputs. Evaluations on NASDAQ and CSI datasets demonstrate the superiority of our method, particularly under high volatility. All code and data are available at https://github.com/wekjsdvnm/Agent-Trading-Arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02896v1">Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 To appear in SIGMOD'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly used as a building block in data systems to process large text datasets. To do so, LLM model providers offer multiple LLMs with different sizes, spanning various cost-quality trade-offs when processing text at scale. Top-of-the-line LLMs (e.g., GPT-4o, Claude Sonnet) operate with high accuracy but are prohibitively expensive when processing many records. To avoid high costs, more affordable but lower quality LLMs (e.g., GPT-4o-mini, Claude Haiku) can be used to process records, but we need to ensure that the overall accuracy does not deviate substantially from that of the top-of-the-line LLMs. The model cascade framework provides a blueprint to manage this trade-off, by using the confidence of LLMs in their output (e.g., log-probabilities) to decide on which records to use the affordable LLM. However, existing solutions following this framework provide only marginal cost savings and weak theoretical guarantees because of poor estimation of the quality of the affordable LLM's outputs. We present BARGAIN, a method that judiciously uses affordable LLMs in data processing to significantly reduce cost while providing strong theoretical guarantees on the solution quality. BARGAIN employs a novel adaptive sampling strategy and statistical estimation procedure that uses data and task characteristics and builds on recent statistical tools to make accurate estimations with tight theoretical guarantees. Variants of BARGAIN can support guarantees on accuracy, precision, or recall of the output. Experimental results across 8 real-world datasets show that BARGAIN reduces cost, on average, by up to 86% more than state-of-the-art, while providing stronger theoretical guarantees on accuracy of output, with similar gains when guaranteeing a desired level of precision or recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12232v2">LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Issue-to-commit link recovery plays an important role in software traceability and improves project management. However, it remains a challenging task. A study on GitHub shows that only 42.2% of the issues are correctly linked to their commits. This highlights the potential for further development and research in this area. Existing studies have employed various AI/ML-based approaches, and with the recent development of large language models, researchers have leveraged LLMs to tackle this problem. These approaches suffer from two main issues. First, LLMs are constrained by limited context windows and cannot ingest all of the available data sources, such as long commit histories, extensive issue comments, and large code repositories. Second, most methods operate on individual issue-commit pairs; that is, given a single issue-commit pair, they determine whether the commit resolves the issue. This quickly becomes impractical in real-world repositories containing tens of thousands of commits. To address these limitations, we present LinkAnchor, the first autonomous LLM-based agent designed for issue-to-commit link recovery. The lazy-access architecture of LinkAnchor enables the underlying LLM to access the rich context of software, spanning commits, issue comments, and code files, without exceeding the token limit by dynamically retrieving only the most relevant contextual data. Additionally, LinkAnchor is able to automatically pinpoint the target commit rather than exhaustively scoring every possible candidate. Our evaluations show that LinkAnchor outperforms state-of-the-art issue-to-commit link recovery approaches by 60-262% in Hit@1 score across all our case study projects. We also publicly release LinkAnchor as a ready-to-use tool, along with our replication package. LinkAnchor is designed and tested for GitHub and Jira, and is easily extendable to other platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02890v1">Grocery to General Merchandise: A Cross-Pollination Recommender using LLMs and Real-Time Cart Context</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Modern e-commerce platforms strive to enhance customer experience by providing timely and contextually relevant recommendations. However, recommending general merchandise to customers focused on grocery shopping -- such as pairing milk with a milk frother -- remains a critical yet under-explored challenge. This paper introduces a cross-pollination (XP) framework, a novel approach that bridges grocery and general merchandise cross-category recommendations by leveraging multi-source product associations and real-time cart context. Our solution employs a two-stage framework: (1) A candidate generation mechanism that uses co-purchase market basket analysis and LLM-based approach to identify novel item-item associations; and (2) a transformer-based ranker that leverages the real-time sequential cart context and optimizes for engagement signals such as add-to-carts. Offline analysis and online A/B tests show an increase of 36\% add-to-cart rate with LLM-based retrieval, and 27\% NDCG\@4 lift using cart context-based ranker. Our work contributes practical techniques for cross-category recommendations and broader insights for e-commerce systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02820v1">Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) involves precisely removing specific information from a pre-trained model. This is crucial to ensure safety of LLMs by deleting private data or harmful knowledge acquired during pre-training. However, existing unlearning methods often fall short when subjected to thorough evaluation. To overcome this, we introduce JensUn, where we leverage the Jensen-Shannon Divergence as the training objective for both forget and retain sets for more stable and effective unlearning dynamics compared to commonly used loss functions. In extensive experiments, JensUn achieves better forget-utility trade-off than competing methods, and even demonstrates strong resilience to benign relearning. Additionally, for a precise unlearning evaluation, we introduce LKF, a curated dataset of lesser-known facts that provides a realistic unlearning scenario. Finally, to comprehensively test unlearning methods, we propose (i) employing an LLM as semantic judge instead of the standard ROUGE score, and (ii) using worst-case unlearning evaluation over various paraphrases and input formats. Our improved evaluation framework reveals that many existing methods are less effective than previously thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10059v2">CodeGrad: Integrating Multi-Step Verification with Gradient-Based LLM Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 6 Pages
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, they often produce solutions that lack guarantees of correctness, robustness, and efficiency. This limitation is particularly acute in domains requiring strict constraints. CodeGrad introduces a principled framework that integrates rigorous verification techniques directly into an iterative LLM-based generation loop. It uniquely treats code as a differentiable variable, converting structured feedback and mathematical constraints into a textual pseudo-gradient. This gradient guides the model to iteratively refine solutions, ensuring they are not only functional but also robust and mathematically justified. We evaluate CodeGrad on the HumanEval, HumanEval+, and LiveCodeBench benchmarks. Our implementation outperforms strong baselines, achieving an absolute improvement of up to 27% on HumanEval and a 41% relative improvement on the challenging LiveCodeBench V6. StructuredGrad generates mathematically justified code that is robust and efficient, paving the way for reliable AI-assisted software development in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v1">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02754v1">Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in large language models (LLMs) have not only advanced natural language processing but also inspired their application in domains with structurally similar problems--most notably, autonomous driving motion generation. Both domains involve autoregressive sequence modeling, token-based representations, and context-aware decision making, making the transfer of LLM components a natural and increasingly common practice. However, despite promising early attempts, a systematic understanding of which LLM modules are truly transferable remains lacking. In this paper, we present a comprehensive evaluation of five key LLM modules--tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation--within the context of motion generation for autonomous driving. Through extensive experiments on the Waymo Sim Agents benchmark, we demonstrate that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation. In addition, we identify which techniques can be effectively transferred, analyze the potential reasons for the failure of others, and discuss the specific adaptations needed for autonomous driving scenarios. We evaluate our method on the Sim Agents task and achieve competitive results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02718v1">Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02550v1">PalmX 2025: The First Shared Task on Benchmarking LLMs on Arabic and Islamic Culture</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 https://palmx.dlnlp.ai/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inherently reflect the vast data distributions they encounter during their pre-training phase. As this data is predominantly sourced from the web, there is a high chance it will be skewed towards high-resourced languages and cultures, such as those of the West. Consequently, LLMs often exhibit a diminished understanding of certain communities, a gap that is particularly evident in their knowledge of Arabic and Islamic cultures. This issue becomes even more pronounced with increasingly under-represented topics. To address this critical challenge, we introduce PalmX 2025, the first shared task designed to benchmark the cultural competence of LLMs in these specific domains. The task is composed of two subtasks featuring multiple-choice questions (MCQs) in Modern Standard Arabic (MSA): General Arabic Culture and General Islamic Culture. These subtasks cover a wide range of topics, including traditions, food, history, religious practices, and language expressions from across 22 Arab countries. The initiative drew considerable interest, with 26 teams registering for Subtask 1 and 19 for Subtask 2, culminating in nine and six valid submissions, respectively. Our findings reveal that task-specific fine-tuning substantially boosts performance over baseline models. The top-performing systems achieved an accuracy of 72.15% on cultural questions and 84.22% on Islamic knowledge. Parameter-efficient fine-tuning emerged as the predominant and most effective approach among participants, while the utility of data augmentation was found to be domain-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02547v1">The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04507v1">From Silent Signals to Natural Language: A Dual-Stage Transformer-LLM Approach</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Silent Speech Interfaces (SSIs) have gained attention for their ability to generate intelligible speech from non-acoustic signals. While significant progress has been made in advancing speech generation pipelines, limited work has addressed the recognition and downstream processing of synthesized speech, which often suffers from phonetic ambiguity and noise. To overcome these challenges, we propose an enhanced automatic speech recognition framework that combines a transformer-based acoustic model with a large language model (LLM) for post-processing. The transformer captures full utterance context, while the LLM ensures linguistic consistency. Experimental results show a 16% relative and 6% absolute reduction in word error rate (WER) over a 36% baseline, demonstrating substantial improvements in intelligibility for silent speech interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16124v2">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14201v2">ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Add code link
    </div>
    <details class="paper-abstract">
      We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development and evaluation of LLM agents, we construct a dataset from a controlled Azure tenant that covers 8 simulated real-world multi-step attacks, 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning. Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research. Code and data are coming soon!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03671v3">TRACE-CS: A Hybrid Logic-LLM System for Explainable Course Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      We present TRACE-CS, a novel hybrid system that combines symbolic reasoning with large language models (LLMs)to address contrastive queries in course scheduling problems. TRACE-CS leverages logic-based techniques to encode scheduling constraints and generate provably correct explanations, while utilizing an LLM to process natural language queries and refine logical explanations into user friendly responses. This system showcases how combining symbolic KR methods with LLMs creates explainable AI agents that balance logical correctness with natural language accessibility, addressing a fundamental challenge in deployed scheduling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01378v2">RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24671v2">Multiple LLM Agents Debate for Equitable Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 ACL 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v2">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14544v2">Adaptively Robust LLM Inference Optimization under Prediction Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request. We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01235v3">NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00134v3">Personalized Causal Graph Reasoning for LLMs: An Implementation for Dietary Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at general-purpose reasoning by leveraging broad commonsense knowledge, but they remain limited in tasks requiring personalized reasoning over multifactorial personal data. This limitation constrains their applicability in domains such as healthcare, where decisions must adapt to individual contexts. We introduce Personalized Causal Graph Reasoning, a framework that enables LLMs to reason over individual-specific causal graphs constructed from longitudinal data. Each graph encodes how user-specific factors influence targeted outcomes. In response to a query, the LLM traverses the graph to identify relevant causal pathways, rank them by estimated impact, simulate potential outcomes, and generate tailored responses. We implement this framework in the context of nutrient-oriented dietary recommendations, where variability in metabolic responses demands personalized reasoning. Using counterfactual evaluation, we assess the effectiveness of LLM-generated food suggestions for glucose control. Our method reduces postprandial glucose iAUC across three time windows compared to prior approaches. Additional LLM-as-a-judge evaluations further confirm improvements in personalization quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19611v2">Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Preparing high-quality instructional materials remains a labor-intensive process that often requires extensive coordination among teaching faculty, instructional designers, and teaching assistants. In this work, we present Instructional Agents, a multi-agent large language model (LLM) framework designed to automate end-to-end course material generation, including syllabus creation, lecture scripts, LaTeX-based slides, and assessments. Unlike existing AI-assisted educational tools that focus on isolated tasks, Instructional Agents simulates role-based collaboration among educational agents to produce cohesive and pedagogically aligned content. The system operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot mode, enabling flexible control over the degree of human involvement. We evaluate Instructional Agents across five university-level computer science courses and show that it produces high-quality instructional materials while significantly reducing development time and human workload. By supporting institutions with limited instructional design capacity, Instructional Agents provides a scalable and cost-effective framework to democratize access to high-quality education, particularly in underserved or resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01822v1">When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has sparked growing interest in their application to time series analysis tasks. However, their ability to perform complex reasoning over temporal data in real-world application domains remains underexplored. To move toward this goal, a first step is to establish a rigorous benchmark dataset for evaluation. In this work, we introduce the TSAIA Benchmark, a first attempt to evaluate LLMs as time-series AI assistants. To ensure both scientific rigor and practical relevance, we surveyed over 20 academic publications and identified 33 real-world task formulations. The benchmark encompasses a broad spectrum of challenges, ranging from constraint-aware forecasting to anomaly detection with threshold calibration: tasks that require compositional reasoning and multi-step time series analysis. The question generator is designed to be dynamic and extensible, supporting continuous expansion as new datasets or task types are introduced. Given the heterogeneous nature of the tasks, we adopt task-specific success criteria and tailored inference-quality metrics to ensure meaningful evaluation for each task. We apply this benchmark to assess eight state-of-the-art LLMs under a unified evaluation protocol. Our analysis reveals limitations in current models' ability to assemble complex time series analysis workflows, underscoring the need for specialized methodologies for domain-specific adaptation. Our benchmark is available at https://huggingface.co/datasets/Melady/TSAIA, and the code is available at https://github.com/USC-Melady/TSAIA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01790v1">Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Prompt sensitivity, referring to the phenomenon where paraphrasing (i.e., repeating something written or spoken using different words) leads to significant changes in large language model (LLM) performance, has been widely accepted as a core limitation of LLMs. In this work, we revisit this issue and ask: Is the widely reported high prompt sensitivity truly an inherent weakness of LLMs, or is it largely an artifact of evaluation processes? To answer this question, we systematically evaluate 7 LLMs (e.g., GPT and Gemini family) across 6 benchmarks, including both multiple-choice and open-ended tasks on 12 diverse prompt templates. We find that much of the prompt sensitivity stems from heuristic evaluation methods, including log-likelihood scoring and rigid answer matching, which often overlook semantically correct responses expressed through alternative phrasings, such as synonyms or paraphrases. When we adopt LLM-as-a-Judge evaluations, we observe a substantial reduction in performance variance and a consistently higher correlation in model rankings across prompts. Our findings suggest that modern LLMs are more robust to prompt templates than previously believed, and that prompt sensitivity may be more an artifact of evaluation than a flaw in the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01750v1">Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Federated learning (FL) for large language models (LLMs) offers a privacy-preserving scheme, enabling clients to collaboratively fine-tune locally deployed LLMs or smaller language models (SLMs) without exchanging raw data. While parameter-sharing methods in traditional FL models solves number of technical challenges, they still incur high communication overhead and struggle with adapting to heterogeneous model architectures. Federated distillation, a framework for mutual knowledge transfer via shared logits, typically offers lower communication overhead than parameter-sharing methods. However, transmitting logits from LLMs remains challenging for bandwidth-limited clients due to their high dimensionality. In this work, we focus on a federated LLM distillation with efficient communication overhead. To achieve this, we first propose an adaptive Top-k logit selection mechanism, dynamically sparsifying logits according to real-time communication conditions. Then to tackle the dimensional inconsistency introduced by the adaptive sparsification, we design an adaptive logits aggregation scheme, effectively alleviating the artificial and uninformative inputs introduced by conventional zero-padding methods. Finally, to enhance the distillation effect, we incorporate LoRA-adapted hidden-layer projection from LLM into the distillation loss, reducing the communication overhead further while providing richer representation. Experimental results demonstrate that our scheme achieves superior performance compared to baseline methods while effectively reducing communication overhead by approximately 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01716v1">An LLM-enabled semantic-centric framework to consume privacy policies</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites, despite claiming otherwise, due to the practical difficulty in comprehending them. The mist of data privacy practices forms a major barrier for user-centred Web approaches, and for data sharing and reusing in an agentic world. Existing research proposed methods for using formal languages and reasoning for verifying the compliance of a specified policy, as a potential cure for ignoring privacy policies. However, a critical gap remains in the creation or acquisition of such formal policies at scale. We present a semantic-centric approach for using state-of-the-art large language models (LLM), to automatically identify key information about privacy practices from privacy policies, and construct $\mathit{Pr}^2\mathit{Graph}$, knowledge graph with grounding from Data Privacy Vocabulary (DPV) for privacy practices, to support downstream tasks. Along with the pipeline, the $\mathit{Pr}^2\mathit{Graph}$ for the top-100 popular websites is also released as a public resource, by using the pipeline for analysis. We also demonstrate how the $\mathit{Pr}^2\mathit{Graph}$ can be used to support downstream tasks by constructing formal policy representations such as Open Digital Right Language (ODRL) or perennial semantic Data Terms of Use (psDToU). To evaluate the technology capability, we enriched the Policy-IE dataset by employing legal experts to create custom annotations. We benchmarked the performance of different large language models for our pipeline and verified their capabilities. Overall, they shed light on the possibility of large-scale analysis of online services' privacy practices, as a promising direction to audit the Web and the Internet. We release all datasets and source code as public resources to facilitate reuse and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01631v1">Unraveling LLM Jailbreaks Through Safety Knowledge Neurons</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01620v1">Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      The rapid development of advanced large language models (LLMs) has made AI-generated text indistinguishable from human-written text. Previous work on detecting AI-generated text has made effective progress, but has not involved modern Chinese poetry. Due to the distinctive characteristics of modern Chinese poetry, it is difficult to identify whether a poem originated from humans or AI. The proliferation of AI-generated modern Chinese poetry has significantly disrupted the poetry ecosystem. Based on the urgency of identifying AI-generated poetry in the real Chinese world, this paper proposes a novel benchmark for detecting LLMs-generated modern Chinese poetry. We first construct a high-quality dataset, which includes both 800 poems written by six professional poets and 41,600 poems generated by four mainstream LLMs. Subsequently, we conduct systematic performance assessments of six detectors on this dataset. Experimental results demonstrate that current detectors cannot be used as reliable tools to detect modern Chinese poems generated by LLMs. The most difficult poetic features to detect are intrinsic qualities, especially style. The detection results verify the effectiveness and necessity of our proposed benchmark. Our work lays a foundation for future detection of AI-generated poetry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01616v1">Automated Generation of Issue-Reproducing Tests by Combining LLMs and Search-Based Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 13 pages, 8 figures, accepted for publication (to appear) in the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025
    </div>
    <details class="paper-abstract">
      Issue-reproducing tests fail on buggy code and pass once a patch is applied, thus increasing developers' confidence that the issue has been resolved and will not be re-introduced. However, past research has shown that developers often commit patches without such tests, making the automated generation of issue-reproducing tests an area of interest. We propose BLAST, a tool for automatically generating issue-reproducing tests from issue-patch pairs by combining LLMs and search-based software testing (SBST). For the LLM part, we complement the issue description and the patch by extracting relevant context through git history analysis, static analysis, and SBST-generated tests. For the SBST part, we adapt SBST for generating issue-reproducing tests; the issue description and the patch are fed into the SBST optimization through an intermediate LLM-generated seed, which we deserialize into SBST-compatible form. BLAST successfully generates issue-reproducing tests for 151/426 (35.4%) of the issues from a curated Python benchmark, outperforming the state-of-the-art (23.5%). Additionally, to measure the real-world impact of BLAST, we built a GitHub bot that runs BLAST whenever a new pull request (PR) linked to an issue is opened, and if BLAST generates an issue-reproducing test, the bot proposes it as a comment in the PR. We deployed the bot in three open-source repositories for three months, gathering data from 32 PRs-issue pairs. BLAST generated an issue-reproducing test in 11 of these cases, which we proposed to the developers. By analyzing the developers' feedback, we discuss challenges and opportunities for researchers and tool builders. Data and material: https://doi.org/10.5281/zenodo.16949042
    </details>
</div>
