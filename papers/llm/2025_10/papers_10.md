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
- [Part 9](papers_9.md)
- Part 10
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13481v1">Tahakom LLM guidelines and receipts: from pre-training data to an Arabic LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the field of natural language processing, enhancing capabilities in both language understanding and generation across diverse domains. However, developing LLMs for Arabic presents unique challenges. This paper explores these challenges by focusing on critical aspects such as data curation, tokenizer design, and evaluation. We detail our approach to the collection and filtration of Arabic pre-training datasets, assess the impact of various tokenizer designs on model performance, and examine the limitations of existing Arabic evaluation frameworks, for which we propose a systematic corrective methodology. To promote transparency and facilitate collaborative development, we share our data and methodologies, contributing to the advancement of language modeling, particularly for the Arabic language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13467v1">NetMCP: Network-Aware Model Context Protocol Platform for LLM Capability Extension</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) remain static in functionality after training, and extending their capabilities requires integration with external data, computation, and services. The Model Context Protocol (MCP) has emerged as a standard interface for such extensions, but current implementations rely solely on semantic matching between users' requests and server function descriptions, which makes current deployments and simulation testbeds fragile under latency fluctuations or server failures. We address this gap by enhancing MCP tool routing algorithms with real-time awareness of network and server status. To provide a controlled test environment for development and evaluation, we construct a heterogeneous experimental platform, namely Network-aware MCP (NetMCP), which offers five representative network states and build a benchmark for latency sequence generation and MCP server datasets. On top of NetMCP platform, we analyze latency sequences and propose a Semantic-Oriented and Network-Aware Routing (SONAR) algorithm, which jointly optimizes semantic similarity and network Quality of Service (QoS) metrics for adaptive tool routing. Results show that SONAR consistently improves task success rate and reduces completion time and failure number compared with semantic-only, LLM-based baselines, demonstrating the value of network-aware design for production-scale LLM systems. The code for NetMCP is available at https://github.com/NICE-HKU/NetMCP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13417v1">Assessing LLM Reasoning Through Implicit Causal Chain Discovery in Climate Discourse</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      How does a cause lead to an effect, and which intermediate causal steps explain their connection? This work scrutinizes the mechanistic causal reasoning capabilities of large language models (LLMs) to answer these questions through the task of implicit causal chain discovery. In a diagnostic evaluation framework, we instruct nine LLMs to generate all possible intermediate causal steps linking given cause-effect pairs in causal chain structures. These pairs are drawn from recent resources in argumentation studies featuring polarized discussion on climate change. Our analysis reveals that LLMs vary in the number and granularity of causal steps they produce. Although they are generally self-consistent and confident about the intermediate causal connections in the generated chains, their judgments are mainly driven by associative pattern matching rather than genuine causal reasoning. Nonetheless, human evaluations confirmed the logical coherence and integrity of the generated chains. Our baseline causal chain discovery approach, insights from our diagnostic evaluation, and benchmark dataset with causal chains lay a solid foundation for advancing future work in implicit, mechanistic causal reasoning in argumentation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13401v1">F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to Workshop on New Approaches for Addressing the Computing Requirements of LLMs and GNNs (LG-ARC) @ ISCA 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly prominent for daily tasks, from improving sound-totext translation to generating additional frames for the latest video games. With the help of LLM inference frameworks, such as llama.cpp, which support optimizations such as KV-caching and quantization, it is now easier than ever to deploy LLMs on edge devices. Quantization is fundamental to enable LLMs on resource-constrained edge devices, and llama.cpp utilizes block floating point (BFP) quantization to drastically reduce the bit width of weights and input tensors, the memory footprint, and the computational power required to run LLMs. LLMs are typically quantized with mixed BFP quantization across the model layers to reduce the loss of model accuracy due to quantization. Therefore, to efficiently accelerate across the layers of BFP-quantized LLMs, specialized accelerators need to support different BFP variants without reconfiguration. To address this issue, we propose a Flexible Block FloatingPoint Quantization (F-BFQ) accelerator, which can dynamically switch between two BFP quantization variants and perform matrix multiplication (MatMul) operations. Our initial F-BFQ accelerator design, deployed on the AMD Kria board, reduces inference time by 1.4x on average over the Arm NEON-based CPU execution across three BFP quantized LLMs while achieving 5.2 tokens per second (~3.9 words per second).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11409v6">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Penetration-testing is crucial for identifying system vulnerabilities, with privilege-escalation being a critical subtask to gain elevated access to protected resources. Language Models (LLMs) presents new avenues for automating these security practices by emulating human behavior. However, a comprehensive understanding of LLMs' efficacy and limitations in performing autonomous Linux privilege-escalation attacks remains under-explored. To address this gap, we introduce hackingBuddyGPT, a fully automated LLM-driven prototype designed for autonomous Linux privilege-escalation. We curated a novel, publicly available Linux privilege-escalation benchmark, enabling controlled and reproducible evaluation. Our empirical analysis assesses the quantitative success rates and qualitative operational behaviors of various LLMs -- GPT-3.5-Turbo, GPT-4-Turbo, and Llama3 -- against baselines of human professional pen-testers and traditional automated tools. We investigate the impact of context management strategies, different context sizes, and various high-level guidance mechanisms on LLM performance. Results show that GPT-4-Turbo demonstrates high efficacy, successfully exploiting 33-83% of vulnerabilities, a performance comparable to human pen-testers (75%). In contrast, local models like Llama3 exhibited limited success (0-33%), and GPT-3.5-Turbo achieved moderate rates (16-50%). We show that both high-level guidance and state-management through LLM-driven reflection significantly boost LLM success rates. Qualitative analysis reveals both LLMs' strengths and weaknesses in generating valid commands and highlights challenges in common-sense reasoning, error handling, and multi-step exploitation, particularly with temporal dependencies. Cost analysis indicates that GPT-4-Turbo can achieve human-comparable performance at competitive costs, especially with optimized context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13371v1">MADREC: A Multi-Aspect Driven LLM Agent for Explainable and Adaptive Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Recent attempts to integrate large language models (LLMs) into recommender systems have gained momentum, but most remain limited to simple text generation or static prompt-based inference, failing to capture the complexity of user preferences and real-world interactions. This study proposes the Multi-Aspect Driven LLM Agent MADRec, an autonomous LLM-based recommender that constructs user and item profiles by unsupervised extraction of multi-aspect information from reviews and performs direct recommendation, sequential recommendation, and explanation generation. MADRec generates structured profiles via aspect-category-based summarization and applies Re-Ranking to construct high-density inputs. When the ground-truth item is missing from the output, the Self-Feedback mechanism dynamically adjusts the inference criteria. Experiments across multiple domains show that MADRec outperforms traditional and LLM-based baselines in both precision and explainability, with human evaluation further confirming the persuasiveness of the generated explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13363v1">D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 8 pages, 6 figures (main content); 25 pages, 18 figures (total)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit factual inconsistencies and logical decay in extended, multi-turn dialogues, a challenge stemming from their reliance on static, pre-trained knowledge and an inability to reason adaptively over the dialogue history. Prevailing mitigation strategies, such as Retrieval-Augmented Generation (RAG) and agentic working memories, improve information recall but still engage with fundamentally static knowledge sources and follow pre-defined single reasoning path. This hinders their ability to preserve factual and logical consistency of their responses in multi-turn dialogues while the context evolves over time. To address this issue, we propose D-SMART, a model-agnostic framework designed to maintain multi-turn dialogue consistency by enabling LLMs to build and reason over a dynamic, structured representation of the conversational context. This is achieved via two synergistic components: (1) a Dynamic Structured Memory (DSM), which incrementally constructs and maintains an authoritative, OWL-compliant knowledge graph of the conversation; and (2) a Reasoning Tree (RT), which executes inferences as an explicit and traceable multi-step search over the graph. As the popular-used quality score (judged by GPT-4) can overlook logical flaws, we introduce new NLI-based metrics to better measure multi-turn dialogue consistency. Comprehensive experiments on the MT-Bench-101 benchmark show that D-SMART significantly outperforms state-of-the-art baselines, elevating the dialogue consistency score by over 48\% for both proprietary and open-source models, and notably improves the quality score of the latter by up to 10.1\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24101v2">BTC-SAM: Leveraging LLMs for Generation of Bias Test Cases for Sentiment Analysis Models</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted at EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      Sentiment Analysis (SA) models harbor inherent social biases that can be harmful in real-world applications. These biases are identified by examining the output of SA models for sentences that only vary in the identity groups of the subjects. Constructing natural, linguistically rich, relevant, and diverse sets of sentences that provide sufficient coverage over the domain is expensive, especially when addressing a wide range of biases: it requires domain experts and/or crowd-sourcing. In this paper, we present a novel bias testing framework, BTC-SAM, which generates high-quality test cases for bias testing in SA models with minimal specification using Large Language Models (LLMs) for the controllable generation of test sentences. Our experiments show that relying on LLMs can provide high linguistic variation and diversity in the test sentences, thereby offering better test coverage compared to base prompting methods even for previously unseen biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21540v2">HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Figure 1 updated, typos corrected, references added, under review
    </div>
    <details class="paper-abstract">
      Process mining has emerged as a powerful analytical technique for understanding complex healthcare workflows. However, its application faces significant barriers, including technical complexity, a lack of standardized approaches, and limited access to practical training resources. We introduce HealthProcessAI, a GenAI framework designed to simplify process mining applications in healthcare and epidemiology by providing a comprehensive wrapper around existing Python (PM4PY) and R (bupaR) libraries. To address unfamiliarity and improve accessibility, the framework integrates multiple Large Language Models (LLMs) for automated process map interpretation and report generation, helping translate technical analyses into outputs that diverse users can readily understand. We validated the framework using sepsis progression data as a proof-of-concept example and compared the outputs of five state-of-the-art LLM models through the OpenRouter platform. To test its functionality, the framework successfully processed sepsis data across four proof-of-concept scenarios, demonstrating robust technical performance and its capability to generate reports through automated LLM analysis. LLM evaluation using five independent LLMs as automated evaluators revealed distinct model strengths: Claude Sonnet-4 and Gemini 2.5-Pro achieved the highest consistency scores (3.79/4.0 and 3.65/4.0) when evaluated by automated LLM assessors. By integrating multiple Large Language Models (LLMs) for automated interpretation and report generation, the framework addresses widespread unfamiliarity with process mining outputs, making them more accessible to clinicians, data scientists, and researchers. This structured analytics and AI-driven interpretation combination represents a novel methodological advance in translating complex process mining results into potentially actionable insights for healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13351v1">Protect: Towards Robust Guardrailing Stack for Trustworthy Enterprise LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The increasing deployment of Large Language Models (LLMs) across enterprise and mission-critical domains has underscored the urgent need for robust guardrailing systems that ensure safety, reliability, and compliance. Existing solutions often struggle with real-time oversight, multi-modal data handling, and explainability -- limitations that hinder their adoption in regulated environments. Existing guardrails largely operate in isolation, focused on text alone making them inadequate for multi-modal, production-scale environments. We introduce Protect, natively multi-modal guardrailing model designed to operate seamlessly across text, image, and audio inputs, designed for enterprise-grade deployment. Protect integrates fine-tuned, category-specific adapters trained via Low-Rank Adaptation (LoRA) on an extensive, multi-modal dataset covering four safety dimensions: toxicity, sexism, data privacy, and prompt injection. Our teacher-assisted annotation pipeline leverages reasoning and explanation traces to generate high-fidelity, context-aware labels across modalities. Experimental results demonstrate state-of-the-art performance across all safety dimensions, surpassing existing open and proprietary models such as WildGuard, LlamaGuard-4, and GPT-4.1. Protect establishes a strong foundation for trustworthy, auditable, and production-ready safety systems capable of operating across text, image, and audio modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13302v1">LLM one-shot style transfer for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01656v3">Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13291v1">Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 36 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13271v1">Do You Get the Hint? Benchmarking LLMs on the Board Game Concept</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In particular, tasks that require abstract reasoning remain challenging, often because they use representations such as grids, symbols, or visual patterns that differ from the natural language data LLMs are trained on. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning in a representation that is much closer to LLM pre-training data: natural language. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13257v1">GRIDAI: Generating and Repairing Intrusion Detection Rules via Collaboration among Multiple LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Rule-based network intrusion detection systems play a crucial role in the real-time detection of Web attacks. However, most existing works primarily focus on automatically generating detection rules for new attacks, often overlooking the relationships between new attacks and existing rules, which leads to significant redundancy within the ever-expanding ruleset. To address this issue, we propose GRIDAI, a novel end-to-end framework for the automated Generation and Repair of Intrusion Detection rules through collaboration among multiple LLM-based agents. Unlike traditional methods, GRIDAI first assesses the nature of incoming attack samples. If the sample represents a new attack type, it is used to generate a new rule. Otherwise, the sample is identified as a variant of an attack already covered by an existing rule and used to repair the rule by updating the corresponding signature, thereby enhancing its generalization capability. Additionally, to mitigate syntactic and semantic errors in rules caused by LLM hallucinations, we incorporate a tool-based real-time validation mechanism and a representative attack sample maintained for each rule, enabling fully automated rule generation and repair. Comprehensive experiments were conducted on a public dataset containing seven types of attacks and a private dataset with 43 attack types. The results demonstrate that GRIDAI accurately identifies the relationships between new attack samples and existing rules, efficiently generates and repairs rules to handle new attacks and variants, and effectively mitigates the impact of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v2">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13248v1">Automated Network Protocol Testing with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Network protocol testing is fundamental for modern network infrastructure. However, traditional network protocol testing methods are labor-intensive and error-prone, requiring manual interpretation of specifications, test case design, and translation into executable artifacts, typically demanding one person-day of effort per test case. Existing model-based approaches provide partial automation but still involve substantial manual modeling and expert intervention, leading to high costs and limited adaptability to diverse and evolving protocols. In this paper, we propose a first-of-its-kind system called NeTestLLM that takes advantage of multi-agent Large Language Models (LLMs) for end-to-end automated network protocol testing. NeTestLLM employs hierarchical protocol understanding to capture complex specifications, iterative test case generation to improve coverage, a task-specific workflow for executable artifact generation, and runtime feedback analysis for debugging and refinement. NeTestLLM has been deployed in a production environment for several months, receiving positive feedback from domain experts. In experiments, NeTestLLM generated 4,632 test cases for OSPF, RIP, and BGP, covering 41 historical FRRouting bugs compared to 11 by current national standards. The process of generating executable artifacts also improves testing efficiency by a factor of 8.65x compared to manual methods. NeTestLLM provides the first practical LLM-powered solution for automated end-to-end testing of heterogeneous network protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v2">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13229v1">Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 ICDM 2025 Accepted Paper
    </div>
    <details class="paper-abstract">
      Recommender systems (RecSys) have become critical tools for enhancing user engagement by delivering personalized content across diverse digital platforms. Recent advancements in large language models (LLMs) demonstrate significant potential for improving RecSys, primarily due to their exceptional generalization capabilities and sophisticated contextual understanding, which facilitate the generation of flexible and interpretable recommendations. However, the direct deployment of LLMs as primary recommendation policies presents notable challenges, including persistent latency issues stemming from frequent API calls and inherent model limitations such as hallucinations and biases. To address these issues, this paper proposes a novel offline reinforcement learning (RL) framework that leverages imitation learning from LLM-generated trajectories. Specifically, inverse reinforcement learning is employed to extract robust reward models from LLM demonstrations. This approach negates the need for LLM fine-tuning, thereby substantially reducing computational overhead. Simultaneously, the RL policy is guided by the cumulative rewards derived from these demonstrations, effectively transferring the semantic insights captured by the LLM. Comprehensive experiments conducted on two benchmark datasets validate the effectiveness of the proposed method, demonstrating superior performance when compared against state-of-the-art RL-based and in-context learning baselines. The code can be found at https://github.com/ArronDZhang/IL-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01427v2">A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Code available: https://github.com/LongfeiYun17/falconer
    </div>
    <details class="paper-abstract">
      At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13223v1">BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in AI infrastructure, driving the need for high throughput, resource efficient serving systems. Disaggregated LLM serving, which separates prompt prefill from auto-regressive decode, has emerged as a promising architecture by isolating their heterogeneous compute and memory demands. However, current disaggregated systems face three key limitations: (i) static resource allocation cannot adapt to highly dynamic workloads, causing over-provisioning that wastes resources or under-provisioning that violates service level objectives (SLOs); (ii) inherent load imbalance between prefill and decode stages, where prefill is compute-bound and decode is memory-bound, causes under-utilization in one tier while the other becomes a bottleneck; and (iii) prefix cache aware routing skews load distribution, as high cache hit rate prefill nodes attract disproportionately more requests, further degrading balance and efficiency. To address these issues, we present BanaServe, a dynamic orchestration framework that continuously rebalances computational and memory resources across prefill and decode instances while eliminating hotspots induced by cache. BanaServe introduces layer level weight migration, attention level Key Value Cache (KV Cache) migration, and Global KV Cache Store sharing with layer wise overlapped transmission, enabling both coarse grained (layer level) and fine grained (attention level) load redistribution with minimal latency overhead. These mechanisms allow routers to perform purely load aware scheduling, unconstrained by cache placement. Compared to vLLM, BanaServe achieves 1.2x-3.9x higher throughput with 3.9%-78.4% lower total processing time, and outperforms DistServe by 1.1x-2.8x in throughput with 1.4%-70.1% latency reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13217v1">LLM-guided Hierarchical Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Modern IR systems are increasingly tasked with answering complex, multi-faceted queries that require deep reasoning rather than simple keyword or semantic matching. While LLM-based IR has shown great promise, the prevailing retrieve-then-rerank paradigm inherits the limitations of embedding-based retrieval; parametric generative approaches are difficult to update with new information; and long-context methods that place the entire corpus in context are computationally infeasible for large document collections. To address these challenges, we introduce LATTICE, a hierarchical retrieval framework that enables an LLM to reason over and navigate large corpora with logarithmic search complexity by imposing a semantic tree structure on the corpus. Our approach consists of two stages: (1) an offline phase that organizes the corpus into a semantic hierarchy via either a bottom-up agglomerative strategy or a top-down divisive strategy using multi-level summaries and (2) an online traversal phase where a search LLM navigates this tree. A central challenge in such LLM-guided search is that the model's relevance judgments are noisy, context-dependent, and unaware of the hierarchy, making cross-branch and cross-level comparisons difficult. To overcome this, we propose a traversal algorithm that estimates calibrated latent relevance scores from local LLM outputs and aggregates them into a global path relevance metric. Our training-free framework achieves state-of-the-art zero-shot performance on the reasoning-intensive BRIGHT benchmark, demonstrating up to 9% improvement in Recall@100 and 5% in nDCG@10 over the next best zero-shot baseline. Furthermore, compared to the fine-tuned SOTA method DIVER-v2, LATTICE attains comparable results on BRIGHT subsets that use a static corpus for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13202v1">LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 11 pages, 4 figures, 1 Table, submitted to an international conference
    </div>
    <details class="paper-abstract">
      Bias in AI systems, especially those relying on natural language data, raises ethical and practical concerns. Underrepresentation of certain groups often leads to uneven performance across demographics. Traditional fairness methods, such as pre-processing, in-processing, and post-processing, depend on protected-attribute labels, involve accuracy-fairness trade-offs, and may not generalize across datasets. To address these challenges, we propose LLM-Guided Synthetic Augmentation (LGSA), which uses large language models to generate counterfactual examples for underrepresented groups while preserving label integrity. We evaluated LGSA on a controlled dataset of short English sentences with gendered pronouns, professions, and binary classification labels. Structured prompts were used to produce gender-swapped paraphrases, followed by quality control including semantic similarity checks, attribute verification, toxicity screening, and human spot checks. The augmented dataset expanded training coverage and was used to train a classifier under consistent conditions. Results show that LGSA reduces performance disparities without compromising accuracy. The baseline model achieved 96.7 percent accuracy with a 7.2 percent gender bias gap. Simple swap augmentation reduced the gap to 0.7 percent but lowered accuracy to 95.6 percent. LGSA achieved 99.1 percent accuracy with a 1.9 percent bias gap, improving performance on female-labeled examples. These findings demonstrate that LGSA is an effective strategy for bias mitigation, enhancing subgroup balance while maintaining high task accuracy and label fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05258v2">Spatio-Temporal LLM: Reasoning about Environments and Actions</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Code and data are available at https://zoezheng126.github.io/STLLM-website/
    </div>
    <details class="paper-abstract">
      Despite significant recent progress of Multimodal Large Language Models (MLLMs), current MLLMs are challenged by "spatio-temporal" prompts, i.e., prompts that refer to 1) the entirety of an environment encoded in a point cloud that the MLLM should consider; and simultaneously also refer to 2) actions that happened in part of the environment and are encoded in a short ego-centric video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this challenge, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent MLLMs indeed struggle to correctly answer "spatio-temporal" prompts. Building on this dataset, we study two spatio-temporal LLM (STLLM) baselines: 1) STLLM-3D, which directly fuses point cloud, video, and text representations as inputs to the LLM; and 2) STLLM-Aligner, which aligns spatial context with video and text before LLM decoding. Both baselines aim to enhance spatial understanding of environments and temporal grounding of egocentric observations. On REA, the STLLM baselines outperform existing models, demonstrating the effectiveness of our designs. Code and data are available at https://zoezheng126.github.io/STLLM-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13195v1">Emotional Cognitive Modeling Framework with Desire-Driven Objective Optimization for LLM-empowered Agent in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has enabled agents to represent virtual humans in societal simulations, facilitating diverse interactions within complex social systems. However, existing LLM-based agents exhibit severe limitations in affective cognition: They fail to simulate the bounded rationality essential for bridging virtual and real-world services; They lack empirically validated integration mechanisms embedding emotions within agent decision architectures. This paper constructs an emotional cognition framework incorporating desire generation and objective management, designed to achieve emotion alignment between LLM-based agents and humans, modeling the complete decision-making process of LLM-based agents, encompassing state evolution, desire generation, objective optimization, decision generation, and action execution. This study implements the proposed framework within our proprietary multi-agent interaction environment. Experimental results demonstrate that agents governed by our framework not only exhibit behaviors congruent with their emotional states but also, in comparative assessments against other agent types, demonstrate superior ecological validity and generate decision outcomes that significantly more closely approximate human behavioral patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13193v1">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at https://github.com/kilgrims/ReMindRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23227v2">Enabling Few-Shot Alzheimer's Disease Diagnosis on Biomarker Data with Tabular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 accepted by ACM-BCB'25: ACM Conference on Bioinformatics, Computational Biology, and Health Informatics [ACM SIGBio Best Paper Award]
    </div>
    <details class="paper-abstract">
      Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13161v1">Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13154v1">I Am Aligned, But With Whom? MENA Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We introduce MENAValues, a novel benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with respect to the beliefs and values of the Middle East and North Africa (MENA) region, an underrepresented area in current AI evaluation efforts. Drawing from large-scale, authoritative human surveys, we curate a structured dataset that captures the sociocultural landscape of MENA with population-level response distributions from 16 countries. To probe LLM behavior, we evaluate diverse models across multiple conditions formed by crossing three perspective framings (neutral, personalized, and third-person/cultural observer) with two language modes (English and localized native languages: Arabic, Persian, Turkish). Our analysis reveals three critical phenomena: "Cross-Lingual Value Shifts" where identical questions yield drastically different responses based on language, "Reasoning-Induced Degradation" where prompting models to explain their reasoning worsens cultural alignment, and "Logit Leakage" where models refuse sensitive questions while internal probabilities reveal strong hidden preferences. We further demonstrate that models collapse into simplistic linguistic categories when operating in native languages, treating diverse nations as monolithic entities. MENAValues offers a scalable framework for diagnosing cultural misalignment, providing both empirical insights and methodological tools for developing more culturally inclusive AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13143v1">Stable LLM Ensemble: Interaction between Example Representativeness and Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable results in wide range of domains. However, the accuracy and robustness of one-shot LLM predictions remain highly sensitive to the examples and the diversity among ensemble members. This study systematically investigates the effects of example representativeness (one-shot strategy) and output diversity (sampling temperature) on LLM ensemble performance. Two one-shot strategies are compared: centroid-based representative examples (proposed) and randomly sampled examples (baseline) and sampling temperature also is varied. The proposed approach with higher temperature setting significantly outperforms random selection by +7.6% (macro-F1) and -10.5% (RMSE). Furthermore, the proposed model exceeds 5-shot prompting by +21.1% (macro-F1) and -24.0% (RMSE). Our findings demonstrate that combining representative example selection with increased temperature provides the appropriate level of diversity to the ensemble. This work highlights the practical importance of both example selection and controlled diversity in designing effective one-shot LLM ensembles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13139v1">Addressing the alignment problem in transportation policy making: an LLM approach</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      A key challenge in transportation planning is that the collective preferences of heterogeneous travelers often diverge from the policies produced by model-driven decision tools. This misalignment frequently results in implementation delays or failures. Here, we investigate whether large language models (LLMs), noted for their capabilities in reasoning and simulating human decision-making, can help inform and address this alignment problem. We develop a multi-agent simulation in which LLMs, acting as agents representing residents from different communities in a city, participate in a referendum on a set of transit policy proposals. Using chain-of-thought reasoning, LLM agents provide ranked-choice or approval-based preferences, which are aggregated using instant-runoff voting (IRV) to model democratic consensus. We implement this simulation framework with both GPT-4o and Claude-3.5, and apply it for Chicago and Houston. Our findings suggest that LLM agents are capable of approximating plausible collective preferences and responding to local context, while also displaying model-specific behavioral biases and modest divergences from optimization-based benchmarks. These capabilities underscore both the promise and limitations of LLMs as tools for solving the alignment problem in transportation decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v4">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23988v2">LLM/Agent-as-Data-Analyst: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 32 page, 11 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) and agent techniques for data analysis (a.k.a LLM/Agent-as-Data-Analyst) have demonstrated substantial impact in both academica and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. The technical evolution further distills five key design goals for intelligent data analysis agents, namely semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., table question answering for relational data and NL2GQL for graph data), (ii) semi-structured data (e.g., markup languages understanding and semi-structured table modeling), (iii) unstructured data (e.g., chart understanding, document understanding, programming languages vulnerable detection), and (iv) heterogeneous data (e.g., data retrieval and modality alignment for data lakes). Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v3">Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) yet often fails to maintain comparable performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Although global pruning aims to identify an optimal sparse model, intuitive methods typically adopt a two-stage paradigm that first evaluates substructure saliency and then applies global pruning, which ignores inter-structure dependencies and fails to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters. Code will be available at https://github.com/AMD-AGI/Tyr-the-Pruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13105v1">EgoSocial: Benchmarking Proactive Intervention Ability of Omnimodal LLMs via Egocentric Social Interaction Perception</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      As AR/VR technologies become integral to daily life, there's a growing need for AI that understands human social dynamics from an egocentric perspective. However, current LLMs often lack the social awareness to discern when to intervene as AI assistant. This leads to constant, socially unaware responses that may disrupt natural conversation and negatively impact user focus. To address these limitations, we introduce EgoSocial, a large-scale egocentric dataset with 13,500 social video-question pairs, specifically designed to benchmark intervention in social interaction perception. We also present an in-depth analysis of current omnimodal LLMs (OLLMs) to assess their effectiveness in detecting diverse social contextual cues. Experiments show that OLLMs still struggle to detect the intervention timing (14.4% for Gemini 2.5 Pro). We also propose EgoSoD (EgoSocial Detection), an end-to-end method for robustly discerning social dynamics. Informed by our OLLM analysis, EgoSoD integrates multimodal contextual cues (e.g., audio and visual cues) into a social thinking graph, dynamically modeling participants and interactions. Our method proactively detects intervention timing and social interactions, precisely determining when to intervene. Our EgoSoD improves Phi-4 by 45.6% and Gemini 2.5 Pro by 9.9% on Intervention Timing performance, and improves Phi-4 by 20.4% and Gemini 2.5 Pro by 6.9% on overall Social Interaction performance. We will release the dataset and code soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v3">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in their reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23694v3">SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18133v4">Self-Evolving LLMs via Continual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening generalization.We propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting self-evolution.Extensive experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13091v1">Unmasking Hiring Bias: Platform Data Analysis and Controlled Experiments on Bias in Online Freelance Marketplaces via RAG-LLM Generated Contents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Online freelance marketplaces, a rapidly growing part of the global labor market, are creating a fair environment where professional skills are the main factor for hiring. While these platforms can reduce bias from traditional hiring, the personal information in user profiles raises concerns about ongoing discrimination. Past studies on this topic have mostly used existing data, which makes it hard to control for other factors and clearly see the effect of things like gender or race. To solve these problems, this paper presents a new method that uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to create realistic, artificial freelancer profiles for controlled experiments. This approach effectively separates individual factors, enabling a clearer statistical analysis of how different variables influence the freelancer project process. In addition to analyzing extracted data with traditional statistical methods for post-project stage analysis, our research utilizes a dataset with highly controlled variables, generated by an RAG-LLM, to conduct a simulated hiring experiment for pre-project stage analysis. The results of our experiments show that, regarding gender, while no significant preference emerged in initial hiring decisions, female freelancers are substantially more likely to receive imperfect ratings post-project stage. Regarding regional bias, a strong and consistent preference favoring US-based freelancers shows that people are more likely to be selected in the simulated experiments, perceived as more leader-like, and receive higher ratings on the live platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10113v3">What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset for benchmarking large language models (LLMs) in fine-grained clinical specialties. S-MedQA has over 20k examples, covers 15 medical specialties, and QA pairs can have multiple specialty annotations (e.g., when a question is cross-disciplinary), constructed with both machine and expert verification to maximize data availability. We use S-MedQA to investigate the role of clinical specialty data in the knowledge-intensive scenario of medical QA. Our results show that 1) training on data from a clinical specialty does not necessarily lead to best performance on that specialty, and 2) regardless of the specialty the LLM was fine-tuned on, token probabilities of clinically relevant terms increase consistently across all specialties. Thus, we hypothesize improvement gains are derived mostly from domain shifting (e.g., general to medical) rather than specialty-specific knowledge injection, and suggest rethinking the role of fine-tuning data in the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02340v2">Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Published at EMNLP 2025; Code and data available at https://github.com/gxx27/time_unlearn
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for temporal prediction, but their reliance on pretraining data raises contamination concerns, as accurate predictions on pre-cutoff test data may reflect memorization rather than reasoning, leading to an overestimation of their generalization capability. With the recent emergence of prompting-based unlearning techniques, a natural question arises: Can LLMs be prompted to simulate an earlier knowledge cutoff? In this work, we investigate the capability of prompting to simulate earlier knowledge cutoff in LLMs. We construct three evaluation datasets to assess the extent to which LLMs can forget (1) direct factual knowledge, (2) semantic shifts, and (3) causally related knowledge. Results demonstrate that while prompt-based simulated knowledge cutoffs show effectiveness when directly queried with the information after that date, they struggle to induce forgetting when the forgotten content is not directly asked but causally related to the query. These findings highlight the need for more rigorous evaluation settings when applying LLMs for temporal prediction tasks. The full dataset and evaluation code are available at https://github.com/gxx27/time_unlearn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14162v1">FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 4 pages, 2 figures, accepted at CIKM 2025 FinAI Workshop
    </div>
    <details class="paper-abstract">
      We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12491v2">Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v3">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23410v2">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14115v1">David vs. Goliath: A comparative study of different-sized LLMs for code generation in the domain of automotive scenario generation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Scenario simulation is central to testing autonomous driving systems. Scenic, a domain-specific language (DSL) for CARLA, enables precise and reproducible scenarios, but NL-to-Scenic generation with large language models (LLMs) suffers from scarce data, limited reproducibility, and inconsistent metrics. We introduce NL2Scenic, an open dataset and framework with 146 NL/Scenic pairs, a difficulty-stratified 30-case test split, an Example Retriever, and 14 prompting variants (ZS, FS, CoT, SP, MoT). We evaluate 13 models: four proprietary (GPT-4o, GPT-5, Claude-Sonnet-4, Gemini-2.5-pro) and nine open-source code models (Qwen2.5Coder 0.5B-32B; CodeLlama 7B/13B/34B), using text metrics (BLEU, ChrF, EDIT-SIM, CrystalBLEU) and execution metrics (compilation and generation), and compare them with an expert study (n=11). EDIT-SIM correlates best with human judgments; we also propose EDIT-COMP (F1 of EDIT-SIM and compilation) as a robust dataset-level proxy that improves ranking fidelity. GPT-4o performs best overall, while Qwen2.5Coder-14B reaches about 88 percent of its expert score on local hardware. Retrieval-augmented prompting, Few-Shot with Example Retriever (FSER), consistently boosts smaller models, and scaling shows diminishing returns beyond mid-size, with Qwen2.5Coder outperforming CodeLlama at comparable scales. NL2Scenic and EDIT-COMP offer a standardized, reproducible basis for evaluating Scenic code generation and indicate that mid-size open-source models are practical, cost-effective options for autonomous-driving scenario programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v3">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We introduce time-to-unsafe-sampling, a novel safety measure for generative models, defined as the number of generations required by a large language model (LLM) to trigger an unsafe (e.g., toxic) response. While providing a new dimension for prompt-adaptive safety evaluation, quantifying time-to-unsafe-sampling is challenging: unsafe outputs are often rare in well-aligned models and thus may not be observed under any feasible sampling budget. To address this challenge, we frame this estimation problem as one of survival analysis. We build on recent developments in conformal prediction and propose a novel calibration technique to construct a lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt with rigorous coverage guarantees. Our key technical innovation is an optimized sampling-budget allocation scheme that improves sample efficiency while maintaining distribution-free guarantees. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04101v2">LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Conference paper, 6 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are very few studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v5">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 10 pages, Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14036v1">One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program. In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v5">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 48 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14030v1">Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 EMNLP Main 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14024v1">Efficiently Executing High-throughput Lightweight LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The rise of Generative AI introduces a new class of HPC workloads that integrates lightweight LLMs with traditional high-throughput applications to accelerate scientific discovery. The current design of HPC clusters is inadequate to support this new class however, either incurring long wait times on static batch queues or repeatedly paying expensive LLM startup costs upon resource preemption. To circumvent both the long queues and high startup costs, we propose to "decouple" the LLM initialization context from the actual LLM inferences, and retain the context in GPUs until it is no longer needed, a technique we term "Pervasive Context Management". We transform a fact verification application to enable this technique, allowing it to reduce its execution time by 72.1% (from 3 hours to 48 minutes) using the same amount of GPUs, and scale opportunistically on 32.8% of all GPUs in the cluster and further reduce the execution time to 13 minutes.
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
