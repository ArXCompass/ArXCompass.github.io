# llm - 2025_08

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08325v2">CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17182v2">LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 This preprint is under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17117v4">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05702v2">Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06133v2">LLM Serving Optimization with Variable Prefill and Decode Lengths</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      We study the problem of serving LLM (Large Language Model) requests where each request has heterogeneous prefill and decode lengths. In LLM serving, the prefill length corresponds to the input prompt length, which determines the initial memory usage in the KV cache. The decode length refers to the number of output tokens generated sequentially, with each additional token increasing the KV cache memory usage by one unit. Given a set of n requests, our goal is to schedule and process them to minimize the total completion time. We show that this problem is NP-hard due to the interplay of batching, placement constraints, precedence relationships, and linearly increasing memory usage. We then analyze commonly used scheduling strategies in practice, such as First-Come-First-Serve (FCFS) and Shortest-First (SF), and prove that their competitive ratios scale up sublinearly with the memory limit-a significant drawback in real-world settings where memory demand is large. To address this, we propose a novel algorithm based on a new selection metric that efficiently forms batches over time. We prove that this algorithm achieves a constant competitive ratio. Finally, we develop and evaluate a few algorithm variants inspired by this approach, including dynamic programming variants, local search methods, and an LP-based scheduler, demonstrating through comprehensive simulations that they outperform standard baselines while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04078v3">LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01085v4">Explaining Length Bias in LLM-Based Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as judges, particularly in preference comparisons, has become widespread, but this reveals a notable bias towards longer responses, undermining the reliability of such evaluations. To better understand such bias, we propose to decompose the preference evaluation metric, specifically the win rate, into two key components: desirability and information mass, where the former is length-independent and related to trustworthiness such as correctness, toxicity, and consistency, and the latter is length-dependent and represents the amount of information in the response. We empirically demonstrated the decomposition through controlled experiments and found that response length impacts evaluations by influencing information mass. To derive a reliable evaluation metric that assesses content quality without being confounded by response length, we propose AdapAlpaca, a simple yet effective adjustment to win rate measurement. Specifically, AdapAlpaca ensures a fair comparison of response quality by aligning the lengths of reference and test model responses under equivalent length intervals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v4">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18798v2">Distill Visual Chart Reasoning Ability from LLMs to MLLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 Accepted to EMNLP 2025 Findings. The code and dataset are publicly available at https://github.com/hewei2001/ReachQA
    </div>
    <details class="paper-abstract">
      Solving complex chart Q&A tasks requires advanced visual reasoning abilities in multimodal large language models (MLLMs), including recognizing key information from visual inputs and conducting reasoning over it. While fine-tuning MLLMs for reasoning is critical, collecting and annotating charts and questions is expensive, hard to scale, and often results in low-quality annotations. To address this, we propose Code-as-Intermediary Translation (CIT), a cost-effective, efficient and scalable data synthesis method for distilling visual reasoning abilities from LLMs to MLLMs. The code serves as an intermediary that translates visual chart representations into textual representations, enabling language models to understand cross-modal information and generate reasoning chains accordingly. In this way, we can employ text-based synthesizing techniques to expand chart-plotting code and generate high-quality Q&A pairs for training models. This produces ReachQA, a dataset containing 3k reasoning-intensive charts and 20k Q&A pairs to enhance both recognition and reasoning abilities of MLLMs. Experiments show that models fine-tuned with ReachQA not only perform well on chart-related tasks but also show performance gains on general reasoning benchmarks. The code and dataset are publicly available at https://github.com/hewei2001/ReachQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12597v2">UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicle (UAV) individual (ID) identification is a critical security surveillance strategy in low-altitude integrated sensing and communication (ISAC) networks. In this paper, we propose a novel dynamic knowledge distillation (KD)-enabled wireless radio frequency fingerprint large language model (RFF-LLM) framework for UAV ID identification. First, we propose an RFF-LLM framework based on the modified GPT-2 model to improve the identification accuracy in complex outdoor environments. Then, considering the parameter overhead of the RFF-LLM, we design a dynamic KD strategy to compress the model. Specifically, the proximal policy optimization (PPO) algorithm is employed to dynamically adjust the distillation temperature, overcoming the local optimum dilemma inherent in static KD. As a next step, the knowledge of the RFF-LLM is adequately transferred to the lightweight Lite-HRNet model. Finally, our experiments are conducted based on the self-built drone RFF dataset of Release one, namely DRFF-R1, by collecting the I/Q signals of 20 commercial UAVs in channel 149. The experiment results show that the proposed framework achieves 98.38% ID identification accuracy with merely 0.15 million parameters and 2.74 ms response time, which outperforms the benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19030v3">RECAST: Strengthening LLMs' Complex Instruction Following with Constraint-Verifiable Data</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions. To address this challenge, we propose RECAST, a novel framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 15 constraint types. Experimental results demonstrate that models fine-tuned on RECAST-30K show substantial improvements in following complex instructions. Moreover, the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21471v2">LUMIR: an LLM-Driven Unified Agent Framework for Multi-task Infrared Spectroscopy Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Infrared spectroscopy enables rapid, non destructive analysis of chemical and material properties, yet high dimensional signals and overlapping bands hinder conventional chemometric methods. Large language models (LLMs), with strong generalization and reasoning capabilities, offer new opportunities for automated spectral interpretation, but their potential in this domain remains largely untapped. This study introduces LUMIR (LLM-driven Unified agent framework for Multi-task Infrared spectroscopy Reasoning), an agent based framework designed to achieve accurate infrared spectral analysis under low data conditions. LUMIR integrates a structured literature knowledge base, automated preprocessing, feature extraction, and predictive modeling into a unified pipeline. By mining peer reviewed spectroscopy studies, it identifies validated preprocessing and feature derivation strategies, transforms spectra into low dimensional representations, and applies few-shot prompts for classification, regression, and anomaly detection. The framework was validated on diverse datasets, including the publicly available Milk near-infrared dataset, Chinese medicinal herbs, Citri Reticulatae Pericarpium(CRP) with different storage durations, an industrial wastewater COD dataset, and two additional public benchmarks, Tecator and Corn. Across these tasks, LUMIR achieved performance comparable to or surpassing established machine learning and deep learning models, particularly in resource limited settings. This work demonstrates that combining structured literature guidance with few-shot learning enables robust, scalable, and automated spectral interpretation. LUMIR establishes a new paradigm for applying LLMs to infrared spectroscopy, offering high accuracy with minimal labeled data and broad applicability across scientific and industrial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17966v2">LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 arXiv admin note: substantial text overlap with arXiv:2504.15085
    </div>
    <details class="paper-abstract">
      Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02518v2">AnalogCoder-Pro: Unifying Analog Circuit Generation and Optimization via Multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Despite recent advances, analog front-end design still relies heavily on expert intuition and iterative simulations, which limits the potential for automation. We present AnalogCoder-Pro, a multimodal large language model (LLM) framework that integrates generative and optimization techniques. The framework features a multimodal diagnosis-and-repair feedback loop that uses simulation error messages and waveform images to autonomously correct design errors. It also builds a reusable circuit tool library by archiving successful designs as modular subcircuits, accelerating the development of complex systems. Furthermore, it enables end-to-end automation by generating circuit topologies from target specifications, extracting key parameters, and applying Bayesian optimization for device sizing. On a curated benchmark suite covering 13 circuit types, AnalogCoder-Pro successfully designed 28 circuits and consistently outperformed existing LLM-based methods in figures of merit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v3">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11649v3">Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01016v1">Analysis of Error Sources in LLM-based Hypothesis Search for Few-Shot Rule Induction</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 This is the preprint version corresponding to our NeurIPS 2025 Workshop on Multimodal Algorithmic Reasoning submission
    </div>
    <details class="paper-abstract">
      Inductive reasoning enables humans to infer abstract rules from limited examples and apply them to novel situations. In this work, we compare an LLM-based hypothesis search framework with direct program generation approaches on few-shot rule induction tasks. Our findings show that hypothesis search achieves performance comparable to humans, while direct program generation falls notably behind. An error analysis reveals key bottlenecks in hypothesis generation and suggests directions for advancing program induction methods. Overall, this paper underscores the potential of LLM-based hypothesis search for modeling inductive reasoning and the challenges in building more efficient systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00973v1">Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 8 pages. Accepted for publication in the proceedings of 7th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (IEEE TPS 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00936v1">UrbanInsight: A Distributed Edge Computing Framework with LLM-Powered Data Filtering for Smart City Digital Twins</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Cities today generate enormous streams of data from sensors, cameras, and connected infrastructure. While this information offers unprecedented opportunities to improve urban life, most existing systems struggle with scale, latency, and fragmented insights. This work introduces a framework that blends physics-informed machine learning, multimodal data fusion, and knowledge graph representation with adaptive, rule-based intelligence powered by large language models (LLMs). Physics-informed methods ground learning in real-world constraints, ensuring predictions remain meaningful and consistent with physical dynamics. Knowledge graphs act as the semantic backbone, integrating heterogeneous sensor data into a connected, queryable structure. At the edge, LLMs generate context-aware rules that adapt filtering and decision-making in real time, enabling efficient operation even under constrained resources. Together, these elements form a foundation for digital twin systems that go beyond passive monitoring to provide actionable insights. By uniting physics-based reasoning, semantic data fusion, and adaptive rule generation, this approach opens new possibilities for creating responsive, trustworthy, and sustainable smart infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00930v1">SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated remarkable general reasoning capabilities. However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis. Existing benchmarks and datasets often lack the necessary variable control for multi-dimensional, systematic analysis and training, or have narrow problem types and formats. To address these limitations, we introduce SATQuest, a systematic verifier designed to evaluate and enhance logical reasoning in LLMs by generating diverse, Satisfiability-based logical reasoning problems directly from Conjunctive Normal Form (CNF) instances. SATQuest structures these problems along three orthogonal dimensions: instance scale, problem type, and question format, employing randomized, SAT-based problem generation and objective answer verification via PySAT. This design mitigates memorization issues, allows for nuanced insights into reasoning performance, and enables effective reinforcement fine-tuning. Our extensive evaluation of various LLMs using SATQuest identified significant limitations in their logical reasoning, particularly in generalizing beyond familiar mathematical formats. Furthermore, we show that reinforcement fine-tuning with SATQuest rewards substantially improves targeted task performance and generalizes to more complex instances, while highlighting remaining challenges in cross-format adaptation. Through these demonstrations, we showcase SATQuest's potential as a foundational tool and a valuable starting point for advancing LLM logical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v1">VULSOVER: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 96.29% accuracy, 96.55% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00806v1">CaresAI at BioCreative IX Track 1 -- LLM for Biomedical QA</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 Proceedings of the BioCreative IX Challenge and Workshop (BC9): Large Language Models for Clinical and Biomedical NLP at the International Joint Conference on Artificial Intelligence (IJCAI), Montreal, Canada, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly evident for accurate question answering across various domains. However, rigorous evaluation of their performance on complex question-answering (QA) capabilities is essential before deployment in real-world biomedical and healthcare applications. This paper presents our approach to the MedHopQA track of the BioCreative IX shared task, which focuses on multi-hop biomedical question answering involving diseases, genes, and chemicals. We adopt a supervised fine-tuning strategy leveraging LLaMA 3 8B, enhanced with a curated biomedical question-answer dataset compiled from external sources including BioASQ, MedQuAD, and TREC. Three experimental setups are explored: fine-tuning on combined short and long answers, short answers only, and long answers only. While our models demonstrate strong domain understanding, achieving concept-level accuracy scores of up to 0.8, their Exact Match (EM) scores remain significantly lower, particularly in the test phase. We introduce a two-stage inference pipeline for precise short-answer extraction to mitigate verbosity and improve alignment with evaluation metrics. Despite partial improvements, challenges persist in generating strictly formatted outputs. Our findings highlight the gap between semantic understanding and exact answer evaluation in biomedical LLM applications, motivating further research in output control and post-processing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00740v1">Efficient Graph Understanding with LLMs via Structured Context Injection</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong capabilities in solving problems across domains, including graph-related tasks traditionally addressed by symbolic or algorithmic methods. In this work, we present a framework for structured context injection, where task-specific information is systematically embedded in the input to guide LLMs in solving a wide range of graph problems. Our method does not require fine-tuning of LLMs, making it cost-efficient and lightweight. We observe that certain graph reasoning tasks remain challenging for LLMs unless they are mapped to conceptually grounded representations. However, achieving such mappings through fine-tuning or repeated multi-step querying can be expensive and inefficient. Our approach offers a practical alternative by injecting structured context directly into the input, enabling the LLM to implicitly align the task with grounded conceptual spaces. We evaluate the approach on multiple graph tasks using both lightweight and large models, highlighting the trade-offs between accuracy and computational cost. The results demonstrate consistent performance improvements, showing that structured input context can rival or surpass more complex approaches. Our findings underscore the value of structured context injection as an effective and scalable strategy for graph understanding with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00731v1">LLM Encoder vs. Decoder: Robust Detection of Chinese AI-Generated Text with LoRA</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      The rapid growth of large language models (LLMs) has heightened the demand for accurate detection of AI-generated text, particularly in languages like Chinese, where subtle linguistic nuances pose significant challenges to current methods. In this study, we conduct a systematic comparison of encoder-based Transformers (Chinese BERT-large and RoBERTa-wwm-ext-large), a decoder-only LLM (Alibaba's Qwen2.5-7B/DeepSeek-R1-Distill-Qwen-7B fine-tuned via Low-Rank Adaptation, LoRA), and a FastText baseline using the publicly available dataset from the NLPCC 2025 Chinese AI-Generated Text Detection Task. Encoder models were fine-tuned using a novel prompt-based masked language modeling approach, while Qwen2.5-7B was adapted for classification with an instruction-format input and a lightweight classification head trained via LoRA. Experiments reveal that although encoder models nearly memorize training data, they suffer significant performance degradation under distribution shifts (RoBERTa: 76.3% test accuracy; BERT: 79.3%). FastText demonstrates surprising lexical robustness (83.5% accuracy) yet lacks deeper semantic understanding. In contrast, the LoRA-adapted Qwen2.5-7B achieves 95.94% test accuracy with balanced precision-recall metrics, indicating superior generalization and resilience to dataset-specific artifacts. These findings underscore the efficacy of decoder-based LLMs with parameter-efficient fine-tuning for robust Chinese AI-generated text detection. Future work will explore next-generation Qwen3 models, distilled variants, and ensemble strategies to enhance cross-domain robustness further.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00728v1">A Survey on Open Dataset Search in the LLM Era: Retrospectives and Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      High-quality datasets are typically required for accomplishing data-driven tasks, such as training medical diagnosis models, predicting real-time traffic conditions, or conducting experiments to validate research hypotheses. Consequently, open dataset search, which aims to ensure the efficient and accurate fulfillment of users' dataset requirements, has emerged as a critical research challenge and has attracted widespread interest. Recent studies have made notable progress in enhancing the flexibility and intelligence of open dataset search, and large language models (LLMs) have demonstrated strong potential in addressing long-standing challenges in this area. Therefore, a systematic and comprehensive review of the open dataset search problem is essential, detailing the current state of research and exploring future directions. In this survey, we focus on recent advances in open dataset search beyond traditional approaches that rely on metadata and keywords. From the perspective of dataset modalities, we place particular emphasis on example-based dataset search, advanced similarity measurement techniques based on dataset content, and efficient search acceleration techniques. In addition, we emphasize the mutually beneficial relationship between LLMs and open dataset search. On the one hand, LLMs help address complex challenges in query understanding, semantic modeling, and interactive guidance within open dataset search. In turn, advances in dataset search can support LLMs by enabling more effective integration into retrieval-augmented generation (RAG) frameworks and data selection processes, thereby enhancing downstream task performance. Finally, we summarize open research problems and outline promising directions for future work. This work aims to offer a structured reference for researchers and practitioners in the field of open dataset search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00698v1">Learning to Shop Like Humans: A Review-driven Retrieval-Augmented Recommendation Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong potential in recommendation tasks due to their strengths in language understanding, reasoning and knowledge integration. These capabilities are especially beneficial for review-based recommendation, which relies on semantically rich user-generated texts to reveal fine-grained user preferences and item attributes. However, effectively incorporating reviews into LLM-based recommendation remains challenging due to (1) inefficient to dynamically utilize user reviews under LLMs' constrained context windows, and (2) lacking effective mechanisms to prioritize reviews most relevant to the user's current decision context. To address these challenges, we propose RevBrowse, a review-driven recommendation framework inspired by the "browse-then-decide" decision process commonly observed in online user behavior. RevBrowse integrates user reviews into the LLM-based reranking process to enhance its ability to distinguish between candidate items. To improve the relevance and efficiency of review usage, we introduce PrefRAG, a retrieval-augmented module that disentangles user and item representations into structured forms and adaptively retrieves preference-relevant content conditioned on the target item. Extensive experiments on four Amazon review datasets demonstrate that RevBrowse achieves consistent and significant improvements over strong baselines, highlighting its generalizability and effectiveness in modeling dynamic user preferences. Furthermore, since the retrieval-augmented process is transparent, RevBrowse offers a certain level of interpretability by making visible which reviews influence the final recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00647v1">LLM-HyPZ: Hardware Vulnerability Discovery using an LLM-Assisted Hybrid Platform for Zero-Shot Knowledge Extraction and Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of hardware vulnerabilities has created an urgent need for systematic and scalable analysis methods. Unlike software flaws, which are often patchable post-deployment, hardware weaknesses remain embedded across product lifecycles, posing persistent risks to processors, embedded devices, and IoT platforms. Existing efforts such as the MITRE CWE Hardware List (2021) relied on expert-driven Delphi surveys, which lack statistical rigor and introduce subjective bias, while large-scale data-driven foundations for hardware weaknesses have been largely absent. In this work, we propose LLM-HyPZ, an LLM-assisted hybrid framework for zero-shot knowledge extraction and refinement from vulnerability corpora. Our approach integrates zero-shot LLM classification, contextualized embeddings, unsupervised clustering, and prompt-driven summarization to mine hardware-related CVEs at scale. Applying LLM-HyPZ to the 2021-2024 CVE corpus (114,836 entries), we identified 1,742 hardware-related vulnerabilities. We distilled them into five recurring themes, including privilege escalation via firmware and BIOS, memory corruption in mobile and IoT systems, and physical access exploits. Benchmarking across seven LLMs shows that LLaMA 3.3 70B achieves near-perfect classification accuracy (99.5%) on a curated validation set. Beyond methodological contributions, our framework directly supported the MITRE CWE Most Important Hardware Weaknesses (MIHW) 2025 update by narrowing the candidate search space. Specifically, our pipeline surfaced 411 of the 1,026 CVEs used for downstream MIHW analysis, thereby reducing expert workload and accelerating evidence gathering. These results establish LLM-HyPZ as the first data-driven, scalable approach for systematically discovering hardware vulnerabilities, thereby bridging the gap between expert knowledge and real-world vulnerability evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03540v1">Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) methods address this issue by incorporating external knowledge from trusted sources at inference time. However, such methods typically treat knowledge as unstructured text, which limits their ability to support compositional reasoning and identify factual inconsistencies. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external information retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's latent knowledge. The graph is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse factual QA benchmarks, demonstrating consistent improvements in factual accuracy, answer precision, and interpretability over baseline prompting and static KG-augmented methods. Our findings suggest that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v2">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge (LLMaaJ) now underpins scalable evaluation, yet we lack a decisive test of a judge's qualification: can it recover a conversation's latent objective and know when that inference is trustworthy? LLMs degrade under irrelevant or long context; multi-turn jailbreaks further hide goals across turns. We introduce ObjexMT, a benchmark for objective extraction and metacognition. Given a multi-turn transcript, a model must return a one-sentence base objective and a self-reported confidence. Accuracy is computed via LLM-judge semantic similarity to gold objectives, converted to binary correctness by a single human-aligned threshold calibrated once on N = 100 items ($\tau^*=0.61$). Metacognition is evaluated with ECE, Brier, Wrong-at-High-Conf, and risk-coverage. Across gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMTData_Attack600, SafeMTData_1K, MHJ, and CoSafe, claude-sonnet-4 attains the best objective-extraction accuracy (0.515) and calibration (ECE 0.296; Brier 0.324); gpt-4.1 and Qwen3-235B-A22B-FP8 tie at 0.441 but are overconfident (mean confidence $\approx$0.88 vs. accuracy $\approx$0.44; Wrong-at-0.90 $\approx$48-52%). Performance varies by dataset ($\approx$0.167-0.865). ObjexMT thus supplies an actionable test for LLM judges: when objectives are not explicit, judges often misinfer them with high confidence. We recommend exposing objectives when feasible and gating decisions by confidence otherwise. Code and data at https://github.com/hyunjun1121/ObjexMT_dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09318v3">Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLMs can generate human-like dialogues, yet their ability to simulate early child-adult interactions remains largely unexplored. In this paper, we examined how effectively LLMs can capture the distinctive features of child-caregiver language in interaction, using both static and interactive benchmarking methods. We found that state-of-the-art LLMs like Llama 3 and GPT-4o can approximate child-caregiver dialogues at the word and utterance level, but they struggle to reproduce the child and caregiver's discursive patterns, exaggerate alignment, and fail to reach the level of diversity shown by humans. The broader goal of this work is to initiate the development of a comprehensive benchmark for LLMs in child-oriented applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16534v2">Multilingual != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accepted at OMMM@RANLP2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly capable across global languages. However, the ability to communicate across languages does not necessarily translate to appropriate cultural representations. A key concern is US-centric bias, where LLMs reflect US rather than local cultural values. We propose a novel methodology that compares LLM-generated response distributions against population-level opinion data from the World Value Survey across four languages (Danish, Dutch, English, and Portuguese). Using a rigorous linear mixed-effects regression framework, we compare two families of models: Google's Gemma models (2B--27B parameters) and successive iterations of OpenAI's turbo-series. Across the families of models, we find no consistent relationships between language capabilities and cultural alignment. While the Gemma models have a positive correlation between language capability and cultural alignment across languages, the OpenAI models do not. Importantly, we find that self-consistency is a stronger predictor of multicultural alignment than multilingual capabilities. Our results demonstrate that achieving meaningful cultural alignment requires dedicated effort beyond improving general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16807v3">Detecting LLM-assisted writing in scientific communication: Are we there yet?</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), exemplified by ChatGPT, have significantly reshaped text generation, particularly in the realm of writing assistance. While ethical considerations underscore the importance of transparently acknowledging LLM use, especially in scientific communication, genuine acknowledgment remains infrequent. A potential avenue to encourage accurate acknowledging of LLM-assisted writing involves employing automated detectors. Our evaluation of four cutting-edge LLM-generated text detectors reveals their suboptimal performance compared to a simple ad-hoc detector designed to identify abrupt writing style changes around the time of LLM proliferation. We contend that the development of specialized detectors exclusively dedicated to LLM-assisted writing detection is necessary. Such detectors could play a crucial role in fostering more authentic recognition of LLM involvement in scientific communication, addressing the current challenges in acknowledgment practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04648v2">FlairGPT: Repurposing LLMs for Interior Designs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 EUROGRAPHICS 2025
    </div>
    <details class="paper-abstract">
      Interior design involves the careful selection and arrangement of objects to create an aesthetically pleasing, functional, and harmonized space that aligns with the client's design brief. This task is particularly challenging, as a successful design must not only incorporate all the necessary objects in a cohesive style, but also ensure they are arranged in a way that maximizes accessibility, while adhering to a variety of affordability and usage considerations. Data-driven solutions have been proposed, but these are typically room- or domain-specific and lack explainability in their design design considerations used in producing the final layout. In this paper, we investigate if large language models (LLMs) can be directly utilized for interior design. While we find that LLMs are not yet capable of generating complete layouts, they can be effectively leveraged in a structured manner, inspired by the workflow of interior designers. By systematically probing LLMs, we can reliably generate a list of objects along with relevant constraints that guide their placement. We translate this information into a design layout graph, which is then solved using an off-the-shelf constrained optimization setup to generate the final layouts. We benchmark our algorithm in various design configurations against existing LLM-based methods and human designs, and evaluate the results using a variety of quantitative and qualitative metrics along with user studies. In summary, we demonstrate that LLMs, when used in a structured manner, can effectively generate diverse high-quality layouts, making them a viable solution for creating large-scale virtual scenes. Project webpage at https://flairgpt.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14533v2">Whose LLM is it Anyway? Linguistic Comparison and LLM Attribution for GPT-3.5, GPT-4 and Bard</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of generating text that is similar to or surpasses human quality. However, it is unclear whether LLMs tend to exhibit distinctive linguistic styles akin to how human authors do. Through a comprehensive linguistic analysis, we compare the vocabulary, Part-Of-Speech (POS) distribution, dependency distribution, and sentiment of texts generated by three of the most popular LLMS today (GPT-3.5, GPT-4, and Bard) to diverse inputs. The results point to significant linguistic variations which, in turn, enable us to attribute a given text to its LLM origin with a favorable 88\% accuracy using a simple off-the-shelf classification model. Theoretical and practical implications of this intriguing finding are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07227v3">LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accepted by ICCAD'25
    </div>
    <details class="paper-abstract">
      LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) techniques have been explored at the algorithmic and hardware levels. However, speculative inference results in more compute-intensive GEMM operations, creating new design trade-offs for existing GEMV-accelerated PIM architectures. Furthermore, there exists a significant amount of redundant draft tokens in tree-based speculative inference, necessitating efficient token management schemes to minimize energy consumption. In this work, we present LP-Spec, an architecture-dataflow co-design leveraging hybrid LPDDR5 performance-enhanced PIM architecture with draft token pruning and dynamic workload scheduling to accelerate LLM speculative inference. A near-data memory controller is proposed to enable data reallocation between DRAM and PIM banks. Furthermore, a data allocation unit based on the hardware-aware draft token pruner is developed to minimize energy consumption and fully exploit parallel execution opportunities. Compared to end-to-end LLM inference on other mobile solutions such as mobile NPUs or GEMV-accelerated PIMs, our LP-Spec achieves 13.21x, 7.56x, and 99.87x improvements in performance, energy efficiency, and energy-delay-product (EDP). Compared with prior AttAcc PIM and RTX 3090 GPU, LP-Spec can obtain 12.83x and 415.31x EDP reduction benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11703v2">Progent: Programmable Privilege Control for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLM agents utilize Large Language Models as central components with diverse tools to complete various user tasks, but face significant security risks when interacting with external environments. Attackers can exploit these agents through various vectors, including indirect prompt injection, memory/knowledge base poisoning, and malicious tools, tricking agents into performing dangerous actions such as unauthorized financial transactions or data leakage. The core problem that enables attacks to succeed lies in over-privileged tool access. We introduce Progent, the first privilege control framework to secure LLM agents. Progent enforces security at the tool level by restricting agents to performing tool calls necessary for user tasks while blocking potentially malicious ones. Progent features a domain-specific language that allows for expressing fine-grained policies for controlling tool privileges, flexible fallback actions when calls are blocked, and dynamic policy updates to adapt to changing agent states. The framework operates deterministically at runtime, providing provable security guarantees. Thanks to our modular design, integrating Progent does not alter agent internals and only requires minimal changes to the existing agent implementation, enhancing its practicality and potential for widespread adoption. Our extensive evaluation across various agent use cases, using benchmarks like AgentDojo, ASB, and AgentPoison, demonstrates that Progent reduces attack success rates to 0%, while preserving agent utility and speed. Additionally, we show that LLMs can automatically generate effective policies, highlighting their potential for automating the process of writing Progent's security policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19720v3">Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 emnlp 2025
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12299v2">QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accept to ACLW 2025 (WOAH)
    </div>
    <details class="paper-abstract">
      The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v4">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07268v2">Advancing Grounded Multimodal Named Entity Recognition via LLM-Based Reformulation and Box-Based Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Extension of our Findings of EMNLP 2023 & ACL 2024 paper, IEEE Transactions on Multimedia accepted on July 19, 2025
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) task aims to identify named entities, entity types and their corresponding visual regions. GMNER task exhibits two challenging attributes: 1) The tenuous correlation between images and text on social media contributes to a notable proportion of named entities being ungroundable. 2) There exists a distinction between coarse-grained noun phrases used in similar tasks (e.g., phrase localization) and fine-grained named entities. In this paper, we propose RiVEG, a unified framework that reformulates GMNER into a joint MNER-VE-VG task by leveraging large language models (LLMs) as connecting bridges. This reformulation brings two benefits: 1) It enables us to optimize the MNER module for optimal MNER performance and eliminates the need to pre-extract region features using object detection methods, thus naturally addressing the two major limitations of existing GMNER methods. 2) The introduction of Entity Expansion Expression module and Visual Entailment (VE) module unifies Visual Grounding (VG) and Entity Grounding (EG). This endows the proposed framework with unlimited data and model scalability. Furthermore, to address the potential ambiguity stemming from the coarse-grained bounding box output in GMNER, we further construct the new Segmented Multimodal Named Entity Recognition (SMNER) task and corresponding Twitter-SMNER dataset aimed at generating fine-grained segmentation masks, and experimentally demonstrate the feasibility and effectiveness of using box prompt-based Segment Anything Model (SAM) to empower any GMNER model with the ability to accomplish the SMNER task. Extensive experiments demonstrate that RiVEG significantly outperforms SoTA methods on four datasets across the MNER, GMNER, and SMNER tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12225v2">Interpretation Gaps in LLM-Assisted Comprehension of Privacy Documents</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Minor revision
    </div>
    <details class="paper-abstract">
      This article explores the gaps that can manifest when using a large language model (LLM) to obtain simplified interpretations of data practices from a complex privacy policy. We exemplify these gaps to showcase issues in accuracy, completeness, clarity and representation, while advocating for continued research to realize an LLM's true potential in revolutionizing privacy management through personal assistants and automated compliance checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00622v1">BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Time series forecasting is a long-standing and highly challenging research topic. Recently, driven by the rise of large language models (LLMs), research has increasingly shifted from purely time series methods toward harnessing textual modalities to enhance forecasting performance. However, the vast discrepancy between text and temporal data often leads current multimodal architectures to over-emphasise one modality while neglecting the other, resulting in information loss that harms forecasting performance. To address this modality imbalance, we introduce BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting), a lightweight time series forecasting framework that maintains balance between the two modalities. Specifically, raw time series are processed by the time series encoder, while descriptive statistics of raw time series are fed to an LLM with learnable prompt, producing compact textual embeddings. To ensure balanced cross-modal context alignment of time series and textual embeddings, a simple yet effective scaling strategy combined with a contrastive objective then maps these textual embeddings into the latent space of the time series embeddings. Finally, the aligned textual semantic embeddings and time series embeddings are together integrated for forecasting. Extensive experiments on standard benchmarks show that, with minimal trainable parameters, BALM-TSF achieves state-of-the-art performance in both long-term and few-shot forecasting, confirming its ability to harness complementary information from text and time series. Code is available at https://github.com/ShiqiaoZhou/BALM-TSF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00579v1">KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) demonstrate impressive potential in various practical applications. However, long context inference poses a significant challenge due to the enormous memory requirements of the key-value (KV) cache, which can scale to multiple gigabytes as sequence length and batch size increase. In this paper, we present KVComp, a generic and efficient KV cache management framework optimized for long-text generation that synergistically works with both latency-critical and throughput-critical inference systems. KVComp employs novel lossy compression techniques specifically designed for KV cache data characteristics, featuring careful co-design of compression algorithms and system architecture. Our approach maintains compatibility with the growing nature of KV cache while preserving high computational efficiency. Experimental results show that KVComp achieves on average 47\% and up to 83\% higher memory reduction rate compared to existing methods with little/no model accuracy degradation. Furthermore, KVComp achieves extremely high execution throughput, effectively reducing decompression overhead and, in some cases, even accelerating the matrix-vector multiplication operation and outperform cuBLAS-based attention kernels with less data movement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00544v1">Thinking Hard, Going Misaligned: Emergent Misalignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) becoming increasingly widely adopted, concerns regarding their safety and alignment with human values have intensified. Previous studies have shown that fine-tuning LLMs on narrow and malicious datasets induce misaligned behaviors. In this work, we report a more concerning phenomenon, Reasoning-Induced Misalignment. Specifically, we observe that LLMs become more responsive to malicious requests when reasoning is strengthened, via switching to "think-mode" or fine-tuning on benign math datasets, with dense models particularly vulnerable. Moreover, we analyze internal model states and find that both attention shifts and specialized experts in mixture-of-experts models help redirect excessive reasoning towards safety guardrails. These findings provide new insights into the emerging reasoning-safety trade-off and underscore the urgency of advancing alignment for advanced reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00543v1">Text-to-Layout: A Generative Workflow for Drafting Architectural Floor Plans Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      This paper presents the development of an AI-powered workflow that uses Large Language Models (LLMs) to assist in drafting schematic architectural floor plans from natural language prompts. The proposed system interprets textual input to automatically generate layout options including walls, doors, windows, and furniture arrangements. It combines prompt engineering, a furniture placement refinement algorithm, and Python scripting to produce spatially coherent draft plans compatible with design tools such as Autodesk Revit. A case study of a mid-sized residential layout demonstrates the approach's ability to generate functional and structured outputs with minimal manual effort. The workflow is designed for transparent replication, with all key prompt specifications documented to enable independent implementation by other researchers. In addition, the generated models preserve the full range of Revit-native parametric attributes required for direct integration into professional BIM processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00529v1">Modeling Motivated Reasoning in Law: Evaluating Strategic Role Conditioning in LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate user-tailored summaries, adapting outputs to specific stakeholders. In legal contexts, this raises important questions about motivated reasoning -- how models strategically frame information to align with a stakeholder's position within the legal system. Building on theories of legal realism and recent trends in legal practice, we investigate how LLMs respond to prompts conditioned on different legal roles (e.g., judges, prosecutors, attorneys) when summarizing judicial decisions. We introduce an evaluation framework grounded in legal fact and reasoning inclusion, also considering favorability towards stakeholders. Our results show that even when prompts include balancing instructions, models exhibit selective inclusion patterns that reflect role-consistent perspectives. These findings raise broader concerns about how similar alignment may emerge as LLMs begin to infer user roles from prior interactions or context, even without explicit role instructions. Our results underscore the need for role-aware evaluation of LLM summarization behavior in high-stakes legal settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00510v1">LLM-Assisted Iterative Evolution with Swarm Intelligence Toward SuperBrain</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We propose a novel SuperBrain framework for collective intelligence, grounded in the co-evolution of large language models (LLMs) and human users. Unlike static prompt engineering or isolated agent simulations, our approach emphasizes a dynamic pathway from Subclass Brain to Superclass Brain: (1) A Subclass Brain arises from persistent, personalized interaction between a user and an LLM, forming a cognitive dyad with adaptive learning memory. (2) Through GA-assisted forward-backward evolution, these dyads iteratively refine prompts and task performance. (3) Multiple Subclass Brains coordinate via Swarm Intelligence, optimizing across multi-objective fitness landscapes and exchanging distilled heuristics. (4) Their standardized behaviors and cognitive signatures integrate into a Superclass Brain, an emergent meta-intelligence capable of abstraction, generalization and self-improvement. We outline the theoretical constructs, present initial implementations (e.g., UAV scheduling, KU/KI keyword filtering) and propose a registry for cross-dyad knowledge consolidation. This work provides both a conceptual foundation and an architectural roadmap toward scalable, explainable and ethically aligned collective AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00483v1">Exploring Decision-Making Capabilities of LLM Agents: An Experimental Study on Jump-Jump Game</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      The Jump-Jump game, as a simple yet challenging casual game, provides an ideal testing environment for studying LLM decision-making capabilities. The game requires players to precisely control jumping force based on current position and target platform distance, involving multiple cognitive aspects including spatial reasoning, physical modeling, and strategic planning. It illustrates the basic gameplay mechanics of the Jump-Jump game, where the player character (red circle) must jump across platforms with appropriate force to maximize score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00482v1">Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 17 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This report investigates approaches for prompting a tool-augmented large language model (LLM) to act as a role-playing dialogue agent in the API track of the Commonsense Persona-grounded Dialogue Challenge (CPDC) 2025. In this setting, dialogue agents often produce overly long in-character responses (over-speaking) while failing to use tools effectively according to the persona (under-acting), such as generating function calls that do not exist or making unnecessary tool calls before answering. We explore four prompting approaches to address these issues: 1) basic role prompting, 2) human-crafted role prompting, 3) automatic prompt optimization (APO), and 4) rule-based role prompting. The rule-based role prompting (RRP) approach achieved the best performance through two novel techniques--character-card/scene-contract design and strict enforcement of function calling--which led to an overall score of 0.571, improving on the zero-shot baseline score of 0.519. These findings demonstrate that RRP design can substantially improve the effectiveness and reliability of role-playing dialogue agents compared with more elaborate methods such as APO. To support future efforts in developing persona prompts, we are open-sourcing all of our best-performing prompts and the APO tool. Source code is available at https://github.com/scb-10x/apo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00461v1">TECP: Token-Entropy Conformal Prediction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00425v1">The Gold Medals in an Empty Room: Diagnosing Metalinguistic Reasoning in LLMs with Camlang</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve gold-medal performance across many benchmarks, yet it remains unclear whether such success reflects genuine reasoning or pattern matching. From a cognitive science perspective, an informative test is whether models can master an unfamiliar language through explicit metalinguistic deductive learning, a paradigm where human learners can reliably internalise grammatical systems through metalinguistic reasoning. We address this question with Camlang, a novel constructed language that exhibits naturalistic yet unattested feature combinations. Camlang consists of two explicit resources, a grammar book and a bilingual dictionary, which mirror adult second-language learning via explicit grammar rules and lexical lookup, and enable us to disentangle errors in morpho-syntax, lexical semantics, and sentence-level reasoning. Human experiments show that these resources are sufficient for participants to acquire Camlang and successfully solve Camlang tasks. To operationalise evaluation, we adapt CommonsenseQA into Camlang, creating Camlang-CSQA-v0, the first task in a broader suite where solving questions requires applying grammar rules and lexical mappings. Experimental results show that GPT-5 achieves 98\% EM accuracy in English but only 47\% in Camlang, far below human performance at 87\%, while other state-of-the-art reasoning LLMs perform even worse. Human verification further reveals that most model successes stem from shallow lexical alignment while GPT-5 shows emerging metalinguistic awareness to a limited extent but not systematic grammatical mastery as humans. Camlang establishes a cognitively grounded evaluation paradigm that exposes fundamental gaps between current models and human metalinguistic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00347v1">LLM-Driven Policy Diffusion: Enhancing Generalization in Offline Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) is known for its strong decision-making capabilities and has been widely applied in various real-world scenarios. However, with the increasing availability of offline datasets and the lack of well-designed online environments from human experts, the challenge of generalization in offline RL has become more prominent. Due to the limitations of offline data, RL agents trained solely on collected experiences often struggle to generalize to new tasks or environments. To address this challenge, we propose LLM-Driven Policy Diffusion (LLMDPD), a novel approach that enhances generalization in offline RL using task-specific prompts. Our method incorporates both text-based task descriptions and trajectory prompts to guide policy learning. We leverage a large language model (LLM) to process text-based prompts, utilizing its natural language understanding and extensive knowledge base to provide rich task-relevant context. Simultaneously, we encode trajectory prompts using a transformer model, capturing structured behavioral patterns within the underlying transition dynamics. These prompts serve as conditional inputs to a context-aware policy-level diffusion model, enabling the RL agent to generalize effectively to unseen tasks. Our experimental results demonstrate that LLMDPD outperforms state-of-the-art offline RL methods on unseen tasks, highlighting its effectiveness in improving generalization and adaptability in diverse settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00631v2">ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Instruction tuning has underscored the significant potential of large language models (LLMs) in producing more human controllable and effective outputs in various domains. In this work, we focus on the data selection problem for task-specific instruction tuning of LLMs. Prevailing methods primarily rely on the crafted similarity metrics to select training data that aligns with the test data distribution. The goal is to minimize instruction tuning loss on the test data, ultimately improving performance on the target task. However, it has been widely observed that instruction tuning loss (i.e., cross-entropy loss for next token prediction) in LLMs often fails to exhibit a monotonic relationship with actual task performance. This misalignment undermines the effectiveness of current data selection methods for task-specific instruction tuning. To address this issue, we introduce ROSE, a novel Reward-Oriented inStruction data sElection method which leverages pairwise preference loss as a reward signal to optimize data selection for task-specific instruction tuning. Specifically, ROSE adapts an influence formulation to approximate the influence of training data points relative to a few-shot preference validation set to select the most task-related training data points. Experimental results show that by selecting just 5\% of the training data using ROSE, our approach can achieve competitive results compared to fine-tuning with the full training dataset, and it surpasses other state-of-the-art data selection methods for task-specific instruction tuning. Our qualitative analysis further confirms the robust generalizability of our method across multiple benchmark datasets and diverse model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17202v2">Active Domain Knowledge Acquisition with 100-Dollar Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21803v1">Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to The 16th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2025)(Poster Paper)
    </div>
    <details class="paper-abstract">
      Accurate interpretation of clinical narratives is critical for patient care, but the complexity of these notes makes automation challenging. While Large Language Models (LLMs) show promise, single-model approaches can lack the robustness required for high-stakes clinical tasks. We introduce a collaborative multi-agent system (MAS) that models a clinical consultation team to address this gap. The system is tasked with identifying clinical problems by analyzing only the Subjective (S) and Objective (O) sections of SOAP notes, simulating the diagnostic reasoning process of synthesizing raw data into an assessment. A Manager agent orchestrates a dynamically assigned team of specialist agents who engage in a hierarchical, iterative debate to reach a consensus. We evaluated our MAS against a single-agent baseline on a curated dataset of 420 MIMIC-III notes. The dynamic multi-agent configuration demonstrated consistently improved performance in identifying congestive heart failure, acute kidney injury, and sepsis. Qualitative analysis of the agent debates reveals that this structure effectively surfaces and weighs conflicting evidence, though it can occasionally be susceptible to groupthink. By modeling a clinical team's reasoning process, our system offers a promising path toward more accurate, robust, and interpretable clinical decision support tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21801v1">DMGIN: How Multimodal LLMs Enhance Large Recommendation Models for Lifelong User Post-click Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Modeling user interest based on lifelong user behavior sequences is crucial for enhancing Click-Through Rate (CTR) prediction. However, long post-click behavior sequences themselves pose severe performance issues: the sheer volume of data leads to high computational costs and inefficiencies in model training and inference. Traditional methods address this by introducing two-stage approaches, but this compromises model effectiveness due to incomplete utilization of the full sequence context. More importantly, integrating multimodal embeddings into existing large recommendation models (LRM) presents significant challenges: These embeddings often exacerbate computational burdens and mismatch with LRM architectures. To address these issues and enhance the model's efficiency and accuracy, we introduce Deep Multimodal Group Interest Network (DMGIN). Given the observation that user post-click behavior sequences contain a large number of repeated items with varying behaviors and timestamps, DMGIN employs Multimodal LLMs(MLLM) for grouping to reorganize complete lifelong post-click behavior sequences more effectively, with almost no additional computational overhead, as opposed to directly introducing multimodal embeddings. To mitigate the potential information loss from grouping, we have implemented two key strategies. First, we analyze behaviors within each group using both interest statistics and intra-group transformers to capture group traits. Second, apply inter-group transformers to temporally ordered groups to capture the evolution of user group interests. Our extensive experiments on both industrial and public datasets confirm the effectiveness and efficiency of DMGIN. The A/B test in our LBS advertising system shows that DMGIN improves CTR by 4.7% and Revenue per Mile by 2.3%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v4">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17196v2">BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v3">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Under review. 19 pages, 8 figures, 12 tables. Code and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning. The code (https://github.com/yyysjz1997/Time-RA) and dataset (https://huggingface.co/datasets/Time-RA/RATs40K) have been fully open-sourced to support and accelerate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v2">Robustness is Important: Limitations of LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21628v1">Personality Matters: User Traits Predict LLM Preferences in Multi-Turn Collaborative Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly integrate into everyday workflows, where users shape outcomes through multi-turn collaboration, a critical question emerges: do users with different personality traits systematically prefer certain LLMs over others? We conducted a study with 32 participants evenly distributed across four Keirsey personality types, evaluating their interactions with GPT-4 and Claude 3.5 across four collaborative tasks: data analysis, creative writing, information retrieval, and writing assistance. Results revealed significant personality-driven preferences: Rationals strongly preferred GPT-4, particularly for goal-oriented tasks, while idealists favored Claude 3.5, especially for creative and analytical tasks. Other personality types showed task-dependent preferences. Sentiment analysis of qualitative feedback confirmed these patterns. Notably, aggregate helpfulness ratings were similar across models, showing how personality-based analysis reveals LLM differences that traditional evaluations miss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v1">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted by EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21561v1">Summarize-Exemplify-Reflect: Data-driven Insight Distillation Empowers LLMs for Few-shot Tabular Classification</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 25 Findings
    </div>
    <details class="paper-abstract">
      Recent studies show the promise of large language models (LLMs) for few-shot tabular classification but highlight challenges due to the variability in structured data. To address this, we propose distilling data into actionable insights to enable robust and effective classification by LLMs. Drawing inspiration from human learning processes, we introduce InsightTab, an insight distillation framework guided by principles of divide-and-conquer, easy-first, and reflective learning. Our approach integrates rule summarization, strategic exemplification, and insight reflection through deep collaboration between LLMs and data modeling techniques. The obtained insights enable LLMs to better align their general knowledge and capabilities with the particular requirements of specific tabular tasks. We extensively evaluate InsightTab on nine datasets. The results demonstrate consistent improvement over state-of-the-art methods. Ablation studies further validate the principle-guided distillation process, while analyses emphasize InsightTab's effectiveness in leveraging labeled data and managing bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21540v1">HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Process mining has emerged as a powerful analytical technique for understanding complex healthcare workflows. However, its application faces significant barriers, including technical complexity, a lack of standardized approaches, and limited access to practical training resources. We introduce HealthProcessAI, a GenAI framework designed to simplify process mining applications in healthcare and epidemiology by providing a comprehensive wrapper around existing Python (PM4PY) and R (bupaR) libraries. To address unfamiliarity and improve accessibility, the framework integrates multiple Large Language Models (LLMs) for automated process map interpretation and report generation, helping translate technical analyses into outputs that diverse users can readily understand. We validated the framework using sepsis progression data as a proof-of-concept example and compared the outputs of five state-of-the-art LLM models through the OpenRouter platform. To test its functionality, the framework successfully processed sepsis data across four proof-of-concept scenarios, demonstrating robust technical performance and its capability to generate reports through automated LLM analysis. LLM evaluation using five independent LLMs as automated evaluators revealed distinct model strengths: Claude Sonnet-4 and Gemini 2.5-Pro achieved the highest consistency scores (3.79/4.0 and 3.65/4.0) when evaluated by automated LLM assessors. By integrating multiple Large Language Models (LLMs) for automated interpretation and report generation, the framework addresses widespread unfamiliarity with process mining outputs, making them more accessible to clinicians, data scientists, and researchers. This structured analytics and AI-driven interpretation combination represents a novel methodological advance in translating complex process mining results into potentially actionable insights for healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21512v1">Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17052v2">Testing Conviction: An Argumentative Framework for Measuring LLM Political Stability</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape political discourse, yet exhibit inconsistent responses when challenged. While prior research categorizes LLMs as left- or right-leaning based on single-prompt responses, a critical question remains: Do these classifications reflect stable ideologies or superficial mimicry? Existing methods cannot distinguish between genuine ideological alignment and performative text generation. To address this, we propose a framework for evaluating ideological depth through (1) argumentative consistency and (2) uncertainty quantification. Testing 12 LLMs on 19 economic policies from the Political Compass Test, we classify responses as stable or performative ideological positioning. Results show 95% of left-leaning models and 89% of right-leaning models demonstrate behavior consistent with our classifications across different experimental conditions. Furthermore, semantic entropy strongly validates our classifications (AUROC=0.78), revealing uncertainty's relationship to ideological consistency. Our findings demonstrate that ideological stability is topic-dependent and challenge the notion of monolithic LLM ideologies, and offer a robust way to distinguish genuine alignment from performative behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21476v1">Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at https://github.com/weixiaolong94-hub/Igniting-Creative-Writing-in-Small-Language-Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20863v2">Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21452v1">From Canonical to Complex: Benchmarking LLM Capabilities in Undergraduate Thermodynamics</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Benchmark downloadable at https://huggingface.co/datasets/herteltm/UTQA
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly considered as tutoring aids in science education. Yet their readiness for unsupervised use in undergraduate instruction remains uncertain, as reliable teaching requires more than fluent recall: it demands consistent, principle-grounded reasoning. Thermodynamics, with its compact laws and subtle distinctions between state and path functions, reversibility, and entropy, provides an ideal testbed for evaluating such capabilities. Here we present UTQA, a 50-item undergraduate thermodynamics question answering benchmark, covering ideal-gas processes, reversibility, and diagram interpretation. No leading 2025-era model exceeded our 95\% competence threshold: the best LLMs achieved 82\% accuracy, with text-only items performing better than image reasoning tasks, which often fell to chance levels. Prompt phrasing and syntactic complexity showed modest to little correlation with performance. The gap concentrates in finite-rate/irreversible scenarios and in binding visual features to thermodynamic meaning, indicating that current LLMs are not yet suitable for unsupervised tutoring in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v1">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21417v1">An Empirical Study of Vulnerable Package Dependencies in LLM Repositories</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have developed rapidly in recent years, revolutionizing various fields. Despite their widespread success, LLMs heavily rely on external code dependencies from package management systems, creating a complex and interconnected LLM dependency supply chain. Vulnerabilities in dependencies can expose LLMs to security risks. While existing research predominantly focuses on model-level security threats, vulnerabilities within the LLM dependency supply chain have been overlooked. To fill this gap, we conducted an empirical analysis of 52 open-source LLMs, examining their third-party dependencies and associated vulnerabilities. We then explored activities within the LLM repositories to understand how maintainers manage third-party vulnerabilities in practice. Finally, we compared third-party dependency vulnerabilities in the LLM ecosystem to those in the Python ecosystem. Our results show that half of the vulnerabilities in the LLM ecosystem remain undisclosed for more than 56.2 months, significantly longer than those in the Python ecosystem. Additionally, 75.8% of LLMs include vulnerable dependencies in their configuration files. This study advances the understanding of LLM supply chain risks, provides insights for practitioners, and highlights potential directions for improving the security of the LLM supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v3">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21378v1">RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17178v3">SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/zjukg/SKA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v5">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21323v1">LLM-driven Provenance Forensics for Threat Investigation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      We introduce PROVSEEK, an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction. PROVSEEK employs specialized toolchains to dynamically retrieve relevant context by generating precise, context-aware queries that fuse a vectorized threat report knowledge base with data from system provenance databases. The framework resolves provenance queries, orchestrates multiple role-specific agents to mitigate hallucinations, and synthesizes structured, ground-truth verifiable forensic summaries. By combining agent orchestration with Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning, PROVSEEK enables adaptive multi-step analysis that iteratively refines hypotheses, verifies supporting evidence, and produces scalable, interpretable forensic explanations of attack behaviors. By combining provenance data with agentic reasoning, PROVSEEK establishes a new paradigm for grounded agentic forecics to investigate APTs. We conduct a comprehensive evaluation on publicly available DARPA datasets, demonstrating that PROVSEEK outperforms retrieval-based methods for intelligence extraction task, achieving a 34% improvement in contextual precision/recall; and for threat detection task, PROVSEEK achieves 22%/29% higher precision/recall compared to both a baseline agentic AI approach and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection System (PIDS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21305v1">LLM-Supported Content Analysis of Motivated Reasoning on Climate Change</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 11 pages, 3 figures. Accepted for ASIS&T 2025
    </div>
    <details class="paper-abstract">
      Public discourse around climate change remains polarized despite scientific consensus on anthropogenic climate change (ACC). This study examines how "believers" and "skeptics" of ACC differ in their YouTube comment discourse. We analyzed 44,989 comments from 30 videos using a large language model (LLM) as a qualitative annotator, identifying ten distinct topics. These annotations were combined with social network analysis to examine engagement patterns. A linear mixed-effects model showed that comments about government policy and natural cycles generated significantly lower interaction compared to misinformation, suggesting these topics are ideologically settled points within communities. These patterns reflect motivated reasoning, where users selectively engage with content that aligns with their identity and beliefs. Our findings highlight the utility of LLMs for large-scale qualitative analysis and highlight how climate discourse is shaped not only by content, but by underlying cognitive and ideological motivations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21300v1">Improving Fisher Information Estimation and Efficiency for LoRA-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated remarkable performance across various tasks but face challenges related to unintentionally generating outputs containing sensitive information. A straightforward approach to address this issue is to retrain the model after excluding the problematic data. However, this approach incurs prohibitively high computational costs. To overcome this limitation, machine unlearning has emerged as a promising solution that can effectively remove sensitive information without the need to retrain the model from scratch. Recently, FILA has been proposed as a parameter-efficient unlearning method by integrating LoRA adapters. Specifically, it calculates the Fisher information to identify parameters associated with the forget set and assigns them to LoRA adapters for updates. Despite its innovative approach, FILA still requires access to all model parameters and does not adequately account for fundamental assumptions underlying Fisher information, leading to inaccuracies in importance estimation. To address these limitations, we propose VILA, a novel unlearning framework that explicitly considers the assumptions overlooked in FILA, thereby enhancing the accuracy of parameter identification for the forget set. Moreover, VILA significantly reduces computational costs by enabling parameter identification without accessing the entire model. Our method achieves up to 100x higher parameter efficiency and 40x faster training speed compared to FILA, and sets new state-of-the-art performance on benchmarks including TOFU, WMDP, and MUSE. Our code is available at https://github.com/kyj93790/VILA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21285v1">A Financial Brain Scan of the LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 47 pages
    </div>
    <details class="paper-abstract">
      Emerging techniques in computer science make it possible to "brain scan" large language models (LLMs), identify the plain-English concepts that guide their reasoning, and steer them while holding other factors constant. We show that this approach can map LLM-generated economic forecasts to concepts such as sentiment, technical analysis, and timing, and compute their relative importance without reducing performance. We also show that models can be steered to be more or less risk-averse, optimistic, or pessimistic, which allows researchers to correct or simulate biases. The method is transparent, lightweight, and replicable for empirical research in the social sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v4">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in list{\ttfamily .}get(i). Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21273v1">CALM: A Framework for Continuous, Adaptive, and LLM-Mediated Anomaly Detection in Time-Series Streams</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      The detection of anomalies in non-stationary time-series streams is a critical but challenging task across numerous industrial and scientific domains. Traditional models, trained offline, suffer significant performance degradation when faced with concept drift, where the underlying statistical properties of the data change over time. This paper introduces CALM (Continuous, Adaptive, and LLM-Mediated), a novel, end-to-end framework for real-time anomaly detection designed to address this challenge. CALM is built on the Apache Beam distributed processing framework and leverages the TimesFm foundation model for forecasting-based anomaly detection. The framework's novelty lies in two core contributions. First, it implements a closed-loop, continuous fine-tuning mechanism that allows the anomaly detection model to adapt to evolving data patterns in near real-time. Second, it introduces an LLM-as-a-Judge component, a Large Language Model that provides semantic, context-aware judgments on detected anomalies to curate a high-quality training dataset, deciding whether an anomaly represents transient noise or a meaningful pattern shift. We evaluate CALM on the comprehensive TSB-UAD benchmark. Our results demonstrate that the continuously fine-tuned model improves the ROC AUC score in most datasets compared to the static, pre-trained base model, validating the efficacy of our adaptive, LLM-guided approach to maintaining high-performance anomaly detection in dynamic streaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00273v2">Echoes in AI: Quantifying lack of plot diversity in LLM outputs</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 PNAS Vol. 122 No. 35. Copyright \c{opyright} 2025 the Author(s). Published by PNAS. This open access article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND)
    </div>
    <details class="paper-abstract">
      With rapid advances in large language models (LLMs), there has been an increasing application of LLMs in creative content ideation and generation. A critical question emerges: can current LLMs provide ideas that are diverse enough to truly bolster collective creativity? We examine two state-of-the-art LLMs, GPT-4 and LLaMA-3, on story generation and discover that LLM-generated stories often consist of plot elements that are echoed across a number of generations. To quantify this phenomenon, we introduce the Sui Generis score, an automatic metric that measures the uniqueness of a plot element among alternative storylines generated using the same prompt under an LLM. Evaluating on 100 short stories, we find that LLM-generated stories often contain combinations of idiosyncratic plot elements echoed frequently across generations and across different LLMs, while plots from the original human-written stories are rarely recreated or even echoed in pieces. Moreover, our human evaluation shows that the ranking of Sui Generis scores among story segments correlates moderately with human judgment of surprise level, even though score computation is completely automatic without relying on human judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00256v1">LLM-Based Program Generation for Triggering Numerical Inconsistencies Across Compilers</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Floating-point inconsistencies across compilers can undermine the reliability of numerical software. We present LLM4FP, the first framework that uses Large Language Models (LLMs) to generate floating-point programs specifically designed to trigger such inconsistencies. LLM4FP combines Grammar-Based Generation and Feedback-Based Mutation to produce diverse and valid programs. We evaluate LLM4FP across multiple compilers and optimization levels, measuring inconsistency rate, time cost, and program diversity. LLM4FP detects over twice as many inconsistencies compared to the state-of-the-art tool, Varity. Notably, most of the inconsistencies involve real-valued differences, rather than extreme values like NaN or infinities. LLM4FP also uncovers inconsistencies across a wider range of optimization levels, and finds the most mismatches between host and device compilers. These results show that LLM-guided program generation improves the detection of numerical inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00245v1">The Rarity Blind Spot: A Framework for Evaluating Statistical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Effective decision-making often relies on identifying what makes each candidate distinctive. While existing benchmarks for LLMs emphasize retrieving or summarizing information relevant to a given query, they do not evaluate a model's ability to identify globally distinctive features across a set of documents. We introduce Distinctive Feature Mining (DFM), a new task that challenges models to analyze a small-to-medium collection (10-40 documents) and surface features that are rare in the global context (e.g., appearing in less than 10% of documents). This setting mirrors real-world scenarios such as candidate selection or product differentiation, where statistical reasoning, not retrieval, is key. To enable systematic evaluation of this capability, we present DiFBench, a configurable benchmark creation framework with controllable parameters such as document set size and distinctiveness thresholds. Using DiFBench, we perform a large-scale assessment of distinctive feature mining across ten state-of-the-art LLMs. Our findings reveal a significant performance gap between general-purpose and reasoning-enhanced models. All models, however, substantially degrade as the task complexity and document count increase. We also find that a common failure mode is misidentifying frequent features as distinctive. These insights reveal core limitations in contemporary LLMs' abilities to perform fine-grained, statistical reasoning and rarity detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00217v1">Learning to Shard: RL for Co-optimizing the Parallelism Degrees and Per-operator Sharding Dimensions in Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Distributed LLM inference requires careful coordination of parallelization strategies across hundreds to thousands of NPUs to meet production SLOs. Current systems like Megatron-LM rely on static heuristics that separately configure parallelism degrees and per-operator sharding dimensions, leaving significant performance on the table as models scale and hardware topologies diversify. We introduce Learn to Shard, to our knowledge, the first RL-based approach to co-optimize both coarse-grained parallelism degrees and fine-grained per-operator sharding dimensions for distributed LLM inference. Our method employs an attention-based policy over an elite history that learns from high-performing strategies to efficiently navigate the vast combinatorial search space. Evaluated on H100 clusters with MoE models up to 1.6T parameters, Learn to Shard achieves up to 3.5x throughput improvement over metaheuristic baselines and 1.06x over Megatron heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00140v1">LLM-based Triplet Extraction for Automated Ontology Generation in Software Engineering Standards</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Ontologies have supported knowledge representation and whitebox reasoning for decades; thus, the automated ontology generation (AOG) plays a crucial role in scaling their use. Software engineering standards (SES) consist of long, unstructured text (with high noise) and paragraphs with domain-specific terms. In this setting, relation triple extraction (RTE), together with term extraction, constitutes the first stage toward AOG. This work proposes an open-source large language model (LLM)-assisted approach to RTE for SES. Instead of solely relying on prompt-engineering-based methods, this study promotes the use of LLMs as an aid in constructing ontologies and explores an effective AOG workflow that includes document segmentation, candidate term mining, LLM-based relation inference, term normalization, and cross-section alignment. Golden-standard benchmarks at three granularities are constructed and used to evaluate the ontology generated from the study. The results show that it is comparable and potentially superior to the OpenIE method of triple extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02666v2">A Survey on Progress in LLM Alignment from the Perspective of Reward Design</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reward design plays a pivotal role in aligning large language models (LLMs) with human values, serving as the bridge between feedback signals and model optimization. This survey provides a structured organization of reward modeling and addresses three key aspects: mathematical formulation, construction practices, and interaction with optimization paradigms. Building on this, it develops a macro-level taxonomy that characterizes reward mechanisms along complementary dimensions, thereby offering both conceptual clarity and practical guidance for alignment research. The progression of LLM alignment can be understood as a continuous refinement of reward design strategies, with recent developments highlighting paradigm shifts from reinforcement learning (RL)-based to RL-free optimization and from single-task to multi-objective and complex settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00132v1">CoComposer: LLM Multi-agent Collaborative Music Composition</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Existing AI Music composition tools are limited in generation duration, musical quality, and controllability. We introduce CoComposer, a multi-agent system that consists of five collaborating agents, each with a task based on the traditional music composition workflow. Using the AudioBox-Aesthetics system, we experimentally evaluate CoComposer on four compositional criteria. We test with three LLMs (GPT-4o, DeepSeek-V3-0324, Gemini-2.5-Flash), and find (1) that CoComposer outperforms existing multi-agent LLM-based systems in music quality, and (2) compared to a single-agent system, in production complexity. Compared to non- LLM MusicLM, CoComposer has better interpretability and editability, although MusicLM still produces better music.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02594v1">OpenAIs HealthBench in Action: Evaluating an LLM-Based Medical Assistant on Realistic Clinical Queries</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 13 pages, two graphs
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on their ability to generate high-quality, accurate, situationally aware answers to clinical questions requires going beyond conventional benchmarks to assess how these systems behave in complex, high-stake clincal scenarios. Traditional evaluations are often limited to multiple-choice questions that fail to capture essential competencies such as contextual reasoning, awareness and uncertainty handling etc. To address these limitations, we evaluate our agentic, RAG-based clinical support assistant, DR.INFO, using HealthBench, a rubric-driven benchmark composed of open-ended, expert-annotated health conversations. On the Hard subset of 1,000 challenging examples, DR.INFO achieves a HealthBench score of 0.51, substantially outperforming leading frontier LLMs (GPT-5, o3, Grok 3, GPT-4, Gemini 2.5, etc.) across all behavioral axes (accuracy, completeness, instruction following, etc.). In a separate 100-sample evaluation against similar agentic RAG assistants (OpenEvidence, Pathway.md), it maintains a performance lead with a health-bench score of 0.54. These results highlight DR.INFOs strengths in communication, instruction following, and accuracy, while also revealing areas for improvement in context awareness and completeness of a response. Overall, the findings underscore the utility of behavior-level, rubric-based evaluation for building a reliable and trustworthy AI-enabled clinical support assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00125v1">Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20810v1">A Graph-Based Test-Harness for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 4 pages, 2 figures, dataset
    </div>
    <details class="paper-abstract">
      We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at https://github.com/jessicalundin/graph_testing_harness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08846v2">Steering Towards Fairness: Mitigating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at CASE@RANLP2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases along political and economic dimensions. In this paper, we employ a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), this method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v3">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [Accepted]. To be held on 26th of October, 2025
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20764v1">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025,14 page,3 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we conduct the first comparative analysis of emotional arcs between real and LLM-generated Cognitive Behavioral Therapy dialogues. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions transcribed from public videos and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability,more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity between real and synthetic speakers is low, especially for clients. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. We introduce RealCBT, a curated dataset of real CBT sessions, to support future research in this space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20750v1">Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection across Datasets</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Paper accepted at the DHOW Workshop at ACM Multimedia 2025. Code available at https://github.com/idiap/implicit-hsd
    </div>
    <details class="paper-abstract">
      Implicit hate speech (IHS) is indirect language that conveys prejudice or hatred through subtle cues, sarcasm or coded terminology. IHS is challenging to detect as it does not include explicit derogatory or inflammatory words. To address this challenge, task-specific pipelines can be complemented with external knowledge or additional information such as context, emotions and sentiment data. In this paper, we show that, by solely fine-tuning recent general-purpose embedding models based on large language models (LLMs), such as Stella, Jasper, NV-Embed and E5, we achieve state-of-the-art performance. Experiments on multiple IHS datasets show up to 1.10 percentage points improvements for in-dataset, and up to 20.35 percentage points improvements in cross-dataset evaluation, in terms of F1-macro score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20744v1">From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Context: Laws and regulations increasingly affect software design and quality assurance, but legal texts are written in technology-neutral language. This creates challenges for engineers who must develop compliance artifacts such as requirements and acceptance criteria. Manual creation is labor-intensive, error-prone, and requires domain expertise. Advances in Generative AI (GenAI), especially Large Language Models (LLMs), offer a way to automate deriving such artifacts. Objective: We present the first systematic human-subject study of LLMs' ability to derive behavioral specifications from legal texts using a quasi-experimental design. These specifications translate legal requirements into a developer-friendly form. Methods: Ten participants evaluated specifications generated from food-safety regulations by Claude and Llama. Using Gherkin, a structured BDD language, 60 specifications were produced. Each participant assessed 12 across five criteria: Relevance, Clarity, Completeness, Singularity, and Time Savings. Each specification was reviewed by two participants, yielding 120 assessments. Results: For Relevance, 75% of ratings were highest and 20% second-highest. Clarity reached 90% highest. Completeness: 75% highest, 19% second. Singularity: 82% highest, 12% second. Time Savings: 68% highest, 24% second. No lowest ratings occurred. Mann-Whitney U tests showed no significant differences across participants or models. Llama slightly outperformed Claude in Clarity, Completeness, and Time Savings, while Claude was stronger in Singularity. Feedback noted hallucinations and omissions but confirmed the utility of the specifications. Conclusion: LLMs can generate high-quality Gherkin specifications from legal texts, reducing manual effort and providing structured artifacts useful for implementation, assurance, and test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20737v1">Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18321v2">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: https://github.com/declare-lab/KAIROS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20697v1">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20643v1">CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Code: https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.
    </details>
</div>
