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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- Part 19
- [Part 20](papers_20.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18262v3">Break the ID-Language Barrier: An Adaption Framework for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The recent breakthrough of large language models (LLMs) in natural language processing has sparked exploration in recommendation systems, however, their limited domain-specific knowledge remains a critical bottleneck. Specifically, LLMs lack key pieces of information crucial for sequential recommendations, such as user behavior patterns. To address this critical gap, we propose IDLE-Adapter, a novel framework that integrates pre-trained ID embeddings, rich in domain-specific knowledge, into LLMs to improve recommendation accuracy. IDLE-Adapter acts as a bridge, transforming sparse user-item interaction data into dense, LLM-compatible representations through a Pre-trained ID Sequential Model, Dimensionality Alignment, Layer-wise Embedding Refinement, and Layer-wise Distribution Alignment. Furthermore, IDLE-Adapter demonstrates remarkable flexibility by seamlessly integrating ID embeddings from diverse ID-based sequential models and LLM architectures. Extensive experiments across various datasets demonstrate the superiority of IDLE-Adapter, achieving over 10\% and 20\% improvements in HitRate@5 and NDCG@5 metrics, respectively, compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02047v1">LLM-Enhanced, Data-Driven Personalized and Equitable Clinician Scheduling: A Predict-then-Optimize Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 10 pages, 5 figures, Accepted to IEEE ICDM 2025 Workshops Proceedings; IEEE Computer Society Press
    </div>
    <details class="paper-abstract">
      Clinician scheduling remains a persistent challenge due to limited clinical resources and fluctuating demands. This complexity is especially acute in large academic anesthesiology departments as physicians balance responsibilities across multiple clinical sites with conflicting priorities. Further, scheduling must account for individual clinical and lifestyle preferences to ensure job satisfaction and well-being. Traditional approaches, often based on statistical or rule-based optimization models, rely on structured data and explicit domain knowledge. However, these methods often overlook unstructured information, e.g., free-text notes from routinely administered clinician well-being surveys and scheduling platforms. These notes may reveal implicit and underutilized clinical resources. Neglecting such information can lead to misaligned schedules, increased burnout, overlooked staffing flexibility, and suboptimal utilization of available resources. To address this gap, we propose a predict-then-optimize framework that integrates classification-based clinician availability predictions with a mixed-integer programming schedule optimization model. Large language models (LLMs) are employed to extract actionable preferences and implicit constraints from unstructured schedule notes, enhancing the reliability of availability predictions. These predictions then inform the schedule optimization considering four objectives: first, ensuring clinical full-time equivalent compliance, second, reducing workload imbalances by enforcing equitable proportions of shift types, third, maximizing clinician availability for assigned shifts, and fourth, schedule consistency. By combining the interpretive power of LLMs with the rigor of mathematical optimization, our framework provides a robust, data-driven solution that enhances operational efficiency while supporting equity and clinician well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v3">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17674v2">Push the Limit of Multi-modal Emotion Recognition by Prompting LLMs with Receptive-Field-Aware Attention Weighting</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Understanding the emotions in a dialogue usually requires external knowledge to accurately understand the contents. As the LLMs become more and more powerful, we do not want to settle on the limited ability of the pre-trained language model. However, the LLMs either can only process text modality or are too expensive to process the multimedia information. We aim to utilize both the power of LLMs and the supplementary features from the multimedia modalities. In this paper, we present a framework, Lantern, that can improve the performance of a certain vanilla model by prompting large language models with receptive-field-aware attention weighting. This framework trained a multi-task vanilla model to produce probabilities of emotion classes and dimension scores. These predictions are fed into the LLMs as references to adjust the predicted probabilities of each emotion class with its external knowledge and contextual understanding. We slice the dialogue into different receptive fields, and each sample is included in exactly t receptive fields. Finally, the predictions of LLMs are merged with a receptive-field-aware attention-driven weighting module. In the experiments, vanilla models CORECT and SDT are deployed in Lantern with GPT-4 or Llama-3.1-405B. The experiments in IEMOCAP with 4-way and 6-way settings demonstrated that the Lantern can significantly improve the performance of current vanilla models by up to 1.23% and 1.80%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25264v2">GeoSQL-Eval: First Evaluation of LLMs on PostGIS-Based NL2GeoSQL Queries</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in natural language to SQL (NL2SQL) tasks within general databases. However, extending to GeoSQL introduces additional complexity from spatial data types, function invocation, and coordinate systems, which greatly increases generation and execution difficulty. Existing benchmarks mainly target general SQL, and a systematic evaluation framework for GeoSQL is still lacking. To fill this gap, we present GeoSQL-Eval, the first end-to-end automated evaluation framework for PostGIS query generation, together with GeoSQL-Bench, a benchmark for assessing LLM performance in NL2GeoSQL tasks. GeoSQL-Bench defines three task categories-conceptual understanding, syntax-level SQL generation, and schema retrieval-comprising 14,178 instances, 340 PostGIS functions, and 82 thematic databases. GeoSQL-Eval is grounded in Webb's Depth of Knowledge (DOK) model, covering four cognitive dimensions, five capability levels, and twenty task types to establish a comprehensive process from knowledge acquisition and syntax generation to semantic alignment, execution accuracy, and robustness. We evaluate 24 representative models across six categories and apply the entropy weight method with statistical analyses to uncover performance differences, common error patterns, and resource usage. Finally, we release a public GeoSQL-Eval leaderboard platform for continuous testing and global comparison. This work extends the NL2GeoSQL paradigm and provides a standardized, interpretable, and extensible framework for evaluating LLMs in spatial database contexts, offering valuable references for geospatial information science and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01995v1">LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Online social media platforms are central to everyday communication and information seeking. While these platforms serve positive purposes, they also provide fertile ground for the spread of hate speech, offensive language, and bullying content targeting individuals, organizations, and communities. Such content undermines safety, participation, and equity online. Reliable detection systems are therefore needed, especially for low-resource languages where moderation tools are limited. In Bangla, prior work has contributed resources and models, but most are single-task (e.g., binary hate/offense) with limited coverage of multi-facet signals (type, severity, target). We address these gaps by introducing the first multi-task Bangla hate-speech dataset, BanglaMultiHate, one of the largest manually annotated corpus to date. Building on this resource, we conduct a comprehensive, controlled comparison spanning classical baselines, monolingual pretrained models, and LLMs under zero-shot prompting and LoRA fine-tuning. Our experiments assess LLM adaptability in a low-resource setting and reveal a consistent trend: although LoRA-tuned LLMs are competitive with BanglaBERT, culturally and linguistically grounded pretraining remains critical for robust performance. Together, our dataset and findings establish a stronger benchmark for developing culturally aligned moderation tools in low-resource contexts. For reproducibility, we will release the dataset and all related scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11997v2">Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted by ACM MM 2025
    </div>
    <details class="paper-abstract">
      Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01929v1">Inverse Language Modeling towards Robust and Grounded LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The current landscape of defensive mechanisms for LLMs is fragmented and underdeveloped, unlike prior work on classifiers. To further promote adversarial robustness in LLMs, we propose Inverse Language Modeling (ILM), a unified framework that simultaneously 1) improves the robustness of LLMs to input perturbations, and, at the same time, 2) enables native grounding by inverting model outputs to identify potentially toxic or unsafe input triggers. ILM transforms LLMs from static generators into analyzable and robust systems, potentially helping RED teaming. ILM can lay the foundation for next-generation LLMs that are not only robust and grounded but also fundamentally more controllable and trustworthy. The code is publicly available at github.com/davegabe/pag-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22255v3">Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14245v2">Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Update with more experiments
    </div>
    <details class="paper-abstract">
      Recent advancements in long chain-of-thought (CoT) reasoning, particularly through the Group Relative Policy Optimization algorithm used by DeepSeek-R1, have led to significant interest in the potential of Reinforcement Learning with Verifiable Rewards (RLVR) for Large Language Models (LLMs). While RLVR promises to improve reasoning by allowing models to learn from free exploration, there remains debate over whether it truly enhances reasoning abilities or simply boosts sampling efficiency. This paper systematically investigates the impact of RLVR on LLM reasoning. We revisit Pass@K experiments and demonstrate that RLVR can extend the reasoning boundary for both mathematical and coding tasks. This is supported by our introduction of a novel evaluation metric, CoT-Pass@K, which captures reasoning success by accounting for both the final answer and intermediate reasoning steps. Furthermore, we present a theoretical framework explaining RLVR's incentive mechanism, demonstrating how it can encourage correct reasoning even when rewards are based solely on answer correctness. Our analysis of RLVR's training dynamics reveals that it incentivizes correct reasoning early in the process, with substantial improvements in reasoning quality confirmed through extensive evaluations. These findings provide strong evidence of RLVR's potential to enhance LLM reasoning, offering valuable insights into its mechanisms and performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01910v1">Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are widely adopted in Web-related applications, serving as a core technique for learning from graph-structured data, such as text-attributed graphs. Yet in real-world scenarios, such graphs exhibit deficiencies that substantially undermine GNN performance. While prior GNN-based augmentation studies have explored robustness against individual imperfections, a systematic understanding of how graph-native and Large Language Models (LLMs) enhanced methods behave under compound deficiencies is still missing. Specifically, there has been no comprehensive investigation comparing conventional approaches and recent LLM-on-graph frameworks, leaving their merits unclear. To fill this gap, we conduct the first empirical study that benchmarks these two lines of methods across diverse graph deficiencies, revealing overlooked vulnerabilities and challenging the assumption that LLM augmentation is consistently superior. Building on empirical findings, we propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement (RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is the first iterative paradigm that leverages Retrieval-Augmented Generation (RAG) to inject retrieval-grounded augmentations by supplying class-consistent, diverse augmentations and enforcing discriminative representations through iterative graph contrastive learning. It transforms LLM augmentation for graphs from static signal injection into dynamic refinement. Extensive experiments demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced baselines, achieving up to 82.43% average improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23281v2">MathArena: Evaluating LLMs on Uncontaminated Math Competitions</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On IMO 2025, top models achieve slightly less than 40%, demonstrating both notable progress and significant room for improvement. So far, we have evaluated over $50$ models across seven competitions, totaling $162$ problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01842v1">Pre-Hoc Predictions in AutoML: Leveraging LLMs to Enhance Model Selection and Benchmarking for Tabular datasets</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Oral Presentations ADAPT Annual Scientific Conference 2025
    </div>
    <details class="paper-abstract">
      The field of AutoML has made remarkable progress in post-hoc model selection, with libraries capable of automatically identifying the most performing models for a given dataset. Nevertheless, these methods often rely on exhaustive hyperparameter searches, where methods automatically train and test different types of models on the target dataset. Contrastingly, pre-hoc prediction emerges as a promising alternative, capable of bypassing exhaustive search through intelligent pre-selection of models. Despite its potential, pre-hoc prediction remains under-explored in the literature. This paper explores the intersection of AutoML and pre-hoc model selection by leveraging traditional models and Large Language Model (LLM) agents to reduce the search space of AutoML libraries. By relying on dataset descriptions and statistical information, we reduce the AutoML search space. Our methodology is applied to the AWS AutoGluon portfolio dataset, a state-of-the-art AutoML benchmark containing 175 tabular classification datasets available on OpenML. The proposed approach offers a shift in AutoML workflows, significantly reducing computational overhead, while still selecting the best model for the given dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01833v1">Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 19 pages and 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches, such as tree-based algorithms and reinforcement learning (RL), attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories. To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization PTA-GRPO, a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness. We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01831v1">Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 14 pages, 5 Tables, 9 Figures; Accepted to MathNLP 2025: The 3rd Workshop on Mathematical Natural Language Processing (co-located with EMNLP 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities but frequently fail on problems that deviate syntactically from their training distribution. We identify a systematic failure mode, syntactic blind spots, in which models misapply familiar reasoning strategies to problems that are semantically straightforward but phrased in unfamiliar ways. These errors are not due to gaps in mathematical competence, but rather reflect a brittle coupling between surface form and internal representation. To test this, we rephrase incorrectly answered questions using syntactic templates drawn from correct examples. These rephrasings, which preserve semantics while reducing structural complexity, often lead to correct answers. We quantify syntactic complexity using a metric based on Dependency Locality Theory (DLT), and show that higher DLT scores are associated with increased failure rates across multiple datasets. Our findings suggest that many reasoning errors stem from structural misalignment rather than conceptual difficulty, and that syntax-aware interventions can reveal and mitigate these inductive failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01801v1">Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has enabled the generation of highly persuasive spam reviews that closely mimic human writing. These reviews pose significant challenges for existing detection systems and threaten the credibility of online platforms. In this work, we first create three realistic LLM-generated spam review datasets using three distinct LLMs, each guided by product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm the high persuasion and deceptive potential of these reviews. To address this threat, we propose FraudSquad, a hybrid detection model that integrates text embeddings from a pre-trained language model with a gated graph transformer for spam node classification. FraudSquad captures both semantic and behavioral signals without relying on manual feature engineering or massive training resources. Experiments show that FraudSquad outperforms state-of-the-art baselines by up to 44.22% in precision and 43.01% in recall on three LLM-generated datasets, while also achieving promising results on two human-written spam datasets. Furthermore, FraudSquad maintains a modest model size and requires minimal labeled training data, making it a practical solution for real-world applications. Our contributions include new synthetic datasets, a practical detection framework, and empirical evidence highlighting the urgency of adapting spam detection to the LLM era. Our code and datasets are available at: https://anonymous.4open.science/r/FraudSquad-5389/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01782v1">Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability. However, existing metrics fail to faithfully measure this ability. On the one hand, simple refusal-based metrics are biased by refusal rates and yield inconsistent scores when models exhibit different refusal tendencies. On the other hand, existing calibration metrics are proxy-based, capturing the performance of auxiliary calibration processes rather than the model's actual refusal behavior. In this work, we propose the Refusal Index (RI), a principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. To make RI practically measurable, we design a lightweight two-pass evaluation method that efficiently estimates RI from observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's intrinsic knowledge-aware refusal capability in factual tasks. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile. This finding highlights the need to complement traditional accuracy metrics with the Refusal Index for comprehensive factuality evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00761v2">Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often fragile: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the optimizer, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the 'grade' of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a hybrid optimizer that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14459v2">Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted by ICDM 2025
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) aim to provide personalized recommendations by capturing user preferences through interactive dialogues. Explainability in CRSs is crucial as it enables users to understand the reasoning behind recommendations, increasing system transparency and trustworthiness. However, current CRSs often leverage knowledge graphs (KGs) or language models to extract and represent user preferences as latent vectors, which limits their explainability. Large language models (LLMs) offer powerful reasoning capabilities that can bridge this gap by generating human-understandable preference summaries. However, effectively reasoning over user preferences in CRSs remains challenging as LLMs pre-trained on large-scale corpora may not be well-suited for analyzing user preferences. While KGs provide rich domain knowledge, integrating them with LLMs encounters a significant modality gap between structured KG information and unstructured conversations. In this paper, we propose COMPASS, a plug-and-play framework that synergizes LLMs and KGs to reason over user preferences, enhancing the performance and explainability of existing CRSs. COMPASS employs a two-stage training approach: first, it bridges the gap between the structured KG and natural language through novel graph entity captioning pre-training. Next, COMPASS optimizes user preference reasoning via knowledge-aware instruction fine-tuning, where the LLM learns to reason and summarize user preferences from dialogue histories and KG-augmented context. This enables COMPASS to perform knowledge-aware reasoning and generate interpretable user preferences that can seamlessly integrate with existing CRS models for improving recommendation performance and explainability. Our experiments on benchmark datasets demonstrate the effectiveness of COMPASS in improving various CRS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v1">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24858v2">MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      A critical component in the trustworthiness of LLMs is reliable uncertainty communication, yet LLMs often use assertive language when conveying false claims, leading to over-reliance and eroded trust. We present the first systematic study of $\textit{faithful confidence calibration}$ of LLMs, benchmarking models' ability to use linguistic expressions of uncertainty that $\textit{faithfully reflect}$ their intrinsic uncertainty, across a comprehensive array of models, datasets, and prompting strategies. Our results demonstrate that LLMs largely fail at this task, and that existing interventions are insufficient: standard prompt approaches provide only marginal gains, and existing, factuality-based calibration techniques can even harm faithful calibration. To address this critical gap, we introduce MetaFaith, a novel prompt-based calibration approach inspired by human metacognition. We show that MetaFaith robustly improves faithful calibration across diverse models and task domains, enabling up to 61% improvement in faithfulness and achieving an 83% win rate over original generations as judged by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01688v1">Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have brought significant improvements to various service domains, including chatbots and medical pre-consultation applications. In the healthcare domain, the most common approach for adapting LLMs to multi-turn dialogue generation is Supervised Fine-Tuning (SFT). However, datasets for SFT in tasks like medical pre-consultation typically exhibit a skewed turn-count distribution. Training on such data induces a novel failure mechanism we term **Format Inertia**, where models tend to generate repetitive, format-correct, but diagnostically uninformative questions in long medical dialogues. To mitigate this observed failure mechanism, we adopt a simple, data-centric method that rebalances the turn-count distribution of the training dataset. Experimental results show that our approach substantially alleviates Format Inertia in medical pre-consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12885v2">VAR-MATH: Probing True Mathematical Reasoning in LLMS via Symbolic Multi-Instance Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have led to substantial improvements in the mathematical reasoning abilities of LLMs, as measured by standard benchmarks. Yet these gains often persist even when models are trained with flawed signals, such as random or inverted rewards. This raises a fundamental question: do such improvements reflect genuine reasoning, or are they merely artifacts of overfitting to benchmark-specific patterns? To answer this question, we adopt an evaluation-centric perspective and highlight two critical shortcomings in existing protocols. First, benchmark contamination arises because test problems are publicly available, thereby increasing the risk of data leakage. Second, evaluation fragility results from reliance on single-instance assessments, which are sensitive to stochastic outputs and fail to capture reasoning consistency. These limitations suggest the need for a new evaluation paradigm that can probe reasoning ability beyond memorization and one-off success. As response, we propose VAR-MATH, a symbolic evaluation framework that converts fixed numerical problems into parameterized templates and requires models to solve multiple instantiations of each. This design enforces consistency across structurally equivalent variants, mitigates contamination, and enhances robustness through bootstrapped metrics. We apply VAR-MATH to transform three popular benchmarks, AMC23, AIME24, and AIME25, into their symbolic counterparts, VAR-AMC23, VAR-AIME24, and VAR-AIME25. Experimental results show substantial performance drops for RL-trained models on these variabilized benchmarks, especially for smaller models, with average declines of 47.9\% on AMC23, 58.8\% on AIME24, and 72.9\% on AIME25. These findings indicate that some existing RL methods rely on superficial heuristics and fail to generalize beyond specific numerical forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01664v1">GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 7 Pages, 2 figures
    </div>
    <details class="paper-abstract">
      This study demonstrates that GuruAgents, prompt-guided AI agents, can systematically operationalize the strategies of legendary investment gurus. We develop five distinct GuruAgents, each designed to emulate an iconic investor, by encoding their distinct philosophies into LLM prompts that integrate financial tools and a deterministic reasoning pipeline. In a backtest on NASDAQ-100 constituents from Q4 2023 to Q2 2025, the GuruAgents exhibit unique behaviors driven by their prompted personas. The Buffett GuruAgent achieves the highest performance, delivering a 42.2\% CAGR that significantly outperforms benchmarks, while other agents show varied results. These findings confirm that prompt engineering can successfully translate the qualitative philosophies of investment gurus into reproducible, quantitative strategies, highlighting a novel direction for automated systematic investing. The source code and data are available at https://github.com/yejining99/GuruAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01656v1">Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01652v1">Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Autoregressive Large Language Models (LLMs) demonstrate exceptional performance in language understanding and generation. However, their application in text embedding tasks has been relatively slow, along with the analysis of their semantic representation in probing tasks, due to the constraints of the unidirectional attention mechanism. This paper aims to explore whether such constraints can be overcome by enabling bidirectional attention in LLMs. We tested different variants of the Llama architecture through additional training steps, progressively enabling bidirectional attention and unsupervised/supervised contrastive learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26306v3">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 The code is available at https://github.com/linhh29/Interactive-Learning-for-LLM-Reasoning
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01650v1">The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Neural network pruning is a promising technique to mitigate the excessive computational and memory requirements of large language models (LLMs). Despite its promise, however, progress in this area has diminished, as conventional methods are seemingly unable to surpass moderate sparsity levels (50-60%) without severely degrading model accuracy. This work breaks through the current impasse, presenting a principled and effective method called $\texttt{Elsa}$, which achieves extreme sparsity levels of up to 90% while retaining high model fidelity. This is done by identifying several limitations in current practice, all of which can be traced back to their reliance on a surrogate objective formulation. $\texttt{Elsa}$ tackles this issue directly and effectively via standard and well-established constrained optimization techniques based on ADMM. Our extensive experiments across a wide range of models and scales show that $\texttt{Elsa}$ achieves substantial improvements over existing methods; e.g., it achieves 7.8$\times$ less perplexity than the best existing method on LLaMA-2-7B at 90% sparsity. Furthermore, we present $\texttt{Elsa}_{\text{-L}}$, a quantized variant that scales to extremely large models (27B), and establish its theoretical convergence guarantees. These results highlight meaningful progress in advancing the frontier of LLM sparsity, while promising that significant opportunities for further advancement may remain in directions that have so far attracted limited exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01644v1">NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BERT</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01639v1">Understanding the Geospatial Reasoning Capabilities of LLMs: A Trajectory Recovery Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      We explore the geospatial reasoning capabilities of Large Language Models (LLMs), specifically, whether LLMs can read road network maps and perform navigation. We frame trajectory recovery as a proxy task, which requires models to reconstruct masked GPS traces, and introduce GLOBALTRACE, a dataset with over 4,000 real-world trajectories across diverse regions and transportation modes. Using road network as context, our prompting framework enables LLMs to generate valid paths without accessing any external navigation tools. Experiments show that LLMs outperform off-the-shelf baselines and specialized trajectory recovery models, with strong zero-shot generalization. Fine-grained analysis shows that LLMs have strong comprehension of the road network and coordinate systems, but also pose systematic biases with respect to regions and transportation modes. Finally, we demonstrate how LLMs can enhance navigation experiences by reasoning over maps in flexible ways to incorporate user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01637v1">Detecting Post-generation Edits to Watermarked LLM Outputs via Combinatorial Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Watermarking has become a key technique for proprietary language models, enabling the distinction between AI-generated and human-written text. However, in many real-world scenarios, LLM-generated content may undergo post-generation edits, such as human revisions or even spoofing attacks, making it critical to detect and localize such modifications. In this work, we introduce a new task: detecting post-generation edits locally made to watermarked LLM outputs. To this end, we propose a combinatorial pattern-based watermarking framework, which partitions the vocabulary into disjoint subsets and embeds the watermark by enforcing a deterministic combinatorial pattern over these subsets during generation. We accompany the combinatorial watermark with a global statistic that can be used to detect the watermark. Furthermore, we design lightweight local statistics to flag and localize potential edits. We introduce two task-specific evaluation metrics, Type-I error rate and detection accuracy, and evaluate our method on open-source LLMs across a variety of editing scenarios, demonstrating strong empirical performance in edit localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01631v1">Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Published as a Main Conference paper at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Training data plays a crucial role in Large Language Models (LLM) scaling, yet high quality data is of limited supply. Synthetic data techniques offer a potential path toward sidestepping these limitations. We conduct a large-scale empirical investigation (>1000 LLMs with >100k GPU hours) using a unified protocol and scaling laws, comparing natural web data, diverse synthetic types (rephrased text, generated textbooks), and mixtures of natural and synthetic data. Specifically, we found pre-training on rephrased synthetic data \textit{alone} is not faster than pre-training on natural web texts; while pre-training on 1/3 rephrased synthetic data mixed with 2/3 natural web texts can speed up 5-10x (to reach the same validation loss) at larger data budgets. Pre-training on textbook-style synthetic data \textit{alone} results in notably higher loss on many downstream domains especially at small data budgets. "Good" ratios of synthetic data in training data mixtures depend on the model size and data budget, empirically converging to ~30% for rephrased synthetic data. Larger generator models do not necessarily yield better pre-training data than ~8B-param models. These results contribute mixed evidence on "model collapse" during large-scale single-round (n=1) model training on synthetic data--training on rephrased synthetic data shows no degradation in performance in foreseeable scales whereas training on mixtures of textbook-style pure-generated synthetic data shows patterns predicted by "model collapse". Our work demystifies synthetic data in pre-training, validates its conditional benefits, and offers practical guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01617v1">AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01609v1">AgentRec: Next-Generation LLM-Powered Multi-Agent Collaborative Recommendation with Adaptive Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Interactive conversational recommender systems have gained significant attention for their ability to capture user preferences through natural language interactions. However, existing approaches face substantial challenges in handling dynamic user preferences, maintaining conversation coherence, and balancing multiple ranking objectives simultaneously. This paper introduces AgentRec, a next-generation LLM-powered multi-agent collaborative recommendation framework that addresses these limitations through hierarchical agent networks with adaptive intelligence. Our approach employs specialized LLM-powered agents for conversation understanding, preference modeling, context awareness, and dynamic ranking, coordinated through an adaptive weighting mechanism that learns from interaction patterns. We propose a three-tier learning strategy combining rapid response for simple queries, intelligent reasoning for complex preferences, and deep collaboration for challenging scenarios. Extensive experiments on three real-world datasets demonstrate that AgentRec achieves consistent improvements over state-of-the-art baselines, with 2.8\% enhancement in conversation success rate, 1.9\% improvement in recommendation accuracy (NDCG@10), and 3.2\% better conversation efficiency while maintaining comparable computational costs through intelligent agent coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03815v2">Synergizing LLMs and Knowledge Graphs: A Novel Approach to Software Repository-Related Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Submitted to ACM Transactions on Software Engineering and Methodology for review
    </div>
    <details class="paper-abstract">
      Software repositories contain valuable information for understanding the development process. However, extracting insights from repository data is time-consuming and requires technical expertise. While software engineering chatbots support natural language interactions with repositories, chatbots struggle to understand questions beyond their trained intents and to accurately retrieve the relevant data. This study aims to improve the accuracy of LLM-based chatbots in answering repository-related questions by augmenting them with knowledge graphs. We use a two-step approach: constructing a knowledge graph from repository data, and synergizing the knowledge graph with an LLM to handle natural language questions and answers. We curated 150 questions of varying complexity and evaluated the approach on five popular open-source projects. Our initial results revealed the limitations of the approach, with most errors due to the reasoning ability of the LLM. We therefore applied few-shot chain-of-thought prompting, which improved accuracy to 84%. We also compared against baselines (MSRBot and GPT-4o-search-preview), and our approach performed significantly better. In a task-based user study with 20 participants, users completed more tasks correctly and in less time with our approach, and they reported that it was useful. Our findings demonstrate that LLMs and knowledge graphs are a viable solution for making repository data accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20177v5">AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Domain reweighting is an emerging research area aimed at adjusting the relative weights of different data sources to improve the effectiveness and efficiency of LLM pre-training. We show that data mixtures that perform well at smaller scales may not retain their advantage at larger scales, challenging the existing practice of determining competitive mixtures in small-scale experiments and directly applying them at much larger scales. To address this, we propose AutoScale, a two-stage, scale-aware data composition framework. First, AutoScale fits a parametric model that predicts the model's loss under different data compositions, then uses it to find an approximate best allocation at smaller, more manageable budgets. Next, leveraging a novel theoretical analysis of how optimal compositions evolve with scale, AutoScale extrapolates that composition to larger budgets without further retraining. Empirically, AutoScale accelerates convergence and improves downstream performance. For instance, when pre-training GPT-2 Large, it achieves a 28% faster perplexity reduction than baselines and up to a 38% speed-up over unweighted training, while yielding best-average results on various downstream tasks. Overall, our findings illustrate how domain importance shifts with training scale, underscoring the need for scale-dependent data curation in LLM training. Our code is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21285v3">Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. Our codes and data are available at https://github.com/XinXU-USTC/DoubleChecker
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01552v1">POLAR: Automating Cyber Threat Prioritization through LLM-Powered Assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01539v1">Executable Counterfactuals: Improving LLMs' Causal Reasoning Through Code</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Counterfactual reasoning, a hallmark of intelligence, consists of three steps: inferring latent variables from observations (abduction), constructing alternatives (interventions), and predicting their outcomes (prediction). This skill is essential for advancing LLMs' causal understanding and expanding their applications in high-stakes domains such as scientific research. However, existing efforts in assessing LLM's counterfactual reasoning capabilities tend to skip the abduction step, effectively reducing to interventional reasoning and leading to overestimation of LLM performance. To address this, we introduce executable counterfactuals, a novel framework that operationalizes causal reasoning through code and math problems. Our framework explicitly requires all three steps of counterfactual reasoning and enables scalable synthetic data creation with varying difficulty, creating a frontier for evaluating and improving LLM's reasoning. Our results reveal substantial drop in accuracy (25-40%) from interventional to counterfactual reasoning for SOTA models like o4-mini and Claude-4-Sonnet. To address this gap, we construct a training set comprising counterfactual code problems having if-else condition and test on out-of-domain code structures (e.g. having while-loop); we also test whether a model trained on code would generalize to counterfactual math word problems. While supervised finetuning on stronger models' reasoning traces improves in-domain performance of Qwen models, it leads to a decrease in accuracy on OOD tasks such as counterfactual math problems. In contrast, reinforcement learning induces the core cognitive behaviors and generalizes to new domains, yielding gains over the base model on both code (improvement of 1.5x-2x) and math problems. Analysis of the reasoning traces reinforces these findings and highlights the promise of RL for improving LLMs' counterfactual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01530v1">LOGicalThought: Logic-Based Ontological Grounding of LLMs for High-Assurance Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      High-assurance reasoning, particularly in critical domains such as law and medicine, requires conclusions that are accurate, verifiable, and explicitly grounded in evidence. This reasoning relies on premises codified from rules, statutes, and contracts, inherently involving defeasible or non-monotonic logic due to numerous exceptions, where the introduction of a single fact can invalidate general rules, posing significant challenges. While large language models (LLMs) excel at processing natural language, their capabilities in standard inference tasks do not translate to the rigorous reasoning required over high-assurance text guidelines. Core reasoning challenges within such texts often manifest specific logical structures involving negation, implication, and, most critically, defeasible rules and exceptions. In this paper, we propose a novel neurosymbolically-grounded architecture called LOGicalThought (LogT) that uses an advanced logical language and reasoner in conjunction with an LLM to construct a dual symbolic graph context and logic-based context. These two context representations transform the problem from inference over long-form guidelines into a compact grounded evaluation. Evaluated on four multi-domain benchmarks against four baselines, LogT improves overall performance by 11.84% across all LLMs. Performance improves significantly across all three modes of reasoning: by up to +10.2% on negation, +13.2% on implication, and +5.5% on defeasible reasoning compared to the strongest baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11027v3">On the Effect of Sampling Diversity in Scaling LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) scaling inference is key to unlocking greater performance, and leveraging diversity has proven an effective way to enhance it. Motivated by the observed relationship between solution accuracy and meaningful response diversity, we systematically study the effect of prompt diversity in scaling inference. We theoretically explain why diversified sampling improves Best-of-$N$ scaling, showing that responses generated from meaningful diverse prompts after Best-of-$N$ selection exhibit significantly lower error rates than those produced from stationary prompts. To promote solution diversity, we analyze perturbation fidelity and show that moderately relevant perturbations improve performance, providing guidance for effective perturbation design. Further, we present a set of effective perturbations, including task-level and query-level ones, and analyze the conditions under which they succeed. We systematically evaluate diversified sampling across tasks, finding relative gains of 10.8% in EM@100 for reasoning, 9.6% for mathematics, and 9.5% in Pass@100 for code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01037v1">CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 25 pages, 10 Figures
    </div>
    <details class="paper-abstract">
      Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by \textbf{+3.30} points and \textbf{+4.82} points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01032v1">Meaningless Tokens, Meaningful Gains: How Activation Shifts Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Motivated by the puzzling observation that inserting long sequences of meaningless tokens before the query prompt can consistently enhance LLM reasoning performance, this work analyzes the underlying mechanism driving this phenomenon and based on these insights proposes a more principled method that allows for similar performance gains. First, we find that the improvements arise from a redistribution of activations in the LLM's MLP layers, where near zero activations become less frequent while large magnitude activations increase. This redistribution enhances the model's representational capacity by suppressing weak signals and promoting stronger, more informative ones. Building on this insight, we propose the Activation Redistribution Module (ARM), a lightweight inference-time technique that modifies activations directly without altering the input sequence. ARM adaptively identifies near-zero activations after the non-linear function and shifts them outward, implicitly reproducing the beneficial effects of meaningless tokens in a controlled manner. Extensive experiments across diverse benchmarks and model architectures clearly show that ARM consistently improves LLM performance on reasoning tasks while requiring only a few lines of simple code to implement. Our findings deliver both a clear mechanistic explanation for the unexpected benefits of meaningless tokens and a simple yet effective technique that harnesses activation redistribution to further improve LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01030v1">Uncovering the Computational Ingredients of Human-Like Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The ability to translate diverse patterns of inputs into structured patterns of behavior has been thought to rest on both humans' and machines' ability to learn robust representations of relevant concepts. The rapid advancement of transformer-based large language models (LLMs) has led to a diversity of computational ingredients -- architectures, fine tuning methods, and training datasets among others -- but it remains unclear which of these ingredients are most crucial for building models that develop human-like representations. Further, most current LLM benchmarks are not suited to measuring representational alignment between humans and models, making benchmark scores unreliable for assessing if current LLMs are making progress towards becoming useful cognitive models. We address these limitations by first evaluating a set of over 70 models that widely vary in their computational ingredients on a triplet similarity task, a method well established in the cognitive sciences for measuring human conceptual representations, using concepts from the THINGS database. Comparing human and model representations, we find that models that undergo instruction-finetuning and which have larger dimensionality of attention heads are among the most human aligned, while multimodal pretraining and parameter size have limited bearing on alignment. Correlations between alignment scores and scores on existing benchmarks reveal that while some benchmarks (e.g., MMLU) are better suited than others (e.g., MUSR) for capturing representational alignment, no existing benchmark is capable of fully accounting for the variance of alignment scores, demonstrating their insufficiency in capturing human-AI alignment. Taken together, our findings help highlight the computational ingredients most essential for advancing LLMs towards models of human conceptual representation and address a key benchmarking gap in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00967v1">QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00962v1">Analyzing Dialectical Biases in LLMs for Knowledge and Reasoning Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP Findings 2025, 12 pages, 11 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are ubiquitous in modern day natural language processing. However, previous work has shown degraded LLM performance for under-represented English dialects. We analyze the effects of typifying "standard" American English language questions as non-"standard" dialectal variants on multiple choice question answering tasks and find up to a 20% reduction in accuracy. Additionally, we investigate the grammatical basis of under-performance in non-"standard" English questions. We find that individual grammatical rules have varied effects on performance, but some are more consequential than others: three specific grammar rules (existential "it", zero copula, and y'all) can explain the majority of performance degradation observed in multiple dialects. We call for future work to investigate bias mitigation methods focused on individual, high-impact grammatical structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00932v1">Opal: A Modular Framework for Optimizing Performance using Analytics and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 12 pages and 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise for automated code optimization but struggle without performance context. This work introduces Opal, a modular framework that connects performance analytics insights with the vast body of published by guiding LLMs to generate informed, trustworthy optimizations. Unlike traditional performance tools that identify bottlenecks but stop short of actionable suggestions, Opal bridges this long-standing gap by linking dynamic insights from hardware counters and Roofline analysis to stall events to optimization decisions. We evaluate Opal across 1640 experiments on real-world GPU kernels and find that in over 98.5% of cases, even a single insight source yields speedups, ranging on average from 19.34% to 52.3%. Our prompt template produced correct code in all but one case, where a vague diagnostic caused an unsafe suggestion. By automatically optimizing GPU kernels using performance analytics and LLMs, Opal marks a leap toward democratizing expert-level performance engineering for all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00911v1">RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00908v1">Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.11812v1">On the Performance of LLMs for Real Estate Appraisal</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted at ECML-PKDD 2025
    </div>
    <details class="paper-abstract">
      The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00881v1">Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted at ASE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software engineering (SE) tools for tasks that extend beyond code synthesis, including judgment under uncertainty and reasoning in ethically significant contexts. We present a fully automated framework for assessing ethical reasoning capabilities across 16 LLMs in a zero-shot setting, using 30 real-world ethically charged scenarios. Each model is prompted to identify the most applicable ethical theory to an action, assess its moral acceptability, and explain the reasoning behind their choice. Responses are compared against expert ethicists' choices using inter-model agreement metrics. Our results show that LLMs achieve an average Theory Consistency Rate (TCR) of 73.3% and Binary Agreement Rate (BAR) on moral acceptability of 86.7%, with interpretable divergences concentrated in ethically ambiguous cases. A qualitative analysis of free-text explanations reveals strong conceptual convergence across models despite surface-level lexical diversity. These findings support the potential viability of LLMs as ethical inference engines within SE pipelines, enabling scalable, auditable, and adaptive integration of user-aligned ethical reasoning. Our focus is the Ethical Interpreter component of a broader profiling pipeline: we evaluate whether current LLMs exhibit sufficient interpretive stability and theory-consistent reasoning to support automated profiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00861v1">Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00857v1">ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at https://github.com/technion-cs-nlp/ManagerBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00844v1">Learning Compact Representations of LLM Abilities via Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors: (i) the model's multi-skill ability vector, (2) the query's discrimination vector that separates models of differing skills, and (3) the query's difficulty scalar. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00841v1">LLM Routing with Dueling Feedback</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We study LLM routing, the problem of selecting the best model for each query while balancing user satisfaction, model expertise, and inference cost. We formulate routing as contextual dueling bandits, learning from pairwise preference feedback rather than absolute scores, thereby yielding label-efficient and dynamic adaptation. Building on this formulation, we introduce Category-Calibrated Fine-Tuning (CCFT), a representation-learning method that derives model embeddings from offline data using contrastive fine-tuning with categorical weighting. These embeddings enable the practical instantiation of Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB), a theoretically grounded posterior-sampling algorithm. We propose four variants of the categorical weighting that explicitly integrate model quality and cost, and we empirically evaluate the proposed methods on the RouterBench and MixInstruct datasets. Across both benchmarks, our methods achieve lower cumulative regret and faster convergence, with better robustness and performance-cost balance than strong baselines built with a general-purpose OpenAI embedding model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00829v1">Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00819v1">Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Reinforcement Learning, particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models. However, the optimization stability of policy gradients in this setting remains understudied. As a result, existing implementations often resort to conservative hyperparameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. We address this gap by formalizing the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. We propose a tractable computational framework that tracks and leverages curvature information during policy updates. We further employ this framework to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out. Theoretically, we establish monotonic improvement guarantees under realistic assumptions. On standard math reasoning benchmarks, we empirically show that CAPO ensures stable updates under aggressive learning regimes where baselines catastrophically fail. With minimal intervention (rejecting fewer than 8% of tokens), CAPO achieves up to 30x improvement in sample efficiency over standard GRPO for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00810v1">Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We investigate how to adapt small, efficient LLMs to Faroese, a low-resource North Germanic language. Starting from English models, we continue pre-training on related Scandinavian languages, either individually or combined via merging, before fine-tuning on Faroese. We compare full fine-tuning with parameter-efficient tuning using LoRA, evaluating their impact on both linguistic accuracy and text comprehension. Due to the lack of existing Faroese evaluation data, we construct two new minimal-pair benchmarks from adapted and newly collected datasets and complement them with human evaluations by Faroese linguists. Our results demonstrate that transfer from related languages is crucial, though the optimal source language depends on the task: Icelandic enhances linguistic accuracy, whereas Danish boosts comprehension. Similarly, the choice between full fine-tuning and LoRA is task-dependent: LoRA improves linguistic acceptability and slightly increases human evaluation scores on the base model, while full fine-tuning yields stronger comprehension performance and better preserves model capabilities during downstream fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00777v1">In-Place Feedback: A New Paradigm for Guiding LLMs in Multi-Turn Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 28 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly studied in the context of multi-turn reasoning, where models iteratively refine their outputs based on user-provided feedback. Such settings are crucial for tasks that require complex reasoning, yet existing feedback paradigms often rely on issuing new messages. LLMs struggle to integrate these reliably, leading to inconsistent improvements. In this work, we introduce in-place feedback, a novel interaction paradigm in which users directly edit an LLM's previous response, and the model conditions on this modified response to generate its revision. Empirical evaluations on diverse reasoning-intensive benchmarks reveal that in-place feedback achieves better performance than conventional multi-turn feedback while using $79.1\%$ fewer tokens. Complementary analyses on controlled environments further demonstrate that in-place feedback resolves a core limitation of multi-turn feedback: models often fail to apply feedback precisely to erroneous parts of the response, leaving errors uncorrected and sometimes introducing new mistakes into previously correct content. These findings suggest that in-place feedback offers a more natural and effective mechanism for guiding LLMs in reasoning-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00662v1">Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00615v1">ACON: Optimizing Context Compression for Long-horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01295v1">The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) transition from static tools to autonomous agents, traditional evaluation benchmarks that measure performance on downstream tasks are becoming insufficient. These methods fail to capture the emergent social and cognitive dynamics that arise when agents communicate, persuade, and collaborate in interactive environments. To address this gap, we introduce a novel evaluation framework that uses multi-agent debate as a controlled "social laboratory" to discover and quantify these behaviors. In our framework, LLM-based agents, instantiated with distinct personas and incentives, deliberate on a wide range of challenging topics under the supervision of an LLM moderator. Our analysis, enabled by a new suite of psychometric and semantic metrics, reveals several key findings. Across hundreds of debates, we uncover a powerful and robust emergent tendency for agents to seek consensus, consistently reaching high semantic agreement ({\mu} > 0.88) even without explicit instruction and across sensitive topics. We show that assigned personas induce stable, measurable psychometric profiles, particularly in cognitive effort, and that the moderators persona can significantly alter debate outcomes by structuring the environment, a key finding for external AI alignment. This work provides a blueprint for a new class of dynamic, psychometrically grounded evaluation protocols designed for the agentic setting, offering a crucial methodology for understanding and shaping the social behaviors of the next generation of AI agents. We have released the code and results at https://github.com/znreza/multi-agent-LLM-eval-for-debate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00579v1">CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing implementations, such as in-context learning and fine-tuning, remain costly and inefficient. To improve CoT reasoning at a lower cost, and inspired by the task vector paradigm, we introduce CoT Vectors, compact representations that encode task-general, multi-step reasoning knowledge. Through experiments with Extracted CoT Vectors, we observe pronounced layer-wise instability, manifesting as a U-shaped performance curve that reflects a systematic three-stage reasoning process in LLMs. To address this limitation, we propose Learnable CoT Vectors, optimized under a teacher-student framework to provide more stable and robust guidance. Extensive evaluations across diverse benchmarks and models demonstrate that CoT Vectors not only outperform existing baselines but also achieve performance comparable to parameter-efficient fine-tuning methods, while requiring fewer trainable parameters. Moreover, by treating CoT Vectors as a probe, we uncover how their effectiveness varies due to latent space structure, information density, acquisition mechanisms, and pre-training differences, offering new insights into the functional organization of multi-step reasoning in LLMs. The source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00555v1">PromptPilot: Improving Human-AI Collaboration Through LLM-Enhanced Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint version. Accepted for presentation at the International Conference on Information Systems (ICIS 2025). Please cite the published version when available
    </div>
    <details class="paper-abstract">
      Effective prompt engineering is critical to realizing the promised productivity gains of large language models (LLMs) in knowledge-intensive tasks. Yet, many users struggle to craft prompts that yield high-quality outputs, limiting the practical benefits of LLMs. Existing approaches, such as prompt handbooks or automated optimization pipelines, either require substantial effort, expert knowledge, or lack interactive guidance. To address this gap, we design and evaluate PromptPilot, an interactive prompting assistant grounded in four empirically derived design objectives for LLM-enhanced prompt engineering. We conducted a randomized controlled experiment with 80 participants completing three realistic, work-related writing tasks. Participants supported by PromptPilot achieved significantly higher performance (median: 78.3 vs. 61.7; p = .045, d = 0.56), and reported enhanced efficiency, ease-of-use, and autonomy during interaction. These findings empirically validate the effectiveness of our proposed design objectives, establishing LLM-enhanced prompt engineering as a viable technique for improving human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00543v1">Flow of Knowledge: Federated Fine-Tuning of LLMs in Healthcare under Non-IID Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great promise in healthcare, but their applications are hindered by data privacy restrictions and the challenges of cross-institution collaboration. Sensitive medical data cannot be centralized, while non-independent and identically distributed (non-IID) characteristics across institutions further complicate convergence and fairness. To address these issues, we present a federated fine-tuning approach based on Low-Rank Adaptation (LoRA), enabling privacy-preserving knowledge flow across institutions. The method iteratively combines local LoRA adaptation with global parameter aggregation, allowing efficient knowledge sharing without exposing raw data. A blockchain identity scheme is used for identifying individual LLM in such a distributed network. We evaluate this approach on heterogeneous and highly non-IID medical text datasets, where experiments demonstrate that federated LoRA not only enhances cross-client generalization but also improves the performance of the weakest client, achieving stable convergence and fairer outcomes. These findings highlight federated LoRA fine-tuning as a practical and effective paradigm for adapting LLMs in healthcare, offering a new path for multi-center medical AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00494v1">Exploring System 1 and 2 communication for latent reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Should LLM reasoning live in a separate module, or within a single model's forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline, a single model with the same forward pass and shared representations, using the same latent-token budget, nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives and communication mechanisms that explicitly shape latent spaces for algorithmic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00482v1">Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted by AIxB 2025
    </div>
    <details class="paper-abstract">
      Agentic large language models (LLMs) have become prominent for autonomously interacting with external environments and performing multi-step reasoning tasks. Most approaches leverage these capabilities via in-context learning with few-shot prompts, but this often results in lengthy inputs and higher computational costs. Agent fine-tuning offers an alternative by enabling LLMs to internalize procedural reasoning and domain-specific knowledge through training on relevant data and demonstration trajectories. While prior studies have focused on general domains, their effectiveness in specialized technical microdomains remains unclear. This paper explores agent fine-tuning for domain adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT operations. We fine-tuned LLMs using JP1-specific datasets derived from domain manuals and distilled reasoning trajectories generated by LLMs themselves, enhancing decision making accuracy and search efficiency. During inference, we used an agentic prompt with retrieval-augmented generation and introduced a context-answer extractor to improve information relevance. On JP1 certification exam questions, our method achieved a 14% performance improvement over the base model, demonstrating the potential of agent fine-tuning for domain-specific reasoning in complex microdomains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00481v1">Make a Video Call with LLM: A Measurement Campaign over Five Mainstream Apps</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      In 2025, Large Language Model (LLM) services have launched a new feature -- AI video chat -- allowing users to interact with AI agents via real-time video communication (RTC), just like chatting with real people. Despite its significance, no systematic study has characterized the performance of existing AI video chat systems. To address this gap, this paper proposes a comprehensive benchmark with carefully designed metrics across four dimensions: quality, latency, internal mechanisms, and system overhead. Using custom testbeds, we further evaluate five mainstream AI video chatbots with this benchmark. This work provides the research community a baseline of real-world performance and identifies unique system bottlenecks. In the meantime, our benchmarking results also open up several research questions for future optimizations of AI video chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00449v1">Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted to EMNLP 2025 PALS Workshop
    </div>
    <details class="paper-abstract">
      Personalizing the outputs of large language models (LLMs) to align with individual user preferences is an active research area. However, previous studies have mainly focused on classification or ranking tasks and have not considered Likert-scale rating prediction, a regression task that requires both language and mathematical reasoning to be solved effectively. This task has significant industrial applications, but the utilization of LLMs remains underexplored, particularly regarding the capabilities of off-the-shelf LLMs. This study investigates the performance of off-the-shelf LLMs on rating prediction, providing different in-context information. Through comprehensive experiments with eight models across three datasets, we demonstrate that user-written reviews significantly improve the rating prediction performance of LLMs. This result is comparable to traditional methods like matrix factorization, highlighting the potential of LLMs as a promising solution for the cold-start problem. We also find that the reviews for concrete items are more effective than general preference descriptions that are not based on any specific item. Furthermore, we discover that prompting LLMs to first generate a hypothetical review enhances the rating prediction performance. Our code is available at https://github.com/ynklab/rating-prediction-with-reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00419v1">Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Zeroth-order optimizers have recently emerged as a practical approach for fine-tuning large language models (LLMs), significantly reducing GPU memory consumption compared to traditional first-order methods. Yet, existing zeroth-order methods rely on hand-crafted, static sampling strategies that are not adaptable to model-specific structures. To address this, we propose ZO Fine-tuner, a learning-based zeroth-order optimizer for LLMs that automatically learns efficient perturbation strategies through a compact and memory-efficient design. Crucially, our approach is motivated by the observation that only a small number of foundation models and their derivatives are widely adopted in practice. Therefore, learning the optimizer once for a given LLM and reusing it across diverse downstream tasks is both feasible and highly desirable. Accordingly, ZO Fine-tuner is designed to scale learning to learn (L2L) to the foundation-model era by supporting one-time training per LLM with minimal overhead. Experiments on 4 LLMs and 7 datasets show that ZO Fine-tuner outperforms prior zeroth-order baselines in 82.1\% of task-model combinations, thereby demonstrating strong performance and scalability for efficient LLM fine-tuning. Our code is available at https://github.com/ASTRAL-Group/ZO_Fine_tuner.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00414v1">RELATE-Sim: Leveraging Turning Point Theory and LLM Agents to Predict and Understand Long-Term Relationship Dynamics through Interactive Narrative Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 10 pages, 3 figures, Submitted to CHI 2026 Conference
    </div>
    <details class="paper-abstract">
      Most dating technologies optimize for getting together, not staying together. We present RELATE-Sim, a theory-grounded simulator that models how couples behave at consequential turning points-exclusivity talks, conflict-and-repair episodes, relocations-rather than static traits. Two persona-aligned LLM agents (one per partner) interact under a centralized Scene Master that frames each turning point as a compact set of realistic options, advances the narrative, and infers interpretable state changes and an auditable commitment estimate after each scene. On a longitudinal dataset of 71 couples with two-year follow-ups, simulation-aware predictions outperform a personas-only baseline while surfacing actionable markers (e.g., repair attempts acknowledged, clarity shifts) that explain why trajectories diverge. RELATE-Sim pushes the relationship research's focus from matchmaking to maintenance, providing a transparent, extensible platform for understanding and forecasting long-term relationship dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01288v1">Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 main pages, 13 appendix pages
    </div>
    <details class="paper-abstract">
      We draw inspiration from microsaccades, tiny involuntary eye movements that reveal hidden dynamics of human perception, to propose an analogous probing method for large language models (LLMs). Just as microsaccades expose subtle but informative shifts in vision, we show that lightweight position encoding perturbations elicit latent signals that indicate model misbehaviour. Our method requires no fine-tuning or task-specific supervision, yet detects failures across diverse settings including factuality, safety, toxicity, and backdoor attacks. Experiments on multiple state-of-the-art LLMs demonstrate that these perturbation-based probes surface misbehaviours while remaining computationally efficient. These findings suggest that pretrained LLMs already encode the internal evidence needed to flag their own failures, and that microsaccade-inspired interventions provide a pathway for detecting and mitigating undesirable behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v4">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14285v2">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 IEEE Conference standard paper
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23571v2">Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. This paper presents CyberTeam, a benchmark designed to guide LLMs in blue teaming practice. CyberTeam constructs a standardized workflow in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of operational modules tailored to its specific analytical requirements. This transforms threat hunting into a structured sequence of reasoning steps, with each step grounded in a discrete operation and ordered according to task-specific dependencies. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modularized steps. Overall, CyberTeam integrates 30 tasks and 9 operational modules to guide LLMs through standardized threat analysis. We evaluate both leading LLMs and state-of-the-art cybersecurity agents, comparing CyberTeam against open-ended reasoning strategies. Our results highlight the improvements enabled by standardized design, while also revealing the limitations of open-ended reasoning in real-world threat hunting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23573v2">Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06920v2">An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 6 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including accuracy, precision, recall, F1, specificity, FAR, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11679v3">Ambiguity in LLMs is a concept missing problem</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 17 pages, 11 figures, title updated
    </div>
    <details class="paper-abstract">
      Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods to ambiguity handling either rely on the ReACT framework to obtain correct mappings through trial and error, or on supervised fine-tuning to bias models toward specific tasks. In this paper, we adopt a different approach that characterizes representation differences of ambiguous text in the latent space and leverages these differences to identify ambiguity before mapping them to structured data. To detect sentence-level ambiguity, we focus on the relationship between ambiguous questions and their interpretations. Unlike distances calculated by dense embeddings, we introduce a new distance measure based on a path kernel over concepts. With this measurement, we identify patterns to distinguish ambiguous from unambiguous questions. Furthermore, we propose a method for improving LLM performance on ambiguous agentic tool calling through missing concept prediction. Both achieve state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23019v2">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04018v2">AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Prepint, under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. While prior research has studied agents' ability to produce harmful outputs or follow malicious instructions, it remains unclear how likely agents are to spontaneously pursue unintended goals in realistic deployments. In this work, we approach misalignment as a conflict between the internal goals pursued by the model and the goals intended by its deployer. We introduce a misalignment propensity benchmark, \textsc{AgentMisalignment}, a benchmark suite designed to evaluate the propensity of LLM agents to misalign in realistic scenarios. Evaluations cover behaviours such as avoiding oversight, resisting shutdown, sandbagging, and power-seeking. Testing frontier models, we find that more capable agents tend to exhibit higher misalignment on average. We also systematically vary agent personalities through different system prompts and observe that persona characteristics can strongly and unpredictably influence misalignment, sometimes more than the choice of model itself. Our results reveal the limitations of current alignment methods for autonomous LLM agents and underscore the need to rethink misalignment in realistic deployment settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07493v3">LLM-guided Task and Motion Planning using Knowledge-based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Submitted to knowledge based systems
    </div>
    <details class="paper-abstract">
      Performing complex manipulation tasks in dynamic environments requires efficient Task and Motion Planning (TAMP) approaches that combine high-level symbolic plans with low-level motion control. Advances in Large Language Models (LLMs), such as GPT-4, are transforming task planning by offering natural language as an intuitive and flexible way to describe tasks, generate symbolic plans, and reason. However, the effectiveness of LLM-based TAMP approaches is limited due to static and template-based prompting, which limits adaptability to dynamic environments and complex task contexts. To address these limitations, this work proposes a novel Onto-LLM-TAMP framework that employs knowledge-based reasoning to refine and expand user prompts with task-contextual reasoning and knowledge-based environment state descriptions. Integrating domain-specific knowledge into the prompt ensures semantically accurate and context-aware task plans. The proposed framework demonstrates its effectiveness by resolving semantic errors in symbolic plan generation, such as maintaining logical temporal goal ordering in scenarios involving hierarchical object placement. The proposed framework is validated through both simulation and real-world scenarios, demonstrating significant improvements over the baseline approach in terms of adaptability to dynamic environments and the generation of semantically correct task plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22255v2">Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25448v2">Fingerprinting LLMs via Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often modified after release through post-processing such as post-training or quantization, which makes it challenging to determine whether one model is derived from another. Existing provenance detection methods have two main limitations: (1) they embed signals into the base model before release, which is infeasible for already published models, or (2) they compare outputs across models using hand-crafted or random prompts, which are not robust to post-processing. In this work, we propose LLMPrint, a novel detection framework that constructs fingerprints by exploiting LLMs' inherent vulnerability to prompt injection. Our key insight is that by optimizing fingerprint prompts to enforce consistent token preferences, we can obtain fingerprints that are both unique to the base model and robust to post-processing. We further develop a unified verification procedure that applies to both gray-box and black-box settings, with statistical guarantees. We evaluate LLMPrint on five base models and around 700 post-trained or quantized variants. Our results show that LLMPrint achieves high true positive rates while keeping false positive rates near zero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18553v2">The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. While GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale, its inner workings are described as a sequence of ad-hoc algebraic updates that obscure geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: first, the GPTQ error propagation step gains an intuitive geometric interpretation; second, GPTQ inherits the error upper bound of Babai's algorithm under the assumption that no weights are clipped. Leveraging this bound, we design post-training quantization methods that avoid clipping, and outperform the original GPTQ. In addition, we provide efficient GPU inference kernels for the resulting representation. Taken together, these results place GPTQ on a firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v2">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 ECIR 2025 demo format
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recent ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26432v2">AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, blockwise semi-autoregressive (semi-AR) approaches are widely adopted due to their natural support for KV caching and their favorable accuracy-speed trade-off. However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size assumption in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing. Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inference-time optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14824v2">Confirmation Bias as a Cognitive Resource in LLM-Supported Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in group decision-making, but their influence risks fostering conformity and reducing epistemic vigilance. Drawing on the Argumentative Theory of Reasoning, we argue that confirmation bias, often seen as detrimental, can be harnessed as a resource when paired with critical evaluation. We propose a three-step process in which individuals first generate ideas independently, then use LLMs to refine and articulate them, and finally engage with LLMs as epistemic provocateurs to anticipate group critique. This framing positions LLMs as tools for scaffolding disagreement, helping individuals prepare for more productive group discussions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12310v2">Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      This paper addresses the critical need for democratizing large language models (LLM) in the Arab world, a region that has seen slower progress in developing models comparable to state-of-the-art offerings like GPT-4 or ChatGPT 3.5, due to a predominant focus on mainstream languages (e.g., English and Chinese). One practical objective for an Arabic LLM is to utilize an Arabic-specific vocabulary for the tokenizer that could speed up decoding. However, using a different vocabulary often leads to a degradation of learned knowledge since many words are initially out-of-vocabulary (OOV) when training starts. Inspired by the vocabulary learning during Second Language (Arabic) Acquisition for humans, the released AraLLaMA employs progressive vocabulary expansion, which is implemented by a modified BPE algorithm that progressively extends the Arabic subwords in its dynamic vocabulary during training, thereby balancing the OOV ratio at every stage. The ablation study demonstrated the effectiveness of Progressive Vocabulary Expansion. Moreover, AraLLaMA achieves decent performance comparable to the best Arabic LLMs across a variety of Arabic benchmarks. Models, training data, benchmarks, and codes will be all open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23384v3">A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Update the system name in the summary
    </div>
    <details class="paper-abstract">
      LLM inference serving typically scales out with a two-tier architecture: a cluster router distributes requests to multiple inference engines, each of which then in turn performs its own internal scheduling. However, this commonly used paradigm suffers from critical, systemic inefficiency caused by the information gaps across two layers. At the cluster-layer, the router mainly relies on lagging, coarse-grained metrics, such as average latency and queue length to make decisions, resulting in "decision lag" that leads to suboptimal request routing. At the engine-layer, static heuristic scheduling policies cannot effectively handle the dynamic workloads, leading a poor balance between latency and throughput. Besides, these gaps may cause SLO violations and resource waste, especially in heterogeneous cloud environments. To bridge such gaps, we propose NexusSched, a cross-layer framework that shifts LLM serving system from reactive load balancing to predictive orchestration. The core of NexusSched lies in a structurally-informed online performance model that provides accurate, forward-looking per-step latency and capacity estimations. This model empowers two key components. At the engine-layer, LENS performs SLO-aware, adaptive scheduling, dynamically optimizing batching to meet SLOs under real-time loads. At the cluster-layer, PRISM uses predictive signals to perform state-driven routing, maximizing cluster-wide performance and SLO attainment. Performance evaluations show that NexusSched improves SLO attainment by 43% on average and achieves up to 3x throughput speedup in long-context and heterogeneous scenarios. Besides, we also deploy NexusSched on FlowGPT's clusters to demonstrate its advantages in production environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19136v2">On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23619v2">Reasoning Scaffolding: Distilling the Flow of Thought from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The prevailing approach to distilling reasoning from Large Language Models (LLMs)-behavioral cloning from textual rationales-is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding}, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1)predict the next semantic signal, anticipating the reasoning flow, and (2)generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18980v2">From latent factors to language: a user study on LLM-generated explanations for an inherently interpretable matrix-based recommender system</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We investigate whether large language models (LLMs) can generate effective, user-facing explanations from a mathematically interpretable recommendation model. The model is based on constrained matrix factorization, where user types are explicitly represented and predicted item scores share the same scale as observed ratings, making the model's internal representations and predicted scores directly interpretable. This structure is translated into natural language explanations using carefully designed LLM prompts. Many works in explainable AI rely on automatic evaluation metrics, which often fail to capture users' actual needs and perceptions. In contrast, we adopt a user-centered approach: we conduct a study with 326 participants who assessed the quality of the explanations across five key dimensions-transparency, effectiveness, persuasion, trust, and satisfaction-as well as the recommendations themselves. To evaluate how different explanation strategies are perceived, we generate multiple explanation types from the same underlying model, varying the input information provided to the LLM. Our analysis reveals that all explanation types are generally well received, with moderate statistical differences between strategies. User comments further underscore how participants react to each type of explanation, offering complementary insights beyond the quantitative results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04215v2">NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted for the ICAPS 2024 Workshop on Human-Aware and Explainable Planning
    </div>
    <details class="paper-abstract">
      Classical planners are powerful systems, but modeling tasks in input formats such as PDDL is tedious and error-prone. In contrast, planning with Large Language Models (LLMs) allows for almost any input text, but offers no guarantees on plan quality or even soundness. In an attempt to merge the best of these two approaches, some work has begun to use LLMs to automate parts of the PDDL creation process. However, these methods still require various degrees of expert input or domain-specific adaptations. We present NL2Plan, the first fully automatic system for generating complete PDDL tasks from minimal natural language descriptions. NL2Plan uses an LLM to incrementally extract the necessary information from the short text input before creating a complete PDDL description of both the domain and the problem which is finally solved by a classical planner. We evaluate NL2Plan on seven planning domains, five of which are novel and thus not in the LLM training data, and find that NL2Plan outperforms directly generating the files with an LLM+validator combination. As such, NL2Plan is a powerful tool for assistive PDDL modeling and a step towards solving natural language planning task with interpretability and guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19552v2">iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20097v2">Integrated Framework for LLM Evaluation with Answer Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 16pages
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21102v3">Exploring and Controlling Diversity in LLM-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Controlling diversity in LLM-agent simulations is essential for balancing stability in structured tasks with variability in open-ended interactions. However, we observe that dialogue diversity tends to degrade over long-term simulations. To explore the role of prompt design in this phenomenon, we modularized the utterance generation prompt and found that reducing contextual information leads to more diverse outputs. Based on this insight, we propose Adaptive Prompt Pruning (APP), a novel method that allows users to control diversity via a single parameter, lambda. APP dynamically prunes prompt segments based on attention scores and is compatible with existing diversity control methods. We demonstrate that APP effectively modulates diversity through extensive experiments and propose a method to balance the control trade-offs. Our analysis reveals that all prompt components impose constraints on diversity, with the Memory being the most influential. Additionally, high-attention contents consistently suppress output diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17251v2">Training-free LLM Verification via Recycling Few-shot Examples</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (ReFeri). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, ReFeri evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25835v2">Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Test-time scaling enables large language models (LLMs) to improve performance on long-horizon reasoning tasks by allocating additional compute at inference. Tree-search-based approaches achieve state-of-the-art results in this setting, but they are notoriously inefficient, often an order of magnitude slower than simpler iterative methods. We introduce Chain-in-Tree (CiT), a plug-in framework that adaptively decides when to branch during search rather than branching at every step. CiT relies on lightweight Branching Necessity (BN) evaluation methods: BN-DP (Direct Prompting), where an auxiliary LLM directly judges whether a step requires branching, and BN-SC (Self-Consistency), which clusters multiple candidate actions to estimate agreement. We integrate CiT into three representative LLM-in-the-loop tree search frameworks: Tree of Thoughts (ToT-BS), ReST-MCTS, and RAP, and evaluate across GSM8K and Math500. Our results show that: (1) BN-DP consistently reduces token generation, model invocations, and runtime by 75-85 percent across all settings, with negligible accuracy loss and sometimes accuracy gains; (2) BN-SC typically yields substantial savings (up to 80 percent) but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce very long reasoning steps; (3) the quality of auxiliary LLMs is critical, not only the BN evaluator in BN-DP, but also the models used in BN-SC for clustering and equivalence checking. When these roles are filled by smaller LLMs, performance degrades. Importantly, BN-SC does not require LLMs in domains with deterministic action spaces, where clustering can be done programmatically. We also provide a theoretical guarantee that BN-DP never increases LLM invocations relative to the baseline and release a unified implementation of CiT across ToT-BS, ReST-MCTS, and RAP to facilitate reproducibility and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24159v2">Latent Collective Preference Optimization: A General Framework for Robust LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Standard human preference-based alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), are a cornerstone technology for aligning Large Language Models (LLMs) with human values. However, these methods are all underpinned by a critical, yet flawed assumption: human preferences are homogeneous (representing a single, unified preference) and the collected data is noiseless (free from error). In reality, neither is true since human preference is pluralistic and annotators can make mistakes. This creates a discrepancy between the recorded data and the ground-truth preferences, which can misguide the model and degrade its performance. To address this challenge, we introduce Latent Collective Preference Optimization (LCPO). LCPO leverages an Expectation-Maximization (EM) algorithm to learn the latent collective consensus from noisy data. It operates by inferring the correctness of each preference label and using this probability as an adaptive weight to re-calibrate each data point's contribution to the training loss, thereby mitigating noise. We generalize this approach by establishing a theoretical link between arbitrary preference losses and their corresponding probabilistic models, elevating LCPO from a specific algorithm to a general framework for robust preference alignment. Theoretically, we prove that under the condition of a perfectly calibrated model, LCPO is guaranteed to converge to the true noise level of the dataset. Our experiments demonstrate LCPO's effectiveness as a general framework, consistently enhancing four state-of-the-art alignment algorithms (DPO, IPO, SimPO, and CPO). When applied to Mistral and Llama 3 models, the LCPO-enhanced methods achieve substantial win rate gains on AlpacaEval 2 and Arena-Hard, with improvements of up to 7.0% on both benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25889v2">A Multimodal LLM Approach for Visual Question Answering on Multiparametric 3D Brain MRI</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 23 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We introduce mpLLM, a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture for visual question answering over multi-parametric 3D brain MRI (mpMRI). mpLLM routes across modality-level and token-level projection experts to fuse multiple interrelated 3D modalities, enabling efficient training without image-report pretraining. To address limited image-text paired supervision, mpLLM integrates a synthetic visual question answering (VQA) protocol that generates medically relevant VQA from segmentation annotations, and we collaborate with medical experts for clinical validation. mpLLM outperforms strong medical VLM baselines by 5.3% on average across multiple mpMRI datasets. Our study features three main contributions: (1) the first clinically validated VQA dataset for 3D brain mpMRI, (2) a novel multimodal LLM that handles multiple interrelated 3D modalities, and (3) strong empirical results that demonstrate the medical utility of our methodology. Ablations highlight the importance of modality-level and token-level experts and prompt-conditioned routing.
    </details>
</div>
