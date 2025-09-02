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

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14862v2">Bitune: Leveraging Bidirectional Attention to Improve Decoder-Only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Decoder-only large language models typically rely solely on masked causal attention, which limits their expressiveness by restricting information flow to one direction. We propose Bitune, a method that enhances pretrained decoder-only LLMs by incorporating bidirectional attention into prompt processing. We evaluate Bitune in instruction-tuning and question-answering settings, showing significant improvements in performance on commonsense reasoning, arithmetic, and language understanding tasks. Furthermore, extensive ablation studies validate the role of each component of the method, and demonstrate that Bitune is compatible with various parameter-efficient finetuning techniques and full model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20996v1">ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20912v1">Research Challenges in Relational Database Management Systems for LLM Queries</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 This paper will appear in the 6th International Workshop on Applied AI for Database Systems and Applications, AIDB Workshop at VLDB 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential for applications such as text summarization, sentiment analysis, and automated question-answering. Recently, LLMs have also been integrated into relational database management systems to enhance querying and support advanced data processing. Companies such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly within SQL, denoted as LLM queries, to boost data insights. However, open-source solutions currently have limited functionality and poor performance. In this work, we present an early exploration of two open-source systems and one enterprise platform, using five representative queries to expose functional, performance, and scalability limits in today's SQL-invoked LLM integrations. We identify three main issues: enforcing structured outputs, optimizing resource utilization, and improving query planning. We implemented initial solutions and observed improvements in accommodating LLM powered SQL queries. These early gains demonstrate that tighter integration of LLM+DBMS is the key to scalable and efficient processing of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20863v1">Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20818v1">cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 A shorter version has been accepted to the 2025 Conference on Information and Knowledge Management
    </div>
    <details class="paper-abstract">
      Many multi-agent reinforcement learning (MARL) algorithms are trained in fixed simulation environments, making them brittle when deployed in real-world scenarios with more complex and uncertain conditions. Contextual MARL (cMARL) addresses this by parameterizing environments with context variables and training a context-agnostic policy that performs well across all environment configurations. Existing cMARL methods attempt to use curriculum learning to help train and evaluate context-agnostic policies, but they often rely on unreliable proxy signals, such as value estimates or generalized advantage estimates that are noisy and unstable in multi-agent settings due to inter-agent dynamics and partial observability. To address these issues, we propose Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending (cMALC-D), a framework that uses Large Language Models (LLMs) to generate semantically meaningful curricula and provide a more robust evaluation signal. To prevent mode collapse and encourage exploration, we introduce a novel diversity-based context blending mechanism that creates new training scenarios by combining features from prior contexts. Experiments in traffic signal control domains demonstrate that cMALC-D significantly improves both generalization and sample efficiency compared to existing curriculum learning baselines. We provide code at https://github.com/DaRL-LibSignal/cMALC-D.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19720v2">Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03661v2">Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 79 pages (29 main), with 6+6 figures and 2 tables, presenting a more concise and updated manuscript
    </div>
    <details class="paper-abstract">
      Gravitational-wave signal detection with unknown source parameters buried in dynamic detector noise remains a formidable computational challenge. Existing approaches face core limitations from restrictive assumptions: traditional methods rely on predefined theoretical priors, while neural networks introduce hidden biases and lack interpretability. We propose Evolutionary Monte Carlo Tree Search (Evo-MCTS), the first integration of large language model (LLM) guidance with domain-aware physical constraints for automated gravitational wave detection. This framework systematically explores algorithmic solution spaces through tree-structured search enhanced by evolutionary optimization, combining MCTS for strategic exploration with evolutionary algorithms for solution refinement. The LLM component provides domain-aware heuristics while maintaining interpretability through explicit algorithmic pathway generation. Experimental validation demonstrates substantial performance improvements, achieving a 20.2% improvement over state-of-the-art gravitational wave detection algorithms on the MLGWSC-1 benchmark dataset and a remarkable 59.1% improvement over other LLM-based algorithm optimization frameworks. Beyond performance improvements, our framework establishes a transferable methodology for automated algorithmic discovery across computational science domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15165v2">SoAy: A Solution-based LLM API-using Methodology for Academic Information Seeking</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 KDD 2025; 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) for academic API usage shows promise in reducing researchers' academic information seeking efforts. However, current LLM API-using methods struggle with complex API coupling commonly encountered in academic queries. To address this, we introduce SoAy, a solution-based LLM API-using methodology for academic information seeking. It uses code with a solution as the reasoning method, where a solution is a pre-constructed API calling sequence. The addition of the solution reduces the difficulty for the model to understand the complex relationships between APIs. Code improves the efficiency of reasoning. To evaluate SoAy, we introduce SoAyBench, an evaluation benchmark accompanied by SoAyEval, built upon a cloned environment of APIs from AMiner. Experimental results demonstrate a 34.58-75.99\% performance improvement compared to state-of-the-art LLM API-based baselines. All datasets, codes, tuned models, and deployed online services are publicly accessible at https://github.com/RUCKBReasoning/SoAy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17387v2">Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20525v1">Enhancing Health Fact-Checking with LLM-Generated Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Fact-checking for health-related content is challenging due to the limited availability of annotated training data. In this study, we propose a synthetic data generation pipeline that leverages large language models (LLMs) to augment training data for health-related fact checking. In this pipeline, we summarize source documents, decompose the summaries into atomic facts, and use an LLM to construct sentence-fact entailment tables. From the entailment relations in the table, we further generate synthetic text-claim pairs with binary veracity labels. These synthetic data are then combined with the original data to fine-tune a BERT-based fact-checking model. Evaluation on two public datasets, PubHealth and SciFact, shows that our pipeline improved F1 scores by up to 0.019 and 0.049, respectively, compared to models trained only on the original data. These results highlight the effectiveness of LLM-driven synthetic data augmentation in enhancing the performance of health-related fact-checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v4">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20514v1">SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06056v2">Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16201v2">SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens to enable efficient speculation without sacrificing accuracy. To achieve this, we performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B. Code is available at https://github.com/zju-jiyicheng/SpecVLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20462v1">Automated Quality Assessment for LLM-Based Complex Qualitative Coding: A Confidence-Diversity Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 21 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      While previous research demonstrated effective automated quality assessment for accessible LLM coding tasks, a fundamental question remains: can confidence-diversity frameworks maintain reliability for complex analytical tasks requiring specialized domain expertise and extensive text comprehension? Traditional inter-coder reliability measures become prohibitively expensive at scale, yet the lack of reliable automated quality assessment methods creates methodological barriers to AI adoption in sophisticated qualitative research. This study extends dual-signal quality assessment combining model confidence and inter-model consensus from accessible to complex analytical domains. We systematically validate this approach across three domains: legal reasoning (390 Supreme Court cases), political analysis (645 hyperpartisan articles), and medical classification (1,000 clinical transcripts). Results demonstrate that uncertainty-based indicators maintain predictive validity in complex tasks, with external entropy showing consistent negative correlations with accuracy (r = -0.179 to -0.273, p < 0.001) and confidence exhibiting positive correlations in two domains (r = 0.104 to 0.429). Systematic weight optimization achieves 6.6 to 113.7 percent improvements over single-signal approaches, with optimized weights transferring effectively across domains (100 percent success rate). An intelligent triage system reduces manual verification effort by 44.6 percent while maintaining quality standards. These findings establish that automated quality assessment can scale from accessible to complex analytical tasks, providing practical tools for expanding AI-assisted qualitative research. Future work will focus on addressing long-tail challenges in high-disagreement, low-confidence cases to further enhance screening efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20453v1">MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We introduce MCP-Bench, a benchmark for evaluating large language models (LLMs) on realistic, multi-step tasks that demand tool use, cross-tool coordination, precise parameter control, and planning/reasoning for solving tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28 representative live MCP servers spanning 250 tools across domains such as finance, traveling, scientific computing, and academic search. Unlike prior API-based benchmarks, each MCP server provides a set of complementary tools designed to work together, enabling the construction of authentic, multi-step tasks with rich input-output coupling. Tasks in MCP-Bench test agents' ability to retrieve relevant tools from fuzzy instructions without explicit tool names, plan multi-hop execution trajectories for complex objectives, ground responses in intermediate tool outputs, and orchestrate cross-domain workflows - capabilities not adequately evaluated by existing benchmarks that rely on explicit tool specifications, shallow few-step workflows, and isolated domain operations. We propose a multi-faceted evaluation framework covering tool-level schema understanding and usage, trajectory-level planning, and task completion. Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench. Code and data: https://github.com/Accenture/mcp-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20444v1">Ransomware 3.0: Self-Composing and LLM-Orchestrated</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20443v1">Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Aware Unlearning with Proxy Constraint</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods lack a sound forgetting boundary, causing some samples to be under-forgotten, leaving residual leakage risks, while others remain over-forgotten at the expense of degraded utility. In this work, we propose EAGLE-PC (Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint), a novel unlearning framework that addresses these limitations through two key components. First, entanglement-awareness guided loss reweighting determines the forgetting effort of each sample by measuring its similarity to retain samples in the embedding space, enabling more targeted and effective unlearning. Second, a proxy constraint leveraging ICL (In-Context Learning) generated test data softly regularizes the forgetting process, effectively mitigating over-forgetting. EAGLE-PC is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EAGLE-PC on the TOFU and MUSE benchmarks, showing consistent improvements in the forgetting-utility trade-off across multiple LLMs. Combined with the NPO+GD optimizer, it approaches full retraining performance, offering a scalable and robust unlearning solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20420v1">CAMB: A comprehensive industrial LLM benchmark on civil aviation maintenance</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Civil aviation maintenance is a domain characterized by stringent industry standards. Within this field, maintenance procedures and troubleshooting represent critical, knowledge-intensive tasks that require sophisticated reasoning. To address the lack of specialized evaluation tools for large language models (LLMs) in this vertical, we propose and develop an industrial-grade benchmark specifically designed for civil aviation maintenance. This benchmark serves a dual purpose: It provides a standardized tool to measure LLM capabilities within civil aviation maintenance, identifying specific gaps in domain knowledge and complex reasoning. By pinpointing these deficiencies, the benchmark establishes a foundation for targeted improvement efforts (e.g., domain-specific fine-tuning, RAG optimization, or specialized prompt engineering), ultimately facilitating progress toward more intelligent solutions within civil aviation maintenance. Our work addresses a significant gap in the current LLM evaluation, which primarily focuses on mathematical and coding reasoning tasks. In addition, given that Retrieval-Augmented Generation (RAG) systems are currently the dominant solutions in practical applications , we leverage this benchmark to evaluate existing well-known vector embedding models and LLMs for civil aviation maintenance scenarios. Through experimental exploration and analysis, we demonstrate the effectiveness of our benchmark in assessing model performance within this domain, and we open-source this evaluation benchmark and code to foster further research and development:https://github.com/CamBenchmark/cambenchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20416v1">DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20401v1">Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 In Proceedings of 2nd Workshop on Evaluating and Applying Recommendation Systems with Large Language Models (EARL) at RecSys 2025 (EARL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for recommendation tasks due to their general-purpose capabilities. While LLMs perform well in rich-context settings, their behavior in cold-start scenarios, where only limited signals such as age, gender, or language are available, raises fairness concerns because they may rely on societal biases encoded during pretraining. We introduce a benchmark specifically designed to evaluate fairness in zero-context recommendation. Our modular pipeline supports configurable recommendation domains and sensitive attributes, enabling systematic and flexible audits of any open-source LLM. Through evaluations of state-of-the-art models (Gemma 3 and Llama 3.2), we uncover consistent biases across recommendation domains (music, movies, and colleges) including gendered and cultural stereotypes. We also reveal a non-linear relationship between model size and fairness, highlighting the need for nuanced analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20395v1">Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision. We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20384v1">Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Under review for AAAI 2026
    </div>
    <details class="paper-abstract">
      In this work, we introduce Entropy Area Score (EAS), a simple yet effective metric to quantify uncertainty in the answer generation process of reasoning large language models (LLMs). EAS requires neither external models nor repeated sampling, it integrates token-level predictive entropy from the model itself to capture the evolution of uncertainty during generation. Empirical results show that EAS is strongly correlated with answer entropy across models and datasets. In training data selection, EAS identifies high-potential samples and consistently outperforms Pass Rate filtering under equal sample budgets, improving student model accuracy on math benchmarks. EAS is both efficient and interpretable, offering a practical tool for uncertainty modeling and data quality assessment in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v3">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in $\texttt{list.get(i)}$. Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v2">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20373v1">Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at https://github.com/Graph-Reasoner/Graph-R1, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20351v1">Joint Enhancement of Relational Reasoning for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 9 pages, 5 pages Accepted by EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite significant progress, large language models (LLMs) still struggle with long contexts due to memory limitations and their inability to tackle complex and long-context tasks. Additionally, LLMs often suffer from a lack of transparency and are prone to producing hallucinations. To address these challenges, we propose \textbf{JERR}, a novel framework designed to enhance long-context comprehension via graph-based reasoning in LLMs. JERR integrates three key components: synopsis extraction, graph construction, and relational reasoning. First, synopsis is extracted by chunking text strategically, allowing the model to summarize and understand information more efficiently. Second, we build a directed acyclic graph (DAG) to resolve redundancy, ensuring logical consistency and clarity. Finally, we incorporate Monte Carlo Tree Search (MCTS) to help the model navigate complex reasoning paths, ensuring more accurate and interpretable outputs. This framework provides a novel solution that enables LLMs to handle extended contexts and complex reasoning tasks with improved reliability and transparency. Experimental results show that JERR consistently outperforms all baselines on the ROUGE and F1 metrics, achieving the highest scores on the LLM-Rater evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20340v1">Boosting Skeleton-Driven SMT Solver Fuzzing by Leveraging LLM to Produce Formula Generators</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with the LLMs introduce substantial computational overhead. In this study, we present Chimera, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of reusable term (i.e., logical expression) generators. Particularly, Chimera uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Chimera populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Chimera requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Chimera on two leading SMT solvers: Z3 and cvc5. Our experiments show that Chimera has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19512v3">Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks often leads to catastrophic forgetting, notably degrading the safety of originally aligned models. While some existing methods attempt to restore safety by incorporating additional safety data, the quality of such data typically falls short of that used in the original alignment process. Moreover, these high-quality safety datasets are generally inaccessible, making it difficult to fully recover the model's original safety. We ask: How can we preserve safety while improving downstream task performance without additional safety data? We show that simply merging the weights of pre- and post-fine-tuned models effectively mitigates safety degradation while enhancing performance. Experiments across different downstream tasks and models validate the method's practicality and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v3">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20333v1">Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20325v1">GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 54 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance. To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations. We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13580v2">LLM Test Generation via Iterative Hybrid Program Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 This paper has been accepted for publication in 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      Automating unit test generation remains a significant challenge, particularly for complex methods in real-world projects. While Large Language Models (LLMs) have made strides in code generation, they struggle to achieve high branch coverage due to their limited ability to reason about intricate control flow structures. To address this limitation, we introduce Panta, a technique that emulates the iterative process human developers follow when analyzing code and constructing test cases. Panta integrates static control flow analysis and dynamic code coverage analysis to systematically guide LLMs in identifying uncovered execution paths and generating better test cases. By incorporating an iterative feedback-driven mechanism, our technique continuously refines test generation based on static and dynamic path coverage insights, ensuring more comprehensive and effective testing. Our empirical evaluation, conducted on classes with high cyclomatic complexity from open-source projects, demonstrates that Panta achieves 26% higher line coverage and 23% higher branch coverage compared to the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21238v1">Addressing accuracy and hallucination of LLMs in Alzheimer's disease research through knowledge graphs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In the past two years, large language model (LLM)-based chatbots, such as ChatGPT, have revolutionized various domains by enabling diverse task completion and question-answering capabilities. However, their application in scientific research remains constrained by challenges such as hallucinations, limited domain-specific knowledge, and lack of explainability or traceability for the response. Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising approach to improving chatbot reliability by integrating domain-specific contextual information before response generation, addressing some limitations of standard LLMs. Despite its potential, there are only limited studies that evaluate GraphRAG on specific domains that require intensive knowledge, like Alzheimer's disease or other biomedical domains. In this paper, we assess the quality and traceability of two popular GraphRAG systems. We compile a database of 50 papers and 70 expert questions related to Alzheimer's disease, construct a GraphRAG knowledge base, and employ GPT-4o as the LLM for answering queries. We then compare the quality of responses generated by GraphRAG with those from a standard GPT-4o model. Additionally, we discuss and evaluate the traceability of several Retrieval-Augmented Generation (RAG) and GraphRAG systems. Finally, we provide an easy-to-use interface with a pre-built Alzheimer's disease database for researchers to test the performance of both standard RAG and GraphRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21204v1">Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21184v1">BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21141v1">Adaptive LLM Routing under Budget Constraints</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025 (findings)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing, but their varying capabilities and costs pose challenges in practical applications. LLM routing addresses this by dynamically selecting the most suitable LLM for each query/task. Previous approaches treat this as a supervised learning problem, assuming complete knowledge of optimal query-LLM pairings. However, real-world scenarios lack such comprehensive mappings and face evolving user queries. We thus propose to study LLM routing as a contextual bandit problem, enabling adaptive decision-making using bandit feedback without requiring exhaustive inference across all LLMs for all queries (in contrast to supervised routing). To address this problem, we develop a shared embedding space for queries and LLMs, where query and LLM embeddings are aligned to reflect their affinity. This space is initially learned from offline human preference data and refined through online bandit feedback. We instantiate this idea through Preference-prior Informed Linucb fOr adaptive rouTing (PILOT), a novel extension of LinUCB. To handle diverse user budgets for model routing, we introduce an online cost policy modeled as a multi-choice knapsack problem, ensuring resource-efficient routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19580v1">ArgCMV: An Argument Summarization Benchmark for the LLM-era</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Key point extraction is an important task in argument summarization which involves extracting high-level short summaries from arguments. Existing approaches for KP extraction have been mostly evaluated on the popular ArgKP21 dataset. In this paper, we highlight some of the major limitations of the ArgKP21 dataset and demonstrate the need for new benchmarks that are more representative of actual human conversations. Using SoTA large language models (LLMs), we curate a new argument key point extraction dataset called ArgCMV comprising of around 12K arguments from actual online human debates spread across over 3K topics. Our dataset exhibits higher complexity such as longer, co-referencing arguments, higher presence of subjective discourse units, and a larger range of topics over ArgKP21. We show that existing methods do not adapt well to ArgCMV and provide extensive benchmark results by experimenting with existing baselines and latest open source models. This work introduces a novel KP extraction dataset for long-context online discussions, setting the stage for the next generation of LLM-driven summarization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19578v1">Towards a Holistic and Automated Evaluation Framework for Multi-Level Comprehension of LLMs in Book-Length Contexts</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      We introduce HAMLET, a holistic and automated framework for evaluating the long-context comprehension of large language models (LLMs). HAMLET structures source texts into a three-level key-fact hierarchy at root-, branch-, and leaf-levels, and employs query-focused summarization to evaluate how well models recall and faithfully represent information at each level. To validate the reliability of our fully automated pipeline, we conduct a systematic human study, showing that our automatic evaluation achieves over 90% agreement with expert human judgments, while reducing the cost by up to 25 times. HAMLET reveals that LLMs struggle with fine-grained comprehension, especially at the leaf level, and are sensitive to positional effects like the lost-in-the-middle. Analytical queries pose greater challenges than narrative ones, and consistent performance gaps emerge between open-source and proprietary models, as well as across model scales. Our code and dataset are publicly available at https://github.com/DISL-Lab/HAMLET.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19576v1">ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We validate the effectiveness of the proposed RL paradigm through extensive experiments on coding problems. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v1">Just Because You Can, Doesn't Mean You Should: LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19559v1">Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) is a GPU-intensive task where traditional autoscalers fall short, particularly for modern Prefill-Decode (P/D) disaggregated architectures. This architectural shift, while powerful, introduces significant operational challenges, including inefficient use of heterogeneous hardware, network bottlenecks, and critical imbalances between prefill and decode stages. We introduce HeteroScale, a coordinated autoscaling framework that addresses the core challenges of P/D disaggregated serving. HeteroScale combines a topology-aware scheduler that adapts to heterogeneous hardware and network constraints with a novel metric-driven policy derived from the first large-scale empirical study of autoscaling signals in production. By leveraging a single, robust metric to jointly scale prefill and decode pools, HeteroScale maintains architectural balance while ensuring efficient, adaptive resource management. Deployed in a massive production environment on tens of thousands of GPUs, HeteroScale has proven its effectiveness, increasing average GPU utilization by a significant 26.6 percentage points and saving hundreds of thousands of GPU-hours daily, all while upholding stringent service level objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19558v1">Functional Consistency of LLM Code Embeddings: A Self-Evolving Data Synthesis Framework for Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Embedding models have demonstrated strong performance in tasks like clustering, retrieval, and feature extraction while offering computational advantages over generative models and cross-encoders. Benchmarks such as MTEB have shown that text embeddings from large language models (LLMs) capture rich semantic information, but their ability to reflect code-level functional semantics remains unclear. Existing studies largely focus on code clone detection, which emphasizes syntactic similarity and overlooks functional understanding. In this paper, we focus on the functional consistency of LLM code embeddings, which determines if two code snippets perform the same function regardless of syntactic differences. We propose a novel data synthesis framework called Functionality-Oriented Code Self-Evolution to construct diverse and challenging benchmarks. Specifically, we define code examples across four semantic and syntactic categories and find that existing datasets predominantly capture syntactic properties. Our framework generates four unique variations from a single code instance, providing a broader spectrum of code examples that better reflect functional differences. Extensive experiments on three downstream tasks-code clone detection, code functional consistency identification, and code retrieval-demonstrate that embedding models significantly improve their performance when trained on our evolved datasets. These results highlight the effectiveness and generalization of our data synthesis framework, advancing the functional understanding of code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v2">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program P and (2) the slicing criteria consisting of a query step s and a query variable v read by s, RecovSlicing computes the runtime definition of v on the trace by estimating the miss-recorded execution of P. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in list.get(i). Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05004v2">R-Zero: Self-Evolving Reasoning LLM from Zero Data</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08292v2">Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 27 pages total (10-page main paper + 17-page appendix), 12 figures, 6 tables. Submitted to ICML 2025 (under review)
    </div>
    <details class="paper-abstract">
      Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at https://github.com/brando90/putnam-axiom.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19504v1">Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) agents augmented with domain tools promise to autonomously execute complex tasks requiring human-level intelligence, such as customer service and digital assistance. However, their practical deployment is often limited by their low success rates under complex real-world environments. To tackle this, prior research has primarily focused on improving the agents themselves, such as developing strong agentic LLMs, while overlooking the role of the system environment in which the agent operates. In this paper, we study a complementary direction: improving agent success rates by optimizing the system environment in which the agent operates. We collect 142 agent traces (3,656 turns of agent-environment interactions) across 5 state-of-the-art agentic benchmarks. By analyzing these agent failures, we propose a taxonomy for agent-environment interaction failures that includes 6 failure modes. Guided by these findings, we design Aegis, a set of targeted environment optimizations: 1) environment observability enhancement, 2) common computation offloading, and 3) speculative agentic actions. These techniques improve agent success rates on average by 6.7-12.5%, without any modifications to the agent and underlying LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19484v1">Rule Synergy Analysis using LLMs: State of the Art and Implications</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Submitted for publication at the IEEE Transactions on Games 2024, Special Issue on Large Language Models and Games (10 pages excluding appendix, 3 figures)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance across a variety of domains, including logical reasoning, mathematics, and more. In this paper, we investigate how well LLMs understand and reason about complex rule interactions in dynamic environments, such as card games. We introduce a dataset of card synergies from the game Slay the Spire, where pairs of cards are classified based on their positive, negative, or neutral interactions. Our evaluation shows that while LLMs excel at identifying non-synergistic pairs, they struggle with detecting positive and, particularly, negative synergies. We categorize common error types, including issues with timing, defining game states, and following game rules. Our findings suggest directions for future research to improve model performance in predicting the effect of rules and their interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v5">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is labor-intensive, especially for strongly typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in LLMs have enabled oracle generation from natural language descriptions, aligning better with user requirements. However, existing LLM-based methods often require fine-tuning or rely on external tools such as EvoSuite for test prefix generation, making them costly or cumbersome to apply in practice. In this work, we propose CANDOR, a novel prompt engineering-based LLM framework for automated unit test generation in Java. CANDOR orchestrates multiple specialized LLM agents to collaboratively generate complete tests. To mitigate the notorious hallucinations in LLMs and improve oracle correctness, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generates accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments show that CANDOR is comparable with EvoSuite in generating tests with high code coverage and clearly superior in terms of mutation score. Moreover, our prompt engineering-based approach CANDOR significantly outperforms the SOTA fine-tuning-based oracle generator TOGLL by at least 21.1 percentage points in oracle correctness on both correct and faulty source code. Further ablation studies confirm the critical contributions of key agents in generating high-quality tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18076v2">Neither Valid nor Reliable? Investigating the Use of LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Prepared for conference submission
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20279v1">How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted by COLM 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20274v1">Predictable LLM Serving on GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Latency-sensitive inference on shared A100 clusters often suffers noisy-neighbor interference on the PCIe fabric, inflating tail latency and SLO violations. We present a fabric-agnostic, VM-deployable host-level controller that combines dynamic Multi-Instance GPU (MIG) reconfiguration, PCIe-aware placement, and lightweight guardrails (MPS quotas, cgroup I/O). It samples per-tenant tails and system signals, uses topology hints to avoid PCIe hot spots, and gates actions with dwell/cool-down to avoid thrash. On a single host and a 2-node (16-GPU) cluster, SLO miss-rate is reduced by \(\approx\)32\% (\(\approx\)1.5) and p99 latency improves \(\approx\)15\% with \(\leq\)5\% throughput cost versus static MIG and naive placement; ablations show MIG and placement contribute comparably. We also evaluate LLM serving with vLLM on OLMo 2 7B Instruct: TTFT p99 improves \(\approx\)10--15\% at \(\leq\)5\% cost without changing the controller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20263v1">Athena: Intermediate Representations for Iterative Scaffolded App Generation with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      It is challenging to generate the code for a complete user interface using a Large Language Model (LLM). User interfaces are complex and their implementations often consist of multiple, inter-related files that together specify the contents of each screen, the navigation flows between the screens, and the data model used throughout the application. It is challenging to craft a single prompt for an LLM that contains enough detail to generate a complete user interface, and even then the result is frequently a single large and difficult to understand file that contains all of the generated screens. In this paper, we introduce Athena, a prototype application generation environment that demonstrates how the use of shared intermediate representations, including an app storyboard, data model, and GUI skeletons, can help a developer work with an LLM in an iterative fashion to craft a complete user interface. These intermediate representations also scaffold the LLM's code generation process, producing organized and structured code in multiple files while limiting errors. We evaluated Athena with a user study that found 75% of participants preferred our prototype over a typical chatbot-style baseline for prototyping apps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v5">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Social networks profoundly influence how humans form opinions, exchange information, and organize collectively. As large language models (LLMs) are increasingly embedded into social and professional environments, it is critical to understand whether their interactions approximate human-like network dynamics. We develop a framework to study the network formation behaviors of multiple LLM agents and benchmark them against human decisions. Across synthetic and real-world settings, including friendship, telecommunication, and employment networks, we find that LLMs consistently reproduce fundamental micro-level principles such as preferential attachment, triadic closure, and homophily, as well as macro-level properties including community structure and small-world effects. Importantly, the relative emphasis of these principles adapts to context: for example, LLMs favor homophily in friendship networks but heterophily in organizational settings, mirroring patterns of social mobility. A controlled human-subject survey confirms strong alignment between LLMs and human participants in link-formation decisions. These results establish that LLMs can serve as powerful tools for social simulation and synthetic data generation, while also raising critical questions about bias, fairness, and the design of AI systems that participate in human networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20258v1">SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown progress in GPU kernel performance engineering using inefficient search-based methods that optimize around runtime. Any existing approach lacks a key characteristic that human performance engineers rely on for near-optimal utilization -- hardware-awareness. By leveraging the workload's specific memory access patterns, architecture specifications, filtered profiling logs, and reflections on historical performance, we can make software-level optimizations that are tailored to the underlying hardware. SwizzlePerf automatically generates spatial optimizations for GPU kernels on disaggregated architectures by giving LLMs explicit hardware-awareness. For a GEMM kernel, SwizzlePerf takes less than 5 minutes to generate the same hardware-specific optimal swizzling pattern that took expert performance engineers 2 weeks to find. On a suite of 10 diverse ML and Science kernels, SwizzlePerf can generate swizzling patterns for 9 of the kernels that achieve up to a 2.06x speedup and 70% improvement in L2 hit rate. This work is the first of many steps toward systematically creating hardware-aware LLM performance engineering agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20151v1">IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 17 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21094v1">Video-LLMs with Temporal Visual Screening</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Humans naturally perform temporal screening by dragging the progress bar and focusing on salient temporal segments, but current Video Large Language Models (Video-LLMs) struggle to capture fine-grained temporal semantics due to sparse frame sampling and insufficient inter-frame reasoning supervision during their training. To address this, Inspired by well-established cognitive science principles, we propose Temporal Visual Screening (TVS), a new task that universally pre-processes video question answering and instruction tuning data by: (1) retaining focus-critical video segments, (2) synchronously reconstructing queries to their most direct form while preserving answer consistency, and (3) keeping the invariance and consistency for any possible answer. TVS is formulated as a modular front-end adapter task that can be seamlessly integrated into both Video Instruction Tuning (training) and Video Question Answering (inference) pipelines. TVS optimizes distribution of reasoning burden and cognitive load; during training, it aligns queries with focus-critical visual information; at inference, it enables query-aware segment focus and streamlined query representations. In particular, we curate the first benchmark for TVS and propose ReSimplifyIt, a baseline outperforming prior approaches on seemingly similar tasks by 0.47 in F-1 score on video trimming while achieving competitive query rewriting performance. Experiments demonstrate that incorporating TVS yields relative gains of 7.33% (training) and 34.6% (inference), demonstrating the effectiveness of temporal information screening for improving video-language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v1">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model(LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20030v1">Large Language Models (LLMs) for Electronic Design Automation (EDA)</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted by IEEE International System-on-Chip Conference
    </div>
    <details class="paper-abstract">
      With the growing complexity of modern integrated circuits, hardware engineers are required to devote more effort to the full design-to-manufacturing workflow. This workflow involves numerous iterations, making it both labor-intensive and error-prone. Therefore, there is an urgent demand for more efficient Electronic Design Automation (EDA) solutions to accelerate hardware development. Recently, large language models (LLMs) have shown remarkable advancements in contextual comprehension, logical reasoning, and generative capabilities. Since hardware designs and intermediate scripts can be represented as text, integrating LLM for EDA offers a promising opportunity to simplify and even automate the entire workflow. Accordingly, this paper provides a comprehensive overview of incorporating LLMs into EDA, with emphasis on their capabilities, limitations, and future opportunities. Three case studies, along with their outlook, are introduced to demonstrate the capabilities of LLMs in hardware design, testing, and optimization. Finally, future directions and challenges are highlighted to further explore the potential of LLMs in shaping the next-generation EDA, providing valuable insights for researchers interested in leveraging advanced AI technologies for EDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20024v1">Using item recommendations and LLMs in marketing email titles</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic. 3 figures
    </div>
    <details class="paper-abstract">
      E-commerce marketplaces make use of a number of marketing channels like emails, push notifications, etc. to reach their users and stimulate purchases. Personalized emails especially are a popular touch point for marketers to inform users of latest items in stock, especially for those who stopped visiting the marketplace. Such emails contain personalized recommendations tailored to each user's interests, enticing users to buy relevant items. A common limitation of these emails is that the primary entry point, the title of the email, tends to follow fixed templates, failing to inspire enough interest in the contents. In this work, we explore the potential of large language models (LLMs) for generating thematic titles that reflect the personalized content of the emails. We perform offline simulations and conduct online experiments on the order of millions of users, finding our techniques useful in improving the engagement between customers and our emails. We highlight key findings and learnings as we productionize the safe and automated generation of email titles for millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20015v1">Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 11+25 pages, 4+11 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning LLMs on narrowly harmful datasets can lead to behavior that is broadly misaligned with respect to human values. To understand when and how this emergent misalignment occurs, we develop a comprehensive framework for detecting and characterizing rapid transitions during fine-tuning using both distributional change detection methods as well as order parameters that are formulated in plain English and evaluated by an LLM judge. Using an objective statistical dissimilarity measure, we quantify how the phase transition that occurs during fine-tuning affects multiple aspects of the model. In particular, we assess what percentage of the total distributional change in model outputs is captured by different aspects, such as alignment or verbosity, providing a decomposition of the overall transition. We also find that the actual behavioral transition occurs later in training than indicated by the peak in the gradient norm alone. Our framework enables the automated discovery and quantification of language-based order parameters, which we demonstrate on examples ranging from knowledge questions to politics and ethics.
    </details>
</div>
