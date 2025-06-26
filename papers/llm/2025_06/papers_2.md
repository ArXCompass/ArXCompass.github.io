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

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09710v2">DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions-varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training. Code: https://github.com/ZhentingWang/DUMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13597v3">LaPuda: LLM-Enabled Policy-Based Query Optimizer for Multi-modal Data</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 Yifan and Haodi contributed equally to the work, accepted by PAKDD 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) has marked a pivotal moment in the field of machine learning and deep learning. Recently its capability for query planning has been investigated, including both single-modal and multi-modal queries. However, there is no work on the query optimization capability of LLM. As a critical (or could even be the most important) step that significantly impacts the execution performance of the query plan, such analysis and attempts should not be missed. From another aspect, existing query optimizers are usually rule-based or rule-based + cost-based, i.e., they are dependent on manually created rules to complete the query plan rewrite/transformation. Given the fact that modern optimizers include hundreds to thousands of rules, designing a multi-modal query optimizer following a similar way is significantly time-consuming since we will have to enumerate as many multi-modal optimization rules as possible, which has not been well addressed today. In this paper, we investigate the query optimization ability of LLM and use LLM to design LaPuda, a novel LLM and Policy based multi-modal query optimizer. Instead of enumerating specific and detailed rules, LaPuda only needs a few abstract policies to guide LLM in the optimization, by which much time and human effort are saved. Furthermore, to prevent LLM from making mistakes or negative optimization, we borrow the idea of gradient descent and propose a guided cost descent (GCD) algorithm to perform the optimization, such that the optimization can be kept in the correct direction. In our evaluation, our methods consistently outperform the baselines in most cases. For example, the optimized plans generated by our methods result in 1~3x higher execution speed than those by the baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17670v1">Online Multi-LLM Selection via Contextual Bandits under Unstructured Context Evolution</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit diverse response behaviors, costs, and strengths, making it challenging to select the most suitable LLM for a given user query. We study the problem of adaptive multi-LLM selection in an online setting, where the learner interacts with users through multi-step query refinement and must choose LLMs sequentially without access to offline datasets or model internals. A key challenge arises from unstructured context evolution: the prompt dynamically changes in response to previous model outputs via a black-box process, which cannot be simulated, modeled, or learned. To address this, we propose the first contextual bandit framework for sequential LLM selection under unstructured prompt dynamics. We formalize a notion of myopic regret and develop a LinUCB-based algorithm that provably achieves sublinear regret without relying on future context prediction. We further introduce budget-aware and positionally-aware (favoring early-stage satisfaction) extensions to accommodate variable query costs and user preferences for early high-quality responses. Our algorithms are theoretically grounded and require no offline fine-tuning or dataset-specific training. Experiments on diverse benchmarks demonstrate that our methods outperform existing LLM routing strategies in both accuracy and cost-efficiency, validating the power of contextual bandits for real-time, adaptive LLM selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13857v2">How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 40 pages, 4 figures, ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite the remarkable success of Transformer-based large language models (LLMs) across various domains, understanding and enhancing their mathematical capabilities remains a significant challenge. In this paper, we conduct a rigorous theoretical analysis of LLMs' mathematical abilities, with a specific focus on their arithmetic performances. We identify numerical precision as a key factor that influences their effectiveness in arithmetical tasks. Our results show that Transformers operating with low numerical precision fail to address arithmetic tasks, such as iterated addition and integer multiplication, unless the model size grows super-polynomially with respect to the input length. In contrast, Transformers with standard numerical precision can efficiently handle these tasks with significantly smaller model sizes. We further support our theoretical findings through empirical experiments that explore the impact of varying numerical precision on arithmetic tasks, providing valuable insights for improving the mathematical reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17642v1">May the Feedback Be with You! Unlocking the Power of Feedback-Driven Deep Learning Framework Fuzzing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) Infrastructures, represented by Deep Learning (DL) frameworks, have served as fundamental DL systems over the last decade. However, the bugs in DL frameworks could lead to catastrophic consequences in some critical scenarios (e.g., healthcare and autonomous driving). A simple yet effective way to find bugs in DL frameworks is fuzz testing (Fuzzing). Unfortunately, existing fuzzing techniques have not comprehensively considered multiple types of feedback. Additionally, they analyze feedback in a coarse-grained manner, such as mutating the test cases only according to whether the coverage increases. Recently, researchers introduced Large Language Models (LLMs) into fuzzing. However, current LLM-based fuzzing techniques only focus on using LLMs to generate test cases while overlooking their potential to analyze feedback information, failing to create more valid and diverse test cases. To fill this gap, we propose FUEL to break the seal of Feedback-driven fuzzing for DL frameworks. The backbone of FUEL comprises two LLM-based agents, namely analysis LLM and generation LLM. Analysis LLM agent infers analysis summaries from feedback information, while the generation LLM agent creates tests guided by these analysis summaries. So far, FUEL has detected 104 bugs for PyTorch and TensorFlow, with 93 confirmed as new bugs, 47 already fixed, and 5 assigned with CVE IDs. Our work indicates that considering multiple types of feedback is beneficial to fuzzing performance, and leveraging LLMs to analyze feedback information is a promising direction. Our artifact is available at https://github.com/NJU-iSE/FUEL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17637v1">Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various domains but encounter substantial challenges in tackling optimization modeling tasks for Operations Research (OR), particularly when dealing with complex problem. In this work, we propose Step-Opt-Instruct, a framework that augments existing datasets and generates high-quality fine-tuning data tailored to optimization modeling. Step-Opt-Instruct employs iterative problem generation to systematically increase problem complexity and stepwise validation to rigorously verify data, preventing error propagation and ensuring the quality of the generated dataset. Leveraging this framework, we fine-tune open-source LLMs, including LLaMA-3-8B and Mistral-7B, to develop Step-Opt--a model that achieves state-of-the-art performance on benchmarks such as NL4OPT, MAMO, and IndustryOR. Extensive experiments demonstrate the superior performance of Step-Opt, especially in addressing complex OR tasks, with a notable 17.01\% improvement in micro average accuracy on difficult problems. These findings highlight the effectiveness of combining structured validation with gradual problem refinement to advance the automation of decision-making processes using LLMs.The code and dataset are available at https://github.com/samwu-learn/Step.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21819v2">Self-Preference Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 Accepted at NeurIPS 2024 Safe Generative AI Workshop
    </div>
    <details class="paper-abstract">
      Automated evaluation leveraging large language models (LLMs), commonly referred to as LLM evaluators or LLM-as-a-judge, has been widely used in measuring the performance of dialogue systems. However, the self-preference bias in LLMs has posed significant risks, including promoting specific styles or policies intrinsic to the LLMs. Despite the importance of this issue, there is a lack of established methods to measure the self-preference bias quantitatively, and its underlying causes are poorly understood. In this paper, we introduce a novel quantitative metric to measure the self-preference bias. Our experimental results demonstrate that GPT-4 exhibits a significant degree of self-preference bias. To explore the causes, we hypothesize that LLMs may favor outputs that are more familiar to them, as indicated by lower perplexity. We analyze the relationship between LLM evaluations and the perplexities of outputs. Our findings reveal that LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated. This suggests that the essence of the bias lies in perplexity and that the self-preference bias exists because LLMs prefer texts more familiar to them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17631v1">LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17630v1">Answer-Centric or Reasoning-Driven? Uncovering the Latent Memory Anchor in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, growing evidence suggests much of their success stems from memorized answer-reasoning patterns rather than genuine inference. In this work, we investigate a central question: are LLMs primarily anchored to final answers or to the textual pattern of reasoning chains? We propose a five-level answer-visibility prompt framework that systematically manipulates answer cues and probes model behavior through indirect, behavioral analysis. Experiments across state-of-the-art LLMs reveal a strong and consistent reliance on explicit answers. The performance drops by 26.90\% when answer cues are masked, even with complete reasoning chains. These findings suggest that much of the reasoning exhibited by LLMs may reflect post-hoc rationalization rather than true inference, calling into question their inferential depth. Our study uncovers the answer-anchoring phenomenon with rigorous empirical validation and underscores the need for a more nuanced understanding of what constitutes reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05651v2">LightVA: Lightweight Visual Analytics with LLM Agent-Based Task Planning and Execution</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Visual analytics (VA) requires analysts to iteratively propose analysis tasks based on observations and execute tasks by creating visualizations and interactive exploration to gain insights. This process demands skills in programming, data processing, and visualization tools, highlighting the need for a more intelligent, streamlined VA approach. Large language models (LLMs) have recently been developed as agents to handle various tasks with dynamic planning and tool-using capabilities, offering the potential to enhance the efficiency and versatility of VA. We propose LightVA, a lightweight VA framework that supports task decomposition, data analysis, and interactive exploration through human-agent collaboration. Our method is designed to help users progressively translate high-level analytical goals into low-level tasks, producing visualizations and deriving insights. Specifically, we introduce an LLM agent-based task planning and execution strategy, employing a recursive process involving a planner, executor, and controller. The planner is responsible for recommending and decomposing tasks, the executor handles task execution, including data analysis, visualization generation and multi-view composition, and the controller coordinates the interaction between the planner and executor. Building on the framework, we develop a system with a hybrid user interface that includes a task flow diagram for monitoring and managing the task planning process, a visualization panel for interactive data exploration, and a chat view for guiding the model through natural language instructions. We examine the effectiveness of our method through a usage scenario and an expert study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14003v2">Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Machine unlearning (MU) for large language models (LLMs), commonly referred to as LLM unlearning, seeks to remove specific undesirable data or knowledge from a trained model, while maintaining its performance on standard tasks. While unlearning plays a vital role in protecting data privacy, enforcing copyright, and mitigating sociotechnical harms in LLMs, we identify a new vulnerability post-unlearning: unlearning trace detection. We discover that unlearning leaves behind persistent ''fingerprints'' in LLMs, detectable traces in both model behavior and internal representations. These traces can be identified from output responses, even when prompted with forget-irrelevant inputs. Specifically, a simple supervised classifier can reliably determine whether a model has undergone unlearning based solely on its textual outputs. Further analysis shows that these traces are embedded in intermediate activations and propagate nonlinearly to the final layer, forming low-dimensional, learnable manifolds in activation space. Through extensive experiments, we show that forget-relevant prompts enable over 90% accuracy in detecting unlearning traces across all model sizes. Even with forget-irrelevant inputs, large LLMs maintain high detectability, demonstrating the broad applicability of unlearning trace detection. These findings reveal that unlearning leaves measurable signatures, introducing a new risk of reverse-engineering forgotten information when a model is identified as unlearned given an input query. Codes are available at https://github.com/OPTML-Group/Unlearn-Trace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15507v4">Steering LLMs for Formal Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in proving formal theorems using proof assistants like Lean. However, current state of the art language models struggles to predict next step in proofs leading practitioners to use different sampling techniques to improve LLMs capabilities. We observe that the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately within the set of candidate tactics, affecting the overall selection process. To overcome this hurdle, we use activation steering to guide LLMs responses to improve the generations at the time of inference. Our results suggest that activation steering offers a promising lightweight alternative to specialized fine-tuning for enhancing theorem proving capabilities in LLMs, particularly valuable in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01048v2">IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional performance across a wide range of natural language tasks. However, selecting the optimal LLM to respond to a user query often necessitates a delicate balance between performance and cost. While powerful models deliver better results, they come at a high cost, whereas smaller models are more cost-effective but less capable. To address this trade-off, we propose IRT-Router, a multi-LLM routing framework that efficiently routes user queries to the most suitable LLM. Inspired by Item Response Theory (IRT), a psychological measurement methodology, IRT-Router explicitly models the relationship between LLM capabilities and user query attributes. This not only enables accurate prediction of response performance but also provides interpretable insights, such as LLM abilities and query difficulty. Additionally, we design an online query warm-up technique based on semantic similarity, further enhancing the online generalization capability of IRT-Router. Extensive experiments on 20 LLMs and 12 datasets demonstrate that IRT-Router outperforms most baseline methods in terms of effectiveness and interpretability. Its superior performance in cold-start scenarios further confirms the reliability and practicality of IRT-Router in real-world applications. Code is available at https://github.com/Mercidaiha/IRT-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v2">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17562v1">LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17132v2">LLM-Aided Testbench Generation and Bug Detection for Finite-State Machines</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      This work investigates the potential of tailoring Large Language Models (LLMs), specifically GPT3.5 and GPT4, for the domain of chip testing. A key aspect of chip design is functional testing, which relies on testbenches to evaluate the functionality and coverage of Register-Transfer Level (RTL) designs. We aim to enhance testbench generation by incorporating feedback from commercial-grade Electronic Design Automation (EDA) tools into LLMs. Through iterative feedback from these tools, we refine the testbenches to achieve improved test coverage. Our case studies present promising results, demonstrating that this approach can effectively enhance test coverage. By integrating EDA tool feedback, the generated testbenches become more accurate in identifying potential issues in the RTL design. Furthermore, we extended our study to use this enhanced test coverage framework for detecting bugs in the RTL implementations
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17539v1">Breaking Single-Tester Limits: Multi-Agent LLMs for Multi-User Feature Testing</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 Accepted to International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      The growing dependence on mobile phones and their apps has made multi-user interactive features, like chat calls, live streaming, and video conferencing, indispensable for bridging the gaps in social connectivity caused by physical and situational barriers. However, automating these interactive features for testing is fraught with challenges, owing to their inherent need for timely, dynamic, and collaborative user interactions, which current automated testing methods inadequately address. Inspired by the concept of agents designed to autonomously and collaboratively tackle problems, we propose MAdroid, a novel multi-agent approach powered by the Large Language Models (LLMs) to automate the multi-user interactive task for app feature testing. Specifically, MAdroid employs two functional types of multi-agents: user agents (Operator) and supervisor agents (Coordinator and Observer). Each agent takes a specific role: the Coordinator directs the interactive task; the Operator mimics user interactions on the device; and the Observer monitors and reviews the task automation process. Our evaluation, which included 41 multi-user interactive tasks, demonstrates the effectiveness of our approach, achieving 82.9% of the tasks with 96.8% action similarity, outperforming the ablation studies and state-of-the-art baselines. Additionally, a preliminary investigation underscores MAdroid's practicality by helping identify 11 multi-user interactive bugs during regression app testing, confirming its potential value in real-world software development contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18931v1">Safe Pruning LoRA: Robust Distance-Guided Pruning for Safety Alignment in Adaptation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) enhances adaptability while reducing computational costs. However, fine-tuning can compromise safety alignment, even with benign data, increasing susceptibility to harmful outputs. Existing safety alignment methods struggle to capture complex parameter shifts, leading to suboptimal safety-utility trade-offs. To address this issue, we propose Safe Pruning LoRA (SPLoRA), a novel pruning-based approach that selectively removes LoRA layers that weaken safety alignment, improving safety while preserving performance. At its core, we introduce Empirical-DIEM (E-DIEM), a dimension-insensitive similarity metric that effectively detects safety misalignment in LoRA-adapted models. We conduct extensive experiments on LLMs fine-tuned with mixed of benign and malicious data, and purely benign datasets, evaluating SPLoRA across utility, safety, and reliability metrics. Results demonstrate that SPLoRA outperforms state-of-the-art safety alignment techniques, significantly reducing safety risks while maintaining or improving model performance and reliability. Additionally, SPLoRA reduces inference overhead, making it a scalable and efficient solution for deploying safer and more reliable LLMs. The code is available at https://github.com/AoShuang92/SPLoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18928v1">Do LLMs Know When to Flip a Coin? Strategic Randomization through Reasoning and Experience</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Strategic randomization is a key principle in game theory, yet it remains underexplored in large language models (LLMs). Prior work often conflates the cognitive decision to randomize with the mechanical generation of randomness, leading to incomplete evaluations. To address this, we propose a novel zero-sum game inspired by the Tian Ji Horse Race, where the Nash equilibrium corresponds to a maximal entropy strategy. The game's complexity masks this property from untrained humans and underdeveloped LLMs. We evaluate five LLMs across prompt styles -- framed, neutral, and hinted -- using competitive multi-tournament gameplay with system-provided random choices, isolating the decision to randomize. Results show that weaker models remain deterministic regardless of prompts, while stronger models exhibit increased randomization under explicit hints. When facing weaker models, strong LLMs adopt deterministic strategies to exploit biases, but converge toward equilibrium play when facing peers. Through win/loss outcomes and Bayes factor analysis, we demonstrate meaningful variation in LLMs' strategic reasoning capabilities, highlighting opportunities for improvement in abstract reasoning and adaptive learning. We make our implementation publicly available at https://github.com/ocelopus/llm-when-to-throw-coin to ensure full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16023v3">Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 Pre-print under-review
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large-scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human-AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users' intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17831v1">Engagement and Disclosures in LLM-Powered Cognitive Behavioral Therapy Exercises: A Factorial Design Comparing the Influence of a Robot vs. Chatbot Over Time</a></div>
    <div class="paper-meta">
      📅 2025-06-21
    </div>
    <details class="paper-abstract">
      Many researchers are working to address the worldwide mental health crisis by developing therapeutic technologies that increase the accessibility of care, including leveraging large language model (LLM) capabilities in chatbots and socially assistive robots (SARs) used for therapeutic applications. Yet, the effects of these technologies over time remain unexplored. In this study, we use a factorial design to assess the impact of embodiment and time spent engaging in therapeutic exercises on participant disclosures. We assessed transcripts gathered from a two-week study in which 26 university student participants completed daily interactive Cognitive Behavioral Therapy (CBT) exercises in their residences using either an LLM-powered SAR or a disembodied chatbot. We evaluated the levels of active engagement and high intimacy of their disclosures (opinions, judgments, and emotions) during each session and over time. Our findings show significant interactions between time and embodiment for both outcome measures: participant engagement and intimacy increased over time in the physical robot condition, while both measures decreased in the chatbot condition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17219v1">No Free Lunch: Rethinking Internal Feedback for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning has emerged as a powerful paradigm for post-training large language models (LLMs) to improve reasoning. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) have shown strong results, but they require extensive external supervision. We investigate an alternative class of methods, Reinforcement Learning from Internal Feedback (RLIF), which relies solely on intrinsic model-derived signals instead of external rewards. In particular, we leverage unsupervised reward proxies such as token-level entropy, trajectory-level entropy, and self-certainty. Our theoretical analysis shows these internal objectives are partially equivalent, and we empirically evaluate various RLIF strategies on challenging math reasoning benchmarks. Experimental results demonstrate that RLIF can boost the reasoning performance of base LLMs at the beginning phase of the training, matching or surpassing RLVR techniques on these tasks. However, when training progresses, performance degrades even below the model before training. Moreover, we find that RLIF yields little improvement for instruction-tuned models, indicating diminishing returns of intrinsic feedback once an LLM is already instruction-tuned. We further analyze this limitation by mixing model weights and explain the reason of RLIF's training behaviors, providing practical guidelines for integrating internal feedback signals into LLM training. We hope our analysis of internal feedback will inform more principled and effective strategies for LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09404v2">AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      This paper introduces AQA-Bench, a novel benchmark to assess the sequential reasoning capabilities of large language models (LLMs) in algorithmic contexts, such as depth-first search (DFS). The key feature of our evaluation benchmark lies in its interactive evaluation protocol - for example, in DFS, the availability of each node's connected edge is contingent upon the model's traversal to that node, thereby necessitating the LLM's ability to effectively remember visited nodes and strategize subsequent moves considering the possible environmental feedback in the future steps. We comprehensively build AQA-Bench with three different algorithms, namely binary search, depth-first search, and breadth-first search, and to evaluate the sequential reasoning ability of 14 different LLMs. Our investigations reveal several interesting findings: (1) Closed-source models like GPT-4 and Gemini generally show much stronger sequential reasoning ability, significantly outperforming open-source LLMs. (2) Naively providing in-context examples may inadvertently hurt few-shot performance in an interactive environment due to over-fitting to examples. (3) Instead of using optimal steps from another test case as the in-context example, a very limited number of predecessor steps in the current test case following the optimal policy can substantially boost small models' performance. (4) The performance gap between weak models and strong models is greatly due to the incapability of weak models to start well. (5) The scaling correlation between performance and model size is not always significant, sometimes even showcasing an inverse trend. We hope our study can catalyze future work on advancing the understanding and enhancement of LLMs' capabilities in sequential reasoning. The code is available at https://github.com/UCSC-VLAA/AQA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17203v1">Confidence Scoring for LLM-Generated SQL in Supply Chain Data Extraction</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 accepted by KDD workshop AI for Supply Chain 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently enabled natural language interfaces that translate user queries into executable SQL, offering a powerful solution for non-technical stakeholders to access structured data. However, one of the limitation that LLMs do not natively express uncertainty makes it difficult to assess the reliability of their generated queries. This paper presents a case study that evaluates multiple approaches to estimate confidence scores for LLM-generated SQL in supply chain data retrieval. We investigated three strategies: (1) translation-based consistency checks; (2) embedding-based semantic similarity between user questions and generated SQL; and (3) self-reported confidence scores directly produced by the LLM. Our findings reveal that LLMs are often overconfident in their own outputs, which limits the effectiveness of self-reported confidence. In contrast, embedding-based similarity methods demonstrate strong discriminative power in identifying inaccurate SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17196v1">Detecting LLM-Generated Short Answers and Effects on Learner Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted for publication at the 19th European Conference on Technology Enhanced Learning (ECTEL 2025). This is the author's accepted manuscript
    </div>
    <details class="paper-abstract">
      The increasing availability of large language models (LLMs) has raised concerns about their potential misuse in online learning. While tools for detecting LLM-generated text exist and are widely used by researchers and educators, their reliability varies. Few studies have compared the accuracy of detection methods, defined criteria to identify content generated by LLM, or evaluated the effect on learner performance from LLM misuse within learning. In this study, we define LLM-generated text within open responses as those produced by any LLM without paraphrasing or refinement, as evaluated by human coders. We then fine-tune GPT-4o to detect LLM-generated responses and assess the impact on learning from LLM misuse. We find that our fine-tuned LLM outperforms the existing AI detection tool GPTZero, achieving an accuracy of 80% and an F1 score of 0.78, compared to GPTZero's accuracy of 70% and macro F1 score of 0.50, demonstrating superior performance in detecting LLM-generated responses. We also find that learners suspected of LLM misuse in the open response question were more than twice as likely to correctly answer the corresponding posttest MCQ, suggesting potential misuse across both question types and indicating a bypass of the learning process. We pave the way for future work by demonstrating a structured, code-based approach to improve LLM-generated response detection and propose using auxiliary statistical indicators such as unusually high assessment scores on related tasks, readability scores, and response duration. In support of open science, we contribute data and code to support the fine-tuning of similar models for similar use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08801v2">LLMs and Stack Overflow Discussions: Reliability, Impact, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 63 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Since its release in November 2022, ChatGPT has shaken up Stack Overflow, the premier platform for developers queries on programming and software development. Demonstrating an ability to generate instant, human-like responses to technical questions, ChatGPT has ignited debates within the developer community about the evolving role of human-driven platforms in the age of generative AI. Two months after ChatGPT release, Meta released its answer with its own Large Language Model (LLM) called LLaMA: the race was on. We conducted an empirical study analyzing questions from Stack Overflow and using these LLMs to address them. This way, we aim to (i) quantify the reliability of LLMs answers and their potential to replace Stack Overflow in the long term; (ii) identify and understand why LLMs fail; (iii) measure users activity evolution with Stack Overflow over time; and (iv) compare LLMs together. Our empirical results are unequivocal: ChatGPT and LLaMA challenge human expertise, yet do not outperform it for some domains, while a significant decline in user posting activity has been observed. Furthermore, we also discuss the impact of our findings regarding the usage and development of new LLMs and provide guidelines for future challenges faced by users and researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17163v1">The MedPerturb Dataset: What Non-Content Perturbations Reveal About Human and Clinical LLM Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Clinical robustness is critical to the safe deployment of medical Large Language Models (LLMs), but key questions remain about how LLMs and humans may differ in response to the real-world variability typified by clinical settings. To address this, we introduce MedPerturb, a dataset designed to systematically evaluate medical LLMs under controlled perturbations of clinical input. MedPerturb consists of clinical vignettes spanning a range of pathologies, each transformed along three axes: (1) gender modifications (e.g., gender-swapping or gender-removal); (2) style variation (e.g., uncertain phrasing or colloquial tone); and (3) format changes (e.g., LLM-generated multi-turn conversations or summaries). With MedPerturb, we release a dataset of 800 clinical contexts grounded in realistic input variability, outputs from four LLMs, and three human expert reads per clinical context. We use MedPerturb in two case studies to reveal how shifts in gender identity cues, language style, or format reflect diverging treatment selections between humans and LLMs. We find that LLMs are more sensitive to gender and style perturbations while human annotators are more sensitive to LLM-generated format perturbations such as clinical summaries. Our results highlight the need for evaluation frameworks that go beyond static benchmarks to assess the similarity between human clinician and LLM decisions under the variability characteristic of clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18110v2">Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Humans naturally understand moments in a video by integrating visual and auditory cues. For example, localizing a scene in the video like "A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding" requires simultaneous processing of visual, audio, and speech signals. However, existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding. To address this, we present TriSense, a triple-modality large language model designed for holistic video temporal understanding through the integration of visual, audio, and speech modalities. Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs. To support TriSense's multimodal capabilities, we introduce TriSense-2M, a high-quality dataset of over 2 million curated samples generated via an automated pipeline powered by fine-tuned LLMs. TriSense-2M includes long-form videos and diverse modality combinations, facilitating broad generalization. Extensive experiments across multiple benchmarks demonstrate the effectiveness of TriSense and its potential to advance multimodal video analysis. Code and dataset will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19675v2">Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted at KDD'25
    </div>
    <details class="paper-abstract">
      The traditional process of creating labeled datasets is labor-intensive and expensive. Recent breakthroughs in open-source large language models (LLMs) have opened up a new avenue in generating labeled datasets automatically for various natural language processing (NLP) tasks, providing an alternative to such an expensive annotation process. However, the reliability of such auto-generated labels remains a significant concern due to inherent inaccuracies. When learning from noisy labels, the model's generalization is likely to be harmed as it is prone to overfit to those label noises. While previous studies in learning from noisy labels mainly focus on synthetic noise and real-world noise, LLM-generated label noise receives less attention. In this paper, we propose SiDyP: Simplex Label Diffusion with Dynamic Prior to calibrate the classifier's prediction, thus enhancing its robustness towards LLM-generated noisy labels. SiDyP retrieves potential true label candidates by neighborhood label distribution in text embedding space and iteratively refines noisy candidates using a simplex diffusion model. Our framework can increase the performance of the BERT classifier fine-tuned on both zero-shot and few-shot LLM-generated noisy label datasets by an average of 7.21% and 7.30% respectively. We demonstrate the effectiveness of SiDyP by conducting extensive benchmarking for different LLMs over a variety of NLP tasks. Our code is available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17104v1">Towards Advanced Mathematical Reasoning for LLMs via First-Order Logic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising first-order logic (FOL) reasoning capabilities with applications in various areas. However, their effectiveness in complex mathematical reasoning involving multi-step FOL deductions is still under-researched. While LLMs perform competitively on established mathematical reasoning benchmarks, they struggle with multi-step FOL tasks, as demonstrated by Deepseek-Prover-V2-7B's low accuracy (4.2%) on our proposed theorem proving dataset. This issue arises from the limited exploration of diverse proof strategies and the potential for early reasoning mistakes to undermine entire proofs. To address these issues, we propose DREAM, a self-adaptive solution that enhances the Diversity and REAsonability of LLMs' generation strategies. DREAM incorporates an Axiom-Driven Strategy Diversification mechanism to promote varied strategic outcomes and a Sub-Proposition Error Feedback to help LLMs reflect on and correct their proofs. Our contributions include pioneering advancements in LLMs' mathematical reasoning through FOL theorem proving, introducing a novel inference stage solution that improves performance by 0.6% to 6.4%, and providing a curated dataset of 447 mathematical theorems in Lean 4 format for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13784v2">ScholarSearch: Benchmarking Scholar Searching Ability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)' search capabilities have garnered significant attention. Existing benchmarks, such as OpenAI's BrowseComp, primarily focus on general search scenarios and fail to adequately address the specific demands of academic search. These demands include deeper literature tracing and organization, professional support for academic databases, the ability to navigate long-tail academic knowledge, and ensuring academic rigor. Here, we proposed ScholarSearch, the first dataset specifically designed to evaluate the complex information retrieval capabilities of Large Language Models (LLMs) in academic research. ScholarSearch possesses the following key characteristics: Academic Practicality, where question content closely mirrors real academic learning and research environments, avoiding deliberately misleading models; High Difficulty, with answers that are challenging for single models (e.g., Grok DeepSearch or Gemini Deep Research) to provide directly, often requiring at least three deep searches to derive; Concise Evaluation, where limiting conditions ensure answers are as unique as possible, accompanied by clear sources and brief solution explanations, greatly facilitating subsequent audit and verification, surpassing the current lack of analyzed search datasets both domestically and internationally; and Broad Coverage, as the dataset spans at least 15 different academic disciplines. Through ScholarSearch, we expect to more precisely measure and promote the performance improvement of LLMs in complex academic information retrieval tasks. The data is available at: https://huggingface.co/datasets/PKU-DS-LAB/ScholarSearch
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17080v1">Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Fine-tuning pretrained LLMs has been shown to be an effective strategy for reaching state-of-the-art performance on specific tasks like machine translation. However, this process of adaptation often implies sacrificing general-purpose capabilities, such as conversational reasoning and instruction-following, hampering the utility of the system in real-world applications that require a mixture of skills. In this paper, we introduce Tower+, a suite of models designed to deliver strong performance across both translation and multilingual general-purpose text capabilities. We achieve a Pareto frontier between translation specialization and multilingual general-purpose capabilities by introducing a novel training recipe that builds on Tower (Alves et al., 2024), comprising continued pretraining, supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. At each stage of training, we carefully generate and curate data to strengthen performance on translation as well as general-purpose tasks involving code generation, mathematics problem solving, and general instruction-following. We develop models at multiple scales: 2B, 9B, and 72B. Our smaller models often outperform larger general-purpose open-weight and proprietary LLMs (e.g., Llama 3.3 70B, GPT-4o). Our largest model delivers best-in-class translation performance for high-resource languages and top results in multilingual Arena Hard evaluations and in IF-MT, a benchmark we introduce for evaluating both translation and instruction-following. Our findings highlight that it is possible to rival frontier models in general capabilities, while optimizing for specific business domains, such as translation and localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17073v1">LLM-Based Bot Broadens the Range of Arguments in Online Discussions, Even When Transparently Disclosed as AI</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      A wide range of participation is essential for democracy, as it helps prevent the dominance of extreme views, erosion of legitimacy, and political polarization. However, engagement in online political discussions often features a limited spectrum of views due to high levels of self-selection and the tendency of online platforms to facilitate exchanges primarily among like-minded individuals. This study examines whether an LLM-based bot can widen the scope of perspectives expressed by participants in online discussions through two pre-registered randomized experiments conducted in a chatroom. We evaluate the impact of a bot that actively monitors discussions, identifies missing arguments, and introduces them into the conversation. The results indicate that our bot significantly expands the range of arguments, as measured by both objective and subjective metrics. Furthermore, disclosure of the bot as AI does not significantly alter these effects. These findings suggest that LLM-based moderation tools can positively influence online political discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17067v1">Empowering Near-Field Communications in Low-Altitude Economy with LLM: Fundamentals, Potentials, Solutions, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      The low-altitude economy (LAE) is gaining significant attention from academia and industry. Fortunately, LAE naturally aligns with near-field communications in extremely large-scale MIMO (XL-MIMO) systems. By leveraging near-field beamfocusing, LAE can precisely direct beam energy to unmanned aerial vehicles, while the additional distance dimension boosts overall spectrum efficiency. However, near-field communications in LAE still face several challenges, such as the increase in signal processing complexity and the necessity of distinguishing between far and near-field users. Inspired by the large language models (LLM) with powerful ability to handle complex problems, we apply LLM to solve challenges of near-field communications in LAE. The objective of this article is to provide a comprehensive analysis and discussion on LLM-empowered near-field communications in LAE. Specifically, we first introduce fundamentals of LLM and near-field communications, including the key advantages of LLM and key characteristics of near-field communications. Then, we reveal the opportunities and challenges of near-field communications in LAE. To address these challenges, we present a LLM-based scheme for near-field communications in LAE, and provide a case study which jointly distinguishes far and near-field users and designs multi-user precoding matrix. Finally, we outline and highlight several future research directions and open issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06751v2">Geopolitical biases in LLMs: what are the "good" and the "bad" countries according to contemporary language models</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      This paper evaluates geopolitical biases in LLMs with respect to various countries though an analysis of their interpretation of historical events with conflicting national perspectives (USA, UK, USSR, and China). We introduce a novel dataset with neutral event descriptions and contrasting viewpoints from different countries. Our findings show significant geopolitical biases, with models favoring specific national narratives. Additionally, simple debiasing prompts had a limited effect in reducing these biases. Experiments with manipulated participant labels reveal models' sensitivity to attribution, sometimes amplifying biases or recognizing inconsistencies, especially with swapped labels. This work highlights national narrative biases in LLMs, challenges the effectiveness of simple debiasing methods, and offers a framework and dataset for future geopolitical bias research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16813v3">Incivility and Rigidity: The Risks of Fine-Tuning LLMs for Political Argumentation</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      The incivility prevalent on platforms like Twitter (now X) and Reddit poses a challenge for developing AI systems that can support productive and rhetorically sound political argumentation. In this study, we report experiments with GPT-3.5 Turbo, fine-tuned on two contrasting datasets of political discussions: high-variance, high-incivility Twitter replies to U.S. Congress, and low-variance, low-incivility posts from Reddit's r/ChangeMyView. We systematically evaluate how these data sources and prompting strategies shape the rhetorical framing and deliberative quality of model-generated arguments. Our results show that Reddit-finetuned models produce safer but rhetorically rigid arguments, while cross-platform fine-tuning amplifies toxicity. Prompting reduces specific toxic behaviors, such as personal attacks, but fails to fully mitigate the influence of high-incivility training data. We introduce and validate a rhetorical evaluation rubric and provide practical guidelines for deploying LLMs in content authoring, moderation, and deliberation support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17006v1">LLM-Generated Feedback Supports Learning If Learners Choose to Use It</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Full research paper accepted at EC-TEL '25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate feedback, yet their impact on learning remains underexplored, especially compared to existing feedback methods. This study investigates how on-demand LLM-generated explanatory feedback influences learning in seven scenario-based tutor training lessons. Analyzing over 2,600 lesson completions from 885 tutor learners, we compare posttest performance among learners across three groups: learners who received feedback generated by gpt-3.5-turbo, those who declined it, and those without access. All groups received non-LLM corrective feedback. To address potential selection bias-where higher-performing learners may be more inclined to use LLM feedback-we applied propensity scoring. Learners with a higher predicted likelihood of engaging with LLM feedback scored significantly higher at posttest than those with lower propensity. After adjusting for this effect, two out of seven lessons showed statistically significant learning benefits from LLM feedback with standardized effect sizes of 0.28 and 0.33. These moderate effects suggest that the effectiveness of LLM feedback depends on the learners' tendency to seek support. Importantly, LLM feedback did not significantly increase completion time, and learners overwhelmingly rated it as helpful. These findings highlight LLM feedback's potential as a low-cost and scalable way to improve learning on open-ended tasks, particularly in existing systems already providing feedback without LLMs. This work contributes open datasets, LLM prompts, and rubrics to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16990v1">TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted to the SDProc Workshop @ ACL 2025
    </div>
    <details class="paper-abstract">
      LaTeX's precision and flexibility in typesetting have made it the gold standard for the preparation of scientific documentation. Large Language Models (LLMs) present a promising opportunity for researchers to produce publication-ready material using LaTeX with natural language instructions, yet current benchmarks completely lack evaluation of this ability. By introducing TeXpert, our benchmark dataset with natural language prompts for generating LaTeX code focused on components of scientific documents across multiple difficulty levels, we conduct an in-depth analysis of LLM performance in this regard and identify frequent error types. Our evaluation across open and closed-source LLMs highlights multiple key findings: LLMs excelling on standard benchmarks perform poorly in LaTeX generation with a significant accuracy drop-off as the complexity of tasks increases; open-source models like DeepSeek v3 and DeepSeek Coder strongly rival closed-source counterparts in LaTeX tasks; and formatting and package errors are unexpectedly prevalent, suggesting a lack of diverse LaTeX examples in the training datasets of most LLMs. Our dataset, code, and model evaluations are available at https://github.com/knowledge-verse-ai/TeXpert.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12911v2">Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Generating SQLs from user queries is a long-standing challenge, where the accuracy of initial schema linking significantly impacts subsequent SQL generation performance. However, current schema linking models still struggle with missing relevant schema elements or an excess of redundant ones. A crucial reason for this is that commonly used metrics, recall and precision, fail to capture relevant element missing and thus cannot reflect actual schema linking performance. Motivated by this, we propose enhanced schema linking metrics by introducing a restricted missing indicator. Accordingly, we introduce Knapsack optimization-based Schema Linking Approach (KaSLA), a plug-in schema linking method designed to prevent the missing of relevant schema elements while minimizing the inclusion of redundant ones. KaSLA employs a hierarchical linking strategy that first identifies the optimal table linking and subsequently links columns within the selected table to reduce linking candidate space. In each linking process, it utilizes a knapsack optimization approach to link potentially relevant elements while accounting for a limited tolerance of potentially redundant ones. With this optimization, KaSLA-1.6B achieves superior schema linking results compared to large-scale LLMs, including deepseek-v3 with the state-of-the-art (SOTA) schema linking method. Extensive experiments on Spider and BIRD benchmarks verify that KaSLA can significantly improve the SQL generation performance of SOTA Text2SQL models by substituting their schema linking processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14352v3">LogProber: Disentangling confidence from contamination in LLM responses</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      In machine learning, contamination refers to situations where testing data leak into the training set. The issue is particularly relevant for the evaluation of the performance of Large Language Models (LLMs), which are generally trained on gargantuan, and generally opaque, corpora of text scraped from the world wide web. Developing tools to detect contamination is therefore crucial to be able to fairly and properly track the evolution of the performance of LLMs. To date, only a few recent studies have attempted to address the issue of quantifying and detecting contamination in short text sequences, such as those commonly found in benchmarks. However, these methods have limitations that can sometimes render them impractical. In the present paper, we introduce LogProber, a novel, efficient algorithm that we show to be able to detect contamination in a black box setting that tries to tackle some of these drawbacks by focusing on the familiarity with the question rather than the answer. Here, we explore the properties of the proposed method in comparison with concurrent approaches, identify its advantages and limitations, and illustrate how different forms of contamination can go undetected depending on the design of the detection algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05692v3">SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce SafeGenBench, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on SafeGenBench, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v2">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      We develop a framework to quantify the time-to-unsafe-sampling - the number of large language model (LLM) generations required to trigger an unsafe (e.g., toxic) response. Estimating this quantity is challenging, since unsafe responses are exceedingly rare in well-aligned LLMs, potentially occurring only once in thousands of generations. As a result, directly estimating time-to-unsafe-sampling would require collecting training data with a prohibitively large number of generations per prompt. However, with realistic sampling budgets, we often cannot generate enough responses to observe an unsafe outcome for every prompt, leaving the time-to-unsafe-sampling unobserved in many cases, making the estimation and evaluation tasks particularly challenging. To address this, we frame this estimation problem as one of survival analysis and develop a provably calibrated lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt, leveraging recent advances in conformal prediction. Our key innovation is designing an adaptive, per-prompt sampling strategy, formulated as a convex optimization problem. The objective function guiding this optimized sampling allocation is designed to reduce the variance of the estimators used to construct the LPB, leading to improved statistical efficiency over naive methods that use a fixed sampling budget per prompt. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04227v2">Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Penetration-testing, while critical for validating defenses and uncovering vulnerabilities, is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Microsoft Active Directory (AD) enterprise networks. Our novel prototype, cochise, represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft AD testbed (GOAD). The evaluation deliberately utilizes GOAD to capture the intricate interactions and sometimes nondeterministic outcomes of live network pen-testing, moving beyond the limitations of synthetic benchmarks. We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through comprehensive quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks, and generate scenario-specific attack parameters. Cochise also exhibits robust self-correction mechanisms, automatically installing missing tools and rectifying invalid command generations. Critically, we find that the associated costs are competitive with those incurred by professional pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v4">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 37 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving scientific problems but often suffer from the issue of hallucination. While integrating LLMs with tools can mitigate this issue, models fine-tuned on tool usage become overreliant on them and incur unnecessary costs. Inspired by how human experts assess problem complexity before selecting solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tool-generated solutions. In the second component, Tool Usage Adaptation (TUA), we categorize problems as easy or hard based on the model's accuracy, and train it to maintain direct reasoning for easy problems while switching to tools for hard ones. We validate our method on six scientific benchmark datasets across climate science, epidemiology, physics, and other domains. Compared to the original instruct model (8B), models post-trained with AWL achieve 29.11% higher answer accuracy and 12.72% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4o and Claude-3.5 on four custom-created datasets. Our code is open-source at https://github.com/Rose-STL-Lab/Adapting-While-Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16813v1">Integrating Traditional Technical Analysis with AI: A Multi-Agent LLM-Based Approach to Stock Market Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 12 pages, 8 figures, 1 table. This is the accepted version of the paper presented at the 17th International Conference on Agents and Artificial Intelligence (ICAART 2025), Porto, Portugal
    </div>
    <details class="paper-abstract">
      Traditional technical analysis methods face limitations in accurately predicting trends in today's complex financial markets. This paper introduces ElliottAgents, an multi-agent system that integrates the Elliott Wave Principle with AI for stock market forecasting. The inherent complexity of financial markets, characterized by non-linear dynamics, noise, and susceptibility to unpredictable external factors, poses significant challenges for accurate prediction. To address these challenges, the system employs LLMs to enhance natural language understanding and decision-making capabilities within a multi-agent framework. By leveraging technologies such as Retrieval-Augmented Generation (RAG) and Deep Reinforcement Learning (DRL), ElliottAgents performs continuous, multi-faceted analysis of market data to identify wave patterns and predict future price movements. The research explores the system's ability to process historical stock data, recognize Elliott wave patterns, and generate actionable insights for traders. Experimental results, conducted on historical data from major U.S. companies, validate the system's effectiveness in pattern recognition and trend forecasting across various time frames. This paper contributes to the field of AI-driven financial analysis by demonstrating how traditional technical analysis methods can be effectively combined with modern AI approaches to create more reliable and interpretable market prediction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15551v2">Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting Attack</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted to ACL Findings, 2025
    </div>
    <details class="paper-abstract">
      Batch prompting, which combines a batch of multiple queries sharing the same context in one inference, has emerged as a promising solution to reduce inference costs. However, our study reveals a significant security vulnerability in batch prompting: malicious users can inject attack instructions into a batch, leading to unwanted interference across all queries, which can result in the inclusion of harmful content, such as phishing links, or the disruption of logical reasoning. In this paper, we construct BATCHSAFEBENCH, a comprehensive benchmark comprising 150 attack instructions of two types and 8k batch instances, to study the batch prompting vulnerability systematically. Our evaluation of both closed-source and open-weight LLMs demonstrates that all LLMs are susceptible to batch-prompting attacks. We then explore multiple defending approaches. While the prompting-based defense shows limited effectiveness for smaller LLMs, the probing-based approach achieves about 95% accuracy in detecting attacks. Additionally, we perform a mechanistic analysis to understand the attack and identify attention heads that are responsible for it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v4">Ask, Fail, Repeat: Meeseeks, an Iterative Feedback Benchmark for LLMs' Multi-turn Instruction-Following Ability</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. For complex instructions, LLMs often struggle to fulfill all requirements in a single attempt. In practice, users typically provide iterative feedback until the LLM generates a response that meets all requirements. However, existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction. To address this gap, we propose Meeseeks. Meeseeks simulates realistic human-LLM interactions through an iterative feedback framework, which enables models to self-correct based on specific requirement failures in each turn, better reflecting real-world user-end usage patterns. Meanwhile, the benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in multi-turn scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16777v1">DistillNote: LLM-based clinical note summaries improve heart failure diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer unprecedented opportunities to generate concise summaries of patient information and alleviate the burden of clinical documentation that overwhelms healthcare providers. We present Distillnote, a framework for LLM-based clinical note summarization, and generate over 64,000 admission note summaries through three techniques: (1) One-step, direct summarization, and a divide-and-conquer approach involving (2) Structured summarization focused on independent clinical insights, and (3) Distilled summarization that further condenses the Structured summaries. We test how useful are the summaries by using them to predict heart failure compared to a model trained on the original notes. Distilled summaries achieve 79% text compression and up to 18.2% improvement in AUPRC compared to an LLM trained on the full notes. We also evaluate the quality of the generated summaries in an LLM-as-judge evaluation as well as through blinded pairwise comparisons with clinicians. Evaluations indicate that one-step summaries are favoured by clinicians according to relevance and clinical actionability, while distilled summaries offer optimal efficiency (avg. 6.9x compression-to-performance ratio) and significantly reduce hallucinations. We release our summaries on PhysioNet to encourage future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16029v3">FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16697v1">From Prompts to Constructs: A Dual-Validity Framework for LLM Research in Psychology</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly being adopted across psychology, serving as research tools, experimental subjects, human simulators, and computational models of cognition. However, the application of human measurement tools to these systems can produce contradictory results, raising concerns that many findings are measurement phantoms--statistical artifacts rather than genuine psychological phenomena. In this Perspective, we argue that building a robust science of AI psychology requires integrating two of our field's foundational pillars: the principles of reliable measurement and the standards for sound causal inference. We present a dual-validity framework to guide this integration, which clarifies how the evidence needed to support a claim scales with its scientific ambition. Using an LLM to classify text may require only basic accuracy checks, whereas claiming it can simulate anxiety demands a far more rigorous validation process. Current practice systematically fails to meet these requirements, often treating statistical pattern matching as evidence of psychological phenomena. The same model output--endorsing "I am anxious"--requires different validation strategies depending on whether researchers claim to measure, characterize, simulate, or model psychological constructs. Moving forward requires developing computational analogues of psychological constructs and establishing clear, scalable standards of evidence rather than the uncritical application of human measurement tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10486v2">LLMs in Disease Diagnosis: A Comparative Study of DeepSeek-R1 and O3 Mini Across Chronic Health Conditions</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing medical diagnostics by enhancing both disease classification and clinical decision-making. In this study, we evaluate the performance of two LLM- based diagnostic tools, DeepSeek R1 and O3 Mini, using a structured dataset of symptoms and diagnoses. We assessed their predictive accuracy at both the disease and category levels, as well as the reliability of their confidence scores. DeepSeek R1 achieved a disease-level accuracy of 76% and an overall accuracy of 82%, outperforming O3 Mini, which attained 72% and 75% respectively. Notably, DeepSeek R1 demonstrated exceptional performance in Mental Health, Neurological Disorders, and Oncology, where it reached 100% accuracy, while O3 Mini excelled in Autoimmune Disease classification with 100% accuracy. Both models, however, struggled with Respiratory Disease classification, recording accuracies of only 40% for DeepSeek R1 and 20% for O3 Mini. Additionally, the analysis of confidence scores revealed that DeepSeek R1 provided high-confidence predictions in 92% of cases, compared to 68% for O3 Mini. Ethical considerations regarding bias, model interpretability, and data privacy are also discussed to ensure the responsible integration of LLMs into clinical practice. Overall, our findings offer valuable insights into the strengths and limitations of LLM-based diagnostic systems and provide a roadmap for future enhancements in AI-driven healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12307v2">Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. Our code is available here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02773v2">SD++: Enhancing Standard Definition Maps by Incorporating Road Knowledge using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 7 pages, 8 figures, 1 table, Accepted at IEEE Intelligent Vehicles Symposium 2025
    </div>
    <details class="paper-abstract">
      High-definition maps (HD maps) are detailed and informative maps capturing lane centerlines and road elements. Although very useful for autonomous driving, HD maps are costly to build and maintain. Furthermore, access to these high-quality maps is usually limited to the firms that build them. On the other hand, standard definition (SD) maps provide road centerlines with an accuracy of a few meters. In this paper, we explore the possibility of enhancing SD maps by incorporating information from road manuals using LLMs. We develop SD++, an end-to-end pipeline to enhance SD maps with location-dependent road information obtained from a road manual. We suggest and compare several ways of using LLMs for such a task. Furthermore, we show the generalization ability of SD++ by showing results from both California and Japan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16659v1">A Minimalist Optimizer Design for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which require significant memory to maintain first- and second-moment matrices, known as optimizer states. While recent works such as GaLore, Fira, and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: What is the minimal amount of optimizer state that is truly necessary to retain state-of-the-art performance in LLM pretraining? In this work, we systematically investigate this question using a bottom-up approach. We find that two memory- and compute-efficient optimization techniques are particularly effective: (1) column-wise gradient normalization significantly boosts the performance of plain SGD without requiring momentum; and (2) adding first-order momentum only to the output layer - where gradient variance is highest - yields performance competitive with fully adaptive methods such as Muon. Based on these insights, we propose SCALE (Stochastic Column-normalized Last-layer Momentum), a new optimizer that combines column-normalized SGD with last-layer momentum, where column normalization refers to normalizing the gradient along the output dimension. Across multiple LLaMA models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira, and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For the LLaMA 7B model, SCALE outperforms the state-of-the-art method APOLLO in terms of both perplexity and memory consumption. In addition, our method serves as a minimalist baseline for more sophisticated optimizer design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17506v1">VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Modern GPUs evolve rapidly, yet production compilers still rely on hand-crafted register allocation heuristics that require substantial re-tuning for each hardware generation. We introduce VeriLocc, a framework that combines large language models (LLMs) with formal compiler techniques to enable generalizable and verifiable register allocation across GPU architectures. VeriLocc fine-tunes an LLM to translate intermediate representations (MIRs) into target-specific register assignments, aided by static analysis for cross-architecture normalization and generalization and a verifier-guided regeneration loop to ensure correctness. Evaluated on matrix multiplication (GEMM) and multi-head attention (MHA), VeriLocc achieves 85-99% single-shot accuracy and near-100% pass@100. Case study shows that VeriLocc discovers more performant assignments than expert-tuned libraries, outperforming rocBLAS by over 10% in runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v4">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains such as time series forecasting. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black box behind the LLM and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numerical downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, a log-linear model for LLMs is considered in which numerical data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). For this model, it is demonstrated that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, it is shown how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numerical data and model architectures have different impacts on isotropy, and this variability directly affects the performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v2">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 6 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Regardless of the advancements in device capabilities, efficient inferencing advanced large language models (LLMs) at the edge remains challenging due to limited device memory and power constraints. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new approach that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a method that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server efficiently batches and verifies the tokens utilizing a more precise target model. This approach supports device heterogeneity and reduces server-side memory footprint by avoiding the need to deploy multiple target models. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: significantly increased system throughput, capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17449v1">OmniReflect: Discovering Transferable Constitutions for LLM agents via Neuro-Symbolic Reflections</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Efforts to improve Large Language Model (LLM) agent performance on complex tasks have largely focused on fine-tuning and iterative self-correction. However, these approaches often lack generalizable mechanisms for longterm learning and remain inefficient in dynamic environments. We introduce OmniReflect, a hierarchical, reflection-driven framework that constructs a constitution, a compact set of guiding principles distilled from task experiences, to enhance the effectiveness and efficiency of an LLM agent. OmniReflect operates in two modes: Self-sustaining, where a single agent periodically curates its own reflections during task execution, and Co-operative, where a Meta-advisor derives a constitution from a small calibration set to guide another agent. To construct these constitutional principles, we employ Neural, Symbolic, and NeuroSymbolic techniques, offering a balance between contextual adaptability and computational efficiency. Empirical results averaged across models show major improvements in task success, with absolute gains of +10.3% on ALFWorld, +23.8% on BabyAI, and +8.3% on PDDL in the Self-sustaining mode. Similar gains are seen in the Co-operative mode, where a lightweight Qwen3-4B ReAct agent outperforms all Reflexion baselines on BabyAI. These findings highlight the robustness and effectiveness of OmniReflect across environments and backbones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17435v1">Beyond the Link: Assessing LLMs' ability to Classify Political Content across Global Media</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) is becoming common in the context of political science, particularly in studies that analyse individuals use of digital media. However, while previous research has demonstrated LLMs ability at labelling tasks, the effectiveness of using LLMs to classify political content (PC) from just URLs is not yet well explored. The work presented in this article bridges this gap by evaluating whether LLMs can accurately identify PC vs. non-PC from both the article text and the URLs from five countries (France, Germany, Spain, the UK, and the US) and different languages. Using cutting-edge LLMs like GPT, Llama, Mistral, Deepseek, Qwen and Gemma, we measure model performance to assess whether URL-level analysis can be a good approximation for full-text analysis of PC, even across different linguistic and national contexts. Model outputs are compared with human-labelled articles, as well as traditional supervised machine learning techniques, to set a baseline of performance. Overall, our findings suggest the capacity of URLs to embed most of the news content, providing a vital perspective on accuracy-cost balancing. We also account for contextual limitations and suggest methodological recommendations to use LLMs within political science studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17419v1">UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 19 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at https://github.com/jinhaoduan/UProp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17410v1">Leveraging LLMs to Assess Tutor Moves in Real-Life Dialogues: A Feasibility Study</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Short research paper accepted at EC-TEL 2025
    </div>
    <details class="paper-abstract">
      Tutoring improves student achievement, but identifying and studying what tutoring actions are most associated with student learning at scale based on audio transcriptions is an open research problem. This present study investigates the feasibility and scalability of using generative AI to identify and evaluate specific tutor moves in real-life math tutoring. We analyze 50 randomly selected transcripts of college-student remote tutors assisting middle school students in mathematics. Using GPT-4, GPT-4o, GPT-4-turbo, Gemini-1.5-pro, and LearnLM, we assess tutors' application of two tutor skills: delivering effective praise and responding to student math errors. All models reliably detected relevant situations, for example, tutors providing praise to students (94-98% accuracy) and a student making a math error (82-88% accuracy) and effectively evaluated the tutors' adherence to tutoring best practices, aligning closely with human judgments (83-89% and 73-77%, respectively). We propose a cost-effective prompting strategy and discuss practical implications for using large language models to support scalable assessment in authentic settings. This work further contributes LLM prompts to support reproducibility and research in AI-supported learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17369v1">Re-Evaluating Code LLM Benchmarks Under Semantic Mutation</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      In the era of large language models (LLMs), code benchmarks have become an important research area in software engineering and are widely used by practitioners. These benchmarks evaluate the performance of LLMs on specific code-related tasks, such as code understanding and generation. A critical step in constructing code benchmarks is the design of prompts. However, as existing code benchmarks typically rely on a single prompt template per task, they are prone to the issue of prompt sensitivity, where minor prompt variations could result in substantial performance variations, leading to unreliable evaluations of model capabilities. While previous studies have explored prompt sensitivity, their experimental designs and findings are limited to traditional natural language processing (NLP) tasks. In this paper, we present an empirical study to investigate prompt sensitivity in code benchmarks. We first propose a general framework that modifies prompt templates in a manner that preserves both their semantics and their structure as much as possible. Based on the framework, we conduct extensive experiments across eight code benchmark tasks on 10 representative open-source LLMs, with each task featuring 100 semantically similar prompt templates. We then analyze the evaluation results using various statistical metrics, focusing on both absolute and relative model performance. Our findings suggest that even slight prompt variations can lead to significant shifts in performance. Additionally, we observe that such variations can introduce inconsistencies in the performance rankings across different models. These insights highlight the need for considering prompt sensitivity when designing future code benchmarks, to ensure more reliable and accurate evaluation of LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17367v1">Cash or Comfort? How LLMs Value Your Inconvenience</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 12 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17363v1">A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 ACL 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Virtual Teaching Assistants (VTAs) powered by Large Language Models (LLMs) have the potential to enhance student learning by providing instant feedback and facilitating multi-turn interactions. However, empirical studies on their effectiveness and acceptance in real-world classrooms are limited, leaving their practical impact uncertain. In this study, we develop an LLM-based VTA and deploy it in an introductory AI programming course with 477 graduate students. To assess how student perceptions of the VTA's performance evolve over time, we conduct three rounds of comprehensive surveys at different stages of the course. Additionally, we analyze 3,869 student--VTA interaction pairs to identify common question types and engagement patterns. We then compare these interactions with traditional student--human instructor interactions to evaluate the VTA's role in the learning process. Through a large-scale empirical study and interaction analysis, we assess the feasibility of deploying VTAs in real-world classrooms and identify key challenges for broader adoption. Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education: \texttt{https://github.com/sean0042/VTA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17353v1">Differentiation-Based Extraction of Proprietary Data from Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 In Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS'25), October 13-17, 2025, Taipei, Taiwan, China. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3719027.3744856
    </div>
    <details class="paper-abstract">
      The increasing demand for domain-specific and human-aligned Large Language Models (LLMs) has led to the widespread adoption of Supervised Fine-Tuning (SFT) techniques. SFT datasets often comprise valuable instruction-response pairs, making them highly valuable targets for potential extraction. This paper studies this critical research problem for the first time. We start by formally defining and formulating the problem, then explore various attack goals, types, and variants based on the unique properties of SFT data in real-world scenarios. Based on our analysis of extraction behaviors of direct extraction, we develop a novel extraction method specifically designed for SFT models, called Differentiated Data Extraction (DDE), which exploits the confidence levels of fine-tuned models and their behavioral differences from pre-trained base models. Through extensive experiments across multiple domains and scenarios, we demonstrate the feasibility of SFT data extraction using DDE. Our results show that DDE consistently outperforms existing extraction baselines in all attack settings. To counter this new attack, we propose a defense mechanism that mitigates DDE attacks with minimal impact on model performance. Overall, our research reveals hidden data leak risks in fine-tuned LLMs and provides insights for developing more secure models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16655v1">Arch-Router: Aligning LLM Routing with Human Preferences</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      With the rapid proliferation of large language models (LLMs) -- each optimized for different strengths, style, or latency/cost profile -- routing has become an essential technique to operationalize the use of different models. However, existing LLM routing approaches are limited in two key ways: they evaluate performance using benchmarks that often fail to capture human preferences driven by subjective evaluation criteria, and they typically select from a limited pool of models. In this work, we propose a preference-aligned routing framework that guides model selection by matching queries to user-defined domains (e.g., travel) or action types (e.g., image editing) -- offering a practical mechanism to encode preferences in routing decisions. Specifically, we introduce \textbf{Arch-Router}, a compact 1.5B model that learns to map queries to domain-action preferences for model routing decisions. Our approach also supports seamlessly adding new models for routing without requiring retraining or architectural modifications. Experiments on conversational datasets demonstrate that our approach achieves state-of-the-art (SOTA) results in matching queries with human preferences, outperforming top proprietary models. Our approach captures subjective evaluation criteria and makes routing decisions more transparent and flexible. Our model is available at: \texttt{https://huggingface.co/katanemo/Arch-Router-1.5B}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16644v1">Semantic Outlier Removal with Embedding Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Accepted to the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025) Industry Track, 10 pages
    </div>
    <details class="paper-abstract">
      Modern text processing pipelines demand robust methods to remove extraneous content while preserving a document's core message. Traditional approaches such as HTML boilerplate extraction or keyword filters often fail in multilingual settings and struggle with context-sensitive nuances, whereas Large Language Models (LLMs) offer improved quality at high computational cost. We introduce SORE (Semantic Outlier Removal), a cost-effective, transparent method that leverages multilingual sentence embeddings and approximate nearest-neighbor search to identify and excise unwanted text segments. By first identifying core content via metadata embedding and then flagging segments that either closely match predefined outlier groups or deviate significantly from the core, SORE achieves near-LLM extraction precision at a fraction of the cost. Experiments on HTML datasets demonstrate that SORE outperforms structural methods and yield high precision in diverse scenarios. Our system is currently deployed in production, processing millions of documents daily across multiple languages while maintaining both efficiency and accuracy. To facilitate reproducibility and further research, we release our implementation and evaluation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13284v3">Learning to Route LLMs with Confidence Tokens</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on several tasks and are increasingly deployed in real-world applications. However, especially in high-stakes settings, it becomes vital to know when the output of an LLM may be unreliable. Depending on whether an answer is trustworthy, a system can then choose to route the question to another expert, or otherwise fall back on a safe default behavior. In this work, we study the extent to which LLMs can reliably indicate confidence in their answers, and how this notion of confidence can translate into downstream accuracy gains. We propose Self-Reflection with Error-based Feedback (Self-REF), a lightweight training strategy to teach LLMs to express confidence in whether their answers are correct in a reliable manner. Self-REF introduces confidence tokens into the LLM, from which a confidence score can be extracted. Compared to conventional approaches such as verbalizing confidence and examining token probabilities, we demonstrate empirically that confidence tokens show significant improvements in downstream routing and rejection learning tasks.
    </details>
</div>
