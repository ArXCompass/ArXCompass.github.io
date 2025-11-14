# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14397v1">Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This paper presents a limit study of transformer-based large language model (LLM) inference, focusing on the fundamental performance bottlenecks imposed by memory bandwidth, memory capacity, and synchronization overhead in distributed inference systems. We develop a hardware-agnostic performance model that abstracts away implementation details, enabling the analysis of a wide range of current and near-future hardware technologies. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. It also covers SRAM-based designs and scaling techniques from distributed clusters with varying numbers of chips to wafer-scale integration. Our key findings for auto-regressive decoding are: i) serving LLMs requires 100s of GB per server to serve a model instance; ii) high memory bandwidth is critical for high per-user throughput; iii) exposed synchronization latencies to achieve collective communication must be around 1us else they make the memory bandwidth ineffective; iv) DRAM-based designs have a fundamental advantage in terms of system-level efficiency as measured in throughput per cost or watt; and v) hardware designs can easily reach 2000+ user token/sec but getting to 10,000+ tokens/sec will need smaller models, smaller context, or other forms of algorithmic advances. This study provides valuable insights into the fundamental performance limits of LLM inference, highlighting the potential benefits of future hardware advancements and guiding the optimization of LLM deployment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10968v2">Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14376v1">Schemora: schema matching via multi-stage recommendation and metadata enrichment using off-the-shelf llms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Schema matching is essential for integrating heterogeneous data sources and enhancing dataset discovery, yet it remains a complex and resource-intensive problem. We introduce SCHEMORA, a schema matching framework that combines large language models with hybrid retrieval techniques in a prompt-based approach, enabling efficient identification of candidate matches without relying on labeled training data or exhaustive pairwise comparisons. By enriching schema metadata and leveraging both vector-based and lexical retrieval, SCHEMORA improves matching accuracy and scalability. Evaluated on the MIMIC-OMOP benchmark, it establishes new state-of-the-art performance, with gains of 7.49% in HitRate@5 and 3.75% in HitRate@3 over previous best results. To our knowledge, this is the first LLM-based schema matching method with an open-source implementation, accompanied by analysis that underscores the critical role of retrieval and provides practical guidance on model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10871v2">Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This study explores how large language models (LLMs) encode interwoven scientific knowledge, using chemical elements and LLaMA-series models as a case study. We identify a 3D spiral structure in the hidden states that aligns with the conceptual structure of the periodic table, suggesting that LLMs can reflect the geometric organization of scientific concepts learned from text. Linear probing reveals that middle layers encode continuous, overlapping attributes that enable indirect recall, while deeper layers sharpen categorical distinctions and incorporate linguistic context. These findings suggest that LLMs represent symbolic knowledge not as isolated facts, but as structured geometric manifolds that intertwine semantic information across layers. We hope this work inspires further exploration of how LLMs represent and reason about scientific knowledge, particularly in domains such as materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v2">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to study representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., ''cat'') and then analyze the corresponding activation patterns. We find that LLM representations captured this way closely align with human representations inferred from behavioral data, matching inter-human alignment levels. Our approach significantly outperforms the alignment captured by word embeddings, which have been the focus of prior work on human-LLM alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts -- we show that LLMs organize concepts in a way that mirrors human concept organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14355v1">Can LLMs Infer Personality from Real World Conversations?</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 21 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as OpenAI's GPT-4 and Meta's LLaMA offer a promising approach for scalable personality assessment from open-ended language. However, inferring personality traits remains challenging, and earlier work often relied on synthetic data or social media text lacking psychometric validity. We introduce a real-world benchmark of 555 semi-structured interviews with BFI-10 self-report scores for evaluating LLM-based personality inference. Three state-of-the-art LLMs (GPT-4.1 Mini, Meta-LLaMA, and DeepSeek) were tested using zero-shot prompting for BFI-10 item prediction and both zero-shot and chain-of-thought prompting for Big Five trait inference. All models showed high test-retest reliability, but construct validity was limited: correlations with ground-truth scores were weak (max Pearson's $r = 0.27$), interrater agreement was low (Cohen's $\kappa < 0.10$), and predictions were biased toward moderate or high trait levels. Chain-of-thought prompting and longer input context modestly improved distributional alignment, but not trait-level accuracy. These results underscore limitations in current LLM-based personality inference and highlight the need for evidence-based development for psychological applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14335v1">ProofCompass: Enhancing Specialized Provers with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 19 pages, 7 figures. Accepted at the 2nd AI for MATH Workshop at the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Language models have become increasingly powerful tools for formal mathematical reasoning. However, most existing approaches rely exclusively on either large general-purpose models or smaller specialized models, each with distinct limitations, while training specialized large models still requires significant computational resources. This paper introduces ProofCompass, a novel hybrid methodology that achieves remarkable computational efficiency by strategically guiding existing specialized prover methods, such as DeepSeek-Prover-v1.5-RL (DSP-v1.5) with a Large Language Model (LLM) without requiring additional model training. The LLM provides natural language proof strategies and analyzes failed attempts to select intermediate lemmas, enabling effective problem decomposition. On the miniF2F benchmark, ProofCompass demonstrates substantial resource efficiency: it outperforms DSP-v1.5 ($54.9\% \rightarrow 55.3\%$) while using 25x fewer attempts ($3200 \rightarrow 128$). Our synergistic approach paves the way for simultaneously improving computational efficiency and accuracy in formal theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v2">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v1">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Submitted to Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [under review]
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14307v1">How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit increasingly sophisticated linguistic capabilities, yet the extent to which these behaviors reflect human-like cognition versus advanced pattern recognition remains an open question. In this study, we investigate how LLMs process the temporal meaning of linguistic aspect in narratives that were previously used in human studies. Using an Expert-in-the-Loop probing pipeline, we conduct a series of targeted experiments to assess whether LLMs construct semantic representations and pragmatic inferences in a human-like manner. Our findings show that LLMs over-rely on prototypicality, produce inconsistent aspectual judgments, and struggle with causal reasoning derived from aspect, raising concerns about their ability to fully comprehend narratives. These results suggest that LLMs process aspect fundamentally differently from humans and lack robust narrative understanding. Beyond these empirical findings, we develop a standardized experimental framework for the reliable assessment of LLMs' cognitive and linguistic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14304v1">Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v2">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://anonymous.4open.science/r/DiZO-E86D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13394v2">Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Evaluating machine-generated text remains a significant challenge in NLP, especially for non-English languages. Current methodologies, including automated metrics, human assessments, and LLM-based evaluations, predominantly focus on English, revealing a significant gap in multilingual evaluation frameworks. We introduce the Cross Lingual Auto Evaluation (CIA) Suite, an extensible framework that includes evaluator LLMs (Hercule) and a novel test set (Recon) specifically designed for multilingual evaluation. Our test set features 500 human-annotated instructions spanning various task capabilities along with human judgment scores across six languages. This would enable benchmarking of general-purpose multilingual LLMs and facilitate meta-evaluation of Evaluator LLMs. The proposed model, Hercule, is a cross-lingual evaluation model that addresses the scarcity of reference answers in the target language by learning to assign scores to responses based on easily available reference answers in English. Our experiments demonstrate that Hercule aligns more closely with human judgments compared to proprietary models, demonstrating the effectiveness of such cross-lingual evaluation in low resource scenarios. Further, it is also effective in zero-shot evaluation on unseen languages. This study is the first comprehensive examination of cross-lingual evaluation using LLMs, presenting a scalable and effective approach for multilingual assessment. All code, datasets, and models will be publicly available to enable further research in this important area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13302v1">The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13290v1">Towards Formal Verification of LLM-Generated Code from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, LLMs often generate incorrect code that users need to fix and the literature suggests users often struggle to detect these errors. In this work we seek to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the experience of using AI Code Assistants and potentially enable natural language programming for users with little or no programming knowledge. To address this challenge we propose to incorporate a formal query language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. Then, using such a query we propose to verify LLM generated code to ensure it matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language which includes such a formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter which is used for the verification. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13266v1">QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10917v2">LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13175v1">Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 42 pages. Supplementary material included at end of article
    </div>
    <details class="paper-abstract">
      The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13123v1">Detecting LLM-generated Code with Subtle Modification by Adversarial Training</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13105v1">SemCSE: Semantic Contrastive Sentence Embeddings Using LLM-Generated Summaries For Scientific Abstracts</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      We introduce SemCSE, an unsupervised method for learning semantic embeddings of scientific texts. Building on recent advances in contrastive learning for text embeddings, our approach leverages LLM-generated summaries of scientific abstracts to train a model that positions semantically related summaries closer together in the embedding space. This resulting objective ensures that the model captures the true semantic content of a text, in contrast to traditional citation-based approaches that do not necessarily reflect semantic similarity. To validate this, we propose a novel benchmark designed to assess a model's ability to understand and encode the semantic content of scientific texts, demonstrating that our method enforces a stronger semantic separation within the embedding space. Additionally, we evaluate SemCSE on the comprehensive SciRepEval benchmark for scientific text embeddings, where it achieves state-of-the-art performance among models of its size, thus highlighting the benefits of a semantically focused training approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06796v3">Write Your Own CodeChecker: An Automated Test-Driven Checker Development Approach with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 update metadata and artifact url
    </div>
    <details class="paper-abstract">
      With the rising demand for code quality assurance, developers are not only utilizing existing static code checkers but also seeking custom checkers to satisfy their specific needs. Nowadays, various code-checking frameworks provide extensive checker customization interfaces to meet this need. However, both the abstract checking logic and the complex API usage of large-scale checker frameworks make this task challenging. To this end, automated code checker generation is anticipated to ease the burden of checker development. In this paper, we propose AutoChecker, an innovative LLM-powered approach that can write code checkers automatically based on only a rule description and a test suite. To achieve comprehensive checking logic, AutoChecker incrementally updates the checker's logic by focusing on solving one selected case each time. To obtain precise API knowledge, during each iteration, it leverages fine-grained logic-guided API-context retrieval, where it first decomposes the checking logic into a series of sub-operations and then retrieves checker-related API-contexts for each sub-operation. For evaluation, we apply AutoChecker, five baselines, and three ablation methods using multiple LLMs to generate checkers for 20 randomly selected PMD rules. Experimental results show that AutoChecker significantly outperforms others across all effectiveness metrics, with an average test pass rate of 82.28%. Additionally, the checkers generated by AutoChecker can be successfully applied to real-world projects, matching the performance of official checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12948v1">Probabilistic Soundness Guarantees in LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      In reasoning chains generated by large language models (LLMs), initial errors often propagate and undermine the reliability of the final conclusion. Current LLM-based error detection methods often fail to detect propagated errors because they do not properly account for how earlier errors might corrupt judgments of downstream reasoning. To better detect such propagated errors, we introduce Autoregressive Reasoning Entailment Stability (ARES), a novel probabilistic framework that prevents error propagation by judging each claim based only on previously-assessed sound premises. This inductive method yields a nuanced score for each step and provides certified statistical guarantees of its soundness, rather than a brittle binary label. ARES achieves state-of-the-art performance across four benchmarks (72.1% Macro-F1, +8.2 points) and demonstrates superior robustness on very long synthetic reasoning chains, where it excels at detecting propagated errors (90.3% F1, +27.6 points).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06208v3">IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04631v2">On the Limitations of Large Language Models (LLMs): False Attribution</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 This paper was accepted for presentation by Recent Advances in NLP (RANLP) 2025 conference
    </div>
    <details class="paper-abstract">
      In this work, we introduce a new hallucination metric - Simple Hallucination Index (SHI) and provide insight into one important limitation of the parametric knowledge of large language models (LLMs), i.e. false attribution. The task of automatic author attribution for relatively small chunks of text is an important NLP task but can be challenging. We empirically evaluate the power of 3 open SotA LLMs in zero-shot setting (Gemma-7B, Mixtral 8x7B, and LLaMA-2-13B). We acquired the top 10 most popular books of a month, according to Project Gutenberg, divided each one into equal chunks of 400 words, and prompted each LLM to predict the author. We then randomly sampled 162 chunks per book for human evaluation, based on the error margin of 7% and a confidence level of 95%. The average results show that Mixtral 8x7B has the highest prediction accuracy, the lowest SHI, and a Pearson's correlation (r) of 0.724, 0.263, and -0.9996, respectively, followed by LLaMA-2-13B and Gemma-7B. However, Mixtral 8x7B suffers from high hallucinations for 3 books, rising as high as a SHI of 0.87 (in the range 0-1, where 1 is the worst). The strong negative correlation of accuracy and SHI, given by r, demonstrates the fidelity of the new hallucination metric, which may generalize to other tasks. We also show that prediction accuracies correlate positively with the frequencies of Wikipedia instances of the book titles instead of the downloads and we perform error analyses of predictions. We publicly release the annotated chunks of data and our codes to aid the reproducibility and evaluation of other models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v2">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v3">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. We release our pre-trained model and benchmark at https://github.com/awsm-research/SEALGuard to support further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09617v2">LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Graph recommendation methods, representing a connected interaction perspective, reformulate user-item interactions as graphs to leverage graph structure and topology to recommend and have proved practical effectiveness at scale. Large language models, representing a textual generative perspective, excel at modeling user languages, understanding behavioral contexts, capturing user-item semantic relationships, analyzing textual sentiments, and generating coherent and contextually relevant texts as recommendations. However, there is a gap between the connected graph perspective and the text generation perspective as the task formulations are different. A research question arises: how can we effectively integrate the two perspectives for more personalized recsys? To fill this gap, we propose to incorporate graph-edge information into LLMs via prompt and attention innovations. We reformulate recommendations as a probabilistic generative problem using prompts. We develop a framework to incorporate graph edge information from the prompt and attention mechanisms for graph-structured LLM recommendations. We develop a new prompt design that brings in both first-order and second-order graph relationships; we devise an improved LLM attention mechanism to embed direct the spatial and connectivity information of edges. Our evaluation of real-world datasets demonstrates the framework's ability to understand connectivity information in graph data and to improve the relevance and quality of recommendation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12820v1">Emotional Support with LLM-based Empathetic Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v4">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 52 pages, updated with new experimental results and implementation details
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12347v2">Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Code available at https://github.com/tanyuqian/synthetic-private-data
    </div>
    <details class="paper-abstract">
      Synthetic data offers a promising path to train models while preserving data privacy. Differentially private (DP) finetuning of large language models (LLMs) as data generator is effective, but is impractical when computation resources are limited. Meanwhile, prompt-based methods such as private evolution depend heavily on the manual prompts, and ineffectively use private information in their iterative data selection process. To overcome these limitations, we propose CTCL (Data Synthesis with ConTrollability and CLustering), a novel framework for generating privacy-preserving synthetic data without extensive prompt engineering or billion-scale LLM finetuning. CTCL pretrains a lightweight 140M conditional generator and a clustering-based topic model on large-scale public data. To further adapt to the private domain, the generator is DP finetuned on private data for fine-grained textual information, while the topic model extracts a DP histogram representing distributional information. The DP generator then samples according to the DP histogram to synthesize a desired number of data examples. Evaluation across five diverse domains demonstrates the effectiveness of our framework, particularly in the strong privacy regime. Systematic ablation validates the design of each framework component and highlights the scalability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02946v6">Scaling Trends for Data Poisoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 This arXiv version of the paper originally included an initial investigation of jailbreak-tuning, which can produce 60+ percentage point increases in vulnerability elicitation compared with standard data poisoning. Jailbreak-tuning has now been separated into a full independent paper, which can be found at arXiv:2507.11630
    </div>
    <details class="paper-abstract">
      LLMs produce harmful and undesirable behavior when trained on datasets containing even a small fraction of poisoned data. We demonstrate that GPT models remain vulnerable to fine-tuning on poisoned data, even when safeguarded by moderation systems. Given the persistence of data poisoning vulnerabilities in today's most capable models, this paper investigates whether these risks increase with model scaling. We evaluate three threat models -- malicious fine-tuning, imperfect data curation, and intentional data contamination -- across 24 frontier LLMs ranging from 1.5 to 72 billion parameters. Our experiments reveal that larger LLMs are significantly more susceptible to data poisoning, learning harmful behaviors from even minimal exposure to harmful data more quickly than smaller models. These findings underscore the need for leading AI companies to thoroughly red team fine-tuning APIs before public release and to develop more robust safeguards against data poisoning, particularly as models continue to scale in size and capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13577v1">LLM-Based Community Surveys for Operational Decision Making in Interconnected Utility Infrastructures</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      We represent interdependent infrastructure systems and communities alike with a hetero-functional graph (HFG) that encodes the dependencies between functionalities. This graph naturally imposes a partial order of functionalities that can inform the sequence of repair decisions to be made during a disaster across affected communities. However, using such technical criteria alone provides limited guidance at the point where the functionalities directly impact the communities, since these can be repaired in any order without violating the system constraints. To address this gap and improve resilience, we integrate community preferences to refine this partial order from the HFG into a total order. Our strategy involves getting the communities' opinions on their preferred sequence for repair crews to address infrastructure issues, considering potential constraints on resources. Due to the delay and cost associated with real-world survey data, we utilize a Large Language Model (LLM) as a proxy survey tool. We use the LLM to craft distinct personas representing individuals, each with varied disaster experiences. We construct diverse disaster scenarios, and each simulated persona provides input on prioritizing infrastructure repair needs across various communities. Finally, we apply learning algorithms to generate a global order based on the aggregated responses from these LLM-generated personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13555v1">Demystifying Feature Requests: Leveraging LLMs to Refine Feature Requests in Open-Source Software</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Accepted at the 33rd IEEE International Requirements Engineering 2025
    </div>
    <details class="paper-abstract">
      The growing popularity and widespread use of software applications (apps) across various domains have driven rapid industry growth. Along with this growth, fast-paced market changes have led to constantly evolving software requirements. Such requirements are often grounded in feature requests and enhancement suggestions, typically provided by users in natural language (NL). However, these requests often suffer from defects such as ambiguity and incompleteness, making them challenging to interpret. Traditional validation methods (e.g., interviews and workshops) help clarify such defects but are impractical in decentralized environments like open-source software (OSS), where change requests originate from diverse users on platforms like GitHub. This paper proposes a novel approach leveraging Large Language Models (LLMs) to detect and refine NL defects in feature requests. Our approach automates the identification of ambiguous and incomplete requests and generates clarification questions (CQs) to enhance their usefulness for developers. To evaluate its effectiveness, we apply our method to real-world OSS feature requests and compare its performance against human annotations. In addition, we conduct interviews with GitHub developers to gain deeper insights into their perceptions of NL defects, the strategies they use to address these defects, and the impact of defects on downstream software engineering (SE) tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05630v2">How Not to Detect Prompt Injections with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13525v1">Revisiting Prompt Engineering: A Comprehensive Evaluation for LLM-based Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Accepted to ACM RecSys2025 reproducibility
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can perform recommendation tasks by taking prompts written in natural language as input. Compared to traditional methods such as collaborative filtering, LLM-based recommendation offers advantages in handling cold-start, cross-domain, and zero-shot scenarios, as well as supporting flexible input formats and generating explanations of user behavior. In this paper, we focus on a single-user setting, where no information from other users is used. This setting is practical for privacy-sensitive or data-limited applications. In such cases, prompt engineering becomes especially important for controlling the output generated by the LLM. We conduct a large-scale comparison of 23 prompt types across 8 public datasets and 12 LLMs. We use statistical tests and linear mixed-effects models to evaluate both accuracy and inference cost. Our results show that for cost-efficient LLMs, three types of prompts are especially effective: those that rephrase instructions, consider background knowledge, and make the reasoning process easier to follow. For high-performance LLMs, simple prompts often outperform more complex ones while reducing cost. In contrast, commonly used prompting styles in natural language processing, such as step-by-step reasoning, or the use of reasoning models often lead to lower accuracy. Based on these findings, we provide practical suggestions for selecting prompts and LLMs depending on the required balance between accuracy and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13490v1">Revisiting LLM Value Probing Strategies: Are They Robust and Expressive?</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      There has been extensive research on assessing the value orientation of Large Language Models (LLMs) as it can shape user experiences across demographic groups. However, several challenges remain. First, while the Multiple Choice Question (MCQ) setting has been shown to be vulnerable to perturbations, there is no systematic comparison of probing methods for value probing. Second, it is unclear to what extent the probed values capture in-context information and reflect models' preferences for real-world actions. In this paper, we evaluate the robustness and expressiveness of value representations across three widely used probing strategies. We use variations in prompts and options, showing that all methods exhibit large variances under input perturbations. We also introduce two tasks studying whether the values are responsive to demographic context, and how well they align with the models' behaviors in value-related scenarios. We show that the demographic context has little effect on the free-text generation, and the models' values only weakly correlate with their preference for value-based actions. Our work highlights the need for a more careful examination of LLM value probing and awareness of its limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13474v1">Paper Summary Attack: Jailbreaking LLMs through LLM Safety Papers</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The safety of large language models (LLMs) has garnered significant research attention. In this paper, we argue that previous empirical studies demonstrate LLMs exhibit a propensity to trust information from authoritative sources, such as academic papers, implying new possible vulnerabilities. To verify this possibility, a preliminary analysis is designed to illustrate our two findings. Based on this insight, a novel jailbreaking method, Paper Summary Attack (\llmname{PSA}), is proposed. It systematically synthesizes content from either attack-focused or defense-focused LLM safety paper to construct an adversarial prompt template, while strategically infilling harmful query as adversarial payloads within predefined subsections. Extensive experiments show significant vulnerabilities not only in base LLMs, but also in state-of-the-art reasoning model like Deepseek-R1. PSA achieves a 97\% attack success rate (ASR) on well-aligned models like Claude3.5-Sonnet and an even higher 98\% ASR on Deepseek-R1. More intriguingly, our work has further revealed diametrically opposed vulnerability bias across different base models, and even between different versions of the same model, when exposed to either attack-focused or defense-focused papers. This phenomenon potentially indicates future research clues for both adversarial methodologies and safety alignment.Code is available at https://github.com/233liang/Paper-Summary-Attack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14240v1">HuggingGraph: Understanding the Supply Chain of LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) leverage deep learning to process and predict sequences of words from context, enabling them to perform various NLP tasks, such as translation, summarization, question answering, and content generation. However, the growing size and complexity of developing, training, and deploying advanced LLMs require extensive computational resources and large datasets. This creates a barrier for users. As a result, platforms that host models and datasets are widely used. For example, Hugging Face, one of the most popular platforms, hosted 1.8 million models and 450K datasets by June 2025, with no sign of slowing down. Since many LLMs are built from base models, pre-trained models, and external datasets, they can inherit vulnerabilities, biases, or malicious components from earlier models or datasets. Therefore, it is critical to understand the origin and development of these components to better detect potential risks, improve model fairness, and ensure compliance. Motivated by this, our project aims to study the relationships between models and datasets, which are core components of the LLM supply chain. First, we design a method to systematically collect LLM supply chain data. Using this data, we build a directed heterogeneous graph to model the relationships between models and datasets, resulting in a structure with 397,376 nodes and 453,469 edges. We then perform various analyses and uncover several findings, such as: (i) the LLM supply chain graph is large, sparse, and follows a power-law degree distribution; (ii) it features a densely connected core and a fragmented periphery; (iii) datasets play pivotal roles in training; (iv) strong interdependence exists between models and datasets; and (v) the graph is dynamic, with daily updates reflecting the ecosystem's ongoing evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15874v1">Why Braking? Scenario Extraction and Reasoning Utilizing LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The growing number of ADAS-equipped vehicles has led to a dramatic increase in driving data, yet most of them capture routine driving behavior. Identifying and understanding safety-critical corner cases within this vast dataset remains a significant challenge. Braking events are particularly indicative of potentially hazardous situations, motivating the central question of our research: Why does a vehicle brake? Existing approaches primarily rely on rule-based heuristics to retrieve target scenarios using predefined condition filters. While effective in simple environments such as highways, these methods lack generalization in complex urban settings. In this paper, we propose a novel framework that leverages Large Language Model (LLM) for scenario understanding and reasoning. Our method bridges the gap between low-level numerical signals and natural language descriptions, enabling LLM to interpret and classify driving scenarios. We propose a dual-path scenario retrieval that supports both category-based search for known scenarios and embedding-based retrieval for unknown Out-of-Distribution (OOD) scenarios. To facilitate evaluation, we curate scenario annotations on the Argoverse 2 Sensor Dataset. Experimental results show that our method outperforms rule-based baselines and generalizes well to OOD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13335v1">Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Humour, as a complex language form, is derived from myriad aspects of life, whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes. In this work, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular humour form. We compare models on simple puns and more complex topical humour that requires knowledge of real-world entities and events. In doing so, we curate a dataset of 600 jokes split across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes, where understanding relies on reasoning beyond "common sense", rooted instead in world knowledge regarding news events and pop culture. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (inc. reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most works in computational humour on overly simple joke forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13323v1">GeoReg: Weight-Constrained Few-Shot Regression for Socio-Economic Estimation using LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 15 pages, 13 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Socio-economic indicators like regional GDP, population, and education levels, are crucial to shaping policy decisions and fostering sustainable development. This research introduces GeoReg a regression model that integrates diverse data sources, including satellite imagery and web-based geospatial information, to estimate these indicators even for data-scarce regions such as developing countries. Our approach leverages the prior knowledge of large language model (LLM) to address the scarcity of labeled data, with the LLM functioning as a data engineer by extracting informative features to enable effective estimation in few-shot settings. Specifically, our model obtains contextual relationships between data features and the target indicator, categorizing their correlations as positive, negative, mixed, or irrelevant. These features are then fed into the linear estimator with tailored weight constraints for each category. To capture nonlinear patterns, the model also identifies meaningful feature interactions and integrates them, along with nonlinear transformations. Experiments across three countries at different stages of development demonstrate that our model outperforms baselines in estimating socio-economic indicators, even for low-income countries with limited data availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12619v1">BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 18 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become a cornerstone of modern AI, driving breakthroughs in natural language processing and expanding into multimodal jobs involving images, audio, and video. As with most computational software, it is important to distinguish between ordinary runtime performance and startup overhead. Prior research has focused on runtime performance: improving training efficiency and stability. This work focuses instead on the increasingly critical issue of startup overhead in training: the delay before training jobs begin execution. Startup overhead is particularly important in large, industrial-scale LLMs, where failures occur more frequently and multiple teams operate in iterative update-debug cycles. In one of our training clusters, more than 3.5% of GPU time is wasted due to startup overhead alone. In this work, we present the first in-depth characterization of LLM training startup overhead based on real production data. We analyze the components of startup cost, quantify its direct impact, and examine how it scales with job size. These insights motivate the design of Bootseer, a system-level optimization framework that addresses three primary startup bottlenecks: (a) container image loading, (b) runtime dependency installation, and (c) model checkpoint resumption. To mitigate these bottlenecks, Bootseer introduces three techniques: (a) hot block record-and-prefetch, (b) dependency snapshotting, and (c) striped HDFS-FUSE. Bootseer has been deployed in a production environment and evaluated on real LLM training workloads, demonstrating a 50% reduction in startup overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08339v2">What Factors Affect LLMs and RLLMs in Financial Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07188v2">Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts - we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also show that all tested models exhibit a consistent recency bias varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19477v2">Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12443v1">LLM-Based Config Synthesis requires Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Beyond hallucinations, another problem in program synthesis using LLMs is ambiguity in user intent. We illustrate the ambiguity problem in a networking context for LLM-based incremental configuration synthesis of route-maps and ACLs. These structures frequently overlap in header space, making the relative priority of actions impossible for the LLM to infer without user interaction. Measurements in a large cloud identify complex ACLs with 100's of overlaps, showing ambiguity is a real problem. We propose a prototype system, Clarify, which uses an LLM augmented with a new module called a Disambiguator that helps elicit user intent. On a small synthetic workload, Clarify incrementally synthesizes routing policies after disambiguation and then verifies them. Our treatment of ambiguities is useful more generally when the intent of updates can be correctly synthesized by LLMs, but their integration is ambiguous and can lead to different global behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12372v1">Web-Browsing LLMs Can Access Social Media Profiles and Infer User Demographics</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have traditionally relied on static training data, limiting their knowledge to fixed snapshots. Recent advancements, however, have equipped LLMs with web browsing capabilities, enabling real time information retrieval and multi step reasoning over live web content. While prior studies have demonstrated LLMs ability to access and analyze websites, their capacity to directly retrieve and analyze social media data remains unexplored. Here, we evaluate whether web browsing LLMs can infer demographic attributes of social media users given only their usernames. Using a synthetic dataset of 48 X (Twitter) accounts and a survey dataset of 1,384 international participants, we show that these models can access social media content and predict user demographics with reasonable accuracy. Analysis of the synthetic dataset further reveals how LLMs parse and interpret social media profiles, which may introduce gender and political biases against accounts with minimal activity. While this capability holds promise for computational social science in the post API era, it also raises risks of misuse particularly in information operations and targeted advertising underscoring the need for safeguards. We recommend that LLM providers restrict this capability in public facing applications, while preserving controlled access for verified research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12370v1">Beyond Single Models: Enhancing LLM Detection of Ambiguity in Requests through Debate</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted at the 2025 SICE Festival with Annual Conference (SICE FES)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant capabilities in understanding and generating human language, contributing to more natural interactions with complex systems. However, they face challenges such as ambiguity in user requests processed by LLMs. To address these challenges, this paper introduces and evaluates a multi-agent debate framework designed to enhance detection and resolution capabilities beyond single models. The framework consists of three LLM architectures (Llama3-8B, Gemma2-9B, and Mistral-7B variants) and a dataset with diverse ambiguities. The debate framework markedly enhanced the performance of Llama3-8B and Mistral-7B variants over their individual baselines, with Mistral-7B-led debates achieving a notable 76.7% success rate and proving particularly effective for complex ambiguities and efficient consensus. While acknowledging varying model responses to collaborative strategies, these findings underscore the debate framework's value as a targeted method for augmenting LLM capabilities. This work offers important insights for developing more robust and adaptive language understanding systems by showing how structured debates can lead to improved clarity in interactive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09477v2">Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 submitted to ARR May
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at https://github.com/DavidZWZ/Awesome-RAG-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10628v2">GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Code avaiable at https://github.com/hkgc-1/GHPO
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12308v1">Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 10 pages (6 content pages + 4 supplementary), 5 figures, Proceedings of the 2024 ACM/IEEE International Symposium on Machine Learning for CAD. 2024 (MLCAD'24)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become widely used across diverse NLP tasks and domains, demonstrating their adaptability and effectiveness. In the realm of Electronic Design Automation (EDA), LLMs show promise for tasks like Register-Transfer Level (RTL) code generation and summarization. However, despite the proliferation of LLMs for general code-related tasks, there's a dearth of research focused on evaluating and refining these models for hardware description languages (HDLs), notably VHDL. In this study, we evaluate the performance of existing code LLMs for VHDL code generation and summarization using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter, an in-house dataset, aims to gauge LLMs' understanding of functionally equivalent code. Our findings reveal consistent underperformance of these models across different metrics, underscoring a significant gap in their suitability for this domain. To address this challenge, we propose Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of LLMs for VHDL code generation and summarization tasks. CoDes involves generating a series of intermediate descriptive steps based on: (i) the problem statement for code generation, and (ii) the VHDL code for summarization. These steps are then integrated with the original input prompt (problem statement or code) and provided as input to the LLMs to generate the final output. Our experiments demonstrate that the CoDes approach significantly surpasses the standard prompting strategy across various metrics on both datasets. This method not only improves the quality of VHDL code generation and summarization but also serves as a framework for future research aimed at enhancing code LLMs for VHDL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12296v1">Humans are more gullible than LLMs in believing common psychological myths</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Despite widespread debunking, many psychological myths remain deeply entrenched. This paper investigates whether Large Language Models (LLMs) mimic human behaviour of myth belief and explores methods to mitigate such tendencies. Using 50 popular psychological myths, we evaluate myth belief across multiple LLMs under different prompting strategies, including retrieval-augmented generation and swaying prompts. Results show that LLMs exhibit significantly lower myth belief rates than humans, though user prompting can influence responses. RAG proves effective in reducing myth belief and reveals latent debiasing potential within LLMs. Our findings contribute to the emerging field of Machine Psychology and highlight how cognitive science methods can inform the evaluation and development of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12295v1">Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived embeddings.In addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at https://github.com/jicongfan/Text-Anomaly-Detection-Benchmark, this work provides a foundation for future research in robust and scalable text anomaly detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12215v1">Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence (AGI). While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner: (1) fine-tuning for legal move prediction to capture basic spatial rules, (2) incorporating strategic annotations to improve decision-making, and (3) applying reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional reward signals to enhance reasoning stability. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in spatially complex areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12207v1">BuildEvo: Designing Building Energy Consumption Forecasting Heuristics via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ICML 2025 CO-Build Workshop Poster
    </div>
    <details class="paper-abstract">
      Accurate building energy forecasting is essential, yet traditional heuristics often lack precision, while advanced models can be opaque and struggle with generalization by neglecting physical principles. This paper introduces BuildEvo, a novel framework that uses Large Language Models (LLMs) to automatically design effective and interpretable energy prediction heuristics. Within an evolutionary process, BuildEvo guides LLMs to construct and enhance heuristics by systematically incorporating physical insights from building characteristics and operational data (e.g., from the Building Data Genome Project 2). Evaluations show BuildEvo achieves state-of-the-art performance on benchmarks, offering improved generalization and transparent prediction logic. This work advances the automated design of robust, physically grounded heuristics, promoting trustworthy models for complex energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12205v1">Toward Efficient SpMV in Sparse LLMs via Block Extraction and Compressed Storage</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Sparse Matrix-Vector Multiplication (SpMV) has become a critical performance bottleneck in the local deployment of sparse Large Language Models (LLMs), where inference predominantly operates on workloads during the decoder phase with a batch size of one. Existing SpMV kernels and sparse matrix formats, originally designed for scientific computing, fail to exploit the unique structure patterns inherent in sparse LLMs, resulting in suboptimal performance and excessive storage overhead. This paper presents EC-SpMV, a GPU-optimized SpMV approach for accelerating sparse LLM inference. EC-SpMV introduces (1) a hierarchical block extraction algorithm that captures multiple granularities of block structures within sparse LLMs, and (2) a novel compressed sparse format (EC-CSR) that employs delta indexing to reduce storage overhead and enhance memory access efficiency. Evaluated on real sparse weight matrices from LLaMA and OPT models, EC-SpMV achieves up to 6.44x speedup over state-of-the-art SpMV libraries and reduces storage overhead by up to 55.4% compared to CSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05843v4">From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Current object detectors excel at entity localization and classification, yet exhibit inherent limitations in event recognition capabilities. This deficiency arises from their architecture's emphasis on discrete object identification rather than modeling the compositional reasoning, inter-object correlations, and contextual semantics essential for comprehensive event understanding. To address this challenge, we present a novel framework that expands the capability of standard object detectors beyond mere object recognition to complex event understanding through LLM-guided symbolic reasoning. Our key innovation lies in bridging the semantic gap between object detection and event understanding without requiring expensive task-specific training. The proposed plug-and-play framework interfaces with any open-vocabulary detector while extending their inherent capabilities across architectures. At its core, our approach combines (i) a symbolic regression mechanism exploring relationship patterns among detected entities and (ii) a LLM-guided strategically guiding the search toward meaningful expressions. These discovered symbolic rules transform low-level visual perception into interpretable event understanding, providing a transparent reasoning path from objects to events with strong transferability across domains.We compared our training-free framework against specialized event recognition systems across diverse application domains. Experiments demonstrate that our framework enhances multiple object detector architectures to recognize complex events such as illegal fishing activities (75% AUROC, +8.36% improvement), construction safety violations (+15.77%), and abnormal crowd behaviors (+23.16%). Code is available at \href{https://github.com/MAC-AutoML/SymbolicDet}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09037v2">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 72 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12143v1">Overview of the Sensemaking Task at the ELOQUENT 2025 Lab: LLMs as Teachers, Students and Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 30 pages, 7 figures, CLEF 2025 Conference and Labs of the Evaluation Forum
    </div>
    <details class="paper-abstract">
      ELOQUENT is a set of shared tasks that aims to create easily testable high-level criteria for evaluating generative language models. Sensemaking is one such shared task. In Sensemaking, we try to assess how well generative models ``make sense out of a given text'' in three steps inspired by exams in a classroom setting: (1) Teacher systems should prepare a set of questions, (2) Student systems should answer these questions, and (3) Evaluator systems should score these answers, all adhering rather strictly to a given set of input materials. We report on the 2025 edition of Sensemaking, where we had 7 sources of test materials (fact-checking analyses of statements, textbooks, transcribed recordings of a lecture, and educational videos) spanning English, German, Ukrainian, and Czech languages. This year, 4 teams participated, providing us with 2 Teacher submissions, 2 Student submissions, and 2 Evaluator submissions. We added baselines for Teacher and Student using commercial large language model systems. We devised a fully automatic evaluation procedure, which we compare to a minimalistic manual evaluation. We were able to make some interesting observations. For the first task, the creation of questions, better evaluation strategies will still have to be devised because it is difficult to discern the quality of the various candidate question sets. In the second task, question answering, the LLMs examined overall perform acceptably, but restricting their answers to the given input texts remains problematic. In the third task, evaluation of question answers, our adversarial tests reveal that systems using the LLM-as-a-Judge paradigm erroneously rate both garbled question-answer pairs and answers to mixed-up questions as acceptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v3">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12104v1">From Static to Intelligent: Evolving SaaS Pricing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 12 pages. Accepted at the SOC4AI Workshop (Service-Oriented Computing for AI Applications), held in conjunction with the 22nd International Conference on Service-Oriented Computing (ICSOC 2024)
    </div>
    <details class="paper-abstract">
      The SaaS paradigm has revolutionized software distribution by offering flexible pricing options to meet diverse customer needs. However, the rapid expansion of the SaaS market has introduced significant complexity for DevOps teams, who must manually manage and evolve pricing structures, an approach that is both time-consuming and prone to errors. The absence of automated tools for pricing analysis restricts the ability to efficiently evaluate, optimize, and scale these models. This paper proposes leveraging intelligent pricing (iPricing), dynamic, machine-readable pricing models, as a solution to these challenges. Intelligent pricing enables competitive analysis, streamlines operational decision-making, and supports continuous pricing evolution in response to market dynamics, leading to improved efficiency and accuracy. We present an LLM-driven approach that automates the transformation of static HTML pricing into iPricing, significantly improving efficiency and consistency while minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic Information Extractor that uses web scraping and LLMs technologies to extract essential pricing components, plans, features, usage limits, and add-ons, from SaaS websites. Validation against a dataset of 30 distinct commercial SaaS, encompassing over 150 intelligent pricings, demonstrates the system's effectiveness in extracting the desired elements across all steps. However, challenges remain in addressing hallucinations, complex structures, and dynamic content. This work highlights the potential of automating intelligent pricing transformation to streamline SaaS pricing management, offering implications for improved consistency and scalability in an increasingly intricate pricing landscape. Future research will focus on refining extraction capabilities and enhancing the system's adaptability to a wider range of SaaS websites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12084v1">LLAMA: Multi-Feedback Smart Contract Fuzzing Framework with LLM-Guided Seed Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Smart contracts play a pivotal role in blockchain ecosystems, and fuzzing remains an important approach to securing smart contracts. Even though mutation scheduling is a key factor influencing fuzzing effectiveness, existing fuzzers have primarily explored seed scheduling and generation, while mutation scheduling has been rarely addressed by prior work. In this work, we propose a Large Language Models (LLMs)-based Multi-feedback Smart Contract Fuzzing framework (LLAMA) that integrates LLMs, evolutionary mutation strategies, and hybrid testing techniques. Key components of the proposed LLAMA include: (i) a hierarchical prompting strategy that guides LLMs to generate semantically valid initial seeds, coupled with a lightweight pre-fuzzing phase to select high-potential inputs; (ii) a multi-feedback optimization mechanism that simultaneously improves seed generation, seed selection, and mutation scheduling by leveraging runtime coverage and dependency feedback; and (iii) an evolutionary fuzzing engine that dynamically adjusts mutation operator probabilities based on effectiveness, while incorporating symbolic execution to escape stagnation and uncover deeper vulnerabilities. Our experiments demonstrate that LLAMA outperforms state-of-the-art fuzzers in both coverage and vulnerability detection. Specifically, it achieves 91% instruction coverage and 90% branch coverage, while detecting 132 out of 148 known vulnerabilities across diverse categories. These results highlight LLAMA's effectiveness, adaptability, and practicality in real-world smart contract security testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12079v1">Findings of MEGA: Maths Explanation with LLMs using the Socratic Method for Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 This paper was accepted for the special issue AI for Education by the IEEE Signal Processing Magazine journal
    </div>
    <details class="paper-abstract">
      This paper presents an intervention study on the effects of the combined methods of (1) the Socratic method, (2) Chain of Thought (CoT) reasoning, (3) simplified gamification and (4) formative feedback on university students' Maths learning driven by large language models (LLMs). We call our approach Mathematics Explanations through Games by AI LLMs (MEGA). Some students struggle with Maths and as a result avoid Math-related discipline or subjects despite the importance of Maths across many fields, including signal processing. Oftentimes, students' Maths difficulties stem from suboptimal pedagogy. We compared the MEGA method to the traditional step-by-step (CoT) method to ascertain which is better by using a within-group design after randomly assigning questions for the participants, who are university students. Samples (n=60) were randomly drawn from each of the two test sets of the Grade School Math 8K (GSM8K) and Mathematics Aptitude Test of Heuristics (MATH) datasets, based on the error margin of 11%, the confidence level of 90%, and a manageable number of samples for the student evaluators. These samples were used to evaluate two capable LLMs at length (Generative Pretrained Transformer 4o (GPT4o) and Claude 3.5 Sonnet) out of the initial six that were tested for capability. The results showed that students agree in more instances that the MEGA method is experienced as better for learning for both datasets. It is even much better than the CoT (47.5% compared to 26.67%) in the more difficult MATH dataset, indicating that MEGA is better at explaining difficult Maths problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v2">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11997v1">Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11981v1">Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by RANLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonyms, words with multiple meanings, where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13789v3">LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by VLDB 2025 Industrial Track
    </div>
    <details class="paper-abstract">
      Display advertising provides significant value to advertisers, publishers, and users. Traditional display advertising systems utilize a multi-stage architecture consisting of retrieval, coarse ranking, and final ranking. However, conventional retrieval methods rely on ID-based learning to rank mechanisms and fail to adequately utilize the content information of ads, which hampers their ability to provide diverse recommendation lists. To address this limitation, we propose leveraging the extensive world knowledge of LLMs. However, three key challenges arise when attempting to maximize the effectiveness of LLMs: "How to capture user interests", "How to bridge the knowledge gap between LLMs and advertising system", and "How to efficiently deploy LLMs". To overcome these challenges, we introduce a novel LLM-based framework called LLM Empowered Display ADvertisement REcommender system (LEADRE). LEADRE consists of three core modules: (1) The Intent-Aware Prompt Engineering introduces multi-faceted knowledge and designs intent-aware <Prompt, Response> pairs that fine-tune LLMs to generate ads tailored to users' personal interests. (2) The Advertising-Specific Knowledge Alignment incorporates auxiliary fine-tuning tasks and Direct Preference Optimization (DPO) to align LLMs with ad semantic and business value. (3) The Efficient System Deployment deploys LEADRE in an online environment by integrating both latency-tolerant and latency-sensitive service. Extensive offline experiments demonstrate the effectiveness of LEADRE and validate the contributions of individual modules. Online A/B test shows that LEADRE leads to a 1.57% and 1.17% GMV lift for serviced users on WeChat Channels and Moments separately. LEADRE has been deployed on both platforms, serving tens of billions of requests each day.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11959v1">PoTPTQ: A Two-step Power-of-Two Post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted at ECAI 2025 (European Conference on Artificial Intelligence)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11953v1">IAM: Efficient Inference through Attention Mapping between Different-scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      LLMs encounter significant challenges in resource consumption nowadays, especially with long contexts. Despite extensive efforts dedicate to enhancing inference efficiency, these methods primarily exploit internal sparsity within the models, without leveraging external information for optimization. We identify the high similarity of attention matrices across different-scale LLMs, which offers a novel perspective for optimization. We first conduct a comprehensive analysis of how to measure similarity, how to select mapping Layers and whether mapping is consistency. Based on these insights, we introduce the IAM framework, which achieves dual benefits of accelerated attention computation and reduced KV cache usage by performing attention mapping between small and large LLMs. Our experimental results demonstrate that IAM can accelerate prefill by 15% and reduce KV cache usage by 22.1% without appreciably sacrificing performance. Experiments on different series of models show the generalizability of IAM. Importantly, it is also orthogonal to many existing KV cache optimization methods, making it a versatile addition to the current toolkit for enhancing LLM efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11898v1">Extremal Testing for Network Software using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Physicists often manually consider extreme cases when testing a theory. In this paper, we show how to automate extremal testing of network software using LLMs in two steps: first, ask the LLM to generate input constraints (e.g., DNS name length limits); then ask the LLM to generate tests that violate the constraints. We demonstrate how easy this process is by generating extremal tests for HTTP, BGP and DNS implementations, each of which uncovered new bugs. We show how this methodology extends to centralized network software such as shortest path algorithms, and how LLMs can generate filtering code to reject extremal input. We propose using agentic AI to further automate extremal testing. LLM-generated extremal testing goes beyond an old technique in software testing called Boundary Value Analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00406v2">Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 This is an extended version of a poster paper accepted and published at ECTEL-2025
    </div>
    <details class="paper-abstract">
      Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11878v1">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11851v1">Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Autoregressive language models are constrained by their inherently sequential nature, generating one token at a time. This paradigm limits inference speed and parallelism, especially during later stages of generation when the direction and semantics of text are relatively certain. In this work, we propose a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques to realize this potential and enable simultaneous prediction of multiple subsequent tokens. Our approach introduces several key innovations: (1) a masked-input formulation where multiple future tokens are jointly predicted from a common prefix; (2) a gated LoRA formulation that preserves the original LLM's functionality, while equipping it for multi-token prediction; (3) a lightweight, learnable sampler module that generates coherent sequences from the predicted future tokens; (4) a set of auxiliary training losses, including a consistency loss, to enhance the coherence and accuracy of jointly generated tokens; and (5) a speculative generation strategy that expands tokens quadratically in the future while maintaining high fidelity. Our method achieves significant speedups through supervised fine-tuning on pretrained models. For example, it generates code and math nearly 5x faster, and improves general chat and knowledge tasks by almost 2.5x. These gains come without any loss in quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06875v2">Eywa: Automating Model Based Testing using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Model-based testing (MBT), whereby a model of the system under test is analyzed to generate high-coverage test cases, has been used to test protocol implementations. A key barrier to the use of MBT is the need for users to understand protocol RFCs in detail to create a compliant model. Our new approach to MBT uses LLMs to automatically build rich models of intended protocol behavior from knowledge embedded in RFCs, blogs, and other natural language sources. Our approach addresses key challenges with using LLMs, including hallucinations and their inability to monolithically generate complex protocol models. We realize our approach through a novel protocol testing framework Eywa,and demonstrate its effectiveness through extensive case studies of DNS and BGP and a smaller study of SMTP. Despite minimal user effort, applying Eywa enabled the discovery of 32 unique bugs across widely used DNS, BGP, and SMTP implementations, 15 of which were previously undiscovered despite extensive prior testing with manually crafted models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v4">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ACL 2025 (main conference)
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have demonstrated gaps in diverse cultural awareness across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on LLMs' ability to display familiarity with various national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on multiple cultural awareness benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., cultural norms, artifacts, and institutions), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models and fails to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional cultural knowledge and open-ended cultural fluency when it comes to evaluating LLMs' cultural awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v4">Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12674v1">ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at \href{https://github.com/mmiroyan/ParaStudent}{\texttt{github.com/mmiroyan/ParaStudent}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24189v2">Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 8 pages, 7 figures. Accepted to Workshop on Structured Knowledge for Large Language Models (SKnowLLM) at KDD 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v3">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 49 pages, updated with new experimental results
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v3">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00838v2">Stylometry recognizes human and LLM-generated texts in short samples</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11210v1">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v3">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11198v1">Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Manuscript submitted for review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including coding and data annotation. While multi-agent systems (MAS) can emulate human coding workflows, their benefits over single-agent coding remain poorly understood. We conducted an experimental study of how agent persona and temperature shape consensus-building and coding accuracy of dialog segments based on a codebook with 8 codes. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic), significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing lead to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Only one model (OpenHermesV2:7B) and code category showed above-chance gains from MAS deliberation when temperature was 0.5 or lower and especially when the agents included at least one assertive persona. Qualitative analysis of MAS collaboration for these configurations suggests that MAS may nonetheless aid in narrowing ambiguous code applications that could improve codebooks and human-AI coding. We contribute new insight into the limits of LLM-based qualitative methods, challenging the notion that diverse MAS personas lead to better outcomes. We open-source our MAS and experimentation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08054v2">FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11128v1">What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 16 pages, 3 figures. Accepted at the 7th Workshop on eXplainable Knowledge Discovery in Data Mining (XKDD 2025), ECML PKDD 2025, Porto, Portugal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can memorize and reveal personal information, raising concerns regarding compliance with the EU's GDPR, particularly the Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the data to forget is already known but do not address how to identify which individual-fact associations are stored in the model. Privacy auditing techniques typically operate at the population level or target a small set of identifiers, limiting applicability to individual-level data inquiries. We introduce WikiMem, a dataset of over 5,000 natural language canaries covering 243 human-related properties from Wikidata, and a model-agnostic metric to quantify human-fact associations in LLMs. Our approach ranks ground-truth values against counterfactuals using calibrated negative log-likelihood across paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B parameters), showing that memorization correlates with subject web presence and model scale. We provide a foundation for identifying memorized personal data in LLMs at the individual level, enabling the dynamic construction of forget sets for machine unlearning and RTBF requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21033v2">Plancraft: an evaluation dataset for planning with LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as a handcrafted planner and Oracle Retriever, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and compare their performance and efficiency to a handcrafted planner. Overall, we find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and offer suggestions on how to improve their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11112v1">Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11097v1">The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 21 pages, 9 figures, work in progress
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11086v1">Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The growing prevalence of cross-border financial activities in global markets has underscored the necessity of accurately identifying and classifying foreign entities. This practice is essential within the Spanish financial system for ensuring robust risk management, regulatory adherence, and the prevention of financial misconduct. This process involves a labor-intensive entity-matching task, where entities need to be validated against available reference sources. Challenges arise from linguistic variations, special characters, outdated names, and changes in legal forms, complicating traditional matching algorithms like Jaccard, cosine, and Levenshtein distances. These methods struggle with contextual nuances and semantic relationships, leading to mismatches. To address these limitations, we explore Large Language Models (LLMs) as a flexible alternative. LLMs leverage extensive training to interpret context, handle abbreviations, and adapt to legal transitions. We evaluate traditional methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases. Results show traditional methods achieve accuracies over 92% but suffer high false positive rates (20-40%). Interface-based LLMs outperform, achieving accuracies above 93%, F1 scores exceeding 96%, and lower false positives (40-80%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11083v1">Function-to-Style Guidance of LLMs for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 This paper has been accepted by ICML 2025. Models and benchmarks can be found at https://www.modelscope.cn/collections/F2STrans-42526ff95dd843
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant strides in code translation tasks. However, ensuring both the correctness and readability of translated code remains a challenge, limiting their effective adoption in real-world software development. In this work, we propose F2STrans, a function-to-style guiding paradigm designed to progressively improve the performance of LLMs in code translation. Our approach comprises two key stages: (1) Functional learning, which optimizes translation correctness using high-quality source-target code pairs mined from online programming platforms, and (2) Style learning, which improves translation readability by incorporating both positive and negative style examples. Additionally, we introduce a novel code translation benchmark that includes up-to-date source code, extensive test cases, and manually annotated ground-truth translations, enabling comprehensive functional and stylistic evaluations. Experiments on both our new benchmark and existing datasets demonstrate that our approach significantly improves code translation performance. Notably, our approach enables Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code translation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11052v1">LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11042v1">Aligned Query Expansion: Efficient Query Expansion for Information Retrieval through LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      With the breakthroughs in large language models (LLMs), query generation techniques that expand documents and queries with related terms are becoming increasingly popular in the information retrieval field. Such techniques have been shown to improve the effectiveness of traditional lexical retrieval methods by dealing with the vocabulary mismatch problem. Recent work has found that generating queries with a greedy decoding strategy can produce sub-optimal queries, including hallucinations, and proposed to filter out queries before expansion. This `generate-then-filter' approach is costly, as it requires generating multiple queries and applying a relevance model to all of them and does not teach the LLM which of the generated queries is more effective for expansion. To overcome such limitations, we propose Aligned Query Expansion (AQE), a novel approach to enhance query expansion for passage retrieval in open-domain question answering. AQE leverages recent techniques in LLM alignment to fine-tune models for generating query expansions that directly optimize the effectiveness of the retrieval task, eliminating the need for additional filtering steps. This alignment ensures that queries are more relevant, reducing computational costs while improving retrieval effectiveness. Empirical evaluations show that AQE outperforms baseline models for query expansion in both in-domain and out-of-domain settings, demonstrating significant improvements in retrieval effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14959v2">Understanding the Dark Side of LLMs' Intrinsic Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at https://x-isc.info/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19502v2">MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at https://github.com/AlgazinovAleksandr/Multi-Agent-MATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v2">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Under Review at Information and Software Technology
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10972v1">Teach Me Sign: Stepwise Prompting LLM for Sign Language Production</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted by IEEE ICIP 2025
    </div>
    <details class="paper-abstract">
      Large language models, with their strong reasoning ability and rich knowledge, have brought revolution to many tasks of AI, but their impact on sign language generation remains limited due to its complexity and unique rules. In this paper, we propose TEAch Me Sign (TEAM-Sign), treating sign language as another natural language. By fine-tuning an LLM, we enable it to learn the correspondence between text and sign language, and facilitate generation. Considering the differences between sign and spoken language, we employ a stepwise prompting strategy to extract the inherent sign language knowledge within the LLM, thereby supporting the learning and generation process. Experimental results on How2Sign and Phoenix14T datasets demonstrate that our approach effectively leverages both the sign language knowledge and reasoning capabilities of LLM to align the different distribution and grammatical rules between sign and spoken language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10920v1">HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10917v1">LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v2">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
    </details>
</div>
