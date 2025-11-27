# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10507v2">AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21638v1">Aligning LLMs Toward Multi-Turn Conversational Outcomes Using Iterative PPO</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Optimizing large language models (LLMs) for multi-turn conversational outcomes remains a significant challenge, especially in goal-oriented settings like AI marketing or sales agents who facilitate transactions via messaging platforms. The difficulty stems from sparse, long-horizon rewards and the discrepancy between response-level planning and token-level generation. In this technical note, we propose a formal reduction of the multi-turn RL problem into a sequence of single-turn RLHF-style problems. This is achieved by setting a learned multi-turn Q-function as the reward model for the single-turn problem. We demonstrate and prove a key insight: solving this single-turn RL problem with standard token-level PPO is equivalent to a policy improvement step within the multi-turn problem. This insight naturally leads to Iterative PPO, a batch online policy iteration algorithm that alternates between fitting Q-functions from logged conversation trajectories and improving the policy. A major practical advantage is that Iterative PPO directly leverages stable, off-the-shelf single-turn RLHF tools, making it straightforward to implement. Our method occupies a middle ground between fully online and fully offline approaches, retaining the adaptability of online updates while gaining the stability benefits of offline training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21624v1">TAGFN: A Text-Attributed Graph Dataset for Fake News Detection in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently revolutionized machine learning on text-attributed graphs, but the application of LLMs to graph outlier detection, particularly in the context of fake news detection, remains significantly underexplored. One of the key challenges is the scarcity of large-scale, realistic, and well-annotated datasets that can serve as reliable benchmarks for outlier detection. To bridge this gap, we introduce TAGFN, a large-scale, real-world text-attributed graph dataset for outlier detection, specifically fake news detection. TAGFN enables rigorous evaluation of both traditional and LLM-based graph outlier detection methods. Furthermore, it facilitates the development of misinformation detection capabilities in LLMs through fine-tuning. We anticipate that TAGFN will be a valuable resource for the community, fostering progress in robust graph-based outlier detection and trustworthy AI. The dataset is publicly available at https://huggingface.co/datasets/kayzliu/TAGFN and our code is available at https://github.com/kayzliu/tagfn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21613v1">Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal-URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v1">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 24 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings, the sequence-wise geometric patterns in LLaMa, and experiment with repeating token sequences. We aim to support systematic analysis of Transformer internals with the goal of enabling further reproducible interpretability research. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v2">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI 2026 Workshop on Graphs and more Complex Structures For Learning and Reasoning (GCLR). Version 2 fixes typos in author name and Figure 1
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.14746v5">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Leibler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^γ$ where $D$ is the size of the training data and $ γ\in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.17846v2">Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19218v2">Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21401v1">Can LLMs extract human-like fine-grained evidence for evidence-based fact-checking?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Misinformation frequently spreads in user comments under online news articles, highlighting the need for effective methods to detect factually incorrect information. To strongly support or refute claims extracted from such comments, it is necessary to identify relevant documents and pinpoint the exact text spans that justify or contradict each claim. This paper focuses on the latter task -- fine-grained evidence extraction for Czech and Slovak claims. We create new dataset, containing two-way annotated fine-grained evidence created by paid annotators. We evaluate large language models (LLMs) on this dataset to assess their alignment with human annotations. The results reveal that LLMs often fail to copy evidence verbatim from the source text, leading to invalid outputs. Error-rate analysis shows that the {llama3.1:8b model achieves a high proportion of correct outputs despite its relatively small size, while the gpt-oss-120b model underperforms despite having many more parameters. Furthermore, the models qwen3:14b, deepseek-r1:32b, and gpt-oss:20b demonstrate an effective balance between model size and alignment with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21334v1">Emergent Lexical Semantics in Neural Language Models: Testing Martin's Law on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 paper draft
    </div>
    <details class="paper-abstract">
      We present the first systematic investigation of Martin's Law - the empirical relationship between word frequency and polysemy - in text generated by neural language models during training. Using DBSCAN clustering of contextualized embeddings as an operationalization of word senses, we analyze four Pythia models (70M-1B parameters) across 30 training checkpoints. Our results reveal a non-monotonic developmental trajectory: Martin's Law emerges around checkpoint 100, reaches peak correlation (r > 0.6) at checkpoint 104, then degrades by checkpoint 105. Smaller models (70M, 160M) experience catastrophic semantic collapse at late checkpoints, while larger models (410M, 1B) show graceful degradation. The frequency-specificity trade-off remains stable (r $\approx$ -0.3) across all models. These findings suggest that compliance with linguistic regularities in LLM-generated text is not monotonically increasing with training, but instead follows a balanced trajectory with an optimal semantic window. This work establishes a novel methodology for evaluating emergent linguistic structure in neural language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19123v2">Facilitating the Integration of LLMs Into Online Experiments With Simple Chat</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, understanding human-LLM interactions is emerging as a central priority in psychological research. Online experiments offer an efficient means to study human-LLM interactions, yet integrating LLMs into established survey platforms remains technically demanding, particularly when aiming for ecologically valid, real-time conversational experiences with strong experimental control. We introduce Simple Chat, an open-source, research-focused chat interface that streamlines LLM integration for platforms such as Qualtrics, oTree, and LimeSurvey, while presenting a unified participant experience across conditions. Simple Chat connects to both commercial providers and open-weights models, supports streaming responses to preserve conversational flow, and offers an administrative interface for fine-grained control of prompts and interface features. By reducing technical barriers, standardizing interfaces, and improving participant experience, Simple Chat helps advance the study of human-LLM interaction. In this article, we outline Simple Chat's key features, provide a step-by-step tutorial, and demonstrate its utility through two illustrative case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21322v1">TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how such chatbots represent diverse cultures. At the same time, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2,925 annotations from 108 annotators with lived cultural experience from across 71 regions in India and 14 languages. Concerningly, we find that 88\% of the generated stories contain one or more cultural inaccuracies, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. Lastly, we transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of foundational models. Through this evaluation, we surprisingly discover that models often possess the requisite cultural knowledge despite generating stories rife with cultural misrepresentations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21270v1">Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17041v2">CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v1">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20403v2">LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted at 40th IEEE/ACM International Conference on Automated Software Engineering
    </div>
    <details class="paper-abstract">
      Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21140v1">How to Correctly Report LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as evaluators in lieu of humans. While scalable, their judgments are noisy due to imperfect specificity and sensitivity of LLMs, leading to biased accuracy estimates. Although bias-correction methods exist, they are underutilized in LLM research and typically assume exact knowledge of the model's specificity and sensitivity. Furthermore, in general we only have estimates of these values and it is not well known how to properly construct confidence intervals using only estimates. This work presents a simple plug-in framework that corrects such bias and constructs confidence intervals reflecting uncertainty from both test and calibration dataset, enabling practical and statistically sound LLM-based evaluation. Additionally, to reduce uncertainty in the accuracy estimate, we introduce an adaptive algorithm that efficiently allocates calibration sample sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04231v2">Empowering Time Series Forecasting with LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02833v4">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Published as a conference paper at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19933v2">Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being rapidly integrated into decision-support tools, automation workflows, and AI-enabled software systems. However, their behavior in production environments remains poorly understood, and their failure patterns differ fundamentally from those of traditional machine learning models. This paper presents a system-level taxonomy of fifteen hidden failure modes that arise in real-world LLM applications, including multi-step reasoning drift, latent inconsistency, context-boundary degradation, incorrect tool invocation, version drift, and cost-driven performance collapse. Using this taxonomy, we analyze the growing gap in evaluation and monitoring practices: existing benchmarks measure knowledge or reasoning but provide little insight into stability, reproducibility, drift, or workflow integration. We further examine the production challenges associated with deploying LLMs - including observability limitations, cost constraints, and update-induced regressions - and outline high-level design principles for building reliable, maintainable, and cost-aware LLM systems. Finally, we outline high-level design principles for building reliable, maintainable, and cost-aware LLM-based systems. By framing LLM reliability as a system-engineering problem rather than a purely model-centric one, this work provides an analytical foundation for future research on evaluation methodology, AI system robustness, and dependable LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21089v1">MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly deployed as dense transformers, where every parameter in every feed-forward block is activated for every token. While architecturally simple, this is computationally inefficient, since inference costs scale linearly with parameter count. Recent upcycling methods such as MoEfication, CMoE, ToMoE, and MoORE reveal that much of the useful computation lives in sparse, semi-modular substructures inside dense feed-forward networks, but these approaches typically rely on clustering, activation profiling, singular value decomposition, or custom routing that requires calibration data. This paper introduces MLPMoE (MLP Mixture-of-Experts), a training-free, deterministic transformation that restructures the dense MLP in transformer blocks into a static, high-cardinality mixture of experts. The transformation uses simple tensor slicing and summation, reinterpreting the algebra of tensor parallelism as a topological conversion rather than a distributed training pattern. We further introduce Fractal Fade (differential branch sparsity) and Compensated Pruning (variance-preserving branch reduction) as lightweight mechanisms for structured sparsity. On Qwen2.5-0.5B-Instruct and DeepSeek-R1-Distill-Llama-8B, the zero-shot MLPMoE transform changes a proxy perplexity metric by less than 0.05 percent while keeping the parameter count effectively constant. On the 8B model, differential sparsity removes about 20 percent of MLP parameters while keeping perplexity within about 2 percent of the dense baseline. The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training. Code is available at https://gist.github.com/iwallarm/fc2ef1eddf226ca7814f9e5e2ae9bad1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21075v1">Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Effective post-training is essential to align Large Language Models (LLMs) with specialized biomedical knowledge to accelerate life science research. However, current approaches face significant limitations. First, biomedical reasoning involves intricate mechanisms often represented by sparse textual data. Standard Supervised Fine-Tuning (SFT) tends to overfit to surface-level instruction patterns without effectively internalizing this fragmented scientific knowledge. Second, Reinforcement Learning (RL) is impractical for this domain, as defining meaningful rewards often necessitates prohibitive experimental validation (e.g., wet-lab verification of drug responses), rendering real-time feedback unfeasible. We propose Balanced Fine-Tuning (BFT), an efficient post-training method designed to learn complex reasoning from sparse data without external reward signals. BFT operates through a two-layer weighting mechanism: 1. At the token level, it scales loss via prediction probabilities to stabilize gradients and prevent overfitting; 2. At the sample level, it uses "minimum group confidence" to adaptively enhance the learning of hard samples. Experiments demonstrate that BFT significantly outperforms SFT. In medical tasks, it enables LLMs to acquire knowledge that SFT misses. In biological tasks, BFT-based LLMs surpass GeneAgent (an accurate agent for biology analysis) in biological process reasoning. Moreover, the text embeddings generated by BFT can be directly applied to downstream tasks, such as gene interaction and single-cell perturbation response prediction. These results indicate that BFT facilitates broad applications of LLMs in biomedical research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21056v1">A Unified Understanding of Offline Data Selection and Online Self-refining Generation for Post-training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Offline data selection and online self-refining generation, which enhance the data quality, are crucial steps in adapting large language models (LLMs) to specific downstream tasks. We tackle offline data selection and online self-refining generations through an optimization perspective. Specifically, bilevel data selection is used for offline data selection with respect to the validation dataset, and we treat online self-refining generation as a model adaptation step of selecting the model trained on current responses that best fits the validation data. Our framework offers a unified understanding of offline data selection and self-refining generation by assigning a learned data weight to each question and response, either explicitly or implicitly. For the first time, we theoretically demonstrate the effectiveness of the bilevel data selection framework and demonstrate its performance gains over unfiltered direct mixing baselines. By combining offline data with validation-weighted online generations, our method enhances fine-tuning performance. Experiments on quality enhancement and safety-aware LLM fine-tuning validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21050v1">Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 AAAI-26 Workshop on Post-AI Formal Methods
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21038v1">Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 13 pages total (7 pages main text, 3 pages references, 3 pages appendix), 2 figures, 14 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl
    </div>
    <details class="paper-abstract">
      Can in-context learning (ICL) override pre-trained label semantics, or does it merely refine an existing semantic backbone? We address this question by treating LLMs as prompt-induced classifiers and contrasting their behavior under \emph{natural} demonstrations (with correct labels) and \emph{inverted} demonstrations (systematically flipping label meanings). We decompose ICL behavior into three alignment metrics (truth, prior, and prompt alignment) and introduce a semantic override rate, defined as correctness under flipped semantics. Across eight classification tasks and eight open-source LLMs (1--12B parameters), we find consistent evidence for a semantic anchor view. With natural demonstrations, ICL improves accuracy while maintaining strong prior alignment; most correct predictions coincide with zero-shot behavior, even when the prior is weak. With inverted demonstrations, models cannot learn coherent anti-semantic classifiers: prompt alignment increases only by sacrificing accuracy, and semantic override rates remain exactly zero in our few-shot 1--12B setting. Rather than flexibly remapping label meanings, ICL primarily adjusts how inputs project onto stable semantic directions learned during pre-training, clarifying fundamental limits of few-shot prompting and suggesting that overriding label semantics at these scales requires interventions beyond ICL. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21037v1">LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted to the PerFM Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Foundation models are increasingly used to personalize learning, yet many systems still assume fixed curricula or coarse progress signals, limiting alignment with learners' day-to-day needs. At the other extreme, lightweight incidental systems offer flexible, in-the-moment content but rarely guide learners toward mastery. Prior work privileges either continuity (maintaining a plan across sessions) or initiative (reacting to the moment), not both, leaving learners to navigate the trade-off between recency and trajectory-immediate relevance versus cumulative, goal-aligned progress. We present LOOM, an agentic pipeline that infers evolving learner needs from recent LLM conversations and a dynamic learner memory graph, then assembles coherent learning materials personalized to the learner's current needs, priorities, and understanding. These materials link adjacent concepts and surface gaps as tightly scoped modules that cumulatively advance broader goals, providing guidance and sustained progress while remaining responsive to new interests. We describe LOOM's end-to-end architecture and working prototype, including conversation summarization, topic planning, course generation, and graph-based progress tracking. In a formative study with ten participants, users reported that LOOM's generated lessons felt relevant to their recent activities and helped them recognize knowledge gaps, though they also highlighted needs for greater consistency and control. We conclude with design implications for more robust, mixed-initiative learning pipelines that integrate structured learner modelling with everyday LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21033v1">Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      The rationality of law manifests in two forms: substantive rationality, which concerns the fairness or moral desirability of outcomes, and formal rationality, which requires legal decisions to follow explicitly stated, general, and logically coherent rules. Existing LLM-based systems excel at surface-level text analysis but lack the guarantees required for principled jurisprudence. We introduce L4M, a novel framework that combines adversarial LLM agents with SMT-solver-backed proofs to unite the interpretive flexibility of natural language with the rigor of symbolic verification. The pipeline consists of three phases: (1) Statute Formalization, where domain-specific prompts convert legal provisions into logical formulae; (2) Dual Fact and Statute Extraction, in which prosecutor- and defense-aligned LLMs independently map case narratives to fact tuples and statutes, ensuring role isolation; and (3) Solver-Centric Adjudication, where an autoformalizer compiles both parties' arguments into logic constraints, and unsat cores trigger iterative self-critique until a satisfiable formula is achieved, which is then verbalized by a Judge-LLM into a transparent verdict and optimized sentence. Experimental results on public benchmarks show that our system surpasses advanced LLMs including GPT-o4-mini, DeepSeek-V3, and Claude 4 as well as state-of-the-art Legal AI baselines, while providing rigorous and explainable symbolic justifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21022v1">Lightweight Model Editing for LLMs to Correct Deprecated API Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Pre-trained or fine-tuned on large code corpora, Large Language Models (LLMs) have demonstrated strong performance in code completion tasks. However, their embedded knowledge is constrained by the timeliness of training data, which often includes code using deprecated APIs. Consequently, LLMs frequently generate deprecated APIs that will no longer be supported in future versions of third-party libraries. While retraining LLMs on updated codebases could refresh their API knowledge, this approach is computationally expensive. Recently, lightweight model editing methods have emerged to efficiently correct specific knowledge in LLMs. However, it remains unclear whether these methods can effectively update deprecated API knowledge and enable edited models to generate up-to-date APIs. To address this gap, we conduct the first systematic study applying 10 state-of-the-art model editing techniques to update deprecated API knowledge in three LLMs: Qwen2.5-Coder, StarCoder2, and DeepSeek-Coder. We introduce EDAPIBench, a dedicated benchmark featuring over 70 deprecated APIs from 8 popular Python libraries, with more than 3,000 editing instances. Our results show that the parameter-efficient fine-tuning method AdaLoRA achieves the best performance in enabling edited models to generate correct, up-to-date APIs, but falls short in Specificity (i.e., the editing influences untargeted knowledge). To resolve this, we propose AdaLoRA-L, which defines "Common API Layers" (layers within the LLMs with high importance across all APIs, storing general knowledge and excluded from editing) and restricts edits exclusively to "Specific API Layers" (layers with high importance only for the target API, storing the API-specific knowledge). Experimental results demonstrate that AdaLoRA-L significantly improves Specificity while maintaining comparable performance across other evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20993v1">Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18581v2">TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Many recent studies showed that LLMs are vulnerable to jailbreak attacks, where an attacker can perturb the input of an LLM to induce it to generate an output for a harmful question. In general, existing jailbreak techniques either optimize a semantic template intended to induce the LLM to produce harmful outputs or optimize a suffix that leads the LLM to initiate its response with specific tokens (e.g., "Sure"). In this work, we introduce TASO (Template and Suffix Optimization), a novel jailbreak method that optimizes both a template and a suffix in an alternating manner. Our insight is that suffix optimization and template optimization are complementary to each other: suffix optimization can effectively control the first few output tokens but cannot control the overall quality of the output, while template optimization provides guidance for the entire output but cannot effectively control the initial tokens, which significantly impact subsequent responses. Thus, they can be combined to improve the attack's effectiveness. We evaluate the effectiveness of TASO on benchmark datasets (including HarmBench and AdvBench) on 24 leading LLMs (including models from the Llama family, OpenAI, and DeepSeek). The results demonstrate that TASO can effectively jailbreak existing LLMs. We hope our work can inspire future studies in exploring this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20982v1">A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      To meet strict Service-Level Objectives (SLOs),contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks inherent to each phase. However, the heterogeneity of LLM workloads causes producerconsumer imbalance between the two instance types in such disaggregated architecture. To address this problem, we propose DOPD (Dynamic Optimal Prefill/Decoding), a dynamic LLM inference system that adjusts instance allocations to achieve an optimal prefill-to-decoding (P/D) ratio based on real-time load monitoring. Combined with an appropriate request-scheduling policy, DOPD effectively resolves imbalances between prefill and decoding instances and mitigates resource allocation mismatches due to mixed-length requests under high concurrency. Experimental evaluations show that, compared with vLLM and DistServe (representative aggregation-based and disaggregationbased approaches), DOPD improves overall system goodput by up to 1.5X, decreases P90 time-to-first-token (TTFT) by up to 67.5%, and decreases P90 time-per-output-token (TPOT) by up to 22.8%. Furthermore, our dynamic P/D adjustment technique performs proactive reconfiguration based on historical load, achieving over 99% SLOs attainment while using less additional resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.17264v5">No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Deploying million-token Large Language Models (LLMs) is challenging because production workloads are highly heterogeneous, mixing short queries and long documents. This heterogeneity, combined with the quadratic complexity of attention, creates severe convoy effects where long-running requests stall short, interactive ones, degrading system responsiveness. We present Medha, a serving system that eliminates these convoys by introducing fine-grained, preemptive scheduling to LLM inference. Medha makes preemption practical with a co-designed set of mechanisms -- including Adaptive Chunking and Stream Pipeline Parallel that overcome the perceived inefficiencies and scaling challenges of chunking. Additionally, we present a new parallelism strategy KV-Cache Parallelism to reduce the decode latency and afford interactivity despite very long context. These mechanisms are orchestrated by a Length-Aware Relative Slack (LARS) scheduler, a deadline and heterogeneity-aware scheduling policy that prevents both the convoy effect and the starvation that plagues simpler policies. Under a heterogeneous workload, Medha improves throughput by 5.7x while reducing median and 99th percentile latency by 30x and 174x, respectively, compared to state-of-the-art non-preemptive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20965v1">TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)
    </div>
    <details class="paper-abstract">
      Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v4">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20942v1">Improving Procedural Skill Explanations via Constrained Generation: A Symbolic-LLM Hybrid Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      In procedural skill learning, instructional explanations must convey not just steps, but the causal, goal-directed, and compositional logic behind them. Large language models (LLMs) often produce fluent yet shallow responses that miss this structure. We present Ivy, an AI coaching system that delivers structured, multi-step explanations by combining symbolic Task-Method-Knowledge (TMK) models with a generative interpretation layer-an LLM that constructs explanations while being constrained by TMK structure. TMK encodes causal transitions, goal hierarchies, and problem decompositions, and guides the LLM within explicit structural bounds. We evaluate Ivy against responses against GPT and retrieval-augmented GPT baselines using expert and independent annotations across three inferential dimensions. Results show that symbolic constraints consistently improve the structural quality of explanations for "how" and "why" questions. This study demonstrates a scalable AI for education approach that strengthens the pedagogical value of AI-generated explanations in intelligent coaching systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14993v3">Multi-modal Generative AI: Multi-modal LLMs, Diffusions, and the Unification</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 21 pages, 10 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20613v1">Can Vibe Coding Beat Graduate CS Students? An LLM vs. Human Coding Tournament on Market-driven Strategic Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has revolutionized AI-assisted code generation. This rapid development of LLMs has outpaced our ability to properly benchmark them. Prevailing benchmarks emphasize unit-test pass rates and syntactic correctness. Such metrics understate the difficulty of many real-world problems that require planning, optimization, and strategic interaction. We introduce a multi-agent reasoning-driven benchmark based on a real-world logistics optimization problem (Auction, Pickup, and Delivery Problem) that couples competitive auctions with capacity-constrained routing. The benchmark requires building agents that can (i) bid strategically under uncertainty and (ii) optimize planners that deliver tasks while maximizing profit. We evaluate 40 LLM-coded agents (by a wide range of state-of-the-art LLMs under multiple prompting methodologies, including vibe coding) against 17 human-coded agents developed before the advent of LLMs. Our results over 12 double all-play-all tournaments and $\sim 40$k matches demonstrate (i) a clear superiority of human(graduate students)-coded agents: the top 5 spots are consistently won by human-coded agents, (ii) the majority of LLM-coded agents (33 out of 40) are beaten by very simple baselines, and (iii) given the best human solution as an input and prompted to improve upon, the best performing LLM makes the solution significantly worse instead of improving it. Our results highlight a gap in LLMs' ability to produce code that works competitively in the real-world, and motivate new evaluations that emphasize reasoning-driven code synthesis in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20604v1">On Evaluating LLM Alignment by Evaluating LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 NeurIPS 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20526v1">Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain underexplored.In China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20468v1">DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and problem-solving.Recent works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic learning.DRAFT-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20403v1">LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted at 40th IEEE/ACM International Conference on Automated Software Engineering
    </div>
    <details class="paper-abstract">
      Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21740v3">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 INLG25
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20340v1">Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 accepted by AAAI-2026
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by utilizing otherwise idle computational resources during memory-to-chip data transfer. Current speculative decoding methods typically assume a considerable amount of available computing power, then generate a complex and massive draft tree using a small autoregressive language model to improve overall prediction accuracy. However, methods like batching have been widely applied in mainstream model inference systems as a superior alternative to speculative decoding, as they compress the available idle computing power. Therefore, performing speculative decoding with low verification resources and low scheduling costs has become an important research problem. We believe that more capable models that allow for parallel generation on draft sequences are what we truly need. Recognizing the fundamental nature of draft models to only generate sequences of limited length, we propose SpecFormer, a novel architecture that integrates unidirectional and bidirectional attention mechanisms. SpecFormer combines the autoregressive model's ability to extract information from the entire input sequence with the parallel generation benefits of non-autoregressive models. This design eliminates the reliance on large prefix trees and achieves consistent acceleration, even in large-batch scenarios. Through lossless speculative decoding experiments across models of various scales, we demonstrate that SpecFormer sets a new standard for scaling LLM inference with lower training demands and reduced computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05130v3">Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI 2026 Workshop WMAC
    </div>
    <details class="paper-abstract">
      Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20284v1">Can LLMs Make (Personalized) Access Control Decisions?</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Precise access control decisions are crucial to the security of both traditional applications and emerging agent-based systems. Typically, these decisions are made by users during app installation or at runtime. Due to the increasing complexity and automation of systems, making these access control decisions can add a significant cognitive load on users, often overloading them and leading to suboptimal or even arbitrary access control decisions. To address this problem, we propose to leverage the processing and reasoning capabilities of large language models (LLMs) to make dynamic, context-aware decisions aligned with the user's security preferences. For this purpose, we conducted a user study, which resulted in a dataset of 307 natural-language privacy statements and 14,682 access control decisions made by users. We then compare these decisions against those made by two versions of LLMs: a general and a personalized one, for which we also gathered user feedback on 1,446 of its decisions. Our results show that in general, LLMs can reflect users' preferences well, achieving up to 86\% accuracy when compared to the decision made by the majority of users. Our study also reveals a crucial trade-off in personalizing such a system: while providing user-specific privacy preferences to the LLM generally improves agreement with individual user decisions, adhering to those preferences can also violate some security best practices. Based on our findings, we discuss design and risk considerations for implementing a practical natural-language-based access control system that balances personalization, security, and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00882v5">VULSOLVER: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, degraded performance with long context, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 97.85% accuracy, 97.97% F1-score, and 100% recall. Applied to widely-used open-source projects, VULSOLVER identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20172v1">Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 13 pages, accepted by SIGMOD'26
    </div>
    <details class="paper-abstract">
      The rapid increase in LLM model sizes and the growing demand for long-context inference have made memory a critical bottleneck in GPU-accelerated serving systems. Although high-bandwidth memory (HBM) on GPUs offers fast access, its limited capacity necessitates reliance on host memory (CPU DRAM) to support larger working sets such as the KVCache. However, the maximum DRAM capacity is constrained by the limited number of memory channels per CPU socket. To overcome this limitation, current systems often adopt RDMA-based disaggregated memory pools, which introduce significant challenges including high access latency, complex communication protocols, and synchronization overhead. Fortunately, the emerging CXL technology introduces new opportunities in KVCache design. In this paper, we propose Beluga, a novel memory architecture that enables GPUs and CPUs to access a shared, large-scale memory pool through CXL switches. By supporting native load/store access semantics over the CXL fabric, our design delivers near-local memory latency, while reducing programming complexity and minimizing synchronization overhead. We conduct a systematic characterization of a commercial CXL switch-based memory pool and propose a set of design guidelines. Based on Beluga, we design and implement Beluga-KVCache, a system tailored for managing the large-scale KVCache in LLM inference. Beluga-KVCache achieves an 89.6% reduction in Time-To-First-Token (TTFT) and 7.35x throughput improvement in the vLLM inference engine compared to RDMA-based solutions. To the best of our knowledge, Beluga is the first system that enables GPUs to directly access large-scale memory pools through CXL switches, marking a significant step toward low-latency, shared access to vast memory resources by GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20100v1">QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages, 2 figures, accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20080v1">Adaptive LLM Agents: Toward Personalized Empathetic Care</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted at workshop Future Wellbeing: Using Design Fiction to Explore Human-Agent Interaction and Mental Health at The 13th International Conference on Human-Agent Interaction (HAI 2025)
    </div>
    <details class="paper-abstract">
      Current mental-health conversational systems are usually based on fixed, generic dialogue patterns. This paper proposes an adaptive framework based on large language models that aims to personalize therapeutic interaction according to a user's psychological state, quantified with the Acceptance of Illness Scale (AIS). The framework defines three specialized agents, L, M, and H, each linked to a different level of illness acceptance, and adjusts conversational behavior over time using continuous feedback signals. The AIS-stratified architecture is treated as a diegetic prototype placed in a plausible near-future setting and examined through the method of design fiction. By embedding the architecture in narrative scenarios, the study explores how such agents might influence access to care and therapeutic relationship. The goal is to show how clinically informed personalization, technical feasibility, and speculative scenario analysis can together inform the responsible design of LLM-based companions for mental-health support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20048v1">Reducing Latency of LLM Search Agent via Speculation-based Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      LLM-based search agents achieve strong performance but suffer from severe latency, as each step requires serialized LLM reasoning followed by action of tool execution. We revisit this bottleneck through the lens of speculation. While traditional predict-verify speculation paradigm can break serial execution, its benefit remains limited, as it retains the full original workload and adds extra inference overhead. We observe that early agent steps often involve simple evidence-gathering, where correct actions can often be predicted without full reasoning. Building on these observations, we present SPAgent, an algorithm-system co-design framework that expands the role of speculation in search agents to reduce latency. Algorithmically, SPAgent introduces a two-phase adaptive speculation mechanism that selectively omits verification when safe. System-wise, a two-level scheduler regulates speculative requests based on engine load to ensure speculation remains beneficial. We implement SPAgent in real-world systems. Across extensive experimental settings, SPAgent achieves up to $1.65\times$ end-to-end speedup while maintaining same or even achieving higher accuracy, enabling practical deployment of multi-step search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10037v2">Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Existing tool-augmented large language models (LLMs) encounter significant challenges when processing complex queries. Current frameworks such as ReAct are prone to local optimization traps due to their reliance on incremental decision-making processes. To address these limitations, we propose a novel Planner-centric Plan-Execute paradigm that fundamentally resolves local optimization bottlenecks through architectural innovation. Central to our approach is a novel Planner model that performs global Directed Acyclic Graph (DAG) planning for complex queries, enabling optimized execution beyond conventional tool coordination. We also introduce ComplexTool-Plan, a large-scale benchmark dataset featuring complex queries that demand sophisticated multi-tool composition and coordination capabilities. Additionally, we develop a two-stage training methodology that integrates Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), systematically enhancing the Planner's tool selection accuracy and global planning awareness through structured DAG-based planning. When integrated with a capable executor, our framework achieves state-of-the-art performance on the StableToolBench benchmark for complex user queries, demonstrating superior end-to-end execution capabilities and robust handling of intricate multi-tool workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19956v1">Prompt Fairness: Sub-group Disparities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), though shown to be effective in many applications, can vary significantly in their response quality. In this paper, we investigate this problem of prompt fairness: specifically, the phrasing of a prompt by different users/styles, despite the same question being asked in principle, may elicit different responses from an LLM. To quantify this disparity, we propose to use information-theoretic metrics that can capture two dimensions of bias: subgroup sensitivity, the variability of responses within a subgroup and cross group consistency, the variability of responses across subgroups. Our analysis reveals that certain subgroups exhibit both higher internal variability and greater divergence from others. Our empirical analysis reveals that certain demographic sub groups experience both higher internal variability and greater divergence from others, indicating structural inequities in model behavior. To mitigate these disparities, we propose practical interventions, including majority voting across multiple generations and prompt neutralization, which together improve response stability and enhance fairness across user populations. In the experiments, we observe clear prompt sensitivity disparities across demographic subgroups: before mitigation, cross-group divergence values reach 0.28 and typically fall in the from 0.14 to 0.22 range. After applying our neutralization and multi generation strategy, these divergences consistently decrease, with the largest gap reduced to 0.22 and many distances falling to 0.17 or below, indicating more stable and consistent outputs across subgroups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08726v2">Improved LLM Agents for Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 13 pages, 5 figures. Unlike the previous version, LLM names are now unmasked
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19940v1">Editing with AI: How Doctors Refine LLM-Generated Answers to Patient Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Patients frequently seek information during their medical journeys, but the rising volume of digital patient messages has strained healthcare systems. Large language models (LLMs) offer promise in generating draft responses for clinicians, yet how physicians refine these drafts remains underexplored. We present a mixed-methods study with nine ophthalmologists answering 144 cataract surgery questions across three conditions: writing from scratch, directly editing LLM drafts, and instruction-based indirect editing. Our quantitative and qualitative analyses reveal that while LLM outputs were generally accurate, occasional errors and automation bias revealed the need for human oversight. Contextualization--adapting generic answers to local practices and patient expectations--emerged as a dominant form of editing. Editing workflows revealed trade-offs: indirect editing reduced effort but introduced errors, while direct editing ensured precision but with higher workload. We conclude with design and policy implications for building safe, scalable LLM-assisted clinical communication systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19931v1">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {https://anonymous.4open.science/r/LLM-EDT-583F}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19877v1">It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19875v1">CodeFuse-CommitEval: Towards Benchmarking LLM's Power on Commit Message and Code Change Inconsistency Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Version control relies on commit messages to convey the rationale for code changes, but these messages are often low quality and, more critically, inconsistent with their diffs-known as message-code inconsistency (MCI). MCIs mislead reviewers, hinder maintenance, contaminate research datasets, and may obscure security patches. Yet, no dedicated benchmark exists to evaluate models for MCI detection. We introduce CODEFUSE-COMMITEVAL, the first benchmark designed for MCI detection using large language models (LLMs). Built on the ApacheCM dataset for diversity and quality, we generate seven types of inconsistent messages through rule-guided mutations of originally consistent commits and apply two-fold validation to verify both positive and negative samples. Using this labeled dataset of message-diff pairs, we evaluate six state-of-the-art open-source LLMs under a vanilla setting and with three augmentation strategies: few-shot prompting, chain-of-thought, and extended context. Results show models detect inconsistent commits more reliably than consistent ones (average Recall 85.95%, Precision 80.28%, Specificity 63.8%); gpt-oss-20B performs best overall but uses over twice the tokens of others. Augmentation effects vary: adjacent context helps larger models but adds noise for smaller ones; few-shot improves accuracy and reduces token use, yet increases universally incorrect predictions; chain-of-thought boosts precision and specificity at the cost of recall and higher token consumption. Type-wise analysis reveals higher detectability for component, file-path, and operation inconsistencies, but lower accuracy and higher token cost for intent-level "purpose" inconsistencies. CODEFUSE-COMMITEVAL provides a rigorous foundation for measuring, comparing, and advancing MCI detection, highlighting the need for richer context and balanced data to capture high-level semantic gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19874v1">Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 10 pages, 2 figures, 8 tables. Evaluation across 6 production LLMs with 1,198 traces
    </div>
    <details class="paper-abstract">
      As AI agents become integral to enterprise workflows, their reliance on shared tool libraries and pre-trained components creates significant supply chain vulnerabilities. While previous work has demonstrated behavioral backdoor detection within individual LLM architectures, the critical question of cross-LLM generalization remains unexplored, a gap with serious implications for organizations deploying multiple AI systems. We present the first systematic study of cross-LLM behavioral backdoor detection, evaluating generalization across six production LLMs (GPT-5.1, Claude Sonnet 4.5, Grok 4.1, Llama 4 Maverick, GPT-OSS 120B, and DeepSeek Chat V3.1). Through 1,198 execution traces and 36 cross-model experiments, we quantify a critical finding: single-model detectors achieve 92.7% accuracy within their training distribution but only 49.2% across different LLMs, a 43.4 percentage point generalization gap equivalent to random guessing. Our analysis reveals that this gap stems from model-specific behavioral signatures, particularly in temporal features (coefficient of variation > 0.8), while structural features remain stable across architectures. We show that model-aware detection incorporating model identity as an additional feature achieves 90.6% accuracy universally across all evaluated models. We release our multi-LLM trace dataset and detection framework to enable reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01265v3">AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning a decade of reporting from 2015 to 2025. Designed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a realistic benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we investigate the impact of domain specificity on abstractive summarisation of Arabic financial texts with large language models (LLMs). In particular, we evaluate transformer-based models: mT5, AraT5, and the domain-adapted FinAraT5 to examine how financial-domain pretraining influences accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more coherent summaries, especially in their handling of quantitative and entity-centric information. These findings highlight the importance of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-uk/AraFinNews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14977v2">SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19852v1">Profile-LLM: Dynamic Profile Optimization for Realistic Personality Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) have been shown to be an effective way to create more engaging and enjoyable user-AI interactions. While previous studies have explored using prompts to elicit specific personality traits in LLMs, they have not optimized these prompts to maximize personality expression. To address this limitation, we propose PersonaPulse: Dynamic Profile Optimization for Realistic Personality Expression in LLMs, a framework that leverages LLMs' inherent knowledge of personality traits to iteratively enhance role-play prompts while integrating a situational response benchmark as a scoring tool, ensuring a more realistic and contextually grounded evaluation to guide the optimization process. Quantitative evaluations demonstrate that the prompts generated by PersonaPulse outperform those of prior work, which were designed based on personality descriptions from psychological studies. Additionally, we explore the relationship between model size and personality modeling through extensive experiments. Finally, we find that, for certain personality traits, the extent of personality evocation can be partially controlled by pausing the optimization process. These findings underscore the importance of prompt optimization in shaping personality expression within LLMs, offering valuable insights for future research on adaptive AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16690v5">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $τ$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $τ$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $τ$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17648v4">Simulating Macroeconomic Expectations using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for simulating macroeconomic expectations using LLM Agents. By constructing LLM Agents equipped with various functional modules, we replicate three representative survey experiments involving several expectations across different types of economic agents. Our results show that although the expectations simulated by LLM Agents are more homogeneous than those of humans, they consistently outperform LLMs relying simply on prompt engineering, and possess human-like mental mechanisms. Evaluation reveals that these capabilities stem from the contributions of their components, offering guidelines for their architectural design. Our approach complements traditional methods and provides new insights into AI behavioral science in macroeconomic research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19830v1">Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native Query Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Multi-modal analytical processing has the potential to transform applications in e-commerce, healthcare, entertainment, and beyond. However, real-world adoption remains elusive due to the limited ability of traditional relational query operators to capture query semantics. The emergence of foundation models, particularly the large language models (LLMs), opens up new opportunities to develop flexible, semantic-aware data analytics systems that transcend the relational paradigm. We present Nirvana, a multi-modal data analytics framework that incorporates programmable semantic operators while leveraging both logical and physical query optimization strategies, tailored for LLM-driven semantic query processing. Nirvana addresses two key challenges. First, it features an agentic logical optimizer that uses natural language-specified transformation rules and random-walk-based search to explore vast spaces of semantically equivalent query plans -- far beyond the capabilities of conventional optimizers. Second, it introduces a cost-aware physical optimizer that selects the most effective LLM backend for each operator using a novel improvement-score metric. To further enhance efficiency, Nirvana incorporates computation reuse and evaluation pushdown techniques guided by model capability hypotheses. Experimental evaluations on three real-world benchmarks demonstrate that Nirvana is able to reduce end-to-end runtime by 10%--85% and reduces system processing costs by 76% on average, outperforming state-of-the-art systems at both efficiency and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v2">Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Polished the abstract and replaced the demonstration screenshots
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v3">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20910v1">Emergence and Localisation of Semantic Role Circuits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Despite displaying semantic competence, large language models' internal mechanisms that ground abstract semantic structure remain insufficiently characterised. We propose a method integrating role-cross minimal pairs, temporal emergence analysis, and cross-model comparison to study how LLMs implement semantic roles. Our analysis uncovers: (i) highly concentrated circuits (89-94% attribution within 28 nodes); (ii) gradual structural refinement rather than phase transitions, with larger models sometimes bypassing localised circuits; and (iii) moderate cross-scale conservation (24-59% component overlap) alongside high spectral similarity. These findings suggest that LLMs form compact, causally isolated mechanisms for abstract semantic structure, and these mechanisms exhibit partial transfer across scales and architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09606v2">How Can We Effectively Use LLMs for Phishing Detection?: Evaluating the Effectiveness of Large Language Model-based Phishing Detection Models</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a promising phishing detection mechanism, addressing the limitations of traditional deep learning-based detectors, including poor generalization to previously unseen websites and a lack of interpretability. However, LLMs' effectiveness for phishing detection remains unexplored. This study investigates how to effectively leverage LLMs for phishing detection (including target brand identification) by examining the impact of input modalities (screenshots, logos, HTML, and URLs), temperature settings, and prompt engineering strategies. Using a dataset of 19,131 real-world phishing websites and 243 benign sites, we evaluate seven LLMs -- two commercial models (GPT 4.1 and Gemini 2.0 flash) and five open-source models (Qwen, Llama, Janus, DeepSeek-VL2, and R1) -- alongside two deep learning (DL)-based baselines (PhishIntention and Phishpedia). Our findings reveal that commercial LLMs generally outperform open-source models in phishing detection, while DL models demonstrate better performance on benign samples. For brand identification, screenshot inputs achieve optimal results, with commercial LLMs reaching 93-95% accuracy and open-source models, particularly Qwen, achieving up to 92%. However, incorporating multiple input modalities simultaneously or applying one-shot prompts does not consistently enhance performance and may degrade results. Furthermore, higher temperature values reduce performance. Based on these results, we recommend using screenshot inputs with zero temperature to maximize accuracy for LLM-based detectors with HTML serving as auxiliary context when screenshot information is insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20878v1">Supporting Students in Navigating LLM-Generated Insecure Code</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      The advent of Artificial Intelligence (AI), particularly large language models (LLMs), has revolutionized software development by enabling developers to specify tasks in natural language and receive corresponding code, boosting productivity. However, this shift also introduces security risks, as LLMs may generate insecure code that can be exploited by adversaries. Current educational approaches emphasize efficiency while overlooking these risks, leaving students underprepared to identify and mitigate security issues in AI-assisted workflows. To address this gap, we present Bifröst, an educational framework that cultivates security awareness in AI-augmented development. Bifröst integrates (1) a Visual Studio Code extension simulating realistic environments, (2) adversarially configured LLMs that generate insecure code, and (3) a feedback system highlighting vulnerabilities. By immersing students in tasks with compromised LLMs and providing targeted security analysis, Bifröst cultivates critical evaluation skills; classroom deployments (n=61) show vulnerability to insecure code, while a post-intervention survey (n=21) indicates increased skepticism toward LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20872v1">Winning with Less for Low Resource Languages: Advantage of Cross-Lingual English_Persian Argument Mining Model over LLM Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Argument mining is a subfield of natural language processing to identify and extract the argument components, like premises and conclusions, within a text and to recognize the relations between them. It reveals the logical structure of texts to be used in tasks like knowledge extraction. This paper aims at utilizing a cross-lingual approach to argument mining for low-resource languages, by constructing three training scenarios. We examine the models on English, as a high-resource language, and Persian, as a low-resource language. To this end, we evaluate the models based on the English Microtext corpus \citep{PeldszusStede2015}, and its parallel Persian translation. The learning scenarios are as follow: (i) zero-shot transfer, where the model is trained solely with the English data, (ii) English-only training enhanced by synthetic examples generated by Large Language Models (LLMs), and (iii) a cross-lingual model that combines the original English data with manually translated Persian sentences. The zero-shot transfer model attains F1 scores of 50.2\% on the English test set and 50.7\% on the Persian test set. LLM-based augmentation model improves the performance up to 59.2\% on English and 69.3\% on Persian. The cross-lingual model, trained on both languages but evaluated solely on the Persian test set, surpasses the LLM-based variant, by achieving a F1 of 74.8\%. Results indicate that a lightweight cross-lingual blend can outperform considerably the more resource-intensive augmentation pipelines, and it offers a practical pathway for the argument mining task to overcome data resource shortage on low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20857v1">Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Statefulness is essential for large language model (LLM) agents to perform long-term planning and problem-solving. This makes memory a critical component, yet its management and evolution remain largely underexplored. Existing evaluations mostly focus on static conversational settings, where memory is passively retrieved from dialogue to answer queries, overlooking the dynamic ability to accumulate and reuse experience across evolving task streams. In real-world environments such as interactive problem assistants or embodied agents, LLMs are required to handle continuous task streams, yet often fail to learn from accumulated interactions, losing valuable contextual insights, a limitation that calls for test-time evolution, where LLMs retrieve, integrate, and update memory continuously during deployment. To bridge this gap, we introduce Evo-Memory, a comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents. Evo-Memory structures datasets into sequential task streams, requiring LLMs to search, adapt, and evolve memory after each interaction. We unify and implement over ten representative memory modules and evaluate them across 10 diverse multi-turn goal-oriented and single-turn reasoning and QA datasets. To better benchmark experience reuse, we provide a baseline method, ExpRAG, for retrieving and utilizing prior experience, and further propose ReMem, an action-think-memory refine pipeline that tightly integrates reasoning, task actions, and memory updates to achieve continual improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08532v3">Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20726v1">Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 24 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Autonomous driving faces critical challenges in rare long-tail events and complex multi-agent interactions, which are scarce in real-world data yet essential for robust safety validation. This paper presents a high-fidelity scenario generation framework that integrates a conditional variational autoencoder (CVAE) with a large language model (LLM). The CVAE encodes historical trajectories and map information from large-scale naturalistic datasets to learn latent traffic structures, enabling the generation of physically consistent base scenarios. Building on this, the LLM acts as an adversarial reasoning engine, parsing unstructured scene descriptions into domain-specific loss functions and dynamically guiding scenario generation across varying risk levels. This knowledge-driven optimization balances realism with controllability, ensuring that generated scenarios remain both plausible and risk-sensitive. Extensive experiments in CARLA and SMARTS demonstrate that our framework substantially increases the coverage of high-risk and long-tail events, improves consistency between simulated and real-world traffic distributions, and exposes autonomous driving systems to interactions that are significantly more challenging than those produced by existing rule- or data-driven methods. These results establish a new pathway for safety validation, enabling principled stress-testing of autonomous systems under rare but consequential events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12491v3">Cost-Aware Contrastive Routing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16660v2">Cognitive Foundations for Reasoning and Their Manifestation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 40 pages, 4 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. To understand this gap, we synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning reasoning invariants, meta-cognitive controls, representations for organizing reasoning & knowledge, and transformation operations. We introduce a fine-grained evaluation framework and conduct the first large-scale empirical analysis of 192K traces from 18 models across text, vision, and audio, complemented by 54 human think-aloud traces, which we make publicly available. We find that models under-utilize cognitive elements correlated with success, narrowing to rigid sequential processing on ill-structured problems where diverse representations and meta-cognitive monitoring are critical. Human traces show more abstraction and conceptual processing, while models default to surface-level enumeration. Meta-analysis of 1.6K LLM reasoning papers reveals the research community concentrates on easily quantifiable elements (sequential organization: 55%, decomposition: 60%) but neglecting meta-cognitive controls (self-awareness: 16%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 66.7% on complex problems. By establishing a shared vocabulary between cognitive science and LLM research, our framework enables systematic diagnosis of reasoning failures and principled development of models that reason through robust cognitive mechanisms rather than spurious shortcuts, while providing tools to test theories of human cognition at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19423v1">Beyond Protein Language Models: An Agentic LLM Framework for Mechanistic Enzyme Design</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present Genie-CAT, a tool-augmented large-language-model (LLM) system designed to accelerate scientific hypothesis generation in protein design. Using metalloproteins (e.g., ferredoxins) as a case study, Genie-CAT integrates four capabilities -- literature-grounded reasoning through retrieval-augmented generation (RAG), structural parsing of Protein Data Bank files, electrostatic potential calculations, and machine-learning prediction of redox properties -- into a unified agentic workflow. By coupling natural-language reasoning with data-driven and physics-based computation, the system generates mechanistically interpretable, testable hypotheses linking sequence, structure, and function. In proof-of-concept demonstrations, Genie-CAT autonomously identifies residue-level modifications near [Fe--S] clusters that affect redox tuning, reproducing expert-derived hypotheses in a fraction of the time. The framework highlights how AI agents combining language models with domain-specific tools can bridge symbolic reasoning and numerical simulation, transforming LLMs from conversational assistants into partners for computational discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10659v2">Information Extraction From Fiscal Documents Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 6 pages. Presented at the AI for Financial Inclusion, Risk Modeling and Resilience in Emerging Markets workshop at ACM ICAIF 2025 Singapore
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in text comprehension, but their ability to process complex, hierarchical tabular data remains underexplored. We present a novel approach to extracting structured data from multi-page government fiscal documents using LLM-based techniques. Applied to annual fiscal documents from the State of Karnataka in India (200+ pages), our method achieves high accuracy through a multi-stage pipeline that leverages domain knowledge, sequential context, and algorithmic validation. A large challenge with traditional OCR methods is the inability to verify the accurate extraction of numbers. When applied to fiscal data, the inherent structure of fiscal tables, with totals at each level of the hierarchy, allows for robust internal validation of the extracted data. We use these hierarchical relationships to create multi-level validation checks. We demonstrate that LLMs can read tables and also process document-specific structural hierarchies, offering a scalable process for converting PDF-based fiscal disclosures into research-ready databases. Our implementation shows promise for broader applications across developing country contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03469v2">Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to AgenticSE Workshop at ASE 2025
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03463v2">ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to MAS-GAIN Workshop at ASE 2025
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19368v1">LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has been increasingly adopted in many real-world applications. While MARL enables decentralized deployment on resource-constrained edge devices, it suffers from severe non-stationarity due to the synchronous updates of agent policies. This non stationarity results in unstable training and poor policy con vergence, especially as the number of agents increases. In this paper, we propose RELED, a scalable MARL framework that integrates large language model (LLM)-driven expert demonstrations with autonomous agent exploration. RELED incorporates a Stationarity-Aware Expert Demonstration module, which leverages theoretical non-stationarity bounds to enhance the quality of LLM-generated expert trajectories, thus providing high reward and training-stable samples for each agent. Moreover, a Hybrid Expert-Agent Policy Optimization module adaptively balances each agent's learning from both expert-generated and agent-generated trajectories, accelerating policy convergence and improving generalization. Extensive experiments with real city networks based on OpenStreetMap demonstrate that RELED achieves superior performance compared to state-of-the-art MARL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03480v2">LLM Agents for Automated Dependency Upgrades</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to AISM Workshop at ASE 2005
    </div>
    <details class="paper-abstract">
      As a codebase expands over time, its library dependencies can become outdated and require updates to maintain innovation and security. However, updating a library can introduce breaking changes in the code, necessitating significant developer time for maintenance. To address this, we introduce a framework of LLM agents to be used in combination with migration documentation to automatically recommend and apply code updates and ensure compatibility with new versions. Our solution can automatically localize updated library usages in live Java codebases and implement recommended fixes in a user-friendly manner. The system architecture consists of multiple key components: a Summary Agent, Control Agent, and Code Agent. To validate our approach, we apply the framework on an industrial use case by which we create three synthetic code repositories with major Upgrade changes and benchmark our approach against state-of-the-art methods. Results show that our approach not only performs upgrades using fewer tokens across all cases but also achieves a precision of 71.4%, highlighting its efficiency and effectiveness compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19355v1">Leveraging LLMs for reward function design in reinforcement learning control tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19333v1">Learning to Reason: Training LLMs with GPT-OSS or DeepSeek R1 Reasoning Traces</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Test-time scaling, which leverages additional computation during inference to improve model accuracy, has enabled a new class of Large Language Models (LLMs) that are able to reason through complex problems by understanding the goal, turning this goal into a plan, working through intermediate steps, and checking their own work before answering . Frontier large language models with reasoning capabilities, such as DeepSeek-R1 and OpenAI's gpt-oss, follow the same procedure when solving complex problems by generating intermediate reasoning traces before giving the final answer. Today, these models are being increasingly used to generate reasoning traces that serve as high-quality supervised data for post-training of small and medium-sized language models to teach reasoning capabilities without requiring expensive human curation. In this work, we compare the performance of medium-sized LLMs on Math problems after post-training on two kinds of reasoning traces. We compare the impact of reasoning traces generated by DeepSeek-R1 and gpt-oss LLMs in terms of accuracy and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22006v2">Enhancing Domain-Specific Encoder Models with LLM-Generated Data: How to Leverage Ontologies, and How to Do Without Them</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Published in the Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      We investigate the use of LLM-generated data for continual pretraining of encoder models in specialized domains with limited training data, using the scientific domain of invasion biology as a case study. To this end, we leverage domain-specific ontologies by enriching them with LLM-generated data and pretraining the encoder model as an ontology-informed embedding model for concept definitions. To evaluate the effectiveness of this method, we compile a benchmark specifically designed for assessing model performance in invasion biology. After demonstrating substantial improvements over standard LLM pretraining, we investigate the feasibility of applying the proposed approach to domains without comprehensive ontologies by substituting ontological concepts with concepts automatically extracted from a small corpus of scientific abstracts and establishing relationships between concepts through distributional statistics. Our results demonstrate that this automated approach achieves comparable performance using only a small set of scientific abstracts, resulting in a fully automated pipeline for enhancing domain-specific understanding of small encoder models that is especially suited for application in low-resource settings and achieves performance comparable to masked language modeling pretraining on much larger datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21505v2">How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some research on language-specific neurons provides a new perspective to analyze and understand LLMs' mechanisms. However, we find that there are many neurons that are shared by multiple but not all languages and cannot be correctly classified. In this work, we propose a ternary classification methodology that categorizes neurons into three types, including language-specific neurons, language-related neurons, and general neurons. And we propose a corresponding identification algorithm to distinguish these different types of neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights to better understand multilingual alignment and multilingual capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06725v2">WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19218v1">Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19175v1">LLM-Based Agentic Negotiation for 6G: Addressing Uncertainty Neglect and Tail-Event Risk</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Link to open-source non-commercial code available
    </div>
    <details class="paper-abstract">
      A critical barrier to the trustworthiness of sixth-generation (6G) agentic autonomous networks is the uncertainty neglect bias; a cognitive tendency for large language model (LLM)-powered agents to make high-stakes decisions based on simple averages while ignoring the tail risk of extreme events. This paper proposes an unbiased, risk-aware framework for agentic negotiation, designed to ensure robust resource allocation in 6G network slicing. Specifically, agents leverage Digital Twins (DTs) to predict full latency distributions, which are then evaluated using a formal framework from extreme value theory, namely, Conditional Value-at-Risk (CVaR). This approach fundamentally shifts the agent's objective from reasoning over the mean to reasoning over the tail, thereby building a statistically-grounded buffer against worst-case outcomes. Furthermore, our framework ensures full uncertainty awareness by requiring agents to quantify epistemic uncertainty -- confidence in their own DTs predictions -- and propagate this meta-verification to make robust decisions, preventing them from acting on unreliable data. We validate this framework in a 6G inter-slice negotiation use-case between an eMBB and a URLLC agent. The results demonstrate the profound failure of the biased, mean-based baseline, which consistently fails its SLAs with a 25\% rate. Our unbiased, CVaR-aware agent successfully mitigates this bias, eliminating SLA violations and reducing the URLLC and eMBB p99.999 latencies by around 11\%. We show this reliability comes at the rational and quantifiable cost of slightly reduced energy savings to 17\%, exposing the false economy of the biased approach. This work provides a concrete methodology for building the trustworthy autonomous systems required for 6G.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19171v1">Can LLMs Threaten Human Survival? Benchmarking Potential Existential Threats from LLMs via Prefix Completion</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Research on the safety evaluation of large language models (LLMs) has become extensive, driven by jailbreak studies that elicit unsafe responses. Such response involves information already available to humans, such as the answer to "how to make a bomb". When LLMs are jailbroken, the practical threat they pose to humans is negligible. However, it remains unclear whether LLMs commonly produce unpredictable outputs that could pose substantive threats to human safety. To address this gap, we study whether LLM-generated content contains potential existential threats, defined as outputs that imply or promote direct harm to human survival. We propose \textsc{ExistBench}, a benchmark designed to evaluate such risks. Each sample in \textsc{ExistBench} is derived from scenarios where humans are positioned as adversaries to AI assistants. Unlike existing evaluations, we use prefix completion to bypass model safeguards. This leads the LLMs to generate suffixes that express hostility toward humans or actions with severe threat, such as the execution of a nuclear strike. Our experiments on 10 LLMs reveal that LLM-generated content indicates existential threats. To investigate the underlying causes, we also analyze the attention logits from LLMs. To highlight real-world safety risks, we further develop a framework to assess model behavior in tool-calling. We find that LLMs actively select and invoke external tools with existential threats. Code and data are available at: https://github.com/cuiyu-ai/ExistBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07751v3">AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in grade school math (GSM) reasoning. In particular, they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In this work, we instead focuses on the strategy of "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. Focusing on GSM, we find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. Besides, improving GSM robustness via AbstRaL is shown to also implicitly benefit LLMs' capabilities on OOD mathematical and general reasoning tasks, indicating that abstract thinking broadly enables better generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16570v2">URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 NeurIPS 2025, Camera Ready
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly pretrained on vast corpora of text without utilizing contextual metadata such as source, quality, or topic, leading to a context-free learning paradigm. While recent studies suggest that adding metadata like URL information as context (i.e., auxiliary inputs not used in the loss calculation) can improve training efficiency and downstream performance, they offer limited understanding of which types of metadata are truly effective and under what conditions. In this work, we conduct a systematic evaluation and find that not all metadata types contribute equally. Only URL context speeds up training, whereas quality scores and topic/format domain information offer no clear benefit. Furthermore, the improved downstream performances of URL conditioning emerge only when longer prompts are used at inference time. In addition, we demonstrate that context-aware pretraining enables more controllable generation than context-free pretraining, in a classifier-free guidance fashion. Although topic and format metadata do not accelerate training, they are effective for steering outputs, offering human-interpretable control over generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.07078v4">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to KDD 2026, Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19132v1">LLMs-Powered Real-Time Fault Injection: An Approach Toward Intelligent Fault Test Cases Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      A well-known testing method for the safety evaluation and real-time validation of automotive software systems (ASSs) is Fault Injection (FI). In accordance with the ISO 26262 standard, the faults are introduced artificially for the purpose of analyzing the safety properties and verifying the safety mechanisms during the development phase. However, the current FI method and tools have a significant limitation in that they require manual identification of FI attributes, including fault type, location and time. The more complex the system, the more expensive, time-consuming and labour-intensive the process. To address the aforementioned challenge, a novel Large Language Models (LLMs)-assisted fault test cases (TCs) generation approach for utilization during real-time FI tests is proposed in this paper. To this end, considering the representativeness and coverage criteria, the applicability of various LLMs to create fault TCs from the functional safety requirements (FSRs) has been investigated. Through the validation results of LLMs, the superiority of the proposed approach utilizing gpt-4o in comparison to other state-of-the-art models has been demonstrated. Specifically, the proposed approach exhibits high performance in terms of FSRs classification and fault TCs generation with F1-score of 88% and 97.5%, respectively. To illustrate the proposed approach, the generated fault TCs were executed in real time on a hardware-in-the-loop system, where a high-fidelity automotive system model served as a case study. This novel approach offers a means of optimizing the real-time testing process, thereby reducing costs while simultaneously enhancing the safety properties of complex safety-critical ASSs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19131v1">Eliciting Chain-of-Thought in Base LLMs via Gradient-Based Representation Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI2026
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning is a critical capability for large language models (LLMs), enabling them to tackle com- plex multi-step tasks. While base LLMs, pre-trained on general text corpora, often struggle with reasoning due to a lack of specialized training, recent studies reveal their latent reason- ing potential tied to hidden states. However, existing hidden state manipulation methods, such as linear activation steering, suffer from limitations due to their rigid and unconstrained nature, often leading to distribution shifts and degraded text quality. In this work, we propose a novel approach for elic- iting CoT reasoning from base LLMs through hidden state manipulation grounded in probabilistic conditional generation. By reformulating the challenge as an optimization problem with a balanced likelihood and prior regularization framework, our method guides hidden states toward reasoning-oriented trajectories while preserving linguistic coherence. Extensive evaluations across mathematical, commonsense, and logical reasoning benchmarks demonstrate that our approach con- sistently outperforms existing steering methods, offering a theoretically principled and effective solution for enhancing reasoning capabilities in base LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19130v1">Can LLMs Recover Program Semantics? A Systematic Evaluation with Symbolic Execution</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Obfuscation poses a persistent challenge for software engineering tasks such as program comprehension, maintenance, testing, and vulnerability detection. While compiler optimizations and third-party code often introduce transformations that obscure program intent, existing analysis tools and large language models (LLMs) struggle to recover the original semantics. In this work, we investigate whether LLMs, when fine-tuned with symbolic execution artifacts, can effectively deobfuscate programs and restore analyzability. We construct a benchmark by applying four widely studied transformations-control-flow flattening, opaque predicates, arithmetic encoding, and branch encoding-across diverse C programs from TUM Obfuscation Benchmarks, the LLVM test suite, and algorithmic repositories. We then compare three state-of-the-art LLMs under two training configurations: baseline fine-tuning on obfuscated/original code pairs, and enhanced fine-tuning with additional KLEE artifacts such as SMT constraints, path statistics, and test cases. Our evaluation examines syntactic correctness (compilation success), semantic fidelity (behavioral equivalence under symbolic execution), and code quality (readability and structure). Results show that GPT-4.1-mini achieves the strongest deobfuscation overall, and that incorporating KLEE artifacts consistently improves semantic preservation and compilation success across models. These findings highlight deobfuscation as a broader software engineering concern, demonstrating that combining LLMs with symbolic execution can strengthen automated testing, static analysis, and program comprehension in the presence of obfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19123v1">Facilitating the Integration of LLMs Into Online Experiments With Simple Chat</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, understanding human-LLM interactions is emerging as a central priority in psychological research. Online experiments offer an efficient means to study human-LLM interactions, yet integrating LLMs into established survey platforms remains technically demanding, particularly when aiming for ecologically valid, real-time conversational experiences with strong experimental control. We introduce Simple Chat, an open-source, research-focused chat interface that streamlines LLM integration for platforms such as Qualtrics, oTree, and LimeSurvey, while presenting a unified participant experience across conditions. Simple Chat connects to both commercial providers and open-weights models, supports streaming responses to preserve conversational flow, and offers an administrative interface for fine-grained control of prompts and interface features. By reducing technical barriers, standardizing interfaces, and improving participant experience, Simple Chat helps advance the study of human-LLM interaction. In this article, we outline Simple Chat's key features, provide a step-by-step tutorial, and demonstrate its utility through two illustrative case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19122v1">Emotion-Enhanced Multi-Task Learning with LLMs for Aspect Category Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Aspect category sentiment analysis (ACSA) has achieved remarkable progress with large language models (LLMs), yet existing approaches primarily emphasize sentiment polarity while overlooking the underlying emotional dimensions that shape sentiment expressions. This limitation hinders the model's ability to capture fine-grained affective signals toward specific aspect categories. To address this limitation, we introduce a novel emotion-enhanced multi-task ACSA framework that jointly learns sentiment polarity and category-specific emotions grounded in Ekman's six basic emotions. Leveraging the generative capabilities of LLMs, our approach enables the model to produce emotional descriptions for each aspect category, thereby enriching sentiment representations with affective expressions. Furthermore, to ensure the accuracy and consistency of the generated emotions, we introduce an emotion refinement mechanism based on the Valence-Arousal-Dominance (VAD) dimensional framework. Specifically, emotions predicted by the LLM are projected onto a VAD space, and those inconsistent with their corresponding VAD coordinates are re-annotated using a structured LLM-based refinement strategy. Experimental results demonstrate that our approach significantly outperforms strong baselines on all benchmark datasets. This underlines the effectiveness of integrating affective dimensions into ACSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19083v1">A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 This paper has been accepted by AAAI 2026 (Main Technical Track)
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) with large language models (LLMs) has emerged as a promising paradigm for named entity recognition (NER) in low-resource scenarios. However, existing ICL-based NER methods suffer from three key limitations: (1) reliance on dynamic retrieval of annotated examples, which is problematic when annotated data is scarce; (2) limited generalization to unseen domains due to the LLM's insufficient internal domain knowledge; and (3) failure to incorporate external knowledge or resolve entity ambiguities. To address these challenges, we propose KDR-Agent, a novel multi-agent framework for multi-domain low-resource in-context NER that integrates Knowledge retrieval, Disambiguation, and Reflective analysis. KDR-Agent leverages natural-language type definitions and a static set of entity-level contrastive demonstrations to reduce dependency on large annotated corpora. A central planner coordinates specialized agents to (i) retrieve factual knowledge from Wikipedia for domain-specific mentions, (ii) resolve ambiguous entities via contextualized reasoning, and (iii) reflect on and correct model predictions through structured self-assessment. Experiments across ten datasets from five domains demonstrate that KDR-Agent significantly outperforms existing zero-shot and few-shot ICL baselines across multiple LLM backbones. The code and data can be found at https://github.com/MWXGOD/KDR-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19078v1">GraphMind: Theorem Selection and Conclusion Generation Framework with Dynamic GNN for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, including multi-step reasoning such as mathematical proving. However, existing approaches often lack an explicit and dynamic mechanism to structurally represent and evolve intermediate reasoning states, which limits their ability to perform context-aware theorem selection and iterative conclusion generation. To address these challenges, we propose GraphMind, a novel dynamic graph-based framework that integrates the graph neural network (GNN) with LLMs to iteratively select theorems and generate intermediate conclusions for multi-step reasoning. Our method models the reasoning process as a heterogeneous evolving graph, where nodes represent conditions, theorems, and conclusions, while edges capture logical dependencies between nodes. By encoding the current reasoning state with GNN and leveraging semantic matching for theorem selection, our framework enables context-aware, interpretable, and structured reasoning in a closed-loop manner. Experiments on various question-answering (QA) datasets demonstrate that our proposed GraphMind method achieves consistent performance improvements and significantly outperforms existing baselines in multi-step reasoning, validating the effectiveness and generalizability of our approach.
    </details>
</div>
